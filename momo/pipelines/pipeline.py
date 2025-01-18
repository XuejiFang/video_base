from diffusers import DiffusionPipeline
from einops import rearrange
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch

class MomoPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            vae=vae, 
            transformer=transformer, 
            scheduler=scheduler
        )

    @torch.no_grad()
    def encode_text(self, prompts, negative_prompt=None):
        text = negative_prompt + prompts if negative_prompt is not None else prompts
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.transformer.config.max_text_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        ).to('cuda')
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return self.text_encoder(input_ids, attention_mask=cond_mask)[0]

    def pos_process(self, frames):
        """BCFHW -> [[PIL.Image, PIL.Image], [...], [...]]"""
        frames = rearrange(frames, 'b c f h w -> b f h w c')
        frames = frames.mul_(127.5).add(127.5).float().cpu().numpy().astype(np.uint8)
        videos = [[Image.fromarray(frame) for frame in batch] for batch in frames]
        return videos

    def prepare_latents(self, bsz, latent_dim, num_frames, height, width):
        noise = torch.randn((latent_dim, num_frames, height, width))
        return torch.cat([noise]*bsz, dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompts,
        num_frames,
        height,
        width,
        guidance_scale=5.5,
        num_diffusion_steps=30,
        negative_prompt=[""],
        num_samples_per_prompt=1,
    ):
        assert isinstance(prompts, list)

        self.scheduler.set_timesteps(num_inference_steps=num_diffusion_steps)
        device = self._execution_device
        
        bsz = num_samples_per_prompt
        # 2B L C
        text_token = self.encode_text(prompts*bsz, negative_prompt*bsz) if guidance_scale > 1 else self.encode_text(prompts*bsz) 
        text_token = self.transformer.text_embedder(text_token)
        weight_type = text_token.dtype
        # 1 C -> 2B 1 C
        bov_token = self.transformer.temporal_encoder.bov_token.unsqueeze(0).expand(text_token.shape[0], -1, -1)
        # KV Cache TODO
        # [setattr(attn, 'kv_cache', True) for attn in self.transformer.temporal_encoder]
        temporal_input = torch.cat([text_token, bov_token], dim=1)
        frames = []

        for f in tqdm(range(1, num_frames+1)):
            # B (L+1+tHW) C
            if len(frames) != 0:
                # TODO: what is uncond?
                temporal_input = torch.cat([temporal_input, 
                                            self.transformer.spatial_decoder.patch_embedder(frames[-1]).expand(text_token.shape[0], -1, -1)], dim=1)
            # B HW C -> 2B HW C
            cur_cond = self.transformer.temporal_encoder(temporal_input, f).last_frame
            # if guidance_scale > 1: # TODO: what serves as uncond?
            #     cur_uncond = torch.randn_like(cur_cond)
            #     cur_cond = torch.cat([cur_uncond, cur_cond], dim=0)
            # prepare latents: B 1 C H W
            latents = self.transformer.spatial_decoder.prepare_latents(
                bsz, height//self.transformer.config.spatial_interpolation_scale, width//self.transformer.config.spatial_interpolation_scale, 
                device, weight_type
            )
            # Spatial Decoder
            for t in tqdm(self.scheduler.timesteps):
                latents_input = torch.cat([latents]*2, dim=0) if guidance_scale > 1 else latents
                timestep = t.expand(latents_input.shape[0]).to(device, dtype=weight_type)
                noise_pred = self.transformer.spatial_decoder(latents_input, cur_cond, timestep).sample
                if guidance_scale > 1:
                    uncond_pred, cond_pred = noise_pred.chunk(2)
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            self.scheduler._init_step_index(self.scheduler.timesteps[0])
            frames.append(latents)
        # B C T H W
        frames = torch.cat(frames, dim=1)
        frames = rearrange(frames, 'b f c h w -> b c f h w')
        # B C T H W, -1 ~ 1
        frames = self.vae.decode(self.vae.unscale_(frames)).sample
        # to PIL
        frames = self.pos_process(frames)
        
        return (frames, )