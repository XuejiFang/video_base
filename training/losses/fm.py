import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from typing import Tuple

class CogFMLoss:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def step(
        self,
        model_pred: torch.FloatTensor,
        x_cur: torch.FloatTensor,
    ) -> Tuple:
        # Upcast to avoid precision issues when computing prev_sample
        x_cur = x_cur.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        x_prev = x_cur + (sigma_next - sigma) * model_pred

        # Cast sample back to model compatible dtype
        x_prev = x_prev.to(model_pred.dtype)

        return (x_prev,)

    
    def __call__(self, model, z_0, prompt_embed, prompt_mask):
        z_0 = z_0.permute(0,2,1,3,4)  # BFCHW for CogVideo
        z_T = torch.randn_like(z_0, dtype=z_0.dtype)
        bsz, _, _, _, _ = z_0.shape
        sigmas = torch.randn((bsz,), device=z_0.device, dtype=z_0.dtype).sigmoid()
        timestep = sigmas*1000
        sigmas = rearrange(sigmas, "b -> b 1 1 1 1")
        z_t = (1 - sigmas) * z_0 + sigmas * z_T
        target = z_T - z_0
        model_pred = model(
            hidden_states           = z_t,
            encoder_hidden_states   = prompt_embed,
            timestep                = timestep.to(z_t.dtype),
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        
        return loss.mean()
    
class OpenSoraFMLoss:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
    
    def __call__(self, model, z_0, prompt_embed, prompt_mask):
        # BCFHW
        noise_offset = 0.02
        z_T = torch.randn_like(z_0, dtype=z_0.dtype)
        z_T += noise_offset * torch.randn((z_0.shape[0], z_0.shape[1], 1, 1, 1), device=z_0.device)
        bsz, _, _, _, _ = z_0.shape
        sigmas = torch.randn((bsz,), device=z_0.device).sigmoid()
        timestep = sigmas*1000
        sigmas = rearrange(sigmas, "b -> b 1 1 1 1")
        z_t = (1 - sigmas) * z_0 + sigmas * z_T
        target = z_T - z_0

        model_pred = model(
            hidden_states           = z_t,
            encoder_hidden_states   = prompt_embed.unsqueeze_(1),    # B1LC for OpenSora
            encoder_attention_mask  = prompt_mask.unsqueeze_(1),
            timestep                = timestep.to(z_t.dtype),
        )[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        return loss.mean()
    
class MomoFMLoss:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
    
    def __call__(self, model, z_0, prompt_embed, prompt_mask):
        """
            z_0:            B C F H W
            prompt_embed:   B L C
        """
        # 1. concat text token and <bov> token
        bsz = z_0.shape[0]
        text_token = model.text_embedder(prompt_embed)
        # HW C  -> B HW C
        bov_token = model.temporal_encoder.bov_token.unsqueeze(0).expand(bsz, -1, -1)
        temporal_input = torch.cat([text_token, bov_token], dim=1)
        # 2. get temporal encoder last frame as condition for later
        cur_cond = model.temporal_encoder(temporal_input, 1).all_frame
        # 3. add noise to z_0 to get z_t, spatial decoder predict flow with condition
        z_0 = z_0.permute(0,2,1,3,4)  # BFCHW for CogVideo
        z_T = torch.randn_like(z_0, dtype=z_0.dtype)
        bsz, _, _, _, _ = z_0.shape
        sigmas = torch.randn((bsz,), device=z_0.device, dtype=z_0.dtype).sigmoid()
        timestep = sigmas*1000
        sigmas = rearrange(sigmas, "b -> b 1 1 1 1")
        z_t = (1 - sigmas) * z_0 + sigmas * z_T
        model_pred = model.spatial_decoder(
            hidden_states           = z_t,
            encoder_hidden_states   = cur_cond,
            timestep                = timestep.to(z_t.dtype),
        )[0]
        # 4. compuse mse loss between added nosise and prediction
        target = z_T - z_0
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        
        return loss.mean()
    
class MomoVidFMLoss:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_degrade: bool = False,
        degrade_scheduler: str = "cosine",
        num_frames: int = None,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.use_degrade = use_degrade
        if self.use_degrade:
            assert num_frames is not None
            if degrade_scheduler == 'cosine':
                self.delta = torch.cos(torch.arange(0, num_frames) / num_frames)
                self.delta = self.delta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                raise NotImplementedError
    
    def __call__(self, model, z_0, prompt_embed, prompt_mask):
        """
            z_0:            B C F H W
            prompt_embed:   B L C
        """
        # 1. concat text token and <bov> token and first (T-1) frames
        bsz, _, f, _, _ = z_0.shape
        text_token = model.text_embedder(prompt_embed)
        # HW C  -> B HW C
        bov_token = model.temporal_encoder.bov_token.unsqueeze(0).expand(bsz, -1, -1)
        # prev frames
        if f > 1:
            prev_frames = rearrange(z_0[:, :, :-1, ...], 'b c f h w -> (b f) 1 c h w')
            prev_frames = model.spatial_decoder.patch_embedder(prev_frames)
            prev_frames = rearrange(prev_frames, '(b f) n c -> b (f n) c', f=f-1)
            temporal_input = torch.cat([text_token, bov_token, prev_frames], dim=1)
        else:
            temporal_input = torch.cat([text_token, bov_token], dim=1)
        # 2. get temporal encoder last frame as condition for later
        cur_cond = model.temporal_encoder(temporal_input, f).all_frame
        if self.use_degrade and f>1:
            self.delta = self.delta.to(cur_cond.device, cur_cond.dtype)
            cur_cond = rearrange(cur_cond, "b (f n) c -> b f n c", f=f)
            eps = torch.randn_like(cur_cond, dtype=cur_cond.dtype, device=cur_cond.device)
            cur_cond = self.delta*cur_cond + (1-self.delta)*eps
            cur_cond = rearrange(cur_cond, "b f n c -> b (f n) c")
        # 3. add noise to z_0 to get z_t, spatial decoder predict flow with condition
        # BCFHW -> (BF)1CHW
        z_0 = rearrange(z_0, 'b c f h w -> (b f) 1 c h w')
        cur_cond = rearrange(cur_cond, 'b (f n) c -> (b f) n c', f=f)
        mini_bsz = None # bsz
        if mini_bsz is None:
            z_T = torch.randn_like(z_0, dtype=z_0.dtype)
            bsz = z_0.shape[0]
            sigmas = torch.randn((bsz,), device=z_0.device, dtype=z_0.dtype).sigmoid()
            timestep = sigmas*1000
            sigmas = rearrange(sigmas, "b -> b 1 1 1 1")
            z_t = (1 - sigmas) * z_0 + sigmas * z_T
            model_pred = model.spatial_decoder(
                hidden_states           = z_t,
                encoder_hidden_states   = cur_cond,
                timestep                = timestep.to(z_t.dtype),
            )[0]
            # 4. compuse mse loss between added nosise and prediction
            target = z_T - z_0
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        else:
            loss = 0.
            for idx in range(0, z_0.shape[0]//mini_bsz):
                z_0_        = z_0[idx*mini_bsz : (idx+1)*mini_bsz, ...]
                cur_cond_   = cur_cond[idx*mini_bsz : (idx+1)*mini_bsz, ...]

                z_T = torch.randn_like(z_0_, dtype=z_0_.dtype)
                bsz = z_0_.shape[0]
                sigmas = torch.randn((bsz,), device=z_0_.device, dtype=z_0_.dtype).sigmoid()
                timestep = sigmas*1000
                sigmas = rearrange(sigmas, "b -> b 1 1 1 1")
                z_t = (1 - sigmas) * z_0_ + sigmas * z_T
                model_pred = model.spatial_decoder(
                    hidden_states           = z_t,
                    encoder_hidden_states   = cur_cond_,
                    timestep                = timestep.to(z_t.dtype),
                )[0]
                # 4. compuse mse loss between added nosise and prediction
                target = z_T - z_0_
                loss += F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # loss: (b f) 1 c h w
        loss_list = rearrange(loss.detach().cpu(), "(b f) 1 c h w -> f (b c h w)", f=f).mean(dim=1).numpy()
        return loss.mean(), loss_list