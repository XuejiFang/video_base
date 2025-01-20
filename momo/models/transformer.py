from dataclasses import dataclass
from diffusers import ModelMixin
from diffusers.models.normalization import AdaLayerNorm
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils import BaseOutput
from typing import Tuple

from .embeddings import MomoPatchEmbed, _prepare_rotary_positional_embeddings
from .modules import BasiceSpatialBlock, BasiceTemporalBlock
import torch
import torch.nn as nn


@dataclass
class MomoOutput(BaseOutput):
    hidden_states: torch.Tensor = None
    last_frame: torch.Tensor = None         # B HW C
    sample: torch.Tensor = None             # B F C H W

class TextEmbedder(nn.Module):
    """Encode text tokens into embeddings."""
    def __init__(self, token_dim, embed_dim):
        super(TextEmbedder, self).__init__()
        self.proj, self.norm = nn.Linear(token_dim, embed_dim), nn.LayerNorm(embed_dim)

    def forward(self, x) -> torch.Tensor:

        return self.norm(self.proj(x))
    
class TemporalEncoder(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_layers, 
        num_heads,
        attn_head_dim,
        dropout,
        text_length, 
        num_frames, 
        height, 
        width,
        vae_scale_factor_spatial: int = 8,
        patch_size: int = 2,
        patch_size_t: int = 1,
        use_rotary_positional_embeddings: bool = True,
        max_width: int = 32,    
        max_height: int = 32,
    ):
        super().__init__()
        self.attn_head_dim = attn_head_dim
        inner_dim = num_heads * attn_head_dim
        self.transformer_blocks = nn.ModuleList([
            BasiceTemporalBlock(
                dim=inner_dim,
                num_attention_heads=num_heads,
                attention_head_dim=attn_head_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.text_length = text_length
        self.num_frams = num_frames
        self.height = height
        self.width = width
        self.frame_token_length = height // 2 * width // 2
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.max_height = max_height
        self.max_width = max_width
        self.bov_token = nn.Parameter(torch.randn(self.frame_token_length, inner_dim))

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def get_attn_mask(self, cur_num_frames):
        total_token_length = self.text_length + cur_num_frames * self.frame_token_length
        attention_mask = torch.ones((total_token_length, total_token_length))
        attention_mask = torch.triu(attention_mask)
        small_indices = torch.arange(cur_num_frames).view(-1, 1)  # shape: (T, 1)
        small_mask = (small_indices < small_indices.T).int()  # shape: (T, T)
        video_casual_mask = small_mask.repeat_interleave(self.frame_token_length, dim=0).repeat_interleave(self.frame_token_length, dim=1)
        attention_mask[:self.text_length, :self.text_length] = 0
        attention_mask[self.text_length:, self.text_length: ] = video_casual_mask
        attention_mask.mul_(-10_000)
        return attention_mask

    def __call__(
        self, 
        hidden_states,
        cur_num_frames,
    ):
        image_rotary_emb = (
            _prepare_rotary_positional_embeddings(
                self.height, self.width, 
                cur_num_frames, self.attn_head_dim, 
                self.patch_size, self.patch_size_t, 
                self.max_width, self.max_height,
                hidden_states.device
            )
            if self.use_rotary_positional_embeddings
            else None
        )
                
        attention_mask = self.get_attn_mask(cur_num_frames).to(hidden_states.device,dtype=hidden_states.dtype)
        for block_id, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    self.text_length,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    self.text_length,
                    image_rotary_emb,
                    attention_mask,
                )

        if cur_num_frames == 1:
            last_frame = hidden_states[:, -self.frame_token_length:, :]
        else:
            last_frame = hidden_states[:, -cur_num_frames*self.frame_token_length:, :]

        return MomoOutput(hidden_states=hidden_states, last_frame=last_frame)


class SpatialDecoder(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_layers, 
        num_heads,
        attn_head_dim,
        dropout,
        height,
        width,
        patch_bias,
        temporal_interpolation_scale,
        use_learned_positional_embeddings,
        patch_size,
        patch_size_t,
        sample_width,
        sample_height,
        sample_frames,
        temporal_compression_ratio,
        max_text_seq_length,
        spatial_interpolation_scale,
        in_channels: int = 16,
        out_channels: int = 16,
        use_rotary_positional_embeddings: bool = True,
        max_width: int = 32,    
        max_height: int = 32,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        timestep_activation_fn: str = 'silu',
        time_embed_dim: int = 512,
        spatial_num_frames: int = 1,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.attn_head_dim = attn_head_dim
        self.num_heads =num_heads
        inner_dim = num_heads * attn_head_dim
        self.transformer_blocks = nn.ModuleList([
            BasiceSpatialBlock(
                dim=inner_dim,
                num_attention_heads=num_heads,
                attention_head_dim=attn_head_dim,
                dropout=dropout,
                time_embed_dim=time_embed_dim,
                norm_eps=norm_eps,
                norm_elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(num_layers)
        ])
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.spatial_num_frames = spatial_num_frames
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.max_width = max_width
        self.max_height = max_height
    
        self.patch_embedder = MomoPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )

        self.image_rotary_emb = (
            _prepare_rotary_positional_embeddings(
                self.height, self.width, 
                self.spatial_num_frames, self.attn_head_dim, 
                self.patch_embedder.patch_size, self.patch_embedder.patch_size_t, 
                self.max_width, self.max_height,
            )
            if self.use_rotary_positional_embeddings
            else None
        )

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        output_dim = self.patch_embedder.patch_size * self.patch_embedder.patch_size * self.patch_embedder.patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value


    def prepare_latents(self, bsz, height, width, device, weight_type):
        """ input pixel level h&w """
        """ return B F C H W """
        return torch.randn(
            (bsz, self.spatial_num_frames, self.in_channels, height, width), 
            device=device, dtype=weight_type)

    def __call__(
        self,
        hidden_states, # B 1 C H W
        encoder_hidden_states, # B HW C
        timestep: int,
    ):
        timesteps = timestep
        t_emb = self.time_proj(timesteps).to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, condition=None)
    
        # B 1 C H W -> B HW C
        bsz, num_frames, _, height, width = hidden_states.shape
        cond_length = encoder_hidden_states.shape[1]
        hidden_states = self.patch_embedder(hidden_states)
        [rope.to(hidden_states.device) for rope in self.image_rotary_emb]

        for block_id, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    self.image_rotary_emb,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    self.image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, cond_length:]

        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        p = self.patch_embedder.patch_size
        p_t = self.patch_embedder.patch_size_t

        output = hidden_states.reshape(
            bsz, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        return MomoOutput(sample=output)
    
class MomoTransformer(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        patch_size,
        patch_size_t,
        text_embed_dim,
        patch_bias,
        sample_width,
        sample_height,
        sample_frames,
        temporal_compression_ratio,
        max_text_seq_length,
        spatial_interpolation_scale,
        temporal_interpolation_scale,
        use_rotary_positional_embeddings,
        use_learned_positional_embeddings,
        dropout,
        temporal_num_layers,
        spatial_num_layers,
        latent_space_channel: int = 16,
        max_height: int = 512,
        max_width: int = 512,
        spatial_num_frames: int = 1,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        timestep_activation_fn: str = 'silu',
        time_embed_dim: int = 512,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.text_embedder = TextEmbedder(text_embed_dim,inner_dim)

        self.temporal_encoder = TemporalEncoder(
            num_layers=temporal_num_layers,
            num_heads=num_attention_heads,
            attn_head_dim=attention_head_dim,
            dropout=dropout,
            text_length=max_text_seq_length,
            num_frames=(sample_frames-1)/temporal_compression_ratio + 1 if sample_frames%2 ==1 else sample_frames//temporal_compression_ratio,
            height=sample_height//spatial_interpolation_scale,
            width=sample_width//spatial_interpolation_scale,
            vae_scale_factor_spatial=spatial_interpolation_scale,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            max_height=max_height//spatial_interpolation_scale,
            max_width=max_width//spatial_interpolation_scale,
        )

        self.spatial_decoder = SpatialDecoder(
            num_layers=spatial_num_layers,
            num_heads=num_attention_heads,
            attn_head_dim=attention_head_dim,
            dropout=dropout,
            in_channels=latent_space_channel,
            out_channels=latent_space_channel,
            height=sample_height//spatial_interpolation_scale,
            width=sample_width//spatial_interpolation_scale,
            spatial_num_frames=spatial_num_frames,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            max_height=max_height//spatial_interpolation_scale,
            max_width=max_width//spatial_interpolation_scale,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            timestep_activation_fn=timestep_activation_fn,
            time_embed_dim=time_embed_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
            # for patch embedder
            patch_bias=patch_bias,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_interpolation_scale=temporal_interpolation_scale,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )


    def set_gradient_checkpointing(self):
        self.temporal_encoder.gradient_checkpointing = True
        self.spatial_decoder.gradient_checkpointing = True
