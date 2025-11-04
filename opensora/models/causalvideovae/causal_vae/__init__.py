from .modeling_causalvae import CausalVAEModel

from einops import rearrange
from torch import nn
import torch

class CausalVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)
        self.vae.enable_tiling()
        self.vae.tile_overlap_factor = 0.125
        self.vae.tile_sample_min_size = 512
        self.vae.tile_latent_min_size = 64
        self.vae.tile_sample_min_size_t = 29
        self.vae.tile_latent_min_size_t = 8
        self.vae_scale_factor = [4,8,8]
    @torch.no_grad()
    def encode(self, x):  # b c t h w
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    @torch.no_grad()
    def decode(self, x):
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype
