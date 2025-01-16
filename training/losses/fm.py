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