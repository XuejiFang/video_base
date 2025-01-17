import torch
from safetensors.torch import load_file

old_path = 'outputs/cog_t2i_1b-tmp/checkpoint-10/model_ema/diffusion_pytorch_model.safetensors'
new_path = 'outputs/cog_t2i_1b-tmp/checkpoint-50/model_ema/diffusion_pytorch_model.safetensors'

old = load_file(old_path)
new = load_file(new_path)

for k in old.keys():
    if k in new:
        if torch.allclose(old[k].float(), new[k].float()):
            print(f'{k} is same')
            # pass
        else:
            pass
            # print(f"{k} is not same")
    else:
        print(f'{k} is missing from new')