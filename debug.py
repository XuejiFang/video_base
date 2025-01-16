import torch
from safetensors.torch import load_file

old_path = 'outputs/baseline_my/checkpoint-200/model/diffusion_pytorch_model.safetensors'
new_path = 'outputs/baseline_my/checkpoint-400/model/diffusion_pytorch_model.safetensors'

old = load_file(old_path)
new = load_file(new_path)

for k in old.keys():
    if k in new:
        if torch.allclose(old[k].float(), new[k].float()):
            print(f'{k} is same')
        else:
            pass
    else:
        print(f'{k} is missing from new')