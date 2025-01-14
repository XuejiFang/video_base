import argparse
import yaml

import torch


from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

from diffusers.utils import export_to_video
from pathlib import Path

from utils import dict_to_namespace, get_instance, register_module, freeze_module

    # from PIL import Image
    # import numpy as np
    # generator = torch.Generator(device=device)
    # x = torch.as_tensor(np.asarray(Image.open('./tmp.png').convert('RGB'))).to(device, dtype=weight_dtype)
    # x = x.sub(127.5).div_(127.5).permute(2, 0, 1).unsqueeze_(0).unsqueeze_(2)
    # with torch.no_grad():
    #     latent = vae.scale_(vae.encode(x).latent_dist.sample(generator))
    #     x_rec = vae.decode(vae.unscale_(latent)).sample
    # x_rec = x_rec.squeeze_(2).squeeze_(0).permute(1,2,0).mul_(127.5).add(127.5).float().cpu().numpy().astype(np.uint8)
    # x_rec = Image.fromarray(x_rec)
    # x_rec.save('./tmp_rec.png')

@torch.no_grad()
def generate_videos(accelerator, vae, tokenizer, text_encoder, transformer, device, weight_dtype):
    from cogvideo.pipelines import CogVideoXPipeline
    from diffusers import CogVideoXDDIMScheduler
    import os
    scheduler = CogVideoXDDIMScheduler.from_config('/storage/qiguojunLab/qiguojun/home/Models/THUDM/CogVideoX-2b/scheduler/')
    pipe = CogVideoXPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, 
                             transformer=transformer, scheduler=scheduler).to(device, dtype=weight_dtype)

    prompts_paths = [
        # === VBench ===
        'prompts/vbench/appearance_style.txt',
        'prompts/vbench/color.txt',
        'prompts/vbench/human_action.txt',
        'prompts/vbench/multiple_objects.txt',
        'prompts/vbench/object_class.txt',
        'prompts/vbench/overall_consistency.txt',
        'prompts/vbench/scene.txt',
        'prompts/vbench/spatial_relationship.txt',
        'prompts/vbench/subject_consistency.txt',
        'prompts/vbench/temporal_flickering.txt',
        'prompts/vbench/temporal_style.txt',
    ]

    save_root = "outputs_eval/vbench"
    for prompt_path in prompts_paths:
        with open(prompt_path, 'r') as f:
            prompts = f.readlines()
            for i, prompt in enumerate(prompts):
                save_dir = os.path.join(save_root, prompt_path.split('/')[-1].split('.')[0])
                os.makedirs(save_dir, exist_ok=True)
                if i % accelerator.num_processes != accelerator.process_index:
                    continue
                prompt = prompt.strip()
                for j in range(5):
                    video = pipe(prompt).frames[0]
                    export_to_video(video, os.path.join(save_dir, f"{prompt}-{j}.mp4"), fps=8)

def main(args):
    # 1. Initialize Accelerator
    training_args = args.training_args
    logging_dir = Path(training_args.output_dir, training_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
    )

    device = accelerator.device
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)

    # 2. Load and Create Models
    vae             = register_module(args.vae, dtype=weight_dtype).to(device)
    tokenizer       = register_module(args.tokenizer)
    text_encoder    = register_module(args.text_encoder, dtype=weight_dtype).to(device)
    freeze_module(vae, trainable=False)
    freeze_module(text_encoder, trainable=False)

    transformer     = get_instance(args.transformer).to(device)
    from safetensors.torch import load_file
    states = load_file('/storage/qiguojunLab/qiguojun/home/Models/THUDM/CogVideoX-2b/transformer/diffusion_pytorch_model.safetensors')
    transformer.load_state_dict(states)
    print(f"Total GPUS: {accelerator.num_processes}, Current GPU: {accelerator.process_index}")
    generate_videos(accelerator, vae, tokenizer, text_encoder, transformer, device, weight_dtype)
    # 3. Prepare Optimizer

    # 4. Prepare Dataloader

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with dynamic config.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    
    main(dict_to_namespace(config))