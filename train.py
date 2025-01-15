import argparse
import diffusers
import logging
import os
import yaml
import torch
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from copy import deepcopy
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import export_to_video
from pathlib import Path
from utils import dict_to_namespace, get_class, get_instance, register_module, freeze_module
from training.train_loop import train_diff_loop

@torch.no_grad()
def generate_videos(prompts, vae, tokenizer, text_encoder, transformer, device, weight_dtype):
    from cogvideo.pipelines import CogVideoXPipeline
    from diffusers import CogVideoXDDIMScheduler
    scheduler = CogVideoXDDIMScheduler.from_config('/storage/qiguojunLab/qiguojun/home/Models/THUDM/CogVideoX-2b/scheduler/')
    pipe = CogVideoXPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, 
                             transformer=transformer, scheduler=scheduler).to(device, dtype=weight_dtype)
    video = pipe(prompts).frames[0]
    export_to_video(video, './tmp-30.mp4')

def main(args):
    # 1. Initialize Accelerator
    train_args = args.training_args
    logging_dir = Path(train_args.output_dir, train_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=train_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.grad_acc,
        mixed_precision=train_args.mixed_precision,
        log_with=train_args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name=train_args.exp_name,
        config={"dropout": 0.1, "learning_rate": 1e-2},
        init_kwargs={"wandb": {"entity": train_args.team_name}}
        )
    train_args.output_dir = Path(train_args.output_dir, train_args.exp_name)
    if accelerator.is_main_process and train_args.output_dir: os.makedirs(train_args.output_dir, exist_ok=True)
    device = accelerator.device
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    logger = get_logger(__name__)
    logger.info("***** Initalized Accelerator *****")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # 2. Load and Create Models
    logger.info("***** Loading and Creating Models *****")
    vae             = register_module(args.vae, dtype=weight_dtype).to(device)
    tokenizer       = register_module(args.tokenizer)
    text_encoder    = register_module(args.text_encoder, dtype=weight_dtype).to(device)
    vae.enable_slicing(), vae.enable_tiling()
    freeze_module(vae, trainable=False)
    freeze_module(text_encoder, trainable=False)

    model = get_instance(args.transformer).to(device, dtype=weight_dtype)
    model.train()
    model.gradient_checkpointing = train_args.gradient_checkpointing

    logger.info("***** Creating EMA Model *****")
    ema_model = deepcopy(model)
    ema_model = EMAModel(
        ema_model.parameters(), decay=train_args.ema_decay, update_after_step=train_args.ema_start_step,
        model_cls=type(model), model_config=ema_model.config
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "model"))
                if weights:  # Don't pop if empty
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

    def load_model_hook(models, input_dir):
        load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), model_cls=get_class(args.transformer))
        ema_model.load_state_dict(load_model.state_dict())
        ema_model.to(accelerator.device)
        del load_model
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            load_model = type(model).from_pretrained(input_dir, subfolder="model")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    # 3. Prepare Optimizer and Dataloader
    logger.info("***** Preparing Optimizer and Dataloader *****")
    params_to_optimize = model.parameters()
    optimizer   = get_instance(args.optimizer, params=params_to_optimize)
    dataloader  = get_instance(args.dataloader, tokenizer=tokenizer)
    dataloader, dataset_length = dataloader.dataloader, dataloader.dataset_length
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.lr_warmup_steps * train_args.grad_acc,
        num_training_steps=train_args.max_train_steps * train_args.grad_acc,
    )
    # 5. Training Loop
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    total_batch_size = args.dataloader.params.batch_size * accelerator.num_processes * train_args.grad_acc
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
    logger.info(f"  Num examples = {dataset_length}")
    logger.info(f"  Instantaneous batch size per device = {args.dataloader.params.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.grad_acc}")
    logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")

    if train_args.resume_from_checkpoint:
        if train_args.resume_from_checkpoint != "latest":
            path = os.path.basename(train_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(train_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{train_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(train_args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
    else:
        initial_global_step = 0

    loss_func = get_instance(args.loss_function)
    train_diff_loop(args, accelerator, loss_func, 
                    model, ema_model, vae, text_encoder, 
                    dataloader, optimizer, 
                    lr_scheduler, initial_global_step, train_args.max_train_steps, logger, weight_dtype)

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with dynamic config.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    
    main(dict_to_namespace(config))