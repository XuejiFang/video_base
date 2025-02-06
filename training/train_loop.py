from accelerate.utils import DistributedType
from tqdm import tqdm
import os
import shutil
import torch

class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss

def sync_gradients_info(args, accelerator, loss, lr_scheduler, progress_bar, progress_info, logger, loss_list=None):
    # Checks if the accelerator has performed an optimization step behind the scenes
    progress_bar.update(1)
    progress_info.global_step += 1
    accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
    if loss_list is not None:
        for i in range(len(loss_list)):
            accelerator.log({f"Frame-{i}": loss_list[i]}, step=progress_info.global_step)
    progress_info.train_loss = 0.0

    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        if progress_info.global_step % args.training_args.checkpointing_steps == 0:
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            if accelerator.is_main_process and args.training_args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.training_args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.training_args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.training_args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.training_args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(args.training_args.output_dir, f"checkpoint-{progress_info.global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
    progress_bar.set_postfix(**logs)

def train_diff_loop(
    args, accelerator, loss_func, model, ema_model, vae, text_encoder, dataloader, optimizer, lr_scheduler, initial_global_step, max_train_steps, logger, weight_dtype
):
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(initial_global_step, train_loss=0.0)

    while progress_info.global_step <= max_train_steps:
        for step, batch in enumerate(dataloader):
            """ [important] for gradient accumulation """
            with accelerator.accumulate(model): 
                x           = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids   = batch["input_ids"].to(accelerator.device).squeeze_(1)
                cond_mask   = batch["cond_mask"].to(accelerator.device).squeeze_(1)
                with torch.no_grad():
                    z_0             = vae.scale_(vae.encode(x).latent_dist.sample()) if x.shape[1] == 3 else vae.scale_(x)
                    prompt_embed    = text_encoder(input_ids, cond_mask)['last_hidden_state']

                loss, loss_list = loss_func(model, z_0, prompt_embed, cond_mask)
                avg_loss = accelerator.gather(loss).mean()
                progress_info.train_loss += avg_loss.detach().item() / args.training_args.grad_acc

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Update progress bar, ema model and save ckpts
                if accelerator.sync_gradients:
                    ema_model.step(model.parameters())
                    sync_gradients_info(args, accelerator, loss, lr_scheduler, progress_bar, progress_info, logger,
                                        loss_list=loss_list)