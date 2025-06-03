import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from src.setup import init_config, init_distributed, init_wandb_and_backup
from src.utils.metric_utils import visualize_intermediate_results, export_metrics
from src.utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from tqdm import tqdm

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

# eval dataset
eval_dataset = Dataset(config, mode='eval')
# print(f"evaluation dataset length {len(eval_dataset)}")
datasampler = DistributedSampler(eval_dataset)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
eval_dataloader_iter = iter(eval_dataloader)

# stage_switch_step
stage_switch_step = config.get("unposed", {}).get("stage_switch_step", 200000)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))


module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

# # debug print model parameters
# def print_model_parameters(model):
#     print("Model parameters with indices:")
#     for idx, (name, param) in enumerate(model.named_parameters()):
#         print(f"Index {idx}: {name}, Shape: {param.shape}, Device: {param.device}")
# print_model_parameters(model)
##

model = DDP(model, device_ids=[ddp_info.local_rank])


optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)


if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
reset_lr_scheduler = config.training.get("reset_lr_scheduler", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    reset_lr_scheduler,
)


enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        # if start_train_step == cur_train_step:
        #     print(f"Current Rank {ddp_info.local_rank} Restarting training from step {cur_train_step}. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        #     datasampler.set_epoch(cur_epoch)
        #     dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}

    # eval during training
    if cur_train_step % config.training.eval_every == 0 and cur_train_step != 0:
        model.eval()
        
        metrics_sum = {"psnr": 0.0, "lpips": 0.0, "ssim": 0.0, "count": 0}
                    
        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            for eval_batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                eval_batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in eval_batch.items()}
                result = model(eval_batch, target_has_input=False)
                if result is None:
                    continue
                batch_metrics = export_metrics(result)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    metrics_sum[k] += v

            dist.barrier()
            # all reduce metrics
            for k, v in metrics_sum.items():
                tensor = torch.tensor(v, device=ddp_info.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics_sum[k] = tensor.item()

            # Compute averages across all batches
            for k in metrics_sum.keys():
                if k != "count":
                    metrics_sum[k] /= metrics_sum["count"]
            
            metrics_to_log = {f"eval/{k}": v for k, v in metrics_sum.items()}

            # Log to wandb
            if ddp_info.is_main_process:
                wandb.log(
                    metrics_to_log,
                    step=cur_train_step,
                )
            if ddp_info.is_main_process:
                print(f"Evaluation metrics: {metrics_to_log}")
        
        model.train()

    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        if cur_train_step <= stage_switch_step:
            ret_dict = model(batch)
            if ret_dict is None:
                continue
        else:
            ret_dict = model(batch, from_camera=True)
            if ret_dict is None:
                continue

    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

    skip_optimizer_step = False
    # Skip optimizer step if loss is NaN or Inf
    if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
        print(f"NaN or Inf loss detected, skip this iteration")
        skip_optimizer_step = True
        ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

    total_grad_norm = None
    # Check gradient norm and update optimizer if everything is fine
    if update_grads and (not skip_optimizer_step):
        # Unscales the gradients
        scaler.unscale_(optimizer) 
        # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
        with torch.no_grad():
            for n, p in optimized_param_dict.items():
                if p.requires_grad and (p.grad is not None):
                    p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
    
        # visualize the grad norm of each layer of our transformer (FOR DEBUG)
        if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
            grad_norms = {}  # Dictionary to store norms per layer
            for name, param in model.named_parameters():
                if param.grad is not None:  # Some parameters might not have gradients
                    grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
            for layer_name, grad_norm in grad_norms.items():
                wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

        total_grad_norm = 0.0
        if config.training.grad_clip_norm > 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

            # if total_grad_norm > config.training.grad_clip_norm * 2.0:
            #     print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

            allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
            if total_grad_norm > allowed_gradnorm:
                skip_optimizer_step = True
                print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

            # show grad norm in wandb if it's too large
            display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
            if display_grad_norm and ddp_info.is_main_process:
                wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

        if not skip_optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            cur_param_update_step += 1

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # log and save checkpoint
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forward step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            checkpoint_dir = config.training.checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_dir = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            vis_dir = vis_dir
            os.makedirs(vis_dir, exist_ok=True)
            visualize_intermediate_results(vis_dir, ret_dict)
            torch.cuda.empty_cache()
            model.train()
    
    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()
        

dist.barrier()
dist.destroy_process_group()
