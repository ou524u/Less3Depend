import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from src.setup import init_config, init_distributed
from src.utils.metric_utils import export_metrics
from tqdm import tqdm

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
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


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()



# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)


if ddp_info.is_main_process:  
    print(f"Running fast inference, computing metrics...")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

metrics_sum = {"psnr": 0.0, "lpips": 0.0, "ssim": 0.0, "count": 0}

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for eval_batch in tqdm(dataloader, desc="Evaluating", leave=False):
        eval_batch = {k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
        result = model(eval_batch, target_has_input=False)
        if result is None:
            continue

        batch_metrics = export_metrics(result)

        for k, v in batch_metrics.items():
            metrics_sum[k] += v

# Synchronize all processes
dist.barrier()

# Aggregate across processes (all_reduce)
for k, v in metrics_sum.items():
    tensor = torch.tensor(v, device=ddp_info.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    metrics_sum[k] = tensor.item()

# Calculate averages (divide by total count)
for k in metrics_sum.keys():
    if k != "count":
        metrics_sum[k] /= metrics_sum["count"]

metrics_to_log = {f"eval/{k}": v for k, v in metrics_sum.items()}

if ddp_info.is_main_process:
    print(f"Evaluation metrics: {metrics_to_log}")

dist.destroy_process_group()
exit(0)