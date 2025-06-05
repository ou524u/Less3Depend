# Less3Depend

This repository contains the PyTorch implementation of the paper **"The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge"**.

## 1. Preparation

### Environment Setup

Create and activate conda environment:

```bash
conda create -n lvsm python=3.11
conda activate lvsm
pip install -r requirements.txt
```

*Recommended*: GPU device with compute capability > 8.0. We used 8*A100 GPUs in our experiments.

### Dataset Setup

We use RealEstate10K dataset from [pixelSplat](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets), and followed [LVSM](https://github.com/haian-jin/LVSM) to do the preprocessing.

Download and unzip RealEstate10K `.torch` chunks. For our scaling experiments, we split the dataset into 4 sizes, each containing the number of chunks listed below:

| Size | Chunks | Scenes |
|------|--------|--------|
| Little | 76 | 1,202 |
| Medium | 304 | 4,121 |
| Large | 1,216 | 16,449 |
| Full | 4,866 | 66,033 |

Process the dataset following [LVSM](https://github.com/Haian-Jin/LVSM/blob/main/process_data.py):

```bash
# process training split
python process_data.py --base_path datasets/re10k --output_dir datasets/re10k-full_processed --mode train --num_processes 80

# process test split
python process_data.py --base_path datasets/re10k --output_dir datasets/re10k-full_processed --mode test --num_processes 80
```

## 2. Evaluation

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1PMEl0RoOwi2wlsMRv6K9YSfeq-KqVYbz/view?usp=sharing), or with the following command:

```bash
# download pre-trained model
mkdir -p checkpoints/uplvsm
gdown 1PMEl0RoOwi2wlsMRv6K9YSfeq-KqVYbz -O checkpoints/uplvsm/uplvsm_x224.pt
```

Run evaluation:

```bash
# fast inference, compute metrics only
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.inference_fast --config config/eval/uplvsm_x224.yaml

# complete inference
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.inference --config config/eval/uplvsm_x224.yaml
```

~~> 📝 **TODO**: Release fine-tuned uplvsm model with 518×518 resolution.~~

✅ Download uplvsm model with 518×518 resolution from [Google Drive](https://drive.google.com/file/d/1DiLCEzHbxtusvA6ic6IhpYuhD93PUjJw/view?usp=sharing), and run evaluation:

```bash
# fast inference, compute metrics only
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.inference_fast --config config/eval/uplvsm_x518.yaml

# complete inference
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.inference --config config/eval/uplvsm_x518.yaml
```

## 3. Training

```bash
# pretraining on 224×224 resolution
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.train --config config/uplvsm_x224.yaml

# finetuning on 518×518 resolution
torchrun --nproc_per_node 8 --nnodes 1 --rdzv_id 18640 --rdzv_backend c10d --rdzv_endpoint localhost:29511 -m src.train --config config/uplvsm_x518.yaml
```

## 📄 Acknowledgments

Our implementation builds upon [LVSM](https://github.com/haian-jin/LVSM). We also recommend [RayZer](https://github.com/hwjiang1510/RayZer) for self-supervised scene reconstruction.

If you find this work useful for your research, please consider citing:

```bibtex
@article{your_paper_2024,
  title={The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge},
  author={Your Authors},
  journal={Your Journal/Conference},
  year={2024}
}
```
