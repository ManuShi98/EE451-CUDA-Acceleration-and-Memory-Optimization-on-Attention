# EE451-CUDA-Acceleration-and-Memory-Optimization-on-Attention
## Setup in CARC
A100 GPU is required.
```bash
module purge
module load gcc/11.3.0
module load python/3.9.12
salloc --partition=gpu --gres=gpu:a100:1 --time=01:00:00
module load nvidia-hpc-sdk
pip install triton

```