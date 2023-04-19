# EE451-CUDA-Acceleration-and-Memory-Optimization-on-Attention
## Setup in CARC
Post-Ampere GPUs is required for triton.
```bash
conda init bash
source ~/.bashrc
mamba create --name triton
conda activate triton
pip install triton

module purge
module load conda
module load gcc/11.3.0 
module load cmake/3.23.2
module load tmux/3.3a
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load git/2.36.1


```

