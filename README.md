# EE451-CUDA-Acceleration-and-Memory-Optimization-on-Attention
## Setup in CARC
Post-Ampere GPUs is required for triton.
salloc --partition=gpu --gres=gpu:a100:1 --time=01:00:00 --mem=16GB --cpus-per-task=4


```bash
module purge
module load conda
module load gcc/11.3.0 
module load cmake/3.23.2
module load tmux/3.3a
module load cuda/11.6.2
module load git/2.36.1

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

only first time
    conda init bash
    source ~/.bashrc
    mamba create --name triton
    conda activate triton
    pip install triton
    conda install nltk matplotlib
conda activate triton

```
sometimes it does not enter the conda environment, conda deactivate and activate environment.

download dataset from https://www.kaggle.com/datasets/bittlingmayer/amazonreviews, unzip them into ./data
prepare_npy.py, benchmark_train_*.py edited from https://github.com/iamirmasoud/amazon_sentiment/tree/main

Execute in order:
```
mkdir data
python prepare_npy.py
python benchmark_train.py --attention_type cpp/cuda/fused
```

