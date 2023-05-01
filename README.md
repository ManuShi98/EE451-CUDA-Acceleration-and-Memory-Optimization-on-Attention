# EE451-CUDA-Acceleration-and-Memory-Optimization-on-Attention
## Setup in CARC
Post-Ampere GPUs is required for triton.
salloc --partition=gpu --gres=gpu:a40:1 --time=01:00:00 --mem=16GB --cpus-per-task=4


```bash
module purge
module load conda
module load gcc/11.3.0 
module load cmake/3.23.2
module load tmux/3.3a
module load cuda/11.6.2
module load git/2.36.1
conda activate triton

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

only first time
    conda init bash
    source ~/.bashrc
    mamba create --name triton
    conda activate triton
    pip install triton
    conda install nltk matplotlib jupyter

```
Sometimes it does not enter the conda environment, deactivating and then reactivating the conda environment may help.

block/ contains the standard block matrix multiplication implementation.
flash/ contains the memory-optimized block matrix multiplication implementation.

Use following demands to benchmark:
```
python benchmark.py --attention_type block/flash
```

Use NVIDIA Nsight Compute to profile the code.
```
cat /proc/sys/kernel/perf_event_paranoid
sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid'
sudo /usr/local/NVIDIA-Nsight-Compute-2023.1/ncu -f --set full -o block_report /home/scruple/miniconda3/envs/triton/bin/python benchmark.py --attention_type flash
sudo /usr/local/NVIDIA-Nsight-Compute-2023.1/ncu -f --set full -o flash_report /home/scruple/miniconda3/envs/triton/bin/python benchmark.py --attention_type block
/usr/local/NVIDIA-Nsight-Compute-2023.1/ncu-ui $report_file$
```
