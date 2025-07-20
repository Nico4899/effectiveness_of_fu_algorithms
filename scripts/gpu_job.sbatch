#!/bin/bash
#SBATCH --job-name=fl_gpu
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/fl_gpu_%j.out
#SBATCH --error=logs/fl_gpu_%j.err

module load devel/cuda/12.8   
module load devel/python/3.11.7-gnu-11.4
source ~/venv/bin/activate

# One server and three clients share GPUs
# srun python src/run_federated.py \
#     --role=$( [ $SLURM_PROCID -eq 0 ] && echo server || echo client ) \
#     --cid $SLURM_PROCID

# Run a single experiment on one GPU
srun python src/experiment_runner.py data/texas100_subset.npz