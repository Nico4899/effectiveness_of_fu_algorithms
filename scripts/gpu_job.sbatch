#!/bin/bash
#SBATCH --job-name=fl_gpu
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/fl_gpu_%j.out
#SBATCH --error=logs/fl_gpu_%j.err

module load cuda
module load Python/3.8
source ~/venv/Scripts/activate

# One server and three clients share GPUs
srun python src/runner.py --role=$( [ $SLURM_PROCID -eq 0 ] && echo server || echo client )