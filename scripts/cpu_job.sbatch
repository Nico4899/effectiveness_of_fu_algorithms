#!/bin/bash
#SBATCH --job-name=fl_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/fl_cpu_%j.out
#SBATCH --error=logs/fl_cpu_%j.err

module load devel/python/3.11.7-gnu-11.4
source ~/venv/bin/activate

# Launch 1 server (PROCID=0) and 10 clients
# srun --ntasks=11 python src/run_federated.py \
#     --role=$( [ $SLURM_PROCID -eq 0 ] && echo server || echo client ) \
#     --cid $SLURM_PROCID

# Run a single experiment
srun python src/experiment_runner.py data/texas100_subset.npz

