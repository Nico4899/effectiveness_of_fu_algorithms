#!/bin/bash
#SBATCH --job-name=fl_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/fl_cpu_%j.out
#SBATCH --error=logs/fl_cpu_%j.err

module load Python/3.8
source ~/venv/bin/activate

# Launch 1 server (PROCID=0) and 10 clients
srun --ntasks=11 python src/experiment_runner.py --role=$( [ $SLURM_PROCID -eq 0 ] && echo server || echo client )

