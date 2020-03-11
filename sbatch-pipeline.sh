#!/bin/sh
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH -p main,viz
#SBATCH --account=snap

srun ./run-pipeline "$@"
