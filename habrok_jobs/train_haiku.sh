#!/bin/bash
#SBATCH --job-name=optuna_haiku
#SBATCH --output=habrok_outputs/%A_%a.out
#SBATCH --error=habrok_outputs/%A_%a.err
#SBATCH --partition=digitallab
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --gpus-per-node=h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load Boost/1.79.0-GCC-11.3.0

source .venv/bin/activate

echo "Starting Optuna worker $SLURM_ARRAY_TASK_ID"
python -u parrellel_tuning.py
