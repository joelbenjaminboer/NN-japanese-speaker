#!/bin/bash
#SBATCH --job-name=aug_optuna
#SBATCH --output=habrok_outputs_v2/%j.out
#SBATCH --error=habrok_outputs_v2/%j.err
#SBATCH --partition=digitallab
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load Boost/1.79.0-GCC-11.3.0

source .venv/bin/activate

# Single worker configuration
echo "Starting optimization on $(hostname)"
python -u exp_augmentation.py