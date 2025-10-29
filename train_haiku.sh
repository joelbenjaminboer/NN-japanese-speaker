#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --output=habrok_outputs/%j.out
#SBATCH --partition=digitallab
#SBATCH --gpus-per-node=h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8000
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load Boost/1.79.0-GCC-11.3.0

source .venv/bin/activate

python main.py