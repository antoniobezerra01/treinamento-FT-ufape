#!/bin/bash
#SBATCH --job-name=ExportGGUF
#SBATCH --output=logs/export_%j.out
#SBATCH --partition=basic
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unsloth_env

python exportar_gguf.py