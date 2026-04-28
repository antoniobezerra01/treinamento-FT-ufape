#!/bin/bash
#SBATCH --job-name=TreinoPNLD
#SBATCH --output=logs/treino_mult%j.out
#SBATCH --error=logs/treino_mult%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unsloth_env

# Variáveis de otimização
export HF_DATASETS_NUM_PROC=1
export OMP_NUM_THREADS=1

echo "Iniciando treinamento na GPU: $(hostname)"
echo "Data/Hora: $(date)"
nvidia-smi

python -u treino_mult_modelos_c1.py

echo "JOB FINALIZADO! Data/Hora: $(date)"