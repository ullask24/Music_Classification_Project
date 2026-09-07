#!/bin/bash
#SBATCH --job-name=resnet_finetune
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/resnet_finetune_out_%j.txt
#SBATCH --error=logs/resnet_finetune_err_%j.txt

# Module laden
module load devel/python/3.11
module load devel/cuda/12.8

# Arbeitsverzeichnis und Environment
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE RESNET FINE-TUNING AUF H100 GPU ==="
python3 scripts/train_resnet_finetune.py
if [ $? -ne 0 ]; then echo "Fehler in train_resnet_finetune.py"; exit 1; fi

echo "=== RESNET FINE-TUNING ERFOLGREICH BEENDET ==="