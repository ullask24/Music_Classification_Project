#!/bin/bash
#SBATCH --job-name=resnet_deep_finetune
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/resnet_deep_out_%j.txt
#SBATCH --error=logs/resnet_deep_err_%j.txt

module load devel/python/3.11
module load devel/cuda/12.8

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE TIEFES RESNET FINE-TUNING AUF H100 ==="
python3 scripts/train_resnet_finetune_deep.py
if [ $? -ne 0 ]; then echo "Fehler im tiefen Fine-Tuning"; exit 1; fi

echo "=== TIEFES FINE-TUNING ERFOLGREICH BEENDET ==="