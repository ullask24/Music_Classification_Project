#!/bin/bash
#SBATCH --job-name=lstm_bilstm
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lstm_bilstm_out_%j.txt
#SBATCH --error=logs/lstm_bilstm_err_%j.txt

module load devel/python/3.11
module load devel/cuda/12.8

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE BiLSTM TRAINING AUF DEV H100 GPU ==="
python3 scripts/specialist_lstm_bilstm.py
if [ $? -ne 0 ]; then echo "Fehler im BiLSTM-Skript"; exit 1; fi

echo "=== BiLSTM TRAINING ERFOLGREICH BEENDET ==="