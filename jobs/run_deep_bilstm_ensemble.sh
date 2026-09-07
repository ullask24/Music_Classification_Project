#!/bin/bash
#SBATCH --job-name=v3_ensemble
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v3_ensemble_out_%j.txt
#SBATCH --error=logs/v3_ensemble_err_%j.txt

module load devel/python/3.11
module load devel/cuda/12.8

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE FINALES ENSEMBLE V3 AUF DEV H100 GPU ==="
python3 scripts/committee_logic_v3.py
if [ $? -ne 0 ]; then echo "Fehler im V3 Ensemble-Skript"; exit 1; fi

echo "=== V3 ENSEMBLE ERFOLGREICH BEENDET ==="