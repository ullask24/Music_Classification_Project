#!/bin/bash
#SBATCH --job-name=deep_ensemble
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/deep_ensemble_out_%j.txt
#SBATCH --error=logs/deep_ensemble_err_%j.txt

module load devel/python/3.11
module load devel/cuda/12.8

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE FINALES ENSEMBLE MIT TIEFEM RESNET ==="
python3 scripts/committee_logic_deep.py
if [ $? -ne 0 ]; then echo "Fehler im Ensemble-Skript"; exit 1; fi

echo "=== ENSEMBLE ERFOLGREICH BEENDET ==="