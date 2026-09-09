#!/bin/bash
#SBATCH --job-name=live_demo_seed_vs_v3
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/live_demo_out_%j.txt
#SBATCH --error=logs/live_demo_err_%j.txt

module load devel/python/3.11
module load devel/cuda/12.8

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE SEED-PAPER VS V3 ENSEMBLE LIVE DEMO AUF DEV H100 GPU ==="
python3 scripts/live_demo_with_seed.py
if [ $? -ne 0 ]; then echo "Fehler im Live-Demo-Skript"; exit 1; fi

echo "=== LIVE DEMO ERFOLGREICH BEENDET ==="