#!/bin/bash
#SBATCH --job-name=music_xgb
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16000
#SBATCH --output=logs/xgb_%j.txt
#SBATCH --error=logs/xgb_%j.txt

cd ..
source .venv/bin/activate
python3 scripts/specialist_xgboost.py