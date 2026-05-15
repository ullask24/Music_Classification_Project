#!/bin/bash
#SBATCH --job-name=music_xgb
#SBATCH --output=logs/xgb_%j.txt
#SBATCH --error=logs/xgb_%j.txt
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd ..
source .venv/bin/activate
python3 scripts/specialist_xgboost.py