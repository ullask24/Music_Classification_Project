#!/bin/bash
#SBATCH --job-name=xgb
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16000
#SBATCH --output=logs/xgb_%j.txt

module load devel/python/3.11
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

python3 scripts/specialist_xgboost.py