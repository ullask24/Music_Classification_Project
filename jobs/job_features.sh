#!/bin/bash
#SBATCH --job-name=music_feat
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16000
#SBATCH --output=logs/feat_%j.txt
#SBATCH --error=logs/feat_%j.txt

cd ..
source .venv/bin/activate
python3 scripts/feature_engineering.py