#!/bin/bash
#SBATCH --job-name=music_feat
#SBATCH --output=logs/feat_%j.txt
#SBATCH --error=logs/feat_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd ..
source .venv/bin/activate
python3 scripts/feature_engineering.py