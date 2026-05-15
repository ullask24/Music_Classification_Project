#!/bin/bash
#SBATCH --job-name=music_feat
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16000
#SBATCH --output=logs/feat_%j.txt

module load devel/python/3.11
cd ~/Music_Classification_Project
source .venv/bin/activate

python3 scripts/feature_engineering.py