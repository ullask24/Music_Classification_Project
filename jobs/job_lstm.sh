#!/bin/bash
#SBATCH --job-name=music_lstm
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=logs/lstm_%j.txt

module load devel/python/3.11
cd ~/Music_Classification_Project
source .venv/bin/activate

python3 scripts/specialist_lstm.py