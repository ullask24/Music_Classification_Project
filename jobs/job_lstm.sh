#!/bin/bash
#SBATCH --job-name=music_lstm
#SBATCH --output=logs/lstm_%j.txt
#SBATCH --error=logs/lstm_%j.txt
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

cd ..
source .venv/bin/activate
python3 scripts/specialist_lstm.py