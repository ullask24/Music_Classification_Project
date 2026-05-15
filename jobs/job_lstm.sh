#!/bin/bash
#SBATCH --job-name=music_lstm
#SBATCH --partition=gpu_4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=logs/lstm_%j.txt
#SBATCH --error=logs/lstm_%j.txt

cd ..
source .venv/bin/activate
python3 scripts/specialist_lstm.py