#!/bin/bash
#SBATCH --job-name=music_genre_ensemble
#SBATCH --output=logs/music_out_%j.txt
#SBATCH --error=logs/music_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dev_gpu_h100

module load devel/python/3.11
module load devel/cuda/12.1
module load lib/cudnn/8.9

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

# 1. Parallele Extraktion aller 3 Daten-Typen
python3 scripts/prepare_dataset.py

# 2. Training der drei unabhängigen Experten
python3 scripts/specialist_resnet.py
python3 scripts/specialist_lstm.py
python3 scripts/specialist_xgboost.py

# 3. Ensemble Soft-Voting und Evaluation
python3 scripts/committee_logic.py