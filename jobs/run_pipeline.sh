#!/bin/bash
#SBATCH --job-name=music_genre_ensemble
#SBATCH --output=logs/music_out_%j.txt
#SBATCH --error=logs/music_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

cd ..

source .venv/bin/activate

python3 scripts/feature_engineering.py
python3 scripts/specialist_resnet.py
python3 scripts/specialist_lstm.py
python3 scripts/specialist_xgboost.py
python3 scripts/committee_logic.py