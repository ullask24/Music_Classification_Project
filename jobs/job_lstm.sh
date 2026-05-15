#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32000
#SBATCH --output=logs/lstm_out_%j.txt

module load devel/python/3.11
module load devel/cuda/12.1
module load lib/cudnn/8.9

cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

python3 scripts/specialist_lstm.py