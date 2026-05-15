#!/bin/bash
#SBATCH --job-name=music_resnet
#SBATCH --output=logs/resnet_%j.txt
#SBATCH --error=logs/resnet_%j.txt
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

cd ..
source .venv/bin/activate
python3 scripts/specialist_ResNet.py