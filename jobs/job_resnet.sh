#!/bin/bash
#SBATCH --job-name=music_resnet
#SBATCH --partition=gpu_4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64000
#SBATCH --output=logs/resnet_%j.txt
#SBATCH --error=logs/resnet_%j.txt

cd ..
source .venv/bin/activate
python3 scripts/specialist_ResNet.py