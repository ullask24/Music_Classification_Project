#!/bin/bash
#SBATCH --job-name=music_resnet
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=64000
#SBATCH --output=logs/resnet_%j.txt

module load devel/python/3.11
cd ~/Music_Classification_Project
source .venv/bin/activate

python3 scripts/specialist_ResNet.py