#!/bin/bash
#SBATCH --job-name=dev_resnet
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64000
#SBATCH --output=logs/resnet_%j.txt

module load devel/python/3.11
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

python3 scripts/specialist_ResNet.py