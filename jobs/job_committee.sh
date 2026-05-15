#!/bin/bash
#SBATCH --job-name=music_final
#SBATCH --output=logs/committee_%j.txt
#SBATCH --error=logs/committee_%j.txt
#SBATCH --mem=8G
#SBATCH --time=00:15:00

cd ..
source .venv/bin/activate
python3 scripts/committee_logic.py