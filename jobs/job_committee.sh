#!/bin/bash
#SBATCH --job-name=music_final
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=8000
#SBATCH --output=logs/committee_%j.txt
#SBATCH --error=logs/committee_%j.txt

cd ..
source .venv/bin/activate
python3 scripts/committee_logic.py