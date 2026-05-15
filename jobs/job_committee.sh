#!/bin/bash
#SBATCH --job-name=final_committee
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=8000
#SBATCH --output=logs/committee_%j.txt

module load devel/python/3.11
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

python3 scripts/committee_logic.py