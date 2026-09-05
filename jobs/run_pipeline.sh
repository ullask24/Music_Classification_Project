#!/bin/bash
#SBATCH --job-name=music_genre_ensemble
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/music_out_%j.txt
#SBATCH --error=logs/music_err_%j.txt

# Korrigierte Modul-Pfade für bwUniCluster 3.0
module load devel/python/3.11
module load devel/cuda/12.8

# Arbeitsverzeichnis und Environment
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE PRODUKTIVE TRAINING-PIPELINE ==="

echo "[1/5] Extrahiere Features (Tabellarisch, Mel-Spec, Wav2Vec)..."
python3 scripts/prepare_dataset.py

echo "[2/5] Trainiere ResNet-Spezialist (2D)..."
python3 scripts/specialist_resnet.py

echo "[3/5] Trainiere LSTM-Spezialist (3D Sequenzen)..."
python3 scripts/specialist_lstm.py

echo "[4/5] Trainiere XGBoost-Spezialist (Statistik)..."
python3 scripts/specialist_xgboost.py

echo "[5/5] Führe Committee Logic (Soft-Voting) aus..."
python3 scripts/committee_logic.py

echo "=== PIPELINE ERFOLGREICH BEENDET ==="