#!/bin/bash
#SBATCH --job-name=dev_pipeline_test
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8        # Reduziert für schnellere Zuteilung im Scheduler
#SBATCH --mem=32G                # Ausreichend für einen Testdurchlauf
#SBATCH --time=00:30:00          # Kurzes Limit für hohe Priorität in der Queue
#SBATCH --output=logs/dev_out_%j.txt
#SBATCH --error=logs/dev_err_%j.txt

# Module laden
module load devel/python/3.11
module load devel/cuda/12.1
module load lib/cudnn/8.9

# Arbeitsverzeichnis und Environment
cd ${HOME}/Music_Classification_Project
source .venv/bin/activate

echo "=== STARTE DEV PIPELINE TESTLAUF ==="

echo "[1/5] Teste Feature Extraktion..."
python3 scripts/prepare_dataset.py
if [ $? -ne 0 ]; then echo "Fehler in prepare_dataset.py"; exit 1; fi

echo "[2/5] Teste ResNet..."
python3 scripts/specialist_resnet.py
if [ $? -ne 0 ]; then echo "Fehler in specialist_resnet.py"; exit 1; fi

echo "[3/5] Teste LSTM..."
python3 scripts/specialist_lstm.py
if [ $? -ne 0 ]; then echo "Fehler in specialist_lstm.py"; exit 1; fi

echo "[4/5] Teste XGBoost..."
python3 scripts/specialist_xgboost.py
if [ $? -ne 0 ]; then echo "Fehler in specialist_xgboost.py"; exit 1; fi

echo "[5/5] Teste Committee Fusion..."
python3 scripts/committee_logic.py
if [ $? -ne 0 ]; then echo "Fehler in committee_logic.py"; exit 1; fi

echo "=== DEV PIPELINE ERFOLGREICH BEENDET ==="