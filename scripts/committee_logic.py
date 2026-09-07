import os
import datetime
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/final_report_finetuned_ensemble_{timestamp}.txt"
matrix_filename = f"logs/ensemble_confusion_matrix_finetuned_{timestamp}.png"

print("=== STARTE ENSEMBLE MIT FINE-GETUNTEM RESNET ===")

# 1. Testdaten laden
X_spec = np.load("features/mel_spectrograms.npy")
X_seq = np.load("features/wav2vec_sequences.npy")
y = np.load("features/labels.npy")

# Kanäle für ResNet anpassen
if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

# Tabellarische Features extrahieren / laden (analog zu den Testdaten-Indizes)
import pandas as pd
df_tab = pd.read_csv("features/statistical_features.csv")
scaler = joblib.load("models/scaler.joblib")
X_tab_scaled = scaler.transform(df_tab.drop(columns=["label"]))

# Train-Test Split (gleicher Random State wie beim Training)
from sklearn.model_selection import train_test_split
_, X_spec_test, _, X_seq_test, _, X_tab_test, _, y_test = train_test_split(
    X_spec, X_seq, X_tab_scaled, y, test_size=0.2, random_state=42
)

# 2. Modelle laden (inklusive des neuen fine-getunten ResNet)
print("Lade Spezialisten...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")
lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras") # Das ist nun das fine-getunte Modell!

# 3. Vorhersagen berechnen
p_xgb = xgb_model.predict_proba(X_tab_test)
p_lstm = lstm_model.predict(X_seq_test, verbose=0)
p_resnet = resnet_model.predict(X_spec_test, verbose=0)

# Gewogenes Soft-Voting (Da das ResNet durch Fine-Tuning nun 80% statt 35% liefert, 
# können wir das Gewicht des ResNets im Ensemble von 0.1 auf 0.2 anheben!)
p_ensemble = (0.40 * p_xgb) + (0.40 * p_lstm) + (0.20 * p_resnet)
y_pred = p_ensemble.argmax(axis=1)

# Auswertung
accuracy = np.mean(y_pred == y_test) * 100
report = classification_report(y_test, y_pred)

print(f"FINAL FINE-TUNED ENSEMBLE ACCURACY: {accuracy:.2f}%")

# 4. Bericht und Dokumentation für die Präsentation abspeichern
with open(log_filename, "w") as f:
    f.write(f"--- ENSEMBLE EXPERIMENT MIT FINE-GETUNTEM RESNET ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN:\n")
    f.write("1. Verwendung des optimierten, fine-getunten ResNet50 (Accuracy von 35% auf 80% gesteigert).\n")
    f.write("2. Fine-Tuning Parameter: Letzte 30 Layer entfroren, Lernrate 1e-5.\n")
    f.write("3. Angepasstes Soft-Voting-Gewicht für ResNet aufgrund verbesserter Performance auf 0.20 erhöht (XGB: 0.40, LSTM: 0.40, ResNet: 0.20).\n\n")
    f.write("ZIELE:\n")
    f.write("- Maximierung der Gesamt-Accuracy durch Behebung der ResNet-Schwachstelle.\n")
    f.write(f"- Erzielte Ensemble Accuracy: {accuracy:.2f}%\n\n")
    f.write(report)

# Konfusionsmatrix plotten
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Fine-Tuned Ensemble Confusion Matrix ({accuracy:.2f}%)")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.savefig(matrix_filename)
plt.close()

print(f"Erfolgreich dokumentiert in: {log_filename}")