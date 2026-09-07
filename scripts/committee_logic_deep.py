import os
import datetime
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/final_report_deep_finetuned_ensemble_{timestamp}.txt"
matrix_filename = f"logs/ensemble_confusion_matrix_deep_finetuned_{timestamp}.png"

print("=== STARTE ENSEMBLE MIT TIEF-FINE-GESUNTEM RESNET ===")

# 1. Testdaten laden
X_spec = np.load("features/mel_spectrograms.npy")
X_seq = np.load("features/wav2vec_sequences.npy")
y = np.load("features/labels.npy")

# Kanäle für ResNet anpassen
if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

# Tabellarische Features laden und skalieren
df_tab = pd.read_csv("features/statistical_features.csv")
scaler = joblib.load("models/scaler.joblib")
X_tab_scaled = scaler.transform(df_tab.drop(columns=["label"]))

# Train-Test Split (identischer Random State)
_, X_spec_test, _, X_seq_test, _, X_tab_test, _, y_test = train_test_split(
    X_spec, X_seq, X_tab_scaled, y, test_size=0.2, random_state=42
)

# 2. Spezialisten laden (Verwendung des tiefen Fine-Tuning ResNets)
print("Lade Spezialisten...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")
lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")

# 3. Vorhersagen berechnen
p_xgb = xgb_model.predict_proba(X_tab_test)
p_lstm = lstm_model.predict(X_seq_test, verbose=0)
p_resnet = resnet_model.predict(X_spec_test, verbose=0)

# Ausgewogenes Soft-Voting (Da das ResNet nun 83% liefert[cite: 15], gleichwertige Gewichtung)
p_ensemble = (0.33 * p_xgb) + (0.33 * p_lstm) + (0.34 * p_resnet)
y_pred = p_ensemble.argmax(axis=1)

# Auswertung
accuracy = np.mean(y_pred == y_test) * 100
report = classification_report(y_test, y_pred)

print(f"FINAL DEEP FINE-TUNED ENSEMBLE ACCURACY: {accuracy:.2f}%")

# 4. Dokumentation für die Präsentation abspeichern
with open(log_filename, "w") as f:
    f.write(f"--- ENSEMBLE EXPERIMENT MIT TIEF-FINE-GESUNTEM RESNET ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN:\n")
    f.write("1. Verwendung des deep fine-getunten ResNet50 (conv4 + conv5 entfroren, Lernrate 5e-6).\n")
    f.write("2. ResNet-Einzelleistung von anfangs 35%[cite: 8] über 80%[cite: 13] auf 83% gesteigert[cite: 15].\n")
    f.write("3. Angepasstes Soft-Voting-Gewicht (XGB: 0.33, LSTM: 0.33, ResNet: 0.34).\n\n")
    f.write(f"ERZIELTE ENSEMBLE ACCURACY: {accuracy:.2f}%\n\n")
    f.write(report)

# Konfusionsmatrix plotten
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Deep Fine-Tuned Ensemble Confusion Matrix ({accuracy:.2f}%)")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.savefig(matrix_filename)
plt.close()

print(f"Erfolgreich dokumentiert in: {log_filename}")