import os
import datetime
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Testdaten laden
print("Lade Testdaten für das Ensemble...")
X_tab = pd.read_csv("features/statistical_features.csv") if 'pd' in globals() else None
# Falls pandas nicht global geladen ist, alternativ:
import pandas as pd
X_tab = pd.read_csv("features/statistical_features.csv")
y = np.load("features/labels.npy")

X_spec = np.load("features/mel_spectrograms.npy")
X_seq = np.load("features/wav2vec_sequences.npy")

# Entsprechenden Test-Split wie im Training herstellen (z.B. letztes Split-Verhältnis)
from sklearn.model_selection import train_test_split

_, X_tab_test, _, X_spec_test, _, X_seq_test, _, y_test = train_test_split(
    X_tab.drop(columns=["label"]), X_spec, X_seq, y, test_size=0.2, random_state=42
)

# Tabellarische Features skalieren (falls Scaler vorhanden ist)
scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
if scaler:
    X_tab_test_scaled = scaler.transform(X_tab_test)
else:
    X_tab_test_scaled = X_tab_test

# WICHTIG: ResNet erwartet 3 Kanäle (RGB-Duplikation der Spektrogramme)
if X_spec_test.shape[-1] != 3:
    X_spec_test = np.repeat(X_spec_test, 3, axis=-1)

# 2. Modelle laden
print("Lade trainierte Einzelspezialisten...")
xgb_model = joblib.load("models/xgb_specialist.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras")

# 3. Vorhersagen (Soft-Voting Wahrscheinlichkeiten) berechnen
print("Berechne Vorhersagen der Spezialisten...")
p_xgb = xgb_model.predict_proba(X_tab_test_scaled)
p_lstm = lstm_model.predict(X_seq_test)
p_resnet = resnet_model.predict(X_spec_test)

# 4. Gewichtiges Soft-Voting anwenden (XGBoost: 0.45, LSTM: 0.45, ResNet: 0.10)
w_xgb, w_lstm, w_resnet = 0.45, 0.45, 0.10
p_ensemble = (w_xgb * p_xgb) + (w_lstm * p_lstm) + (w_resnet * p_resnet)
y_pred = p_ensemble.argmax(axis=1)

# 5. Metriken und Berichte erstellen
acc = np.mean(y_pred == y_test)
report = classification_report(y_test, y_pred)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"final_report_extended_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- ENSEMBLE EXPERIMENT (GEWICHTETES SOFT-VOTING) ---\n")
    f.write(f"Zeitpunkt: {timestamp}\n")
    f.write(f"VERÄNDERUNGEN: Gewichtung angepasst (XGBoost: {w_xgb}, LSTM: {w_lstm}, ResNet: {w_resnet}), Pretrained ResNet50 integriert.\n")
    f.write(f"ZIELE: Kompensation schwächerer Modelle und Maximierung der Ensemble-Accuracy.\n\n")
    f.write(f"FINAL ENSEMBLE ACCURACY: {acc * 100:.2f}%\n\n")
    f.write(report)

print(f"Ensemble-Auswertung beendet! Accuracy: {acc * 100:.2f}%. Bericht: {report_filename}")

# 6. Konfusionsmatrix visualisieren und speichern
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title(f"Ensemble Confusion Matrix - Gewichtet ({timestamp})")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.savefig(f"ensemble_confusion_matrix_extended_{timestamp}.png")
plt.close()