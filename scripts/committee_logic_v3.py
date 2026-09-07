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
log_filename = f"logs/final_report_v3_ensemble_{timestamp}.txt"
matrix_filename = f"logs/ensemble_confusion_matrix_v3_{timestamp}.png"

print("=== STARTE FINALES ENSEMBLE V3 (BILSTM + DEEP RESNET + XGB) ===")

# 1. Daten laden
X_spec = np.load("features/mel_spectrograms.npy")
X_seq = np.load("features/wav2vec_sequences.npy")
y = np.load("features/labels.npy")

if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

df_tab = pd.read_csv("features/statistical_features.csv")
scaler = joblib.load("models/scaler.joblib")
X_tab_scaled = scaler.transform(df_tab.drop(columns=["label"]))

_, X_spec_test, _, X_seq_test, _, X_tab_test, _, y_test = train_test_split(
    X_spec, X_seq, X_tab_scaled, y, test_size=0.2, random_state=42
)

# 2. Laden der verbesserten Spezialisten
print("Lade optimierte Spezialisten...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")
lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras") # BiLSTM (91%)
resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras") # Deep ResNet (83%)

# 3. Vorhersagen berechnen
p_xgb = xgb_model.predict_proba(X_tab_test)
p_lstm = lstm_model.predict(X_seq_test, verbose=0)
p_resnet = resnet_model.predict(X_spec_test, verbose=0)

# Angepasste Gewichtung basierend auf der neuen Leistungsverteilung:
# BiLSTM hat nun mit 91% das stärkste Gewicht, gefolgt von ResNet und XGBoost
p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
y_pred = p_ensemble.argmax(axis=1)

# Auswertung
accuracy = np.mean(y_pred == y_test) * 100
report = classification_report(y_test, y_pred)

print(f"FINAL V3 ENSEMBLE ACCURACY: {accuracy:.2f}%")

# 4. Dokumentation abspeichern
with open(log_filename, "w") as f:
    f.write(f"--- FINALES ENSEMBLE EXPERIMENT V3 ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN & EVIDENZ:\n")
    f.write("1. Integration des BiLSTM-Spezialisten (Accuracy von 71% auf 91% gesteigert)[cite: 4].\n")
    f.write("2. Integration des Deep Fine-Tuned ResNet50 (Accuracy von 35% auf 83% gesteigert)[cite: 8].\n")
    f.write("3. Bewährter XGBoost-Spezialist als stabiles Fundament (82% Accuracy).\n")
    f.write("4. Angepasste Soft-Voting-Gewichte (XGB: 0.25, BiLSTM: 0.45, ResNet: 0.30).\n\n")
    f.write(f"ERZIELTE GESAMT-ACCURACY: {accuracy:.2f}%\n\n")
    f.write(report)

# Konfusionsmatrix plotten
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Final V3 Ensemble Confusion Matrix ({accuracy:.2f}%)")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.savefig(matrix_filename)
plt.close()

print(f"Erfolgreich dokumentiert in: {log_filename}")