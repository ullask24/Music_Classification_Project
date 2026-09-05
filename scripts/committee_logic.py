import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten laden
df_tab = pd.read_csv("features/statistical_features.csv")
X_tab = df_tab.drop("label", axis=1).values
y = np.load("features/labels.npy")

X_spec = np.expand_dims(np.load("features/mel_spectrograms.npy"), axis=-1)
X_seq = np.load("features/wav2vec_sequences.npy")

genre_names = sorted(df_tab["label"].unique())

# Einheitlicher Test-Split über identischen Seed
indices = np.arange(len(y))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_tab_test = X_tab[test_idx]
X_spec_test = X_spec[test_idx]
X_seq_test = X_seq[test_idx]
y_test = y[test_idx]

# 2. Modelle laden
scaler = joblib.load("models/scaler.joblib")
X_tab_test_scaled = scaler.transform(X_tab_test)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")

lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras")

# 3. Soft-Voting-Wahrscheinlichkeiten berechnen
p_xgb = xgb_model.predict_proba(X_tab_test_scaled)
p_lstm = lstm_model.predict(X_seq_test)
p_resnet = resnet_model.predict(X_spec_test)

# Arithmetischer Mittelwert über alle 3 Wahrscheinlichkeitsverteilungen
ensemble_probs = (p_xgb + p_lstm + p_resnet) / 3.0
final_preds = np.argmax(ensemble_probs, axis=1)

# 4. Metriken und Visualisierung
acc = accuracy_score(y_test, final_preds)
print(f"\n==========================================")
print(f" FINAL ENSEMBLE ACCURACY: {acc * 100:.2f}%")
print(f"==========================================\n")

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=genre_names, yticklabels=genre_names)
plt.title("Ensemble Confusion Matrix (XGBoost + LSTM + ResNet)")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix.png")

with open("final_report.txt", "w") as f:
    f.write(f"FINAL ENSEMBLE ACCURACY: {acc * 100:.2f}%\n\n")
    f.write(classification_report(y_test, final_preds, target_names=genre_names))

print("Evaluation abgeschlossen. Bericht und Konfusionsmatrix gespeichert.")