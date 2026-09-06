import os
import datetime
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df_tab = pd.read_csv("features/statistical_features.csv")
X_tab = df_tab.drop("label", axis=1).values
y = np.load("features/labels.npy")

X_spec = np.expand_dims(np.load("features/mel_spectrograms.npy"), axis=-1)
X_seq = np.load("features/wav2vec_sequences.npy")

genre_names = sorted(df_tab["label"].unique())

indices = np.arange(len(y))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_tab_test = X_tab[test_idx]
X_spec_test = X_spec[test_idx]
X_seq_test = X_seq[test_idx]
y_test = y[test_idx]

scaler = joblib.load("models/scaler.joblib")
X_tab_test_scaled = scaler.transform(X_tab_test)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")

lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras")

p_xgb = xgb_model.predict_proba(X_tab_test_scaled)
p_lstm = lstm_model.predict(X_seq_test)
p_resnet = resnet_model.predict(X_spec_test)

# VERÄNDERUNG: Gewichtetes Soft-Voting statt starrem Mittelwert (1/3)
# ZIELE: Starke Modelle (XGBoost/LSTM) stärken und den ResNet-Einbruch kompensieren
w_xgb, w_lstm, w_resnet = 0.45, 0.45, 0.10
ensemble_probs = (w_xgb * p_xgb) + (w_lstm * p_lstm) + (w_resnet * p_resnet)
final_preds = np.argmax(ensemble_probs, axis=1)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"final_report_extended_{timestamp}.txt"
matrix_filename = f"ensemble_confusion_matrix_extended_{timestamp}.png"

acc = accuracy_score(y_test, final_preds)
print(f"\n==========================================")
print(f" FINAL ENSEMBLE ACCURACY (Gewichtet): {acc * 100:.2f}%")
print(f"==========================================\n")

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=genre_names, yticklabels=genre_names)
plt.title(f"Ensemble Confusion Matrix - Gewichtet ({timestamp})")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.tight_layout()
plt.savefig(matrix_filename)
plt.close()

with open(report_filename, "w") as f:
    f.write(f"--- ENSEMBLE EXPERIMENT (GEWICHTETES SOFT-VOTING) ---\n")
    f.write(f"Zeitpunkt: {timestamp}\n")
    f.write(f"VERÄNDERUNGEN: Gewichtung angepasst (XGBoost: {w_xgb}, LSTM: {w_lstm}, ResNet: {w_resnet})\n")
    f.write(f"ZIELE: Bessere Kompensation schwächerer Modelle und Maximierung der Ensemble-Accuracy.\n\n")
    f.write(f"FINAL ENSEMBLE ACCURACY: {acc * 100:.2f}%\n\n")
    f.write(classification_report(y_test, final_preds, target_names=genre_names))

print(f"Evaluierung abgeschlossen. Ergebnisse gesichert unter:\n- Bericht: {report_filename}\n- Grafik: {matrix_filename}")