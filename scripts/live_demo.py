import sys
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa

# GTZAN Genre-Mapping
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

if len(sys.argv) < 2:
    print("Verwendung: python live_demo.py <pfad_zu_audio.wav>")
    sys.exit(1)

audio_path = sys.argv[1]
print(analysiere Datei: {audio_path}...)

# 1. Audio laden und Features für die drei Spezialisten extrahieren
y_audio, sr = librosa.load(audio_path, duration=30.0, sr=22050)

# A) Tabellarische / Statistische Features (für XGBoost)
# Hier müssen exakt dieselben Features extrahiert werden wie im Training (z.B. Spectral Centroid, RMS, MFCC-Mittelwerte etc.)
# Als vereinfachtes Beispiel laden wir hier Dummy- oder Live-Extraktion:
mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
stat_features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1)

scaler = joblib.load("models/scaler.joblib")
stat_features_scaled = scaler.transform(stat_features)

# B) Mel-Spektrogramm (für ResNet - 128x128x3)
mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# Auf feste Größe 128x128 bringen (z.B. via Resize oder Padding)
if mel_spec_db.shape[1] < 128:
    mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
else:
    mel_spec_db = mel_spec_db[:, :128]
X_spec = np.expand_dims(mel_spec_db, axis=-1)
X_spec = np.repeat(X_spec, 3, axis=-1) # Auf 3 Kanäle für ResNet50
X_spec = np.expand_dims(X_spec, axis=0)

# C) Sequenz-Features (für LSTM - z.B. Wav2Vec oder MFCC-Sequenzen)
# Entspricht der im Training verwendeten Sequenz-Form (Batch, Time, Features)
X_seq = mfcap = mfcc.T # Vereinfacht als Sequenz
if X_seq.shape[0] < 128:
    X_seq = np.pad(X_seq, ((0, 128 - X_seq.shape[0]), (0, 0)))
else:
    X_seq = X_seq[:128, :]
X_seq = np.expand_dims(X_seq, axis=0)

# 2. Modelle laden
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_specialist.json")
lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras")

# 3. Vorhersagen berechnen
p_xgb = xgb_model.predict_proba(stat_features_scaled)
p_lstm = lstm_model.predict(X_seq)
p_resnet = resnet_model.predict(X_spec)

# 4. Gewichtiges Soft-Voting
p_ensemble = (0.45 * p_xgb) + (0.45 * p_lstm) + (0.10 * p_resnet)
predicted_idx = p_ensemble.argmax(axis=1)[0]
confidence = p_ensemble[0][predicted_idx] * 100

print(f"\nERGEBNIS DER LIVE-KLASSIFIKATION:")
print(f"Erkanntes Genre: {GENRES[predicted_idx].upper()} (Sicherheit: {confidence:.2f}%)")