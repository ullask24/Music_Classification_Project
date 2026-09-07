import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from sklearn.metrics import accuracy_score, classification_report

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_DIR = "data/gtzan"

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def extract_tabular_features(y_audio, sr):
    features = []
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfccs, axis=1))
    contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    hpss = librosa.effects.hpss(y_audio)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y_audio), sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y_audio)))
    features.append(np.mean(librosa.feature.rms(y=y_audio)))
    while len(features) < 39:
        features.append(0.0)
    return np.array(features[:39]).reshape(1, -1)

def extract_wav2vec_sequence(y_audio, orig_sr, processor, model):
    if orig_sr != 16000:
        y_audio = librosa.resample(y_audio, orig_sr=orig_sr, target_sr=16000)
    inputs = processor(y_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.numpy()
    target_len = 1000
    if embeddings.shape[1] < target_len:
        embeddings = np.pad(embeddings, ((0,0), (0, target_len - embeddings.shape[1]), (0,0)))
    else:
        embeddings = embeddings[:, :target_len, :]
    return embeddings

def evaluate_gtzan():
    print("Lade V3-Modelle und Wav2Vec2 vorab...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")
    scaler = joblib.load("models/scaler.joblib")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    y_true = []
    y_pred = []

    if not os.path.exists(DATA_DIR):
        print(f"Fehler: Verzeichnis {DATA_DIR} nicht gefunden.")
        return

    print(Starte Evaluation über GTZAN-Dateien in {DATA_DIR}...")
    for genre in GENRES:
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            continue
        
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        # Optional: Nur eine Stichprobe pro Genre testen (z.B. die ersten 5 Dateien), um Zeit zu sparen
        for file in files[:5]:
            file_path = os.path.join(genre_dir, file)
            y_audio, sr = librosa.load(file_path, sr=22050)
            
            # Features extrahieren
            mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            if mel_spec_db.shape[1] < 128:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
            else:
                mel_spec_db = mel_spec_db[:, :128]
            X_spec = np.expand_dims(mel_spec_db, axis=-1)
            X_spec = np.repeat(X_spec, 3, axis=-1)
            X_spec = np.expand_dims(X_spec, axis=0)
            
            X_tab_raw = extract_tabular_features(y_audio, sr)
            X_tab_scaled = scaler.transform(X_tab_raw)
            X_seq = extract_wav2vec_sequence(y_audio, sr, processor, w2v_model)
            
            p_xgb = xgb_model.predict_proba(X_tab_scaled)
            p_lstm = softmax(lstm_model.predict(X_seq, verbose=0))
            p_resnet = softmax(resnet_model.predict(X_spec, verbose=0))
            
            p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
            pred_idx = p_ensemble.argmax()
            
            y_true.append(GENRES.index(genre))
            y_pred.append(pred_idx)
            print(f"[{genre}] {file} -> Vorhersage: {GENRES[pred_idx]} (Echt: {genre})")

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        print(f"\nGesamt-Accuracy auf Test-Stichprobe: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=GENRES, zero_division=0))

if __name__ == "__main__":
    evaluate_gtzan()