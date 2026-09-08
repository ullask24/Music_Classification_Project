import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
import torch
import torchaudio
from sklearn.metrics import accuracy_score, classification_report

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_DIR = "data/gtzan"
TARGET_SR = 16000
DURATION = 30
FIXED_SPEC_WIDTH = 128

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def extract_tabular_features(y, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

    temp_row = [
        np.mean(chroma), np.mean(rmse), np.mean(spec_cent),
        np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
    ]
    for m in mfcc:
        temp_row.append(np.mean(m))
    for sc in spec_contrast:
        temp_row.append(np.mean(sc))
    for tn in tonnetz:
        temp_row.append(np.mean(tn))
    return np.array(temp_row).reshape(1, -1)

def get_individual_probabilities(xgb_model, bilstm_model, resnet_model, tabular_features, wav2vec_seq, spec_numpy, device):
    """
    Holt und normalisiert die Einzelwahrscheinlichkeiten aller drei Ensemble-Modelle.
    """
    xgb_probs = xgb_model.predict_proba(tabular_features)
    
    bilstm_model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor(wav2vec_seq, dtype=torch.float32).to(device)
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        bilstm_out = bilstm_model(seq_tensor)
        bilstm_probs = softmax(bilstm_out.cpu().numpy())

    # ResNet ist ein TensorFlow/Keras-Modell und erwartet NumPy Arrays im Format (Batch, H, W, C)
    resnet_out = resnet_model.predict(spec_numpy, verbose=0)
    resnet_probs = softmax(resnet_out)

    return xgb_probs, bilstm_probs, resnet_probs

def evaluate_gtzan():
    print("Lade V3-Modelle und Wav2Vec2 vorab...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")
    scaler = joblib.load("models/scaler.joblib")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    w2v_model = bundle.get_model().to(device)
    w2v_model.eval()

    y_true = []
    y_pred = []

    if not os.path.exists(DATA_DIR):
        print(f"Fehler: Verzeichnis {DATA_DIR} nicht gefunden.")
        return

    print(f"Starte Evaluation über GTZAN-Dateien in {DATA_DIR}...")
    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            continue
        
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        for file in files[:3]:
            file_path = os.path.join(genre_dir, file)
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, duration=DURATION)
            if len(y) < TARGET_SR * DURATION:
                y = np.pad(y, (0, TARGET_SR * DURATION - len(y)))
            else:
                y = y[:TARGET_SR * DURATION]
            
            # ResNet Spektrogramm mit exakter Min-Max Normalisierung (Keras Format: Batch, H, W, C)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)
            temp_spec = np.expand_dims(temp_spec, axis=-1)
            X_spec = np.repeat(temp_spec, 3, axis=-1)
            X_spec_np = np.expand_dims(X_spec, axis=0) # Shape: (1, 128, 128, 3)
            
            # Tabellarische Features
            X_tab_raw = extract_tabular_features(y, sr)
            X_tab_scaled = scaler.transform(X_tab_raw)
            
            # Wav2Vec2 Sequence für BiLSTM
            waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = w2v_model.extract_features(waveform)
                latent_seq = out[0] if isinstance(out, tuple) else out
                rep = latent_seq[-1].squeeze(0).cpu().numpy()
                step = max(1, rep.shape[0] // 100)
                temp_seq = rep[::step][:100]
                if temp_seq.shape[0] < 100:
                    temp_seq = np.pad(temp_seq, ((0, 100 - temp_seq.shape[0]), (0, 0)))
            X_seq = np.expand_dims(temp_seq, axis=0)
            
            # Einzelwahrscheinlichkeiten abrufen
            p_xgb, p_lstm, p_resnet = get_individual_probabilities(
                xgb_model, lstm_model, resnet_model, 
                X_tab_scaled, X_seq, X_spec_np, device
            )
            
            # Gewichtetes Soft-Voting (Ensemble V3)
            p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
            pred_idx = p_ensemble.argmax()
            
            y_true.append(genre_idx)
            y_pred.append(pred_idx)
            print(f"[{genre}] {file} -> XGB: {GENRES[p_xgb.argmax()]} | LSTM: {GENRES[p_lstm.argmax()]} | ResNet: {GENRES[p_resnet.argmax()]} -> Ensemble: {GENRES[pred_idx]}")

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        print(f"\nGesamt-Accuracy auf Test-Stichprobe: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=GENRES, zero_division=0))

if __name__ == "__main__":
    evaluate_gtzan()