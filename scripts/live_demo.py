import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
import torch
import torchaudio

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
LIVE_DIR = "../live"
TARGET_SR = 16000
DURATION = 30
FIXED_SPEC_WIDTH = 128

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def extract_tabular_features(y, sr):
    y_harmonic, _ = librosa.effects.hpss(y)
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

def evaluate_live_folders():
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

    target_folders = ['country', 'metal', 'jazz']

    if not os.path.exists(LIVE_DIR):
        print(f"Fehler: Live-Verzeichnis '{LIVE_DIR}' nicht gefunden.")
        return

    for folder_name in target_folders:
        folder_path = os.path.join(LIVE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Überspringe: Ordner '{folder_name}' existiert nicht unter {LIVE_DIR}.")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            print(f"Keine .wav Dateien im Ordner '{folder_name}' gefunden.")
            continue

        print(f"\n" + "="*60)
        print(f" ANALYSE DES ORDNERS: {folder_name.upper()} (Live-Daten Domain Shift)")
        print("="*60)

        for file in files:
            file_path = os.path.join(folder_path, file)
            y_full, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            
            chunk_samples = TARGET_SR * DURATION
            chunk_probs_ensemble = []
            
            # Verarbeite den ersten validen Ausschnitt (oder gemittelt über Segmente)
            for i in range(0, max(1, len(y_full)), chunk_samples):
                y_chunk = y_full[i:i + chunk_samples]
                if len(y_chunk) < TARGET_SR * 3:
                    break
                if len(y_chunk) < chunk_samples:
                    y_chunk = np.pad(y_chunk, (0, chunk_samples - len(y_chunk)))
                else:
                    y_chunk = y_chunk[:chunk_samples]
                
                # Features extrahieren
                mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=TARGET_SR, n_mels=128, n_fft=1024, hop_length=512)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)
                temp_spec = np.expand_dims(temp_spec, axis=-1)
                X_spec = np.repeat(temp_spec, 3, axis=-1)
                X_spec_np = np.expand_dims(X_spec, axis=0)
                
                X_tab_raw = extract_tabular_features(y_chunk, TARGET_SR)
                X_tab_scaled = scaler.transform(X_tab_raw)
                
                waveform = torch.tensor(y_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = w2v_model.extract_features(waveform)
                    latent_seq = out[0] if isinstance(out, tuple) else out
                    rep = latent_seq[-1].squeeze(0).cpu().numpy()
                    step = max(1, rep.shape[0] // 100)
                    temp_seq = rep[::step][:100]
                    if temp_seq.shape[0] < 100:
                        temp_seq = np.pad(temp_seq, ((0, 100 - temp_seq.shape[0]), (0, 0)))
                X_seq = np.expand_dims(temp_seq, axis=0)
                
                # Einzelwahrscheinlichkeiten holen
                p_xgb = xgb_model.predict_proba(X_tab_scaled)
                p_lstm = softmax(lstm_model.predict(X_seq, verbose=0))
                p_resnet = softmax(resnet_model.predict(X_spec_np, verbose=0))
                
                p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
                chunk_probs_ensemble.append(p_ensemble[0])
                
                # Zeige exemplarisch das erste Segment zum Vergleich des Domain Shifts
                print(f"Datei: {file} [Segment {i/TARGET_SR:.1f}s]")
                print(f"  -> XGBoost:  {GENRES[p_xgb.argmax()]:<10} ({p_xgb[0, p_xgb.argmax()]*100:5.2f}% Konfidenz) [Anfällig für Drift]")
                print(f"  -> BiLSTM:   {GENRES[p_lstm.argmax()]:<10} ({p_lstm[0, p_lstm.argmax()]*100:5.2f}% Konfidenz)")
                print(f"  -> ResNet:   {GENRES[p_resnet.argmax()]:<10} ({p_resnet[0, p_resnet.argmax()]*100:5.2f}% Konfidenz)")
                print(f"  -> ENSEMBLE: {GENRES[p_ensemble.argmax()]:<10} ({p_ensemble[0, p_ensemble.argmax()]*100:5.2f}% Konfidenz)\n")
                break # Analysiert das erste prägende Segment pro Datei

if __name__ == "__main__":
    evaluate_live_folders()