import os
import sys
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
import torch
import torchaudio

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
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

def run_live_inference(file_path):
    print(f"Lade Audiodatei: {file_path}")
    y_full, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
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
    
    chunk_samples = TARGET_SR * DURATION
    probabilities = []
    
    for i in range(0, max(1, len(y_full)), chunk_samples):
        y_chunk = y_full[i:i + chunk_samples]
        if len(y_chunk) < TARGET_SR * 3:
            break
            
        if len(y_chunk) < chunk_samples:
            y_chunk = np.pad(y_chunk, (0, chunk_samples - len(y_chunk)))
        else:
            y_chunk = y_chunk[:chunk_samples]
            
        print(f"Analysiere Segment ab {i/TARGET_SR:.1f}s...")
        
        # ResNet Spektrogramm mit exakter Min-Max Normalisierung[cite: 1]
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=TARGET_SR, n_mels=128, n_fft=1024, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)
        temp_spec = np.expand_dims(temp_spec, axis=-1)
        X_spec = np.repeat(temp_spec, 3, axis=-1)
        X_spec = np.expand_dims(X_spec, axis=0)
        
        # Tabellarische Features
        X_tab_raw = extract_tabular_features(y_chunk, TARGET_SR)
        X_tab_scaled = scaler.transform(X_tab_raw)
        
        # Wav2Vec2 Sequence für BiLSTM[cite: 1]
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
        
        p_xgb = xgb_model.predict_proba(X_tab_scaled)
        p_lstm = softmax(lstm_model.predict(X_seq, verbose=0))
        p_resnet = softmax(resnet_model.predict(X_spec, verbose=0))
        
        p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
        probabilities.append(p_ensemble[0])
        
    if len(probabilities) == 0:
        print("Fehler: Audiodatei ist zu kurz.")
        return
        
    mean_probabilities = np.mean(probabilities, axis=0)
    pred_idx = mean_probabilities.argmax()
    confidence = mean_probabilities[pred_idx] * 100
    
    print("\n" + "="*40)
    print(" ERGEBNIS DER LIVE-DEMO (LANGER TRACK)")
    print("="*40)
    print(f"Datei: {os.path.basename(file_path)}")
    print(f"Erkanntes Genre: {GENRES[pred_idx].upper()}")
    print(f"Konfidenz (Ensemble V3): {confidence:.2f}%")
    print("\nWahrscheinlichkeiten aller Genres:")
    for idx, prob in enumerate(mean_probabilities):
        print(f"  - {GENRES[idx].capitalize()}: {prob*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python3 scripts/live_demo.py <pfad_zu_audio.wav>")
    else:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            run_live_inference(audio_path)
        else:
            print(f"Fehler: Datei nicht gefunden unter {audio_path}")