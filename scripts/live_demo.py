import sys
import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa

# Absoluten Basis-Projektpfad ermitteln
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def process_live_audio(audio_path):
    print(f"Analysiere Datei: {audio_path}...")
    y_audio, sr = librosa.load(audio_path, sr=22050)
    
    # Aufteilung in 30-Sekunden-Fenster (GTZAN-Standard)
    chunk_duration = 30.0  
    chunk_samples = int(chunk_duration * sr)
    
    # Modelle und Scaler mit absoluten Pfaden laden
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(BASE_DIR, "models", "xgb_specialist.json"))
    lstm_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "lstm_specialist.keras"))
    resnet_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "resnet_specialist.keras"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    
    all_probabilities = []
    
    # Schleife über alle 30-Sekunden-Blöcke des Songs
    for i in range(0, max(1, len(y_audio)), chunk_samples):
        chunk = y_audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        # 1. XGBoost Features (Exakt 39 Features analog zu prepare_dataset.py)[cite: 6]
        y_harmonic, y_percussive = librosa.effects.hpss(chunk)
        
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        rmse = librosa.feature.rms(y=chunk)
        spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(chunk)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        spec_contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

        stat_row = [
            np.mean(chroma), np.mean(rmse), np.mean(spec_cent),
            np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
        ]
        for m in mfcc:
            stat_row.append(np.mean(m))
        for sc in spec_contrast:
            stat_row.append(np.mean(sc))
        for tn in tonnetz:
            stat_row.append(np.mean(tn))
            
        stat_features = np.array(stat_row).reshape(1, -1)
        stat_scaled = scaler.transform(stat_features)
        
        # 2. ResNet Spektrogramm (128x128x3)
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :128]
        X_spec = np.expand_dims(mel_db, axis=-1)
        X_spec = np.repeat(X_spec, 3, axis=-1)
        X_spec = np.expand_dims(X_spec, axis=0)
        
        # 3. LSTM Sequenz Features
        X_seq = mfcc.T
        if X_seq.shape[0] < 128:
            X_seq = np.pad(X_seq, ((0, 128 - X_seq.shape[0]), (0, 0)))
        else:
            X_seq = X_seq[:128, :]
        X_seq = np.expand_dims(X_seq, axis=0)
        
        # Vorhersagen der Einzelspezialisten berechnen
        p_xgb = xgb_model.predict_proba(stat_scaled)
        p_lstm = lstm_model.predict(X_seq, verbose=0)
        p_resnet = resnet_model.predict(X_spec, verbose=0)
        
        # Gewichtiges Soft-Voting für diesen Chunk
        p_chunk = (0.45 * p_xgb) + (0.45 * p_lstm) + (0.10 * p_resnet)
        all_probabilities.append(p_chunk)
        
    # Aggregation über alle Fenster (Mittelwertbildung)
    final_p = np.mean(all_probabilities, axis=0)
    pred_idx = final_p.argmax(axis=1)[0]
    confidence = final_p[0][pred_idx] * 100
    
    print(f"\nERGEBNIS DER LIVE-KLASSIFIKATION:")
    print(f"Erkanntes Genre: {GENRES[pred_idx].upper()} (Sicherheit: {confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python scripts/live_demo.py <pfad_zu_audio.wav>")
        sys.exit(1)
    process_live_audio(sys.argv[1])