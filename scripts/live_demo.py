import sys
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def process_live_audio(audio_path):
    print(f"Analysiere Datei: {audio_path}...")
    y_audio, sr = librosa.load(audio_path, sr=22050)
    
    # Aufteilung in 30-Sekunden-Fenster (GTZAN-Standard)[cite: 1]
    chunk_duration = 30.0  
    chunk_samples = int(chunk_duration * sr)
    
    # Modelle und Scaler laden
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist.keras")
    scaler = joblib.load("models/scaler.joblib")
    
    all_probabilities = []
    
    # Schleife über alle 30-Sekunden-Blöcke des Songs
    for i in range(0, max(1, len(y_audio)), chunk_samples):
        chunk = y_audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        # 1. XGBoost Features (Statistik)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        stat_features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1)
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
        print("Verwendung: python live_demo.py <pfad_zu_audio.wav>")
        sys.exit(1)
    process_live_audio(sys.argv[1])