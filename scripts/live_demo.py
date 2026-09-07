import os
import sys
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_tabular_features(y_audio, sr):
    # Extrahiert die 39 statistischen Features analog zum Training
    features = []
    
    # MFCCs (20 Mittelwerte)
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfccs, axis=1))
    
    # Spectral Contrast (7 Bands)
    contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    
    # Tonnetz (6 Dimensionen)
    hpss = librosa.effects.hpss(y_audio)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y_audio), sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    
    # Weitere Spektral- und Rhythmus-Features zum Auffüllen auf 39 Features
    features.append(np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y_audio, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y_audio)))
    features.append(np.mean(librosa.feature.rms(y=y_audio)))
    
    # Rest mit Nullen auffüllen falls Dimension abweicht
    while len(features) < 39:
        features.append(0.0)
        
    return np.array(features[:39]).reshape(1, -1)

def extract_wav2vec_sequence(y_audio, sr):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    inputs = processor(y_audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.numpy() # Shape: (1, Frames, 768)
        
    # Auf feste Sequenzlänge bringen (padding/truncating je nach Modell-Erwartung)
    target_len = 1000 # Entspricht typischer GTZAN Sequenzlänge
    if embeddings.shape[1] < target_len:
        embeddings = np.pad(embeddings, ((0,0), (0, target_len - embeddings.shape[1]), (0,0)))
    else:
        embeddings = embeddings[:, :target_len, :]
    return embeddings

def run_live_inference(file_path):
    print(1)
    print(f"Lade Audiodatei: {file_path}")
    y_full, sr = librosa.load(file_path, sr=22050)
    
    # Spezialisten und Scaler laden
    print("Lade V3-Modelle...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")
    scaler = joblib.load("models/scaler.joblib")
    
    # Audio in 30-Sekunden-Fenster unterteilen (GTZAN Standard)
    chunk_duration = 30.0
    chunk_samples = int(chunk_duration * sr)
    
    probabilities = []
    
    for i in range(0, len(y_full), chunk_samples):
        y_chunk = y_full[i:i + chunk_samples]
        if len(y_chunk) < sr * 5: # Ignoriere zu kurze Reststücke (< 5 Sek)
            break
            
        print(f"Analysiere Segment ab {i/sr:.1f}s...")
        
        # 1. ResNet Feature (Mel-Spektrogramm)
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
        else:
            mel_spec_db = mel_spec_db[:, :128]
        X_spec = np.expand_dims(mel_spec_db, axis=-1)
        X_spec = np.repeat(X_spec, 3, axis=-1)
        X_spec = np.expand_dims(X_spec, axis=0)
        
        # 2. Tabellarische Features für XGBoost
        X_tab_raw = extract_tabular_features(y_chunk, sr)
        X_tab_scaled = scaler.transform(X_tab_raw)
        
        # 3. Wav2Vec Sequence für BiLSTM
        X_seq = extract_wav2vec_sequence(y_chunk, sr)
        
        # Vorhersagen der Spezialisten für dieses Segment
        p_xgb = xgb_model.predict_proba(X_tab_scaled)
        p_lstm = lstm_model.predict(X_seq, verbose=0)
        p_resnet = resnet_model.predict(X_spec, verbose=0)
        
        # V3 Ensemble Soft-Voting (XGB: 0.25, BiLSTM: 0.45, ResNet: 0.30)
        p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
        probabilities.append(p_ensemble[0])
        
    # Mittelwert über alle Segmente bilden
    if len(probabilities) == 0:
        print("Fehler: Audiodatei ist zu kurz.")
        return
        
    mean_probabilities = np.mean(probabilities, axis=0)
    pred_idx = mean_probabilities.argmax()
    confidence = mean_probabilities[pred_idx] * 100
    
    print("\n" + "="*40)
    print(f" ERGEBNIS DER LIVE-DEMO (LANGER TRACK)")
    print("="*40)
    print(f"Datei: {os.path.basename(file_path)}")
    print(f"Erkanntes Genre: {GENRES[pred_idx].upper()}")
    print(f"Konfidenz (Ensemble V3): {confidence:.2f}%")
    print("\nWahrscheinlichkeiten aller Genres:")
    for idx, prob in enumerate(mean_probabilities):
        print(f"  - {GENRES[idx].capitalize()}: {prob*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    audio_path = "/home/ulas/Uni/Neuroinformatik/Music_Classification_Project/live/audiodollar-epic-metal-epic-453481.wav"
    if os.path.exists(audio_path):
        run_live_inference(audio_path)
    else:
        print(f"Fehler: Datei nicht gefunden unter {audio_path}")