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

def run_live_inference(file_path):
    print(f"Lade Audiodatei: {file_path}")
    y_full, sr = librosa.load(file_path, sr=22050)
    
    print("Lade V3-Modelle und Wav2Vec2 vorab...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")
    scaler = joblib.load("models/scaler.joblib")
    
    # Wav2Vec2 einmalig laden
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    chunk_duration = 30.0
    chunk_samples = int(chunk_duration * sr)
    probabilities = []
    
    for i in range(0, len(y_full), chunk_samples):
        y_chunk = y_full[i:i + chunk_samples]
        if len(y_chunk) < sr * 5:
            break
            
        print(f"Analysiere Segment ab {i/sr:.1f}s...")
        
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
        else:
            mel_spec_db = mel_spec_db[:, :128]
        X_spec = np.expand_dims(mel_spec_db, axis=-1)
        X_spec = np.repeat(X_spec, 3, axis=-1)
        X_spec = np.expand_dims(X_spec, axis=0)
        
        X_tab_raw = extract_tabular_features(y_chunk, sr)
        X_tab_scaled = scaler.transform(X_tab_raw)
        X_seq = extract_wav2vec_sequence(y_chunk, sr, processor, w2v_model)
        
        p_xgb = xgb_model.predict_proba(X_tab_scaled)
        p_lstm = lstm_model.predict(X_seq, verbose=0)
        p_resnet = resnet_model.predict(X_spec, verbose=0)
        
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
        print("Verwendung: python3 scripts/live_demo_long.py <pfad_zu_audio.wav>")
    else:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            run_live_inference(audio_path)
        else:
            print(f"Fehler: Datei nicht gefunden unter {audio_path}")