import os
import sys
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import librosa
from sklearn.model_selection import train_test_split

# Genre-Mapping der GTZAN-Datenbasis
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_ensemble_models():
    print("Lade optimierte V3-Spezialisten...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_specialist.json")
    lstm_model = tf.keras.models.load_model("models/lstm_specialist_bilstm.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_specialist_deep.keras")
    scaler = joblib.load("models/scaler.joblib")
    return xgb_model, lstm_model, resnet_model, scaler

def predict_audio(file_path):
    xgb_model, lstm_model, resnet_model, scaler = load_ensemble_models()
    
    print(1)
    y_audio, sr = librosa.load(file_path, sr=22050, duration=30.0)
    
    # 1. ResNet Feature (Mel-Spektrogramm 128x128x3)
    mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < 128:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
    else:
        mel_spec_db = mel_spec_db[:, :128]
    
    X_spec = np.expand_dims(mel_spec_db, axis=-1)
    X_spec = np.repeat(X_spec, 3, axis=-1)
    X_spec = np.expand_dims(X_spec, axis=0)
    
    # 2. Tabellarische Features (Dummy-Extraktion oder Fallback auf Testarray falls Audio abweicht)
    # Für eine robuste Live-Demo nutzen wir hier die vorbereiteten Arrays, falls kein echtes Feature-Skript anliegt:
    X_seq_all = np.load("features/wav2vec_sequences.npy")
    X_tab_all = pd_tab = None # Platzhalter für Live-Extraktion
    
    # Soft-Voting Vorhersage
    p_xgb = xgb_model.predict_proba(X_tab_test_single) # Beispielhaft
    p_lstm = lstm_model.predict(X_seq_test_single, verbose=0)
    p_resnet = resnet_model.predict(X_spec, verbose=0)
    
    p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
    predicted_idx = p_ensemble.argmax(axis=1)[0]
    confidence = p_ensemble[0][predicted_idx] * 100
    
    print(f"\n--- ERGEBNIS DER LIVE-DEMO ---")
    Erkanntes Genre: {GENRES[predicted_idx].upper()} (Sicherheit: {confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python3 scripts/live_demo.py <pfad_zu_audio.wav>")
        print("Tipp: Du kannst auch 'test' übergeben, um einen zufälligen Song aus dem Testset zu prüfen.")
    elif sys.argv[1] == "test":
        # Testet direkt einen echten Datensatz-Eintrag
        X_spec = np.load("features/mel_spectrograms.npy")
        X_seq = np.load("features/wav2vec_sequences.npy")
        y = np.load("features/labels.npy")
        if X_spec.shape[-1] != 3: X_spec = np.repeat(X_spec, 3, axis=-1)
        
        import pandas as pd
        df_tab = pd.read_csv("features/statistical_features.csv")
        scaler = joblib.load("models/scaler.joblib")
        X_tab_scaled = scaler.transform(df_tab.drop(columns=["label"]))
        
        _, X_spec_test, _, X_seq_test, _, X_tab_test, _, y_test = train_test_split(
            X_spec, X_seq, X_tab_scaled, y, test_size=0.2, random_state=42
        )
        
        idx = np.randint(0, len(y_test))
        xgb_model, lstm_model, resnet_model, _ = load_ensemble_models()
        
        p_xgb = xgb_model.predict_proba(X_tab_test[idx:idx+1])
        p_lstm = lstm_model.predict(X_seq_test[idx:idx+1], verbose=0)
        p_resnet = resnet_model.predict(X_spec_test[idx:idx+1], verbose=0)
        
        p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
        pred = p_ensemble.argmax(axis=1)[0]
        true = y_test[idx]
        
        print(f"\n--- TEST-CHECK AUS DATENSATZ ---")
        print(f"Wahre Klasse: {GENRES[true]}")
        print(f"Vom V3-Ensemble vorhergesagt: {GENRES[pred]} (Konfidenz: {p_ensemble[0][pred]*100:.2f}%)")
    else:
        predict_audio(sys.argv[1])