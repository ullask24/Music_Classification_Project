import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Input
import xgboost as xgb
import joblib
import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
LIVE_DIR = "live"
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

def build_cheng_et_al_cnn(input_shape=(128, 128, 1), num_classes=10):
    """Baut das exakte 5-Layer CNN aus dem Seed-Paper nach."""
    model = Sequential(name="Seed_CNN_Cheng_2020")
    model.add(Input(shape=input_shape))
    for f in [128, 64, 32, 16, 8]:
        model.add(Conv2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(ReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_or_train_seed_model():
    os.makedirs("models", exist_ok=True)
    seed_model_path = "models/seed_cnn.keras"
    if os.path.exists(seed_model_path):
        print("Lade bereits trainiertes Seed-Paper CNN...")
        return tf.keras.models.load_model(seed_model_path)
    
    print("Trainiere Seed-Paper CNN (Cheng et al.) auf GTZAN-Daten...")
    X_spec_gtzan = np.load("features/mel_spectrograms.npy")
    y_gtzan = np.load("features/labels.npy")
    
    # Sicherstellen, dass der Input 1-Kanal (Graustufe) ist
    if len(X_spec_gtzan.shape) == 3:
        X_spec_gtzan = np.expand_dims(X_spec_gtzan, axis=-1)
    elif X_spec_gtzan.shape[-1] == 3:
        X_spec_gtzan = X_spec_gtzan[..., :1]
        
    seed_model = build_cheng_et_al_cnn(input_shape=X_spec_gtzan.shape[1:])
    seed_model.fit(X_spec_gtzan, y_gtzan, epochs=50, batch_size=32, verbose=1)
    seed_model.save(seed_model_path)
    return seed_model

def evaluate_live_folders_multi_segment():
    os.makedirs("logs", exist_ok=True)
    
    # 1. Seed-Modell vorbereiten
    seed_model = get_or_train_seed_model()
    
    # 2. V3-Spezialisten vorbereiten (Aus committee_logic_v3.py & live_demo.py)
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

    all_seed_preds = []
    all_ensemble_preds = []

    for folder_name in target_folders:
        folder_path = os.path.join(LIVE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            continue

        print(f"\n" + "="*85)
        print(f" MULTI-SEGMENT ANALYSE DES ORDNERS: {folder_name.upper()}")
        print("="*85)

        for file in files:
            file_path = os.path.join(folder_path, file)
            y_full, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            chunk_samples = TARGET_SR * DURATION
            
            print(f"\n--- Datei: {file} (Gesamtlänge: {len(y_full)/TARGET_SR:.1f}s) ---")
            
            chunk_idx = 0
            for i in range(0, max(1, len(y_full)), chunk_samples):
                y_chunk = y_full[i:i + chunk_samples]
                if len(y_chunk) < TARGET_SR * 3:
                    break
                if len(y_chunk) < chunk_samples:
                    y_chunk = np.pad(y_chunk, (0, chunk_samples - len(y_chunk)))
                else:
                    y_chunk = y_chunk[:chunk_samples]
                
                # Spectrograms
                mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=TARGET_SR, n_mels=128, n_fft=1024, hop_length=512)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)
                
                # Seed erwartet 1 Kanal, ResNet erwartet 3 Kanäle
                X_spec_seed = np.expand_dims(np.expand_dims(temp_spec, axis=-1), axis=0)
                X_spec_resnet = np.expand_dims(np.repeat(np.expand_dims(temp_spec, axis=-1), 3, axis=-1), axis=0)
                
                # Tabular & Seq
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
                
                # Vorhersagen
                p_seed = softmax(seed_model.predict(X_spec_seed, verbose=0))
                p_xgb = xgb_model.predict_proba(X_tab_scaled)
                p_lstm = softmax(lstm_model.predict(X_seq, verbose=0))
                p_resnet = softmax(resnet_model.predict(X_spec_resnet, verbose=0))
                
                # V3 Ensemble Gewichtung: 25% XGB, 45% BiLSTM, 30% ResNet[cite: 3]
                p_ensemble = (0.25 * p_xgb) + (0.45 * p_lstm) + (0.30 * p_resnet)
                
                all_seed_preds.append(p_seed[0])
                all_ensemble_preds.append(p_ensemble[0])
                
                print(f"  [Segment {chunk_idx+1} | Start: {i/TARGET_SR:.1f}s]")
                print(f"    -> SEED-CNN: {GENRES[p_seed.argmax()]:<10} ({p_seed[0, p_seed.argmax()]*100:5.2f}%)  <-- Monolith")
                print(f"    -> XGBoost:  {GENRES[p_xgb.argmax()]:<10} ({p_xgb[0, p_xgb.argmax()]*100:5.2f}%)")
                print(f"    -> BiLSTM:   {GENRES[p_lstm.argmax()]:<10} ({p_lstm[0, p_lstm.argmax()]*100:5.2f}%)")
                print(f"    -> ResNet:   {GENRES[p_resnet.argmax()]:<10} ({p_resnet[0, p_resnet.argmax()]*100:5.2f}%)")
                print(f"    -> ENSEMBLE: {GENRES[p_ensemble.argmax()]:<10} ({p_ensemble[0, p_ensemble.argmax()]*100:5.2f}%)  <-- Hybrid")
                
                chunk_idx += 1

    # --- PLOT GENERIEREN ---
    if all_seed_preds and all_ensemble_preds:
        print("\nGeneriere finales Vergleichs-Diagramm...")
        avg_seed_conf = np.mean(all_seed_preds, axis=0)
        avg_ensemble_conf = np.mean(all_ensemble_preds, axis=0)

        x = np.arange(len(GENRES))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, avg_seed_conf, width, label='Seed-Paper 5-Layer CNN', color='#f28c8c')
        ax.bar(x + width/2, avg_ensemble_conf, width, label='Ensemble V3 (Robust bei Domain Shift)', color='#8ccef2')

        ax.set_ylabel('Durchschnittliche Vorhersage-Wahrscheinlichkeit')
        ax.set_title('Domain Shift Analyse: Live-Daten Segment-Auswertung')
        ax.set_xticks(x)
        ax.set_xticklabels(GENRES, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"logs/live_domain_shift_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Grafik erfolgreich gespeichert unter: {plot_path}")

if __name__ == "__main__":
    evaluate_live_folders_multi_segment()