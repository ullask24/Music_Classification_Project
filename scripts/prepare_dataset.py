import os
import datetime
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio

warnings.filterwarnings("ignore")

DATA_PATH = "data/gtzan"
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock".split()
TARGET_SR = 16000
DURATION = 30
FIXED_SPEC_WIDTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
w2v_model = bundle.get_model().to(device)
w2v_model.eval()

tabular_rows = []
spectrograms = []
sequential_features = []
labels = []

os.makedirs("features", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Timestamp für das Versions- und Änderungsprotokoll
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/data_prep_log_{timestamp}.txt"

print("Starte erweiterte Feature-Extraktion mit Data Augmentation auf Gerät:", device)

def process_audio_variant(y, sr, g, genre_idx):
    """Hilfsfunktion zur Feature-Extraktion für Original- oder augmentierte Audiospuren"""
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
    temp_row.append(g)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)
    temp_spec = np.expand_dims(temp_spec, axis=-1)

    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = w2v_model.extract_features(waveform)
        latent_seq = out[0] if isinstance(out, tuple) else out
        rep = latent_seq[-1].squeeze(0).cpu().numpy()
        
        step = max(1, rep.shape[0] // 100)
        temp_seq = rep[::step][:100]
        if temp_seq.shape[0] < 100:
            temp_seq = np.pad(temp_seq, ((0, 100 - temp_seq.shape[0]), (0, 0)))

    return temp_row, temp_spec, temp_seq, genre_idx

for genre_idx, g in enumerate(GENRES):
    genre_folder = os.path.join(DATA_PATH, g)
    if not os.path.isdir(genre_folder):
        continue

    for filename in sorted(os.listdir(genre_folder)):
        songpath = os.path.join(genre_folder, filename)
        try:
            y, sr = librosa.load(songpath, sr=TARGET_SR, mono=True, duration=DURATION)
            if len(y) < TARGET_SR * DURATION:
                y = np.pad(y, (0, TARGET_SR * DURATION - len(y)))
            else:
                y = y[:TARGET_SR * DURATION]

            # A) Original
            row, spec, seq, lbl = process_audio_variant(y, sr, g, genre_idx)
            tabular_rows.append(row)
            spectrograms.append(spec)
            sequential_features.append(seq)
            labels.append(lbl)

            # B) Data Augmentation: Pitch Shifting
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0)
            row_p, spec_p, seq_p, lbl_p = process_audio_variant(y_pitch, sr, g, genre_idx)
            tabular_rows.append(row_p)
            spectrograms.append(spec_p)
            sequential_features.append(seq_p)
            labels.append(lbl_p)

            # C) Data Augmentation: Time Stretching
            y_stretch = librosa.effects.time_stretch(y, rate=1.05)
            if len(y_stretch) < TARGET_SR * DURATION:
                y_stretch = np.pad(y_stretch, (0, TARGET_SR * DURATION - len(y_stretch)))
            else:
                y_stretch = y_stretch[:TARGET_SR * DURATION]
            
            row_s, spec_s, seq_s, lbl_s = process_audio_variant(y_stretch, sr, g, genre_idx)
            tabular_rows.append(row_s)
            spectrograms.append(spec_s)
            sequential_features.append(seq_s)
            labels.append(lbl_s)

        except Exception as e:
            print(f"Überspringe Datei {filename} wegen Fehler: {e}")
            continue

col_names = "chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zcr".split()
col_names += [f"mfcc{i}" for i in range(1, 21)]
col_names += [f"spec_contrast_{i}" for i in range(7)]
col_names += [f"tonnetz_{i}" for i in range(6)]
col_names += ["label"]

df_tabular = pd.DataFrame(tabular_rows, columns=col_names)
df_tabular.to_csv("features/statistical_features.csv", index=False)

np.save("features/mel_spectrograms.npy", np.array(spectrograms, dtype=np.float32))
np.save("features/wav2vec_sequences.npy", np.array(sequential_features, dtype=np.float32))
np.save("features/labels.npy", np.array(labels, dtype=np.int64))

# Protokollierung der Änderungen und Ziele
with open(log_filename, "w") as f:
    f.write(f"--- DATA PREPARATION & AUGMENTATION LOG ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN:\n")
    f.write("1. Integration von Harmonic-Percussive Source Separation (HPSS).\n")
    f.write("2. Hinzufügen von 7 Spectral-Contrast-Bändern und 6 Tonnetz-Dimensionen.\n")
    f.write("3. Data Augmentation: Pitch-Shifting (+2 Halbtöne) und Time-Stretching (Faktor 1.05).\n\n")
    f.write("ZIELE:\n")
    f.write("- Verdreifachung des Trainingsdatensatzes zur Erhöhung der Modellrobustheit.\n")
    f.write("- Gezielte Verbesserung der Erkennungsrate bei akustisch dichten Problemgenres (wie Rock).\n")
    f.write(f"- Generierte Gesamt-Tracks (inkl. Augmentierung): {len(labels)}\n")

print(f"Extraktion & Augmentierung beendet! Protokoll gespeichert unter: {log_filename}")