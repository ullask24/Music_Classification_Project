import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import traceback

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
print("Starte multimodale Feature-Extraktion...")

for genre_idx, g in enumerate(GENRES):
    genre_folder = os.path.join(DATA_PATH, g)
    if not os.path.isdir(genre_folder):
        continue

for filename in sorted(os.listdir(genre_folder)):
            songpath = os.path.join(genre_folder, filename)
            try:
                # Audio laden mit Fallback auf librosa's Standard-Load falls sr abweicht
                y, sr = librosa.load(songpath, sr=TARGET_SR, mono=True, duration=DURATION)
                if len(y) < TARGET_SR * DURATION:
                    y = np.pad(y, (0, TARGET_SR * DURATION - len(y)))
                else:
                    y = y[:TARGET_SR * DURATION]

                # ... (restlicher Extraktionscode für Tabular, Mel-Spec und Wav2Vec bleibt gleich)
                
            except Exception as e:
                print(f"Überspringe beschädigte Datei {filename}: {e}")
                continue

            # 2. Tabellarisch
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            temp_row = [
                np.mean(chroma), np.mean(rmse), np.mean(spec_cent),
                np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
            ]
            for m in mfcc:
                temp_row.append(np.mean(m))
            temp_row.append(g)

            # 3. Spektrogramm
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            temp_spec = librosa.util.fix_length(mel_norm, size=FIXED_SPEC_WIDTH, axis=1)

            # 4. Wav2Vec (LSTM)
            waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                # Extraktion (Wav2Vec2 gibt oft ein Tuple zurück)
                out = w2v_model.extract_features(waveform)
                # Falls Tuple (features, lengths), nimm features
                if isinstance(out, tuple):
                    latent_seq = out[0]
                else:
                    latent_seq = out
                
                rep = latent_seq[-1].squeeze(0).cpu().numpy()
                
                step = max(1, rep.shape[0] // 100)
                temp_seq = rep[::step][:100]
                if temp_seq.shape[0] < 100:
                    temp_seq = np.pad(temp_seq, ((0, 100 - temp_seq.shape[0]), (0, 0)))

            # --- ALLES ERFOLGREICH: ERST JETZT IN DIE LISTEN PACKEN ---
            tabular_rows.append(temp_row)
            spectrograms.append(temp_spec)
            sequential_features.append(temp_seq)
            labels.append(genre_idx)

        except Exception as e:
            print(f"Überspringe {filename} wegen Fehler: {e}")
            traceback.print_exc() # Gibt den exakten Fehler aus!
            continue

# Tabellarische Daten abspeichern
col_names = "chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zcr".split()
col_names += [f"mfcc{i}" for i in range(1, 21)] + ["label"]
df_tabular = pd.DataFrame(tabular_rows, columns=col_names)
df_tabular.to_csv("features/statistical_features.csv", index=False)

# numpy Arrays speichern
np.save("features/mel_spectrograms.npy", np.array(spectrograms, dtype=np.float32))
np.save("features/wav2vec_sequences.npy", np.array(sequential_features, dtype=np.float32))
np.save("features/labels.npy", np.array(labels, dtype=np.int64))

print(f"Extraktion abgeschlossen! {len(labels)} Tracks verarbeitet.")