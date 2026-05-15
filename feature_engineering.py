import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "gtzan")
FEAT_DIR = os.path.join(ROOT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

SR = 22050
CLIP_SEC = 3
N_MFCC = 20
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_ventral_features(segment, sr):
    sc = librosa.feature.spectral_centroid(y=segment, sr=sr)
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    
    return {
        'centroid_mean': np.mean(sc),
        'centroid_var': np.var(sc),
        'chroma_mean': np.mean(chroma),
        'chroma_var': np.var(chroma),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms)
    }

def extract_dorsal_features(segment, sr):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T 

stats_data = []
temporal_data = []
labels = []
samples_per_clip = SR * CLIP_SEC

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Data directory not found at {RAW_DATA_PATH}")

for genre in GENRES:
    folder = os.path.join(RAW_DATA_PATH, genre)
    if not os.path.exists(folder):
        continue
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.wav', '.au'))])
    
    for fname in tqdm(files, desc=f"Processing {genre}"):
        path = os.path.join(folder, fname)
        try:
            y, _ = librosa.load(path, sr=SR, duration=30.0)
            
            if len(y) < samples_per_clip:
                continue

            for start in range(0, len(y) - samples_per_clip + 1, samples_per_clip):
                seg = y[start : start + samples_per_clip]
                
                v_feat = extract_ventral_features(seg, SR)
                d_feat = extract_dorsal_features(seg, SR)
                
                if v_feat is not None and d_feat is not None:
                    stats_data.append(v_feat)
                    temporal_data.append(d_feat)
                    labels.append(genre)
        except Exception as e:
            continue

le = LabelEncoder()
y_encoded = le.fit_transform(labels)

np.save(os.path.join(FEAT_DIR, 'y_labels.npy'), y_encoded)
np.save(os.path.join(FEAT_DIR, 'label_names.npy'), le.classes_)

df = pd.DataFrame(stats_data)
df['label'] = y_encoded
df.to_csv(os.path.join(FEAT_DIR, 'statistical_features.csv'), index=False)

np.save(os.path.join(FEAT_DIR, 'temporal_sequences.npy'), np.array(temporal_data))