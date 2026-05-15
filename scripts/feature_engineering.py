import os
import librosa
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def extract_features(data_path='data/gtzan'):
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    data = []
    
    print("Starting feature extraction...")
    for g in genres:
        print(f"Processing genre: {g}")
        genre_folder = os.path.join(data_path, g)
        
        for filename in os.listdir(genre_folder):
            songpath = os.path.join(genre_folder, filename)
            
            try:
                # Load audio
                y, sr = librosa.load(songpath, mono=True, duration=30)
                
                # Extraction
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rmse = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                
                to_append = [
                    np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), 
                    np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
                ]
                for e in mfcc:
                    to_append.append(np.mean(e))
                to_append.append(g)
                data.append(to_append)
                
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

    column_names = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zcr'.split()
    for i in range(1, 21):
        column_names.append(f'mfcc{i}')
    column_names.append('label')
    
    df = pd.DataFrame(data, columns=column_names)
    os.makedirs('features', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    df.to_csv('features/statistical_features.csv', index=False)
    
    with open('logs/features_summary.txt', 'w') as f:
        f.write(f"Total processed successfully: {len(df)} tracks\n")
        f.write(df.groupby('label').size().to_string())
    
    print(f"Extraction finished. Saved {len(df)} tracks.")

if __name__ == "__main__":
    extract_features()