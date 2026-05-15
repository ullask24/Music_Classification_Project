import os
import librosa
import numpy as np
import pandas as pd

def extract_features(data_path='data/gtzan'):
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    data = []
    
    print("Starting feature extraction...")
    for g in genres:
        for filename in os.listdir(os.path.join(data_path, g)):
            songpath = os.path.join(data_path, g, filename)
            y, sr = librosa.load(songpath, mono=True, duration=30)
            
            # Statistical Features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            
            # Row assembly
            to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            data.append(to_append.split())

    # Save CSV
    column_names = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zcr'.split()
    for i in range(1, 21):
        column_names.append(f'mfcc{i}')
    column_names.append('label')
    
    df = pd.DataFrame(data, columns=column_names)
    os.makedirs('features', exist_ok=True)
    df.to_csv('features/statistical_features.csv', index=False)
    
    # Save Summary for GitHub
    with open('logs/features_summary.txt', 'w') as f:
        f.write(f"Total processed: {len(df)} tracks\n")
        f.write(df.groupby('label').size().to_string())
    print("Features saved to features/statistical_features.csv")

if __name__ == "__main__":
    extract_features()