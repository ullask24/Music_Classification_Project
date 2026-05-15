import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('features/statistical_features.csv')
X = df.drop('label', axis=1).values
y_labels = df['label']
label_names = sorted(y_labels.unique())
label_map = {name: i for i, name in enumerate(label_names)}
y_true = y_labels.map(label_map).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lstm_model = tf.keras.models.load_model('models/lstm_specialist.keras')
resnet_model = tf.keras.models.load_model('models/resnet_specialist.keras')

X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
p1 = lstm_model.predict(X_lstm)
p2 = resnet_model.predict(X_scaled)

ensemble_probs = (p1 + p2) / 2
final_preds = np.argmax(ensemble_probs, axis=1)

acc = accuracy_score(y_true, final_preds)
cm = confusion_matrix(y_true, final_preds)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
plt.savefig('ensemble_confusion_matrix.png')

with open('final_report.txt', 'w') as f:
    f.write(f"FINAL ENSEMBLE ACCURACY: {acc * 100:.2f}%\n")