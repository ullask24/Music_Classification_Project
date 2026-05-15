import os
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FEAT_DIR = os.path.join(ROOT_DIR, 'features')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

resnet_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'specialist_resnet.keras'))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'specialist_lstm.keras'))
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'specialist_xgboost.joblib'))

X_stats = pd.read_csv(os.path.join(FEAT_DIR, 'statistical_features.csv'))
X_seq = np.load(os.path.join(FEAT_DIR, 'temporal_sequences.npy'))
y_true_all = np.load(os.path.join(FEAT_DIR, 'y_labels.npy'))

from sklearn.model_selection import train_test_split
_, _, _, y_test = train_test_split(y_true_all, y_true_all, test_size=0.2, random_state=42, stratify=y_true_all)
indices = train_test_split(np.arange(len(y_true_all)), test_size=0.2, random_state=42, stratify=y_true_all)[1]

X_stats_test = X_stats.iloc[indices].drop('label', axis=1)
X_seq_test = X_seq[indices]

resnet_probs = resnet_model.predict(X_seq_test, verbose=0) # Note: Ensure input matches ResNet requirements
lstm_probs = lstm_model.predict(X_seq_test, verbose=0)
xgb_probs = xgb_model.predict_proba(X_stats_test)

w_resnet = 0.4
w_lstm = 0.4
w_xgb = 0.2

committee_probs = (w_resnet * resnet_probs) + (w_lstm * lstm_probs) + (w_xgb * xgb_probs)
y_pred = np.argmax(committee_probs, axis=1)
y_true = y_test

final_acc = accuracy_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='magma', xticklabels=GENRES, yticklabels=GENRES)
plt.title(f'Final Committee Decision (Accuracy: {final_acc:.2f})')
plt.savefig(os.path.join(ROOT_DIR, 'committee_confusion_matrix.png'))
plt.close()

with open(os.path.join(ROOT_DIR, 'final_report.txt'), 'w') as f:
    f.write("--- Committee Ensemble Report ---\n")
    f.write(classification_report(y_true, y_pred, target_names=GENRES))