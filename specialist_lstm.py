import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FEAT_DIR = os.path.join(ROOT_DIR, 'features')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(FEAT_DIR, 'temporal_sequences.npy')) 
y = np.load(os.path.join(FEAT_DIR, 'y_labels.npy'))

N, TimeSteps, Features = X.shape
scaler = StandardScaler()
X_flatten = X.reshape(-1, Features)
X_scaled_2d = scaler.fit_transform(X_flatten)
X = X_scaled_2d.reshape(N, TimeSteps, Features)

y_onehot = tf.keras.utils.to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(TimeSteps, Features)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, 
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test), 
    shuffle=True,
    verbose=0
)

model.save(os.path.join(MODEL_DIR, 'specialist_lstm.keras'))

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(history.history['accuracy'], label='train')
ax[0].plot(history.history['val_accuracy'], label='test')
ax[0].set_title('Accuracy')
ax[1].plot(history.history['loss'], label='train')
ax[1].plot(history.history['val_loss'], label='test')
ax[1].set_title('Loss')
plt.savefig(os.path.join(ROOT_DIR, 'lstm_training_report.png'))
plt.close()

cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=genres, yticklabels=genres, cmap='magma')
plt.savefig(os.path.join(ROOT_DIR, 'lstm_confusion_matrix.png'))
plt.close()