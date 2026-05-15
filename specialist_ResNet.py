import os
import librosa
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "gtzan")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def get_spectrogram(path):
    try:
        y, sr = librosa.load(path, duration=30.0)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db_norm = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_resized = cv2.resize(S_db_norm, IMG_SIZE)
        img_rgb = cv2.merge([img_resized, img_resized, img_resized])
        return img_rgb
    except Exception:
        return None

X, y = [], []
for label, genre in enumerate(GENRES):
    folder = os.path.join(DATA_DIR, genre)
    if not os.path.exists(folder): continue
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.au'))]
    for file in tqdm(files, desc=f"Processing {genre}"):
        img = get_spectrogram(os.path.join(folder, file))
        if img is not None:
            X.append(img)
            y.append(label)

X = np.array(X, dtype='float32')
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_val_cat = tf.keras.utils.to_categorical(y_val, 10)

X_train = tf.keras.applications.resnet_v2.preprocess_input(X_train)
X_val = tf.keras.applications.resnet_v2.preprocess_input(X_val)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)

base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(val_ds, validation_data=val_ds, epochs=10, verbose=0)

model.save(os.path.join(MODEL_DIR, 'specialist_resnet.keras'))

y_pred_probs = model.predict(val_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val_cat, axis=1)

cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
plt.savefig(os.path.join(ROOT_DIR, 'resnet_confusion_matrix.png'))
plt.close()