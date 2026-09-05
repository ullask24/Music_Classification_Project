import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_seq = np.load("features/wav2vec_sequences.npy")
y = np.load("features/labels.npy")
y_cat = tf.keras.utils.to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_cat, test_size=0.2, random_state=42)

# Input-Shape: (100 Timesteps, 768 Wav2Vec Embedding-Dimension)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 768)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=32, verbose=1)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model.save("models/lstm_specialist.keras")
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test.argmax(axis=1), y_pred)
with open("logs/lstm_report.txt", "w") as f:
    f.write(report)

print("LSTM erfolgreich trainiert und gespeichert.")