import os
import datetime
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
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Callbacks für Lernraten-Decay und Early Stopping gegen Overfitting
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
)

# Erhöhte Epochenanzahl
model.fit(
    X_train, y_train, 
    validation_split=0.1, 
    epochs=70, 
    batch_size=32, 
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model.save("models/lstm_specialist.keras")
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test.argmax(axis=1), y_pred)

# Versionssichere Protokollierung mit Zeitstempel
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"logs/lstm_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- LSTM SPECIALIST EXPERIMENT ({timestamp}) ---\n")
    f.write(report)

print(f"LSTM erfolgreich trainiert. Bericht gespeichert unter: {report_filename}")