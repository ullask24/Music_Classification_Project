import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

print("=== STARTE LSTM-OPTIMIERUNG (BIDIRECTIONAL LSTM) ===")

# 1. Sequenz-Daten laden
X_seq = np.load("features/wav2vec_sequences.npy")
y = np.load("features/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

# 2. BiLSTM Modell aufbauen
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_seq.shape[1], X_seq.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Training mit Callbacks
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=70,
    batch_size=32,
    callbacks=callbacks
)

# Modell abspeichern
model.save("models/lstm_specialist_bilstm.keras")

# 4. Auswertung und Evidenz-Dokumentation für den Vergleich
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test, y_pred)
report_filename = f"logs/lstm_bilstm_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- LSTM SPECIALIST EXPERIMENT - BIDIRECTIONAL ARCHITECTURE ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN & VERGLEICH:\n")
    f.write("1. Architekturwechsel von einfachem LSTM auf Bidirektionales LSTM (BiLSTM).\n")
    f.write("2. Hinzufügen von BatchNormalization und erweitertem Dropout (0.3/0.2) zur Regularisierung.\n")
    f.write("3. Vergleich zur Baseline: Standard-LSTM erreichte 71% Accuracy.\n\n")
    f.write(report)

print(f"BiLSTM erfolgreich trainiert. Bericht gespeichert unter: {report_filename}")