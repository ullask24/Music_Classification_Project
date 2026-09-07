import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

print("=== STARTE FINE-TUNING RESNET-SPEZIALIST ===")

# 1. Daten laden
X_spec = np.load("features/mel_spectrograms.npy")
y = np.load("features/labels.npy")

# Auf 3 Kanäle erweitern (RGB-Duplikation) falls noch nicht geschehen
if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.2, random_state=42)

# 2. Basis-Modell (ResNet50) mit ImageNet-Gewichten laden
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Zunächst Basismodell komplett eingefroren lassen für die ersten Epochen oder gezieltes Unfreezing
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Nur die letzten 30 Layer für Fine-Tuning öffnen
    layer.trainable = False

# 3. Klassifikationskopf aufbauen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Kompilieren mit sehr kleiner Lernrate für Fine-Tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Training mit Fine-Tuning
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Modell abspeichern
model.save("models/resnet_specialist.keras")

# 6. Auswertung und Bericht mit klarem Hinweis auf Fine-Tuning
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test, y_pred)
report_filename = f"logs/resnet_finetuned_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- RESNET SPECIALIST (FINE-TUNED EXPERIMENT) ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN: Fine-Tuning aktiv (letzte 30 Layer des ResNet50 entfroren, Lernrate 1e-5).\n")
    f.write("VERGLEICHSHINWEIS: Dieses Modell verwendet Fine-Tuning im Gegensatz zum reinen Feature-Extraction-Ansatz.\n\n")
    f.write(report)

print(f"Fine-Tuned ResNet erfolgreich trainiert und gespeichert. Bericht: {report_filename}")