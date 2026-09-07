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

print("=== STARTE TIEFERES RESNET FINE-TUNING (CONV4 + CONV5) ===")

X_spec = np.load("features/mel_spectrograms.npy")
y = np.load("features/labels.npy")

if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.2, random_state=42)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Aggressiveres Unfreezing: Alle Layer ab dem Block 'conv4_block1_out' freigeben
base_model.trainable = True
trainable_flag = False
for layer in base_model.layers:
    if 'conv4_block1_out' in layer.name:
        trainable_flag = True
    layer.trainable = trainable_flag

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=5e-6), # Noch kleinere Lernrate wegen größerer Anpassungstiefe
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=35,
    batch_size=32,
    callbacks=callbacks
)

model.save("models/resnet_specialist_deep.keras")

y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test, y_pred)
report_filename = f"logs/resnet_deep_finetuned_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- RESNET SPECIALIST (DEEP FINE-TUNING EXPERIMENT) ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN: Erweitertes Unfreezing ab conv4_block aufwärts, Lernrate 5e-6.\n")
    f.write(report)

print(f"Deep Fine-Tuned ResNet gespeichert. Bericht: {report_filename}")