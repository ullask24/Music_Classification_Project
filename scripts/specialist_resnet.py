import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_spec = np.load("features/mel_spectrograms.npy")
y = np.load("features/labels.npy")

X_spec = np.expand_dims(X_spec, axis=-1)
y_cat = tf.keras.utils.to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X_spec, y_cat, test_size=0.2, random_state=42)

def residual_block_2d(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

inputs = tf.keras.Input(shape=(128, 128, 1))
x = tf.keras.layers.Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

x = residual_block_2d(x, 32)
x = residual_block_2d(x, 64)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = residual_block_2d(x, 128)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Callbacks für dynamische Lernraten-Anpassung und Stopp bei Stagnation
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# Erhöhte Epochenanzahl
model.fit(
    X_train, y_train, 
    validation_split=0.1, 
    epochs=60, 
    batch_size=32, 
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model.save("models/resnet_specialist.keras")
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test.argmax(axis=1), y_pred)

# Versionssichere Protokollierung mit Zeitstempel
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"logs/resnet_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- RESNET SPECIALIST EXPERIMENT ({timestamp}) ---\n")
    f.write(report)

print(f"ResNet erfolgreich trainiert. Bericht gespeichert unter: {report_filename}")