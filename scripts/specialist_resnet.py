import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_spec = np.load("features/mel_spectrograms.npy")
y = np.load("features/labels.npy")

# Anpassung von 1 Kanal (Graustufen) auf 3 Kanäle (RGB) für ImageNet-ResNet
if X_spec.shape[-1] != 3:
    X_spec = np.repeat(X_spec, 3, axis=-1)

y_cat = tf.keras.utils.to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X_spec, y_cat, test_size=0.2, random_state=42)

# Vortrainiertes ResNet50 laden (ohne Klassifikationskopf)
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", 
    include_top=False, 
    input_shape=(128, 128, 3)
)

# Basis-Modell zunächst einfrieren, um stabile Initialisierung zu sichern
base_model.trainable = False

inputs = tf.keras.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

model.fit(
    X_train, y_train, 
    validation_split=0.1, 
    epochs=40, 
    batch_size=32, 
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model.save("models/resnet_specialist.keras")
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test.argmax(axis=1), y_pred)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"logs/resnet_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(f"--- RESNET SPECIALIST (PRETRAINED IMAGENET) ({timestamp}) ---\n")
    f.write("VERÄNDERUNGEN: Nutzung von ResNet50 mit ImageNet-Gewichten, Kanäle auf 3 erweitert, Feature Extractor eingefroren.\n")
    f.write("ZIELE: Verhinderung des Trainingskollapses bei kleinen Audio-Datasets durch vortrainierte visuelle Muster.\n\n")
    f.write(report)

print(f"Vortrainiertes ResNet erfolgreich trainiert. Bericht: {report_filename}")