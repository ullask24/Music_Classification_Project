import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv('features/statistical_features.csv')

X = df.drop('label', axis=1).values
y_labels = df['label']

label_names = sorted(y_labels.unique())
label_map = {name: i for i, name in enumerate(label_names)}
y_ints = y_labels.map(label_map).values
y = tf.keras.utils.to_categorical(y_ints, num_classes=10)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 26)), 
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train, 
    validation_split=0.1, 
    epochs=50, 
    batch_size=32, 
    verbose=1
)

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

pd.DataFrame(history.history).to_csv('logs/lstm_training_metrics.csv', index=False)

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
report = classification_report(y_true, y_pred, target_names=label_names)
with open('logs/lstm_report.txt', 'w') as f:
    f.write(report)

model.save('models/lstm_specialist.keras')