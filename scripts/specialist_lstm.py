import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load Data
df = pd.read_csv('features/statistical_features.csv')
X = df.drop('label', axis=1).values
y = pd.get_dummies(df['label']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(1, 26), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and Log
print("Training LSTM...")
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=0)

# Save Metrics for GitHub
os.makedirs('logs', exist_ok=True)
pd.DataFrame(history.history).to_csv('logs/lstm_training_metrics.csv', index=False)

# Detailed Report
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
report = classification_report(y_true, y_pred)
with open('logs/lstm_report.txt', 'w') as f:
    f.write(report)

model.save('models/lstm_specialist.keras')