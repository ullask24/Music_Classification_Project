import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Simplified ResNet-style block for 1D data
def res_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Dense(filters, activation='relu')(x)
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation('relu')(x)

df = pd.read_csv('features/statistical_features.csv')
X = df.drop('label', axis=1).values
y = pd.get_dummies(df['label']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

inputs = tf.keras.Input(shape=(26,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = res_block(x, 64)
x = res_block(x, 64)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, verbose=0)

# Logging
pd.DataFrame(history.history).to_csv('logs/resnet_training_metrics.csv', index=False)
y_pred = model.predict(X_test).argmax(axis=1)
with open('logs/resnet_report.txt', 'w') as f:
    f.write(classification_report(y_test.argmax(axis=1), y_pred))

model.save('models/resnet_specialist.keras')