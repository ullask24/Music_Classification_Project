import tensorflow as tf
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and models
df = pd.read_csv('features/statistical_features.csv')
# ... (Prep X_test, y_test same as others) ...

lstm_model = tf.keras.models.load_model('models/lstm_specialist.keras')
resnet_model = tf.keras.models.load_model('models/resnet_specialist.keras')
# xgb_model = ... load xgb ...

# Get Probabilities
p1 = lstm_model.predict(X_test_reshaped)
p2 = resnet_model.predict(X_test)
# p3 = xgb_model.predict_proba(X_test)

# Simple Average Ensemble
final_preds = np.argmax((p1 + p2) / 2, axis=1)
accuracy = accuracy_score(y_test_labels, final_preds)

# Save Confusion Matrix
cm = confusion_matrix(y_test_labels, final_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, cmap='viridis')
plt.savefig('results_confusion_matrix.png')

with open('final_report.txt', 'w') as f:
    f.write(f"FINAL ENSEMBLE ACCURACY: {accuracy * 100:.2f}%\n")