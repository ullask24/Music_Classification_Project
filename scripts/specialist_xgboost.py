import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FEAT_DIR = os.path.join(ROOT_DIR, 'features')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FEAT_CSV = os.path.join(FEAT_DIR, 'statistical_features.csv')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'specialist_xgboost.joblib')

df = pd.read_csv(FEAT_CSV)
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, XGB_MODEL_PATH)

y_pred = model.predict(X_test)
genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

importances = model.feature_importances_
feat_import_series = pd.Series(importances, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 12))
feat_import_series.plot(kind='barh', color='teal')
plt.title("Ventral Stream Specialist: Feature Importance")
plt.savefig(os.path.join(ROOT_DIR, 'xgb_feature_importance.png'))
plt.close()

cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
plt.title('Ventral Stream (XGBoost) Confusion Matrix')
plt.savefig(os.path.join(ROOT_DIR, 'xgb_confusion_matrix.png'))
plt.close()