import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('features/statistical_features.csv')
le = LabelEncoder()
X = df.drop('label', axis=1)
y = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Log Importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).to_csv('logs/xgb_feature_importance.csv')

# Log Report
y_pred = model.predict(X_test)
with open('logs/xgb_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))