import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv("features/statistical_features.csv")
le = LabelEncoder()
y = le.fit_transform(df["label"])
X = df.drop("label", axis=1).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Modell und Scaler sichern
model.save_model("models/xgb_specialist.json")
joblib.dump(scaler, "models/scaler.joblib")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)
with open("logs/xgb_report.txt", "w") as f:
    f.write(report)

print("XGBoost erfolgreich trainiert und gespeichert.")