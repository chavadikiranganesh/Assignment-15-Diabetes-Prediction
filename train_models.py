import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "diabetes.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

X = df.drop(columns=["Outcome"]).copy()
y = df["Outcome"]

# Follow the assignment preprocessing: zero values in these measurements
# are treated as missing and imputed with the training median.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    X[col] = X[col].replace(0, float("nan"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

lgbm_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LGBMClassifier(random_state=42, verbosity=-1))
])

xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05
    ))
])

lgbm_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

joblib.dump(lgbm_pipe, MODEL_DIR / "lightgbm_model.pkl")
joblib.dump(xgb_pipe, MODEL_DIR / "xgboost_model.pkl")

print("Saved:")
print(MODEL_DIR / "lightgbm_model.pkl")
print(MODEL_DIR / "xgboost_model.pkl")
