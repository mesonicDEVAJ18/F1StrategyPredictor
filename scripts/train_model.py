import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import os
import warnings

from feature_config import MODEL_FEATURE_COLUMNS as FEATURES

warnings.filterwarnings("ignore")
os.makedirs("./results", exist_ok=True)

# LOAD DATA
df = pd.read_csv("../data/final_engineered_dataset.csv")
df = df.sort_values(["stint_number", "lap_number"]).reset_index(drop=True)

# BUILD TARGET
df["target_next_fcp"] = df.groupby("stint_number")["fuel_corrected_pace"].shift(-1)
df = df.dropna(subset=["target_next_fcp"]).reset_index(drop=True)

# FEATURES & TARGET
X = df[FEATURES].copy()
y = df["target_next_fcp"]

# CLEAN
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median(numeric_only=True))

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

# MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
}

preds = {}
fitted_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
    preds[name] = model.predict(X_test)

# SAVE MODEL WITH FEATURES
joblib.dump(
    {"model": fitted_models["XGBoost"], "features": FEATURES},
    "./results/xgb_model.pkl"
)
print("Saved model with feature metadata.")