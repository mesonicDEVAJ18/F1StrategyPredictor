import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------
# LOAD MODEL + FEATURES
# ------------------------------------------------------
bundle = joblib.load("./results/xgb_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]        # exact model input order

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
df = pd.read_csv("../data/final_engineered_dataset.csv")

# SAFE SORT (driver_number removed from dataset)
sort_cols = []
if "driver_number" in df.columns:
    sort_cols.append("driver_number")

sort_cols += ["stint_number", "lap_number"]
df = df.sort_values(sort_cols).reset_index(drop=True)

# ------------------------------------------------------
# NEXT-LAP TARGET
# ------------------------------------------------------
if "driver_number" in df.columns:
    df["target_next_fcp"] = df.groupby(["driver_number", "stint_number"])["fuel_corrected_pace"].shift(-1)
else:
    df["target_next_fcp"] = df.groupby(["stint_number"])["fuel_corrected_pace"].shift(-1)

df = df.dropna(subset=["target_next_fcp"]).reset_index(drop=True)

X = df[FEATURES].values
y = df["target_next_fcp"].values

# ------------------------------------------------------
# CROSS-VALIDATION
# ------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_mae = []
cv_rmse = []

for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    mae = mean_absolute_error(y[test_idx], pred)
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))

    cv_mae.append(mae)
    cv_rmse.append(rmse)

print("CV MAE:", np.mean(cv_mae))
print("CV RMSE:", np.mean(cv_rmse))

# ------------------------------------------------------
# RESIDUAL ANALYSIS
# ------------------------------------------------------
pred_full = model.predict(X)
residuals = y - pred_full

N = min(3000, len(y))

plt.figure(figsize=(8,4))
plt.scatter(pred_full[:N], residuals[:N], s=4, alpha=0.5)
plt.axhline(0, color="red")
plt.title("Residual Plot")
plt.xlabel("Predicted Fuel-Corrected Pace")
plt.ylabel("Residual")
plt.savefig("./results/residual_plot.png", dpi=300)

# ------------------------------------------------------
# STRATEGY VALIDATION — MONTE CARLO TEMPLATE
# (You will plug in your simulator here)
# ------------------------------------------------------
def monte_carlo_sim(strategy_sim_fn, start_features, base_time, runs=500):
    results = []
    for _ in range(runs):
        noisy_pit = 22 + np.random.normal(0, 1.5)
        noisy_model = lambda x: model.predict(x) + np.random.normal(0, 0.05)

        t = strategy_sim_fn(noisy_model, noisy_pit)
        results.append(t)
    return np.array(results)

# ------------------------------------------------------
# T-TEST EXAMPLE
# ------------------------------------------------------
s1 = np.random.normal(4800, 10, size=500)
s2 = np.random.normal(4820, 10, size=500)

t, p = ttest_ind(s1, s2)
print("T-test p-value:", p)

# ------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------
summary = pd.DataFrame({
    "CV_MAE": [np.mean(cv_mae)],
    "CV_RMSE": [np.mean(cv_rmse)],
    "t_test_p": [p]
})
summary.to_csv("./results/validation_summary.csv", index=False)

print("\nValidation complete. Summary saved.")