import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
df["sm_0_7"] = df["sm_0_7"].clip(lower=0)
df = df.dropna(subset=["sm_0_7"])

FC = ["t_max", "t_min", "rh_mean", "rain"]   # the paper's forecast-weather inputs
COLS = ["sm_l1", "sm_l2"] + FC
H = 7                                         # forecast horizon (days)

sm = df["sm_0_7"].values
met = df[FC].values
n = len(df)
cut = int(n * 0.8)

# ---- training table: predict SM(t) from SM(t-1), SM(t-2), weather(t) ----
t_idx = np.arange(2, cut)                     # training targets only in train period
X_tr = pd.DataFrame(np.column_stack([sm[t_idx - 1], sm[t_idx - 2], met[t_idx]]), columns=COLS)
y_tr = sm[t_idx]

models = {
    "LR": make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr),
    "XGB": GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]},
        cv=TimeSeriesSplit(5), scoring="r2", n_jobs=-1,
    ).fit(X_tr, y_tr),
}
print("Best XGB params:", models["XGB"].best_params_)

# ---- recursive forecasting from every origin in the test period ----
# origin s: we know SM(s) and SM(s-1), forecast days s+1 ... s+H
origins = np.arange(cut - 1, n - H)           # first target (s+1) is >= cut -> no leakage
actual = np.column_stack([sm[origins + h] for h in range(1, H + 1)])

def recursive_forecast(model):
    prev1, prev2 = sm[origins].copy(), sm[origins - 1].copy()
    out = np.zeros((len(origins), H))
    for h in range(1, H + 1):
        Xh = pd.DataFrame(np.column_stack([prev1, prev2, met[origins + h]]), columns=COLS)
        p = np.clip(model.predict(Xh), 0, None)
        out[:, h - 1] = p
        prev2, prev1 = prev1, p               # feed prediction into next day
    return out

forecasts = {name: recursive_forecast(m) for name, m in models.items()}
forecasts["Persistence"] = np.column_stack([sm[origins]] * H)

rows = []
for name, pred in forecasts.items():
    for h in range(H):
        rows.append(dict(model=name, day=h + 1,
                         R2=round(r2_score(actual[:, h], pred[:, h]), 4),
                         RMSE=round(np.sqrt(mean_squared_error(actual[:, h], pred[:, h])), 4)))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "data/processed/forecast_results.csv", index=False)

print("\nR2 by forecast day:")
print(res.pivot(index="day", columns="model", values="R2").to_string())
print("\nRMSE by forecast day:")
print(res.pivot(index="day", columns="model", values="RMSE").to_string())

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name in forecasts:
    d = res[res.model == name]
    ax[0].plot(d.day, d.R2, marker="o", label=name)
    ax[1].plot(d.day, d.RMSE, marker="o", label=name)
ax[0].set_xlabel("Forecast day"); ax[0].set_ylabel("R²"); ax[0].legend()
ax[1].set_xlabel("Forecast day"); ax[1].set_ylabel("RMSE (m³/m³)")
plt.tight_layout()
plt.savefig(ROOT / "data/processed/forecast_horizon.png", dpi=150)
print("\nSaved results CSV and forecast_horizon.png in data/processed/")
