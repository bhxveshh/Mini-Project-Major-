import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
df["sm_0_7"] = df["sm_0_7"].clip(lower=0)
df = df.dropna(subset=["sm_0_7"])

FC = ["t_max", "t_min", "rh_mean", "rain"]
H = 7
sm = df["sm_0_7"].values
met = df[FC].values.astype(float)
n = len(df)
cut = int(n * 0.8)

# train on OBSERVED weather (as in the paper), best params from the previous run
t_idx = np.arange(2, cut)
X_tr = np.column_stack([sm[t_idx - 1], sm[t_idx - 2], met[t_idx]])
model = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.1,
                     random_state=42, n_jobs=-1).fit(X_tr, sm[t_idx])

origins = np.arange(cut - 1, n - H)
actual = np.column_stack([sm[origins + h] for h in range(1, H + 1)])

def forecast(met_future):
    prev1, prev2 = sm[origins].copy(), sm[origins - 1].copy()
    out = np.zeros((len(origins), H))
    for h in range(1, H + 1):
        X = np.column_stack([prev1, prev2, met_future[origins + h]])
        p = np.clip(model.predict(X), 0, None)
        out[:, h - 1] = p
        prev2, prev1 = prev1, p
    return out

def degraded(rain_sigma, t_sd, rh_sd, seed=0):
    rng = np.random.RandomState(seed)
    m = met.copy()
    m[:, 0] += rng.normal(0, t_sd, n)                  # t_max
    m[:, 1] += rng.normal(0, t_sd, n)                  # t_min
    m[:, 2] = np.clip(m[:, 2] + rng.normal(0, rh_sd, n), 0, 100)
    m[:, 3] = m[:, 3] * rng.lognormal(0, rain_sigma, n)  # multiplicative rain error
    return m

scenarios = {
    "Oracle": met,
    "Mild noise": degraded(0.5, 1.0, 5.0),
    "Heavy noise": degraded(1.0, 2.0, 10.0),
    "No forecast": np.tile(met[:cut].mean(axis=0), (n, 1)),
}

table_r2, table_rmse = {}, {}
for name, mf in scenarios.items():
    pred = forecast(mf)
    table_r2[name] = [r2_score(actual[:, h], pred[:, h]) for h in range(H)]
    table_rmse[name] = [np.sqrt(mean_squared_error(actual[:, h], pred[:, h])) for h in range(H)]

persist = np.column_stack([sm[origins]] * H)
table_r2["Persistence"] = [r2_score(actual[:, h], persist[:, h]) for h in range(H)]
table_rmse["Persistence"] = [np.sqrt(mean_squared_error(actual[:, h], persist[:, h])) for h in range(H)]

idx = pd.Index(range(1, H + 1), name="day")
r2 = pd.DataFrame(table_r2, index=idx).round(4)
rmse = pd.DataFrame(table_rmse, index=idx).round(4)
print("R2 by forecast day:\n", r2.to_string())
print("\nRMSE by forecast day:\n", rmse.to_string())

r2.to_csv(ROOT / "data/processed/robustness_r2.csv")
rmse.to_csv(ROOT / "data/processed/robustness_rmse.csv")
