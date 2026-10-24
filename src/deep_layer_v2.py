import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
    df[c] = df[c].clip(lower=0)

# ---- long-memory features (windows END today: no future information) ----
for w in [7, 14, 30]:
    df[f"rain_{w}d"] = df["rain"].rolling(w).sum()
df["sm_mean7"] = df["sm_0_7"].rolling(7).mean()
df["sm_mean30"] = df["sm_0_7"].rolling(30).mean()
df["t_mean7"] = df["t_mean"].rolling(7).mean()
df = df.dropna()          # same rows used for BOTH feature sets

MET = ["t_mean", "t_max", "t_min", "t_range", "rh_mean", "rh_max", "rh_min",
       "pressure", "wind_mean", "wind_max", "rain", "rain_1d", "rain_2d"]
LONG = ["rain_7d", "rain_14d", "rain_30d", "sm_mean7", "sm_mean30", "t_mean7"]
FEATURE_SETS = {"base (paper-style)": MET, "base + long memory": MET + LONG}

def get(name):
    if name == "LR":
        return make_pipeline(StandardScaler(), LinearRegression()), {}
    if name == "RF":
        return (RandomForestRegressor(random_state=42, n_jobs=-1),
                {"n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_leaf": [1, 3]})
    return (XGBRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]})

def fit(name, X, y):
    est, grid = get(name)
    if grid:
        gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(5), scoring="r2", n_jobs=-1).fit(X, y)
        return gs.best_estimator_
    return est.fit(X, y)

n = len(df); cut = int(n * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
print(f"train {tr.index[0].date()} to {tr.index[-1].date()} | test {te.index[0].date()} to {te.index[-1].date()}")

rows = []
for fs_name, extra in FEATURE_SETS.items():
    f1 = ["sm_0_7"] + extra                 # -> 7-28 cm
    f2 = ["sm_0_7", "sm_7_28"] + extra      # -> 28-100 cm (uses ESTIMATED 7-28 at test time)
    for name in ["LR", "RF", "XGB"]:
        m1 = fit(name, tr[f1], tr["sm_7_28"])
        m2 = fit(name, tr[f2], tr["sm_28_100"])
        p1 = m1.predict(te[f1])
        te2 = te.copy(); te2["sm_7_28"] = p1
        p2 = m2.predict(te2[f2])
        for layer, y, p in [("7-28cm", te["sm_7_28"], p1), ("28-100cm", te["sm_28_100"], p2)]:
            rows.append(dict(features=fs_name, model=name, layer=layer,
                             R2=round(r2_score(y, p), 4),
                             RMSE=round(np.sqrt(mean_squared_error(y, p)), 4),
                             MAE=round(mean_absolute_error(y, p), 4)))
        print("done:", fs_name, name)

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv(ROOT / "data/processed/deep_layer_v2_results.csv", index=False)
