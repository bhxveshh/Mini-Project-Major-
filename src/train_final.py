import json
import warnings
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
    df[c] = df[c].clip(lower=0)

# ---------- 1. 7-day forecaster: SM(t) from SM(t-1), SM(t-2), weather(t) ----------
FC = ["t_max", "t_min", "rh_mean", "rain"]
fdf = df.dropna(subset=["sm_0_7"]).copy()
fdf["sm_l1"] = fdf["sm_0_7"].shift(1)
fdf["sm_l2"] = fdf["sm_0_7"].shift(2)
fdf = fdf.dropna(subset=["sm_l1", "sm_l2"])
FCOLS = ["sm_l1", "sm_l2"] + FC

fc_model = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.1,
                        random_state=42, n_jobs=-1).fit(fdf[FCOLS], fdf["sm_0_7"])
joblib.dump(fc_model, ROOT / "models/forecast_xgb.joblib")
print("Forecast model trained on", fdf.index[0].date(), "to", fdf.index[-1].date(), f"({len(fdf)} days)")

# ---------- 2. deep-layer chain with long-memory features (same as deep_layer_v2) ----------
for w in [7, 14, 30]:
    df[f"rain_{w}d"] = df["rain"].rolling(w).sum()
df["sm_mean7"] = df["sm_0_7"].rolling(7).mean()
df["sm_mean30"] = df["sm_0_7"].rolling(30).mean()
df["t_mean7"] = df["t_mean"].rolling(7).mean()
ddf = df.dropna()

MET = ["t_mean", "t_max", "t_min", "t_range", "rh_mean", "rh_max", "rh_min",
       "pressure", "wind_mean", "wind_max", "rain", "rain_1d", "rain_2d"]
LONG = ["rain_7d", "rain_14d", "rain_30d", "sm_mean7", "sm_mean30", "t_mean7"]
F1 = ["sm_0_7"] + MET + LONG                      # -> 7-28 cm
F2 = ["sm_0_7", "sm_7_28"] + MET + LONG           # -> 28-100 cm

grid = {"n_estimators": [200, 400], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]}
def fit(X, y):
    gs = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), grid,
                      cv=TimeSeriesSplit(5), scoring="r2", n_jobs=-1).fit(X, y)
    print("  best params:", gs.best_params_)
    return gs.best_estimator_

print("Deep-layer models trained on", ddf.index[0].date(), "to", ddf.index[-1].date(), f"({len(ddf)} days)")
m1 = fit(ddf[F1], ddf["sm_7_28"])
m2 = fit(ddf[F2], ddf["sm_28_100"])
joblib.dump(m1, ROOT / "models/deep_7_28.joblib")
joblib.dump(m2, ROOT / "models/deep_28_100.joblib")

meta = {"forecast_cols": FCOLS, "deep1_cols": F1, "deep2_cols": F2,
        "trained_through": str(df.index[-1].date()),
        "note": "Trained on ERA5-Land (Open-Meteo archive) data for Mangaluru; deployment models, not test scores."}
(ROOT / "models/meta.json").write_text(json.dumps(meta, indent=2))
print("\nSaved to models/:", sorted(p.name for p in (ROOT / "models").iterdir()))
