import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)

# ---- cleaning ----
for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
    df[c] = df[c].clip(lower=0)
df = df.dropna()

# ---- quick sanity check (run once, then you can delete) ----
print("Layer correlations:\n", df[["sm_0_7", "sm_7_28", "sm_28_100"]].corr().round(3))
print("\nMonthly mean surface moisture:\n", df["sm_0_7"].resample("MS").mean().round(3).to_string())

MET = ["t_mean", "t_max", "t_min", "t_range", "rh_mean", "rh_max", "rh_min",
       "pressure", "wind_mean", "wind_max", "rain", "rain_1d", "rain_2d"]

def get_models():
    return {
        "LR": (make_pipeline(StandardScaler(), LinearRegression()), {}),
        "RF": (RandomForestRegressor(random_state=42, n_jobs=-1),
               {"n_estimators": [100, 200], "max_depth": [5, 7, 10],
                "min_samples_leaf": [1, 2, 4]}),
        "XGB": (XGBRegressor(random_state=42, n_jobs=-1),
                {"n_estimators": [200, 400], "max_depth": [2, 3, 5],
                 "learning_rate": [0.03, 0.1]}),
    }

def fit(name, X, y, cv):
    est, grid = get_models()[name]
    gs = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=-1) if grid else est
    gs.fit(X, y)
    return gs.best_estimator_ if grid else gs

def metrics(y, p):
    return (r2_score(y, p), np.sqrt(mean_squared_error(y, p)), mean_absolute_error(y, p))

results = []
n = len(df); cut = int(n * 0.8)

splits = {
    "chronological": (np.arange(cut), np.arange(cut, n), TimeSeriesSplit(5)),
    "random": None,
}
rng = np.random.RandomState(42)
perm = rng.permutation(n)
splits["random"] = (perm[:cut], perm[cut:], KFold(5, shuffle=True, random_state=42))

for split_name, (tr_i, te_i, cv) in splits.items():
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    for use_met in [False, True]:
        met = MET if use_met else []
        f1 = ["sm_0_7"] + met                 # -> 7-28 cm
        f2 = ["sm_0_7", "sm_7_28"] + met      # -> 28-100 cm
        for name in ["LR", "RF", "XGB"]:
            m1 = fit(name, tr[f1], tr["sm_7_28"], cv)
            m2 = fit(name, tr[f2], tr["sm_28_100"], cv)

            p1 = m1.predict(te[f1])                           # layer 2 estimate
            te2 = te.copy(); te2["sm_7_28"] = p1              # feed ESTIMATE into next step
            p2 = m2.predict(te2[f2])                          # layer 3 estimate (chained)

            for layer, y, p in [("7-28cm", te["sm_7_28"], p1),
                                ("28-100cm", te["sm_28_100"], p2)]:
                r2, rmse, mae = metrics(y, p)
                results.append(dict(split=split_name, weather=use_met, model=name,
                                    layer=layer, R2=round(r2, 4),
                                    RMSE=round(rmse, 4), MAE=round(mae, 4)))
            print(f"done: {split_name} | met={use_met} | {name}")

res = pd.DataFrame(results)
out = ROOT / "data/processed/deep_layer_results.csv"
res.to_csv(out, index=False)
print("\nSaved:", out)
print(res.to_string(index=False))