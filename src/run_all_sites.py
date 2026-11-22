"""
Run the FULL analysis pipeline for all three sites:
  1. Surface forecast (7-day XGBoost recursive)
  2. Deep layer estimation v1 (random vs chronological split)
  3. Deep layer estimation v2 (long-memory features)
  4. Robustness analysis
  5. Train final models
"""
import json, warnings, shutil
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

SITES = ["mangaluru", "mysore", "shivamogga"]

FC = ["t_max", "t_min", "rh_mean", "rain"]
MET = ["t_mean", "t_max", "t_min", "t_range", "rh_mean", "rh_max", "rh_min",
       "pressure", "wind_mean", "wind_max", "rain", "rain_1d", "rain_2d"]
LONG = ["rain_7d", "rain_14d", "rain_30d", "sm_mean7", "sm_mean30", "t_mean7"]
H = 7  # forecast horizon


# ─────────────────────────── helpers ───────────────────────────
def load_data(site):
    df = pd.read_csv(ROOT / f"data/raw/{site}_daily.csv", index_col=0, parse_dates=True)
    return df


def make_dirs(site):
    proc = ROOT / f"data/processed/{site}"
    proc.mkdir(parents=True, exist_ok=True)
    mdl = ROOT / f"models/{site}"
    mdl.mkdir(parents=True, exist_ok=True)
    return proc, mdl


# ─────────────────────────── Step 1: surface forecast ───────────────────────────
def run_forecast(df, proc):
    df = df.copy()
    df["sm_0_7"] = df["sm_0_7"].clip(lower=0)
    df = df.dropna(subset=["sm_0_7"])

    COLS = ["sm_l1", "sm_l2"] + FC
    sm = df["sm_0_7"].values
    met = df[FC].values
    n = len(df)
    cut = int(n * 0.8)

    t_idx = np.arange(2, cut)
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
    print("    Best XGB params:", models["XGB"].best_params_)

    origins = np.arange(cut - 1, n - H)
    actual = np.column_stack([sm[origins + h] for h in range(1, H + 1)])

    def recursive_forecast(model):
        prev1, prev2 = sm[origins].copy(), sm[origins - 1].copy()
        out = np.zeros((len(origins), H))
        for h in range(1, H + 1):
            Xh = pd.DataFrame(np.column_stack([prev1, prev2, met[origins + h]]), columns=COLS)
            p = np.clip(model.predict(Xh), 0, None)
            out[:, h - 1] = p
            prev2, prev1 = prev1, p
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
    res.to_csv(proc / "forecast_results.csv", index=False)
    print("    Forecast results saved.")
    return res


# ─────────────────────────── Step 2: deep layer v1 ───────────────────────────
def run_deep_layer_v1(df, proc):
    df = df.copy()
    for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
        df[c] = df[c].clip(lower=0)
    df = df.dropna()

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
    }
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    splits["random"] = (perm[:cut], perm[cut:], KFold(5, shuffle=True, random_state=42))

    for split_name, (tr_i, te_i, cv) in splits.items():
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        for use_met in [False, True]:
            met_cols = MET if use_met else []
            f1 = ["sm_0_7"] + met_cols
            f2 = ["sm_0_7", "sm_7_28"] + met_cols
            for name in ["LR", "RF", "XGB"]:
                m1 = fit(name, tr[f1], tr["sm_7_28"], cv)
                m2 = fit(name, tr[f2], tr["sm_28_100"], cv)
                p1 = m1.predict(te[f1])
                te2 = te.copy(); te2["sm_7_28"] = p1
                p2 = m2.predict(te2[f2])
                for layer, y, p in [("7-28cm", te["sm_7_28"], p1),
                                    ("28-100cm", te["sm_28_100"], p2)]:
                    r2, rmse, mae = metrics(y, p)
                    results.append(dict(split=split_name, weather=use_met, model=name,
                                        layer=layer, R2=round(r2, 4),
                                        RMSE=round(rmse, 4), MAE=round(mae, 4)))
                print(f"      done: {split_name} | met={use_met} | {name}")

    res = pd.DataFrame(results)
    res.to_csv(proc / "deep_layer_results.csv", index=False)
    print("    Deep layer v1 results saved.")
    return res


# ─────────────────────────── Step 3: deep layer v2 ───────────────────────────
def run_deep_layer_v2(df, proc):
    df = df.copy()
    for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
        df[c] = df[c].clip(lower=0)

    for w in [7, 14, 30]:
        df[f"rain_{w}d"] = df["rain"].rolling(w).sum()
    df["sm_mean7"] = df["sm_0_7"].rolling(7).mean()
    df["sm_mean30"] = df["sm_0_7"].rolling(30).mean()
    df["t_mean7"] = df["t_mean"].rolling(7).mean()
    df = df.dropna()

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

    rows = []
    for fs_name, extra in FEATURE_SETS.items():
        f1 = ["sm_0_7"] + extra
        f2 = ["sm_0_7", "sm_7_28"] + extra
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
            print(f"      done: {fs_name} | {name}")

    res = pd.DataFrame(rows)
    res.to_csv(proc / "deep_layer_v2_results.csv", index=False)
    print("    Deep layer v2 results saved.")
    return res


# ─────────────────────────── Step 4: robustness ───────────────────────────
def run_robustness(df, proc):
    df = df.copy()
    df["sm_0_7"] = df["sm_0_7"].clip(lower=0)
    df = df.dropna(subset=["sm_0_7"])

    sm = df["sm_0_7"].values
    met = df[FC].values.astype(float)
    n = len(df)
    cut = int(n * 0.8)

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
        m[:, 0] += rng.normal(0, t_sd, n)
        m[:, 1] += rng.normal(0, t_sd, n)
        m[:, 2] = np.clip(m[:, 2] + rng.normal(0, rh_sd, n), 0, 100)
        m[:, 3] = m[:, 3] * rng.lognormal(0, rain_sigma, n)
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
    r2.to_csv(proc / "robustness_r2.csv")
    rmse.to_csv(proc / "robustness_rmse.csv")
    print("    Robustness results saved.")
    return r2, rmse


# ─────────────────────────── Step 5: train final models ───────────────────────────
def run_train_final(df, proc, mdl):
    df = df.copy()
    for c in ["sm_0_7", "sm_7_28", "sm_28_100", "rain_1d", "rain_2d"]:
        df[c] = df[c].clip(lower=0)

    # ---- forecast model ----
    FCOLS = ["sm_l1", "sm_l2"] + FC
    fdf = df.dropna(subset=["sm_0_7"]).copy()
    fdf["sm_l1"] = fdf["sm_0_7"].shift(1)
    fdf["sm_l2"] = fdf["sm_0_7"].shift(2)
    fdf = fdf.dropna(subset=["sm_l1", "sm_l2"])
    fc_model = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.1,
                            random_state=42, n_jobs=-1).fit(fdf[FCOLS], fdf["sm_0_7"])
    joblib.dump(fc_model, mdl / "forecast_xgb.joblib")
    print(f"    Forecast model saved ({len(fdf)} days)")

    # ---- deep-layer chain ----
    for w in [7, 14, 30]:
        df[f"rain_{w}d"] = df["rain"].rolling(w).sum()
    df["sm_mean7"] = df["sm_0_7"].rolling(7).mean()
    df["sm_mean30"] = df["sm_0_7"].rolling(30).mean()
    df["t_mean7"] = df["t_mean"].rolling(7).mean()
    ddf = df.dropna()

    F1 = ["sm_0_7"] + MET + LONG
    F2 = ["sm_0_7", "sm_7_28"] + MET + LONG

    grid = {"n_estimators": [200, 400], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]}
    def fit_gs(X, y):
        gs = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), grid,
                          cv=TimeSeriesSplit(5), scoring="r2", n_jobs=-1).fit(X, y)
        return gs.best_estimator_

    m1 = fit_gs(ddf[F1], ddf["sm_7_28"])
    m2 = fit_gs(ddf[F2], ddf["sm_28_100"])
    joblib.dump(m1, mdl / "deep_7_28.joblib")
    joblib.dump(m2, mdl / "deep_28_100.joblib")

    meta = {"forecast_cols": FCOLS, "deep1_cols": F1, "deep2_cols": F2,
            "trained_through": str(df.index[-1].date())}
    (mdl / "meta.json").write_text(json.dumps(meta, indent=2))
    print("    Final models saved.")


# ─────────────────────────── Copy existing Mangaluru results ───────────────────────────
def copy_existing_mangaluru():
    """Copy already-existing Mangaluru results into the site-specific subdirectory."""
    proc = ROOT / "data/processed"
    dst = proc / "mangaluru"
    dst.mkdir(parents=True, exist_ok=True)
    for f in proc.glob("*.csv"):
        shutil.copy2(f, dst / f.name)
    for f in proc.glob("*.png"):
        shutil.copy2(f, dst / f.name)
    # models
    mdl_src = ROOT / "models"
    mdl_dst = ROOT / "models/mangaluru"
    mdl_dst.mkdir(parents=True, exist_ok=True)
    for f in mdl_src.glob("*.joblib"):
        shutil.copy2(f, mdl_dst / f.name)
    for f in mdl_src.glob("*.json"):
        shutil.copy2(f, mdl_dst / f.name)
    print("  Existing Mangaluru results copied to site subdirectory.")


# ─────────────────────────── main ───────────────────────────
def run_site(site):
    print(f"\n{'='*60}")
    print(f"  RUNNING PIPELINE FOR: {site.upper()}")
    print(f"{'='*60}")
    df = load_data(site)
    proc, mdl = make_dirs(site)

    print("  [1/5] Surface forecast …")
    fc_res = run_forecast(df, proc)

    print("  [2/5] Deep layer v1 …")
    dl_res = run_deep_layer_v1(df, proc)

    print("  [3/5] Deep layer v2 …")
    dl2_res = run_deep_layer_v2(df, proc)

    print("  [4/5] Robustness analysis …")
    rob_r2, rob_rmse = run_robustness(df, proc)

    print("  [5/5] Train final models …")
    run_train_final(df, proc, mdl)

    print(f"  ✓ Pipeline complete for {site}")
    return fc_res, dl_res, dl2_res, rob_r2, rob_rmse


if __name__ == "__main__":
    # Copy existing Mangaluru results first
    copy_existing_mangaluru()

    for site in SITES:
        run_site(site)

    print("\n✓ All sites complete.")
