import sys, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_data import _get, _daily_weather, _daily_sm, W_VARS, SM_VARS, LAT, LON, TZ, ROOT

H = 7
FC = ["t_max", "t_min", "rh_mean", "rain"]
BASE = ["sm_l1", "sm_l2"] + FC
COLS = BASE + ["lead"]
TEST_START = pd.Timestamp("2026-01-01")
today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
end = (today - pd.Timedelta(days=1)).date().isoformat()
PERIODS = [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", end)]

def retry(fn, *args):
    for _ in range(3):
        try:
            return fn(*args)
        except Exception as e:
            print("retrying after error:", e); time.sleep(5)
    raise SystemExit("Download failed three times; rerun later.")

def get_obs(a, b):
    h = _get("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": LAT, "longitude": LON, "start_date": a, "end_date": b,
        "hourly": ",".join(W_VARS + SM_VARS), "timezone": TZ})
    return pd.concat([_daily_weather(h), _daily_sm(h)], axis=1)

def get_prev(a, b):
    cols = [f"{v}_previous_day{k}" for v in ["temperature_2m", "relative_humidity_2m", "precipitation"]
            for k in range(1, H + 1)]
    r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", timeout=120, params={
        "latitude": LAT, "longitude": LON, "start_date": a, "end_date": b,
        "hourly": ",".join(cols), "models": "ecmwf_ifs025", "timezone": TZ})
    r.raise_for_status()
    x = pd.DataFrame(r.json()["hourly"]); x["time"] = pd.to_datetime(x["time"])
    return x.set_index("time")

obs = pd.concat([retry(get_obs, a, b) for a, b in PERIODS]).asfreq("D")
hp = pd.concat([retry(get_prev, a, b) for a, b in PERIODS])

def daily(s, how):
    g = s.resample("D"); d = getattr(g, how)(); d[g.count() < 23] = np.nan
    return d

lead = {k: pd.DataFrame({
    "t_max": daily(hp[f"temperature_2m_previous_day{k}"], "max"),
    "t_min": daily(hp[f"temperature_2m_previous_day{k}"], "min"),
    "rh_mean": daily(hp[f"relative_humidity_2m_previous_day{k}"], "mean"),
    "rain": daily(hp[f"precipitation_previous_day{k}"], "sum")}) for k in range(1, H + 1)}

dates = obs.index; n = len(dates)
sm = obs["sm_0_7"].clip(lower=0).values
W = {k: lead[k].reindex(dates)[FC].values for k in range(1, H + 1)}
cut = int(np.searchsorted(dates, TEST_START))

# ---- training rows: observed lags + lead-k archived forecast weather for the target day ----
Xs, ys, t = [], [], np.arange(2, cut)
for k in range(1, H + 1):
    X = np.column_stack([sm[t - 1], sm[t - 2], W[k][t], np.full(len(t), k)])
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(sm[t])
    Xs.append(X[keep]); ys.append(sm[t][keep])
Xtr = pd.DataFrame(np.vstack(Xs), columns=COLS); ytr = np.concatenate(ys)
print(f"Training rows: {len(Xtr)} (2024-2025 archived forecasts, leads 1-{H})")
new_model = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                         random_state=42, n_jobs=-1).fit(Xtr, ytr)
era_model = joblib.load(ROOT / "models/forecast_xgb.joblib")   # deployed model, trained on ERA5 weather

# ---- 2026 origins with complete observations and complete forecasts ----
org = np.array([i for i in range(2, n - H)
                if dates[i] >= TEST_START and not np.isnan(sm[i - 2:i + H + 1]).any()
                and all(not np.isnan(W[k][i + k]).any() for k in range(1, H + 1))])
print(f"{len(org)} origins, {dates[org[0]].date()} to {dates[org[-1]].date()}")
act = np.column_stack([sm[org + k] for k in range(1, H + 1)])

def run(model, with_lead):
    p1, p2 = sm[org].copy(), sm[org - 1].copy()
    out = np.zeros((len(org), H))
    for k in range(1, H + 1):
        parts = [p1, p2, W[k][org + k]] + ([np.full(len(org), k)] if with_lead else [])
        X = pd.DataFrame(np.column_stack(parts), columns=COLS if with_lead else BASE)
        p = np.clip(model.predict(X), 0, None)
        out[:, k - 1] = p; p2, p1 = p1, p
    return out

preds = {"Persistence": np.column_stack([sm[org]] * H),
         "Trained on ERA5": run(era_model, False),
         "Trained on forecasts": run(new_model, True)}
idx = pd.Index(range(1, H + 1), name="day")
r2 = pd.DataFrame({k: [r2_score(act[:, j], v[:, j]) for j in range(H)] for k, v in preds.items()}, index=idx).round(4)
rm = pd.DataFrame({k: [np.sqrt(mean_squared_error(act[:, j], v[:, j])) for j in range(H)] for k, v in preds.items()}, index=idx).round(4)
print("\nR2:\n", r2.to_string()); print("\nRMSE:\n", rm.to_string())

mon = np.isin(dates[org].month, [6, 7, 8, 9])
for name, msk in [("dry/pre-monsoon", ~mon), ("monsoon Jun-Sep", mon)]:
    t_ = pd.DataFrame({k: [np.sqrt(mean_squared_error(act[msk, j], v[msk, j])) for j in range(H)]
                       for k, v in preds.items()}, index=idx).round(4)
    print(f"\nRMSE, {name} (n={int(msk.sum())}):\n", t_.iloc[[0, 2, 6]].to_string())

r2.to_csv(ROOT / "data/processed/forecast_trained_on_forecasts_r2.csv")
rm.to_csv(ROOT / "data/processed/forecast_trained_on_forecasts_rmse.csv")
# ---------- deployment model: same recipe, trained on ALL rows including 2026 ----------
Xs, ys, t = [], [], np.arange(2, n)
for k in range(1, H + 1):
    X = np.column_stack([sm[t - 1], sm[t - 2], W[k][t], np.full(len(t), k)])
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(sm[t])
    Xs.append(X[keep]); ys.append(sm[t][keep])
final = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                     random_state=42, n_jobs=-1).fit(
    pd.DataFrame(np.vstack(Xs), columns=COLS), np.concatenate(ys))
joblib.dump(final, ROOT / "models/forecast_xgb_fc.joblib")
print("Saved models/forecast_xgb_fc.joblib, trained through", dates[n - 1].date())
# ---------- deployment model: same recipe, trained on ALL rows including 2026 ----------
Xs, ys, t = [], [], np.arange(2, n)
for k in range(1, H + 1):
    X = np.column_stack([sm[t - 1], sm[t - 2], W[k][t], np.full(len(t), k)])
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(sm[t])
    Xs.append(X[keep]); ys.append(sm[t][keep])
final = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                     random_state=42, n_jobs=-1).fit(
    pd.DataFrame(np.vstack(Xs), columns=COLS), np.concatenate(ys))
joblib.dump(final, ROOT / "models/forecast_xgb_fc.joblib")
print("Saved models/forecast_xgb_fc.joblib, trained through", dates[n - 1].date())
