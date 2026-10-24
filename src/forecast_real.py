
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

LAT, LON = 12.9141, 74.8560
H = 7
FC = ["t_max", "t_min", "rh_mean", "rain"]

# ---------- 1. download archived forecasts at lead times 1..7 days ----------
def fetch_previous_runs(model):
    hourly = []
    for var in ["temperature_2m", "relative_humidity_2m", "precipitation"]:
        hourly += [f"{var}_previous_day{k}" for k in range(1, H + 1)]
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": "2024-01-01", "end_date": "2025-12-31",
        "hourly": ",".join(hourly), "timezone": "Asia/Kolkata",
    }
    if model:
        params["models"] = model
    r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast",
                     params=params, timeout=120)
    if r.status_code != 200:
        print(f"model={model}: HTTP {r.status_code}: {r.text[:300]}")
        return None
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")

hourly = None
for model in ["ecmwf_ifs025", None]:
    hourly = fetch_previous_runs(model)
    if hourly is not None:
        print("Using model:", model or "default")
        break
if hourly is None:
    raise SystemExit("Could not download previous-runs data; paste the error above.")

# daily features for each lead time k: lead[k] is a DataFrame indexed by date
lead = {}
for k in range(1, H + 1):
    t = hourly[f"temperature_2m_previous_day{k}"]
    rh = hourly[f"relative_humidity_2m_previous_day{k}"]
    pr = hourly[f"precipitation_previous_day{k}"]
    lead[k] = pd.DataFrame({
        "t_max": t.resample("D").max(), "t_min": t.resample("D").min(),
        "rh_mean": rh.resample("D").mean(), "rain": pr.resample("D").sum(),
    })

# ---------- 2. same model as before, trained on observed (ERA5) weather ----------
df = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
df["sm_0_7"] = df["sm_0_7"].clip(lower=0)
df = df.dropna(subset=["sm_0_7"])
sm = df["sm_0_7"].values
met = df[FC].values.astype(float)
n = len(df); cut = int(n * 0.8)
dates = df.index

t_idx = np.arange(2, cut)
model = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.1,
                     random_state=42, n_jobs=-1).fit(
    np.column_stack([sm[t_idx - 1], sm[t_idx - 2], met[t_idx]]), sm[t_idx])

# ---------- 3. evaluate on test origins where real forecasts exist ----------
origins = np.arange(cut - 1, n - H)

def real_met(h):   # real archived forecast for target day (origin + h) at lead h
    d = lead[h].reindex(dates[origins + h])
    return d[FC].values

real = {h: real_met(h) for h in range(1, H + 1)}
ok = np.all([~np.isnan(real[h]).any(axis=1) for h in real], axis=0)
print(f"Origins with complete forecasts: {ok.sum()} of {len(origins)}")
origins = origins[ok]
real = {h: real[h][ok] for h in real}
actual = np.column_stack([sm[origins + h] for h in range(1, H + 1)])

def run(get_met):
    prev1, prev2 = sm[origins].copy(), sm[origins - 1].copy()
    out = np.zeros((len(origins), H))
    for h in range(1, H + 1):
        X = np.column_stack([prev1, prev2, get_met(h)])
        p = np.clip(model.predict(X), 0, None)
        out[:, h - 1] = p
        prev2, prev1 = prev1, p
    return out

preds = {
    "Oracle (ERA5)": run(lambda h: met[origins + h]),
    "Real forecast": run(lambda h: real[h]),
    "Persistence": np.column_stack([sm[origins]] * H),
}

r2 = pd.DataFrame({k: [r2_score(actual[:, h], v[:, h]) for h in range(H)]
                   for k, v in preds.items()}, index=pd.Index(range(1, H + 1), name="day")).round(4)
rmse = pd.DataFrame({k: [np.sqrt(mean_squared_error(actual[:, h], v[:, h])) for h in range(H)]
                     for k, v in preds.items()}, index=r2.index).round(4)
print("\nR2:\n", r2.to_string())
print("\nRMSE:\n", rmse.to_string())

# how good are the real rain forecasts themselves?
for h in [1, 3, 7]:
    obs_rain = met[origins + h, 3]
    print(f"\nLead {h}d rain: corr(forecast, ERA5) = "
          f"{np.corrcoef(real[h][:, 3], obs_rain)[0, 1]:.2f}, "
          f"mean forecast {real[h][:, 3].mean():.1f} mm vs ERA5 {obs_rain.mean():.1f} mm")

r2.to_csv(ROOT / "data/processed/real_forecast_r2.csv")
rmse.to_csv(ROOT / "data/processed/real_forecast_rmse.csv")
