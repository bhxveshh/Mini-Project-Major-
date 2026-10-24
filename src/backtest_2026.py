import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_data import _get, _daily_weather, _daily_sm, W_VARS, SM_VARS, WEATHER_COLS, LAT, LON, TZ, ROOT

H = 7
FC = ["t_max", "t_min", "rh_mean", "rain"]
meta = json.loads((ROOT / "models/meta.json").read_text())
fc_model = joblib.load(ROOT / "models/forecast_xgb.joblib")
m1 = joblib.load(ROOT / "models/deep_7_28.joblib")
m2 = joblib.load(ROOT / "models/deep_28_100.joblib")

# ---------- 1. download ERA5-Land from Nov 2025 to the latest day ----------
today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
h = None
for back in [1, 3, 7]:
    try:
        h = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": LAT, "longitude": LON, "start_date": "2025-11-01",
            "end_date": (today - pd.Timedelta(days=back)).date().isoformat(),
            "hourly": ",".join(W_VARS + SM_VARS), "timezone": TZ})
        break
    except Exception as e:
        print("archive attempt failed:", e)
if h is None:
    raise SystemExit("Could not download the archive.")

obs = pd.concat([_daily_weather(h), _daily_sm(h)], axis=1)
obs = obs.dropna(subset=["sm_0_7"] + WEATHER_COLS).asfreq("D")
obs["sm_0_7"] = obs["sm_0_7"].clip(lower=0)
print(f"ERA5 days: {obs.index[0].date()} to {obs.index[-1].date()}")

# ---------- 2. deep-layer chain on 2026 (inputs are OBSERVED surface values) ----------
f = obs.copy()
f["t_range"] = f["t_max"] - f["t_min"]
f["rain_1d"] = f["rain"].shift(1).clip(lower=0)
f["rain_2d"] = (f["rain"].rolling(3).sum() - f["rain"]).clip(lower=0)
for w in [7, 14, 30]:
    f[f"rain_{w}d"] = f["rain"].rolling(w).sum()
f["sm_mean7"] = f["sm_0_7"].rolling(7).mean()
f["sm_mean30"] = f["sm_0_7"].rolling(30).mean()
f["t_mean7"] = f["t_mean"].rolling(7).mean()
f = f[f.index >= "2026-01-01"].dropna(subset=meta["deep1_cols"] + ["era_7_28", "era_28_100"])

est1 = m1.predict(f[meta["deep1_cols"]])
g = f.copy(); g["sm_7_28"] = est1                    # chain: feed the ESTIMATE forward
est2 = m2.predict(g[meta["deep2_cols"]])

def summarize(y, p, benchmark):
    d = pd.DataFrame({"y": y.values, "p": p}, index=y.index)
    groups = [("ALL 2026", d), ("last 14 days", d.tail(14))]
    groups += [(f"2026-{m:02d}", gg) for m, gg in d.groupby(d.index.month)]
    out = pd.DataFrame({name: {"n": len(gg),
                               "R2": r2_score(gg.y, gg.p) if len(gg) > 2 else np.nan,
                               "RMSE": np.sqrt(mean_squared_error(gg.y, gg.p)),
                               "bias": (gg.p - gg.y).mean()} for name, gg in groups}).T.round(4)
    print(out.to_string())
    print(f"(chronological-test RMSE from earlier evaluation: {benchmark})")
    return out

print("\n=== Deep layer 7-28 cm vs ERA5 ===")
s1 = summarize(f["era_7_28"], est1, 0.0142)
print("\n=== Deep layer 28-100 cm vs ERA5 ===")
s2 = summarize(f["era_28_100"], est2, 0.0395)
s1.to_csv(ROOT / "data/processed/backtest2026_deep_7_28.csv")
s2.to_csv(ROOT / "data/processed/backtest2026_deep_28_100.csv")

# ---------- 3. 7-day recursive surface forecast: oracle vs real archived forecasts ----------
sm = obs["sm_0_7"].values
met = obs[FC].values.astype(float)
dates = obs.index
n = len(obs)
origins = np.array([i for i in range(2, n - H)
                    if dates[i] >= pd.Timestamp("2026-01-01")
                    and not np.isnan(sm[i - 2:i + H + 1]).any()
                    and not np.isnan(met[i + 1:i + H + 1]).any()])

def fetch_previous_runs(model):
    hourly = []
    for var in ["temperature_2m", "relative_humidity_2m", "precipitation"]:
        hourly += [f"{var}_previous_day{k}" for k in range(1, H + 1)]
    params = {"latitude": LAT, "longitude": LON, "start_date": "2026-01-01",
              "end_date": (today - pd.Timedelta(days=1)).date().isoformat(),
              "hourly": ",".join(hourly), "timezone": TZ}
    if model:
        params["models"] = model
    r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params=params, timeout=120)
    if r.status_code != 200:
        print(f"previous-runs model={model}: HTTP {r.status_code}: {r.text[:200]}")
        return None
    x = pd.DataFrame(r.json()["hourly"]); x["time"] = pd.to_datetime(x["time"])
    return x.set_index("time")

hp = None
for model in ["ecmwf_ifs025", None]:
    hp = fetch_previous_runs(model)
    if hp is not None:
        break

def daily(series, how):
    gp = series.resample("D")
    d = getattr(gp, how)()
    d[gp.count() < 23] = np.nan                      # drop incomplete days
    return d

def run(get_met, orig):
    p1, p2 = sm[orig].copy(), sm[orig - 1].copy()
    out = np.zeros((len(orig), H))
    for k in range(1, H + 1):
        X = pd.DataFrame(np.column_stack([p1, p2, get_met(k)]), columns=meta["forecast_cols"])
        p = np.clip(fc_model.predict(X), 0, None)
        out[:, k - 1] = p
        p2, p1 = p1, p
    return out

real = None
if hp is not None:
    lead = {k: pd.DataFrame({
        "t_max": daily(hp[f"temperature_2m_previous_day{k}"], "max"),
        "t_min": daily(hp[f"temperature_2m_previous_day{k}"], "min"),
        "rh_mean": daily(hp[f"relative_humidity_2m_previous_day{k}"], "mean"),
        "rain": daily(hp[f"precipitation_previous_day{k}"], "sum")}) for k in range(1, H + 1)}
    r = {k: lead[k].reindex(dates[origins + k])[FC].values for k in range(1, H + 1)}
    ok = np.all([~np.isnan(r[k]).any(axis=1) for k in r], axis=0)
    origins = origins[ok]
    real = {k: r[k][ok] for k in r}

print(f"\n=== 7-day forecast backtest: {len(origins)} origins, "
      f"{dates[origins[0]].date()} to {dates[origins[-1]].date()} ===")
actual = np.column_stack([sm[origins + k] for k in range(1, H + 1)])
preds = {"Oracle (ERA5 weather)": run(lambda k: met[origins + k], origins),
         "Persistence": np.column_stack([sm[origins]] * H)}
if real is not None:
    preds["Real forecast"] = run(lambda k: real[k], origins)
idx = pd.Index(range(1, H + 1), name="day")
r2 = pd.DataFrame({k: [r2_score(actual[:, j], v[:, j]) for j in range(H)] for k, v in preds.items()}, index=idx).round(4)
rm = pd.DataFrame({k: [np.sqrt(mean_squared_error(actual[:, j], v[:, j])) for j in range(H)] for k, v in preds.items()}, index=idx).round(4)
print("R2:\n", r2.to_string()); print("\nRMSE:\n", rm.to_string())
r2.to_csv(ROOT / "data/processed/backtest2026_forecast_r2.csv")
rm.to_csv(ROOT / "data/processed/backtest2026_forecast_rmse.csv")

# ---------- 4. where does the real-forecast model lose to persistence? ----------
org_dates = dates[origins]
season = np.where(np.isin(org_dates.month, [6, 7, 8, 9]), "monsoon Jun-Sep", "dry/pre-monsoon")
short = {"Oracle (ERA5 weather)": "oracle", "Persistence": "persist", "Real forecast": "real"}
rows = []
for sname in ["dry/pre-monsoon", "monsoon Jun-Sep"]:
    msk = season == sname
    for j in [0, 2, 6]:                                  # forecast day 1, 3, 7
        row = {"season": sname, "day": j + 1, "n": int(msk.sum())}
        for name, v in preds.items():
            row["RMSE " + short[name]] = round(float(np.sqrt(mean_squared_error(actual[msk, j], v[msk, j]))), 4)
        rows.append(row)
print("\nRMSE by season:\n", pd.DataFrame(rows).to_string(index=False))

if real is not None:
    print()
    for sname in ["dry/pre-monsoon", "monsoon Jun-Sep"]:
        msk = season == sname
        for k in [1, 7]:
            f_, o_ = real[k][msk, 3], met[origins[msk] + k, 3]
            print(f"{sname} lead {k}d rain: corr {np.corrcoef(f_, o_)[0, 1]:.2f}, "
                  f"forecast mean {f_.mean():.1f} mm vs ERA5 {o_.mean():.1f} mm")

# ---------- 5. do heavy-rain days explain the deep-layer errors? ----------
heavy = f["rain"] >= 25
after = heavy.shift(1, fill_value=False)
e = pd.DataFrame({"e7": est1 - f["era_7_28"].values, "e28": est2 - f["era_28_100"].values}, index=f.index)
print(f"\nHeavy-rain days (>=25 mm) in 2026: {int(heavy.sum())}")
for lab, msk in [("heavy-rain day", heavy), ("day after", after), ("all other days", ~(heavy | after))]:
    print(f"  {lab:15s} n={int(msk.sum()):3d} | 7-28: bias {e.e7[msk].mean():+.3f}, RMSE {np.sqrt((e.e7[msk]**2).mean()):.3f}"
          f" | 28-100: bias {e.e28[msk].mean():+.3f}, RMSE {np.sqrt((e.e28[msk]**2).mean()):.3f}")
# ---------- 6. does the 28-100 cm error depend on how wet the layer is? ----------
e28 = pd.Series(est2 - f["era_28_100"].values, index=f.index)
lvl = pd.qcut(f["era_28_100"], 3, labels=["driest third", "middle third", "wettest third"])
print("\n28-100 cm error by ERA5 wetness level (2026):")
for lab in lvl.cat.categories:
    sel = lvl == lab
    print(f"  {lab:14s} ERA5 {f['era_28_100'][sel].min():.3f}-{f['era_28_100'][sel].max():.3f} | "
          f"n={int(sel.sum()):3d} | bias {e28[sel].mean():+.3f}, RMSE {np.sqrt((e28[sel]**2).mean()):.3f}")
print(f"Day-to-day change, 28-100 cm: model std {pd.Series(est2).diff().std():.4f} "
      f"vs ERA5 std {f['era_28_100'].diff().std():.4f}")