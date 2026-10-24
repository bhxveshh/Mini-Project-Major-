import json
import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAT, LON = 12.9141, 74.8560          # same site as the training data
TZ = "Asia/Kolkata"

W_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
          "wind_speed_10m", "precipitation"]
SM_VARS = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]
WEATHER_COLS = ["t_mean", "t_max", "t_min", "rh_mean", "rh_max", "rh_min",
                "pressure", "wind_mean", "wind_max", "rain"]

def _get(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")

def _daily_weather(h):
    g = h.resample("D")
    d = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(), "t_max": g["temperature_2m"].max(),
        "t_min": g["temperature_2m"].min(),
        "rh_mean": g["relative_humidity_2m"].mean(), "rh_max": g["relative_humidity_2m"].max(),
        "rh_min": g["relative_humidity_2m"].min(),
        "pressure": g["surface_pressure"].mean(),
        "wind_mean": g["wind_speed_10m"].mean(), "wind_max": g["wind_speed_10m"].max(),
        "rain": g["precipitation"].sum(min_count=1),
    })
    d.loc[~(g["temperature_2m"].count() >= 23)] = np.nan   # drop incomplete days
    return d

def _daily_sm(h):
    g = h.resample("D")
    d = pd.DataFrame({"sm_0_7": g["soil_moisture_0_to_7cm"].mean(),
                      "era_7_28": g["soil_moisture_7_to_28cm"].mean(),
                      "era_28_100": g["soil_moisture_28_to_100cm"].mean()})
    d.loc[~(g["soil_moisture_0_to_7cm"].count() >= 23)] = np.nan
    return d

def fetch_observed(today):
    """ERA5-Land from the Open-Meteo archive; find the latest complete day."""
    start = (today - pd.Timedelta(days=150)).date().isoformat()
    last_err = None
    for back in [1, 3, 7, 10]:                     # retry if the end date is not yet available
        try:
            h = _get("https://archive-api.open-meteo.com/v1/archive", {
                "latitude": LAT, "longitude": LON, "start_date": start,
                "end_date": (today - pd.Timedelta(days=back)).date().isoformat(),
                "hourly": ",".join(W_VARS + SM_VARS), "timezone": TZ})
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Archive download failed: {last_err}")
    obs = pd.concat([_daily_weather(h), _daily_sm(h)], axis=1)
    return obs.dropna(subset=["sm_0_7"] + WEATHER_COLS)

def fetch_forecast_weather():
    h = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "hourly": ",".join(W_VARS),
        "past_days": 30, "forecast_days": 8, "timezone": TZ})
    return _daily_weather(h)

def build_live_table(today=None):
    today = today or pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    obs = fetch_observed(today)
    last_obs = obs.index[-1]
    fw = fetch_forecast_weather()
    fut = fw[fw.index > last_obs].dropna(subset=WEATHER_COLS)
    daily = pd.concat([obs, fut]).asfreq("D")
    daily[WEATHER_COLS] = daily[WEATHER_COLS].interpolate(limit=2, limit_area="inside")
    if daily[WEATHER_COLS].isna().any().any():
        raise RuntimeError("Missing weather values between the last ERA5 day and the forecast.")

    meta = json.loads((ROOT / "models/meta.json").read_text())
    fc_model = joblib.load(ROOT / "models/forecast_xgb_fc.joblib")
    m1 = joblib.load(ROOT / "models/deep_7_28.joblib")
    m2 = joblib.load(ROOT / "models/deep_28_100.joblib")

    # ---- recursive surface forecast through the gap and ahead ----
    sm = daily["sm_0_7"].copy()
    one = pd.Timedelta(days=1)
    for d in daily.index[daily.index > last_obs]:
        lead = int(np.clip((d - today).days, 1, 7))   # days ahead of today; gap days use 1
        row = {"sm_l1": sm.loc[d - one], "sm_l2": sm.loc[d - 2 * one],
               "t_max": daily.loc[d, "t_max"], "t_min": daily.loc[d, "t_min"],
               "rh_mean": daily.loc[d, "rh_mean"], "rain": daily.loc[d, "rain"],
               "lead": lead}
        X = pd.DataFrame([row])[["sm_l1", "sm_l2", "t_max", "t_min", "rh_mean", "rain", "lead"]]
        sm.loc[d] = max(float(fc_model.predict(X)[0]), 0.0)

    # ---- deep-layer chain on observed + forecast surface series ----
    f = daily.copy()
    f["sm_0_7"] = sm
    f["t_range"] = f["t_max"] - f["t_min"]
    f["rain_1d"] = f["rain"].shift(1)
    f["rain_2d"] = f["rain"].rolling(3).sum() - f["rain"]
    for w in [7, 14, 30]:
        f[f"rain_{w}d"] = f["rain"].rolling(w).sum()
    f["sm_mean7"] = f["sm_0_7"].rolling(7).mean()
    f["sm_mean30"] = f["sm_0_7"].rolling(30).mean()
    f["t_mean7"] = f["t_mean"].rolling(7).mean()
    f = f.dropna(subset=meta["deep1_cols"][1:])
    f["est_7_28"] = m1.predict(f[meta["deep1_cols"]])
    g = f.copy(); g["sm_7_28"] = f["est_7_28"]                 # feed the ESTIMATE forward
    f["est_28_100"] = m2.predict(g[meta["deep2_cols"]])

    out = f[["sm_0_7", "est_7_28", "est_28_100", "rain"]].copy()
    out["era_7_28"] = daily["era_7_28"]
    out["era_28_100"] = daily["era_28_100"]
    out["is_obs"] = out.index <= last_obs
    out["horizon"] = np.maximum((out.index - last_obs).days, 0)
    out = out[out.index >= today - pd.Timedelta(days=30)]
    out = out[out.index <= today + pd.Timedelta(days=7)]
    return out, last_obs, today

if __name__ == "__main__":
    t, last_obs, today = build_live_table()
    print("today:", today.date(), "| last ERA5 day:", last_obs.date(),
          "| gap:", (today - last_obs).days, "days")
    print(t.tail(14).round(3).to_string())
