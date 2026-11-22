"""
Fetch ERA5-Land hourly data from Open-Meteo Archive API for three sites,
aggregate to daily, and save to data/raw/<site>_daily.csv.
"""
import time, requests, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SITES = {
    "mangaluru":  (12.9141, 74.8560),
    "mysore":     (12.3537, 76.4054),
    "shivamogga": (13.9719, 75.4704),
}

START, END = "2023-01-01", "2025-12-31"

HOURLY_VARS = [
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m", "surface_pressure",
]

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_site(name, lat, lon, retries=5, backoff=10):
    """Fetch hourly data, aggregate to daily, return DataFrame."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(1, retries + 1):
        try:
            print(f"  [{name}] attempt {attempt} …")
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            break
        except Exception as e:
            print(f"  [{name}] error: {e}")
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)

    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")

    d = pd.DataFrame({
        "sm_0_7":    h["soil_moisture_0_to_7cm"].resample("D").mean(),
        "sm_7_28":   h["soil_moisture_7_to_28cm"].resample("D").mean(),
        "sm_28_100": h["soil_moisture_28_to_100cm"].resample("D").mean(),
        "t_mean":    h["temperature_2m"].resample("D").mean(),
        "t_max":     h["temperature_2m"].resample("D").max(),
        "t_min":     h["temperature_2m"].resample("D").min(),
        "rh_mean":   h["relative_humidity_2m"].resample("D").mean(),
        "rh_max":    h["relative_humidity_2m"].resample("D").max(),
        "rh_min":    h["relative_humidity_2m"].resample("D").min(),
        "pressure":  h["surface_pressure"].resample("D").mean(),
        "wind_mean": h["wind_speed_10m"].resample("D").mean(),
        "wind_max":  h["wind_speed_10m"].resample("D").max(),
        "rain":      h["precipitation"].resample("D").sum(),
    })
    d["t_range"] = d["t_max"] - d["t_min"]
    d["rain_1d"]  = d["rain"].shift(1)
    d["rain_2d"]  = d["rain"].rolling(3).sum() - d["rain"]
    return d


if __name__ == "__main__":
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in SITES.items():
        print(f"\nFetching {name} ({lat}, {lon}) …")
        d = fetch_site(name, lat, lon)
        out = ROOT / f"data/raw/{name}_daily.csv"
        d.to_csv(out)
        print(f"  Saved {out}  ({len(d)} days)")
        print(d.describe().T[["count", "mean", "min", "max"]])
        time.sleep(2)  # polite pause between API calls
    print("\n✓ All sites fetched.")
