import requests, pandas as pd

LAT, LON = 12.9141, 74.8560   # Mangaluru centre; change to your exact site later
START, END = "2023-01-01", "2025-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT, "longitude": LON,
    "start_date": START, "end_date": END,
    "hourly": ",".join([
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m", "surface_pressure",
    ]),
    "timezone": "Asia/Kolkata",
}
r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
h = pd.DataFrame(r.json()["hourly"])
h["time"] = pd.to_datetime(h["time"])
h = h.set_index("time")

# Aggregate to daily, mirroring the paper's feature set (Table 1)
d = pd.DataFrame({
    "sm_0_7":    h["soil_moisture_0_to_7cm"].resample("D").mean(),
    "sm_7_28":   h["soil_moisture_7_to_28cm"].resample("D").mean(),
    "sm_28_100": h["soil_moisture_28_to_100cm"].resample("D").mean(),
    "t_mean": h["temperature_2m"].resample("D").mean(),
    "t_max":  h["temperature_2m"].resample("D").max(),
    "t_min":  h["temperature_2m"].resample("D").min(),
    "rh_mean": h["relative_humidity_2m"].resample("D").mean(),
    "rh_max":  h["relative_humidity_2m"].resample("D").max(),
    "rh_min":  h["relative_humidity_2m"].resample("D").min(),
    "pressure": h["surface_pressure"].resample("D").mean(),
    "wind_mean": h["wind_speed_10m"].resample("D").mean(),
    "wind_max":  h["wind_speed_10m"].resample("D").max(),
    "rain": h["precipitation"].resample("D").sum(),
})
d["t_range"] = d["t_max"] - d["t_min"]
d["rain_1d"] = d["rain"].shift(1)                   # previous-day rain
d["rain_2d"] = d["rain"].rolling(3).sum() - d["rain"]  # cumulative of previous 2 days

d.to_csv("data/raw/mangaluru_daily.csv")
print(d.describe().T[["count", "mean", "min", "max"]])
