import ee
import pandas as pd
from datetime import date, timedelta

PROJECT = "soil-moisture-mini"
LAT, LON = 12.9141, 74.8560          # same as your other scripts
ee.Initialize(project=PROJECT)
aoi = ee.Geometry.Point([LON, LAT])

s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(aoi)
      .filterDate("2026-06-01", (date.today() + timedelta(days=1)).isoformat())
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

times = s1.aggregate_array("system:time_start").getInfo()
orbits = s1.aggregate_array("relativeOrbitNumber_start").getInfo()
dirs = s1.aggregate_array("orbitProperties_pass").getInfo()
df = pd.DataFrame({"date": pd.to_datetime(times, unit="ms").normalize(),
                   "rel_orbit": orbits, "pass": dirs}).sort_values("date")
print("Recent Sentinel-1 passes over the site:")
print(df.tail(10).to_string(index=False))

if len(df):
    last = df["date"].iloc[-1].date()
    print(f"\nProjected next passes (assuming a 12-day repeat; VERIFY before planning):")
    d = last
    while d < date.today() + timedelta(days=60):
        d += timedelta(days=12)
        if d >= date.today():
            print("  ", d.isoformat(), (d.strftime("%a")))
