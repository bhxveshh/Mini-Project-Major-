import ee
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = "soil-moisture-mini"      # your Google Cloud project registered for Earth Engine
LAT, LON = 12.9141, 74.8560        # CHANGE to a farmland/paddy site
ee.Initialize(project=PROJECT)

ROOT = Path(__file__).resolve().parent.parent
aoi = ee.Geometry.Point([LON, LAT]).buffer(150)     # ~150 m radius; averages out speckle

s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(aoi)
      .filterDate("2023-01-01", "2025-12-31")
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
print("Sentinel-1 scenes found:", s1.size().getInfo())

def stats(img):
    d = img.select(["VV", "VH", "angle"]).reduceRegion(ee.Reducer.mean(), aoi, 10)
    return ee.Feature(None, d).set({
        "time": img.date().format("YYYY-MM-dd"),
        "orbit": img.get("orbitProperties_pass"),
        "rel_orbit": img.get("relativeOrbitNumber_start"),
    })

feats = s1.map(stats).getInfo()["features"]
s = pd.DataFrame([f["properties"] for f in feats])
s["time"] = pd.to_datetime(s["time"])
s = s.dropna(subset=["VV", "VH"]).sort_values("time")
s.to_csv(ROOT / "data/raw/s1_site.csv", index=False)
print(s.head(), "\n", len(s), "usable scenes")

# ---- does backscatter track ERA5 surface moisture? (compare within one relative orbit) ----
d = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
d["sm_0_7"] = d["sm_0_7"].clip(lower=0)
m = s.merge(d[["sm_0_7", "rain"]], left_on="time", right_index=True)
m["VV_VH"] = m["VV"] - m["VH"]                      # ratio in dB

print("\nCorrelation with ERA5 surface moisture, by relative orbit:")
for ro, g in m.groupby("rel_orbit"):
    if len(g) >= 15:
        print(f"orbit {ro} ({g['orbit'].iloc[0]}, n={len(g)}): "
              f"VV {g['VV'].corr(g['sm_0_7']):+.2f} | VH {g['VH'].corr(g['sm_0_7']):+.2f} "
              f"| VV-VH {g['VV_VH'].corr(g['sm_0_7']):+.2f}")
