import ee
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
PROJECT = "soil-moisture-mini"
LAT, LON = 12.9141, 74.8560          # <-- same as s1_feasibility.py
ROOT = Path(__file__).resolve().parent.parent

ee.Initialize(project=PROJECT)
aoi = ee.Geometry.Point([LON, LAT]).buffer(150)

# ---------- Sentinel-2 NDVI, cloud-masked with the scene classification ----------
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi).filterDate("2023-01-01", "2025-12-31"))

def ndvi_feature(img):
    scl = img.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5))                 # vegetation or bare soil only
    ndvi = img.normalizedDifference(["B8", "B4"]).updateMask(clear).rename("NDVI")
    d = ndvi.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True), aoi, 10)
    return ee.Feature(None, d).set("time", img.date().format("YYYY-MM-dd"))

feats = s2.map(ndvi_feature).getInfo()["features"]
n2 = pd.DataFrame([f["properties"] for f in feats])
n2["time"] = pd.to_datetime(n2["time"])
n2 = n2.dropna(subset=["NDVI_mean"])
n2 = n2[n2["NDVI_count"] >= 350]                    # need roughly half the pixels clear
n2 = n2.groupby("time", as_index=False)["NDVI_mean"].mean().sort_values("time")
n2.to_csv(ROOT / "data/raw/s2_ndvi_site.csv", index=False)
print(f"Usable Sentinel-2 NDVI dates: {len(n2)}")
print(n2["time"].dt.month.value_counts().sort_index().to_string())

# ---------- merge with Sentinel-1 and ERA5 ----------
s1 = pd.read_csv(ROOT / "data/raw/s1_site.csv", parse_dates=["time"]).sort_values("time")
era = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
era["sm_0_7"] = era["sm_0_7"].clip(lower=0)

m = s1.merge(era[["sm_0_7"]], left_on="time", right_index=True)
m = pd.merge_asof(m, n2, on="time", direction="nearest", tolerance=pd.Timedelta("20D"))
lin = lambda db: 10 ** (db / 10)
m["RVI"] = 4 * lin(m["VH"]) / (lin(m["VV"]) + lin(m["VH"]))
m["VV_VH"] = m["VV"] - m["VH"]
m["month"] = m["time"].dt.month
m["year"] = m["time"].dt.year
print(f"\nS1 scenes with an NDVI within +/-20 days: {m['NDVI_mean'].notna().sum()} of {len(m)}")

# ---------- leave-one-year-out CV ----------
def loyo(df, feats, make, kind="model"):
    pred = pd.Series(index=df.index, dtype=float)
    for yr in df["year"].unique():
        tr, te = df[df.year != yr], df[df.year == yr]
        if kind == "month":
            clim = tr.groupby("month")["sm_0_7"].mean()
            pred[te.index] = te["month"].map(clim).fillna(tr["sm_0_7"].mean())
        else:
            pred[te.index] = make().fit(tr[feats], tr["sm_0_7"]).predict(te[feats])
    return r2_score(df["sm_0_7"], pred), np.sqrt(mean_squared_error(df["sm_0_7"], pred))

RF = lambda: RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)
LR = LinearRegression

radar_models = {
    "Month climatology (baseline)": (None, None, "month"),
    "LR: VV":                       (["VV"], LR, "model"),
    "LR: VV, VH":                   (["VV", "VH"], LR, "model"),
    "RF: VV, VH, RVI":              (["VV", "VH", "RVI"], RF, "model"),
    "RF: VV, VH, RVI + month":      (["VV", "VH", "RVI", "month"], RF, "model"),
}
ndvi_models = {
    "RF: VV, VH, RVI + NDVI":       (["VV", "VH", "RVI", "NDVI_mean"], RF, "model"),
    "RF: VV, VH + NDVI":            (["VV", "VH", "NDVI_mean"], RF, "model"),
}

def report(df, models, title):
    print(f"\n{title} (n={len(df)})")
    for name, (f, mk, kind) in models.items():
        r2, rmse = loyo(df, f, mk, kind)
        print(f"  {name:32s} R2 {r2:6.3f}   RMSE {rmse:.4f}")

report(m, radar_models, "Radar-only models, all scenes, leave-one-year-out")
common = m.dropna(subset=["NDVI_mean"])
if len(common) >= 30:
    report(common, {**radar_models, **ndvi_models},
           "Scenes with NDVI, same rows for every model")
else:
    print("\nToo few scenes with NDVI to compare fairly; radar-only models are the way forward.")

m.to_csv(ROOT / "data/processed/retrieval_dataset.csv", index=False)
