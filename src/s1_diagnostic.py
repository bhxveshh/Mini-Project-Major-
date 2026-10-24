import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
s = pd.read_csv(ROOT / "data/raw/s1_site.csv", parse_dates=["time"])
d = pd.read_csv(ROOT / "data/raw/mangaluru_daily.csv", index_col=0, parse_dates=True)
d["sm_0_7"] = d["sm_0_7"].clip(lower=0)

m = (s.merge(d[["sm_0_7", "rain"]], left_on="time", right_index=True)
       .sort_values("time").reset_index(drop=True))
m["VV_VH"] = m["VV"] - m["VH"]
m["month"] = m["time"].dt.month
print("Scenes:", len(m), "| mean gap (days):", m["time"].diff().dt.days.mean().round(1))

# 1. raw
print("\n[1] Raw correlation with ERA5 moisture:")
for c in ["VV", "VH", "VV_VH"]:
    print(f"  {c}: {m[c].corr(m['sm_0_7']):+.2f}")

# 2. remove the seasonal cycle (subtract each calendar month's mean)
for c in ["VV", "VH", "sm_0_7"]:
    m[c + "_anom"] = m[c] - m.groupby("month")[c].transform("mean")
print("\n[2] Correlation of DESEASONALISED values:")
for c in ["VV", "VH"]:
    print(f"  {c}: {m[c + '_anom'].corr(m['sm_0_7_anom']):+.2f}")

# 3. scene-to-scene changes
dm = m[["VV", "VH", "sm_0_7"]].diff().dropna()
print("\n[3] Correlation of changes between consecutive scenes:")
for c in ["VV", "VH"]:
    print(f"  d{c} vs dSM: {dm[c].corr(dm['sm_0_7']):+.2f}  (n={len(dm)})")

# season split
mon = m["month"].isin([6, 7, 8, 9])
print("\n[4] Raw VV vs moisture by season:")
for name, g in [("monsoon Jun-Sep", m[mon]), ("rest of year", m[~mon])]:
    print(f"  {name}: n={len(g)}, VV corr {g['VV'].corr(g['sm_0_7']):+.2f}, "
          f"VH corr {g['VH'].corr(g['sm_0_7']):+.2f}")

# 5. leave-one-out prediction of moisture
def loo_r2(X, y):
    p = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return r2_score(y, p)

months = pd.get_dummies(m["month"], prefix="m").astype(float)
y = m["sm_0_7"].values
print("\n[5] Leave-one-out R2 for predicting ERA5 surface moisture:")
print(f"  month-of-year only:        {loo_r2(months, y):.3f}")
print(f"  VV only:                   {loo_r2(m[['VV']], y):.3f}")
print(f"  VV + VH:                   {loo_r2(m[['VV', 'VH']], y):.3f}")
print(f"  month + VV + VH:           {loo_r2(pd.concat([months, m[['VV', 'VH']]], axis=1), y):.3f}")

# plot (z-scored so both fit one axis)
z = lambda v: (v - v.mean()) / v.std()
plt.figure(figsize=(10, 4))
plt.plot(m["time"], z(m["VV"]), "o-", label="VV backscatter (z)")
plt.plot(m["time"], z(m["sm_0_7"]), "s-", label="ERA5 surface moisture (z)")
plt.legend(); plt.tight_layout()
plt.savefig(ROOT / "data/processed/s1_vs_era5.png", dpi=150)
print("\nSaved plot: data/processed/s1_vs_era5.png")
