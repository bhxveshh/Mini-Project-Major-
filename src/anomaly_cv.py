import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
m = pd.read_csv(ROOT / "data/processed/retrieval_dataset.csv", parse_dates=["time"])

FEATS = {"VV anomaly": ["VV_a"], "VV + VH anomaly": ["VV_a", "VH_a"]}
rows, pooled = [], {k: [] for k in FEATS}
pooled_true, pooled_base = [], []

for yr in sorted(m["year"].unique()):
    tr, te = m[m.year != yr].copy(), m[m.year == yr].copy()
    for col in ["sm_0_7", "VV", "VH"]:
        clim = tr.groupby("month")[col].mean()
        tr[col + "_a"] = tr[col] - tr["month"].map(clim)
        base = te["month"].map(clim).fillna(tr[col].mean())
        te[col + "_a"] = te[col] - base
        if col == "sm_0_7":
            te["base_sm"] = base
    for name, f in FEATS.items():
        mdl = LinearRegression().fit(tr[f], tr["sm_0_7_a"])
        a_pred = mdl.predict(te[f])
        pooled[name].append(pd.DataFrame({"a_true": te["sm_0_7_a"], "a_pred": a_pred,
                                          "true": te["sm_0_7"], "lvl": te["base_sm"] + a_pred}))
        rows.append(dict(held_out=yr, model=name, n=len(te),
                         baseline_R2=round(r2_score(te["sm_0_7"], te["base_sm"]), 3),
                         radar_R2=round(r2_score(te["sm_0_7"], te["base_sm"] + a_pred), 3),
                         anomaly_R2=round(r2_score(te["sm_0_7_a"], a_pred), 3),
                         anomaly_corr=round(np.corrcoef(te["sm_0_7_a"], a_pred)[0, 1], 3)))
    pooled_base.append(pd.DataFrame({"true": te["sm_0_7"], "lvl": te["base_sm"]}))

print(pd.DataFrame(rows).to_string(index=False))
b = pd.concat(pooled_base)
print(f"\nPOOLED over held-out years: baseline level R2 = {r2_score(b['true'], b['lvl']):.3f}")
for name, parts in pooled.items():
    p = pd.concat(parts)
    print(f"  {name}: level R2 = {r2_score(p['true'], p['lvl']):.3f} | "
          f"anomaly R2 = {r2_score(p['a_true'], p['a_pred']):.3f} | "
          f"anomaly corr = {np.corrcoef(p['a_true'], p['a_pred'])[0, 1]:.2f}")
