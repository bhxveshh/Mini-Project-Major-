from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig); print("saved", name)

# ---- Fig 1: 7-day surface forecast, 2026 out-of-sample test ----
r2 = pd.read_csv(P / "forecast_trained_on_forecasts_r2.csv", index_col="day")
rm = pd.read_csv(P / "forecast_trained_on_forecasts_rmse.csv", index_col="day")
orc = pd.read_csv(P / "backtest2026_forecast_rmse.csv", index_col="day")["Oracle (ERA5 weather)"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for col, st in [("Persistence", "k--o"), ("Trained on ERA5", "C1-s"), ("Trained on forecasts", "C0-o")]:
    ax[0].plot(r2.index, r2[col], st, label=col, ms=4)
    ax[1].plot(rm.index, rm[col], st, label=col, ms=4)
ax[1].plot(orc.index, orc, "C2:^", label="Oracle (perfect weather)", ms=4)
ax[0].set_ylabel("R²"); ax[1].set_ylabel("RMSE (m³/m³)")
for a in ax: a.set_xlabel("Forecast day")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
fig.suptitle("Surface (0-7 cm) forecast, out-of-sample test Jan-Oct 2026 (276 origins)")
save(fig, "fig1_forecast_horizon.png")

# ---- Fig 2: random vs chronological split (weather features) ----
d = pd.read_csv(P / "deep_layer_results.csv")
d = d[d["weather"].astype(str) == "True"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, layer in zip(axs, ["7-28cm", "28-100cm"]):
    t = d[d.layer == layer].pivot(index="model", columns="split", values="R2").loc[["LR", "RF", "XGB"]]
    t.plot.bar(ax=ax, rot=0); ax.set_title(layer); ax.set_xlabel(""); ax.set_ylabel("R²")
fig.suptitle("Deep-layer R²: random split (paper's method) vs chronological split")
save(fig, "fig2_split_comparison.png")

# ---- Fig 3: effect of long-memory features (chronological split) ----
v = pd.read_csv(P / "deep_layer_v2_results.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, layer in zip(axs, ["7-28cm", "28-100cm"]):
    t = v[v.layer == layer].pivot(index="model", columns="features", values="R2").loc[["LR", "RF", "XGB"]]
    t.plot.bar(ax=ax, rot=0); ax.set_title(layer); ax.set_xlabel(""); ax.set_ylabel("R²")
fig.suptitle("Deep-layer R²: paper-style features vs + long-memory features (test Jun-Dec 2025)")
save(fig, "fig3_long_memory.png")

# ---- Fig 4: 2026 deep-layer error by month (models never saw 2026) ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name, f in [("7-28 cm", "backtest2026_deep_7_28.csv"), ("28-100 cm", "backtest2026_deep_28_100.csv")]:
    t = pd.read_csv(P / f, index_col=0)
    t = t[t.index.astype(str).str.startswith("2026-")]
    m = t.index.astype(str).str[-2:]
    axs[0].plot(m, t["RMSE"], "-o", ms=4, label=name)
    axs[1].plot(m, t["bias"], "-o", ms=4, label=name)
axs[0].set_ylabel("RMSE (m³/m³)"); axs[1].set_ylabel("Bias (estimate - ERA5)")
axs[1].axhline(0, color="k", lw=0.8)
for a in axs: a.set_xlabel("Month of 2026"); a.legend(fontsize=8)
fig.suptitle("Deep-layer estimates vs ERA5, by month of 2026")
save(fig, "fig4_deep_2026_monthly.png")

# ---- Fig 5: satellite feasibility (values copied from the printed output of
#      retrieval_cv.py and anomaly_cv.py; they were not saved to CSV) ----
sat = {"Month climatology\n(no radar)": 0.725, "LR: VV": 0.628, "LR: VV+VH": 0.613,
       "RF: VV,VH,RVI": 0.645, "RF: VV,VH,RVI\n+ month": 0.698,
       "Climatology +\nVV anomaly": 0.811}
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(list(sat), list(sat.values()),
       color=["gray", "C0", "C0", "C0", "C0", "C2"])
ax.set_ylabel("R² vs ERA5 surface moisture"); ax.set_ylim(0.4, 0.9)
ax.set_title("Sentinel-1 feasibility: leave-one-year-out, 75 scenes, one site")
ax.tick_params(axis="x", labelsize=8)
save(fig, "fig5_satellite_feasibility.png")
