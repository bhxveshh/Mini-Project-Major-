"""
Cross-site comparison: load results from all 3 sites and produce
  fig6_cross_site_forecast.png  – R² and RMSE by forecast day
  fig7_cross_site_deep.png      – Deep layer R² comparison
  fig8_site_climatology.png     – Monthly mean SM and rainfall (3 panels)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

SITES = ["mangaluru", "mysore", "shivamogga"]
COLORS = {"mangaluru": "#e41a1c", "mysore": "#377eb8", "shivamogga": "#4daf4a"}
LABELS = {"mangaluru": "Mangaluru", "mysore": "Mysore", "shivamogga": "Shivamogga"}


# ──────────────────── Figure 6: Cross-site forecast performance ────────────────────
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for site in SITES:
        df = pd.read_csv(ROOT / f"data/processed/{site}/forecast_results.csv")
        xgb = df[df.model == "XGB"]
        axes[0].plot(xgb.day, xgb.R2, marker="o", color=COLORS[site], label=LABELS[site], linewidth=2)
        axes[1].plot(xgb.day, xgb.RMSE, marker="o", color=COLORS[site], label=LABELS[site], linewidth=2)

    axes[0].set_xlabel("Forecast Day", fontsize=12)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("Surface SM Forecast R² (XGBoost)", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Forecast Day", fontsize=12)
    axes[1].set_ylabel("RMSE (m³/m³)", fontsize=12)
    axes[1].set_title("Surface SM Forecast RMSE (XGBoost)", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGDIR / "fig6_cross_site_forecast.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────── Figure 7: Cross-site deep layer comparison ────────────────────
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bar_width = 0.25
    x_labels_7 = []
    x_labels_28 = []

    for i, site in enumerate(SITES):
        df = pd.read_csv(ROOT / f"data/processed/{site}/deep_layer_v2_results.csv")
        # Best feature set: "base + long memory", all models
        best = df[df.features == "base + long memory"]
        r2_7 = best[best.layer == "7-28cm"]
        r2_28 = best[best.layer == "28-100cm"]

        x = np.arange(len(r2_7))
        axes[0].bar(x + i * bar_width, r2_7.R2.values, bar_width,
                    color=COLORS[site], label=LABELS[site], alpha=0.85)
        axes[1].bar(x + i * bar_width, r2_28.R2.values, bar_width,
                    color=COLORS[site], label=LABELS[site], alpha=0.85)
        if i == 0:
            x_labels_7 = r2_7.model.values
            x_labels_28 = r2_28.model.values

    for ax, title, labels in [
        (axes[0], "Deep Layer Estimation R² (7–28 cm)", x_labels_7),
        (axes[1], "Deep Layer Estimation R² (28–100 cm)", x_labels_28),
    ]:
        ax.set_xticks(np.arange(len(labels)) + bar_width)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Long-Memory Feature Set", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGDIR / "fig7_cross_site_deep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────── Figure 8: Climatology comparison ────────────────────
def fig8():
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for i, site in enumerate(SITES):
        df = pd.read_csv(ROOT / f"data/raw/{site}_daily.csv", index_col=0, parse_dates=True)
        monthly = df.resample("MS").mean()
        rain_monthly = df["rain"].resample("MS").sum()

        ax = axes[i]
        ax2 = ax.twinx()
        ax.plot(monthly.index, monthly["sm_0_7"], color="brown", linewidth=2, label="SM 0–7 cm")
        ax.plot(monthly.index, monthly["sm_7_28"], color="orange", linewidth=2, label="SM 7–28 cm")
        ax.plot(monthly.index, monthly["sm_28_100"], color="goldenrod", linewidth=2, label="SM 28–100 cm")
        ax2.bar(rain_monthly.index, rain_monthly.values, width=25, alpha=0.3,
                color="steelblue", label="Rainfall")
        ax.set_ylabel("Soil Moisture (m³/m³)", fontsize=11)
        ax2.set_ylabel("Monthly Rainfall (mm)", fontsize=11)
        ax.set_title(f"{LABELS[site]}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Date", fontsize=12)
    plt.suptitle("Monthly Soil Moisture and Rainfall Climatology", fontsize=15, y=1.01)
    plt.tight_layout()
    out = FIGDIR / "fig8_site_climatology.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────── Summary table ────────────────────
def print_summary():
    print("\n" + "=" * 80)
    print("CROSS-SITE COMPARISON SUMMARY")
    print("=" * 80)

    # Forecast: XGB day-1 and day-7 R²/RMSE
    print("\n--- Surface Forecast (XGBoost) ---")
    print(f"{'Site':<15} {'Day-1 R²':>10} {'Day-1 RMSE':>12} {'Day-7 R²':>10} {'Day-7 RMSE':>12}")
    for site in SITES:
        df = pd.read_csv(ROOT / f"data/processed/{site}/forecast_results.csv")
        xgb = df[df.model == "XGB"]
        d1 = xgb[xgb.day == 1].iloc[0]
        d7 = xgb[xgb.day == 7].iloc[0]
        print(f"{LABELS[site]:<15} {d1.R2:>10.4f} {d1.RMSE:>12.4f} {d7.R2:>10.4f} {d7.RMSE:>12.4f}")

    # Deep layer: best XGB R² from v2
    print("\n--- Deep Layer (XGBoost, long-memory features) ---")
    print(f"{'Site':<15} {'7-28cm R²':>10} {'7-28cm RMSE':>13} {'28-100cm R²':>12} {'28-100cm RMSE':>15}")
    for site in SITES:
        df = pd.read_csv(ROOT / f"data/processed/{site}/deep_layer_v2_results.csv")
        best = df[(df.features == "base + long memory") & (df.model == "XGB")]
        r7 = best[best.layer == "7-28cm"].iloc[0]
        r28 = best[best.layer == "28-100cm"].iloc[0]
        print(f"{LABELS[site]:<15} {r7.R2:>10.4f} {r7.RMSE:>13.4f} {r28.R2:>12.4f} {r28.RMSE:>15.4f}")

    # Robustness: Oracle day-1 R²
    print("\n--- Robustness (Oracle day-1 R²) ---")
    for site in SITES:
        df = pd.read_csv(ROOT / f"data/processed/{site}/robustness_r2.csv", index_col=0)
        print(f"  {LABELS[site]:<15} Oracle day-1: {df.loc[1, 'Oracle']:.4f}   Persistence day-1: {df.loc[1, 'Persistence']:.4f}")

    # Climatology: mean SM and annual rainfall
    print("\n--- Mean SM and Annual Rainfall ---")
    print(f"{'Site':<15} {'SM 0-7':>8} {'SM 7-28':>8} {'SM 28-100':>10} {'Ann. Rain (mm)':>16}")
    for site in SITES:
        df = pd.read_csv(ROOT / f"data/raw/{site}_daily.csv", index_col=0, parse_dates=True)
        ann_rain = df["rain"].sum() / max(1, (df.index[-1] - df.index[0]).days / 365.25)
        print(f"{LABELS[site]:<15} {df['sm_0_7'].mean():>8.4f} {df['sm_7_28'].mean():>8.4f} "
              f"{df['sm_28_100'].mean():>10.4f} {ann_rain:>16.1f}")

    print("=" * 80)


if __name__ == "__main__":
    print("Generating cross-site comparison figures …\n")
    fig6()
    fig7()
    fig8()
    print_summary()
    print("\n✓ All comparison figures and summary generated.")
