import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_data import build_live_table, ROOT

st.set_page_config(page_title="Mangaluru soil moisture", layout="wide")
st.title("Soil moisture forecast: Mangaluru")
st.warning(
    "Research prototype. Values are modeled (ERA5-Land via Open-Meteo), not sensor measurements, "
    "and the models were trained on one site's reanalysis data. Not for irrigation decisions.")
st.caption("Note: the 28-100 cm estimate is noisy from day to day (test RMSE 0.04, on a signal that "
               "mostly stays between 0.05 and 0.12) and biased high on average. Use it for the "
               "general level only, not for daily changes.")
@st.cache_data(ttl=3600)
def load():
    return build_live_table()

try:
    tbl, last_obs, today = load()
except Exception as e:
    st.error(f"Could not build the live forecast: {e}")
    st.stop()

gap = (today - last_obs).days
max_h = int(tbl["horizon"].max())
c1, c2, c3 = st.columns(3)
c1.metric("Today", str(today.date()))
c2.metric("Last ERA5 day", str(last_obs.date()), f"{gap} days old", delta_color="off")
c3.metric("Forecast steps from last ERA5 day", max_h)
if max_h > 7:
    st.info(f"The forecast reaches {max_h} steps from the last real value, past the 7-day horizon "
            "that was validated. Days beyond step 7 are drawn dashed with no error band.")

rmse = pd.read_csv(ROOT / "data/processed/forecast_trained_on_forecasts_rmse.csv", index_col="day")["Trained on forecasts"]
tab1, tab2, tab3, tab4 = st.tabs(["Surface (0-7 cm)", "Deeper layers", "Evaluation", "About"])

with tab1:
    obs = tbl[tbl["is_obs"]]; fc = tbl[~tbl["is_obs"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(obs.index, obs["sm_0_7"], "k-", label="ERA5-Land (observed)")
    if len(fc):
        join = pd.concat([obs.tail(1), fc])
        valid = join[join["horizon"] <= 7]; beyond = join[join["horizon"] >= 7]
        ax.plot(valid.index, valid["sm_0_7"], "C0-o", ms=3, label="Model forecast (validated range)")
        if len(beyond) > 1:
            ax.plot(beyond.index, beyond["sm_0_7"], "C0--", label="Beyond validated horizon")
        v = fc[fc["horizon"].between(1, 7)]
        e = v["horizon"].map(rmse)
        ax.fill_between(v.index, v["sm_0_7"] - e, v["sm_0_7"] + e, color="C0", alpha=0.2,
                        label="±1 RMSE (from test evaluation)")
    ax.axvline(today, color="gray", ls=":"); ax.text(today, ax.get_ylim()[1], " today", va="top")
    ax.set_ylabel("Volumetric moisture (m³/m³)"); ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate(); st.pyplot(fig)
    st.dataframe(fc[["sm_0_7", "rain", "horizon"]].rename(
        columns={"sm_0_7": "forecast_SM", "rain": "weather_rain_mm", "horizon": "steps_from_last_ERA5"}).round(3))

with tab2:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, est, era, title in [(axs[0], "est_7_28", "era_7_28", "7-28 cm"),
                                (axs[1], "est_28_100", "era_28_100", "28-100 cm")]:
        ax.plot(tbl.index, tbl[era], "k-", label="ERA5-Land (observed)")
        ax.plot(tbl.index, tbl[est], "C1-", label="Model estimate")
        ax.axvline(today, color="gray", ls=":"); ax.set_title(title); ax.legend(fontsize=8)
    fig.autofmt_xdate(); st.pyplot(fig)
    st.caption("Estimates for forecast days chain the surface forecast through the deep-layer models. "
               "On past days the surface input is observed ERA5, so the lines can be compared.")

with tab3:
    st.markdown("**Frozen results from the evaluation scripts.** Surface forecast results are an "
                "out-of-sample test on Jan-Oct 2026; the other tables use the Jun-Dec 2025 test period.")
    for title, f in [
        ("Surface forecast, 2026 out-of-sample test, trained on forecasts vs persistence (R²)",
         "forecast_trained_on_forecasts_r2.csv"),
        ("7-day surface forecast, 2025 test: real ECMWF forecasts vs oracle weather vs persistence (R²)",
         "real_forecast_r2.csv"),
        ("Same forecasts, stress-tested with simulated weather noise (R²)",
         "robustness_r2.csv"),
        ("Deep-layer estimation, base vs long-memory features",
         "deep_layer_v2_results.csv"),
    ]:
        p = ROOT / "data/processed" / f
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p))
with tab4:
    st.markdown("""
**Method (adapted from Gao et al., 2026, Smart Agricultural Technology):** a recursive XGBoost surface
forecast plus a progressive deep-layer estimator, for Mangaluru. The surface forecaster is trained on
archived ECMWF weather forecasts (2024 to Oct 2026) with lead time as an input; the deep-layer models are
trained on daily ERA5-Land data (2023-2025).

**Key limitations**
- Target data is modeled reanalysis (about 9 km), not in-situ sensors.
- Single location; the 2026 evaluation covers one year, with heavily overlapping forecast origins.
- Forecast skill falls with horizon and with rainfall-forecast error.
- The 28-100 cm estimate is noisy day to day and biased high on average.
- Satellite (Sentinel-1) module is a feasibility study only and is not part of this dashboard.
""")