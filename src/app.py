import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_data import build_live_table, ROOT

st.set_page_config(page_title="Soil Moisture Forecasting — Karnataka", layout="wide")

# Sidebar site selector
st.sidebar.title("Configuration")
site_option = st.sidebar.selectbox(
    "Select Study Area:",
    ["Mangaluru", "Mysore", "Shivamogga"],
    index=0
)
site_key = site_option.lower()

st.title(f"Soil Moisture Forecast: {site_option}")
st.warning(
    f"Research prototype for {site_option}. Values are modeled (ERA5-Land via Open-Meteo), not sensor measurements, "
    "and models are trained on reanalysis data. Not for operational irrigation decisions."
)

@st.cache_data(ttl=3600)
def load(site):
    return build_live_table(site)

try:
    tbl, last_obs, today = load(site_key)
except Exception as e:
    st.error(f"Could not build the live forecast for {site_option}: {e}")
    st.stop()

gap = (today - last_obs).days
max_h = int(tbl["horizon"].max())
c1, c2, c3 = st.columns(3)
c1.metric("Today", str(today.date()))
c2.metric("Last ERA5 day", str(last_obs.date()), f"{gap} days old", delta_color="off")
c3.metric("Forecast steps from last ERA5 day", max_h)
if max_h > 7:
    st.info(f"The forecast reaches {max_h} steps from the last real value, past the 7-day horizon "
            "that was validated. Days beyond step 7 are drawn dashed.")

# Load RMSE for error band if available
rmse_file = ROOT / f"data/processed/{site_key}/forecast_results.csv"
if not rmse_file.exists():
    rmse_file = ROOT / "data/processed/forecast_trained_on_forecasts_rmse.csv"
if rmse_file.exists():
    rdf = pd.read_csv(rmse_file)
    if "model" in rdf.columns:
        rmse_series = rdf[rdf["model"] == "XGB"].set_index("day")["RMSE"]
    elif "Trained on forecasts" in rdf.columns:
        rmse_series = rdf.set_index("day")["Trained on forecasts"]
    else:
        rmse_series = rdf.iloc[:, 1]
else:
    rmse_series = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Surface (0-7 cm)", "Deeper layers", "Evaluation", "Study Area & Cross-Site", "About"
])

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
        if rmse_series is not None:
            v = fc[fc["horizon"].between(1, 7)]
            e = v["horizon"].map(rmse_series).fillna(0.03)
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
    st.markdown(f"**Frozen results for {site_option}.**")
    site_proc = ROOT / f"data/processed/{site_key}"
    if not site_proc.exists():
        site_proc = ROOT / "data/processed"
    
    for title, f in [
        ("Surface Forecast Results by Day", "forecast_results.csv"),
        ("Deep-Layer Estimation (Base vs Long Memory)", "deep_layer_v2_results.csv"),
        ("Deep-Layer Split Comparison (Random vs Chronological)", "deep_layer_results.csv"),
        ("Robustness to Weather Forecast Noise (R²)", "robustness_r2.csv"),
    ]:
        p = site_proc / f
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p))

with tab4:
    st.header("Study Area Locations and Cross-Site Analysis")
    
    map_png = ROOT / "figures/fig_study_area_map.png"
    if map_png.exists():
        st.subheader("Geographical Distribution of Study Areas")
        st.image(str(map_png), caption="Three Study Areas in Karnataka: Mangaluru, Mysore, and Shivamogga", use_container_width=True)

    html_map = ROOT / "figures/study_area_interactive.html"
    if html_map.exists():
        st.subheader("Interactive Study Area Map")
        with open(html_map, "r", encoding="utf-8") as hf:
            st.components.v1.html(hf.read(), height=500, scrolling=True)

    cfig6 = ROOT / "figures/fig6_cross_site_forecast.png"
    if cfig6.exists():
        st.subheader("Surface Forecast Comparison Across Sites")
        st.image(str(cfig6), use_container_width=True)

    cfig7 = ROOT / "figures/fig7_cross_site_deep.png"
    if cfig7.exists():
        st.subheader("Deep-Layer Estimation Comparison Across Sites")
        st.image(str(cfig7), use_container_width=True)

    cfig8 = ROOT / "figures/fig8_site_climatology.png"
    if cfig8.exists():
        st.subheader("Soil Moisture & Rainfall Climatology Across Sites")
        st.image(str(cfig8), use_container_width=True)

with tab5:
    st.markdown("""
**Method (adapted from Gao et al., 2026, Smart Agricultural Technology):** a recursive XGBoost surface
forecast plus a progressive deep-layer estimator, applied across three distinct agro-climatic zones in Karnataka, India:
- **Mangaluru**: Coastal belt with high monsoonal rainfall (>3,500 mm).
- **Mysore**: Southern Deccan plateau with semi-arid rain-shadow climate (~1,000 mm).
- **Shivamogga**: Western Ghats transition zone with intermediate heavy rainfall (~1,200 mm).

**Key limitations**
- Target data is modeled reanalysis (about 9 km), not in-situ sensors.
- Forecast skill falls with horizon and with rainfall-forecast error.
- Deep layer dynamics vary significantly between tropical wet and semi-arid plateau regimes.
- Satellite (Sentinel-1) module is a feasibility study only.
""")
