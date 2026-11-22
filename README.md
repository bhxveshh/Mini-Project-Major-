# Soil Moisture Retrieval and Multi-Day Forecasting

A machine learning framework for soil moisture forecasting and deep-layer profile estimation using meteorological inputs and remote sensing data. This project replicates and extends the methodology from **Gao et al. (2026)** (*Smart Agricultural Technology*), adapted for Indian study sites (Mangaluru, Mysore, and Shivamogga).

---

## 📁 Repository Structure

```
.
├── data/                    # Datasets
│   ├── raw/                 # Raw weather and soil moisture observations
│   └── processed/           # Feature-engineered tables and cross-validation datasets
├── figures/                 # Generated evaluation plots, maps, and climatology charts
├── models/                  # Trained XGBoost models and metadata
│   ├── mangaluru/           # Mangaluru site models
│   ├── mysore/              # Mysore site models
│   └── shivamogga/          # Shivamogga site models
├── notebooks/               # Jupyter notebooks for exploratory data analysis
├── outputs/                 # Model evaluation logs and backtest summaries
│   └── backtest_output.txt  # 2026 chronological backtest evaluation log
├── reports/                 # Academic deliverables and presentations
│   ├── compile.sh           # One-click LaTeX compilation helper script
│   ├── report/              # Final written project report
│   │   ├── report.tex       # LaTeX source for the comprehensive report
│   │   ├── report.pdf       # Compiled final report PDF
│   │   └── multirow.sty     # LaTeX table styling stub
│   ├── presentation/        # Project viva / seminar presentation
│   │   ├── presentation.tex # LaTeX Beamer presentation source
│   │   ├── presentation.pdf # Compiled presentation slides
│   │   ├── nitk_logo.png    # NITK institution logo asset
│   │   └── slide_previews/  # Rendered PNG slide previews
│   └── templates/           # Official institute formatting templates
│       ├── Mini project template-front pages.docx
│       └── nitk_screenshot.png
├── scripts/                 # Utility scripts (e.g., dashboard screenshot capture)
│   ├── capture_screenshots.py
│   └── capture_full_screenshots.py
├── src/                     # Core Python source code
│   ├── app.py               # Interactive Streamlit dashboard
│   ├── live_data.py         # Live weather fetch & multi-layer inference pipeline
│   ├── backtest_2026.py     # 2026 evaluation & backtesting script
│   ├── compare_sites.py     # Multi-site comparative analysis
│   ├── make_figures.py      # Publication figure generator
│   └── ...                  # Model training and validation modules
├── study area/              # GIS boundary shapefiles for Mysore and Shivamogga
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Interactive Dashboard

Launch the Streamlit web application:

```bash
streamlit run src/app.py
```

### 3. Compiling Reports and Slides

A helper script is provided in `reports/` to compile LaTeX documents into PDFs:

```bash
# Compile both the report and presentation:
./reports/compile.sh all

# Or compile individually:
./reports/compile.sh report
./reports/compile.sh presentation

# Clean auxiliary build files (.aux, .log, .toc, etc.):
./reports/compile.sh clean
```

---

## 📊 Methodology Overview

1. **Surface Forecasting (0–7 cm):** Recursive multi-step forecasting with XGBoost using autoregressive lags ($t-1, t-2$), lead-time encoding, and ECMWF meteorological forecasts.
2. **Deep-Layer Profile Estimation (7–28 cm & 28–100 cm):** Gradient boosting regressor mapping surface moisture trajectories and cumulative rainfall memory to root-zone moisture.
3. **Multi-Site Analysis:** Evaluated across Coastal Karnataka (Mangaluru), Southern Dry Zone (Mysore), and Central Malnad (Shivamogga).
