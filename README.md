# NHS Scotland A&E Crisis Forecaster
 
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Open%20Dashboard-brightgreen?style=for-the-badge)](https://nhs-scotland-ae-forecaster.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion%20Model-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-Logistic%20Growth-purple?style=flat-square)](https://facebook.github.io/prophet/)
[![Data](https://img.shields.io/badge/Data-Public%20Health%20Scotland-005EB8?style=flat-square)](https://opendata.nhs.scot/)
 
> **A multivariate time-series forecasting system that predicts NHS Scotland A&E waiting time breaches up to 4 weeks ahead — across all 11 mainland health boards — using open government data, staffing ratios, flu search trends, and an XGBoost + Prophet ensemble.**
 
**→ [Open the Live Dashboard](https://nhs-scotland-ae-forecaster.streamlit.app/)**
 
---
 
## What This Project Does
 
The Scottish Government mandates that **95% of A&E patients are seen within 4 hours**. Since 2022, nearly every mainland health board has been in permanent breach of this target — with some boards failing up to 65% of patients per week.
 
This tool forecasts **how severe that breach will be** in the coming weeks and explains *why* — so NHS planners can act before a crisis deepens rather than react after.
 
**Four tabs in the dashboard:**
 
| Tab | What it shows |
|-----|---------------|
| **Proof of Accuracy** | Model trained on 2015–2022, blindfolded on 2023–2026. Green line vs blue dotted line = predicted vs actual |
| **National Leaderboard** | All 11 boards ranked by predicted breach severity this week |
| **How It Decides** | SHAP waterfall explaining exactly which factors drove a high-risk prediction |
| **Hiring Simulator** | Policy counterfactual — slide to simulate the impact of more clinical staff on breach numbers |
 
---
 
## 🏆 Model Performance
 
| Model | MAPE | RMSE | Notes |
|-------|------|------|-------|
| SARIMAX + Exog | 67.30% | 0.4061 | Defeated by post-COVID structural break |
| Prophet Baseline | 66.51% | 0.3762 | Trend extrapolated beyond bounds |
| Prophet Logistic (cap=0.85) | 28.45% | 0.1604 | Logistic growth solved runaway trend |
| **XGBoost (All Boards)** | **10.55%** | **0.0410** | **Champion — handles regime shifts natively** |
| **Ensemble (XGB 98.3% + Prophet 1.7%)** | **10.51%** | **0.0409** | **Best overall — optimised weights** |
 
All metrics computed on held-out test data (Jan 2023 → May 2026, never seen during training).
 
---
 
## 📁 Project Structure
 
```
nhs-scotland-ae-forecaster/
│
├── data/
│   ├── raw/                    ← Source CSVs (gitignored)
│   └── processed/              ← Cleaned features, model outputs, plots
│       ├── ae_features.csv         (6,446 rows × 43 features)
│       ├── ensemble_predictions.csv
│       ├── model_results.csv
│       ├── xgb_shap_waterfall.png
│       └── counterfactual_staffing.png
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Data loading, filtering, EDA plots
│   ├── 02_sarimax.ipynb        ← SARIMAX baseline + exog + ADF tests
│   ├── 03_prophet.ipynb        ← Prophet additive → logistic growth
│   ├── 04_xgboost.ipynb        ← XGBoost + SHAP explainability
│   ├── 05_ensemble.ipynb       ← Optimised ensemble, weight tuning
│   └── 06_counterfactual.ipynb ← Policy "What-If" staffing analysis
│
├── dashboard/
│   └── app.py                  ← Streamlit dashboard (4 tabs)
│
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Data Sources (All Free & Open)
 
| Source | What it provides | URL |
|--------|-----------------|-----|
| Public Health Scotland | Weekly A&E activity & waiting times (2015–present) | [opendata.nhs.scot](https://opendata.nhs.scot) |
| NES Turas Data Intelligence | NHS Scotland quarterly workforce WTE by board & job family | [turasdata.nes.nhs.scot](https://turasdata.nes.nhs.scot) |
| Google Trends (pytrends) | Weekly flu search index for Scotland (flu symptoms, NHS 24, fever, sore throat) | via `pytrends` |
| Scottish Government | School holiday calendar 2015–2026 | [gov.scot](https://www.gov.scot) |
| gov.scot / NRS Scotland | SIMD deprivation scores & population estimates per health board | [gov.scot](https://www.gov.scot) |
 
---
 
## Feature Engineering (43 Features)
 
### Target Variable
- `BreachRate` — proportion of attendances exceeding 4-hour wait (continuous, 0–1)
- `BreachedTarget` — binary flag: 1 if board missed the 95% government target that week
### Calendar Features
`WeekOfYear`, `Month`, `FluSeasonFlag` (Oct–Mar), `WinterFlag` (Dec–Feb), `ChristmasWeek`
 
### COVID Structural Break Variables
`CovidPhase` (0–3), `CovidEra`, `PostCovidStress` (Apr 2022+), `VaxRollout`
 
> These were the single most important modelling decision. Without them, SARIMAX was confused by the 2020–2022 anomaly and could not forecast the post-2022 plateau.
 
### Lag & Rolling Features
`Lag1/2/4/8_BreachRate`, `Lag1/4_Attendances`, `RollingMean4W/8W`, `RollingStd4W`, `RollingMax4W`, `BreachRateDelta`
 
> SHAP analysis confirmed these dominate XGBoost predictions — recent system state is the strongest predictor of near-term future performance.
 
### External Demand Signals
- `FluSearchIndex` — composite Google Trends score (4 keywords, Scotland, resampled monthly→weekly)
- `IsSchoolHoliday`, `DaysToNextHoliday`, `DaysSinceHoliday` — 586 holiday days hardcoded from Scottish Government calendar
### Staffing & Capacity
- `TotalClinicalWTE` — sum of Nursing, Medical, AHP, Healthcare Science WTE; interpolated quarterly→weekly
- `StaffingRatioPer1000` — WTE per 1,000 weekly attendances
- `StaffingPressureFlag` — 1 if ratio falls below board's own historical 10th percentile
### Board Fixed Effects
`SIMDScore` (deprivation), `IsUrban`, `PopulationK`, `AttendancesPer1000`
 
---
 
## Modelling Journey
 
### Why SARIMAX failed
Raw BreachRate is non-stationary (ADF: p=0.63). After first-differencing (p=0.0000) and fitting SARIMAX(1,1,1)(1,1,0,52), the model achieved 71.77% MAPE. Adding exogenous features reduced this to 67.30% — but the model extrapolated the post-2022 trend beyond 1.0 (physically impossible). Only `CovidEra` was statistically significant (p=0.014). SARIMAX contributes seasonal structure to the ensemble but cannot model structural breaks.
 
### Why Prophet needed a logistic cap
Prophet's default linear trend learned the 2021–2022 surge and extrapolated it indefinitely. Switching to `growth='logistic'` with `cap=0.85` forced the model to recognise the system reached a new stable crisis level. This single change dropped MAPE from 67.74% → **28.45%**. Adding a `PostCovidPlateau` feature made things marginally worse, proving the logistic curve already captured the plateau.
 
### Why XGBoost won
Tree-based models route post-2022 data through different decision paths from pre-2022 data — they handle structural breaks natively without any explicit regime variable. Combined with direct access to lag features (XGBoost sees last week's breach rate as an input), the model achieved **8.39% MAPE on Forth Valley** and **10.55% across all 11 boards**.
 
### Ensemble optimisation
`scipy.optimize.minimize` tested weight combinations under the constraint that weights sum to 1. Optimal: **XGBoost 98.3%, Prophet 1.7%**. The 1.7% Prophet contribution shaved ensemble MAPE from 10.55% → 10.51%, confirming the ensemble improves over any single model.
 
---
 
## Key Findings (SHAP Analysis)
 
From `shap.TreeExplainer` on the XGBoost test set:
 
1. **`RollingMean4W`** and **`Lag1_BreachRate`** are an order of magnitude more important than all external features combined — recent system history dominates
2. **`StaffingRatioPer1000`** is the strongest externally-engineered feature (rank 7 of 30)
3. **`VaxRollout`** captures the post-vaccination demand shift (rank 5)
4. `FluSearchIndex`, `SIMDScore`, and school holidays have measurable but small impact
**Waterfall interpretation (week of 2024-10-13, Forth Valley):**
Starting from a base expectation of 0.125, the model predicted 0.576 because `RollingMean4W=0.55` (+0.16) and `Lag1_BreachRate=0.564` (+0.16) indicated persistent high pressure over the preceding 8 weeks.
 
---
 
## Policy Counterfactual
 
**Question:** If NHS Forth Valley had 10% more clinical staff in Q1 2023, how many patients would have avoided a 4-hour breach?
 
**Result:**
- Baseline predicted breaches: **6,189**
- With +10% staffing: **5,971**
- **Estimated 218 patients saved** over 12 weeks
This demonstrates the model functions as a prescriptive policy tool, not just a forecasting exercise.
 
---
 
## Run Locally
 
```bash
# Clone
git clone https://github.com/angelvbenit/nhs-scotland-ae-forecaster.git
cd nhs-scotland-ae-forecaster
 
# Set up environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
 
# Run dashboard
streamlit run dashboard/app.py
```
 
**Note:** Raw data files are gitignored. To reproduce from scratch, download:
- Weekly A&E data from [opendata.nhs.scot](https://opendata.nhs.scot) → save as `data/raw/ae_weekly_raw.csv`
- Workforce data from [turasdata.nes.nhs.scot](https://turasdata.nes.nhs.scot) → save as `data/raw/workforce_raw.xlsx`
Then run notebooks 01 through 06 in order.
 
---
 
## Dependencies
 
```
pandas / numpy / matplotlib / seaborn
statsmodels          # SARIMAX
prophet              # Facebook Prophet
xgboost / shap       # XGBoost + explainability
pytrends             # Google Trends API
plotly / streamlit   # Dashboard
scipy                # Ensemble weight optimisation
openpyxl             # Workforce Excel parsing
python-dotenv        # API key management
```
 
Full list in `requirements.txt`.
 
---
 
## Limitations
 
- **Weather data not yet integrated** — Met Office DataPoint API was blocked by institutional firewall during development. KNMI Climate Explorer or CEDA Archive are planned fallbacks
- **Island boards excluded** — Orkney, Shetland, Western Isles have volumes too small for meaningful breach rate forecasting
- **No rolling-origin cross-validation** — metrics are from a single 2023+ holdout. Full expanding-window CV is the next methodological improvement
- **Staffing data is quarterly** — interpolated linearly to weekly, which smooths out genuine workforce spikes
---
 
## Roadmap
 
- [ ] Add Met Office weather features (WinterPressureIndex, FrostDays)
- [ ] Implement rolling-origin cross-validation for robust metric reporting
- [ ] Add Temporal Fusion Transformer (TFT) as fourth ensemble component
- [ ] Extend counterfactual simulator to all 11 boards
- [ ] Add 4-week ahead forecast ribbon (currently shows test-period backtesting)
---
 
## Author
 
Built as a personal project.  
Data sources: Public Health Scotland, NES Turas, Scottish Government, Google Trends.  
All data is open and freely available.
 
---
 
*This project is for educational and research purposes. It is not a clinical decision support tool and should not be used to make operational NHS decisions.*