# Data Science & AI Capstone Projects

This repository hosts two industrial-grade Data Science & AI Capstone projects developed under the **DSAI** specialization:

1. **Capstone 1: Walmart Store Sales Demand Forecasting & Outlets Inventory Optimization** (Status: **Completed & Verified** ✅)
2. **Capstone 2: Online Retail Customer Purchasing Behavior & Segmentation** (Status: *In Planning* ⏳)

---

## 🏬 Capstone 1: Walmart Store Sales Demand Forecasting

### Quick Access & Key Artifacts

| Deliverable | Description | Direct Link |
| :--- | :--- | :--- |
| 📓 **Interactive Notebook** | Complete end-to-end modular pipeline (EDA, stats tests, CV, models, forward forecasts) | [`01_walmart_sales_forecasting.ipynb`](./capstone_1_walmart/notebooks/01_walmart_sales_forecasting.ipynb) |
| 📄 **Formal Project Report (PDF)** | Publication-grade 12-section PDF report conforming strictly to `Draft.pdf` guidelines | [`Walmart_Store_Sales_Forecasting_Report.pdf`](./capstone_1_walmart/reports/Walmart_Store_Sales_Forecasting_Report.pdf) |
| 📝 **Editable Report (Word)** | Microsoft Word document with embedded 300 DPI figures and native OpenXML page breaks | [`Walmart_Sales_Forecasting_Report.docx`](./capstone_1_walmart/reports/Walmart_Sales_Forecasting_Report.docx) |
| 📋 **Master Markdown Report** | Complete markdown source document with LaTeX mathematical formulations | [`report_walmart_sales_forecasting.md`](./capstone_1_walmart/reports/report_walmart_sales_forecasting.md) |
| 🏆 **Champion Model Weights** | Serialized tuned XGBoost regressor (Depth 7, LR 0.04, 300 estimators; 670 KB) | [`walmart_xgboost_iter02.joblib`](./capstone_1_walmart/models/walmart_xgboost_iter02.joblib) |
| 📊 **Forward 12-Week Projections** | Multi-step store-level forward sales predictions (Nov 2, 2012 to Jan 18, 2013) | [`forward_12week-forecast01.csv`](./capstone_1_walmart/data_processed/forward_12week-forecast01.csv) |

---

### Project Overview & Operational Mandate
A national retail chain operating **45 superstores** across heterogeneous geographic and macroeconomic microclimates experiences acute inventory replenishment friction. The objective is twofold:
1. **Operational Diagnostics**: Audit historical demand patterns across 6,435 observations (143 balanced weeks per store) to categorize stores into operational tiers and identify seasonal demand multipliers.
2. **Predictive Modeling & Forward Forecast**: Build a machine learning forecasting architecture to predict weekly sales per store for the **upcoming 12 weeks**, accounting for seasonal dynamics, macroeconomic indicators (CPI, Unemployment, Fuel Price), and holiday surges.

---

### Key Empirical Findings
- **Store Volume Stratification**: Mean weekly sales range eight-fold from **\$259,862** (Store 33) to **\$2,107,677** (Store 20).
  - *Tier 1 (Flagship)*: Stores 20, 4, 14, 13, and 2 consistently average between **\$1.93M and \$2.11M** weekly. Within Tier 1, **Store 14** exhibits the highest volatility ($\text{CV} \approx 0.157$, std of \$317.6K on a \$2.02M base), creating severe stockout exposure during demand surges.
  - *Tier 3 (Boutique)*: Stores 33, 44, 5, 36, and 38 average under **\$400K** weekly (Store 35 has highest nationwide CV of 0.230).
- **Holiday Demand Multipliers**:
  - **Thanksgiving / Black Friday**: Drives the largest single holiday uplift at **+41.3%** over baseline (\$1,471,273 vs. \$1,041,256).
  - **Pre-Christmas Peak (Week 51)**: While official calendar Christmas week registers a post-holiday lull (-7.7%), the penultimate week of December represents the single highest revenue week of the year (**\$80.9M** in 2010 and **\$77.0M** in 2011, representing a **+55% to +70%** national demand expansion).
  - **Super Bowl**: Modest **+3.6%** lift; **Labor Day**: **+0.1%** lift.
- **Statistical Stationarity**: Augmented Dickey-Fuller (ADF) tests decisively reject the unit root null hypothesis ($p < 0.0001$; National ADF = `-5.9083`, Store 1 ADF = `-5.1022`), confirming series stability around strong **52-week deterministic annual seasonality** (confirmed via ACF/PACF and seasonal decomposition).

---

### Model Evaluation & Out-of-Sample Benchmark

Models were validated using a 3-fold expanding chronological `TimeSeriesSplit` cross-validation scheme to strictly prevent future data leakage. Out-of-sample performance was benchmarked on the final 12-week holdout test partition (540 observations across 45 stores):

| Model Architecture | Scope | Out-of-Sample RMSE | Out-of-Sample MAE | Out-of-Sample MAPE | Out-of-Sample WAPE | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (Lag 1 Persistence)** | All 45 Stores | \$72,442.05 | \$46,965.55 | 4.71% | 4.57% | Lower Bound Benchmark |
| **Random Forest Regressor (150 Trees)** | All 45 Stores | \$56,656.71 | \$38,418.35 | 3.78% | 3.74% | Ensemble Bagging |
| 🏆 **Tuned Champion XGBoost** | **All 45 Stores** | **\$49,140.51** | **\$33,619.68** | **3.40%** | **3.27%** | **Promoted Champion** |
| **Facebook Prophet (Bayesian GAM)** | Sample Tiers (20, 1, 33) | \$86,284.53 | \$66,089.59 | 5.36% | 5.10% | Additive Decomposition |
| **Statistical SARIMAX (1,1,1)x(0,1,1)₁₂** | Sample Tiers (20, 1, 33) | \$104,025.64 | \$73,133.91 | 5.94% | 5.64% | Stochastic Regressor |
| 🏆 **Tuned XGBoost** | **Sample Tiers (20, 1, 33)** | **\$51,860.80** | **\$39,052.79** | **3.44%** | **3.01%** | **Promoted Champion** |

> **Business Impact**: The tuned champion XGBoost model reduces national forecast dollar error by **>28%** relative to the naive persistence model, achieving an unprecedented **3.27% WAPE** across all 45 outlets.

---

### Inventory Optimization & Scientific Safety Stock Policy

By combining point forecasts with the champion model's empirical error distribution ($\text{RMSE}_{\text{store}}$), store replenishment policies are mathematically formulated for a 2-week lead time ($L=2$) and a **95% service level** ($Z_{0.95} = 1.645$):

$$\text{Safety Stock} = Z_{\alpha} \times \text{RMSE}_{\text{store}} \times \sqrt{L}, \quad \text{Reorder Point (ROP)} = \mu_{\text{demand}} \times L + \text{Safety Stock}$$

- **Tier 1 Stores (Stores 4, 14, 20)**: Baseline safety stock buffer of **\$127,900**, scaling to **\$450,000–\$600,000** in Q4 to prevent stockouts during double-digit holiday demand spikes.
- **Tier 3 Stores (Stores 33, 44)**: Lean safety stock buffer of **\$41,800**, supporting weekly cross-docking without holding cost inflation.
- **Post-Holiday De-escalation**: Procurement orders must be tapered starting **December 10** to manage the sharp 60% post-holiday demand contraction in January and avoid Q1 dead stock.

---

### Detailed Directory Structure: `capstone_1_walmart/`

```text
capstone_1_walmart/
├── notebooks/
│   └── 01_walmart_sales_forecasting.ipynb   # Master executable notebook (10 sections, deepmind kernel)
├── reports/
│   ├── Walmart_Store_Sales_Forecasting_Report.pdf  # Formal 19-page publication PDF (Draft.pdf format)
│   ├── Walmart_Sales_Forecasting_Report.docx       # Editable Microsoft Word document with embedded plots
│   └── report_walmart_sales_forecasting.md         # Master source markdown report with LaTeX math
├── models/
│   ├── walmart_xgboost_iter02.joblib        # Champion tuned model (Depth 7, LR 0.04, 300 Trees; 670 KB)
│   ├── walmart_rf_iter02.joblib             # Random Forest benchmark model (12.7 MB, compressed)
│   ├── walmart_sarimax_store1_iter01.joblib # Statistical SARIMAX model for median-volume Store 1 (6.4 MB)
│   ├── walmart_sarimax_store20_iter01.joblib# Statistical SARIMAX model for flagship Store 20 (6.3 MB)
│   └── walmart_sarimax_store33_iter01.joblib# Statistical SARIMAX model for low-volume Store 33 (6.3 MB)
├── plots/
│   ├── 01_store_sales_distribution_boxplot.png      # 45-store sales distribution & volume tiers (300 DPI)
│   ├── 02_aggregate_sales_trend_with_holidays.png   # National weekly sales trend with annotated peaks
│   ├── 03_holiday_sales_uplift_comparison.png       # Holiday demand uplift multipliers (+41.3% Thanksgiving)
│   ├── 04_macro_correlation_heatmap.png             # Correlation matrix: Sales vs. CPI, Unemp, Fuel, Temp
│   ├── 05_time_series_decomposition.png             # 52-week additive seasonal decomposition
│   ├── 05b_acf_pacf_plots.png                       # ACF and PACF autocorrelation diagnostic plots
│   ├── 06_model_error_comparison_barchart.png       # Out-of-sample forecast error comparison (WAPE %)
│   ├── 07_predicted_vs_actual_12week_test.png       # Backtest trajectories across volume tiers (Stores 20, 1, 33)
│   └── 08_forward_12week_store_forecasts.png        # 12-week forward projections (Nov 2012 – Jan 2013)
└── data_processed/
    ├── baseline_split-train01.csv                   # Chronological raw training split (5,895 rows)
    ├── baseline_split-test01.csv                    # Chronological holdout test split (540 rows)
    ├── lag_rolling_features-train02.csv             # Autoregressive lags & rolling window features (train)
    ├── lag_rolling_features-test02.csv              # Autoregressive lags & rolling window features (test)
    ├── macro_cyclical_engineered-train03.csv        # Model-ready feature matrix with cyclical encodings (train)
    ├── macro_cyclical_engineered-test03.csv        # Model-ready feature matrix with cyclical encodings (test)
    └── forward_12week-forecast01.csv                # Forward 12-week recursive sales forecasts (540 rows)
```

---

## 🛒 Capstone 2: Online Retail Customer Purchasing Behavior & Segmentation

- **Objective**: Customer purchasing pattern discovery, transaction frequency modeling, customer lifetime value (CLV) estimation, and customer segmentation for an online retailer.
- **Approach**: RFM (Recency, Frequency, Monetary) analytics, power-law log transformations, and unsupervised ML clustering (K-Means, Agglomerative Hierarchical Clustering, DBSCAN, PCA).
- **Raw Dataset**: `data_raw/OnlineRetail.csv` (~541k transactions, ~388k clean records with verified CustomerID).
- **Directory**: [`capstone_2_onlineretail/`](./capstone_2_onlineretail/)

---

## 💻 Python Environment & Reproducibility

- **Python Version**: `3.11.15`
- **Conda Environment**: `DL_AI_env_311`
- **Interpreter Path**: `C:\Users\amans\anaconda3\envs\DL_AI_env_311\python.exe`
- **Jupyter Kernel**: `deepmind` (`DeepMind (Python 3.11.15)`)
- **Core Dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `prophet`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `pandoc`

To replicate the execution of the Walmart pipeline:
```powershell
# Activate conda environment
conda activate DL_AI_env_311

# Execute notebook builder & validator
python .agent/tools/build_walmart_nb.py

# Run notebook linter & naming convention audits
python .agent/tools/lint_notebook.py
python .agent/tools/validate_naming.py
```
