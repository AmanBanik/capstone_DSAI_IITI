# Data Science & AI Capstone Projects

This repository hosts two industrial-grade Data Science & AI Capstone projects developed under the **DSAI** specialization:

1. **Capstone 1: Walmart Store Sales Demand Forecasting & Outlets Inventory Optimization** (Status: **Completed & Verified** ✅)
2. **Capstone 2: Online Retail Customer Purchasing Behavior & Segmentation** (Status: **Completed & Verified** ✅)

---

## 🏬 Capstone 1: Walmart Store Sales Demand Forecasting

### Quick Access & Key Artifacts

| Deliverable | Description | Direct Link |
| :--- | :--- | :--- |
| 📓 **Interactive Notebook** | Complete end-to-end modular pipeline (EDA, stats tests, CV, models, forward forecasts) | [`01_walmart_sales_forecasting.ipynb`](./capstone_1_walmart/notebooks/01_walmart_sales_forecasting.ipynb) |
| 📄 **Formal Project Report (PDF)** | Comprehensive 12-section PDF report conforming strictly to `Draft.pdf` guidelines | [`Walmart_Store_Sales_Forecasting_Report.pdf`](./capstone_1_walmart/reports/Walmart_Store_Sales_Forecasting_Report.pdf) |
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

Candidate models were validated and selected using a 3-fold date-grouped expanding chronological panel cross-validation scheme across all 45 stores to strictly prevent future data leakage. Hyperparameters were tuned on cross-validation only (Candidate 3 Depth 7 selected as Champion: CV Mean RMSE $78,917.02 \pm $19,502.68, CV Mean WAPE 4.74% \pm 1.07%). Out-of-sample performance was evaluated once on the final 12-week holdout test partition (540 observations across 45 stores):

| Model Architecture | Scope | Out-of-Sample RMSE | Out-of-Sample MAE | Out-of-Sample MAPE | Out-of-Sample WAPE | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (Lag 1 Persistence)** | All 45 Stores | \$72,442.05 | \$46,965.55 | 4.71% | 4.57% | Lower Bound Benchmark |
| **Random Forest Regressor (150 Trees)** | All 45 Stores | \$56,656.71 | \$38,418.35 | 3.78% | 3.74% | Ensemble Bagging |
| 🏆 **Tuned Champion XGBoost** | **All 45 Stores** | **\$49,140.51** | **\$33,619.68** | **3.40%** | **3.27%** | **Promoted Champion** |
| **Facebook Prophet (Bayesian GAM)** | Sample Tiers (20, 1, 33) | \$86,284.53 | \$66,089.59 | 5.36% | 5.10% | Additive Decomposition |
| **Statistical SARIMAX (1,1,1)x(0,1,1)₁₂** | Sample Tiers (20, 1, 33) | \$104,025.64 | \$73,133.91 | 5.94% | 5.64% | Stochastic Regressor |
| 🏆 **Tuned XGBoost** | **Sample Tiers (20, 1, 33)** | **\$51,860.80** | **\$39,052.79** | **3.44%** | **3.01%** | **Promoted Champion** |

> **Business Impact**: The tuned champion XGBoost model reduces national forecast dollar error by **>28% (specifically 32.17%)** relative to the naive persistence model, achieving an unprecedented **3.27% WAPE** across all 45 outlets.
> **Calibrated Uncertainty**: 90% prediction intervals calibrated strictly on training CV out-of-fold residuals achieved **91.30% empirical coverage** across the 540 holdout test observations with an average width of **$150,396.66**.

---

### Inventory Optimization & Scientific Safety Stock Policy

By combining point forecasts with the champion model's empirical error distribution ($\text{RMSE}_{\text{store}}$), store replenishment policies are mathematically formulated for a 2-week lead time ($L=2$) and a **95% service level** ($Z_{0.95} = 1.645$):

$$\text{Safety Stock} = Z_{\alpha} \times \text{RMSE}_{\text{store}} \times \sqrt{L}, \quad \text{Reorder Point (ROP)} = \mu_{\text{demand}} \times L + \text{Safety Stock}$$

- **Tier 1 Stores (Stores 4, 14, 20)**: Baseline safety stock buffer of **\$127,900**, scaling to **\$450,000–\$600,000** in Q4 to prevent stockouts during double-digit holiday demand spikes.
- **Tier 3 Stores (Stores 33, 44)**: Lean safety stock buffer of **\$41,800**, supporting weekly cross-docking without holding cost inflation.
- **Post-Holiday De-escalation**: Procurement orders must be tapered starting **December 10** to manage the sharp 60% post-holiday demand contraction in January and avoid Q1 dead stock.

> **Methodological Qualification**: The dollar-based safety stock formula ($Z_{\alpha} \times \text{RMSE}_{\text{store}} \times \sqrt{L}$) provides an illustrative strategic framework for capital allocation. Operational warehouse deployment requires granular SKU-level physical unit demand, stochastic lead-time variance, and discrete-event inventory simulation.
> **Baseline Macro Persistence Scenario**: The 12-week forward projections assume frozen macroeconomic indicators (`CPI`, `Unemployment`, `Fuel_Price`) and seasonal temperatures at their final observed values as a baseline demand planning scenario.

---

### Detailed Directory Structure: `capstone_1_walmart/`

```text
capstone_1_walmart/
├── notebooks/
│   └── 01_walmart_sales_forecasting.ipynb   # Master executable notebook (10 sections, deepmind kernel)
├── reports/
│   ├── Walmart_Store_Sales_Forecasting_Report.pdf  # Formal 24-page PDF report (Draft.pdf format)
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

### Quick Access & Key Artifacts

| Deliverable | Description | Direct Link |
| :--- | :--- | :--- |
| 📓 **Interactive Notebook** | Complete end-to-end unsupervised pipeline (Hygiene, RFM, Log1p, PCA, K-Means, Agglomerative, DBSCAN) | [`01_retail_customer_segmentation.ipynb`](./capstone_2_onlineretail/notebooks/01_retail_customer_segmentation.ipynb) |
| 📄 **Formal Project Report (PDF)** | Comprehensive 12-section 27-page PDF report conforming strictly to `Draft.pdf` guidelines | [`Online_Retail_Customer_Segmentation_Report.pdf`](./capstone_2_onlineretail/reports/Online_Retail_Customer_Segmentation_Report.pdf) |
| 📝 **Editable Report (Word)** | Microsoft Word document with embedded 300 DPI figures and native OpenXML page breaks | [`Retail_Customer_Segmentation_Report.docx`](./capstone_2_onlineretail/reports/Retail_Customer_Segmentation_Report.docx) |
| 📋 **Master Markdown Report** | Complete markdown source document with LaTeX formulations and persona tables | [`report_retail_customer_segmentation.md`](./capstone_2_onlineretail/reports/report_retail_customer_segmentation.md) |
| 🏆 **Champion Clustering Model** | Serialized Champion K-Means model ($k=4$, `k-means++` seeding, `random_state=42`) | [`retail_kmeans_champion_k4.joblib`](./capstone_2_onlineretail/models/retail_kmeans_champion_k4.joblib) |
| 🔬 **Feature Scaler & PCA** | Preserved StandardScaler and 2-component PCA transformers (93.87% variance explained) | [`retail_scaler_iter01.joblib`](./capstone_2_onlineretail/models/retail_scaler_iter01.joblib), [`retail_pca_iter01.joblib`](./capstone_2_onlineretail/models/retail_pca_iter01.joblib) |
| 📊 **Segmented Customer Table** | Final entity matrix with assigned Personas, RFM metrics, AOV, and 2D PCA coordinates | [`segmented_customers-final01.csv`](./capstone_2_onlineretail/data_processed/segmented_customers-final01.csv) |

---

### Project Overview & Operational Mandate
An international e-commerce retailer based in the UK operating across **38 global markets** specialized in gifts and novelty homeware experienced severe marketing friction from uniform, blanket promotional blasts. The objective is twofold:
1. **Behavioral Feature Engineering**: Clean 541,909 transaction records (filtering 5,268 duplicates, 9,288 cancellations, and 135,080 unauthenticated checkouts) to extract customer-level **Recency, Frequency, Monetary, and Average Order Value (RFM-AOV)** metrics for 4,338 verified commercial accounts.
2. **Unsupervised Segmentation & Value Optimization**: Overcome severe power-law skewness, project behavioral axes via PCA, benchmark multiple clustering paradigms, and establish actionable commercial playbooks.

---

### Key Empirical Findings & Mathematical Profiling
- **Severe Transactional Skewness Rectified**:
  - *Frequency ($F$)*: Mean 4.27 orders, Max 209 orders; extreme positive skewness of **+12.07** successfully normalized to **+1.21** via monotonic $\log(1+x)$.
  - *Monetary ($M$)*: Mean £2,048.69, Median £668.57, Max £280,206.02; extreme positive skewness of **+19.32** normalized to **+0.39** via $\log(1+x)$.
  - *Recency ($R$)*: Mean 92.5 days, Median 51.0 days; skewness shifted from +1.25 to **-0.38**.
- **PCA Dimensionality Reduction**:
  - **PC1 (Customer Engagement & Lifetime Value Spectrum)**: Explains **75.08%** of total variance (strong positive loadings on Frequency +0.60, Monetary +0.61; negative on Recency -0.51).
  - **PC2 (Order Structure & Basket Dynamics)**: Explains **18.79%** of variance.
  - **Cumulative Variance**: **93.87% explained variance** across 2 dimensions, enabling an high-fidelity 2D visualization (retaining 93.87% of behavioral variance).
- **Top Product & Geographic Dynamics**:
  - `PAPER CRAFT , LITTLE BIRDIE`: Highest single-transaction revenue (£168.5K via 1 wholesale bulk order of 80,995 units).
  - `REGENCY CAKESTAND 3 TIER`: Premier recurring commercial bestseller (£142.3K across 1,703 distinct customer orders).
  - `United Kingdom`: Generates **£7.29M (81.97% of total revenue)** and accounts for **3,920 of 4,338 customers (90.36%)**. Top export markets: Netherlands (£285.4K, 3.21%), EIRE (£265.3K, 2.98%), Germany (£228.7K, 2.57%), and France (£208.9K, 2.35%).
  - **Holiday Surge**: Sales surge to **£1.16M in November 2011** (+132% over annual baseline).

---

### Unsupervised Clustering Benchmark

Algorithms were systematically evaluated across $k \in [2, 6]$ using Within-Cluster Sum of Squares (Inertia), the Silhouette Coefficient, and the Davies-Bouldin Index:

| Algorithm Architecture | Clusters ($k$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means Baseline** | $k=2$ | 6,483.59 | 0.4328 | 0.8925 | High separation; binary active vs. inactive split (too coarse) |
| **K-Means** | $k=3$ | 4,869.49 | 0.3365 | 1.0483 | Identifies active/dormant; fails to distinguish high-yield VIPs |
| 🏆 **Champion K-Means** | **$k=4$** | **3,939.05** | **0.3375** | **1.0086** | **Optimal commercial granularity; isolates 4 distinct personas** |
| **K-Means** | $k=5$ | 3,296.71 | 0.3162 | 0.9878 | Diminishing returns; over-fragments regular repeaters |
| **K-Means** | $k=6$ | 2,855.76 | 0.3124 | 1.0210 | High complexity; increases campaign management overhead |
| **Agglomerative (Ward Linkage)** | $k=4$ | — | 0.2800 | 1.0992 | Confirms 4-branch dendrogram topology; lower silhouette |
| **DBSCAN ($\epsilon=0.75$, $MinPts=10$)** | 1 + Noise | — | N/A (Degenerate) | N/A (Degenerate) | Fails density test (lumps 4,303 into 1 cluster; 35 noise points) |

---

### Discovered Customer Personas & The Pareto Principle

> **Dimensionality & Cluster Stability**: PCA is utilized as an orthogonal 2D visual communication tool retaining 93.87% variance, while K-Means is trained on standardized 3D RFM space. Cluster stability was verified via an exceptional Adjusted Rand Index (ARI) of **0.9668** across random initialization seeds. "Lapsed / Inactive Customers" represents an operational observation-window classification rather than permanent churn.

Applying the champion $k=4$ model to the 4,338 customer base reveals four actionable commercial archetypes:

| Customer Persona | Headcount | Headcount Share | Mean Recency | Mean Frequency | Mean Spend | Mean AOV | Total Gross Revenue | Revenue Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **Champions (Whales)** | **709** | **16.34%** | **12.08 days** | **13.79 orders** | **£8,112.93** | **£623.83** | **£5,752,065.07** | **64.72%** |
| 💎 **Loyal Active Repeaters** | **1,173** | **27.04%** | **70.13 days** | **4.11 orders** | **£1,804.82** | **£571.27** | **£2,117,050.02** | **23.82%** |
| 🌱 **Promising Recent Shoppers** | **839** | **19.34%** | **18.21 days** | **2.14 orders** | **£543.06** | **£278.34** | **£455,624.56** | **5.13%** |
| 💤 **Lapsed / Inactive Customers** | **1,617** | **37.28%** | **182.63 days** | **1.32 orders** | **£347.85** | **£288.08** | **£562,469.24** | **6.33%** |
| **Enterprise Total** | **4,338** | **100.0%** | **92.54 days** | **4.27 orders** | **£2,048.69** | **£417.65** | **£8,887,208.89** | **100.0%** |

> **Empirical Validation of the Pareto Principle**:
> - **Top 43.38% of customers (Champions + Loyalists = 1,882 accounts) generate 88.54% of total gross sales (£7.87M of £8.89M)**.
> - Conversely, the **Lapsed / Inactive Customers** cohort represents **37.28% of customer headcount** but generates merely **6.33% of revenue**, demonstrating that retargeting ad spend must be drastically reallocated toward Champions retention and Promising nurture sequences.

---

### Prescriptive Strategic Playbooks
1. **Champions (Whales)**: Dedicated VIP account representatives, early product catalog access for Christmas lines, and tiered volume rebates (e.g. 5% credit on spend $>£10,000$).
2. **Loyal Active Repeaters**: Personalized cross-selling engines (association rule pairings), gamified milestone loyalty progression ("Order 5 times for free annual shipping").
3. **Promising Recent Shoppers**: 3-part educational email nurture series within 7 days, 15% second-purchase acceleration incentive valid for 21 days.
4. **Lapsed / Inactive Customers**: Automated win-back campaign with £10 gift voucher; 1-question exit survey; suppress ad spend after 3 unresponsive attempts.

---

### Detailed Directory Structure: `capstone_2_onlineretail/`

```text
capstone_2_onlineretail/
├── notebooks/
│   └── 01_retail_customer_segmentation.ipynb    # Master executable notebook (10 sections, deepmind kernel)
├── reports/
│   ├── Online_Retail_Customer_Segmentation_Report.pdf  # Formal 27-page PDF report (Draft.pdf format)
│   ├── Retail_Customer_Segmentation_Report.docx        # Editable Microsoft Word document with embedded plots
│   └── report_retail_customer_segmentation.md          # Master source markdown report with LaTeX math
├── models/
│   ├── retail_kmeans_champion_k4.joblib         # Champion K-Means model (k=4, random_state=42; 18.2 KB)
│   ├── retail_scaler_iter01.joblib              # Fitted StandardScaler transformer (927 bytes)
│   └── retail_pca_iter01.joblib                 # Fitted 2-component PCA transformer (1.08 KB)
├── plots/
│   ├── 01_top_products_revenue.png              # Top 10 revenue-generating products (300 DPI)
│   ├── 02_temporal_order_patterns.png           # Hourly and daily purchasing temporal dynamics
│   ├── 03_country_sales_distribution.png        # Domestic UK vs. International export distribution
│   ├── 04_monthly_revenue_trend.png             # Monthly revenue growth & Q4 holiday surge annotation
│   ├── 05_raw_rfm_distributions.png             # Raw RFM metric distributions & power-law skewness
│   ├── 06_log_transformed_rfm_distributions.png # Normalized log-transformed RFM distributions
│   ├── 07_pca_feature_loadings.png              # PCA explained variance & feature loading vectors
│   ├── 08_kmeans_elbow_silhouette.png           # K-Means Elbow Inertia curve & Silhouette trajectory
│   ├── 09_hierarchical_dendrogram.png           # Ward-linkage hierarchical clustering dendrogram
│   ├── 10_pca_customer_clusters.png             # 2D PCA customer cluster scatter with persona boundaries
│   ├── 11_cluster_persona_metrics.png           # Multi-metric comparison (Recency, Freq, Spend, AOV)
│   └── 12_segment_revenue_contribution.png      # Pareto imbalance: Headcount share vs. Revenue contribution
└── data_processed/
    ├── cleaned_transactions-stage01.csv         # Cleaned, authenticated commercial purchases (392,692 rows)
    ├── rfm_raw-features01.csv                   # Raw customer RFM-AOV feature matrix (4,338 customers)
    ├── rfm_transformed-cluster01.csv            # Log-transformed and standardized feature matrix
    ├── rfm_pca_projected-cluster02.csv          # 2D PCA coordinates (PC1, PC2) per customer
    └── segmented_customers-final01.csv          # Final entity table with assigned Personas and metrics
```

---

## 💻 Python Environment, Reproducibility & Automated Testing

### Environment Setup
- **Python Version**: `3.11.15`
- **Conda Environment**: `DL_AI_env_311` (or create via [`environment.yml`](./environment.yml))
- **Jupyter Kernel**: `deepmind` (`DeepMind (Python 3.11.15)`)
- **Dependency Specifications**: Pinned via [`environment.yml`](./environment.yml) and [`requirements.txt`](./requirements.txt)

To set up a clean reproducible environment:
```powershell
# Option A: Create fresh conda environment
conda env create -f environment.yml
conda activate DL_AI_env_311

# Option B: Install via pip
pip install -r requirements.txt
```

### Automated Unit & Regression Tests
The repository includes an automated regression test suite built on Python's native `unittest` framework (zero extra testing dependencies required):
- `tests/test_walmart_pipeline.py`: Verifies strict chronological split ordering (no overlap), absence of rolling window boundary leakage across stores, forecast output dimensions (540 rows across 45 stores), and validity of 90% prediction intervals.
- `tests/test_retail_pipeline.py`: Verifies gross revenue conservation (£8,887,208.89), RFM domain constraints ($R \ge 0, F \ge 1, M > 0$), 4-cluster completeness, and persona profile monotonicity.

Execute all automated tests via:
```powershell
python -m unittest discover tests
```
