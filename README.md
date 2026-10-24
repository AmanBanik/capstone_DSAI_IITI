# Capstone DSAI Projects

Initial workspace and baseline setup for two Data Science & Machine Learning Capstone projects:

1. **Capstone 1: Walmart Store Sales Demand Forecasting**
   - Outlets demand & supply forecasting across 45 stores nationwide.
   - 12-week forward sales forecasting per store incorporating holiday dynamics and macroeconomic indicators (CPI, Unemployment, Fuel Price).
   - Core Approach: Classical ML & Statistical Time-Series (ARIMA/SARIMAX, XGBoost, Random Forest).
   - Directory: [`capstone_1_walmart/`](./capstone_1_walmart/)

2. **Capstone 2: Online Retail Customer Purchasing Behavior & Segmentation**
   - Customer purchasing patterns, transaction frequency, and customer segmentation for an online retailer.
   - Core Approach: RFM (Recency, Frequency, Monetary) analytics, data preprocessing, and unsupervised ML clustering (K-Means, Agglomerative/Hierarchical Clustering, DBSCAN, PCA).
   - Directory: [`capstone_2_onlineretail/`](./capstone_2_onlineretail/)

---

## Workspace Structure

```text
capstone_DSAI/
├── capstone_1_walmart/
│   ├── notebooks/             # Starter and experiment notebooks (.ipynb)
│   ├── data_processed/        # Model-ready data: <preprocess-type>-<stage><number>.csv
│   ├── models/                # Saved models, scalers, and pipelines (.joblib)
│   ├── plots/                 # Saved visualizations (.png)
│   └── reports/               # Submission drafts and reports (PDF)
├── capstone_2_onlineretail/
│   ├── notebooks/             # Starter and experiment notebooks (.ipynb)
│   ├── data_processed/        # Model-ready data: <preprocess-type>-<stage><number>.csv
│   ├── models/                # Saved models, scalers, and pipelines (.joblib)
│   ├── plots/                 # Saved visualizations (.png)
│   └── reports/               # Submission drafts and reports (PDF)
├── data_raw/                  # Raw immutable datasets (Walmart.csv, OnlineRetail.csv)
├── problem-statements-and-report-draft/
│   ├── Draft.pdf              # 12-section report structure template
│   ├── guideline.md           # Submission guidelines
│   ├── problem 1.md           # Problem 1 specification
│   └── problem 2.md           # Problem 2 specification
└── README.md
```

---

## Python Environment

- **Python**: 3.11.15
- **Conda Environment**: `DL_AI_env_311` (Kernel: `deepmind`)
- **Core ML Stack**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `prophet`, `xgboost`, `matplotlib`, `seaborn`, `joblib`
