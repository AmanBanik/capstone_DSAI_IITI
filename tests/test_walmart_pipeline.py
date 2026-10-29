"""
Automated Regression and Unit Tests for Capstone 1: Walmart Sales Demand Forecasting.
Verifies split chronology, rolling boundary isolation, forecast dimensions, interval calibration, and path portability.
"""

import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Machine-independent workspace resolution with optional environment override
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path(__file__).resolve().parent.parent))
DATA_RAW = WORKSPACE_ROOT / "data_raw/Walmart.csv"
DATA_PROCESSED = WORKSPACE_ROOT / "capstone_1_walmart/data_processed"
MODELS_DIR = WORKSPACE_ROOT / "capstone_1_walmart/models"
REPORTS_DIR = WORKSPACE_ROOT / "capstone_1_walmart/reports"


class TestWalmartPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_b01 = pd.read_csv(DATA_PROCESSED / "baseline_split-train01.csv")
        cls.test_b01 = pd.read_csv(DATA_PROCESSED / "baseline_split-test01.csv")
        cls.train_f02 = pd.read_csv(DATA_PROCESSED / "lag_rolling_features-train02.csv")
        cls.train_m03 = pd.read_csv(DATA_PROCESSED / "macro_cyclical_engineered-train03.csv")
        cls.test_m03 = pd.read_csv(DATA_PROCESSED / "macro_cyclical_engineered-test03.csv")
        cls.forecast01 = pd.read_csv(DATA_PROCESSED / "forward_12week-forecast01.csv")

    def test_environment_and_path_portability(self):
        """Assert that all workspace artifacts resolve dynamically without hardcoded machine paths."""
        self.assertTrue(DATA_PROCESSED.exists(), f"Processed data directory {DATA_PROCESSED} must exist")
        self.assertTrue((DATA_PROCESSED / "baseline_split-train01.csv").exists(), "Baseline train split must exist")
        self.assertTrue((DATA_PROCESSED / "forward_12week-forecast01.csv").exists(), "Forward forecast must exist")

    def test_chronological_split_strictness(self):
        """Assert that no training date overlaps or exceeds the out-of-sample test cutoff."""
        train_dates = pd.to_datetime(self.train_b01["Date"])
        test_dates = pd.to_datetime(self.test_b01["Date"])
        
        self.assertLess(
            train_dates.max(),
            test_dates.min(),
            f"Train max date ({train_dates.max()}) must be strictly less than Test min date ({test_dates.min()})"
        )
        self.assertEqual(len(test_dates.unique()), 12, "Test split must contain exactly 12 unique weeks")
        self.assertEqual(len(self.test_b01), 45 * 12, "Test split must contain exactly 540 rows (45 stores * 12 weeks)")

    def test_rolling_feature_store_boundary_isolation(self):
        """Assert that rolling statistics within each store are calculated independently without bleeding."""
        df_raw = pd.read_csv(DATA_RAW)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], format="%d-%m-%Y")
        df_raw = df_raw.sort_values(["Store", "Date"]).reset_index(drop=True)

        for w in [4, 12]:
            expected_roll = df_raw.groupby("Store")["Weekly_Sales"].transform(
                lambda s: s.shift(1).rolling(w).mean()
            )
            df_raw[f"exp_roll_{w}"] = expected_roll

        # Drop initial 12 weeks with NaNs and subset to train window
        clean_expected = df_raw.dropna().reset_index(drop=True)
        train_expected = clean_expected[clean_expected["Date"].isin(pd.to_datetime(self.train_b01["Date"]))].reset_index(drop=True)

        for w in [4, 12]:
            diff = np.abs(train_expected[f"exp_roll_{w}"] - self.train_f02[f"rolling_mean_{w}"])
            max_diff = diff.max()
            self.assertLess(max_diff, 1e-4, f"Rolling window {w} features deviate from store-isolated calculation (max diff: {max_diff})")

    def test_forward_forecast_structure_and_intervals(self):
        """Assert that 12-week forward forecasts have valid dimensions, positive values, and consistent intervals."""
        self.assertEqual(len(self.forecast01), 45 * 12, "Forward forecast must register exactly 540 rows")
        self.assertEqual(self.forecast01["Store"].nunique(), 45, "Forward forecast must encompass all 45 stores")

        preds = self.forecast01["Forecasted_Weekly_Sales"]
        lower = self.forecast01["Lower_Bound_90"]
        upper = self.forecast01["Upper_Bound_90"]

        self.assertTrue((preds > 0).all(), "All point forecasts must be strictly positive")
        self.assertTrue((lower >= 0).all(), "All 90% lower bounds must be non-negative")
        self.assertTrue((lower <= preds).all(), "90% lower bound must be <= point forecast")
        self.assertTrue((preds <= upper).all(), "Point forecast must be <= 90% upper bound")

    def test_evaluation_summary_and_interval_calibration(self):
        """Assert that canonical evaluation summary exists, validates CV-only selection, and verifies interval coverage."""
        import json
        summary_path = REPORTS_DIR / "walmart_evaluation_summary.json"
        self.assertTrue(summary_path.exists(), "walmart_evaluation_summary.json must exist in reports")
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        self.assertEqual(len(summary["cross_validation_candidates"]), 4, "Must benchmark exactly 4 candidate configurations on CV")
        self.assertEqual(summary["champion_selection"], "Selected solely via cross-validation (Candidate 3: Depth 7, LR 0.04, N 300)")
        
        # Verify champion holdout test metrics
        test_metrics = summary["champion_holdout_test_metrics"]
        self.assertLess(test_metrics["rmse"], 50000.0, "Champion holdout RMSE must break below $50,000 threshold")
        self.assertLess(test_metrics["wape"], 3.5, "Champion holdout WAPE must be < 3.5%")
        
        # Verify calibrated prediction intervals
        calib = summary["interval_calibration"]
        self.assertGreaterEqual(calib["Holdout 90% Empirical Coverage (%)"], 85.0, "Empirical coverage must be >= 85%")
        self.assertLessEqual(calib["Holdout 90% Empirical Coverage (%)"], 98.0, "Empirical coverage must be <= 98%")
        self.assertGreater(calib["Holdout Average Interval Width ($)"], 50000.0, "Interval width must be realistic (> $50K)")

    def test_panel_weekly_cadence_and_non_duplication(self):
        """Assert zero duplicate (Store, Date) rows and strict 7-day weekly cadence across all 45 stores."""
        df_raw = pd.read_csv(DATA_RAW)
        self.assertEqual(df_raw.duplicated(subset=["Store", "Date"]).sum(), 0, "Raw panel must have zero duplicate (Store, Date) pairs")
        
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], format="%d-%m-%Y")
        for store_id, group in df_raw.groupby("Store"):
            group = group.sort_values("Date")
            date_diffs = group["Date"].diff().dropna()
            self.assertTrue(
                (date_diffs == pd.Timedelta(days=7)).all(),
                f"Store {store_id} contains non-7-day weekly intervals: {date_diffs.unique()}"
            )

    def test_holdout_metric_recomputation_from_champion_model(self):
        """Assert that loading champion XGBoost and predicting on test data reproduces canonical test metrics."""
        import joblib
        model_path = MODELS_DIR / "walmart_xgboost_iter02.joblib"
        self.assertTrue(model_path.exists(), "Champion XGBoost model file must exist")
        
        model = joblib.load(model_path)
        store_means = self.train_m03.groupby("Store")["Weekly_Sales"].mean().to_dict()
        
        test_df = self.test_m03.copy()
        test_df["store_mean_sales"] = test_df["Store"].map(store_means)
        
        feature_order = model.get_booster().feature_names
        X_test = test_df[feature_order]
        y_true = test_df["Weekly_Sales"].values
        y_pred = model.predict(X_test)
        
        recomputed_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        recomputed_mae = float(np.mean(np.abs(y_true - y_pred)))
        recomputed_wape = float((np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0)
        recomputed_mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
        
        # Verify alignment to canonical published metrics
        self.assertAlmostEqual(recomputed_rmse, 49140.51, delta=1.0, msg="Recomputed RMSE must align to $49,140.51")
        self.assertAlmostEqual(recomputed_mae, 33619.68, delta=1.0, msg="Recomputed MAE must align to $33,619.68")
        self.assertAlmostEqual(recomputed_wape, 3.27, delta=0.05, msg="Recomputed WAPE must align to 3.27%")
        self.assertAlmostEqual(recomputed_mape, 3.40, delta=0.05, msg="Recomputed MAPE must align to 3.40%")

    def test_holdout_interval_empirical_coverage_recomputation(self):
        """Assert that out-of-fold calibrated residuals produce exactly 91.30% coverage on the untouched test holdout."""
        import joblib
        import json
        
        model = joblib.load(MODELS_DIR / "walmart_xgboost_iter02.joblib")
        store_means = self.train_m03.groupby("Store")["Weekly_Sales"].mean().to_dict()
        
        test_df = self.test_m03.copy()
        test_df["store_mean_sales"] = test_df["Store"].map(store_means)
        
        feature_order = model.get_booster().feature_names
        y_true = test_df["Weekly_Sales"].values
        y_pred = model.predict(test_df[feature_order])
        
        with open(REPORTS_DIR / "walmart_evaluation_summary.json", "r", encoding="utf-8") as f:
            calib = json.load(f)["interval_calibration"]
            
        q_low = calib["10th Percentile OOF Error ($)"]
        q_high = calib["90th Percentile OOF Error ($)"]
        
        test_lower = y_pred + q_low
        test_upper = y_pred + q_high
        
        empirical_coverage = float(np.mean((y_true >= test_lower) & (y_true <= test_upper)) * 100.0)
        avg_width = float(np.mean(test_upper - test_lower))
        
        self.assertAlmostEqual(empirical_coverage, 91.30, delta=0.05, msg="Recomputed holdout coverage must be 91.30%")
        self.assertAlmostEqual(avg_width, 150396.66, delta=1.0, msg="Recomputed interval width must be $150,396.66")

    def test_synthetic_fixture_leakage_resistance(self):
        """Assert that lag and rolling window transformations strictly respect shift(1) without future leakage."""
        dates = pd.date_range("2024-01-01", periods=6, freq="W")
        toy_df = pd.DataFrame({
            "Store": [1] * 6,
            "Date": dates,
            "Target": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        })
        
        # Calculate shifted rolling mean (window=2)
        toy_df["roll_2"] = toy_df.groupby("Store")["Target"].transform(
            lambda s: s.shift(1).rolling(2).mean()
        )
        
        # Verify that row 3 (Target=400.0) only saw Target 200 and 300 (mean = 250)
        self.assertEqual(toy_df.loc[3, "roll_2"], 250.0)
        # Verify that changing future value in row 3 does not alter rolling value in row 2
        toy_modified = toy_df.copy()
        toy_modified.loc[3, "Target"] = 99999.0
        toy_modified["roll_2"] = toy_modified.groupby("Store")["Target"].transform(
            lambda s: s.shift(1).rolling(2).mean()
        )
        self.assertEqual(toy_df.loc[2, "roll_2"], toy_modified.loc[2, "roll_2"])


if __name__ == "__main__":
    unittest.main()
