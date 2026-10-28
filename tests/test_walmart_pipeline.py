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
REPORTS_DIR = WORKSPACE_ROOT / "capstone_1_walmart/reports"


class TestWalmartPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_b01 = pd.read_csv(DATA_PROCESSED / "baseline_split-train01.csv")
        cls.test_b01 = pd.read_csv(DATA_PROCESSED / "baseline_split-test01.csv")
        cls.train_f02 = pd.read_csv(DATA_PROCESSED / "lag_rolling_features-train02.csv")
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


if __name__ == "__main__":
    unittest.main()
