"""
Automated Regression and Unit Tests for Capstone 2: Online Retail Customer Segmentation.
Verifies gross revenue conservation, RFM domain constraints, cluster completeness, and profile monotonicity.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path("C:/Devfield/capstone_DSAI")
DATA_PROCESSED = WORKSPACE_ROOT / "capstone_2_onlineretail/data_processed"


class TestRetailPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cleaned_tx = pd.read_csv(DATA_PROCESSED / "cleaned_transactions-stage01.csv")
        cls.rfm_raw = pd.read_csv(DATA_PROCESSED / "rfm_raw-features01.csv")
        cls.segmented = pd.read_csv(DATA_PROCESSED / "segmented_customers-final01.csv")

    def test_gross_revenue_conservation(self):
        """Assert that total monetary sum across all customer entities matches the transaction log."""
        tx_total = self.cleaned_tx["TotalAmount"].sum()
        rfm_total = self.rfm_raw["Monetary"].sum()
        seg_total = self.segmented["Monetary"].sum()

        self.assertAlmostEqual(tx_total, rfm_total, places=2, msg="RFM total spend must match transaction total spend")
        self.assertAlmostEqual(rfm_total, seg_total, places=2, msg="Segmented total spend must match raw RFM total spend")
        self.assertAlmostEqual(tx_total, 8887208.89, places=1, msg="Gross enterprise sales must align to £8,887,208.89")

    def test_rfm_domain_validity(self):
        """Assert that Recency, Frequency, and Monetary values are within valid retail domains."""
        self.assertEqual(len(self.segmented), 4338, "Segmented customer table must contain exactly 4,338 customer records")
        self.assertTrue((self.segmented["Recency"] >= 0).all(), "Recency days must be non-negative")
        self.assertTrue((self.segmented["Frequency"] >= 1).all(), "Frequency must be at least 1 invoice")
        self.assertTrue((self.segmented["Monetary"] > 0).all(), "Monetary spend must be strictly positive")
        self.assertTrue((self.segmented["AOV"] > 0).all(), "Average order value must be strictly positive")

    def test_cluster_completeness_and_persona_mapping(self):
        """Assert that exactly 4 clusters exist and all records are mapped to one of the 4 designated personas."""
        expected_personas = {
            "Champions (Whales)",
            "Loyal Active Repeaters",
            "Promising Recent Shoppers",
            "Lapsed / Inactive Customers"
        }
        actual_personas = set(self.segmented["Persona"].unique())
        self.assertEqual(actual_personas, expected_personas, f"Discovered personas {actual_personas} must match {expected_personas}")
        self.assertEqual(self.segmented["Persona"].isnull().sum(), 0, "No customer persona may be null")
        self.assertEqual(self.segmented["Cluster"].nunique(), 4, "Must contain exactly 4 distinct clusters")

    def test_persona_profile_monotonicity(self):
        """Assert that Champions have higher average spend and frequency than Inactive accounts."""
        champs = self.segmented[self.segmented["Persona"] == "Champions (Whales)"]
        inactive = self.segmented[self.segmented["Persona"] == "Lapsed / Inactive Customers"]

        self.assertGreater(
            champs["Monetary"].mean(),
            inactive["Monetary"].mean(),
            "Champions mean spend must strictly exceed Lapsed/Inactive mean spend"
        )
        self.assertGreater(
            champs["Frequency"].mean(),
            inactive["Frequency"].mean(),
            "Champions mean frequency must strictly exceed Lapsed/Inactive mean frequency"
        )
        self.assertLess(
            champs["Recency"].median(),
            inactive["Recency"].median(),
            "Champions median recency must be significantly fresher (lower) than Lapsed/Inactive recency"
        )


if __name__ == "__main__":
    unittest.main()
