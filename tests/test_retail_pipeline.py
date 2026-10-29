"""
Automated Regression and Unit Tests for Capstone 2: Online Retail Customer Segmentation.
Verifies gross revenue conservation, RFM domain constraints, cluster completeness, persona ordering,
reproducibility across reloaded model serializations, cluster stability across random seeds, and path portability.
"""

import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

# Machine-independent workspace resolution with optional environment override
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path(__file__).resolve().parent.parent))
DATA_PROCESSED = WORKSPACE_ROOT / "capstone_2_onlineretail/data_processed"
MODELS_DIR = WORKSPACE_ROOT / "capstone_2_onlineretail/models"
REPORTS_DIR = WORKSPACE_ROOT / "capstone_2_onlineretail/reports"


class TestRetailPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cleaned_tx = pd.read_csv(DATA_PROCESSED / "cleaned_transactions-stage01.csv")
        cls.rfm_raw = pd.read_csv(DATA_PROCESSED / "rfm_raw-features01.csv")
        cls.segmented = pd.read_csv(DATA_PROCESSED / "segmented_customers-final01.csv")
        cls.scaler = joblib.load(MODELS_DIR / "retail_scaler_iter01.joblib")
        cls.champion_kmeans = joblib.load(MODELS_DIR / "retail_kmeans_champion_k4.joblib")

    def test_environment_and_path_portability(self):
        """Assert that all workspace artifacts resolve dynamically without hardcoded machine paths."""
        self.assertTrue(DATA_PROCESSED.exists(), f"Processed data directory {DATA_PROCESSED} must exist")
        self.assertTrue((DATA_PROCESSED / "cleaned_transactions-stage01.csv").exists(), "Cleaned transactions must exist")
        self.assertTrue((MODELS_DIR / "retail_kmeans_champion_k4.joblib").exists(), "Champion model weights must exist")

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

    def test_persona_profile_hierarchy_and_business_ordering(self):
        """Assert that all four commercial personas satisfy strict multidimensional behavioral ordering."""
        champs = self.segmented[self.segmented["Persona"] == "Champions (Whales)"]
        loyal = self.segmented[self.segmented["Persona"] == "Loyal Active Repeaters"]
        promising = self.segmented[self.segmented["Persona"] == "Promising Recent Shoppers"]
        inactive = self.segmented[self.segmented["Persona"] == "Lapsed / Inactive Customers"]

        # Spend Monotonicity: Champions > Loyal > Promising > Inactive
        self.assertGreater(champs["Monetary"].mean(), loyal["Monetary"].mean())
        self.assertGreater(loyal["Monetary"].mean(), promising["Monetary"].mean())
        self.assertGreater(promising["Monetary"].mean(), inactive["Monetary"].mean())

        # Frequency Hierarchy: Champions > Loyal > Promising / Inactive
        self.assertGreater(champs["Frequency"].mean(), loyal["Frequency"].mean())
        self.assertGreater(loyal["Frequency"].mean(), promising["Frequency"].mean())

        # Recency Dynamics: Champions and Promising are active; Inactive are decayed
        self.assertLess(champs["Recency"].median(), 25, "Champions must have median recency < 25 days")
        self.assertLess(promising["Recency"].median(), 40, "Promising shoppers must have median recency < 40 days")
        self.assertGreater(inactive["Recency"].median(), 120, "Inactive customers must have median recency > 120 days")
        self.assertGreater(inactive["Recency"].mean(), 150, "Inactive customers must have mean recency > 150 days")

    def test_model_reloading_reproducibility(self):
        """Assert that reloaded StandardScaler and KMeans pipelines exactly reproduce assigned cluster labels."""
        feature_names = ["Recency_Log", "Frequency_Log", "Monetary_Log"]
        log_rfm = pd.DataFrame(
            np.log1p(self.segmented[["Recency", "Frequency", "Monetary"]].values),
            columns=feature_names
        )
        scaled_rfm = self.scaler.transform(log_rfm)
        
        reproduced_labels = self.champion_kmeans.predict(scaled_rfm)
        match_rate = np.mean(reproduced_labels == self.segmented["Cluster"].values)
        self.assertEqual(match_rate, 1.0, f"Reloaded model reproduction rate must be exactly 100% (got {match_rate*100}%)")

    def test_cluster_stability_across_initialization_seeds(self):
        """Assert that K-Means clustering demonstrates high topological stability (ARI >= 0.90) across random seeds."""
        feature_names = ["Recency_Log", "Frequency_Log", "Monetary_Log"]
        log_rfm = pd.DataFrame(
            np.log1p(self.segmented[["Recency", "Frequency", "Monetary"]].values),
            columns=feature_names
        )
        scaled_rfm = self.scaler.transform(log_rfm)
        base_labels = self.segmented["Cluster"].values

        ari_scores = []
        for seed in [10, 42, 99, 123, 777]:
            km_test = KMeans(n_clusters=4, init="k-means++", n_init=15, random_state=seed)
            test_preds = km_test.fit_predict(scaled_rfm)
            ari = adjusted_rand_score(base_labels, test_preds)
            ari_scores.append(ari)

        mean_ari = np.mean(ari_scores)
        self.assertGreaterEqual(mean_ari, 0.90, f"Mean Adjusted Rand Index across initialization seeds must be >= 0.90 (got {mean_ari:.4f})")

    def test_evaluation_summary_and_lineage_audit(self):
        """Assert that canonical evaluation summary exists and matches single source of truth figures."""
        import json
        summary_path = REPORTS_DIR / "retail_evaluation_summary.json"
        self.assertTrue(summary_path.exists(), "retail_evaluation_summary.json must exist in reports")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        lineage = summary["data_hygiene_lineage"]
        self.assertEqual(lineage["raw_transactions"], 541909)
        self.assertEqual(lineage["clean_commercial_transactions"], 392692)
        self.assertEqual(lineage["unique_commercial_customers"], 4338)
        self.assertAlmostEqual(lineage["gross_enterprise_revenue_gbp"], 8887208.89, places=2)

        pareto = summary["pareto_proof"]
        self.assertAlmostEqual(pareto["power_tier_revenue_share_pct"], 88.54, places=1)
        self.assertAlmostEqual(pareto["power_tier_headcount_share_pct"], 43.38, places=1)

    def test_pca_artifact_and_variance_decomposition(self):
        """Assert that PCA transformer is valid and preserves 93.87% cumulative explained variance across 2 components."""
        pca_path = MODELS_DIR / "retail_pca_iter01.joblib"
        self.assertTrue(pca_path.exists(), "retail_pca_iter01.joblib must exist")
        pca = joblib.load(pca_path)
        self.assertEqual(pca.n_components, 2, "PCA must preserve exactly 2 principal components for visualization")
        
        cum_var = float(np.sum(pca.explained_variance_ratio_) * 100.0)
        self.assertAlmostEqual(cum_var, 93.87, delta=0.05, msg="PCA cumulative variance must equal 93.87%")
        self.assertAlmostEqual(float(pca.explained_variance_ratio_[0] * 100.0), 75.08, delta=0.05)
        self.assertAlmostEqual(float(pca.explained_variance_ratio_[1] * 100.0), 18.79, delta=0.05)


if __name__ == "__main__":
    unittest.main()
