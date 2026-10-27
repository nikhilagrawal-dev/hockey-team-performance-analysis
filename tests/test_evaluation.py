"""
Unit tests for Model Evaluation Lab (Phase 2B).
Verifies candidate models comparison, test metrics, baseline comparison,
season breakdowns, residual diagnostics, and chart builders.
"""

import unittest
import pandas as pd
import numpy as np

from src.config import (
    CANDIDATE_MODELS_EVALUATION,
    DATA_CLEANED_PATH,
    MODEL_PATH,
    METADATA_PATH,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
)
from src.data import load_cleaned_data, get_model_evaluation_data
from src.model import load_model_and_metadata
from src.charts import (
    build_cv_rmse_comparison_chart,
    build_test_rmse_comparison_chart,
    build_actual_vs_predicted_chart,
    build_residual_distribution_chart,
    build_error_by_season_chart,
)


class TestModelEvaluation(unittest.TestCase):
    """Test suite for candidate models, test metrics, and baseline comparisons."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_cleaned_data(DATA_CLEANED_PATH)
        cls.pipeline, cls.metadata, cls.error = load_model_and_metadata(MODEL_PATH, METADATA_PATH)

    def test_candidate_models_verified_values(self):
        """Verifies candidate models match the authoritative ML notebook v4."""
        self.assertEqual(len(CANDIDATE_MODELS_EVALUATION), 5)
        names = [m["model"] for m in CANDIDATE_MODELS_EVALUATION]
        self.assertIn("Linear Regression", names)
        self.assertIn("Ridge Regression", names)
        self.assertIn("Lasso Regression", names)
        self.assertIn("Decision Tree", names)
        self.assertIn("Random Forest", names)

        # Selected model check
        selected_models = [m for m in CANDIDATE_MODELS_EVALUATION if m["selected"]]
        self.assertEqual(len(selected_models), 1)
        lr = selected_models[0]
        self.assertEqual(lr["model"], "Linear Regression")
        self.assertAlmostEqual(lr["cv_rmse"], 0.0414, places=4)
        self.assertAlmostEqual(lr["cv_rmse_sd"], 0.0113, places=4)
        self.assertAlmostEqual(lr["test_rmse"], 0.0332, places=4)
        self.assertAlmostEqual(lr["test_mae"], 0.0264, places=4)
        self.assertAlmostEqual(lr["test_r2"], 0.8397, places=4)

    def test_get_model_evaluation_data_test_metrics(self):
        """Verifies test set metrics calculation on actual held-out test data (2008-2011)."""
        eval_data = get_model_evaluation_data(self.pipeline, self.df)
        test_df = eval_data["test_df"]
        test_metrics = eval_data["test_metrics"]

        # 120 held-out test records (30 teams * 4 seasons)
        self.assertEqual(len(test_df), 120)
        self.assertEqual(test_metrics["sample_size"], 120)

        # Check metric values
        self.assertAlmostEqual(test_metrics["mae"], 0.0264, places=3)
        self.assertAlmostEqual(test_metrics["rmse"], 0.0332, places=3)
        self.assertAlmostEqual(test_metrics["r2"], 0.8397, places=2)

    def test_get_model_evaluation_data_baseline(self):
        """Verifies training-mean baseline comparisons on held-out test data."""
        eval_data = get_model_evaluation_data(self.pipeline, self.df)
        base = eval_data["baseline_metrics"]

        # Training set (1990-2007) mean is ~0.4477
        self.assertAlmostEqual(base["train_mean"], 0.4477, places=3)
        self.assertAlmostEqual(base["mae"], 0.0817, places=3)
        self.assertAlmostEqual(base["rmse"], 0.0981, places=3)
        # Negative R2 due to post-lockout era shift
        self.assertLess(base["r2"], 0.0)

    def test_season_by_season_metrics_breakdown(self):
        """Verifies test seasons [2008, 2009, 2010, 2011] are evaluated individually."""
        eval_data = get_model_evaluation_data(self.pipeline, self.df)
        season_df = eval_data["season_metrics"]

        self.assertEqual(len(season_df), 4)
        seasons = season_df["Season"].tolist()
        self.assertEqual(seasons, [2008, 2009, 2010, 2011])
        for _, row in season_df.iterrows():
            self.assertEqual(row["Teams"], 30)
            self.assertGreater(row["MAE"], 0.02)
            self.assertLess(row["MAE"], 0.04)
            self.assertGreater(row["RMSE"], 0.025)
            self.assertLess(row["RMSE"], 0.045)

    def test_residual_diagnostics(self):
        """Verifies residual diagnostics including Durbin-Watson statistic."""
        eval_data = get_model_evaluation_data(self.pipeline, self.df)
        res_diag = eval_data["residual_diagnostics"]

        self.assertIn("mean", res_diag)
        self.assertIn("durbin_watson_test", res_diag)
        self.assertIn("durbin_watson_full_notebook", res_diag)

        # Test set DW should be near 1.8-1.9
        self.assertGreater(res_diag["durbin_watson_test"], 1.5)
        self.assertLess(res_diag["durbin_watson_test"], 2.2)
        # Full notebook DW is verified 1.0792
        self.assertAlmostEqual(res_diag["durbin_watson_full_notebook"], 1.0792, places=3)

    def test_fallback_on_empty_or_none(self):
        """Verifies graceful handling of empty data or None pipeline."""
        empty_res = get_model_evaluation_data(None, pd.DataFrame())
        self.assertEqual(len(empty_res["test_df"]), 0)
        self.assertEqual(empty_res["candidate_models"], CANDIDATE_MODELS_EVALUATION)

    def test_evaluation_chart_builders(self):
        """Verifies Altair evaluation charts compile properly."""
        eval_data = get_model_evaluation_data(self.pipeline, self.df)

        c1 = build_cv_rmse_comparison_chart(eval_data["candidate_models"])
        self.assertIsNotNone(c1)
        self.assertTrue(hasattr(c1, "to_dict"))

        c2 = build_test_rmse_comparison_chart(eval_data["candidate_models"])
        self.assertIsNotNone(c2)
        self.assertTrue(hasattr(c2, "to_dict"))

        c3 = build_actual_vs_predicted_chart(eval_data["test_df"])
        self.assertIsNotNone(c3)
        self.assertTrue(hasattr(c3, "to_dict"))

        c4 = build_residual_distribution_chart(eval_data["test_df"])
        self.assertIsNotNone(c4)
        self.assertTrue(hasattr(c4, "to_dict"))

        c5 = build_error_by_season_chart(eval_data["season_metrics"])
        self.assertIsNotNone(c5)
        self.assertTrue(hasattr(c5, "to_dict"))


if __name__ == "__main__":
    unittest.main()
