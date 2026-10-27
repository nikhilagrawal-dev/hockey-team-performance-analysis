"""
Automated Unit Tests: ML Pipeline Inference, Consistency & Explainability
Verifies model loading, exact feature ordering, preset reproduction,
mathematical contribution decomposition, and input boundary validation.
"""

import unittest
import numpy as np
from src.config import (
    REQUIRED_FEATURES,
    PRESET_SCENARIOS,
    HISTORICAL_MIN_WIN_PCT,
    HISTORICAL_MAX_WIN_PCT,
)
from src.model import (
    load_model_and_metadata,
    validate_contribution_breakdown,
    validate_prediction_inputs,
    predict_win_pct,
    calculate_feature_contributions,
)


class TestModelPipeline(unittest.TestCase):
    """Test suite for the fitted Linear Regression pipeline and explainability engine."""

    @classmethod
    def setUpClass(cls):
        """Loads model and metadata once for test suite."""
        cls.pipeline, cls.metadata, cls.load_error = load_model_and_metadata()

    def test_model_loaded_successfully(self):
        """Model artifact must deserialize without errors."""
        self.assertIsNone(self.load_error, f"Model loading failed: {self.load_error}")
        self.assertIsNotNone(self.pipeline, "Pipeline object is None.")
        self.assertIsInstance(self.metadata, dict, "Metadata is not a valid dictionary.")

    def test_pipeline_architecture(self):
        """Pipeline must contain exactly 'scaler' (StandardScaler) and 'model' (LinearRegression)."""
        self.assertTrue(hasattr(self.pipeline, "named_steps"))
        self.assertIn("scaler", self.pipeline.named_steps)
        self.assertIn("model", self.pipeline.named_steps)
        self.assertEqual(self.pipeline.named_steps["scaler"].__class__.__name__, "StandardScaler")
        self.assertEqual(self.pipeline.named_steps["model"].__class__.__name__, "LinearRegression")

    def test_exact_feature_ordering(self):
        """Pipeline and metadata must strictly enforce ['Goals_For', 'Goals_Against', 'Year']."""
        scaler = self.pipeline.named_steps["scaler"]
        if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
            self.assertEqual(list(scaler.feature_names_in_), REQUIRED_FEATURES)

        meta_features = self.metadata.get("feature_columns")
        self.assertEqual(meta_features, REQUIRED_FEATURES)

    def test_learned_coefficients_and_intercept(self):
        """
        Coefficients must reflect hockey logic:
        Positive coefficient for Goals_For, negative for Goals_Against.
        """
        model = self.pipeline.named_steps["model"]
        coefs = model.coef_
        intercept = model.intercept_

        # GF coefficient > 0 (scoring increases win %)
        self.assertGreater(coefs[0], 0.08, "Goals_For coefficient must be positive (~ +0.093)")
        # GA coefficient < 0 (conceding decreases win %)
        self.assertLess(coefs[1], -0.07, "Goals_Against coefficient must be negative (~ -0.085)")
        # Intercept ~ 0.4477
        self.assertAlmostEqual(intercept, 0.4477, places=3)

    def test_preset_scenarios_reproduction(self):
        """
        Inference test on verified canonical scenarios:
        Model predictions must match established benchmark outputs within 0.0005.
        """
        for preset_name, preset_data in PRESET_SCENARIOS.items():
            gf = preset_data["goals_for"]
            ga = preset_data["goals_against"]
            yr = preset_data["year"]
            expected_win = preset_data["expected_win_pct"]

            pred = predict_win_pct(self.pipeline, gf, ga, yr)
            self.assertAlmostEqual(
                pred,
                expected_win,
                places=3,
                msg=f"Preset '{preset_name}' yielded {pred:.4f}, expected ~{expected_win:.4f}"
            )

    def test_feature_contribution_mathematical_identity(self):
        """
        Mathematical proof:
        Reconstructed prediction: intercept + sum(z_i * beta_i)
        must exactly equal pipeline.predict() within 1e-5 floating-point tolerance.
        """
        test_cases = [
            (250, 230, 2009),
            (310, 190, 2011),
            (180, 280, 1995),
            (234, 234, 2006),
        ]
        for gf, ga, yr in test_cases:
            result = calculate_feature_contributions(self.pipeline, self.metadata, gf, ga, yr)
            self.assertTrue(result["is_valid"])

            reconstructed = result["reconstructed_prediction"]
            direct = result["direct_prediction"]

            diff = abs(reconstructed - direct)
            self.assertLess(
                diff,
                1e-5,
                f"Decomposition sum ({reconstructed}) does not match direct predict ({direct}); diff={diff}"
            )

    def test_input_validation_errors(self):
        """Input validator must reject non-positive goals, strings, and out-of-range years."""
        # Negative Goals For
        errs1 = validate_prediction_inputs(-10, 200, 2010)
        self.assertGreater(len(errs1), 0)
        self.assertIn("Goals For must be greater than 0.", errs1[0])

        # Zero Goals Against
        errs2 = validate_prediction_inputs(200, 0, 2010)
        self.assertGreater(len(errs2), 0)
        self.assertIn("Goals Against must be greater than 0.", errs2[0])

        # Non-numeric string
        errs3 = validate_prediction_inputs("abc", 200, 2010)
        self.assertGreater(len(errs3), 0)

        # Implausible year
        errs4 = validate_prediction_inputs(250, 230, 1850)
        self.assertGreater(len(errs4), 0)

    def test_unclamped_linear_extrapolation_behavior(self):
        """
        Scientific integrity constraint:
        OLS regressor must NOT silently clamp extreme predictions.
        Extreme lopsided inputs (e.g. 50 GF and 450 GA) should produce an unclamped estimate.
        """
        extreme_low = predict_win_pct(self.pipeline, 50, 450, 1992)
        # 50 GF vs 450 GA will yield a negative unconstrained estimate
        self.assertLess(extreme_low, 0.0, "Model should produce unclamped negative estimate for extreme input.")

        extreme_high = predict_win_pct(self.pipeline, 500, 100, 2011)
        # 500 GF vs 100 GA will yield an unclamped estimate > 1.0
        self.assertGreater(extreme_high, 1.0, "Model should produce unclamped >1.0 estimate for extreme high input.")


if __name__ == "__main__":
    unittest.main()
