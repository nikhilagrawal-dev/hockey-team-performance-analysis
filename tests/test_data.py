"""
Automated Unit Tests: Dataset Integrity & Schema Validation
Verifies historical records count, core column types, target distributions,
and preservation of pre-1999 unrecorded OT losses.
"""

import unittest
import pandas as pd
from src.config import (
    HISTORICAL_TOTAL_RECORDS,
    HISTORICAL_MIN_YEAR,
    HISTORICAL_MAX_YEAR,
    HISTORICAL_LOCKOUT_YEAR,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
    HISTORICAL_MIN_WIN_PCT,
    HISTORICAL_MAX_WIN_PCT,
)
from src.data import (
    load_cleaned_data,
    load_raw_data,
    validate_dataset_schema,
    get_historical_summary_stats,
    get_unique_teams,
    get_unique_years,
)


class TestDataIntegrity(unittest.TestCase):
    """Test suite for NHL team-seasons dataset validation."""

    @classmethod
    def setUpClass(cls):
        """Loads dataset once for the test case."""
        cls.df_clean = load_cleaned_data()
        cls.df_raw = load_raw_data()

    def test_cleaned_dataset_row_and_column_count(self):
        """Cleaned dataset must have exactly 582 rows and at least 10 expected columns."""
        self.assertEqual(
            len(self.df_clean),
            HISTORICAL_TOTAL_RECORDS,
            f"Expected {HISTORICAL_TOTAL_RECORDS} rows in cleaned CSV, got {len(self.df_clean)}"
        )
        self.assertGreaterEqual(len(self.df_clean.columns), 10)

    def test_raw_dataset_row_count(self):
        """Raw scraped dataset must have exactly 582 rows and 9 columns."""
        self.assertEqual(len(self.df_raw), HISTORICAL_TOTAL_RECORDS)
        self.assertEqual(len(self.df_raw.columns), 9)

    def test_required_columns_exist(self):
        """All model features and target column must exist in cleaned dataset."""
        for feat in REQUIRED_FEATURES:
            self.assertIn(feat, self.df_clean.columns, f"Feature '{feat}' missing from cleaned dataset.")
        self.assertIn(TARGET_COLUMN, self.df_clean.columns, f"Target '{TARGET_COLUMN}' missing from cleaned dataset.")

    def test_preservation_of_unrecorded_ot_losses(self):
        """
        Historical integrity test:
        OT_Losses must have exactly 224 missing/null values corresponding to the pre-1999 era.
        They must NOT be imputed as zero.
        """
        null_counts = self.df_clean.isnull().sum()
        self.assertEqual(
            null_counts["OT_Losses"],
            224,
            f"Expected exactly 224 missing OT_Losses values, found {null_counts['OT_Losses']}"
        )
        # Verify that all 224 nulls are strictly for years < 1999
        null_years = self.df_clean[self.df_clean["OT_Losses"].isnull()]["Year"]
        self.assertTrue(
            (null_years < 1999).all(),
            "Missing OT_Losses values must only exist in pre-1999 seasons when ties were recorded."
        )

    def test_core_columns_have_zero_nulls(self):
        """Team_Name, Year, Wins, Losses, Win_Pct, Goals_For, Goals_Against must have zero nulls."""
        core_cols = ["Team_Name", "Year", "Wins", "Losses", TARGET_COLUMN, "Goals_For", "Goals_Against", "Goal_Diff"]
        for col in core_cols:
            self.assertEqual(
                self.df_clean[col].isnull().sum(),
                0,
                f"Column '{col}' unexpectedly has null values in cleaned dataset."
            )

    def test_target_variable_bounds(self):
        """Win_Pct values must fall within the verified historical range [0.119, 0.756]."""
        min_win_pct = self.df_clean[TARGET_COLUMN].min()
        max_win_pct = self.df_clean[TARGET_COLUMN].max()

        self.assertAlmostEqual(min_win_pct, HISTORICAL_MIN_WIN_PCT / 100.0, places=3)
        self.assertAlmostEqual(max_win_pct, HISTORICAL_MAX_WIN_PCT / 100.0, places=3)
        self.assertTrue((self.df_clean[TARGET_COLUMN] >= 0.0).all())
        self.assertTrue((self.df_clean[TARGET_COLUMN] <= 1.0).all())

    def test_year_range_and_lockout_season(self):
        """Dataset spans 1990 to 2011; calendar year 2004 must be strictly absent."""
        years = get_unique_years(self.df_clean)
        self.assertEqual(min(years), HISTORICAL_MIN_YEAR)
        self.assertEqual(max(years), HISTORICAL_MAX_YEAR)
        self.assertNotIn(
            HISTORICAL_LOCKOUT_YEAR,
            years,
            f"Lockout year {HISTORICAL_LOCKOUT_YEAR} must not be present in historical records."
        )

    def test_franchise_count(self):
        """Dataset must contain exactly 35 distinct franchise names across 1990-2011."""
        teams = get_unique_teams(self.df_clean)
        self.assertEqual(len(teams), 35)

    def test_schema_validator_function(self):
        """validate_dataset_schema helper must report valid=True and 0 duplicate rows."""
        report = validate_dataset_schema(self.df_clean)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["row_count"], HISTORICAL_TOTAL_RECORDS)
        self.assertEqual(report["ot_losses_missing"], 224)

    def test_historical_summary_stats(self):
        """Summary stats calculation must accurately compute means and ranges."""
        stats = get_historical_summary_stats(self.df_clean)
        self.assertEqual(stats["total_records"], 582)
        self.assertEqual(stats["total_teams"], 35)
        # Symmetrical league scoring property
        self.assertAlmostEqual(stats["goals_for_mean"], stats["goals_against_mean"], places=2)


if __name__ == "__main__":
    unittest.main()
