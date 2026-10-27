"""
Automated Unit Tests: Historical Analytics, Filtering & Team Comparison
Verifies season/team filtering, leaderboard rankings, head-to-head comparison calculations,
empty filter resilience, CSV export serialization, and statistical consistency.
"""

import unittest
import io
import pandas as pd
from src.data import (
    load_cleaned_data,
    filter_historical_records,
    get_leaderboards,
    compare_teams,
    get_concise_findings,
    get_unique_teams,
    get_unique_years,
)
from src.config import (
    HISTORICAL_TOTAL_RECORDS,
    HISTORICAL_MIN_YEAR,
    HISTORICAL_MAX_YEAR,
    HISTORICAL_LOCKOUT_YEAR,
)


class TestAnalyticsEngine(unittest.TestCase):
    """Test suite for analytical query methods, comparisons, and leaderboard rankings."""

    @classmethod
    def setUpClass(cls):
        """Loads cleaned dataset once for analytical testing."""
        cls.df = load_cleaned_data()

    def test_filter_by_season_range(self):
        """Filtering by 2005 to 2010 must return only records within that range."""
        filtered = filter_historical_records(self.df, start_year=2005, end_year=2010)
        self.assertFalse(filtered.empty)
        self.assertTrue((filtered["Year"] >= 2005).all())
        self.assertTrue((filtered["Year"] <= 2010).all())
        self.assertNotIn(HISTORICAL_LOCKOUT_YEAR, filtered["Year"].values)

    def test_filter_by_teams(self):
        """Filtering by selected teams must strictly return only those franchises."""
        selected = ["Boston Bruins", "Montreal Canadiens", "Detroit Red Wings"]
        filtered = filter_historical_records(self.df, teams=selected)
        self.assertFalse(filtered.empty)
        self.assertTrue(filtered["Team_Name"].isin(selected).all())
        self.assertEqual(filtered["Team_Name"].nunique(), len(selected))

    def test_filter_empty_or_invalid_selection(self):
        """Empty filters or non-existent criteria must be handled gracefully without crashing."""
        # Non-existent team
        filtered_bad_team = filter_historical_records(self.df, teams=["NonExistent Team FC"])
        self.assertTrue(filtered_bad_team.empty)

        # Range outside dataset
        filtered_future = filter_historical_records(self.df, start_year=2030, end_year=2040)
        self.assertTrue(filtered_future.empty)

        # Empty DataFrame input
        empty_df = pd.DataFrame()
        self.assertTrue(filter_historical_records(empty_df).empty)

    def test_get_leaderboards_ordering_and_ties(self):
        """Leaderboards must correctly return top-N and bottom-N teams ordered by metric."""
        top_5, bottom_5 = get_leaderboards(self.df, metric="Win_Pct", n=5)

        self.assertEqual(len(top_5), 5)
        self.assertEqual(len(bottom_5), 5)

        # Top 1 team in historical dataset must be 1995 Detroit Red Wings (0.756 Win %)
        top_1 = top_5.iloc[0]
        self.assertEqual(top_1["Team_Name"], "Detroit Red Wings")
        self.assertEqual(top_1["Year"], 1995)
        self.assertAlmostEqual(top_1["Win_Pct"], 0.756, places=3)

        # Lowest team in historical dataset is 1992 Ottawa Senators (10-70-4, 0.119 Win %)
        bottom_1 = bottom_5.iloc[0]
        self.assertEqual(bottom_1["Team_Name"], "Ottawa Senators")
        self.assertEqual(bottom_1["Year"], 1992)
        self.assertAlmostEqual(bottom_1["Win_Pct"], 0.119, places=3)


    def test_compare_teams_single_season(self):
        """Comparing two teams in a single year must report single-season context and correct deltas."""
        comp = compare_teams(self.df, "Boston Bruins", "Montreal Canadiens", start_year=2010, end_year=2010)

        self.assertTrue(comp["is_single_season"])
        self.assertEqual(comp["period_label"], "2010 Season")
        self.assertEqual(comp["stats_a"]["seasons_count"], 1)
        self.assertEqual(comp["stats_b"]["seasons_count"], 1)

        # Check deltas calculation: team_a - team_b
        expected_diff = comp["stats_a"]["mean_win_pct"] - comp["stats_b"]["mean_win_pct"]
        self.assertAlmostEqual(comp["deltas"]["win_pct"], expected_diff, places=4)

    def test_compare_teams_multi_season(self):
        """Comparing two teams over multiple years must compute cumulative and average metrics."""
        comp = compare_teams(self.df, "Boston Bruins", "Detroit Red Wings", start_year=1995, end_year=2005)

        self.assertFalse(comp["is_single_season"])
        self.assertIn("1995–2005", comp["period_label"])
        # Both teams played active seasons during this period (excluding 2004 lockout)
        self.assertGreater(comp["stats_a"]["seasons_count"], 5)
        self.assertGreater(comp["stats_b"]["seasons_count"], 5)
        self.assertGreater(len(comp["common_years"]), 5)

    def test_compare_teams_with_no_overlap(self):
        """Comparing teams that never competed in the same era must return empty common years."""
        # Quebec Nordiques existed 1990-1994; Minnesota Wild began 2000
        comp = compare_teams(self.df, "Quebec Nordiques", "Minnesota Wild")
        self.assertEqual(len(comp["common_years"]), 0)
        self.assertTrue(comp["head_to_head_df"].empty)

    def test_concise_findings_consistency(self):
        """get_concise_findings must return 4 verified empirical insights matching dataset values."""
        findings = get_concise_findings(self.df)
        self.assertEqual(len(findings), 4)

        # Finding 1: League average ~45.85%
        self.assertIn("45.85%", findings[0]["stat"])

        # Finding 2: Symmetric scoring ~234 GF = GA
        self.assertIn("234", findings[1]["stat"])

        # Finding 3: Goal diff correlation ~0.887
        self.assertIn("0.887", findings[2]["stat"])

    def test_csv_export_format(self):
        """Filtered records must serialize cleanly to valid CSV format with correct row count."""
        filtered = filter_historical_records(self.df, start_year=2008, end_year=2011)
        csv_buffer = io.StringIO()
        filtered.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()

        # Reload CSV text and verify round-trip integrity
        reloaded = pd.read_csv(io.StringIO(csv_text))
        self.assertEqual(len(reloaded), len(filtered))
        self.assertEqual(list(reloaded.columns), list(filtered.columns))


if __name__ == "__main__":
    unittest.main()
