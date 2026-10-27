"""
Unit tests for Statistical Insights & SI Lab Practicals (Phase 2B).
Verifies One-Sample t-Test, Historical Era ANOVA, Chi-Square Independence,
Central Limit Theorem simulations, and 16 Practicals directory.
"""

import unittest
import pandas as pd
import numpy as np

from src.config import (
    ACADEMIC_PRACTICALS_DIRECTORY,
    DATA_CLEANED_PATH,
    TARGET_COLUMN,
)
from src.data import (
    load_cleaned_data,
    get_statistical_insights_data,
    run_clt_simulation,
)
from src.charts import (
    build_clt_simulation_chart,
    build_anova_decade_boxplot,
    build_chisquare_heatmap,
)


class TestStatisticalInsights(unittest.TestCase):
    """Test suite for SI Lab practicals and statistical inference calculations."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_cleaned_data(DATA_CLEANED_PATH)

    def test_practicals_directory_completeness(self):
        """Verifies all 16 academic practicals are correctly indexed."""
        self.assertEqual(len(ACADEMIC_PRACTICALS_DIRECTORY), 16)
        nums = [p["num"] for p in ACADEMIC_PRACTICALS_DIRECTORY]
        self.assertEqual(nums, list(range(1, 17)))

        # Verify key practical topics
        p4 = next(p for p in ACADEMIC_PRACTICALS_DIRECTORY if p["num"] == 4)
        self.assertIn("Central Limit Theorem", p4["topic"])
        self.assertEqual(p4["status"], "Interactive in App")

        p5 = next(p for p in ACADEMIC_PRACTICALS_DIRECTORY if p["num"] == 5)
        self.assertIn("One-Sample t-Test", p5["topic"])

        p13 = next(p for p in ACADEMIC_PRACTICALS_DIRECTORY if p["num"] == 13)
        self.assertIn("ANOVA", p13["topic"])

        p16 = next(p for p in ACADEMIC_PRACTICALS_DIRECTORY if p["num"] == 16)
        self.assertIn("Chi-Square", p16["topic"])

    def test_one_sample_ttest_verified_results(self):
        """Verifies one-sample t-test against mu=0.500 matches academic notebook."""
        stats_data = get_statistical_insights_data(self.df)
        t_res = stats_data["t_test"]

        self.assertEqual(t_res["sample_size"], 582)
        self.assertEqual(t_res["degrees_of_freedom"], 581)
        self.assertAlmostEqual(t_res["sample_mean"], 0.45852, places=4)
        self.assertAlmostEqual(t_res["sample_std"], 0.10224, places=4)
        self.assertAlmostEqual(t_res["t_statistic"], -9.7882, places=3)
        self.assertLess(t_res["p_value"], 1e-15)

        # 95% Confidence Interval check
        self.assertAlmostEqual(t_res["ci_95_low"], 0.45020, places=4)
        self.assertAlmostEqual(t_res["ci_95_high"], 0.46684, places=4)

    def test_one_way_anova_verified_results(self):
        """Verifies historical era ANOVA and post-hoc contrasts match academic notebook."""
        stats_data = get_statistical_insights_data(self.df)
        anova = stats_data["anova"]

        # F-statistic and p-value
        self.assertAlmostEqual(anova["f_statistic"], 14.6872, places=3)
        self.assertLess(anova["p_value"], 1e-5)
        self.assertAlmostEqual(anova["eta_squared"], 0.0483, places=3)

        # Group counts
        g_df = anova["group_stats"]
        self.assertEqual(len(g_df), 3)
        counts = g_df["Count"].tolist()
        self.assertEqual(counts, [252, 270, 60])

        # Post-hoc comparisons
        ph_df = anova["post_hoc"]
        self.assertEqual(len(ph_df), 3)
        # 1990s vs 2000s and 1990s vs 2010s significant; 2000s vs 2010s not significant
        sig_list = ph_df["Significant"].tolist()
        self.assertEqual(sig_list, [True, True, False])

    def test_chi_square_test_verified_results(self):
        """Verifies Chi-Square test of independence matches academic notebook."""
        stats_data = get_statistical_insights_data(self.df)
        chi2_res = stats_data["chi_square"]

        self.assertAlmostEqual(chi2_res["chi2_statistic"], 28.0782, places=3)
        self.assertEqual(chi2_res["degrees_of_freedom"], 4)
        self.assertLess(chi2_res["p_value"], 1e-4)
        self.assertAlmostEqual(chi2_res["cramers_v"], 0.1553, places=3)

        # Expected frequencies >= 5 check
        self.assertTrue(chi2_res["all_expected_ge_5"])
        self.assertAlmostEqual(chi2_res["min_expected_frequency"], 10.72, places=2)

    def test_clt_simulation_reproducibility_and_properties(self):
        """Verifies CLT simulation properties and consistency with theoretical standard error."""
        sim = run_clt_simulation(self.df, sample_size=30, n_simulations=1000, seed=42)

        self.assertEqual(len(sim["sim_df"]), 1000)
        self.assertAlmostEqual(sim["pop_mean"], 0.4585, places=3)

        # Theoretical SE = pop_std / sqrt(30) ~ 0.0186
        expected_se = sim["pop_std"] / np.sqrt(30)
        self.assertAlmostEqual(sim["theoretical_se"], expected_se, places=4)

        # Empirical mean of sample means should be very close to population mean
        self.assertAlmostEqual(sim["empirical_mean"], sim["pop_mean"], places=2)
        # Empirical standard deviation of sample means should be close to theoretical SE
        self.assertAlmostEqual(sim["empirical_se"], sim["theoretical_se"], delta=0.003)

    def test_fallback_on_empty(self):
        """Verifies statistical insights and CLT simulation handle empty input safely."""
        empty_stats = get_statistical_insights_data(pd.DataFrame())
        self.assertEqual(empty_stats["practicals"], ACADEMIC_PRACTICALS_DIRECTORY)

        empty_sim = run_clt_simulation(pd.DataFrame())
        self.assertEqual(len(empty_sim["sim_df"]), 0)

    def test_statistical_charts(self):
        """Verifies Altair statistical charts compile properly."""
        sim = run_clt_simulation(self.df, sample_size=30, n_simulations=500, seed=42)
        c1 = build_clt_simulation_chart(sim["sim_df"], sim["pop_mean"], sim["theoretical_se"])
        self.assertIsNotNone(c1)
        self.assertTrue(hasattr(c1, "to_dict"))

        c2 = build_anova_decade_boxplot(self.df)
        self.assertIsNotNone(c2)
        self.assertTrue(hasattr(c2, "to_dict"))

        stats_data = get_statistical_insights_data(self.df)
        c3 = build_chisquare_heatmap(stats_data["chi_square"]["observed_table"])
        self.assertIsNotNone(c3)
        self.assertTrue(hasattr(c3, "to_dict"))


if __name__ == "__main__":
    unittest.main()
