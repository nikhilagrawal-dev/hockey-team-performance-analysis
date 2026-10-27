"""
Hockey Performance Intelligence Configuration & Design Tokens
Central definitions for file paths, feature schemas, empirical historical constants,
preset scenarios, and UI design tokens.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATH DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_RAW_PATH = os.path.join(BASE_DIR, "hockey_team_performance_raw.csv")
DATA_CLEANED_PATH = os.path.join(BASE_DIR, "hockey_team_performance_cleaned.csv")

# Models directory and model artifacts
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hockey_win_pct_regressor_v2.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata_v2.json")

# Fallback artifacts for backward compatibility
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, "hockey_win_pct_regressor.joblib")
FALLBACK_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SCHEMA & LEAKAGE CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Win_Pct"

# Features must be provided in this exact order to the model pipeline
REQUIRED_FEATURES = ["Goals_For", "Goals_Against", "Year"]

# Strictly excluded features to prevent data/target leakage
LEAKAGE_EXCLUSIONS = [
    "Wins",
    "Losses",
    "OT_Losses",
    "Goal_Diff",
    "Goal_Difference",
]

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATASET BENCHMARKS (582 NHL team-seasons, 1990–2011)
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_TOTAL_RECORDS = 582
HISTORICAL_MIN_YEAR = 1990
HISTORICAL_MAX_YEAR = 2011
HISTORICAL_LOCKOUT_YEAR = 2004  # 2004–05 season cancelled due to NHL labour lockout

HISTORICAL_MIN_WIN_PCT = 11.90   # 1992 Ottawa Senators (10-70-4, Win_Pct = 0.1190)
HISTORICAL_MEAN_WIN_PCT = 45.85  # League-wide historical mean across all 582 team-seasons
HISTORICAL_MAX_WIN_PCT = 75.60   # 1995 Detroit Red Wings (62-13-7, Win_Pct = 0.7560)
HISTORICAL_AVG_GF = 234.06       # League average regular-season goals scored
HISTORICAL_AVG_GA = 234.06       # League average regular-season goals conceded


# Evaluation window splits
TRAIN_PERIOD_YEARS = (1990, 2007)
TRAIN_PERIOD_ROWS = 462
TEST_PERIOD_YEARS = (2008, 2011)
TEST_PERIOD_ROWS = 120

# ─────────────────────────────────────────────────────────────────────────────
# PRESET SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
PRESET_SCENARIOS = {
    "Test Sample (2009)": {
        "goals_for": 250,
        "goals_against": 230,
        "year": 2009,
        "description": "Representative mid-tier post-lockout season from holdout evaluation set.",
        "expected_win_pct": 0.5300,
    },
    "Top Contender": {
        "goals_for": 300,
        "goals_against": 210,
        "year": 2010,
        "description": "Elite offensive team with +90 goal differential.",
        "expected_win_pct": 0.6773,
    },
    "League Average": {
        "goals_for": 234,
        "goals_against": 234,
        "year": 2006,
        "description": "Balanced baseline team with zero goal differential.",
        "expected_win_pct": 0.4767,
    },
    "Rebuilding Team": {
        "goals_for": 185,
        "goals_against": 295,
        "year": 2003,
        "description": "Severely out-scored rebuilding franchise with -110 goal differential.",
        "expected_win_pct": 0.2474,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# BRANDING & DESIGN SYSTEM TOKENS
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_NAME = "Hockey Performance Intelligence"
PRODUCT_TAGLINE = "NHL Regular-Season Statistical Inference & ML Platform"
PRODUCT_BADGE = "NHL Regular-Season Analytics · Statistical Inference & ML Regression"
PRODUCT_SUBTITLE = (
    "Retrospective end-of-season NHL team winning percentage estimation using "
    "completed regular-season goal aggregates and structural era adjustments (1990–2011)."
)
NON_FORECASTING_DISCLOSURE = (
    "This platform evaluates completed seasons from final regular-season records and is "
    "not an in-game or pre-season forecasting tool."
)

# Hockey sports theme palette
THEME_BG = "#EAF6FC"            # Light ice-blue canvas
THEME_CARD_BG = "#FFFFFF"       # White card surface
THEME_NAVY = "#0B1F33"          # Deep navy for header & sidebar
THEME_BORDER = "#D8E7EF"        # Subtle ice borders
THEME_PRIMARY = "#0797D5"       # Hockey blue
THEME_CYAN = "#20C4F4"          # Bright cyan accent
THEME_TEXT_PRIMARY = "#142D42"  # Dark navy text for high readability
THEME_TEXT_MUTED = "#6B8294"    # Slate muted text
THEME_TEXT_DIM = "#94A3B8"      # Light slate for captions
THEME_POSITIVE = "#10B981"      # Accessible emerald green
THEME_NEGATIVE = "#EF4444"      # Accessible crimson red
THEME_WARNING = "#F59E0B"       # Accessible amber

# ─────────────────────────────────────────────────────────────────────────────
# VERIFIED 5-MODEL EVALUATION TABLE (ML Notebook v4, Cell 54)
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATE_MODELS_EVALUATION = [
    {
        "model": "Linear Regression",
        "cv_mae": 0.0342,
        "cv_mae_sd": 0.0096,
        "cv_rmse": 0.0414,
        "cv_rmse_sd": 0.0113,
        "cv_r2": 0.8038,
        "cv_r2_sd": 0.1079,
        "test_mae": 0.0264,
        "test_rmse": 0.0332,
        "test_r2": 0.8397,
        "selected": True,
        "selection_notes": "Selected by lowest mean CV RMSE across 5 season-aware expanding-window folds (0.0414). Superior test generalization (0.0332).",
    },
    {
        "model": "Ridge Regression",
        "cv_mae": 0.0343,
        "cv_mae_sd": 0.0096,
        "cv_rmse": 0.0416,
        "cv_rmse_sd": 0.0112,
        "cv_r2": 0.8023,
        "cv_r2_sd": 0.1076,
        "test_mae": 0.0265,
        "test_rmse": 0.0333,
        "test_r2": 0.8389,
        "selected": False,
        "selection_notes": "L2 regularization provides negligible shrinkage benefit given only 3 low-collinearity features.",
    },
    {
        "model": "Lasso Regression",
        "cv_mae": 0.0349,
        "cv_mae_sd": 0.0102,
        "cv_rmse": 0.0423,
        "cv_rmse_sd": 0.0119,
        "cv_r2": 0.7939,
        "cv_r2_sd": 0.1180,
        "test_mae": 0.0274,
        "test_rmse": 0.0343,
        "test_r2": 0.8295,
        "selected": False,
        "selection_notes": "L1 penalty unnecessarily dampens informative weights without sparsity requirements.",
    },
    {
        "model": "Decision Tree",
        "cv_mae": 0.0532,
        "cv_mae_sd": 0.0071,
        "cv_rmse": 0.0661,
        "cv_rmse_sd": 0.0093,
        "cv_r2": 0.5179,
        "cv_r2_sd": 0.1706,
        "test_mae": 0.0459,
        "test_rmse": 0.0567,
        "test_r2": 0.5329,
        "selected": False,
        "selection_notes": "Step-function boundaries cause high variance and discretization error on continuous targets.",
    },
    {
        "model": "Random Forest",
        "cv_mae": 0.0417,
        "cv_mae_sd": 0.0081,
        "cv_rmse": 0.0510,
        "cv_rmse_sd": 0.0104,
        "cv_r2": 0.7055,
        "cv_r2_sd": 0.1297,
        "test_mae": 0.0340,
        "test_rmse": 0.0427,
        "test_r2": 0.7354,
        "selected": False,
        "selection_notes": "Ensemble averages trees effectively but underperforms OLS because the underlying physics is predominantly linear.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 16 SI LAB PRACTICALS DIRECTORY (Statistical Notebook Mapping)
# ─────────────────────────────────────────────────────────────────────────────
ACADEMIC_PRACTICALS_DIRECTORY = [
    {
        "num": 1,
        "topic": "Web Scraping & Data Acquisition",
        "tools": "requests, BeautifulSoup4",
        "status": "Source Curation",
        "aim": "Scrape paginated multi-season NHL records (24 pages, 582 records) from ScrapeThisSite."
    },
    {
        "num": 2,
        "topic": "Data Cleaning & Era Engineering",
        "tools": "pandas, numpy",
        "status": "Verified",
        "aim": "Preserve unrecorded pre-1999 OT losses, drop duplicate team-years, engineer Decade, and compute descriptive statistics."
    },
    {
        "num": 3,
        "topic": "Probability & Sampling Techniques",
        "tools": "pandas sampling",
        "status": "Verified",
        "aim": "Calculate exact empirical probabilities; demonstrate Simple Random Sampling (SRS), Systematic, and Stratified sampling."
    },
    {
        "num": 4,
        "topic": "Central Limit Theorem Demonstration",
        "tools": "scipy.stats, numpy",
        "status": "Interactive in App",
        "aim": "Verify CLT by simulating sampling distributions (sample size n=30) converging to normal distribution."
    },
    {
        "num": 5,
        "topic": "One-Sample t-Test against 0.500",
        "tools": "scipy.stats.ttest_1samp",
        "status": "Interactive in App",
        "aim": "Test if mean team Win % differs significantly from 0.500 (t = -9.788, p = 4.81e-21, df = 581)."
    },
    {
        "num": 6,
        "topic": "95% Confidence Interval for Population Mean",
        "tools": "scipy.stats.t",
        "status": "Interactive in App",
        "aim": "Estimate unknown population mean team win percentage with 95% confidence [0.4502, 0.4668]."
    },
    {
        "num": 7,
        "topic": "Illustrative One-Sample Z-Test",
        "tools": "scipy.stats.norm",
        "status": "Verified",
        "aim": "Contrast classical Z-test against t-test; illustrates why substituting sample s for sigma offers no independent confirmation."
    },
    {
        "num": 8,
        "topic": "Pearson Linear Correlation",
        "tools": "scipy.stats.pearsonr",
        "status": "Interactive in App",
        "aim": "Evaluate linear associations (GF vs Win Pct: r=0.480; Diff vs Win Pct: r=0.887; GA vs Win Pct: r=-0.487)."
    },
    {
        "num": 9,
        "topic": "Sample Covariance Matrix",
        "tools": "pandas.cov",
        "status": "Verified",
        "aim": "Compute covariance matrix and analyze scale/unit dependency across goal metrics."
    },
    {
        "num": 10,
        "topic": "Spearman Rank Correlation",
        "tools": "scipy.stats.spearmanr",
        "status": "Verified",
        "aim": "Assess monotonic relationships without assuming bivariate normality (GF vs Win Pct: rho=0.517)."
    },
    {
        "num": 11,
        "topic": "Pearson vs. Spearman Comparative Contrast",
        "tools": "scipy.stats",
        "status": "Verified",
        "aim": "Systematically contrast Pearson r and Spearman rho; agreement (|r - rho| <= 0.037) confirms strong linearity."
    },
    {
        "num": 12,
        "topic": "Multiple Linear Regression (Statsmodels OLS)",
        "tools": "statsmodels.api.OLS",
        "status": "Interactive in App",
        "aim": "Fit Win_Pct ~ GF + GA (R2=0.788, F=1075); evaluate Durbin-Watson autocorrelation (1.079) and residuals."
    },
    {
        "num": 13,
        "topic": "One-Way ANOVA Across Historical Decades",
        "tools": "scipy.stats.f_oneway, Tukey HSD",
        "status": "Interactive in App",
        "aim": "Test mean Win % differences across 1990s, 2000s, 2010s (F=14.687, p=5.998e-07)."
    },
    {
        "num": 14,
        "topic": "Two-Sample Independent Welch's t-Test",
        "tools": "scipy.stats.ttest_ind",
        "status": "Verified",
        "aim": "Compare 1990s vs 2000s win percentages (t = -4.140, p = 4.06e-05, Cohen's d = -0.363)."
    },
    {
        "num": 15,
        "topic": "Paired t-Test for Consecutive Seasons",
        "tools": "scipy.stats.ttest_rel",
        "status": "Verified",
        "aim": "Test stability of matched franchise win percentages between 2010 and 2011 (t = -0.086, p = 0.932)."
    },
    {
        "num": 16,
        "topic": "Chi-Square Test of Independence",
        "tools": "scipy.stats.chi2_contingency",
        "status": "Interactive in App",
        "aim": "Test association between Decade and Performance Tier (chi2=28.078, p=1.203e-05, Cramer's V=0.155)."
    },
]

