# Hockey Team Performance Analysis

A statistical inference, machine learning regression, and interactive dashboard project analyzing NHL team-season performance using data collected from [Scrape This Site — Hockey Teams](https://www.scrapethissite.com/pages/forms/).

The repository implements an end-to-end data science workflow: web scraping, data cleaning, exploratory data analysis (EDA), classical statistical hypothesis testing, time-aware machine learning regression, model explainability, and an interactive Streamlit GUI.

---

## 1. Project Objectives

1. **Data Acquisition & Curation:** Scrape paginated multi-season NHL records into structured datasets while handling historical era differences (e.g., preserving unrecorded pre-shootout overtime loss values as missing rather than imputing zeros).
2. **Exploratory & Inferential Statistics:** Apply probability distributions, Central Limit Theorem demonstrations, confidence intervals, correlation analyses, parametric and non-parametric hypothesis tests, ANOVA, and categorical association tests.
3. **Machine Learning Regression:** Train and evaluate multiple regression pipelines to estimate a team's retrospective end-of-season winning percentage (`Win_Pct`) from season goal aggregates and historical era (`Goals_For`, `Goals_Against`, `Year`).
4. **Leakage Prevention & Temporal Validation:** Enforce strict feature selection to prevent target leakage and evaluate generalization via chronological holdout and expanding-window cross-validation.
5. **Interactive Model Deployment:** Provide a Streamlit web application with scenario presets, standardized feature explainability, and statistical model diagnostics.

---

## 2. Dataset Overview

- **Source:** [https://www.scrapethissite.com/pages/forms/](https://www.scrapethissite.com/pages/forms/)
- **Total Records:** 582 NHL team-seasons across 21 historical seasons (1990–2011).
  *(Note: The 2004–2005 NHL season was cancelled due to a labour lockout and is absent from the historical records).*
- **Raw Dataset (`hockey_team_performance_raw.csv`):** 582 rows, 9 columns.
- **Cleaned Dataset (`hockey_team_performance_cleaned.csv`):** 582 rows, 11 columns, including engineered features and decade categorizations.

### Key Data Fields

| Column | Type | Description |
|---|---|---|
| `Team_Name` | String | Recorded team name (e.g., Boston Bruins, Detroit Red Wings). |
| `Year` | Integer | Season start year (1990 to 2011). |
| `Wins` | Integer | Regular season wins. |
| `Losses` | Integer | Regular season regulation losses. |
| `OT_Losses` | Float | Overtime / shootout losses (224 missing values preserved for pre-1999 era). |
| `Win_Pct` | Float | Season winning percentage (Target variable, range: 0.119 to 0.756). |
| `Goals_For` | Integer | Total regular season goals scored by the team (GF). |
| `Goals_Against` | Integer | Total regular season goals conceded by the team (GA). |
| `Goal_Diff` | Integer | Goal differential ($Goals\_For - Goals\_Against$). |
| `Decade` | String | Categorical era bin (`1990s`, `2000s`, `2010s`). |
| `Goal_Difference`| Integer | Validated goal difference feature check. |

---

## 3. Statistical Analysis

Implemented and documented in the primary statistical notebook [`Hockey_Team_Performance_Analysis_working.ipynb`](Hockey_Team_Performance_Analysis_working.ipynb):

1. **Data Cleaning & Missing Value Handling:** Preservation of 224 missing `OT_Losses` values representing the pre-shootout era where ties existed.
2. **Probability & Sampling:** Empirical probabilities and demonstration of Simple Random Sampling (SRS), Systematic Sampling, and Stratified Sampling.
3. **Sampling Distributions:** Verification of the Central Limit Theorem across repeated sample simulations.
4. **Confidence Intervals & Hypothesis Testing:** One-sample $t$-test and illustrative $Z$-test for team winning percentage against 0.500.
5. **Correlation & Covariance:** Analysis of Pearson and Spearman rank correlation between goal scoring and winning percentage.
6. **Group Comparisons & ANOVA:** One-way ANOVA across decades with Tukey HSD post-hoc testing, independent two-sample $t$-tests, and paired $t$-tests.
7. **Categorical Association:** Chi-square test of independence and Cramér's V effect size across performance tiers and eras.

---

## 4. Machine Learning Methodology

Implemented, evaluated, and verified in the primary ML notebook [`Hockey_Team_Performance_ML_working_v4.ipynb`](Hockey_Team_Performance_ML_working_v4.ipynb):

### Prediction Objective & Target
- **Target Variable:** `Win_Pct` (continuous end-of-season winning percentage in $[0, 1]$).
- **Scope:** Retrospective performance estimation using completed regular-season goal aggregates.
- **Important Distinction:** This is **not** a game-level or pre-season forecasting model; regular season goal totals are only finalized at the conclusion of a season.

### Feature Selection & Target Leakage Prevention
To prevent data and target leakage, the feature space is strictly constrained:
- **Included Features (3):** `['Goals_For', 'Goals_Against', 'Year']`
- **Excluded Features:**
  - `Wins` and `Losses`: Excluded because they directly reconstruct the target ($Win\_Pct = Wins / (Wins + Losses)$).
  - `OT_Losses`: Excluded as part of the denominator in post-shootout hockey; missing in 224 of 582 records.
  - `Goal_Diff`: Excluded as a direct collinear linear combination of $Goals\_For - Goals\_Against$ (Pearson $r \approx 0.88$).

### Chronological Train / Test Split
To evaluate temporal generalization rather than random interpolation:
- **Training Period:** 1990–2007 (462 team-seasons across 17 calendar seasons).
- **Held-Out Test Period:** 2008–2011 (120 team-seasons across 4 calendar seasons).

### Season-Aware Expanding-Window Cross-Validation
Cross-validation is conducted strictly on the training set (1990–2007) using 5 expanding temporal folds (2 validation seasons per fold):
- **Fold 1:** Train 1990–1996 (171 teams) → Val 1997–1998 (53 teams)
- **Fold 2:** Train 1990–1998 (224 teams) → Val 1999–2000 (58 teams)
- **Fold 3:** Train 1990–2000 (282 teams) → Val 2001–2002 (60 teams)
- **Fold 4:** Train 1990–2002 (342 teams) → Val 2003, 2005 (60 teams; 2004 lockout omitted)
- **Fold 5:** Train 1990–2005 (402 teams) → Val 2006–2007 (60 teams)

### Model Comparison & Selection
Five pipeline architectures (with `StandardScaler` fitted strictly on training folds) were evaluated:

| Model Pipeline | CV MAE (± sd) | CV RMSE (± sd) | CV $R^2$ (± sd) | Test MAE | Test RMSE | Test $R^2$ |
|---|---|---|---|---|---|---|
| **Linear Regression (Selected)** | **0.0342 (± 0.0096)** | **0.0414 (± 0.0113)** | **0.8038 (± 0.1079)** | **0.0264** | **0.0332** | **0.8397** |
| Ridge Regression | 0.0343 (± 0.0096) | 0.0416 (± 0.0112) | 0.8023 (± 0.1076) | 0.0265 | 0.0333 | 0.8389 |
| Lasso Regression | 0.0349 (± 0.0102) | 0.0423 (± 0.0119) | 0.7939 (± 0.1180) | 0.0274 | 0.0343 | 0.8295 |
| Decision Tree Regressor | 0.0532 (± 0.0071) | 0.0661 (± 0.0093) | 0.5179 (± 0.1706) | 0.0459 | 0.0567 | 0.5329 |
| Random Forest Regressor | 0.0417 (± 0.0053) | 0.0514 (± 0.0076) | 0.6974 (± 0.0988) | 0.0340 | 0.0427 | 0.7354 |

*Selection Decision:* **Linear Regression** was selected based on the lowest mean CV RMSE across the 5 season-aware folds (`0.0414`) and superior test generalization (`Test RMSE = 0.0332`, `Test R² = 0.8397`).

---

## 5. Saved Model Artifacts

The final production pipeline refitted on all 462 training team-seasons is saved in `models/`:

- **Primary Pipeline Artifact:** [`models/hockey_win_pct_regressor_v2.joblib`](models/hockey_win_pct_regressor_v2.joblib)
  - Components: `StandardScaler()`, `LinearRegression()`.
  - Feature input order: `['Goals_For', 'Goals_Against', 'Year']`.
- **Primary Metadata Artifact:** [`models/model_metadata_v2.json`](models/model_metadata_v2.json)
  - Contains full cross-validation and test metrics, fold splits, scaler statistics, coefficients, intercept, and assumptions.
- **Fallback Artifacts:** `models/hockey_win_pct_regressor.joblib` and `models/model_metadata.json` (preserved for backward compatibility).

 
---

## 6. Interactive Streamlit Dashboard

The web application [`app.py`](app.py) provides an interactive GUI to explore model predictions and feature contributions:

- **Scenario Presets:** Instant loading for canonical team profiles:
  - *Test Sample (2009):* 250 GF, 230 GA, 2009 $\rightarrow$ `53.00%` Win Pct
  - *Top Contender:* 300 GF, 210 GA, 2010 $\rightarrow$ `67.73%` Win Pct
  - *League Average:* 234 GF, 234 GA, 2006 $\rightarrow$ `47.67%` Win Pct
  - *Rebuilding Team:* 185 GF, 295 GA, 2003 $\rightarrow$ `24.74%` Win Pct
- **Dynamic Metric Display:** Estimated Win %, Goal Differential, and qualitative performance tiering.
- **Comparative Visual Scale:** Custom SVG visual indicator showing predicted Win % positioned against historical minimum (11.9%), mean (45.9%), and maximum (75.6%).
- **Model Explainability (Collapsed Expander):** Interactive standardized $z$-score breakdown, latex formula display, and Altair feature-contribution bar chart demonstrating how each feature shifts the estimate away from the baseline intercept.
- **Methodology & Limitations (Collapsed Expander):** Explicitly documents leakage exclusions, retrospective non-forecasting scope, and statistical assumptions.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+ (recommended: Python 3.10 to 3.12).

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Hockey_Team_Performance_Analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On macOS/Linux
   # venv\Scripts\activate         # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8502` (or the URL printed in the terminal).

5. **Run the Jupyter Notebooks:**
   ```bash
   jupyter lab
   # Or: jupyter notebook
   ```
   - Open [`Hockey_Team_Performance_Analysis_working.ipynb`](Hockey_Team_Performance_Analysis_working.ipynb) for the statistical analysis.
   - Open [`Hockey_Team_Performance_ML_working_v4.ipynb`](Hockey_Team_Performance_ML_working_v4.ipynb) for the machine learning pipeline.

---

## 8. Repository Directory Structure

```text
.
├── .gitignore                                      # Git ignore rules for environments, caches, and secrets
├── .streamlit/
│   └── config.toml                                 # Streamlit custom theme configuration
├── Hockey_Team_Performance_Analysis_working.ipynb  # Primary statistical analysis notebook (fully executed)
├── Hockey_Team_Performance_ML_working_v4.ipynb     # Primary ML regression & evaluation notebook (fully executed)
├── Hockey_Team_Performance_ML_working.ipynb        # Historical ML iteration (initial exploration)
├── Hockey_Team_Performance_ML_working_v2.ipynb     # Historical ML iteration (v2 pipeline development)
├── Hockey_Team_Performance_ML_working_v3.ipynb     # Historical ML iteration (v3 refinement)
├── README.md                                       # Comprehensive project documentation
├── app.py                                          # Interactive Streamlit web application
├── hockey_team_performance_cleaned.csv             # Cleaned NHL dataset with engineered features (582 records)
├── hockey_team_performance_raw.csv                 # Raw scraped NHL dataset (582 records)
├── models/
│   ├── hockey_win_pct_regressor_v2.joblib          # Active production trained linear regression pipeline
│   ├── model_metadata_v2.json                      # Comprehensive evaluation metadata & parameters
│   ├── hockey_win_pct_regressor.joblib             # Fallback regression pipeline
│   └── model_metadata.json                         # Fallback evaluation metadata
└── requirements.txt                                # Reproducible project dependency specification
```

---
