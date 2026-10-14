\# 🏒 Hockey Team Performance Analysis

An end-to-end data science project combining statistical inference, machine learning regression, and interactive visualization to analyze historical NHL team performance.

The project uses historical hockey data from [Scrape This Site — Hockey Teams](https://www.scrapethissite.com/pages/forms/) to explore performance trends, conduct statistical hypothesis testing, evaluate regression models using chronological validation, and deploy an interactive Streamlit dashboard.

## Table of Contents

- [Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset-overview)
- [Statistical Analysis](#-statistical-analysis)
- [Machine Learning](#-machine-learning-methodology)
- [Model Performance](#-model-performance)
- [Interactive Dashboard](#-interactive-dashboard)
- [Tech Stack](#-technology-stack)
- [Installation](#-installation-and-setup)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Project Overview

This project implements a complete data science workflow, from web scraping and data preparation to statistical analysis, predictive modeling, and interactive deployment.

The primary objective is to investigate the relationship between team scoring performance and season winning percentage while demonstrating sound statistical practices and leakage-aware machine learning evaluation.

### Project Objectives

- Collect and clean historical NHL team-season records.
- Explore probability distributions, sampling techniques, and performance trends.
- Apply statistical inference to compare teams and historical eras.
- Develop and compare multiple regression models.
- Prevent target leakage and evaluate models using chronological validation.
- Deploy an interactive dashboard for exploring predictions and feature contributions.

## ✨ Key Features

- **Data Engineering:** Paginated web scraping, data cleaning, feature engineering, and missing-value analysis.
- **Statistical Inference:** Confidence intervals, hypothesis testing, ANOVA, correlation analysis, and categorical association tests.
- **Machine Learning:** Comparison of five regression pipelines using season-aware cross-validation.
- **Temporal Validation:** Chronological train-test separation and expanding-window validation.
- **Model Explainability:** Standardized feature contributions and regression diagnostics.
- **Interactive Deployment:** Streamlit dashboard with scenario presets, performance indicators, and visual explanations.

## 📊 Dataset Overview

**Source:** [Scrape This Site — Hockey Teams](https://www.scrapethissite.com/pages/forms/)

| Dataset Attribute | Details |
|---|---|
| Total records | 582 team-seasons |
| Historical coverage | 1990–2011 |
| Raw dataset | `hockey_team_performance_raw.csv` |
| Processed dataset | `hockey_team_performance_cleaned.csv` |
| Raw columns | 9 |
| Processed columns | 11 |
| Prediction target | `Win_Pct` |

The dataset contains team-level regular-season statistics, including wins, losses, goals scored, and goals conceded.

The 2004–2005 NHL season is absent because the season was cancelled due to a labour lockout.

### Key Variables

| Feature | Description |
|---|---|
| `Team_Name` | Recorded NHL team name |
| `Year` | Season start year |
| `Wins` | Regular-season wins |
| `Losses` | Regulation losses |
| `OT_Losses` | Overtime or shootout losses, where recorded |
| `Win_Pct` | Season winning percentage |
| `Goals_For` | Total goals scored |
| `Goals_Against` | Total goals conceded |
| `Goal_Diff` | Difference between goals scored and conceded |
| `Decade` | Historical era category |
| `Goal_Difference` | Additional goal-difference validation feature |

**Data quality consideration:** The 224 missing `OT_Losses` values are preserved to represent the historical period before overtime losses were recorded in the same way as in later seasons. They are not automatically replaced with zero.

## 📈 Statistical Analysis

Notebook: [`Hockey_Team_Performance_Analysis_working.ipynb`](Hockey_Team_Performance_Analysis_working.ipynb)

The statistical analysis explores historical team performance through descriptive statistics, probability, sampling, and inferential methods.

| Analysis Area | Techniques Implemented |
|---|---|
| Data preparation | Data cleaning, missing-value analysis, feature engineering |
| Probability | Empirical probability calculations |
| Sampling | Simple random, systematic, and stratified sampling |
| Sampling distributions | Central Limit Theorem simulations |
| Statistical inference | Confidence intervals, one-sample t-tests, illustrative z-tests |
| Correlation | Pearson correlation, Spearman rank correlation, covariance |
| Group comparisons | Independent and paired t-tests |
| Analysis of variance | One-way ANOVA and Tukey HSD post-hoc testing |
| Categorical analysis | Chi-square test of independence and Cramér's V |

These analyses investigate the relationships between scoring, winning percentage, historical eras, and categorical performance groups.

## 🤖 Machine Learning Methodology

Notebook: [`Hockey_Team_Performance_ML_working_v4.ipynb`](Hockey_Team_Performance_ML_working_v4.ipynb)

### Prediction Objective

The model estimates a team's season winning percentage using its goal-scoring statistics and season year.

- **Target:** `Win_Pct`
- **Input features:** `Goals_For`, `Goals_Against`, `Year`
- **Task:** Supervised regression
- **Selected model:** Linear Regression

This is a retrospective estimation task using completed-season statistics, not a pre-season prediction or game-level forecasting system.

### Feature Selection and Leakage Prevention

The feature set is deliberately restricted to three variables.

| Feature | Decision | Reason |
|---|---|---|
| `Goals_For` | Included | Measures offensive production |
| `Goals_Against` | Included | Measures defensive performance |
| `Year` | Included | Captures historical changes across seasons |
| `Wins` | Excluded | Directly determines the target when combined with losses |
| `Losses` | Excluded | Directly involved in calculating the target |
| `OT_Losses` | Excluded | Has historical missing values and affects some season win-percentage definitions |
| `Goal_Diff` | Excluded | Derived from the two goal-related input features |

This approach reduces direct target leakage and avoids introducing a redundant feature derived from the selected goal aggregates.

### Chronological Train-Test Split

A time-aware split is used to assess performance on later seasons.

| Dataset | Season Range | Team-Season Records |
|---|---|---:|
| Training | 1990–2007 | 462 |
| Testing | 2008–2011 | 120 |
| Total | 1990–2011, excluding the cancelled season | 582 |

### Expanding-Window Cross-Validation

Five expanding temporal folds are evaluated using the training period. Each validation window contains two available seasons.

| Fold | Training Seasons | Validation Seasons |
|---|---|---|
| 1 | 1990–1996 | 1997–1998 |
| 2 | 1990–1998 | 1999–2000 |
| 3 | 1990–2000 | 2001–2002 |
| 4 | 1990–2002 | 2003, 2005 |
| 5 | 1990–2005 | 2006–2007 |

The cancelled 2004–2005 season is not represented in the dataset. Feature scaling is fitted within the relevant training folds to prevent information from validation data influencing preprocessing.

## 🏆 Model Performance

Five regression pipelines were compared using cross-validation and a held-out chronological test set.

| Model | CV RMSE (Mean ± SD) | Test RMSE | Test R² |
|---|---:|---:|---:|
| **Linear Regression** | **0.0414 ± 0.0113** | **0.0332** | **0.8397** |
| Ridge Regression | 0.0416 ± 0.0112 | 0.0333 | 0.8389 |
| Lasso Regression | 0.0423 ± 0.0119 | 0.0343 | 0.8295 |
| Decision Tree Regressor | 0.0661 ± 0.0093 | 0.0567 | 0.5329 |
| Random Forest Regressor | 0.0514 ± 0.0076 | 0.0427 | 0.7354 |

### Evaluation Metrics

- **RMSE:** Measures prediction error while penalizing larger errors more heavily.
- **MAE:** Measures the average absolute difference between actual and predicted values.
- **R²:** Measures the proportion of target variance explained by the model relative to a mean-prediction baseline.

### Final Model Selection

**Linear Regression** was selected because it achieved the lowest mean cross-validation RMSE among the evaluated models and performed best on the chronological test set.

- Cross-validation RMSE: `0.0414`
- Test RMSE: `0.0332`
- Test MAE: `0.0264`
- Test R²: `0.8397`

Since the target is represented as a proportion, the test RMSE of `0.0332` corresponds to approximately **3.32 percentage points** on the winning-percentage scale.

The results indicate a strong fit on the evaluated historical data. They should not be interpreted as evidence of pre-season forecasting accuracy.

## 🖥️ Interactive Dashboard

Application: [`app.py`](app.py)

The Streamlit application provides an interactive interface for exploring regression predictions, comparing scenarios, and understanding model behavior.

### Dashboard Capabilities

- **Scenario Presets:** Load predefined team profiles for quick comparisons.
- **Prediction Metrics:** View estimated winning percentage and goal differential.
- **Performance Classification:** Explore qualitative performance tiers.
- **Historical Reference Scale:** Compare predictions with historical minimum, mean, and maximum winning percentages.
- **Feature Explainability:** Inspect standardized feature values and their contributions to the linear regression estimate.
- **Methodology and Limitations:** Review model assumptions, feature exclusions, and the retrospective prediction scope.

### Example Scenarios

| Scenario | Goals For | Goals Against | Year |
|---|---:|---:|---:|
| Test Sample | 250 | 230 | 2009 |
| Top Contender | 300 | 210 | 2010 |
| League Average | 234 | 234 | 2006 |
| Rebuilding Team | 185 | 295 | 2003 |

The application loads the saved regression pipeline from the `models/` directory.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data manipulation | Pandas, NumPy |
| Statistical analysis | SciPy, statistical testing libraries |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Altair |
| Interactive application | Streamlit |
| Web scraping | Requests, Beautiful Soup |
| Notebook environment | Jupyter Notebook / JupyterLab |
| Model persistence | Joblib |

The project dependencies are specified in [`requirements.txt`](requirements.txt).

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- A terminal or command-line interface

### 1. Clone the Repository

Replace `<repository-url>` with the actual Git repository URL.

```bash
git clone <repository-url>
cd Hockey_Team_Performance_Analysis
```

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. If the application is configured to use port `8502`, the address will be:

`http://localhost:8502`

Otherwise, use the URL provided by Streamlit.

### 5. Explore the Notebooks

Launch JupyterLab:

```bash
jupyter lab
```

Open the following notebooks:

- **Statistical analysis:** [`Hockey_Team_Performance_Analysis_working.ipynb`](Hockey_Team_Performance_Analysis_working.ipynb)
- **Final machine learning pipeline:** [`Hockey_Team_Performance_ML_working_v4.ipynb`](Hockey_Team_Performance_ML_working_v4.ipynb)

The earlier ML notebooks are retained for reference and document previous iterations of the modeling workflow.

## 📂 Repository Structure

```text
Hockey_Team_Performance_Analysis/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── Hockey_Team_Performance_Analysis_working.ipynb
├── Hockey_Team_Performance_ML_working.ipynb
├── Hockey_Team_Performance_ML_working_v2.ipynb
├── Hockey_Team_Performance_ML_working_v3.ipynb
├── Hockey_Team_Performance_ML_working_v4.ipynb
├── app.py
├── hockey_team_performance_raw.csv
├── hockey_team_performance_cleaned.csv
├── models/
│   ├── hockey_win_pct_regressor_v2.joblib
│   ├── model_metadata_v2.json
│   ├── hockey_win_pct_regressor.joblib
│   └── model_metadata.json
├── requirements.txt
└── README.md
```
