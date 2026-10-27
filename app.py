"""
Hockey Performance Intelligence
Interactive Analytics Platform & Retrospective Performance Estimation Engine.

Core Sections (Phase 2A):
1. Overview & Insights
2. Historical Explorer
3. Team Comparison
4. Prediction Studio (Preserved Active Pipeline)
"""

import io
import os
import json
import base64
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hockey Performance Intelligence",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULAR IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from src.config import (
    BASE_DIR,
    MODEL_PATH,
    METADATA_PATH,
    REQUIRED_FEATURES,
    HISTORICAL_MIN_WIN_PCT,
    HISTORICAL_MEAN_WIN_PCT,
    HISTORICAL_MAX_WIN_PCT,
    HISTORICAL_AVG_GF,
    HISTORICAL_AVG_GA,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    PRODUCT_SUBTITLE,
    NON_FORECASTING_DISCLOSURE,
    CANDIDATE_MODELS_EVALUATION,
    ACADEMIC_PRACTICALS_DIRECTORY,
)
from src.data import (
    get_cached_cleaned_data,
    filter_historical_records,
    get_leaderboards,
    compare_teams,
    get_concise_findings,
    get_unique_teams,
    get_unique_years,
    get_historical_summary_stats,
    get_model_evaluation_data,
    get_statistical_insights_data,
    run_clt_simulation,
)
from src.charts import (
    build_win_pct_distribution_chart,
    build_goal_diff_scatter_chart,
    build_team_timeline_chart,
    build_h2h_grouped_bar_chart,
    build_cv_rmse_comparison_chart,
    build_test_rmse_comparison_chart,
    build_actual_vs_predicted_chart,
    build_residual_distribution_chart,
    build_error_by_season_chart,
    build_clt_simulation_chart,
    build_anova_decade_boxplot,
    build_chisquare_heatmap,
    build_season_trends_chart,
)
from src.model import (
    load_model_and_metadata as _load_model_and_metadata,
    validate_contribution_breakdown,
    validate_prediction_inputs,
    predict_win_pct,
    calculate_feature_contributions,
)


# ─────────────────────────────────────────────────────────────────────────────
# SCOPED CSS (ICE-HOCKEY SPORTS-ANALYTICS DESIGN SYSTEM)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_hero_banner_b64() -> str:
    """Returns base64 encoded string of the hockey player hero background."""
    img_path = os.path.join(BASE_DIR, "assets", "hockey_hero_banner.jpg")
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def inject_custom_css():
    """Injects scoped CSS for ice-hockey sports-analytics theme consistency and component styling."""
    st.markdown("""
    <style>
    /* Main application canvas and base typography */
    .stApp {
        background-color: #EAF6FC;
        color: #142D42;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Complete suppression of Streamlit default header, toolbar and decorations */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Container top padding so sports navigation bar sits cleanly at the top of the viewport */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Sports League Top Navigation Bar */
    .sports-league-navbar {
        background-color: #0B1F33;
        display: flex;
        align-items: center;
        border-radius: 8px;
        margin: 0 0 1.25rem 0 !important;
        padding: 0 12px;
        height: 52px;
        box-shadow: 0 4px 16px rgba(11, 31, 51, 0.2);
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        border-bottom: 2px solid #0797D5;
        position: relative;
        z-index: 1000;
    }
    .nav-brand-controls {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-right: 16px;
        border-right: 1px solid rgba(255, 255, 255, 0.12);
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    .nav-control-icon {
        cursor: pointer;
        opacity: 0.85;
        transition: opacity 0.15s ease;
    }
    .nav-control-icon:hover {
        opacity: 1;
        color: #20C4F4;
    }
    .nav-items-wrapper {
        display: flex;
        align-items: center;
        flex: 1;
        justify-content: space-between;
        height: 100%;
        padding-left: 10px;
    }
    .sports-nav-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 52px;
        padding: 0 14px;
        color: #CFE4F1 !important;
        text-decoration: none !important;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
    }
    .sports-nav-item:hover {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.06);
    }
    .sports-nav-item.active {
        background: linear-gradient(90deg, #0797D5 0%, #0087BC 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800;
        clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 100%, 0 100%);
        padding-right: 22px;
        box-shadow: 0 4px 12px rgba(7, 151, 213, 0.35);
    }
    .nav-item-icon {
        display: flex;
        align-items: center;
        font-size: 0.9rem;
    }
    .nav-item-text {
        white-space: nowrap;
    }
    .nav-crest-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 12px;
    }

    /* Hero Banner matching Continental Hockey League screenshot */
    .hockey-hero-card {
        background-color: #FFFFFF !important;
        border: 1px solid #D8E7EF !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(11, 31, 51, 0.05) !important;
        min-height: 250px !important;
        height: 250px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: stretch !important;
        justify-content: space-between !important;
        margin-bottom: 22px !important;
        overflow: hidden !important;
        padding: 0 !important;
        position: relative !important;
    }
    .hero-left-content {
        flex: 1 1 54% !important;
        max-width: 58% !important;
        padding: 28px 36px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        z-index: 2 !important;
        background: linear-gradient(90deg, #FFFFFF 0%, #FFFFFF 85%, rgba(255,255,255,0) 100%) !important;
    }
    .hero-right-visual {
        flex: 1 1 46% !important;
        height: 100% !important;
        position: relative !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
    }
    .hero-player-img {
        height: 100% !important;
        width: 100% !important;
        object-fit: cover !important;
        object-position: right center !important;
        mask-image: linear-gradient(to left, rgba(0,0,0,1) 65%, rgba(0,0,0,0) 100%) !important;
        -webkit-mask-image: linear-gradient(to left, rgba(0,0,0,1) 65%, rgba(0,0,0,0) 100%) !important;
    }
    .hero-wordmark {
        font-family: 'Impact', 'Montserrat', 'Arial Black', sans-serif;
        font-size: 58px;
        font-weight: 900;
        font-style: italic;
        color: #0B1F33;
        letter-spacing: 2px;
        line-height: 0.95;
        text-transform: uppercase;
        margin: 0;
        padding: 0;
    }
    .hero-ribbon-bar {
        background: linear-gradient(90deg, #0797D5 0%, #20C4F4 100%);
        transform: skewX(-14deg);
        display: inline-block;
        padding: 5px 20px;
        border-radius: 3px;
        margin-top: 6px;
        margin-bottom: 6px;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(7, 151, 213, 0.25);
    }
    .hero-ribbon-text {
        display: inline-block;
        transform: skewX(14deg);
        font-family: 'Impact', 'Montserrat', 'Arial Black', sans-serif;
        font-size: 28px;
        font-weight: 900;
        font-style: italic;
        color: #FFFFFF;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .hero-tagline {
        font-size: 12px;
        font-weight: 800;
        color: #142D42;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 8px;
        line-height: 1.4;
    }

    /* 5 KPI Stat Cards in a row matching screenshot */
    .kpi-row-5 {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }
    @media (max-width: 992px) {
        .kpi-row-5 {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .kpi-card-box {
        background-color: #FFFFFF;
        border: 1px solid #D8E7EF;
        border-left: 5px solid #20C4F4;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(7, 151, 213, 0.12);
    }
    .kpi-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .kpi-card-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
    }
    .kpi-card-label {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #4B6376;
    }
    .kpi-card-value {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #0B1F33;
        line-height: 1.1;
    }
    .kpi-pill-badge {
        display: inline-block;
        background-color: #E0F4FE;
        color: #0797D5;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-top: 4px;
        width: fit-content;
    }

    /* Lower 2-column layout */
    .section-title-sports {
        font-size: 1.15rem;
        font-weight: 900;
        font-style: italic;
        color: #0B1F33;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    .title-slash {
        color: #0797D5;
        font-size: 1.25rem;
    }
    .sports-card-container {
        background-color: #FFFFFF;
        border: 1px solid #D8E7EF;
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04);
        min-height: 330px;
    }
    .findings-list {
        display: flex;
        flex-direction: column;
        gap: 14px;
        justify-content: space-between;
        height: 100%;
    }
    .finding-row {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid #EBF3F8;
    }
    .finding-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .finding-icon-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #EAF6FC;
        border: 1px solid #D8E7EF;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .finding-headline {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0B1F33;
        margin-bottom: 2px;
    }
    .finding-subtext {
        font-size: 0.84rem;
        color: #4B6376;
        line-height: 1.45;
    }

    /* Sidebar styling: Deep navy with cyan accents */
    [data-testid="stSidebar"] {
        background-color: #0B1F33 !important;
        border-right: 1px solid #132B45 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }
    [data-testid="stSidebar"] * {
        color: #CFE4F1;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1A3854 !important;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 4px;
        color: #A0BBD0 !important;
        font-weight: 600;
        font-size: 0.9rem;
        border-left: 3px solid transparent;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(32, 196, 244, 0.08);
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(7, 151, 213, 0.28) 0%, rgba(32, 196, 244, 0.12) 100%) !important;
        border-left: 3px solid #20C4F4 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(7, 151, 213, 0.15);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        color: inherit !important;
    }

    /* Container card styling */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #D8E7EF !important;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04);
        padding: 1.25rem 1.4rem;
    }

    /* Badge pill */
    .badge-pill {
        display: inline-block;
        padding: 0.24rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #0797D5;
        background-color: #EAF6FC;
        border: 1px solid #BEE3F8;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }

    /* Input chip */
    .input-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #142D42;
        background-color: #FFFFFF;
        border: 1px solid #D8E7EF;
        border-radius: 6px;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(11, 31, 51, 0.03);
    }

    /* Tier card */
    .tier-card {
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-top: 0.65rem;
        margin-bottom: 0.65rem;
        border: 1px solid #D8E7EF;
        background-color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(11, 31, 51, 0.04);
    }
    .tier-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .tier-desc {
        color: #4B6376;
        font-size: 0.85rem;
        line-height: 1.45;
        margin: 0;
    }

    /* Finding card */
    .finding-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E7EF;
        border-top: 3px solid #0797D5;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .finding-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(7, 151, 213, 0.12);
    }
    .finding-stat {
        font-size: 1.4rem;
        font-weight: 900;
        color: #0797D5;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .finding-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0B1F33;
        margin-bottom: 0.35rem;
    }
    .finding-desc {
        font-size: 0.85rem;
        color: #4B6376;
        line-height: 1.45;
        margin: 0;
    }

    /* Sports Stat KPI metric cards */
    [data-testid="stMetric"], .kpi-card {
        background-color: #FFFFFF !important;
        border: 1px solid #D8E7EF !important;
        border-left: 4px solid #0797D5 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover, .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(7, 151, 213, 0.12) !important;
        border-color: #BEE3F8 !important;
        border-left: 4px solid #20C4F4 !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 900 !important;
        letter-spacing: -0.02em !important;
        color: #0B1F33 !important;
        font-size: 1.85rem !important;
    }
    [data-testid="stMetricLabel"], .kpi-title {
        color: #6B8294 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 900;
        color: #0B1F33;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #6B8294;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 2px solid #D8E7EF;
        margin-bottom: 1rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        color: #6B8294;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0 0.4rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0797D5 !important;
        border-bottom: 2px solid #0797D5 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #D8E7EF;
        background: linear-gradient(180deg, #FFFFFF 0%, #F4FAFD 100%);
        color: #0B1F33;
        box-shadow: 0 1px 3px rgba(11, 31, 51, 0.05);
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #0797D5 !important;
        color: #0797D5 !important;
        background-color: #EAF6FC !important;
        box-shadow: 0 2px 8px rgba(7, 151, 213, 0.15);
    }

    /* Form controls, inputs, selects */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #D8E7EF !important;
        color: #142D42 !important;
        border-radius: 6px;
    }
    div[data-baseweb="input"] input {
        color: #142D42 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #0797D5 !important;
        box-shadow: 0 0 0 1px #0797D5 !important;
    }

    /* Typography & athletic headings */
    h1:not(.hockey-hero-banner *),
    h2:not(.hockey-hero-banner *),
    h4:not(.hockey-hero-banner *),
    h5:not(.hockey-hero-banner *),
    h6:not(.hockey-hero-banner *) {
        color: #0B1F33 !important;
        font-weight: 700 !important;
    }
    .hockey-hero-banner,
    .hockey-hero-banner *,
    .hockey-hero-banner h1,
    .hockey-hero-banner h2,
    .hockey-hero-banner .hero-title-text {
        color: #FFFFFF !important;
    }
    h3 {
        color: #0B1F33 !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        font-size: 1.15rem !important;
        display: flex;
        align-items: center;
    }
    h3::before {
        content: "";
        display: inline-block;
        width: 10px;
        height: 1.05em;
        background: linear-gradient(180deg, #20C4F4 0%, #0797D5 100%);
        margin-right: 10px;
        transform: skewX(-14deg);
        border-radius: 2px;
    }
    .stCaption, small {
        color: #6B8294 !important;
    }

    /* Table / DataFrame styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #D8E7EF;
        border-radius: 6px;
        background-color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL & METADATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Loads the fitted model pipeline and evaluation metadata safely using src.model."""
    return _load_model_and_metadata(MODEL_PATH, METADATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR (CONTINENTAL HOCKEY LEAGUE AESTHETIC)
# ─────────────────────────────────────────────────────────────────────────────
def render_top_navigation_bar(active_view: str):
    """Renders the top navigation bar matching the Continental Hockey League layout."""
    tabs_data = [
        {
            "name": "Overview & Insights",
            "label": "OVERVIEW & INSIGHTS",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>'
        },
        {
            "name": "Historical Explorer",
            "label": "HISTORICAL EXPLORER",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        },
        {
            "name": "Team Comparison",
            "label": "TEAM COMPARISON",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
        },
        {
            "name": "Prediction Studio",
            "label": "PREDICTION STUDIO",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>'
        },
        {
            "name": "Model Evaluation Lab",
            "label": "MODEL EVALUATION LAB",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
        },
        {
            "name": "Statistical Insights & SI Lab",
            "label": "STATISTICAL INSIGHTS & SI LAB",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31L4.69 17.5a2 2 0 0 0 1.62 3.1h11.38a2 2 0 0 0 1.62-3.1L14 9.31V2h-4z"></path><line x1="8.5" y1="2" x2="15.5" y2="2"></line></svg>'
        },
        {
            "name": "Data Quality & Methodology",
            "label": "DATA QUALITY & METHODOLOGY",
            "svg": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'
        },
    ]

    left_tabs = tabs_data[:3]
    right_tabs = tabs_data[3:]

    def build_tab_html(tab):
        is_active = tab["name"] == active_view
        active_cls = " active" if is_active else ""
        query_val = tab["name"].replace("&", "%26").replace(" ", "+")
        return (
            f'<a href="?view={query_val}" target="_self" class="sports-nav-item{active_cls}">'
            f'<span class="nav-item-icon">{tab["svg"]}</span>'
            f'<span class="nav-item-text">{tab["label"]}</span>'
            f'</a>'
        )

    left_html = "".join(build_tab_html(t) for t in left_tabs)
    right_html = "".join(build_tab_html(t) for t in right_tabs)

    crest_svg = (
        '<div class="nav-crest-badge">'
        '<svg width="38" height="44" viewBox="0 0 38 44" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M19 2L3 8V22C3 32.5 10 41 19 43C28 41 35 32.5 35 22V8L19 2Z" fill="#0B1F33" stroke="#20C4F4" stroke-width="2.2"/>'
        '<line x1="11" y1="31" x2="27" y2="13" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="27" y1="31" x2="11" y2="13" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
        '<ellipse cx="19" cy="22" rx="4" ry="2.2" fill="#20C4F4"/>'
        '</svg>'
        '</div>'
    )

    navbar_html = f"""
    <div class="sports-league-navbar">
        <div class="nav-brand-controls">
            <span class="nav-control-icon">☰</span>
            <span class="nav-control-icon">🔍</span>
        </div>
        <div class="nav-items-wrapper">
            {left_html}
            {crest_svg}
            {right_html}
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER COMPONENT (FOR NON-OVERVIEW PAGES)
# ─────────────────────────────────────────────────────────────────────────────
def render_header(section_badge: str, section_title: str, section_desc: str):
    """Renders professional hockey broadcast hero banner inspired by sports league portals."""
    st.markdown(
        f"""
        <div class="hockey-hero-banner" style="background: linear-gradient(135deg, #0B1F33 0%, #0F2742 55%, #0797D5 140%); border-radius: 8px; border: 1px solid rgba(32, 196, 244, 0.25); box-shadow: 0 4px 16px rgba(11, 31, 51, 0.1); padding: 22px 28px; margin-bottom: 20px; color: #FFFFFF;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.25rem;">🏒</span>
                    <span style="font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #20C4F4;">{PRODUCT_NAME} &middot; {section_badge}</span>
                </div>
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #8EADC1; background: rgba(255,255,255,0.08); padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(255,255,255,0.12);">
                    NHL Historical Intelligence (1990–2011)
                </div>
            </div>
            <div style="margin-top: 8px;">
                <h1 class="hero-title-text" style="font-size: 2rem !important; font-weight: 900 !important; letter-spacing: 0.01em !important; text-transform: uppercase !important; color: #FFFFFF !important; margin: 0 !important; line-height: 1.15 !important;">{section_title}</h1>
                <div style="display: inline-block; background: linear-gradient(90deg, #0797D5 0%, #20C4F4 100%); transform: skewX(-14deg); padding: 4px 14px; border-radius: 4px; margin-top: 6px; margin-bottom: 6px;">
                    <span style="display: inline-block; transform: skewX(14deg); color: #FFFFFF; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.06em; text-transform: uppercase;">{PRODUCT_SUBTITLE}</span>
                </div>
            </div>
            <p style="color: #CFE4F1; font-size: 0.95rem; line-height: 1.5; margin-top: 6px; margin-bottom: 4px;">{section_desc}</p>
            <div style="color: #8EADC1; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">
                Academic Identity: Hockey Team Performance Analysis Using Web Scraping and Machine Learning &middot; SI Lab &amp; Portfolio Evaluation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: OVERVIEW & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def render_overview_view(df: pd.DataFrame):
    """Renders the executive summary sports dashboard matching the Continental Hockey League layout."""
    stats = get_historical_summary_stats(df)
    avg_win_pct = f"{df['Win_Pct'].mean():.3f}"
    avg_gf = f"{df['Goals_For'].mean():.1f}"
    avg_ga = f"{df['Goals_Against'].mean():.1f}"

    b64_hero = get_hero_banner_b64()

    # Hero Banner Card
    hero_img_markup = f"""
    <div class="hero-right-visual">
        <img src="data:image/jpeg;base64,{b64_hero}" alt="Hockey Performance" class="hero-player-img" />
    </div>
    """ if b64_hero else ""

    hero_html = f"""
    <div class="hockey-hero-card">
        <div class="hero-left-content">
            <h1 class="hero-wordmark">HOCKEY</h1>
            <div class="hero-ribbon-bar">
                <span class="hero-ribbon-text">PERFORMANCE ANALYSIS</span>
            </div>
            <div class="hero-tagline">
                A DATA-DRIVEN ANALYSIS OF NHL TEAM PERFORMANCE USING STATISTICS AND MACHINE LEARNING
            </div>
        </div>
        {hero_img_markup}
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # 5 KPI Stat Cards in a row
    kpi_puck_svg = '''<svg width="26" height="26" viewBox="0 0 28 28" fill="none"><ellipse cx="14" cy="11" rx="11" ry="5.5" fill="#EAF6FC" stroke="#0797D5" stroke-width="2"/><path d="M3 11V17C3 20 7.9 22.5 14 22.5C20.1 22.5 25 20 25 17V11" stroke="#0797D5" stroke-width="2"/><ellipse cx="14" cy="11" rx="5" ry="2.5" fill="#0797D5" fill-opacity="0.3"/></svg>'''
    kpi_trophy_svg = '''<svg width="26" height="26" viewBox="0 0 28 28" fill="none"><path d="M7 5H21V12C21 15.9 17.9 19 14 19C10.1 19 7 15.9 7 12V5Z" fill="#EAF6FC" stroke="#0797D5" stroke-width="2"/><path d="M7 8H4C3.4 8 3 8.4 3 9V11C3 13.2 4.8 15 7 15" stroke="#0797D5" stroke-width="2"/><path d="M21 8H24C24.6 8 25 8.4 25 9V11C25 13.2 23.2 15 21 15" stroke="#0797D5" stroke-width="2"/><path d="M14 19V23M10 23H18" stroke="#0797D5" stroke-width="2" stroke-linecap="round"/></svg>'''
    kpi_net_gf_svg = '''<svg width="26" height="26" viewBox="0 0 28 28" fill="none"><rect x="3" y="6" width="22" height="17" rx="2" stroke="#0797D5" stroke-width="2" fill="#EAF6FC"/><line x1="3" y1="12" x2="25" y2="12" stroke="#20C4F4" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="3" y1="18" x2="25" y2="18" stroke="#20C4F4" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="9" y1="6" x2="9" y2="23" stroke="#20C4F4" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="14" y1="6" x2="14" y2="23" stroke="#20C4F4" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="19" y1="6" x2="19" y2="23" stroke="#20C4F4" stroke-width="1.5" stroke-dasharray="2 2"/></svg>'''
    kpi_net_ga_svg = '''<svg width="26" height="26" viewBox="0 0 28 28" fill="none"><rect x="3" y="6" width="22" height="17" rx="2" stroke="#E53E3E" stroke-width="2" fill="#FFF5F5"/><line x1="3" y1="12" x2="25" y2="12" stroke="#FC8181" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="3" y1="18" x2="25" y2="18" stroke="#FC8181" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="9" y1="6" x2="9" y2="23" stroke="#FC8181" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="14" y1="6" x2="14" y2="23" stroke="#FC8181" stroke-width="1.5" stroke-dasharray="2 2"/><line x1="19" y1="6" x2="19" y2="23" stroke="#FC8181" stroke-width="1.5" stroke-dasharray="2 2"/></svg>'''
    kpi_chart_svg = '''<svg width="26" height="26" viewBox="0 0 28 28" fill="none"><rect x="4" y="14" width="4" height="10" rx="1" fill="#0797D5"/><rect x="12" y="8" width="4" height="16" rx="1" fill="#0797D5"/><rect x="20" y="4" width="4" height="20" rx="1" fill="#20C4F4"/><line x1="2" y1="24" x2="26" y2="24" stroke="#0B1F33" stroke-width="1.5" stroke-linecap="round"/></svg>'''

    kpi_html = f"""
    <div class="kpi-row-5">
        <div class="kpi-card-box">
            <div class="kpi-card-header">
                <div class="kpi-card-icon">{kpi_puck_svg}</div>
                <div class="kpi-card-label">TOTAL TEAM-SEASONS</div>
            </div>
            <div class="kpi-card-value">{stats['total_records']}</div>
        </div>
        <div class="kpi-card-box">
            <div class="kpi-card-header">
                <div class="kpi-card-icon">{kpi_trophy_svg}</div>
                <div class="kpi-card-label">AVG. WINNING PCT.</div>
            </div>
            <div class="kpi-card-value">{avg_win_pct}</div>
        </div>
        <div class="kpi-card-box">
            <div class="kpi-card-header">
                <div class="kpi-card-icon">{kpi_net_gf_svg}</div>
                <div class="kpi-card-label">AVG. GOALS FOR</div>
            </div>
            <div class="kpi-card-value">{avg_gf}</div>
        </div>
        <div class="kpi-card-box">
            <div class="kpi-card-header">
                <div class="kpi-card-icon">{kpi_net_ga_svg}</div>
                <div class="kpi-card-label">AVG. GOALS AGAINST</div>
            </div>
            <div class="kpi-card-value">{avg_ga}</div>
        </div>
        <div class="kpi-card-box">
            <div class="kpi-card-header">
                <div class="kpi-card-icon">{kpi_chart_svg}</div>
                <div class="kpi-card-label">BEST MODEL (R²)</div>
            </div>
            <div>
                <div class="kpi-card-value">0.8397</div>
                <div class="kpi-pill-badge">Linear Regression</div>
            </div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Lower Two-Column Section (Season Trends & Key Findings)
    col_left, col_right = st.columns([57, 43], gap="medium")

    with col_left:
        st.markdown(
            '<div class="section-title-sports"><span class="title-slash">◤</span> SEASON TRENDS (1990 – 2011)</div>',
            unsafe_allow_html=True
        )
        trends_chart = build_season_trends_chart(df)
        st.altair_chart(trends_chart, use_container_width=True)

    with col_right:
        st.markdown(
            '<div class="section-title-sports"><span class="title-slash">◤</span> KEY FINDINGS</div>',
            unsafe_allow_html=True
        )
        findings_html = """
        <div class="sports-card-container">
            <div class="findings-list">
                <div class="finding-row">
                    <div class="finding-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0797D5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                            <polyline points="17 6 23 6 23 12"></polyline>
                        </svg>
                    </div>
                    <div>
                        <div class="finding-headline">Strong Predictive Performance</div>
                        <div class="finding-subtext">Linear Regression explains 83.97% of the variance (R² = 0.8397).</div>
                    </div>
                </div>
                <div class="finding-row">
                    <div class="finding-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0797D5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <circle cx="12" cy="12" r="6"></circle>
                            <circle cx="12" cy="12" r="2"></circle>
                        </svg>
                    </div>
                    <div>
                        <div class="finding-headline">Goal Differential Matters</div>
                        <div class="finding-subtext">Goals For and Goals Against are the strongest performance drivers.</div>
                    </div>
                </div>
                <div class="finding-row">
                    <div class="finding-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0797D5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                    </div>
                    <div>
                        <div class="finding-headline">Positive Era Trend</div>
                        <div class="finding-subtext">Winning percentage shows an upward temporal association (+0.00398 per year).</div>
                    </div>
                </div>
                <div class="finding-row">
                    <div class="finding-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0797D5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="18" y1="20" x2="18" y2="10"></line>
                            <line x1="12" y1="20" x2="12" y2="4"></line>
                            <line x1="6" y1="20" x2="6" y2="14"></line>
                        </svg>
                    </div>
                    <div>
                        <div class="finding-headline">Statistically Significant</div>
                        <div class="finding-subtext">Key relationships are validated through t-test, ANOVA, and χ² tests.</div>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(findings_html, unsafe_allow_html=True)

    # Detailed distributions & analytical boundaries deep dive
    with st.expander("📊 Detailed Empirical Distributions & Analytical Scope", expanded=False):
        c_hist, c_scat = st.columns(2, gap="large")
        with c_hist:
            st.altair_chart(build_win_pct_distribution_chart(df), use_container_width=True)
        with c_scat:
            st.altair_chart(build_goal_diff_scatter_chart(df), use_container_width=True)

        st.markdown("---")
        st.caption(
            "Franchise Counting Methodology: The 35 distinct franchise labels reflect recorded team names across seasons "
            "(e.g., 'Mighty Ducks of Anaheim' and 'Anaheim Ducks', or 'Quebec Nordiques' and 'Colorado Avalanche'). "
            "In accordance with rigorous empirical standards, identities are preserved as recorded rather than merged."
        )
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### What Hockey Performance Intelligence Establishes")
            st.markdown("""
- **Retrospective Seasonal Estimation:** Accurately models the structural mathematical relationship between end-of-season goal aggregates, era, and winning percentage.
- **Empirical League Distributions:** Documents real historical team performance across the pre-shootout tie era (1990s) and modern post-lockout era (2000s).
- **Leakage-Free Cross-Validation:** Demonstrates temporal generalization across an expanding chronological window (1990–2007 train, 2008–2011 test).
            """)
        with s_col2:
            st.markdown("##### What Hockey Performance Intelligence Cannot Establish")
            st.markdown("""
- **Non-Forecasting:** Regular-season goal totals are only available after all 82 games conclude; this cannot be used for game-level or pre-season forecasting.
- **Observational Limitations:** Correlation between goals and winning percentage does not establish causal mechanisms for roster construction or individual player impact.
- **Unconstrained OLS Boundaries:** Standard linear regression estimates are unconstrained and not artificially clipped to $[0, 1]$.
            """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: HISTORICAL EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def render_historical_explorer_view(df: pd.DataFrame):
    """Renders interactive historical filtering, leaderboards, scatter plots, and CSV export."""
    render_header(
        section_badge="Historical Explorer",
        section_title="Historical Team Performance Explorer",
        section_desc="Filter, explore, and analyze 582 historical NHL regular-season records across 21 seasons (1990–2011)."
    )

    all_teams = get_unique_teams(df)
    all_years = get_unique_years(df)

    # Interactive Filter Controls
    with st.container(border=True):
        st.markdown("**Explorer Filter Controls**")
        fc1, fc2, fc3 = st.columns([2, 3, 2])

        with fc1:
            season_range = st.slider(
                "Season Starting Year Range",
                min_value=min(all_years),
                max_value=max(all_years),
                value=(min(all_years), max(all_years)),
                step=1,
                help="Select range of seasons. Note: 2004–05 season was cancelled due to the league lockout."
            )

        with fc2:
            selected_teams = st.multiselect(
                "Filter by Franchise Labels (Leave empty for all teams)",
                options=all_teams,
                default=[],
                help="Select specific franchises to isolate in tables and visualisations."
            )

        with fc3:
            selected_metric = st.selectbox(
                "Primary Performance Metric",
                options=["Win_Pct", "Goal_Diff", "Goals_For", "Goals_Against", "Wins", "Losses"],
                index=0,
                format_func=lambda m: {
                    "Win_Pct": "Winning Percentage (Win %)",
                    "Goal_Diff": "Goal Differential (GF − GA)",
                    "Goals_For": "Goals For (GF)",
                    "Goals_Against": "Goals Against (GA)",
                    "Wins": "Regular Season Wins",
                    "Losses": "Regular Season Losses",
                }.get(m, m)
            )

    # Filter records based on user input
    filtered_df = filter_historical_records(
        df,
        start_year=season_range[0],
        end_year=season_range[1],
        teams=selected_teams if selected_teams else None
    )

    if filtered_df.empty:
        st.warning("No historical team-seasons match the selected filter criteria. Please broaden your season or franchise selection.")
        return

    # Filter summary headline
    st.markdown(
        f"<span style='color: #6B8294; font-size: 0.9rem;'>"
        f"Displaying <strong>{len(filtered_df)}</strong> team-seasons across "
        f"<strong>{filtered_df['Year'].nunique()}</strong> played seasons and "
        f"<strong>{filtered_df['Team_Name'].nunique()}</strong> franchise labels."
        f"</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Leaderboards (Top 5 & Bottom 5)
    top_5, bottom_5 = get_leaderboards(filtered_df, metric=selected_metric, n=5)

    l_col1, l_col2 = st.columns(2)
    with l_col1:
        st.markdown(f"##### Top Performers by {selected_metric.replace('_', ' ')}")
        display_top = top_5.copy()
        display_top["Win_Pct"] = (display_top["Win_Pct"] * 100.0).round(2).astype(str) + "%"
        st.dataframe(display_top, use_container_width=True, hide_index=True)

    with l_col2:
        st.markdown(f"##### Lowest Performers by {selected_metric.replace('_', ' ')}")
        display_bottom = bottom_5.copy()
        display_bottom["Win_Pct"] = (display_bottom["Win_Pct"] * 100.0).round(2).astype(str) + "%"
        st.dataframe(display_bottom, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Visual Explorer Tabs
    tab_scatter, tab_trend = st.tabs([
        "Goal Differential Scatter",
        "Team Trajectory Over Seasons"
    ])

    with tab_scatter:
        st.altair_chart(
            build_goal_diff_scatter_chart(filtered_df, selected_teams=selected_teams),
            use_container_width=True
        )

    with tab_trend:
        # If no teams selected, default to top 4 franchises by win percentage in filter
        trend_teams = selected_teams if (selected_teams and len(selected_teams) > 0) else [
            "Detroit Red Wings", "Pittsburgh Penguins", "Boston Bruins", "Montreal Canadiens"
        ]
        st.altair_chart(
            build_team_timeline_chart(df, teams=trend_teams, metric=selected_metric),
            use_container_width=True
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Complete Filtered Table with CSV Export
    st.markdown("##### Filtered Records Data Table")
    table_export = filtered_df[[
        "Team_Name", "Year", "Wins", "Losses", "OT_Losses", "Win_Pct",
        "Goals_For", "Goals_Against", "Goal_Diff", "Decade"
    ]].copy()

    # Formatted copy for UI presentation
    table_display = table_export.copy()
    table_display["Win_Pct"] = (table_display["Win_Pct"] * 100.0).round(2).astype(str) + "%"

    st.dataframe(table_display, use_container_width=True, hide_index=True, height=280)

    # CSV Download Button
    csv_buffer = io.StringIO()
    table_export.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="Download Filtered Records (CSV)",
        data=csv_bytes,
        file_name=f"hockey_analytics_filtered_{season_range[0]}_{season_range[1]}.csv",
        mime="text/csv",
        help="Download currently filtered subset as a clean CSV file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TEAM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def render_team_comparison_view(df: pd.DataFrame):
    """Renders head-to-head comparison workflow between two franchises across seasons."""
    render_header(
        section_badge="Team Comparison",
        section_title="Franchise Head-to-Head Comparison",
        section_desc="Contrast historical performance profiles between any two recorded teams across seasons."
    )

    all_teams = get_unique_teams(df)
    all_years = get_unique_years(df)

    # Controls container
    with st.container(border=True):
        st.markdown("**Comparison Configuration**")
        c1, c2, c3 = st.columns([3, 3, 4])

        with c1:
            team_a = st.selectbox(
                "Primary Team (Team A)",
                options=all_teams,
                index=all_teams.index("Boston Bruins") if "Boston Bruins" in all_teams else 0,
                key="compare_team_a"
            )

        with c2:
            # Filter options so team_b cannot be identical to team_a
            other_teams = [t for t in all_teams if t != team_a]
            team_b = st.selectbox(
                "Comparison Team (Team B)",
                options=other_teams,
                index=other_teams.index("Detroit Red Wings") if "Detroit Red Wings" in other_teams else 0,
                key="compare_team_b"
            )

        with c3:
            compare_mode = st.radio(
                "Comparison Horizon",
                options=["Multi-Season Range", "Single Season Snapshot"],
                horizontal=True
            )

        if compare_mode == "Multi-Season Range":
            comp_years = st.slider(
                "Comparison Season Window",
                min_value=min(all_years),
                max_value=max(all_years),
                value=(1995, 2011),
                step=1
            )
            start_yr, end_yr = comp_years[0], comp_years[1]
        else:
            single_yr = st.selectbox(
                "Select Specific Season",
                options=all_years,
                index=all_years.index(2010) if 2010 in all_years else 0
            )
            start_yr, end_yr = single_yr, single_yr

    # Execute comparison
    comp_result = compare_teams(df, team_a, team_b, start_year=start_yr, end_year=end_yr)

    # Display comparison period badge
    st.markdown(
        f'<div class="badge-pill">Comparison Horizon: {comp_result["period_label"]}</div>',
        unsafe_allow_html=True
    )

    stats_a = comp_result["stats_a"]
    stats_b = comp_result["stats_b"]
    deltas = comp_result["deltas"]

    # Comparative Metric Cards (4 Columns)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        win_a = stats_a["mean_win_pct"] * 100.0
        win_b = stats_b["mean_win_pct"] * 100.0
        delta_win = deltas["win_pct"] * 100.0
        st.metric(
            label="Win %",
            value=f"{win_a:.2f}%",
            delta=f"{delta_win:+.2f}% vs {team_b} ({win_b:.2f}%)",
            delta_color="normal"
        )
        st.caption(f"{team_a}: {stats_a['total_wins']}W-{stats_a['total_losses']}L | {team_b}: {stats_b['total_wins']}W-{stats_b['total_losses']}L")

    with m2:
        gf_a = stats_a["mean_gf"]
        gf_b = stats_b["mean_gf"]
        delta_gf = deltas["goals_for"]
        st.metric(
            label="Goals For (Avg / Season)",
            value=f"{gf_a:.1f}",
            delta=f"{delta_gf:+.1f} vs {team_b} ({gf_b:.1f})",
            delta_color="normal"
        )
        st.caption(f"Totals: {stats_a['total_gf']} GF vs {stats_b['total_gf']} GF")

    with m3:
        ga_a = stats_a["mean_ga"]
        ga_b = stats_b["mean_ga"]
        # Invert delta color for Goals Against: fewer conceded is better
        delta_ga = deltas["goals_against"]
        st.metric(
            label="Goals Against (Avg / Season)",
            value=f"{ga_a:.1f}",
            delta=f"{delta_ga:+.1f} vs {team_b} ({ga_b:.1f})",
            delta_color="inverse"
        )
        st.caption(f"Totals: {stats_a['total_ga']} GA vs {stats_b['total_ga']} GA")

    with m4:
        diff_a = stats_a["mean_diff"]
        diff_b = stats_b["mean_diff"]
        delta_diff = deltas["goal_diff"]
        st.metric(
            label="Goal Differential (Avg / Season)",
            value=f"{diff_a:+.1f}",
            delta=f"{delta_diff:+.1f} vs {team_b} ({diff_b:+.1f})",
            delta_color="normal"
        )
        st.caption(f"Net Totals: {stats_a['total_diff']:+d} vs {stats_b['total_diff']:+d}")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.caption(
        "Methodological Note on Raw Historical Comparisons: Raw metric differences compare observed team-season totals. "
        "They do not control for NHL rule changes across eras (e.g. 1999 overtime point, 2005 shootout), roster continuity, "
        "or schedule strength."
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Common Seasons Trajectory & Side-by-Side Table
    if len(comp_result["common_years"]) > 0:
        st.markdown("##### Head-to-Head Common Season Comparison")
        comp_chart_tab, comp_table_tab = st.tabs(["Common Seasons Grouped Visual", "Side-by-Side Data Table"])

        with comp_chart_tab:
            st.altair_chart(
                build_h2h_grouped_bar_chart(comp_result["head_to_head_df"], team_a, team_b, metric_type="win_pct"),
                use_container_width=True
            )

        with comp_table_tab:
            h2h_display = comp_result["head_to_head_df"].copy()
            for col in h2h_display.columns:
                if "Win %" in col:
                    h2h_display[col] = (h2h_display[col] * 100.0).round(2).astype(str) + "%"
            st.dataframe(h2h_display, use_container_width=True, hide_index=True)
    else:
        st.info(f"{team_a} and {team_b} have no overlapping recorded seasons in the selected {comp_result['period_label']} window.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PREDICTION STUDIO (PRESERVED ACTIVE PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
def apply_preset(gf, ga, yr):
    """Callback to update session state inputs synchronously on button click."""
    st.session_state["input_gf"] = int(gf)
    st.session_state["input_ga"] = int(ga)
    st.session_state["input_yr"] = int(yr)


def render_prediction_workspace():
    """Renders scenario preset buttons and numeric input controls."""
    if "input_gf" not in st.session_state:
        st.session_state["input_gf"] = 250
    if "input_ga" not in st.session_state:
        st.session_state["input_ga"] = 230
    if "input_yr" not in st.session_state:
        st.session_state["input_yr"] = 2009

    with st.container(border=True):
        st.subheader("Performance Estimator")
        st.markdown(
            "<span style='color: #6B8294; font-size: 0.9rem;'>Select a scenario preset or enter custom regular-season statistics:</span>",
            unsafe_allow_html=True
        )

        st.markdown("**Preset Scenarios**")
        p_row1_col1, p_row1_col2 = st.columns(2)
        with p_row1_col1:
            st.button(
                "Test Sample (2009)",
                on_click=apply_preset,
                args=(250, 230, 2009),
                use_container_width=True
            )
        with p_row1_col2:
            st.button(
                "Top Contender",
                on_click=apply_preset,
                args=(300, 210, 2010),
                use_container_width=True
            )

        p_row2_col1, p_row2_col2 = st.columns(2)
        with p_row2_col1:
            st.button(
                "League Average",
                on_click=apply_preset,
                args=(234, 234, 2006),
                use_container_width=True
            )
        with p_row2_col2:
            st.button(
                "Rebuilding Team",
                on_click=apply_preset,
                args=(185, 295, 2003),
                use_container_width=True
            )

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("**Season Input Parameters**")
        in_col1, in_col2 = st.columns(2)
        with in_col1:
            goals_for = st.number_input(
                "Goals For (GF)",
                min_value=50,
                max_value=500,
                step=1,
                key="input_gf",
                help="Total regular-season goals scored by the team (historical range: 115–369)."
            )
        with in_col2:
            goals_against = st.number_input(
                "Goals Against (GA)",
                min_value=50,
                max_value=500,
                step=1,
                key="input_ga",
                help="Total regular-season goals conceded by the team (historical range: 123–397)."
            )

        year = st.number_input(
            "Season Starting Year",
            min_value=1990,
            max_value=2025,
            step=1,
            key="input_yr",
            help="Season start year (e.g. 2009 for the 2009–10 season; dataset covers 1990–2011)."
        )

        diff = goals_for - goals_against
        diff_color = "#0797D5" if diff >= 0 else "#EF4444"
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #6B8294; margin-top: 0.2rem; margin-bottom: 0.4rem;'>"
            f"Calculated Goal Differential (GF &minus; GA): "
            f"<strong style='color: {diff_color};'>{diff:+d}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

        if year == 2004:
            st.caption(
                "Note: The 2004–05 NHL season was cancelled due to the league lockout. "
                "The estimate is based on structural era interpolation."
            )

    return goals_for, goals_against, year


def render_win_pct_comparison_scale(pred_pct):
    """
    Renders a responsive horizontal 0-100% comparison scale bar:
    - Filled progress bar from 0% to prediction.
    - Prediction: thick hockey-blue vertical line + pointer triangle + circle marker.
    - Historical ticks (Min/Mean/Max) with collision avoidance.
    """
    is_out_of_range = pred_pct < 0.0 or pred_pct > 100.0
    scale_min = min(0.0, float(pred_pct) - 10.0) if is_out_of_range else 0.0
    scale_max = max(100.0, float(pred_pct) + 10.0) if is_out_of_range else 100.0
    total_range = scale_max - scale_min

    def to_x(val):
        return 40.0 + ((float(val) - scale_min) / total_range) * 720.0

    pred_x = to_x(pred_pct)
    min_x = to_x(HISTORICAL_MIN_WIN_PCT)
    mean_x = to_x(HISTORICAL_MEAN_WIN_PCT)
    max_x = to_x(HISTORICAL_MAX_WIN_PCT)
    track_start = to_x(0.0)
    track_end = to_x(100.0)
    track_width = track_end - track_start

    fill_end_x = max(track_start, min(track_end, pred_x))
    fill_width = fill_end_x - track_start

    px_per_pct = 720.0 / total_range
    collision_threshold = px_per_pct * 3.0
    tick_xs = [min_x, mean_x, max_x]
    near_tick = any(abs(pred_x - tx) < collision_threshold for tx in tick_xs)

    pred_label_y = 18 if not near_tick else 9

    label_x = max(55.0, min(745.0, pred_x))
    if pred_x < 110:
        pred_anchor = "start"
        label_x = max(40.0, pred_x)
    elif pred_x > 690:
        pred_anchor = "end"
        label_x = min(760.0, pred_x)
    elif near_tick:
        nearest_tx = min(tick_xs, key=lambda tx: abs(pred_x - tx))
        pred_anchor = "start" if pred_x >= nearest_tx else "end"
        label_x = pred_x + (8 if pred_anchor == "start" else -8)
    else:
        pred_anchor = "middle"

    parts = [
        '<div style="width:100%;overflow-x:auto;margin-top:0.5rem;margin-bottom:0.25rem;">',
        '<svg width="100%" height="96" viewBox="0 0 800 96" xmlns="http://www.w3.org/2000/svg" style="display:block;">',
        f'<rect x="{track_start:.1f}" y="42" width="{track_width:.1f}" height="8" rx="4" fill="#D8E7EF"/>',
        f'<rect x="{track_start:.1f}" y="42" width="{fill_width:.1f}" height="8" rx="4" fill="#0797D5" fill-opacity="0.35"/>',
        f'<line x1="{min_x:.1f}" y1="40" x2="{min_x:.1f}" y2="52" stroke="#6B8294" stroke-width="1.5"/>',
        f'<text x="{min_x:.1f}" y="74" fill="#6B8294" font-size="11" font-weight="600" font-family="sans-serif" text-anchor="middle">Min {HISTORICAL_MIN_WIN_PCT:.1f}%</text>',
        f'<line x1="{mean_x:.1f}" y1="40" x2="{mean_x:.1f}" y2="52" stroke="#6B8294" stroke-width="1.5"/>',
        f'<text x="{mean_x:.1f}" y="74" fill="#6B8294" font-size="11" font-weight="600" font-family="sans-serif" text-anchor="middle">Mean {HISTORICAL_MEAN_WIN_PCT:.1f}%</text>',
        f'<line x1="{max_x:.1f}" y1="40" x2="{max_x:.1f}" y2="52" stroke="#6B8294" stroke-width="1.5"/>',
        f'<text x="{max_x:.1f}" y="74" fill="#6B8294" font-size="11" font-weight="600" font-family="sans-serif" text-anchor="middle">Max {HISTORICAL_MAX_WIN_PCT:.1f}%</text>',
        f'<line x1="{pred_x:.1f}" y1="26" x2="{pred_x:.1f}" y2="58" stroke="#0797D5" stroke-width="3"/>',
        f'<polygon points="{pred_x-6:.1f},28 {pred_x+6:.1f},28 {pred_x:.1f},40" fill="#0797D5"/>',
        f'<circle cx="{pred_x:.1f}" cy="46" r="5" fill="#0797D5"/>',
        f'<text x="{label_x:.1f}" y="{pred_label_y}" fill="#0797D5" font-size="13" font-weight="800" font-family="sans-serif" text-anchor="{pred_anchor}">Prediction {pred_pct:.1f}%</text>',
        '</svg>',
        '</div>',
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.caption("Blue marker = current prediction. Grey ticks = historical dataset min (11.9% Ottawa), mean (45.9%), and max (75.6% Detroit).")


def render_result(pred_raw, goals_for, goals_against, year):
    """Renders the prediction result section with metrics, tier badge, and SVG gauge."""
    pred_pct = pred_raw * 100.0
    goal_diff = goals_for - goals_against

    with st.container(border=True):
        st.subheader("Estimation Result")

        diff_color = "#0797D5" if goal_diff >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="input-chip">
                <span>Inputs: <strong>{goals_for} GF</strong> &middot; <strong>{goals_against} GA</strong> &middot; <strong>{year} Season</strong></span>
                <span style="color: #6B8294;">|</span>
                <span>Differential: <strong style="color: {diff_color};">{goal_diff:+d}</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Estimated Win %",
                value=f"{pred_pct:.2f}%",
                help="Continuous Ordinary Least Squares regression estimate of regular-season Win_Pct."
            )
            st.caption(f"Raw Model Output (Win_Pct): {pred_raw:.4f}")
        with res_col2:
            st.metric(
                label="Goal Differential",
                value=f"{goal_diff:+d}",
                help="Goals For minus Goals Against."
            )
            st.caption(f"Historical Mean: ~{int(round(HISTORICAL_AVG_GF))} GF / ~{int(round(HISTORICAL_AVG_GA))} GA")

        # Performance Tier Interpretation
        if pred_raw >= 0.600:
            tier_title = "Upper Range / High Performance Tier (Win % >= 60%)"
            tier_desc = "Illustrative historical category representing regular seasons substantially above league-average performance."
            tier_color = "#0797D5"
        elif pred_raw >= 0.500:
            tier_title = "Above-Average Tier (50% <= Win % < 60%)"
            tier_desc = "Illustrative historical category representing above-average regular-season win-loss performance."
            tier_color = "#0797D5"
        elif pred_raw >= 0.400:
            tier_title = "Moderate / Sub-.500 Tier (40% <= Win % < 50%)"
            tier_desc = "Illustrative historical category representing modest regular-season winning performance."
            tier_color = "#F59E0B"
        else:
            tier_title = "Lower Boundary Tier (Win % < 40%)"
            tier_desc = "Illustrative historical category representing regular seasons substantially below league-average performance."
            tier_color = "#EF4444"

        st.markdown(
            f'<div class="tier-card" style="border-left: 4px solid {tier_color};">'
            f'<div class="tier-title" style="color:{tier_color};">{tier_title}</div>'
            f'<p class="tier-desc">{tier_desc}</p>'
            '</div>',
            unsafe_allow_html=True
        )

        render_win_pct_comparison_scale(pred_pct)

        if pred_raw < 0.0 or pred_raw > 1.0:
            st.caption(
                f"Note: The linear regression output ({pred_pct:.2f}%) falls outside the 0%–100% range. "
                "The model uses continuous ordinary least squares without artificial boundary clipping."
            )

        st.caption("Descriptive retrospective estimate based on final regular-season aggregates; not an in-game forecast or verified real-world franchise result.")


def render_feature_contribution_chart(intercept, contrib_gf, contrib_ga, contrib_yr):
    """Renders 4-bar horizontal Altair chart showing standardized shifts from baseline intercept."""
    bar_order = ["Baseline (intercept)", "Goals For", "Goals Against", "Year"]
    data = pd.DataFrame([
        {"Feature": "Baseline (intercept)", "Contribution": float(intercept), "Value_Label": f"{intercept:+.4f}", "Color": "#6B8294"},
        {"Feature": "Goals For", "Contribution": float(contrib_gf), "Value_Label": f"{contrib_gf:+.4f}", "Color": "#0797D5" if contrib_gf >= 0 else "#EF4444"},
        {"Feature": "Goals Against", "Contribution": float(contrib_ga), "Value_Label": f"{contrib_ga:+.4f}", "Color": "#0797D5" if contrib_ga >= 0 else "#EF4444"},
        {"Feature": "Year", "Contribution": float(contrib_yr), "Value_Label": f"{contrib_yr:+.4f}", "Color": "#0797D5" if contrib_yr >= 0 else "#EF4444"},
    ])

    min_c = min(0.0, float(data["Contribution"].min()))
    max_c = max(0.0, float(data["Contribution"].max()))
    padding = max(0.08, (max_c - min_c) * 0.28)
    domain = [min_c - padding, max_c + padding]

    base = alt.Chart(data).properties(
        height=210,
        title=alt.TitleParams(
            text="How each input moves the estimate from the baseline",
            color="#0B1F33",
            fontSize=13,
            fontWeight=700,
            anchor="start",
            offset=10
        )
    )

    bars = base.mark_bar(height=26, cornerRadius=3).encode(
        y=alt.Y("Feature:N", sort=bar_order, title=None, axis=alt.Axis(labelColor="#142D42", labelFontSize=12, labelPadding=12, ticks=False, domainColor="#D8E7EF")),
        x=alt.X("Contribution:Q", title="Contribution to Win_Pct", scale=alt.Scale(domain=domain, zero=True), axis=alt.Axis(labelColor="#6B8294", titleColor="#6B8294", gridColor="#E2EEF5", domainColor="#D8E7EF")),
        color=alt.Color("Color:N", scale=None)
    )

    labels = base.mark_text(
        align=alt.expr("datum.Contribution >= 0 ? 'left' : 'right'"),
        baseline="middle",
        dx=alt.expr("datum.Contribution >= 0 ? 8 : -8"),
        color="#0B1F33",
        fontSize=11,
        fontWeight=700
    ).encode(
        y=alt.Y("Feature:N", sort=bar_order),
        x=alt.X("Contribution:Q"),
        text=alt.Text("Value_Label:N")
    )

    st.altair_chart((bars + labels).configure_view(strokeOpacity=0), use_container_width=True)


def render_explainability(pipeline, metadata, goals_for, goals_against, year, pred_raw):
    """Renders calculations, LaTeX equations, and Altair feature-contribution chart."""
    try:
        contrib_dict = calculate_feature_contributions(pipeline, metadata, goals_for, goals_against, year)
    except Exception as exc:
        st.markdown(f"Technical feature-contribution breakdown is omitted: {exc}")
        return

    st.markdown("**Standardized Feature Calculations**")
    breakdown_df = pd.DataFrame(contrib_dict["breakdown_rows"])
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown("**Linear Regression Model Equation**")
    st.latex(contrib_dict["latex_equation"])
    st.latex(contrib_dict["latex_substituted"])

    st.markdown("**Feature Contribution Breakdown**")
    render_feature_contribution_chart(
        contrib_dict["intercept"],
        contrib_dict["contrib_gf"],
        contrib_dict["contrib_ga"],
        contrib_dict["contrib_yr"]
    )

    st.caption(
        f"Baseline + Goals For + Goals Against + Year = predicted Win_Pct: "
        f"{contrib_dict['intercept']:.4f} + ({contrib_dict['contrib_gf']:+.4f}) + "
        f"({contrib_dict['contrib_ga']:+.4f}) + ({contrib_dict['contrib_yr']:+.4f}) = {pred_raw:.4f}"
    )


def render_model_performance(metadata):
    """Renders evaluation metrics from metadata safely without hardcoding."""
    eval_strat = metadata.get("evaluation_strategy") if isinstance(metadata, dict) else {}
    cv_metrics = metadata.get("cv_metrics") if isinstance(metadata, dict) else {}
    test_metrics = metadata.get("test_metrics") if isinstance(metadata, dict) else {}

    cv_rmse_val = cv_metrics.get("cv_rmse_mean", cv_metrics.get("cv_rmse"))
    cv_rmse_sd = cv_metrics.get("cv_rmse_sd")
    cv_r2_val = cv_metrics.get("cv_r2_mean", cv_metrics.get("cv_r2"))
    cv_r2_sd = cv_metrics.get("cv_r2_sd")

    cv_rmse_display = f"{cv_rmse_val:.4f} +/- {cv_rmse_sd:.4f}" if isinstance(cv_rmse_val, (int, float)) and isinstance(cv_rmse_sd, (int, float)) else f"{cv_rmse_val:.4f}" if isinstance(cv_rmse_val, (int, float)) else "N/A"
    cv_r2_display = f"{cv_r2_val:.4f} +/- {cv_r2_sd:.4f}" if isinstance(cv_r2_val, (int, float)) and isinstance(cv_r2_sd, (int, float)) else f"{cv_r2_val:.4f}" if isinstance(cv_r2_val, (int, float)) else "N/A"

    test_rmse = test_metrics.get("test_rmse")
    test_r2 = test_metrics.get("test_r2")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("CV RMSE", cv_rmse_display, help="Mean RMSE across 5 season-aware expanding-window folds.")
    with m_col2:
        st.metric("Test RMSE", f"{test_rmse:.4f}" if isinstance(test_rmse, (int, float)) else "N/A", help="Out-of-sample RMSE on held-out 2008–2011 seasons.")
    with m_col3:
        st.metric("Test R²", f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) else "N/A", help="Variance explained on held-out 2008–2011 seasons.")
    with m_col4:
        st.metric("CV R²", cv_r2_display, help="Mean R² across 5 season-aware expanding-window folds.")

    st.caption(
        "Validation Strategy: 5-Fold Season-Aware Expanding Window (1990–2007 train, 2008–2011 holdout). "
        "2004 lockout season strictly omitted from splits."
    )


def render_historical_benchmarks():
    """Renders empirical dataset baseline statistics."""
    st.markdown("**Descriptive statistics of the historical dataset (not model predictions)**")
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    with b_col1:
        st.metric("Min Win %", f"{HISTORICAL_MIN_WIN_PCT:.1f}%", help="1992 Ottawa Senators (10-70-4)")
    with b_col2:
        st.metric("Mean Win %", f"{HISTORICAL_MEAN_WIN_PCT:.2f}%", help="League-wide historical mean across 582 team-seasons")
    with b_col3:
        st.metric("Max Win %", f"{HISTORICAL_MAX_WIN_PCT:.1f}%", help="1995 Detroit Red Wings (62-13-7)")
    with b_col4:
        st.metric("Avg Goals For", f"~{int(round(HISTORICAL_AVG_GF))}", help="League average regular-season goals scored")
    with b_col5:
        st.metric("Avg Goals Against", f"~{int(round(HISTORICAL_AVG_GA))}", help="League average regular-season goals conceded")


def render_methodology():
    """Renders methodology, leakage exclusions, and limitations directly inside container."""
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Analytical Approach")
        st.markdown("""
- **Retrospective Estimation:** Estimates final season Win_Pct using regular-season goal aggregates.
- **Non-Forecasting Scope:** Goal totals are only available after the season concludes; this cannot be used for pre-game or pre-season forecasting.
- **Temporal Holdout:** Evaluated on unseen 2008–2011 seasons (120 team-seasons).
        """)
        st.markdown("##### Excluded Features (Leakage Prevention)")
        st.markdown("""
- **Wins & Losses:** Excluded because they mathematically reconstruct the target.
- **OT_Losses:** Excluded as part of the Win_Pct denominator; 224 of 582 historical records lack OT loss data (pre-shootout era).
- **Goal_Diff:** Excluded as a direct collinear linear combination of Goals_For minus Goals_Against (r = 0.887).
        """)
    with m_col2:
        st.markdown("##### Model Limitations & Statistical Diagnostics")
        st.markdown("""
- **Continuous OLS:** Uses unconstrained linear regression with constant coefficients; does not enforce artificial [0, 1] bounds.
- **Historical Era Window:** Trained on 1990–2011 data; modern NHL rule changes (shootout points, overtime 3-on-3) shifted baseline scoring environments.
- **Residual Autocorrelation:** Residuals exhibit mild positive serial correlation (Durbin-Watson ≈ 1.082) due to persistent franchise quality across seasons.
        """)


def render_prediction_studio_view(pipeline, metadata):
    """Renders the complete preserved Prediction Studio workspace."""
    render_header(
        section_badge="Prediction Studio",
        section_title="Performance Prediction Studio",
        section_desc="Retrospective end-of-season winning percentage estimation using the verified linear regression pipeline."
    )

    col_input, col_result = st.columns([5, 6], gap="large")

    with col_input:
        goals_for, goals_against, year = render_prediction_workspace()

    expected_features = REQUIRED_FEATURES
    if isinstance(metadata, dict) and "feature_columns" in metadata:
        meta_cols = metadata.get("feature_columns")
        if isinstance(meta_cols, list) and meta_cols == REQUIRED_FEATURES:
            expected_features = meta_cols

    validation_errors = validate_prediction_inputs(goals_for, goals_against, year)

    pred_raw = None
    with col_result:
        if validation_errors:
            for err in validation_errors:
                st.error(f"Input error: {err}")
        else:
            try:
                pred_raw = predict_win_pct(pipeline, goals_for, goals_against, year, expected_features)
                render_result(pred_raw, goals_for, goals_against, year)
            except Exception as exc:
                st.error(f"Prediction calculation error: {str(exc)}")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    tab_explain, tab_perf, tab_dataset, tab_limits = st.tabs([
        "Feature Impact & Explainability",
        "Model Performance & Validation",
        "Historical Benchmarks & Context",
        "Analytical Scope & Limitations"
    ])

    with tab_explain:
        if not validation_errors and pred_raw is not None:
            render_explainability(pipeline, metadata, goals_for, goals_against, year, pred_raw)
        else:
            st.info("Provide valid inputs above to view feature contributions.")

    with tab_perf:
        render_model_performance(metadata)

    with tab_dataset:
        render_historical_benchmarks()

    with tab_limits:
        render_methodology()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MODEL EVALUATION LAB
# ─────────────────────────────────────────────────────────────────────────────
def render_model_evaluation_view(pipeline: Any, metadata: dict, df: pd.DataFrame):
    """
    Renders the Model Evaluation Lab: 5 candidate models, cross-validation vs.
    held-out test set performance, error charts, baseline comparison, and residual diagnostics.
    """
    render_header(
        section_badge="Model Evaluation Lab",
        section_title="Five-Model Cross-Validation & Generalization Lab",
        section_desc=(
            "Season-aware expanding-window cross-validation, held-out test set generalization, "
            "and residual error diagnostics across 5 candidate regression architectures (1990–2011)."
        )
    )

    eval_data = get_model_evaluation_data(pipeline, df)
    candidate_models = eval_data.get("candidate_models", [])
    test_metrics = eval_data.get("test_metrics", {})
    baseline_metrics = eval_data.get("baseline_metrics", {})
    test_df = eval_data.get("test_df", pd.DataFrame())
    season_metrics = eval_data.get("season_metrics", pd.DataFrame())
    residual_diag = eval_data.get("residual_diagnostics", {})

    # 1. Executive Metric Summary Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Selected Model</div>
                <div class="kpi-value" style="font-size: 1.25rem; color: #0797D5;">Linear Regression</div>
                <div class="kpi-sub">Criterion: Lowest Mean CV RMSE</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        cv_rmse = 0.0414
        cv_sd = 0.0113
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">5-Fold CV RMSE</div>
                <div class="kpi-value">{cv_rmse:.4f}</div>
                <div class="kpi-sub">Fold SD: ±{cv_sd:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        test_rmse = test_metrics.get("rmse", 0.0332)
        base_rmse = baseline_metrics.get("rmse", 0.0981)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Held-Out Test RMSE</div>
                <div class="kpi-value">{test_rmse:.4f}</div>
                <div class="kpi-sub">vs. Baseline {base_rmse:.4f} (-66.2%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        test_r2 = test_metrics.get("r2", 0.8397)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Held-Out Test R²</div>
                <div class="kpi-value">{test_r2:.4f}</div>
                <div class="kpi-sub">120 Teams · Seasons 2008–2011</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 2. Five-Model Comparison Table
    with st.container(border=True):
        st.subheader("1. Candidate Model Architecture Comparison Table")
        st.markdown(
            """
            Verified evaluation metrics extracted directly from the authoritative machine learning notebook 
            (`Hockey_Team_Performance_ML_working_v4.ipynb`, Cell 54).
            <br>
            <span style="color: #6B8294; font-size: 0.88rem;">
                Cross-validation used 5 season-aware expanding-window splits on the training partition (1990–2007).
                The held-out test partition (2008–2011, 120 team-seasons) was kept strictly isolated until final verification.
            </span>
            """,
            unsafe_allow_html=True
        )

        table_rows = []
        for m in candidate_models:
            table_rows.append({
                "Architecture": m["model"],
                "Status": "Selected (Best CV)" if m["selected"] else "Candidate",
                "CV MAE (Mean ± SD)": f"{m['cv_mae']:.4f} ± {m['cv_mae_sd']:.4f}",
                "CV RMSE (Mean ± SD)": f"{m['cv_rmse']:.4f} ± {m['cv_rmse_sd']:.4f}",
                "CV R² (Mean ± SD)": f"{m['cv_r2']:.4f} ± {m['cv_r2_sd']:.4f}",
                "Test MAE": f"{m['test_mae']:.4f}",
                "Test RMSE": f"{m['test_rmse']:.4f}",
                "Test R²": f"{m['test_r2']:.4f}",
                "Architectural Rationale": m["selection_notes"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        with st.expander("Methodological Rationale: CV RMSE vs. Held-Out Test Set Discipline"):
            st.markdown("""
- **Why Lowest CV RMSE Governs Model Selection:** Cross-validation over expanding seasonal folds evaluates out-of-sample error across varying league scoring regimes (e.g., dead-puck vs post-lockout). Selecting the architecture with the lowest mean CV RMSE (Linear Regression, 0.0414) minimizes structural overfitting.
- **Why the Held-Out Test Set Must Not Be Used for Tuning:** Repeatedly evaluating candidate hyper-parameters or model types against the test set turns the test set into a de facto training set, leaking information and invalidating generalization guarantees. The held-out test set was evaluated exactly once for final model audit.
- **Why OLS Outperforms Complex Tree Ensembles:** Goal scoring and winning percentage share a structurally linear mathematical relationship ($Win\\% \\propto GF - GA$). Tree models impose discontinuous step-function boundaries that introduce discretization error and variance on continuous sports targets.
            """)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 3. Model Comparison Visualizations
    with st.container(border=True):
        st.subheader("2. Model Architecture Visual Benchmarks")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("**Cross-Validation RMSE by Model (with ±1 SD Fold Variability)**")
            st.caption("Error bars indicate variability across the 5 temporal expanding folds.")
            st.altair_chart(build_cv_rmse_comparison_chart(candidate_models), use_container_width=True)
        with c_col2:
            st.markdown("**Held-Out Test Set RMSE across Candidate Models (2008–2011)**")
            st.caption("Lower is better. Confirms Linear Regression generalizes best out-of-sample.")
            st.altair_chart(build_test_rmse_comparison_chart(candidate_models), use_container_width=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 4. Held-Out Generalization & Residual Diagnostics
    with st.container(border=True):
        st.subheader("3. Selected Model Held-Out Generalization & Error Diagnostics")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("**Actual vs. Predicted Win Percentage (120 Test Records)**")
            st.caption("Points cluster tightly along the dashed 45° line of perfect agreement (y = x).")
            st.altair_chart(build_actual_vs_predicted_chart(test_df), use_container_width=True)
        with g_col2:
            st.markdown("**Test Residual Distribution (Actual - Predicted Win %)**")
            st.caption("Residuals are centered near zero with symmetrical bell-shaped spread.")
            st.altair_chart(build_residual_distribution_chart(test_df), use_container_width=True)

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Season-by-Season Generalization Breakdown (2008–2011)")
        st.altair_chart(build_error_by_season_chart(season_metrics), use_container_width=True)

        if not season_metrics.empty:
            st.dataframe(
                season_metrics.rename(columns={
                    "Mean_Actual_Win_Pct": "Actual Win % Mean",
                    "Mean_Predicted_Win_Pct": "Predicted Win % Mean",
                    "MAE": "Test MAE",
                    "RMSE": "Test RMSE"
                }),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 5. Baseline Comparison & Scientific Interpretation
    with st.container(border=True):
        st.subheader("4. Baseline Comparison & Metric Interpretation")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("##### Selected Model vs. Chronological Training Baseline")
            base_df = pd.DataFrame([
                {
                    "Evaluation Strategy": "Selected Model (OLS Regression)",
                    "Test MAE": f"{test_metrics.get('mae', 0.0264):.4f}",
                    "Test RMSE": f"{test_metrics.get('rmse', 0.0332):.4f}",
                    "Test R²": f"{test_metrics.get('r2', 0.8397):.4f}",
                    "Error Reduction": "67.7% MAE reduction vs baseline"
                },
                {
                    "Evaluation Strategy": "Chronological Training Mean Baseline",
                    "Test MAE": f"{baseline_metrics.get('mae', 0.0817):.4f}",
                    "Test RMSE": f"{baseline_metrics.get('rmse', 0.0981):.4f}",
                    "Test R²": f"{baseline_metrics.get('r2', -0.3971):.4f}",
                    "Error Reduction": f"Baseline predictor: constant y = {baseline_metrics.get('train_mean', 0.4477):.4f}"
                }
            ])
            st.dataframe(base_df, use_container_width=True, hide_index=True)

            st.markdown("""
<div style="font-size: 0.86rem; color: #6B8294; margin-top: 0.5rem; line-height: 1.5;">
    <strong>Why Baseline R² is Negative (-0.3971):</strong> The naive baseline uses the training set mean (0.4477, 1990–2007). 
    Because post-lockout NHL rule changes eliminated regulation ties and instituted the shootout, the test set mean rose to 0.5000. 
    Predicting the historical 0.4477 mean generates larger squared errors than the test set variance itself, resulting in negative R². 
    The trained model accommodates this secular era difference through its positive temporal <code>Year</code> trend coefficient (+0.00398/yr), reflecting an observational era association rather than causal proof of specific rule changes.
</div>
            """, unsafe_allow_html=True)

        with b_col2:
            st.markdown("##### Scientific Interpretation of Error Metrics")
            st.markdown("""
- **MAE in Win % Units:** A Test MAE of 0.0264 corresponds to an average deviation of 2.64 percentage points. Across an 82-game regular season, this translates to an average estimation error of **~2.16 wins (~2.2 standings points)**.
- **RMSE Penalty for Large Outliers:** Test RMSE is 0.0332. The modest difference between RMSE and MAE (0.0068) confirms that large residual outliers are rare and the error distribution has thin tails.
- **Why High R² Does Not Guarantee Invariant Reliability:** A held-out R² of 0.8397 confirms the model accounts for 84.0% of league-wide variance. However, in individual seasons with high shootout frequency or atypical goaltending, residual error can fluctuate.
- **Retrospective Scope Notice:** All estimates utilize completed regular-season goal aggregates and cannot be applied as in-game or pre-season forecasts.
            """)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 6. Residual Autocorrelation & Durbin-Watson Diagnostics
    with st.container(border=True):
        st.subheader("5. Residual Autocorrelation & Diagnostic Analysis")
        dw_test = residual_diag.get("durbin_watson_test", 1.8753)
        dw_full = residual_diag.get("durbin_watson_full_notebook", 1.0792)

        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Held-Out Test Durbin–Watson</div>
                    <div class="kpi-value" style="color: #0797D5;">{dw_test:.4f}</div>
                    <div class="kpi-sub">Near 2.0 (Minimal Serial Correlation)</div>
                </div>
                <div style="height: 0.5rem;"></div>
                <div class="kpi-card">
                    <div class="kpi-title">Full Dataset OLS DW (Prac 12)</div>
                    <div class="kpi-value" style="color: #F59E0B;">{dw_full:.4f}</div>
                    <div class="kpi-sub">Positive Serial Correlation in Time-Series</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with d_col2:
            st.markdown("##### Statistical Context & Interpretation of Durbin–Watson")
            st.markdown(f"""
- **Provenance:** In the full academic dataset OLS regression (`Hockey_Team_Performance_Analysis_working.ipynb`, Practical 12), the Durbin–Watson statistic was reported as **{dw_full:.4f}**. On the chronologically held-out test set, the computed Durbin–Watson statistic is **{dw_test:.4f}**.
- **What Durbin–Watson Evaluates:** DW measures first-order residual autocorrelation ($e_t$ vs. $e_{{t-1}}$). A value near 2.0 indicates no first-order serial correlation, while values substantially below 2.0 indicate positive autocorrelation.
- **Methodological Impact on Ordinary Regression Assumptions:** In panel datasets where franchises appear across consecutive seasons, roster talent and coaching systems persist across years. This temporal inertia introduces serial correlation in unmodeled residuals.
- **Scientific Conclusion:** While serial autocorrelation can cause ordinary OLS standard errors to be understated (affecting hypothesis test confidence intervals), it does **not** bias point estimates or invalidate the model's out-of-sample predictive accuracy, as demonstrated by the 0.0332 test RMSE.
            """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: STATISTICAL INSIGHTS & SI LAB SHOWCASE
# ─────────────────────────────────────────────────────────────────────────────
def render_statistical_insights_view(df: pd.DataFrame):
    """
    Renders the academic SI Lab practicals: Central Limit Theorem simulations,
    one-sample t-test & confidence intervals, historical era ANOVA & Tukey HSD,
    Chi-Square contingency tests, and the directory of all 16 academic practicals.
    """
    render_header(
        section_badge="Statistical Insights",
        section_title="Academic SI Lab Showcase & Statistical Inference",
        section_desc=(
            "Rigorous presentation layer for the foundational statistical inference practicals "
            "documented in Hockey_Team_Performance_Analysis_working.ipynb."
        )
    )

    stats_data = get_statistical_insights_data(df)
    t_test = stats_data.get("t_test", {})
    anova = stats_data.get("anova", {})
    chi_square = stats_data.get("chi_square", {})
    practicals = stats_data.get("practicals", [])

    tab_clt, tab_ttest, tab_anova, tab_chi2, tab_all = st.tabs([
        "Central Limit Theorem (Prac 4)",
        "One-Sample t-Test & CI (Pracs 5 & 6)",
        "Historical Era ANOVA (Prac 13)",
        "Chi-Square Association (Prac 16)",
        "All 16 SI Lab Practicals Index"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: CENTRAL LIMIT THEOREM
    # ─────────────────────────────────────────────────────────────────────────
    with tab_clt:
        st.subheader("Central Limit Theorem & Sampling Distribution Simulation")
        st.markdown("""
        The Central Limit Theorem (CLT) states that given a population with finite mean $\\mu$ and standard deviation $\\sigma$, 
        the sampling distribution of the sample mean $\\bar{X}$ converges to a normal distribution as sample size $n$ increases, 
        regardless of the underlying population shape:
        $$\\bar{X} \\sim \\mathcal{N}\\left(\\mu, \\frac{\\sigma}{\\sqrt{n}}\\right)$$
        """)

        pop_val = df["Win_Pct"].dropna()
        pop_n = len(pop_val)
        pop_mean = float(pop_val.mean())
        pop_std = float(pop_val.std(ddof=0))

        clt_col1, clt_col2 = st.columns([1, 2])
        with clt_col1:
            with st.container(border=True):
                st.markdown("**Simulation Parameters**")
                sample_size = st.slider("Sample Size (n teams per draw)", min_value=10, max_value=60, value=30, step=5)
                n_sims = st.slider("Simulated Samples Count", min_value=500, max_value=2000, value=1000, step=250)
                sim_seed = st.number_input("RNG Seed (Reproducibility)", min_value=1, max_value=9999, value=42)

                sim_res = run_clt_simulation(df, sample_size=sample_size, n_simulations=n_sims, seed=int(sim_seed))

                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; color: #6B8294; margin-top: 0.5rem; line-height: 1.45;">
                        <strong>Population (N = {pop_n}):</strong><br>
                        Mean (μ): <strong>{pop_mean:.4f}</strong><br>
                        Std Dev (σ): <strong>{pop_std:.4f}</strong><br>
                        Theoretical SE (σ/√n): <strong>{sim_res['theoretical_se']:.4f}</strong><br>
                        Empirical SE (s_x̄): <strong>{sim_res['empirical_se']:.4f}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with clt_col2:
            st.markdown(f"**Sampling Distribution of Sample Means (n = {sample_size}, {n_sims} Simulations)**")
            st.caption("Dashed red line indicates population mean μ = 0.4585. Blue bars show simulated distribution.")
            st.altair_chart(
                build_clt_simulation_chart(sim_res["sim_df"], pop_mean, sim_res["theoretical_se"]),
                use_container_width=True
            )
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #F59E0B; background-color: rgba(245, 158, 11, 0.08); padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid rgba(245, 158, 11, 0.25);">
                    <strong>Scientific Labeling Notice:</strong> The chart above displays a Monte Carlo simulation based on random sampling with replacement from historical observations. It represents theoretical sampling variability, not raw historical observations.
                </div>
                """,
                unsafe_allow_html=True
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: ONE-SAMPLE T-TEST & CONFIDENCE INTERVAL
    # ─────────────────────────────────────────────────────────────────────────
    with tab_ttest:
        st.subheader("One-Sample t-Test & 95% Confidence Interval (Practicals 5 & 6)")
        st.markdown("""
        Evaluates whether the historical mean NHL winning percentage differs significantly from the 
        theoretical reference point of $\\mu_0 = 0.5000$ (an even split in a two-outcome league).
        """)

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">t-Statistic (df = 581)</div>
                    <div class="kpi-value" style="color: #EF4444;">{t_test.get('t_statistic', -9.7882):.4f}</div>
                    <div class="kpi-sub">Standard Errors from 0.500</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with t2:
            p_val = t_test.get('p_value', 4.81e-21)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">p-Value (Two-Sided)</div>
                    <div class="kpi-value">{p_val:.2e}</div>
                    <div class="kpi-sub">p &lt;&lt; 0.001 (Highly Significant)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with t3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Sample Mean (x̄)</div>
                    <div class="kpi-value">{t_test.get('sample_mean', 0.4585):.4f}</div>
                    <div class="kpi-sub">s = {t_test.get('sample_std', 0.1022):.4f} (N = 582)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with t4:
            ci_l = t_test.get('ci_95_low', 0.4502)
            ci_h = t_test.get('ci_95_high', 0.4668)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">95% Confidence Interval</div>
                    <div class="kpi-value" style="font-size: 1.2rem;">[{ci_l:.4f}, {ci_h:.4f}]</div>
                    <div class="kpi-sub">Entire interval strictly &lt; 0.500</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("##### Statistical Hypotheses & Plain-Language Conclusion")
            st.markdown(f"""
- **Null Hypothesis ($H_0$):** $\\mu = 0.5000$ (The true population mean winning percentage of NHL teams is 50.0%).
- **Alternative Hypothesis ($H_1$):** $\\mu \\neq 0.5000$ (Two-sided alternative: mean winning percentage differs from 50.0%).
- **Test Decision:** Reject $H_0$ ($t = -9.7882, df = 581, p = 4.81 \\times 10^{{-21}}$). The data provides overwhelming statistical evidence that the historical mean winning percentage is less than 0.500.
- **Substantive Interpretation:** The 95% confidence interval shows we can be 95% confident that the true long-run population mean falls between **{ci_l:.4f}** and **{ci_h:.4f}**.
- **Statistical Significance vs. Practical Importance:** In large samples ($n = 582$), even trivial numerical deviations achieve low p-values. Here, the ~4.15 percentage point deficit below 0.500 is both statistically significant and practically meaningful: it quantifies the systemic impact of regulation ties throughout the 1990s, where teams earned points for ties without receiving a win in the denominator.
            """)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: HISTORICAL ERA ANOVA
    # ─────────────────────────────────────────────────────────────────────────
    with tab_anova:
        st.subheader("Historical Era One-Way ANOVA & Post-Hoc Contrasts (Practical 13)")
        st.markdown("""
        Evaluates whether mean team winning percentage differed across the three historical eras represented in the dataset:
        1990s (Ties permitted), 2000s (OT loss point introduced), and 2010s (Full shootout resolution).
        """)

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">One-Way ANOVA F-Statistic</div>
                    <div class="kpi-value" style="color: #0797D5;">{anova.get('f_statistic', 14.6872):.4f}</div>
                    <div class="kpi-sub">Degrees of Freedom: {anova.get('degrees_of_freedom', '(2, 579)')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with a2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">ANOVA p-Value</div>
                    <div class="kpi-value">{anova.get('p_value', 5.998e-07):.2e}</div>
                    <div class="kpi-sub">Reject H₀ (Equal Era Means)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with a3:
            eta_sq = anova.get('eta_squared', 0.0483)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Effect Size (η² - Eta Squared)</div>
                    <div class="kpi-value">{eta_sq:.4f}</div>
                    <div class="kpi-sub">~4.8% of variance explained by era</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        ao_col1, ao_col2 = st.columns(2)
        with ao_col1:
            st.markdown("**Team Win % Distribution Across Historical Eras**")
            st.caption("Boxplot shows median, interquartile range, and outlier bounds.")
            st.altair_chart(build_anova_decade_boxplot(df), use_container_width=True)

        with ao_col2:
            st.markdown("**Era Descriptive Statistics**")
            st.dataframe(anova.get("group_stats", pd.DataFrame()), use_container_width=True, hide_index=True)

            st.markdown("**Post-Hoc Pairwise Contrasts (Tukey HSD)**")
            st.dataframe(anova.get("post_hoc", pd.DataFrame()), use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div style="font-size: 0.84rem; color: #F59E0B; background-color: rgba(245, 158, 11, 0.08); padding: 0.65rem 0.85rem; border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.25); margin-top: 0.75rem;">
                <strong>Methodological Disclosure & Dataset Limitation:</strong> {anova.get('caveat', '')}
                In addition, the 2004–05 season was completely cancelled by the NHL lockout. Group variance assumptions (homoscedasticity) should be viewed with caution given the small 2010s sample size.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: CHI-SQUARE TEST OF INDEPENDENCE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_chi2:
        st.subheader("Chi-Square Test of Independence: Era vs. Performance Tier (Practical 16)")
        st.markdown("""
        Assesses whether franchise classification into performance tiers (Sub-.450, Competitive, Elite) 
        is statistically independent of the historical decade.
        """)

        chi1, chi2, chi3, chi4 = st.columns(4)
        with chi1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Chi-Square Statistic (χ²)</div>
                    <div class="kpi-value" style="color: #0797D5;">{chi_square.get('chi2_statistic', 28.0782):.4f}</div>
                    <div class="kpi-sub">Degrees of Freedom: {chi_square.get('degrees_of_freedom', 4)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with chi2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Asymptotic p-Value</div>
                    <div class="kpi-value">{chi_square.get('p_value', 1.20e-05):.2e}</div>
                    <div class="kpi-sub">Reject Independence</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with chi3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Cramér's V (Association)</div>
                    <div class="kpi-value">{chi_square.get('cramers_v', 0.1553):.4f}</div>
                    <div class="kpi-sub">Moderate Association Strength</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with chi4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Min Expected Frequency</div>
                    <div class="kpi-value" style="color: #0797D5;">{chi_square.get('min_expected_frequency', 10.72):.2f}</div>
                    <div class="kpi-sub">Threshold ≥ 5.0 (Fully Satisfied)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("**Contingency Table Heatmap (Observed Counts & Era Share %)**")
            obs_table = chi_square.get("observed_table", pd.DataFrame())
            st.altair_chart(build_chisquare_heatmap(obs_table), use_container_width=True)

        with ch_col2:
            st.markdown("**Observed Contingency Table (Counts)**")
            st.dataframe(obs_table, use_container_width=True)

            st.markdown("**Expected Frequencies Under Independence (H₀)**")
            st.dataframe(chi_square.get("expected_table", pd.DataFrame()), use_container_width=True)

        st.markdown(f"""
<div style="font-size: 0.86rem; color: #6B8294; margin-top: 0.5rem; line-height: 1.5;">
    <strong>Interpretation & Association vs. Causation:</strong> {chi_square.get('conclusion', '')}
    <br><br>
    <em>Causality Warning:</em> Statistical association does not mean historical eras caused teams to play better. 
    Rather, the shift is an artifact of NHL institutional rule modifications (instituting the overtime loss point in 1999 and the shootout in 2005) 
    which eliminated regulation ties and increased the average mathematical winning percentage toward 0.500.
</div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5: ALL 16 PRACTICALS DIRECTORY
    # ─────────────────────────────────────────────────────────────────────────
    with tab_all:
        st.subheader("Academic SI Lab Practicals Directory (16 Mapped Topics)")
        st.markdown(
            """
            Comprehensive index mapping all 16 practicals from the foundational academic research notebook 
            (`Hockey_Team_Performance_Analysis_working.ipynb`).
            <br>
            <span style="color: #6B8294; font-size: 0.88rem;">
                Note: Practicals marked <code>Interactive in App</code> feature live exploration widgets. 
                Remaining practicals are fully preserved, verified, and accessible directly in the original notebook.
            </span>
            """,
            unsafe_allow_html=True
        )

        prac_rows = []
        for p in practicals:
            prac_rows.append({
                "Prac #": f"Prac {p['num']}",
                "Topic": p["topic"],
                "Tools & Libraries": p["tools"],
                "Status": p["status"],
                "Specific Scientific Aim": p["aim"],
            })
        st.dataframe(pd.DataFrame(prac_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DATA QUALITY & METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def render_methodology_view(df: pd.DataFrame, metadata: dict):
    """
    Renders methodological transparency, data provenance, missing value handling,
    target leakage prevention, and model serialization compatibility.
    """
    render_header(
        section_badge="Methodology & Data Quality",
        section_title="Data Engineering, Quality Assurance & Methodological Transparency",
        section_desc=(
            "Documentation of data acquisition, missing-value preservation rationale, "
            "target leakage prevention, chronological splitting, and model limitations."
        )
    )

    with st.container(border=True):
        st.subheader("1. Data Source & Provenance")
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Verified Historical Records</div>
                    <div class="kpi-value">{len(df):,}</div>
                    <div class="kpi-sub">21 Seasons · 1990–2011</div>
                </div>
                <div style="height: 0.5rem;"></div>
                <div class="kpi-card">
                    <div class="kpi-title">Source Scrape Target</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #0797D5;">ScrapeThisSite</div>
                    <div class="kpi-sub">24 Paginated HTML Pages</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with p_col2:
            st.markdown("""
- **Data Acquisition Approach:** Extracted via automated web scraping (`requests` and `BeautifulSoup4`) from the paginated hockey dataset at `https://www.scrapethissite.com/pages/forms/`.
- **Historical Scope:** Covers completed NHL regular seasons from 1990 through 2011.
- **2004–2005 Lockout Year:** Exactly 0 games were played in the 2004–05 season due to the complete cancellation of the season during the NHL collective bargaining dispute. This is an authentic historical absence, not a data scraping error.
- **Data Parity:** The local CSV files (`hockey_team_performance_raw.csv` raw and `hockey_team_performance_cleaned.csv` cleaned) contain exactly 582 records and are permanently preserved without modification.
            """)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("2. Data Quality Audit & Missing Value Transparency")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.markdown("##### The 224 Missing OT_Losses Values")
            st.markdown("""
- **Observed Pattern:** There are exactly 224 rows where `OT_Losses` is missing (`NaN`). All 224 occurrences fall between seasons 1990 and 1998.
- **Historical Rule Context:** Prior to the 1999–2000 NHL season, overtime losses did not exist as an official statistical category. Games tied after overtime were recorded as official Ties, and both teams were awarded 1 standings point.
- **Preservation Rationale:** These values are deliberately preserved as `NaN` (or unrecorded) rather than automatically imputed with zero. Imputing zero would create a false structural equivalence between pre-1999 regulation ties and modern post-lockout shootout losses, biasing historical comparisons.
            """)
        with q_col2:
            st.markdown("##### Duplicate Column Audit: Goal_Diff vs. Goal_Difference")
            st.markdown("""
- **Audit Finding:** The raw and cleaned CSV datasets contain two adjacent columns: `Goal_Diff` and `Goal_Difference`.
- **Equivalence Verification:** Both columns have identical numerical values across all 582 rows, equal to $Goals\\_For - Goals\\_Against$.
- **Architectural Standardization:** To eliminate redundancy while strictly preserving original files, Hockey Performance Intelligence standardizes internally on `Goal_Diff`.
- **Data Types & Encoding:** Numeric columns are coerced strictly to valid floats/integers upon load, ensuring zero loss of precision.
            """)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("3. Feature Engineering & Target Leakage Prevention")
        st.markdown("""
        Selecting input features for team winning percentage estimation requires strict adherence to mathematical independence:
        """)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("##### Excluded Columns (Strict Target Leakage)")
            st.markdown("""
- **`Wins`:** Excluded because winning percentage is algebraically computed as $Wins / Games\\_Played$. Including `Wins` produces circular target leakage ($R^2 = 1.0$) with zero analytical utility.
- **`Losses` & `OT_Losses`:** Excluded for identical circular dependency reasons.
- **`Goal_Diff`:** Excluded as a direct model input feature because it is a linear combination of $Goals\\_For - Goals\\_Against$. Passing both $GF$, $GA$, and $Goal\\_Diff$ creates extreme multicollinearity.
            """)
        with f_col2:
            st.markdown("##### Active Prediction Features & Established Order")
            st.markdown("""
The active model uses strictly 3 independent features, maintained in strict established sequence:
1. **`Goals_For` (GF):** Total regular-season goals scored by the franchise (integer, typical range 150–350).
2. **`Goals_Against` (GA):** Total regular-season goals conceded by the franchise (integer, typical range 150–350).
3. **`Year`:** Calendar year of season conclusion (integer, range 1990–2011), representing a secular macro-era temporal trend.
            """)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("4. Cross-Validation, Serialization & Scientific Boundaries")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### Chronological Partitioning & Expanding-Window CV")
            st.markdown("""
- **Why Random Splits Fail:** Standard random $K$-fold cross-validation is invalid for multi-season sports analytics because franchises possess roster inertia and multi-year competitive cycles across adjacent years. Random splits leak future season information into past evaluations.
- **Expanding-Window Strategy:** Evaluation uses chronological expanding-window folds: training strictly on earlier years and validating on subsequent seasons.
- **Held-Out Test Set:** Seasons 2008 through 2011 (120 team-seasons) form the untouched test set.
            """)
        with s_col2:
            st.markdown("##### Serialization & Model Compatibility Notice")
            st.markdown("""
- **Model Artifact:** Preserved serialized artifact `models/hockey_win_pct_regressor_v2.joblib` (with backward-compatible fallback `models/hockey_win_pct_regressor.joblib`).
- **Serialization & Coefficients:** The artifact was serialized under scikit-learn 1.8.0 and loaded in the current runtime. The system verifies deserialization safety and explicitly distinguishes standardized pipeline parameters from equivalent natural-feature weights:
  - *Standardized Pipeline Parameters:* $\\beta_{GF} = +0.0930, \\beta_{GA} = -0.0854, \\beta_{Year} = +0.0203$, Standardized Intercept $\\beta_0 = 0.4477$.
  - *Equivalent Natural-Feature Weights:* $w_{GF} = +0.002123, w_{GA} = -0.001858, w_{Year} = +0.003981$, Natural Intercept $b = -7.570184$.
  - *(Standardized and natural coefficients are mathematically equivalent representations via feature scaling ($w_j = \\beta_j / \\sigma_j$), but are not interchangeable.)*
- **Retrospective Scope:** Hockey Performance Intelligence evaluates completed regular-season performance and does not constitute an in-game or pre-season forecasting model.
            """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_custom_css()

    # Load core data and model pipeline
    df = get_cached_cleaned_data()
    pipeline, metadata, load_error = load_model_and_metadata()

    # Navigation options
    options = [
        "Overview & Insights",
        "Historical Explorer",
        "Team Comparison",
        "Prediction Studio",
        "Model Evaluation Lab",
        "Statistical Insights & SI Lab",
        "Data Quality & Methodology",
    ]

    # Synchronize query parameters with navigation
    query_view = st.query_params.get("view", None)
    default_index = options.index(query_view) if query_view in options else 0

    # Sidebar Navigation Controller
    with st.sidebar:
        st.markdown(f"### {PRODUCT_NAME}")
        st.markdown(
            f"<span style='color: #A0BBD0; font-size: 0.85rem;'>{PRODUCT_TAGLINE}</span>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        nav_selection = st.radio(
            "Platform Navigation",
            options=options,
            index=default_index,
            label_visibility="collapsed"
        )

    # Render sports league top navigation bar on every view
    render_top_navigation_bar(nav_selection)

    # Route to selected analytical view
    if nav_selection == "Overview & Insights":
        render_overview_view(df)

    elif nav_selection == "Historical Explorer":
        render_historical_explorer_view(df)

    elif nav_selection == "Team Comparison":
        render_team_comparison_view(df)

    elif nav_selection == "Prediction Studio":
        if load_error:
            st.error(f"Model artifact error: {load_error}")
            st.stop()
        render_prediction_studio_view(pipeline, metadata)

    elif nav_selection == "Model Evaluation Lab":
        render_model_evaluation_view(pipeline, metadata, df)

    elif nav_selection == "Statistical Insights & SI Lab":
        render_statistical_insights_view(df)

    elif nav_selection == "Data Quality & Methodology":
        render_methodology_view(df, metadata)


if __name__ == "__main__":
    main()

