"""
Hockey Team Performance Analysis — Interactive Streamlit Dashboard
Retrospective end-of-season NHL Win_Pct estimation using regular-season goal aggregates.
Fitted pipeline: 'models/hockey_win_pct_regressor_v2.joblib' (StandardScaler + LinearRegression)
Evaluation metadata: 'models/model_metadata_v2.json'
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hockey Team Performance Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hockey_win_pct_regressor_v2.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata_v2.json")

# Fallback paths
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "hockey_win_pct_regressor.joblib")
FALLBACK_METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

REQUIRED_FEATURES = ["Goals_For", "Goals_Against", "Year"]

# Historical Dataset Reference Statistics (582 NHL team-seasons, 1990–2011)
HISTORICAL_MIN_WIN_PCT = 11.90   # 1992 San Jose Sharks (11-71-2)
HISTORICAL_MEAN_WIN_PCT = 45.85  # League-wide mean
HISTORICAL_MAX_WIN_PCT = 75.60   # 1995 Detroit Red Wings (62-13-7)
HISTORICAL_AVG_GF = 234.06
HISTORICAL_AVG_GA = 234.06


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL SCOPED CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_custom_css():
    """Injects minimal styling for dark theme consistency and tier card layout."""
    st.markdown("""
    <style>
    .stApp {
        background-color: #0B1220;
        color: #F1F5F9;
    }
    .tier-card {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        border: 1px solid #1E293B;
        background-color: #131C2E;
    }
    .tier-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .tier-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        margin: 0;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL & METADATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Loads the fitted model pipeline and evaluation metadata safely."""
    target_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    target_meta_path = METADATA_PATH if os.path.exists(METADATA_PATH) else FALLBACK_METADATA_PATH

    if not os.path.exists(target_model_path):
        return None, None, f"Model artifact not found at '{target_model_path}'."

    try:
        pipeline = joblib.load(target_model_path)
    except Exception as e:
        return None, None, f"Failed to load model artifact: {str(e)}"

    metadata = {}
    if os.path.exists(target_meta_path):
        try:
            with open(target_meta_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    metadata = loaded
                else:
                    metadata = {"load_error": "Metadata file does not contain a valid JSON object."}
        except Exception as e:
            metadata = {"load_error": str(e)}

    return pipeline, metadata, None


# ─────────────────────────────────────────────────────────────────────────────
# SAFE VALIDATION OF FEATURE-CONTRIBUTION BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def validate_contribution_breakdown(pipeline, metadata, expected_features):
    """
    Validates whether the technical feature-contribution breakdown can be safely computed.
    Verifies steps, attributes, feature dimensions, and feature ordering.
    Returns (is_valid: bool, detail: dict or str).
    """
    if not hasattr(pipeline, "named_steps"):
        return False, "Pipeline does not expose named steps."

    scaler = pipeline.named_steps.get("scaler")
    regressor = pipeline.named_steps.get("model")

    if scaler is None or regressor is None:
        return False, "Pipeline does not contain expected 'scaler' and 'model' steps."

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        return False, "Scaler does not expose required 'mean_' and 'scale_' attributes."
    if scaler.mean_ is None or scaler.scale_ is None:
        return False, "Scaler parameters are uninitialized."

    if not hasattr(regressor, "coef_") or not hasattr(regressor, "intercept_"):
        return False, "Regressor does not expose required 'coef_' and 'intercept_' attributes."
    if regressor.coef_ is None or regressor.intercept_ is None:
        return False, "Regressor parameters are uninitialized."

    n_features = len(expected_features)
    if len(scaler.mean_) != n_features or len(scaler.scale_) != n_features:
        return False, f"Scaler parameter dimension ({len(scaler.mean_)}) does not match expected features ({n_features})."
    if len(regressor.coef_) != n_features:
        return False, f"Regressor coefficient dimension ({len(regressor.coef_)}) does not match expected features ({n_features})."

    verified_by_model = False
    for step_name, step_obj in [("pipeline", pipeline), ("scaler", scaler), ("regressor", regressor)]:
        if hasattr(step_obj, "feature_names_in_") and step_obj.feature_names_in_ is not None:
            names_in = list(step_obj.feature_names_in_)
            if names_in != expected_features:
                return False, f"Step '{step_name}' feature names {names_in} do not match expected input order {expected_features}."
            verified_by_model = True

    meta_cols = metadata.get("feature_columns") if isinstance(metadata, dict) else None
    if meta_cols is not None:
        if not isinstance(meta_cols, list) or meta_cols != expected_features:
            return False, f"Metadata feature columns {meta_cols} do not match expected input order {expected_features}."
        verified_by_metadata = True
    else:
        verified_by_metadata = False

    if not verified_by_model and not verified_by_metadata:
        return False, "Feature ordering cannot be verified from pipeline 'feature_names_in_' attributes or metadata 'feature_columns'."

    return True, {
        "scaler": scaler,
        "regressor": regressor,
        "means": scaler.mean_,
        "scales": scaler.scale_,
        "coefs": regressor.coef_,
        "intercept": regressor.intercept_
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    """Renders the title and single plain retrospective estimate caption."""
    st.title("Hockey Team Performance Analysis")
    st.caption("This model estimates a team's final Win % from season-total Goals For, Goals Against and season year; totals are only available after the season concludes, so it is not a pre-season forecasting tool.")


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: PREDICTION WORKSPACE
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
        st.subheader("Prediction")

        st.markdown("**Preset Scenarios**")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.button(
                "Test Sample (2009)",
                on_click=apply_preset,
                args=(250, 230, 2009),
                use_container_width=True
            )
        with p_col2:
            st.button(
                "Top Contender",
                on_click=apply_preset,
                args=(300, 210, 2010),
                use_container_width=True
            )
        with p_col3:
            st.button(
                "League Average",
                on_click=apply_preset,
                args=(234, 234, 2006),
                use_container_width=True
            )
        with p_col4:
            st.button(
                "Rebuilding Team",
                on_click=apply_preset,
                args=(185, 295, 2003),
                use_container_width=True
            )

        st.markdown("**Team Season Inputs**")
        in_col1, in_col2, in_col3 = st.columns(3)
        with in_col1:
            goals_for = st.number_input(
                "Goals For (GF)",
                min_value=50,
                max_value=500,
                step=1,
                key="input_gf"
            )
        with in_col2:
            goals_against = st.number_input(
                "Goals Against (GA)",
                min_value=50,
                max_value=500,
                step=1,
                key="input_ga"
            )
        with in_col3:
            year = st.number_input(
                "Season Starting Year",
                min_value=1990,
                max_value=2025,
                step=1,
                key="input_yr"
            )

        if year == 2004:
            st.caption(
                "Note: The 2004–05 NHL season was cancelled due to the league lockout. "
                "The estimate is based on structural era interpolation."
            )

    return goals_for, goals_against, year


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: WIN % COMPARISON SCALE (REBUILT)
# ─────────────────────────────────────────────────────────────────────────────
def render_win_pct_comparison_scale(pred_pct):
    """
    Renders a responsive horizontal 0-100% comparison scale bar.
    - Filled progress bar (accent color, low opacity) from 0% to prediction.
    - Prediction: thick icy-blue vertical line + downward triangle + small circle
      on the track — clearly distinct from the thin grey historical ticks.
    - Historical ticks (Min/Mean/Max): thin, short, muted grey lines with small labels below.
    - Prediction label is bold, larger, accent-colored, above the track.
    - Collision handling: when prediction is within ~3pp of a tick, both a vertical
      raise and a horizontal nudge (via text-anchor flip) are applied so text never overlaps.
    - Label x is clamped to SVG bounds so text is never clipped at container edges.
    - All HTML built as a compact joined list — no blank lines, no indented lines —
      so Markdown never misinterprets it as a fenced code block.
    """
    is_out_of_range = pred_pct < 0.0 or pred_pct > 100.0
    scale_min = min(0.0, float(pred_pct) - 10.0) if is_out_of_range else 0.0
    scale_max = max(100.0, float(pred_pct) + 10.0) if is_out_of_range else 100.0
    total_range = scale_max - scale_min

    def to_x(val):
        # 800px coordinate space: 40px left margin, 40px right margin = 720px usable
        return 40.0 + ((float(val) - scale_min) / total_range) * 720.0

    pred_x = to_x(pred_pct)
    min_x = to_x(HISTORICAL_MIN_WIN_PCT)
    mean_x = to_x(HISTORICAL_MEAN_WIN_PCT)
    max_x = to_x(HISTORICAL_MAX_WIN_PCT)
    track_start = to_x(0.0)
    track_end = to_x(100.0)
    track_width = track_end - track_start

    # Filled progress bar: from track_start to pred_x (clamped to track bounds)
    fill_end_x = max(track_start, min(track_end, pred_x))
    fill_width = fill_end_x - track_start

    # Threshold for collision detection: 3 percentage points in SVG coordinate space
    px_per_pct = 720.0 / total_range
    collision_threshold = px_per_pct * 3.0
    tick_xs = [min_x, mean_x, max_x]
    near_tick = any(abs(pred_x - tx) < collision_threshold for tx in tick_xs)

    # Prediction label vertical position: raise higher when close to a tick label
    pred_label_y = 18 if not near_tick else 9

    # Prediction label anchor: flip from middle to start/end to avoid horizontal overlap,
    # but also clamp x so text never clips at the SVG left/right edges
    label_x = max(55.0, min(745.0, pred_x))
    if pred_x < 110:
        pred_anchor = "start"
        label_x = max(40.0, pred_x)
    elif pred_x > 690:
        pred_anchor = "end"
        label_x = min(760.0, pred_x)
    elif near_tick:
        # Horizontally nudge label to whichever side has more room from the nearest tick
        nearest_tx = min(tick_xs, key=lambda tx: abs(pred_x - tx))
        pred_anchor = "start" if pred_x >= nearest_tx else "end"
        label_x = pred_x + (8 if pred_anchor == "start" else -8)
    else:
        pred_anchor = "middle"

    # Build compact SVG as a joined list — no blank lines, no leading indentation
    parts = [
        '<div style="width:100%;overflow-x:auto;margin-top:0.5rem;margin-bottom:0.25rem;">',
        '<svg width="100%" height="96" viewBox="0 0 800 96" xmlns="http://www.w3.org/2000/svg" style="display:block;">',
        # Grey base track (full 0-100% extent)
        f'<rect x="{track_start:.1f}" y="42" width="{track_width:.1f}" height="8" rx="4" fill="#1E293B"/>',
        # Accent-colored filled bar from 0% up to the prediction (30% opacity)
        f'<rect x="{track_start:.1f}" y="42" width="{fill_width:.1f}" height="8" rx="4" fill="#38BDF8" fill-opacity="0.30"/>',
        # Historical min tick (thin, short, muted grey) and label below track
        f'<line x1="{min_x:.1f}" y1="40" x2="{min_x:.1f}" y2="52" stroke="#94A3B8" stroke-width="1.5"/>',
        f'<text x="{min_x:.1f}" y="74" fill="#64748B" font-size="11" font-family="sans-serif" text-anchor="middle">Min {HISTORICAL_MIN_WIN_PCT:.1f}%</text>',
        # Historical mean tick and label below track
        f'<line x1="{mean_x:.1f}" y1="40" x2="{mean_x:.1f}" y2="52" stroke="#94A3B8" stroke-width="1.5"/>',
        f'<text x="{mean_x:.1f}" y="74" fill="#64748B" font-size="11" font-family="sans-serif" text-anchor="middle">Mean {HISTORICAL_MEAN_WIN_PCT:.1f}%</text>',
        # Historical max tick and label below track
        f'<line x1="{max_x:.1f}" y1="40" x2="{max_x:.1f}" y2="52" stroke="#94A3B8" stroke-width="1.5"/>',
        f'<text x="{max_x:.1f}" y="74" fill="#64748B" font-size="11" font-family="sans-serif" text-anchor="middle">Max {HISTORICAL_MAX_WIN_PCT:.1f}%</text>',
        # Prediction: thick icy-blue vertical line spanning from above to below the track
        f'<line x1="{pred_x:.1f}" y1="26" x2="{pred_x:.1f}" y2="58" stroke="#38BDF8" stroke-width="3"/>',
        # Downward-pointing triangle pointer (above track)
        f'<polygon points="{pred_x-6:.1f},28 {pred_x+6:.1f},28 {pred_x:.1f},40" fill="#38BDF8"/>',
        # Small circle on the track at the prediction position
        f'<circle cx="{pred_x:.1f}" cy="46" r="5" fill="#38BDF8"/>',
        # Prediction label — bold, larger, accent-colored, above the track
        f'<text x="{label_x:.1f}" y="{pred_label_y}" fill="#38BDF8" font-size="13" font-weight="700" font-family="sans-serif" text-anchor="{pred_anchor}">Prediction {pred_pct:.1f}%</text>',
        '</svg>',
        '</div>',
    ]
    svg_html = "".join(parts)

    # st.markdown with unsafe_allow_html=True renders the SVG correctly because the
    # string is compact (no blank lines, no leading indentation) — Markdown will not
    # treat it as a fenced code block or stop the HTML block at a blank line.
    st.markdown(svg_html, unsafe_allow_html=True)
    st.caption("Blue fill and marker = current prediction. Thin grey ticks = historical dataset min, mean and max (descriptive statistics, not model predictions).")


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: RESULT
# ─────────────────────────────────────────────────────────────────────────────
def render_result(pred_raw, goals_for, goals_against):
    """
    Renders the Result section containing:
    1. Estimated Win % (large) with the raw decimal caption beneath it.
    2. Goal Differential (GF minus GA, with sign).
    3. Performance tier (plain text name, range, and one-line explanation).
    4. Win % comparison scale.
    """
    pred_pct = pred_raw * 100.0
    goal_diff = goals_for - goals_against

    with st.container(border=True):
        st.subheader("Result")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Estimated Win %",
                value=f"{pred_pct:.2f}%"
            )
            st.caption(f"Model output (Win_Pct): {pred_raw:.4f}")
        with res_col2:
            st.metric(
                label="Goal Differential",
                value=f"{goal_diff:+d}"
            )

        # Performance Tier Interpretation (Plain text without emojis)
        if pred_raw >= 0.600:
            tier_title = "Elite / Top Contender (Win % >= 60%)"
            tier_desc = "Teams in this tier consistently secure high playoff seeding and contend for division titles."
            tier_color = "#10B981"
        elif pred_raw >= 0.500:
            tier_title = "Playoff Contender (50% <= Win % < 60%)"
            tier_desc = "Above-average performance typically qualifying for post-season contention."
            tier_color = "#38BDF8"
        elif pred_raw >= 0.400:
            tier_title = "Mid-to-Lower Tier (40% <= Win % < 50%)"
            tier_desc = "Below-average record, usually missing the playoffs or competing for wild-card spots."
            tier_color = "#F59E0B"
        else:
            tier_title = "Rebuilding Tier (Win % < 40%)"
            tier_desc = "Significant performance deficit; typically in draft-lottery contention."
            tier_color = "#E11D48"

        # Build tier card as compact HTML with no blank lines or leading indentation
        # to avoid Markdown treating indented lines as a code block
        tier_html = (
            '<div class="tier-card">'
            f'<div class="tier-title" style="color:{tier_color};">{tier_title}</div>'
            f'<p class="tier-desc">{tier_desc}</p>'
            '</div>'
        )
        st.markdown(tier_html, unsafe_allow_html=True)

        # 4. Rebuilt Win % comparison scale
        render_win_pct_comparison_scale(pred_pct)

        # Plain text note for out-of-range predictions (never clamped)
        if pred_raw < 0.0 or pred_raw > 1.0:
            st.caption(
                f"Note: The linear regression output ({pred_pct:.2f}%) falls outside the 0%–100% range. "
                "The model uses continuous ordinary least squares without artificial boundary clipping."
            )


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def render_model_performance(metadata):
    """Renders evaluation metrics from metadata safely without hardcoding."""
    with st.container(border=True):
        st.subheader("Model Performance")

        eval_strat = metadata.get("evaluation_strategy") if isinstance(metadata, dict) else None
        cv_metrics = metadata.get("cv_metrics") if isinstance(metadata, dict) else None
        test_metrics = metadata.get("test_metrics") if isinstance(metadata, dict) else None

        train_period = "Not available"
        test_period = "Not available"
        train_rows_str = ""
        test_rows_str = ""
        cv_method = "5-Fold Season-Aware Expanding-Window CV"

        if isinstance(eval_strat, dict):
            train_period = str(eval_strat.get("train_period", eval_strat.get("train_years", "Not available")))
            test_period = str(eval_strat.get("test_period", eval_strat.get("test_years", "Not available")))
            if "train_rows" in eval_strat:
                train_rows_str = f" ({eval_strat['train_rows']} teams)"
            if "test_rows" in eval_strat:
                test_rows_str = f" ({eval_strat['test_rows']} teams)"
            if eval_strat.get("cv_method"):
                cv_method = str(eval_strat["cv_method"])

        # Extract CV RMSE and optional SD from metadata
        cv_rmse_val = None
        cv_rmse_sd = None
        cv_r2_val = None
        cv_r2_sd = None

        if isinstance(cv_metrics, dict):
            cv_rmse_val = cv_metrics.get("cv_rmse_mean", cv_metrics.get("cv_rmse"))
            cv_rmse_sd = cv_metrics.get("cv_rmse_sd")
            cv_r2_val = cv_metrics.get("cv_r2_mean", cv_metrics.get("cv_r2"))
            cv_r2_sd = cv_metrics.get("cv_r2_sd")

        # Format CV RMSE with SD if present in metadata
        if isinstance(cv_rmse_val, (int, float)):
            if isinstance(cv_rmse_sd, (int, float)):
                cv_rmse_display = f"{cv_rmse_val:.4f} +/- {cv_rmse_sd:.4f}"
            else:
                cv_rmse_display = f"{cv_rmse_val:.4f}"
        else:
            cv_rmse_display = "Not available"

        # Format CV R2 with SD if present in metadata
        if isinstance(cv_r2_val, (int, float)):
            if isinstance(cv_r2_sd, (int, float)):
                cv_r2_display = f"{cv_r2_val:.4f} +/- {cv_r2_sd:.4f}"
            else:
                cv_r2_display = f"{cv_r2_val:.4f}"
        else:
            cv_r2_display = "Not available"

        test_rmse_val = None
        test_r2_val = None
        if isinstance(test_metrics, dict):
            test_rmse_val = test_metrics.get("test_rmse")
            test_r2_val = test_metrics.get("test_r2")

        test_rmse_display = f"{test_rmse_val:.4f}" if isinstance(test_rmse_val, (int, float)) else "Not available"
        test_r2_display = f"{test_r2_val:.4f}" if isinstance(test_r2_val, (int, float)) else "Not available"

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="CV RMSE", value=cv_rmse_display)
        with m_col2:
            st.metric(label="Test RMSE", value=test_rmse_display)
        with m_col3:
            st.metric(label="Test R²", value=test_r2_display)
        with m_col4:
            st.metric(label="CV R²", value=cv_r2_display)

        st.caption(
            f"Validation Framework: {cv_method}. "
            f"Training Window: {train_period}{train_rows_str}. "
            f"Holdout Test Window: {test_period}{test_rows_str}. "
            "CV metrics show fold mean +/- standard deviation across folds where available; "
            "Test metrics are evaluated on the 2008–2011 holdout seasons."
        )


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: HISTORICAL DATASET REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def render_historical_benchmarks():
    """Renders empirical dataset baseline statistics clearly separated from model predictions."""
    with st.container(border=True):
        st.subheader("Historical Dataset Reference")
        st.markdown("**Descriptive statistics of the historical dataset (not model predictions)**")

        b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
        with b_col1:
            st.metric("Min Win %", f"{HISTORICAL_MIN_WIN_PCT:.1f}%")
        with b_col2:
            st.metric("Mean Win %", f"{HISTORICAL_MEAN_WIN_PCT:.2f}%")
        with b_col3:
            st.metric("Max Win %", f"{HISTORICAL_MAX_WIN_PCT:.1f}%")
        with b_col4:
            st.metric("Avg Goals For", f"~{int(round(HISTORICAL_AVG_GF))}")
        with b_col5:
            st.metric("Avg Goals Against", f"~{int(round(HISTORICAL_AVG_GA))}")

        st.caption("Source: Historical dataset (582 NHL team-seasons, 1990–2011). Min: 1992 San Jose Sharks; Max: 1995 Detroit Red Wings.")


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: HOW THE PREDICTION IS CALCULATED & REBUILT FEATURE CHART
# ─────────────────────────────────────────────────────────────────────────────
def render_feature_contribution_chart(intercept, contrib_gf, contrib_ga, contrib_yr):
    """
    Renders a simple, clean horizontal bar chart with exactly four bars:
    Baseline (intercept), Goals For, Goals Against, Year.
    Labels do not truncate or overlap; values are displayed directly beside each bar.
    """
    bar_order = ["Baseline (intercept)", "Goals For", "Goals Against", "Year"]

    data = pd.DataFrame([
        {
            "Feature": "Baseline (intercept)",
            "Contribution": float(intercept),
            "Value_Label": f"{intercept:+.4f}",
            "Color": "#64748B"
        },
        {
            "Feature": "Goals For",
            "Contribution": float(contrib_gf),
            "Value_Label": f"{contrib_gf:+.4f}",
            "Color": "#38BDF8" if contrib_gf >= 0 else "#E11D48"
        },
        {
            "Feature": "Goals Against",
            "Contribution": float(contrib_ga),
            "Value_Label": f"{contrib_ga:+.4f}",
            "Color": "#38BDF8" if contrib_ga >= 0 else "#E11D48"
        },
        {
            "Feature": "Year",
            "Contribution": float(contrib_yr),
            "Value_Label": f"{contrib_yr:+.4f}",
            "Color": "#38BDF8" if contrib_yr >= 0 else "#E11D48"
        },
    ])

    min_c = min(0.0, float(data["Contribution"].min()))
    max_c = max(0.0, float(data["Contribution"].max()))
    padding = max(0.08, (max_c - min_c) * 0.28)
    domain = [min_c - padding, max_c + padding]

    base = alt.Chart(data).properties(
        height=210,
        title=alt.TitleParams(
            text="How each input moves the estimate from the baseline",
            color="#F1F5F9",
            fontSize=13,
            fontWeight=600,
            anchor="start",
            offset=10
        )
    )

    bars = base.mark_bar(height=26, cornerRadius=3).encode(
        y=alt.Y(
            "Feature:N",
            sort=bar_order,
            title=None,
            axis=alt.Axis(
                labelColor="#F1F5F9",
                labelFontSize=12,
                labelPadding=12,
                ticks=False,
                domainColor="#334155"
            )
        ),
        x=alt.X(
            "Contribution:Q",
            title="Contribution to Win_Pct",
            scale=alt.Scale(domain=domain, zero=True),
            axis=alt.Axis(
                labelColor="#94A3B8",
                titleColor="#94A3B8",
                gridColor="#1E293B",
                domainColor="#334155"
            )
        ),
        color=alt.Color("Color:N", scale=None)
    )

    labels = base.mark_text(
        align=alt.expr("datum.Contribution >= 0 ? 'left' : 'right'"),
        baseline="middle",
        dx=alt.expr("datum.Contribution >= 0 ? 8 : -8"),
        color="#F1F5F9",
        fontSize=11,
        fontWeight=600
    ).encode(
        y=alt.Y("Feature:N", sort=bar_order),
        x=alt.X("Contribution:Q"),
        text=alt.Text("Value_Label:N")
    )

    chart = (bars + labels).configure_view(strokeOpacity=0)
    st.altair_chart(chart, use_container_width=True)


def render_explainability(pipeline, metadata, goals_for, goals_against, year, pred_raw):
    """Renders calculations, equations, and 4-bar horizontal chart inside expander."""
    with st.expander("How the Prediction Is Calculated", expanded=False):
        is_valid, check_result = validate_contribution_breakdown(pipeline, metadata, REQUIRED_FEATURES)

        if is_valid:
            means = check_result["means"]
            scales = check_result["scales"]
            coefs = check_result["coefs"]
            intercept = check_result["intercept"]

            z_gf = (goals_for - means[0]) / scales[0]
            z_ga = (goals_against - means[1]) / scales[1]
            z_yr = (year - means[2]) / scales[2]

            contrib_gf = coefs[0] * z_gf
            contrib_ga = coefs[1] * z_ga
            contrib_yr = coefs[2] * z_yr

            st.markdown("**Standardized Feature Calculations**")
            breakdown_df = pd.DataFrame([
                {
                    "Feature": "Goals_For",
                    "Raw Input": goals_for,
                    "Training Mean": round(means[0], 2),
                    "Training Std": round(scales[0], 2),
                    "Z-Score": round(z_gf, 3),
                    "Coefficient": round(coefs[0], 4),
                    "Contribution to Win_Pct": round(contrib_gf, 4),
                },
                {
                    "Feature": "Goals_Against",
                    "Raw Input": goals_against,
                    "Training Mean": round(means[1], 2),
                    "Training Std": round(scales[1], 2),
                    "Z-Score": round(z_ga, 3),
                    "Coefficient": round(coefs[1], 4),
                    "Contribution to Win_Pct": round(contrib_ga, 4),
                },
                {
                    "Feature": "Year",
                    "Raw Input": year,
                    "Training Mean": round(means[2], 2),
                    "Training Std": round(scales[2], 2),
                    "Z-Score": round(z_yr, 3),
                    "Coefficient": round(coefs[2], 4),
                    "Contribution to Win_Pct": round(contrib_yr, 4),
                },
            ])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            st.markdown("**Linear Regression Model Equation**")
            st.latex(
                r"\text{Win\_Pct} = \beta_0 + \beta_1 \cdot z_{\text{GF}} + \beta_2 \cdot z_{\text{GA}} + \beta_3 \cdot z_{\text{Year}}"
            )
            st.latex(
                rf"\text{{Win\_Pct}} = {intercept:.4f} + ({coefs[0]:.4f} \cdot {z_gf:.3f}) + ({coefs[1]:.4f} \cdot {z_ga:.3f}) + ({coefs[2]:.4f} \cdot {z_yr:.3f}) = {pred_raw:.4f}"
            )

            st.markdown("**Feature Contribution Breakdown**")
            render_feature_contribution_chart(intercept, contrib_gf, contrib_ga, contrib_yr)

            st.caption(
                f"Baseline + Goals For + Goals Against + Year = predicted Win_Pct: "
                f"{intercept:.4f} + ({contrib_gf:+.4f}) + ({contrib_ga:+.4f}) + ({contrib_yr:+.4f}) = {pred_raw:.4f}"
            )
            st.caption(
                "Each bar shows how that standardized input shifts the estimate away from the regression baseline "
                "for this prediction. These are model contributions, not causal effects."
            )

        else:
            st.markdown(
                f"Technical feature-contribution breakdown is omitted: {check_result} "
                "The primary regression prediction above remains fully operational."
            )


# ─────────────────────────────────────────────────────────────────────────────
# UI SECTION: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def render_methodology():
    """Renders methodology, leakage exclusions, and limitations inside a collapsed expander."""
    with st.expander("Methodology & Model Limitations", expanded=False):
        st.markdown("##### Approach")
        st.markdown("""
- Retrospective Estimation: Estimates final season Win_Pct using regular-season goal aggregates.
- Non-Forecasting: Goal totals are only available after the season concludes, so this cannot be used for pre-game or pre-season forecasting.
- Temporal Holdout: Generalization evaluated on unseen 2008–2011 seasons.
        """)

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("##### Excluded Features")
            st.markdown("""
- Wins and Losses: Excluded because they directly reconstruct the target.
- OT_Losses: Excluded as part of the Win_Pct denominator; 224 of 582 values are missing (pre-shootout era).
- Goal_Diff: Excluded as a direct mathematical composite of Goals_For minus Goals_Against (near-perfect predictor, r = 0.88).
            """)

        with c_m2:
            st.markdown("##### Limitations")
            st.markdown("""
- Linearity: Uses OLS with constant coefficients; does not enforce hard 0 to 1 bounds.
- Historical Era: Trained on 1990–2011; modern NHL rule changes may shift baseline goals.
- Autocorrelation: Residuals exhibit mild positive serial correlation (Durbin-Watson ≈ 1.08).
            """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION FLOW
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_custom_css()
    pipeline, metadata, load_error = load_model_and_metadata()

    # Header
    render_header()

    if load_error:
        st.markdown(f"Model artifact error: {load_error}")
        st.stop()

    # Prediction Workspace
    goals_for, goals_against, year = render_prediction_workspace()

    # Determine feature schema safely
    expected_features = REQUIRED_FEATURES
    if isinstance(metadata, dict) and "feature_columns" in metadata:
        meta_cols = metadata.get("feature_columns")
        if isinstance(meta_cols, list) and meta_cols == REQUIRED_FEATURES:
            expected_features = meta_cols

    # Input validation checks
    validation_errors = []
    if goals_for <= 0:
        validation_errors.append("Goals For must be greater than 0.")
    if goals_against <= 0:
        validation_errors.append("Goals Against must be greater than 0.")

    if validation_errors:
        for err in validation_errors:
            st.markdown(f"Input error: {err}")
        return

    # Generate prediction from saved pipeline
    try:
        input_df = pd.DataFrame([{
            "Goals_For": float(goals_for),
            "Goals_Against": float(goals_against),
            "Year": float(year)
        }])[expected_features]

        # Raw continuous model prediction (unclamped)
        pred_raw = float(pipeline.predict(input_df)[0])

    except Exception as exc:
        st.markdown(f"Prediction calculation error: {str(exc)}")
        return

    # Result Section
    render_result(pred_raw, goals_for, goals_against)

    # Model Performance
    render_model_performance(metadata)

    # Historical Dataset Reference
    render_historical_benchmarks()

    # How the Prediction Is Calculated
    render_explainability(pipeline, metadata, goals_for, goals_against, year, pred_raw)

    # Methodology
    render_methodology()


if __name__ == "__main__":
    main()
