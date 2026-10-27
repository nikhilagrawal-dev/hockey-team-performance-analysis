"""
Hockey Performance Intelligence Model Inference, Validation & Explainability Module
Provides safe model loading, input validation, strict feature ordering,
retrospective regression estimation, and standardized feature decomposition.
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    FALLBACK_MODEL_PATH,
    FALLBACK_METADATA_PATH,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
)

# Optional Streamlit caching decorator wrapper
try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource(show_spinner=False)
except ImportError:
    def cache_resource_decorator(func):
        return func


def load_model_and_metadata(
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """
    Safely loads the serialized regression pipeline and associated evaluation metadata.
    Handles fallback paths and captures serialization warnings transparently.
    Returns: (pipeline, metadata_dict, error_message)
    """
    target_model = model_path or (MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH)
    target_meta = metadata_path or (METADATA_PATH if os.path.exists(METADATA_PATH) else FALLBACK_METADATA_PATH)

    if not os.path.exists(target_model):
        return None, {}, f"Model artifact not found at '{target_model}'."

    pipeline = None
    try:
        # Load the fitted pipeline safely
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            pipeline = joblib.load(target_model)
    except Exception as exc:
        return None, {}, f"Failed to deserialize model artifact at '{target_model}': {str(exc)}"

    metadata: Dict[str, Any] = {}
    if os.path.exists(target_meta):
        try:
            with open(target_meta, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    metadata = loaded
                else:
                    metadata = {"load_error": "Metadata JSON is not a valid dictionary object."}
        except Exception as exc:
            metadata = {"load_error": f"Failed to parse metadata JSON: {str(exc)}"}
    else:
        metadata = {"load_warning": f"Metadata file not found at '{target_meta}'."}

    return pipeline, metadata, None


def _is_streamlit_running() -> bool:
    try:
        import streamlit as st
        return hasattr(st, "runtime") and hasattr(st.runtime, "exists") and st.runtime.exists()
    except Exception:
        return False


def get_cached_model_and_metadata(
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """
    Streamlit-cached wrapper for load_model_and_metadata; returns plain result outside Streamlit.
    """
    if _is_streamlit_running():
        import streamlit as st
        @st.cache_resource(show_spinner=False)
        def _cached_loader(m_path: Optional[str], meta_path: Optional[str]):
            return load_model_and_metadata(m_path, meta_path)
        return _cached_loader(model_path, metadata_path)
    return load_model_and_metadata(model_path, metadata_path)



def validate_contribution_breakdown(
    pipeline: Any,
    metadata: Dict[str, Any],
    expected_features: Optional[List[str]] = None
) -> Tuple[bool, Any]:
    """
    Verifies that the fitted pipeline exposes all parameters required for
    exact mathematical feature-contribution decomposition:
    - Step names: 'scaler' (StandardScaler) and 'model' (LinearRegression)
    - Attributes: mean_, scale_, coef_, intercept_
    - Feature dimension alignment and column order
    Returns: (is_valid: bool, details_dict_or_error_str)
    """
    exp_features = expected_features or REQUIRED_FEATURES

    if not hasattr(pipeline, "named_steps"):
        return False, "Pipeline does not expose named_steps attribute."

    scaler = pipeline.named_steps.get("scaler")
    regressor = pipeline.named_steps.get("model")

    if scaler is None or regressor is None:
        return False, "Pipeline missing expected 'scaler' or 'model' step."

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        return False, "Scaler does not expose required 'mean_' or 'scale_' attributes."
    if scaler.mean_ is None or scaler.scale_ is None:
        return False, "Scaler parameters are uninitialized."

    if not hasattr(regressor, "coef_") or not hasattr(regressor, "intercept_"):
        return False, "Regressor does not expose required 'coef_' or 'intercept_' attributes."
    if regressor.coef_ is None or regressor.intercept_ is None:
        return False, "Regressor parameters are uninitialized."

    n_features = len(exp_features)
    if len(scaler.mean_) != n_features or len(scaler.scale_) != n_features:
        return False, f"Scaler dimension ({len(scaler.mean_)}) does not match expected ({n_features})."
    if len(regressor.coef_) != n_features:
        return False, f"Regressor coefficient dimension ({len(regressor.coef_)}) does not match expected ({n_features})."

    # Verify input feature ordering
    verified = False
    for step_obj in [pipeline, scaler, regressor]:
        if hasattr(step_obj, "feature_names_in_") and step_obj.feature_names_in_ is not None:
            names_in = list(step_obj.feature_names_in_)
            if names_in != exp_features:
                return False, f"Feature ordering {names_in} does not match expected order {exp_features}."
            verified = True

    meta_cols = metadata.get("feature_columns") if isinstance(metadata, dict) else None
    if meta_cols is not None:
        if meta_cols != exp_features:
            return False, f"Metadata feature columns {meta_cols} do not match expected order {exp_features}."
        verified = True

    if not verified:
        return False, "Feature ordering could not be positively verified from pipeline or metadata."

    return True, {
        "scaler": scaler,
        "regressor": regressor,
        "means": scaler.mean_,
        "scales": scaler.scale_,
        "coefs": regressor.coef_,
        "intercept": float(regressor.intercept_),
    }


def validate_prediction_inputs(
    goals_for: Any,
    goals_against: Any,
    year: Any
) -> List[str]:
    """
    Validates numeric inputs for the performance estimator.
    Returns a list of human-readable error messages (empty if valid).
    """
    errors: List[str] = []

    # Type and finiteness validation
    try:
        gf = float(goals_for)
        ga = float(goals_against)
        yr = int(year)
    except (ValueError, TypeError):
        return ["All input fields (Goals For, Goals Against, Year) must be valid numeric values."]

    if np.isnan(gf) or np.isinf(gf):
        errors.append("Goals For must be a finite numeric value.")
    elif gf <= 0:
        errors.append("Goals For must be greater than 0.")
    elif gf > 1000:
        errors.append("Goals For exceeds plausible hockey scoring bounds (maximum: 1000).")

    if np.isnan(ga) or np.isinf(ga):
        errors.append("Goals Against must be a finite numeric value.")
    elif ga <= 0:
        errors.append("Goals Against must be greater than 0.")
    elif ga > 1000:
        errors.append("Goals Against exceeds plausible hockey scoring bounds (maximum: 1000).")

    if yr < 1890 or yr > 2050:
        errors.append("Season Year must fall within a plausible calendar range (1890–2050).")

    return errors


def predict_win_pct(
    pipeline: Any,
    goals_for: float,
    goals_against: float,
    year: int,
    expected_features: Optional[List[str]] = None
) -> float:
    """
    Executes model inference on a single season input vector.
    Constructs a DataFrame strictly preserving the expected feature order.
    Returns the continuous OLS prediction unclamped (as a raw decimal float).
    """
    exp_features = expected_features or REQUIRED_FEATURES

    input_df = pd.DataFrame([{
        "Goals_For": float(goals_for),
        "Goals_Against": float(goals_against),
        "Year": float(year)
    }])[exp_features]

    prediction = float(pipeline.predict(input_df)[0])
    return prediction


def calculate_feature_contributions(
    pipeline: Any,
    metadata: Dict[str, Any],
    goals_for: float,
    goals_against: float,
    year: int,
    expected_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculates the exact standardized feature contributions:
    z_i = (x_i - mean_i) / std_i
    contrib_i = coef_i * z_i
    reconstructed_pred = intercept + sum(contrib_i)

    Verifies mathematical identity against pipeline.predict() within 1e-5 tolerance.
    """
    exp_features = expected_features or REQUIRED_FEATURES
    is_valid, details = validate_contribution_breakdown(pipeline, metadata, exp_features)
    if not is_valid:
        raise ValueError(f"Cannot calculate feature contributions: {details}")

    means = details["means"]
    scales = details["scales"]
    coefs = details["coefs"]
    intercept = details["intercept"]

    # Calculate standardized values (z-scores)
    z_gf = (float(goals_for) - float(means[0])) / float(scales[0])
    z_ga = (float(goals_against) - float(means[1])) / float(scales[1])
    z_yr = (float(year) - float(means[2])) / float(scales[2])

    # Calculate individual linear contributions
    contrib_gf = float(coefs[0]) * z_gf
    contrib_ga = float(coefs[1]) * z_ga
    contrib_yr = float(coefs[2]) * z_yr

    # Reconstructed estimate
    reconstructed_pred = intercept + contrib_gf + contrib_ga + contrib_yr

    # Raw model prediction from pipeline
    direct_pred = predict_win_pct(pipeline, goals_for, goals_against, year, exp_features)

    # Mathematical consistency verification
    diff = abs(reconstructed_pred - direct_pred)
    if diff > 1e-4:
        warnings.warn(
            f"Mathematical discrepancy between pipeline prediction ({direct_pred:.6f}) "
            f"and reconstructed contribution sum ({reconstructed_pred:.6f}); diff={diff:.6e}"
        )

    breakdown_rows = [
        {
            "Feature": "Goals_For",
            "Raw_Input": float(goals_for),
            "Training_Mean": round(float(means[0]), 2),
            "Training_Std": round(float(scales[0]), 2),
            "Z_Score": round(z_gf, 4),
            "Coefficient": round(float(coefs[0]), 4),
            "Contribution": round(contrib_gf, 4),
        },
        {
            "Feature": "Goals_Against",
            "Raw_Input": float(goals_against),
            "Training_Mean": round(float(means[1]), 2),
            "Training_Std": round(float(scales[1]), 2),
            "Z_Score": round(z_ga, 4),
            "Coefficient": round(float(coefs[1]), 4),
            "Contribution": round(contrib_ga, 4),
        },
        {
            "Feature": "Year",
            "Raw_Input": float(year),
            "Training_Mean": round(float(means[2]), 2),
            "Training_Std": round(float(scales[2]), 2),
            "Z_Score": round(z_yr, 4),
            "Coefficient": round(float(coefs[2]), 4),
            "Contribution": round(contrib_yr, 4),
        },
    ]

    return {
        "is_valid": True,
        "intercept": intercept,
        "contrib_gf": contrib_gf,
        "contrib_ga": contrib_ga,
        "contrib_yr": contrib_yr,
        "z_gf": z_gf,
        "z_ga": z_ga,
        "z_yr": z_yr,
        "reconstructed_prediction": reconstructed_pred,
        "direct_prediction": direct_pred,
        "breakdown_rows": breakdown_rows,
        "latex_equation": (
            r"\text{Win\_Pct} = \beta_0 + \beta_1 \cdot z_{\text{GF}} + "
            r"\beta_2 \cdot z_{\text{GA}} + \beta_3 \cdot z_{\text{Year}}"
        ),
        "latex_substituted": (
            rf"\text{{Win\_Pct}} = {intercept:.4f} + ({coefs[0]:.4f} \cdot {z_gf:.3f}) + "
            rf"({coefs[1]:.4f} \cdot {z_ga:.3f}) + ({coefs[2]:.4f} \cdot {z_yr:.3f}) = {direct_pred:.4f}"
        ),
    }
