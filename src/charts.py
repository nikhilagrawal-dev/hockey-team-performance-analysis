"""
Hockey Performance Intelligence Chart Factory Module
Constructs accessible, responsive, dark-theme Altair visualisations
for hockey performance distributions, trends, scatter plots, and team comparisons.
"""

from typing import List, Optional
import pandas as pd
import altair as alt

from src.config import (
    THEME_BG,
    THEME_CARD_BG,
    THEME_NAVY,
    THEME_BORDER,
    THEME_PRIMARY,
    THEME_CYAN,
    THEME_TEXT_PRIMARY,
    THEME_TEXT_MUTED,
    THEME_TEXT_DIM,
    THEME_POSITIVE,
    THEME_NEGATIVE,
    THEME_WARNING,
    HISTORICAL_MEAN_WIN_PCT,
)


def apply_theme_config(chart: alt.Chart) -> alt.Chart:
    """Applies consistent ice-hockey sports analytics styling across Altair charts."""
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=THEME_TEXT_MUTED,
            titleColor=THEME_TEXT_PRIMARY,
            gridColor="#E2EEF5",
            domainColor="#D8E7EF",
            tickColor="#D8E7EF",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=700,
        )
        .configure_legend(
            labelColor=THEME_TEXT_PRIMARY,
            titleColor=THEME_TEXT_MUTED,
            fillColor="transparent",
            strokeColor="transparent",
            titleFontSize=11,
            labelFontSize=11,
        )
    )


def build_win_pct_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Constructs a distribution histogram of team winning percentages with mean
    and .500 reference markers.
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    chart_df = df[["Team_Name", "Year", "Win_Pct"]].copy()
    chart_df["Win_Pct_Display"] = (chart_df["Win_Pct"] * 100.0).round(1)

    mean_val = float(df["Win_Pct"].mean()) * 100.0

    base = alt.Chart(chart_df).properties(
        height=260,
        title=alt.TitleParams(
            text="Historical Distribution of Team Winning Percentages (1990–2011)",
            color=THEME_TEXT_PRIMARY,
            fontSize=13,
            fontWeight=600,
            anchor="start",
            offset=10,
        ),
    )

    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=THEME_PRIMARY, opacity=0.85).encode(
        x=alt.X(
            "Win_Pct_Display:Q",
            bin=alt.Bin(step=5.0),
            title="Season Winning Percentage (Win %)",
            axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'"),
        ),
        y=alt.Y("count():Q", title="Number of Team-Seasons"),
        tooltip=[
            alt.Tooltip("count():Q", title="Team-Seasons"),
            alt.Tooltip("min(Win_Pct_Display):Q", title="Bin Min (%)", format=".1f"),
            alt.Tooltip("max(Win_Pct_Display):Q", title="Bin Max (%)", format=".1f"),
        ],
    )

    # Reference rule for historical mean (45.85%)
    mean_rule = (
        alt.Chart(pd.DataFrame({"Mean": [mean_val]}))
        .mark_rule(color=THEME_NAVY, strokeWidth=2, strokeDash=[4, 4])
        .encode(x="Mean:Q")
    )

    # Reference rule for .500 benchmark (50.00%)
    bench_rule = (
        alt.Chart(pd.DataFrame({"Benchmark": [50.0]}))
        .mark_rule(color=THEME_WARNING, strokeWidth=1.5, strokeDash=[2, 2])
        .encode(x="Benchmark:Q")
    )

    chart = (bars + mean_rule + bench_rule)
    return apply_theme_config(chart)


def build_season_trends_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Constructs the season-by-season average winning percentage trend chart (1990-2011)
    with points, line, and translucent area fill, matching the reference sports dashboard.
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"Season": [], "Winning Percentage": []})).mark_point()

    season_df = df.groupby("Year", as_index=False)["Win_Pct"].mean()
    season_df.rename(columns={"Year": "Season", "Win_Pct": "Winning Percentage"}, inplace=True)
    season_df["Winning Percentage"] = season_df["Winning Percentage"].round(4)

    base = alt.Chart(season_df).properties(
        height=260,
    )

    area = base.mark_area(
        color=THEME_PRIMARY,
        opacity=0.15,
        line=False
    ).encode(
        x=alt.X("Season:Q", axis=alt.Axis(format="d", values=[1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010], title="Season (Year)", labelFontSize=11, labelAngle=0)),
        y=alt.Y("Winning Percentage:Q", scale=alt.Scale(domain=[0.30, 0.60]), axis=alt.Axis(format=".2f", values=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60], title="Winning Percentage", labelFontSize=11))
    )

    line = base.mark_line(
        color=THEME_PRIMARY,
        strokeWidth=2.2
    ).encode(
        x=alt.X("Season:Q"),
        y=alt.Y("Winning Percentage:Q")
    )

    points = base.mark_point(
        color=THEME_PRIMARY,
        fill=THEME_PRIMARY,
        size=50
    ).encode(
        x=alt.X("Season:Q"),
        y=alt.Y("Winning Percentage:Q"),
        tooltip=[
            alt.Tooltip("Season:Q", format="d", title="Season (Year)"),
            alt.Tooltip("Winning Percentage:Q", format=".3f", title="Avg Win %")
        ]
    )

    return apply_theme_config(area + line + points)


def build_goal_diff_scatter_chart(
    df: pd.DataFrame,
    selected_teams: Optional[List[str]] = None
) -> alt.Chart:
    """
    Constructs an interactive scatter plot comparing Goal Differential with Win %,
    featuring trendline, contextual tooltips, and optional team highlights.
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    chart_df = df.copy()
    chart_df["Win_Pct_Pct"] = (chart_df["Win_Pct"] * 100.0).round(2)
    if selected_teams and len(selected_teams) > 0:

        chart_df["Status"] = "Other Teams"
        is_sel = chart_df["Team_Name"].isin(selected_teams)
        chart_df.loc[is_sel & (chart_df["Goal_Diff"] >= 0), "Status"] = "Positive Diff (Selected)"
        chart_df.loc[is_sel & (chart_df["Goal_Diff"] < 0), "Status"] = "Negative Diff (Selected)"
        color_scale = alt.Scale(
            domain=["Positive Diff (Selected)", "Negative Diff (Selected)", "Other Teams"],
            range=[THEME_PRIMARY, THEME_NEGATIVE, "#94A3B8"]
        )
    else:
        chart_df["Status"] = "Positive Differential"
        chart_df.loc[chart_df["Goal_Diff"] < 0, "Status"] = "Negative Differential"
        color_scale = alt.Scale(
            domain=["Positive Differential", "Negative Differential"],
            range=[THEME_PRIMARY, THEME_NEGATIVE]
        )

    base = alt.Chart(chart_df).properties(
        height=340,
        title=alt.TitleParams(
            text="Goal Differential vs. Regular-Season Winning Percentage",
            subtitle="Strong linear association (r = 0.8872) illustrates regular-season determinism",
            color=THEME_TEXT_PRIMARY,
            subtitleColor=THEME_TEXT_MUTED,
            fontSize=13,
            fontWeight=600,
            anchor="start",
            offset=10,
        ),
    )

    # Base points
    points = base.mark_circle(size=55).encode(
        x=alt.X(
            "Goal_Diff:Q",
            title="Goal Differential (GF − GA)",
            scale=alt.Scale(zero=True),
        ),
        y=alt.Y(
            "Win_Pct_Pct:Q",
            title="Winning Percentage (Win %)",
            scale=alt.Scale(domain=[10, 80]),
            axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'"),
        ),
        color=alt.Color("Status:N", title=None, scale=color_scale),
        opacity=alt.condition(
            alt.datum.Status == "Other Teams",
            alt.value(0.20),
            alt.value(0.85)
        ),


        tooltip=[
            alt.Tooltip("Team_Name:N", title="Team"),
            alt.Tooltip("Year:O", title="Season"),
            alt.Tooltip("Win_Pct_Pct:Q", title="Win %", format=".2f"),
            alt.Tooltip("Goals_For:Q", title="Goals For (GF)"),
            alt.Tooltip("Goals_Against:Q", title="Goals Against (GA)"),
            alt.Tooltip("Goal_Diff:Q", title="Differential (GF-GA)"),
            alt.Tooltip("Wins:Q", title="Wins"),
            alt.Tooltip("Losses:Q", title="Losses"),
        ],
    )

    # Zero differential reference rule
    zero_rule = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#CBD5E1", strokeWidth=1)
        .encode(x="zero:Q")
    )

    # 50% Win Pct reference rule
    mean_rule = (
        alt.Chart(pd.DataFrame({"fifty": [50.0]}))
        .mark_rule(color="#CBD5E1", strokeWidth=1, strokeDash=[3, 3])
        .encode(y="fifty:Q")
    )

    # Linear trendline
    trend = base.transform_regression("Goal_Diff", "Win_Pct_Pct").mark_line(
        color=THEME_PRIMARY,
        strokeWidth=2,
        opacity=0.75
    ).encode(
        x="Goal_Diff:Q",
        y="Win_Pct_Pct:Q",
    )

    chart = (points + zero_rule + mean_rule + trend)
    return apply_theme_config(chart)


def build_team_timeline_chart(
    df: pd.DataFrame,
    teams: List[str],
    metric: str = "Win_Pct"
) -> alt.Chart:
    """
    Constructs a multi-team historical trajectory chart across season starting years.
    """
    if df.empty or not teams:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    chart_df = df[df["Team_Name"].isin(teams)].copy()
    if chart_df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    is_pct = metric == "Win_Pct"
    if is_pct:
        chart_df["Metric_Val"] = (chart_df["Win_Pct"] * 100.0).round(2)
        y_title = "Winning Percentage (Win %)"
        y_axis = alt.Axis(format=".0f", labelExpr="datum.value + '%'")
    else:
        chart_df["Metric_Val"] = chart_df[metric]
        y_title = metric.replace("_", " ")
        y_axis = alt.Axis(format="d")

    base = alt.Chart(chart_df).properties(
        height=320,
        title=alt.TitleParams(
            text=f"Historical Trajectory: {y_title} Over Time",
            subtitle="Comparing franchise performance across active recorded seasons",
            color=THEME_TEXT_PRIMARY,
            subtitleColor=THEME_TEXT_MUTED,
            fontSize=13,
            fontWeight=600,
            anchor="start",
            offset=10,
        ),
    )

    lines = base.mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=alt.X(
            "Year:O",
            title="Season Starting Year",
            axis=alt.Axis(labelAngle=-45),
        ),
        y=alt.Y("Metric_Val:Q", title=y_title, axis=y_axis),
        color=alt.Color("Team_Name:N", title="Team", scale=alt.Scale(scheme="tableau10")),
        tooltip=[
            alt.Tooltip("Team_Name:N", title="Team"),
            alt.Tooltip("Year:O", title="Season"),
            alt.Tooltip("Metric_Val:Q", title=y_title, format=".2f" if is_pct else "d"),
            alt.Tooltip("Wins:Q", title="Wins"),
            alt.Tooltip("Losses:Q", title="Losses"),
            alt.Tooltip("Goal_Diff:Q", title="Goal Diff"),
        ],
    )

    # 2004 Lockout vertical rule annotation
    lockout_df = pd.DataFrame({"Year": [2004], "Label": ["2004 Lockout"]})
    lockout_rule = (
        alt.Chart(lockout_df)
        .mark_rule(color="#F43F5E", strokeWidth=1.5, strokeDash=[4, 4], opacity=0.7)
        .encode(x="Year:O")
    )

    chart = (lines + lockout_rule)
    return apply_theme_config(chart)


def build_h2h_grouped_bar_chart(
    h2h_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    metric_type: str = "win_pct"
) -> alt.Chart:
    """
    Constructs a grouped side-by-side bar chart comparing two teams across common seasons.
    """
    if h2h_df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    # Reshape h2h_df into long format for Altair
    long_rows = []
    for _, row in h2h_df.iterrows():
        yr = int(row["Year"])
        if metric_type == "win_pct":
            val_a = float(row[f"{team_a} Win %"]) * 100.0
            val_b = float(row[f"{team_b} Win %"]) * 100.0
            metric_title = "Winning Percentage (Win %)"
        elif metric_type == "goals":
            val_a = int(row[f"{team_a} GF"])
            val_b = int(row[f"{team_b} GF"])
            metric_title = "Goals For (GF)"
        else:
            val_a = int(row[f"{team_a} Diff"])
            val_b = int(row[f"{team_b} Diff"])
            metric_title = "Goal Differential (Diff)"

        long_rows.append({"Year": yr, "Team": team_a, "Value": val_a})
        long_rows.append({"Year": yr, "Team": team_b, "Value": val_b})

    long_df = pd.DataFrame(long_rows)

    chart = (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Team:N", title=None, axis=None),
            y=alt.Y(
                "Value:Q",
                title=metric_title,
                axis=alt.Axis(
                    format=".0f" if metric_type != "win_pct" else ".0f",
                    labelExpr="datum.value + '%'" if metric_type == "win_pct" else "datum.value",
                ),
            ),
            color=alt.Color(
                "Team:N",
                title="Team",
                scale=alt.Scale(domain=[team_a, team_b], range=[THEME_PRIMARY, "#F59E0B"]),
            ),
            column=alt.Column(
                "Year:O",
                title="Common Seasons",
                header=alt.Header(
                    labelColor=THEME_TEXT_MUTED,
                    titleColor=THEME_TEXT_PRIMARY,
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Year:O", title="Season"),
                alt.Tooltip("Value:Q", title=metric_title, format=".2f" if metric_type == "win_pct" else "d"),
            ],
        )
        .properties(height=240)
    )

    return apply_theme_config(chart)


def build_cv_rmse_comparison_chart(candidate_models: List[Dict[str, Any]]) -> alt.Chart:
    """
    Constructs a cross-validation RMSE comparison bar chart with ±1 SD fold
    variability error bars for all 5 candidate models.
    """
    if not candidate_models:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    df = pd.DataFrame(candidate_models).copy()
    df["Status"] = df["selected"].apply(lambda s: "Selected Model (Lowest CV RMSE)" if s else "Candidate Model")
    df["cv_rmse_low"] = df["cv_rmse"] - df["cv_rmse_sd"]
    df["cv_rmse_high"] = df["cv_rmse"] + df["cv_rmse_sd"]
    df["RMSE_Pct"] = (df["cv_rmse"] * 100.0).round(2)
    df["SD_Pct"] = (df["cv_rmse_sd"] * 100.0).round(2)

    base = alt.Chart(df).encode(
        x=alt.X("model:N", title="Candidate Model Architecture", sort=None, axis=alt.Axis(labelAngle=0)),
    )

    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=alt.Y(
            "cv_rmse:Q",
            title="Mean Cross-Validation RMSE (Lower is Better)",
            axis=alt.Axis(format=".3f"),
            scale=alt.Scale(domain=[0.0, 0.085]),
        ),
        color=alt.Color(
            "Status:N",
            title="Model Selection",
            scale=alt.Scale(
                domain=["Selected Model (Lowest CV RMSE)", "Candidate Model"],
                range=[THEME_PRIMARY, "#94A3B8"],
            ),
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("cv_rmse:Q", title="CV RMSE", format=".4f"),
            alt.Tooltip("cv_rmse_sd:Q", title="Fold SD (±)", format=".4f"),
            alt.Tooltip("cv_mae:Q", title="CV MAE", format=".4f"),
            alt.Tooltip("cv_r2:Q", title="CV R²", format=".4f"),
            alt.Tooltip("selection_notes:N", title="Selection Note"),
        ],
    )

    error_bars = base.mark_rule(color=THEME_NAVY, strokeWidth=2).encode(
        y="cv_rmse_low:Q",
        y2="cv_rmse_high:Q",
    )

    tick_top = base.mark_tick(color=THEME_NAVY, thickness=2, size=14).encode(
        y="cv_rmse_high:Q",
    )
    tick_bottom = base.mark_tick(color=THEME_NAVY, thickness=2, size=14).encode(
        y="cv_rmse_low:Q",
    )

    chart = (bars + error_bars + tick_top + tick_bottom).properties(height=280)
    return apply_theme_config(chart)


def build_test_rmse_comparison_chart(candidate_models: List[Dict[str, Any]]) -> alt.Chart:
    """
    Constructs a held-out test set RMSE comparison bar chart across the 5 models.
    """
    if not candidate_models:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    df = pd.DataFrame(candidate_models).copy()
    df["Status"] = df["selected"].apply(lambda s: "Selected Model (OLS)" if s else "Candidate Model")

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("model:N", title="Candidate Model", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "test_rmse:Q",
                title="Held-Out Test RMSE (2008–2011)",
                axis=alt.Axis(format=".3f"),
                scale=alt.Scale(domain=[0.0, 0.070]),
            ),
            color=alt.Color(
                "Status:N",
                title="Status",
                scale=alt.Scale(
                    domain=["Selected Model (OLS)", "Candidate Model"],
                    range=[THEME_PRIMARY, "#94A3B8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("test_rmse:Q", title="Test RMSE", format=".4f"),
                alt.Tooltip("test_mae:Q", title="Test MAE", format=".4f"),
                alt.Tooltip("test_r2:Q", title="Test R²", format=".4f"),
            ],
        )
        .properties(height=280)
    )
    return apply_theme_config(chart)


def build_actual_vs_predicted_chart(test_df: pd.DataFrame) -> alt.Chart:
    """
    Constructs an Actual vs. Predicted Win Percentage scatter plot on the held-out test set
    with a 45-degree reference line (y = x).
    """
    if test_df.empty or "Predicted_Win_Pct" not in test_df.columns:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    chart_df = test_df.copy()
    chart_df["Actual_Pct"] = (chart_df["Win_Pct"] * 100.0).round(1)
    chart_df["Pred_Pct"] = (chart_df["Predicted_Win_Pct"] * 100.0).round(1)
    chart_df["Resid_Pct"] = (chart_df["Residual"] * 100.0).round(1)

    min_val = min(float(test_df["Win_Pct"].min()), float(test_df["Predicted_Win_Pct"].min())) - 0.03
    max_val = max(float(test_df["Win_Pct"].max()), float(test_df["Predicted_Win_Pct"].max())) + 0.03

    diag_df = pd.DataFrame({"Actual": [min_val, max_val], "Predicted": [min_val, max_val]})
    ref_line = (
        alt.Chart(diag_df)
        .mark_line(strokeDash=[4, 4], color="#64748B", size=1.5)
        .encode(x="Actual:Q", y="Predicted:Q")
    )

    points = (
        alt.Chart(chart_df)
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X(
                "Win_Pct:Q",
                title="Actual Team Win Percentage",
                scale=alt.Scale(domain=[min_val, max_val]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "Predicted_Win_Pct:Q",
                title="Predicted Team Win Percentage",
                scale=alt.Scale(domain=[min_val, max_val]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Abs_Error:Q",
                title="|Error| (Win %)",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                alt.Tooltip("Team_Name:N", title="Franchise"),
                alt.Tooltip("Year:O", title="Season"),
                alt.Tooltip("Actual_Pct:Q", title="Actual Win %", format=".1f"),
                alt.Tooltip("Pred_Pct:Q", title="Predicted Win %", format=".1f"),
                alt.Tooltip("Resid_Pct:Q", title="Residual (Actual - Pred %)", format=".1f"),
                alt.Tooltip("Goals_For:Q", title="Goals For"),
                alt.Tooltip("Goals_Against:Q", title="Goals Against"),
            ],
        )
    )

    chart = (ref_line + points).properties(height=320)
    return apply_theme_config(chart)


def build_residual_distribution_chart(test_df: pd.DataFrame) -> alt.Chart:
    """
    Constructs a histogram of test residuals (Actual - Predicted) with a vertical zero marker.
    """
    if test_df.empty or "Residual" not in test_df.columns:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    hist = (
        alt.Chart(test_df)
        .mark_bar(color=THEME_PRIMARY, opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Residual:Q", bin=alt.Bin(maxbins=20), title="Residual (Actual - Predicted Win %)"),
            y=alt.Y("count()", title="Frequency (Team-Seasons)"),
            tooltip=[
                alt.Tooltip("count()", title="Count"),
            ],
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="#F43F5E", strokeDash=[3, 3], size=2)
        .encode(x="x:Q")
    )

    chart = (hist + zero_line).properties(height=240)
    return apply_theme_config(chart)


def build_error_by_season_chart(season_df: pd.DataFrame) -> alt.Chart:
    """
    Constructs a bar chart showing Test MAE and RMSE across test seasons (2008–2011).
    """
    if season_df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    long_rows = []
    for _, row in season_df.iterrows():
        long_rows.append({"Season": str(int(row["Season"])), "Metric": "MAE", "Value": float(row["MAE"])})
        long_rows.append({"Season": str(int(row["Season"])), "Metric": "RMSE", "Value": float(row["RMSE"])})
    long_df = pd.DataFrame(long_rows)

    chart = (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Metric:N", title=None, axis=None),
            y=alt.Y(
                "Value:Q",
                title="Error in Win % Units",
                axis=alt.Axis(format=".3f"),
                scale=alt.Scale(domain=[0.0, 0.045]),
            ),
            color=alt.Color(
                "Metric:N",
                title="Metric",
                scale=alt.Scale(domain=["MAE", "RMSE"], range=[THEME_PRIMARY, "#F59E0B"]),
            ),
            column=alt.Column(
                "Season:O",
                title="Test Season (30 Teams Each)",
                header=alt.Header(
                    labelColor=THEME_TEXT_MUTED,
                    titleColor=THEME_TEXT_PRIMARY,
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip("Season:O", title="Season"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Value:Q", title="Value", format=".4f"),
            ],
        )
        .properties(height=220)
    )
    return apply_theme_config(chart)


def build_clt_simulation_chart(
    sim_df: pd.DataFrame,
    pop_mean: float,
    theo_se: float,
) -> alt.Chart:
    """
    Constructs a histogram of 1,000 simulated sample means with reference line at population mean.
    Demonstrates Central Limit Theorem normal convergence.
    """
    if sim_df.empty or "Sample_Mean" not in sim_df.columns:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    hist = (
        alt.Chart(sim_df)
        .mark_bar(color=THEME_PRIMARY, opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "Sample_Mean:Q",
                bin=alt.Bin(maxbins=25),
                title="Simulated Sample Mean (n = 30)",
                axis=alt.Axis(format=".3f"),
            ),
            y=alt.Y("count()", title="Simulations Count"),
            tooltip=[
                alt.Tooltip("count()", title="Simulations in Bin"),
            ],
        )
    )

    pop_line = (
        alt.Chart(pd.DataFrame({"x": [pop_mean]}))
        .mark_rule(color="#F43F5E", strokeDash=[4, 4], size=2)
        .encode(x="x:Q")
    )

    chart = (hist + pop_line).properties(height=260)
    return apply_theme_config(chart)


def build_anova_decade_boxplot(df: pd.DataFrame) -> alt.Chart:
    """
    Constructs a boxplot showing Win % distribution across 1990s, 2000s, and 2010s.
    """
    if df.empty or "Win_Pct" not in df.columns:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    chart_df = df.copy()
    if "Decade" not in chart_df.columns:
        chart_df["Decade"] = chart_df["Year"].apply(
            lambda y: "1990s" if y < 2000 else ("2000s" if y < 2010 else "2010s")
        )

    chart = (
        alt.Chart(chart_df)
        .mark_boxplot(
            size=45,
            color=THEME_PRIMARY,
            median=dict(color=THEME_NAVY, size=2),
            ticks=dict(color=THEME_TEXT_MUTED),
        )
        .encode(
            x=alt.X("Decade:N", title="Historical Era / Decade", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Win_Pct:Q",
                title="Team Winning Percentage",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Decade:N",
                title="Era",
                scale=alt.Scale(
                    domain=["1990s", "2000s", "2010s"],
                    range=["#64748B", THEME_PRIMARY, "#F59E0B"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Decade:N", title="Era"),
                alt.Tooltip("count()", title="Sample Size"),
                alt.Tooltip("mean(Win_Pct):Q", title="Mean Win %", format=".3f"),
                alt.Tooltip("median(Win_Pct):Q", title="Median Win %", format=".3f"),
            ],
        )
        .properties(height=280)
    )
    return apply_theme_config(chart)


def build_chisquare_heatmap(contingency_df: pd.DataFrame) -> alt.Chart:
    """
    Constructs a contingency table heatmap showing franchise counts across Eras and Performance Tiers.
    """
    if contingency_df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    # Flatten crosstab
    long_rows = []
    for decade, row in contingency_df.iterrows():
        total_era = float(row.sum())
        for tier, count in row.items():
            pct = (count / total_era * 100.0) if total_era > 0 else 0.0
            long_rows.append({
                "Decade": decade,
                "Tier": str(tier),
                "Count": int(count),
                "Percentage": pct,
                "Label": f"{int(count)} ({pct:.1f}%)",
            })
    flat_df = pd.DataFrame(long_rows)

    base = alt.Chart(flat_df).encode(
        x=alt.X("Tier:N", title="Performance Tier", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Decade:N", title="Historical Era", sort=["1990s", "2000s", "2010s"]),
    )

    rect = base.mark_rect().encode(
        color=alt.Color(
            "Count:Q",
            title="Team Count",
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=[
            alt.Tooltip("Decade:N", title="Era"),
            alt.Tooltip("Tier:N", title="Tier"),
            alt.Tooltip("Count:Q", title="Franchise Seasons"),
            alt.Tooltip("Percentage:Q", title="Era Share", format=".1f"),
        ],
    )

    text = base.mark_text(color="#142D42", fontSize=12, fontWeight=700).encode(
        text="Label:N"
    )

    chart = (rect + text).properties(height=240)
    return apply_theme_config(chart)

