"""
evaluation.py
-------------
Evaluation plots built with Plotly using a clean light-mode clinical design (pure white background, zero dark blue).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_EMERALD = "#10B981"
COLOR_CRIMSON = "#E11D48"
COLOR_AMBER = "#F59E0B"
COLOR_PURPLE = "#8B5CF6"
LIGHT_PAPER_BG = "#FFFFFF"
LIGHT_PLOT_BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"
TEXT_DARK = "#0F172A"


def actual_vs_predicted_scatter(y_true, y_pred, title, axis_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(size=8, color=COLOR_EMERALD, opacity=0.8, line=dict(width=1, color="#059669")),
        name="Patients",
    ))
    lo, hi = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=COLOR_CRIMSON, dash="dash", width=2), name="Perfect Prediction (1:1)",
    ))
    fig.update_layout(
        title=title, xaxis_title=f"Actual {axis_label}", yaxis_title=f"Predicted {axis_label}",
        paper_bgcolor=LIGHT_PAPER_BG, plot_bgcolor=LIGHT_PLOT_BG,
        font=dict(color=TEXT_DARK, family="Inter, sans-serif"),
        height=420,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


def feature_importance_bar(fi_df: pd.DataFrame, top_n=12, title="Feature Importance"):
    top = fi_df.head(top_n).iloc[::-1]
    fig = px.bar(top, x="Importance", y="Feature", orientation="h", title=title, color_discrete_sequence=[COLOR_EMERALD])
    fig.update_layout(
        paper_bgcolor=LIGHT_PAPER_BG, plot_bgcolor=LIGHT_PLOT_BG,
        font=dict(color=TEXT_DARK, family="Inter, sans-serif"), height=420,
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR)
    )
    return fig


def residual_plot(y_true, y_pred, title, axis_label):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(size=8, color=COLOR_AMBER, opacity=0.8)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_CRIMSON)
    fig.update_layout(
        title=title, xaxis_title=f"Predicted {axis_label}", yaxis_title="Residual",
        paper_bgcolor=LIGHT_PAPER_BG, plot_bgcolor=LIGHT_PLOT_BG,
        font=dict(color=TEXT_DARK, family="Inter, sans-serif"), height=420,
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR)
    )
    return fig


def error_distribution_hist(y_true, y_pred, title):
    errors = np.asarray(y_pred) - np.asarray(y_true)
    fig = px.histogram(errors, nbins=30, title=title, labels={"value": "Prediction Error"}, color_discrete_sequence=[COLOR_PURPLE])
    fig.update_layout(
        paper_bgcolor=LIGHT_PAPER_BG, plot_bgcolor=LIGHT_PLOT_BG,
        font=dict(color=TEXT_DARK, family="Inter, sans-serif"), height=420, showlegend=False,
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR)
    )
    return fig
