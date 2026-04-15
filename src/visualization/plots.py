"""Gráficos Plotly para el dashboard de anomalías."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Paleta de colores Ecuador-themed
COLORS = {
    "normal": "#2196F3",
    "anomaly": "#F44336",
    "trend": "#4CAF50",
    "bg": "#FAFAFA",
    "grid": "#E0E0E0",
    "ecuador_yellow": "#FFD100",
    "ecuador_blue": "#0033A0",
    "ecuador_red": "#CE1126",
}


def plot_timeseries_with_anomalies(
    df: pd.DataFrame,
    date_col: str = "fecha",
    value_col: str = None,
    title: str = "Serie Temporal con Anomalías Detectadas",
) -> go.Figure:
    """Serie temporal con anomalías resaltadas en rojo."""
    if value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = [c for c in numeric_cols if c not in ["is_anomaly", "anomaly_score"]][0]

    normal = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    fig = go.Figure()

    # Línea principal
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col],
        mode="lines",
        name="Datos",
        line=dict(color=COLORS["normal"], width=1.5),
        opacity=0.7,
    ))

    # Trend si existe
    trend_col = f"{value_col}_trend"
    if trend_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[trend_col],
            mode="lines",
            name="Tendencia",
            line=dict(color=COLORS["trend"], width=2, dash="dash"),
        ))

    # Puntos de anomalía
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies[date_col], y=anomalies[value_col],
            mode="markers",
            name=f"Anomalías ({len(anomalies)})",
            marker=dict(
                color=COLORS["anomaly"],
                size=8,
                symbol="x",
                line=dict(width=1, color="darkred"),
            ),
            text=anomalies["anomaly_score"].round(3).astype(str),
            hovertemplate=(
                "<b>Fecha</b>: %{x}<br>"
                f"<b>{value_col}</b>: %{{y:.2f}}<br>"
                "<b>Score</b>: %{text}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=value_col.replace("_", " ").title(),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_anomaly_heatmap(
    df: pd.DataFrame,
    date_col: str = "fecha",
    title: str = "Mapa de Calor de Anomalías",
) -> go.Figure:
    """Heatmap de anomalías por mes y día de la semana."""
    if "mes" not in df.columns or "dia_semana" not in df.columns:
        return go.Figure()

    pivot = df.pivot_table(
        values="is_anomaly",
        index="dia_semana",
        columns="mes",
        aggfunc="mean",
    )

    day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[month_names[i-1] for i in pivot.columns],
        y=[day_names[i] for i in pivot.index],
        colorscale=[[0, "#E3F2FD"], [0.5, "#FFF9C4"], [1, "#F44336"]],
        hovertemplate=(
            "<b>%{y}</b> - <b>%{x}</b><br>"
            "Tasa anomalías: %{z:.1%}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
    )

    return fig


def plot_score_distribution(
    df: pd.DataFrame,
    title: str = "Distribución de Anomaly Scores",
) -> go.Figure:
    """Histograma de anomaly scores con separación normal/anomalía."""
    fig = go.Figure()

    normal = df[df["is_anomaly"] == 0]["anomaly_score"]
    anomalies = df[df["is_anomaly"] == 1]["anomaly_score"]

    fig.add_trace(go.Histogram(
        x=normal, name="Normal",
        marker_color=COLORS["normal"], opacity=0.7,
        nbinsx=50,
    ))

    if len(anomalies) > 0:
        fig.add_trace(go.Histogram(
            x=anomalies, name="Anomalía",
            marker_color=COLORS["anomaly"], opacity=0.7,
            nbinsx=30,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Anomaly Score",
        yaxis_title="Frecuencia",
        barmode="overlay",
        template="plotly_white",
    )

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Importancia de Variables (SHAP)",
) -> go.Figure:
    """Gráfico de barras horizontales de importancia de features."""
    top = importance_df.head(top_n).sort_values("mean_abs_shap")

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature"],
        orientation="h",
        marker_color=COLORS["ecuador_blue"],
        text=top["mean_abs_shap"].round(4),
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        template="plotly_white",
        height=max(400, top_n * 30),
    )

    return fig


def plot_anomaly_timeline(
    df: pd.DataFrame,
    date_col: str = "fecha",
    title: str = "Timeline de Anomalías",
) -> go.Figure:
    """Visualiza anomalías como puntos en una timeline con severity."""
    anomalies = df[df["is_anomaly"] == 1].copy()
    if len(anomalies) == 0:
        return go.Figure().update_layout(title="No se encontraron anomalías")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anomalies[date_col],
        y=anomalies["anomaly_score"],
        mode="markers",
        marker=dict(
            size=10,
            color=anomalies["anomaly_score"],
            colorscale="RdYlBu",
            colorbar=dict(title="Score"),
            line=dict(width=1, color="black"),
        ),
        hovertemplate=(
            "<b>Fecha</b>: %{x}<br>"
            "<b>Score</b>: %{y:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Anomaly Score (menor = más anómalo)",
        template="plotly_white",
    )

    return fig


def plot_overview_kpis(df: pd.DataFrame) -> go.Figure:
    """Panel de KPIs principales."""
    n_total = len(df)
    n_anomalies = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
    anomaly_rate = n_anomalies / n_total * 100 if n_total > 0 else 0
    avg_score = df["anomaly_score"].mean() if "anomaly_score" in df.columns else 0

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
    )

    fig.add_trace(go.Indicator(
        mode="number",
        value=n_total,
        title={"text": "Total Registros"},
        number={"font": {"size": 40}},
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=n_anomalies,
        title={"text": "Anomalías"},
        number={"font": {"size": 40, "color": COLORS["anomaly"]}},
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=anomaly_rate,
        title={"text": "Tasa Anomalías (%)"},
        number={"font": {"size": 40}, "suffix": "%"},
        delta={"reference": 5, "relative": False},
    ), row=1, col=3)

    fig.add_trace(go.Indicator(
        mode="number",
        value=avg_score,
        title={"text": "Score Promedio"},
        number={"font": {"size": 40}, "valueformat": ".4f"},
    ), row=1, col=4)

    fig.update_layout(
        template="plotly_white",
        height=200,
        margin=dict(t=50, b=10),
    )

    return fig
