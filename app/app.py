"""Ecuador Energy Anomaly Detector — Streamlit App."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
IMAGES    = ROOT / "docs" / "images"

st.set_page_config(
    page_title="Ecuador Energy · Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar limpio */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem;
    padding: 6px 0;
}

/* Tipografía base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* KPI cards */
.kpi-row { display: flex; gap: 14px; margin-bottom: 20px; }
.kpi {
    flex: 1;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 10px 10px 0 0;
    background: var(--accent, #3b82f6);
}
.kpi-label { font-size: 0.72rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-sub   { font-size: 0.75rem; color: #64748b; margin-top: 3px; }
.kpi-green  { --accent: #22c55e; }
.kpi-blue   { --accent: #3b82f6; }
.kpi-orange { --accent: #f59e0b; }
.kpi-red    { --accent: #ef4444; }
.kpi-purple { --accent: #8b5cf6; }

/* Highlight banner */
.banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.banner-icon { font-size: 2rem; }
.banner-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
.banner-sub   { font-size: 0.82rem; color: #94a3b8; }

/* Sección */
.sec-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #64748b;
    margin: 24px 0 10px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
}

/* Tag badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-orange { background: #fef3c7; color: #92400e; }
.badge-gray   { background: #f1f5f9; color: #475569; }

/* Ocultar footer de Streamlit */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_results() -> pd.DataFrame:
    p = PROCESSED / "latam_multitechnique_results.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(PROCESSED / "latam_multitechnique_results.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

@st.cache_data
def load_metrics() -> dict:
    return json.loads((PROCESSED / "metrics.json").read_text())

@st.cache_data
def load_baselines() -> dict:
    return json.loads((PROCESSED / "baselines_comparison.json").read_text())

@st.cache_data
def load_ci() -> dict:
    return json.loads((PROCESSED / "confidence_intervals.json").read_text())

@st.cache_data
def load_cross() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "cross_country_validation.csv")

@st.cache_data
def load_hydro() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "hydro_dependency_analysis.csv")

@st.cache_data
def load_sens() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "sensitivity_analysis.csv")


# ── Constantes ────────────────────────────────────────────────────────────────
CRISIS_BANDS = [
    {"label": "Apagones · abr–jun 2024", "start": "2024-04-01", "end": "2024-06-30", "color": "rgba(245,158,11,0.12)"},
    {"label": "Crisis · oct–dic 2024",   "start": "2024-09-15", "end": "2024-12-31", "color": "rgba(239,68,68,0.15)"},
]
PLOTLY_TEMPLATE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#334155"),
)
M = lambda t=36, b=44, l=12, r=12: dict(margin=dict(t=t, b=b, l=l, r=r))
METHOD_COLOR = {
    "IF":        "#3b82f6",
    "STL":       "#f59e0b",
    "CUSUM":     "#8b5cf6",
    "Consensus": "#ef4444",
}


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df_all    = load_results()
    metrics   = load_metrics()
    baselines = load_baselines()
    ci_data   = load_ci()
    df_cross  = load_cross()
    df_hydro  = load_hydro()
    df_sens   = load_sens()
except Exception as exc:
    st.error(f"**No se encontraron datos procesados.** `{exc}`\n\nEjecuta: `python scripts/train_model.py`")
    st.stop()

df_ec = df_all[df_all["pais"] == "Ecuador"].sort_values("fecha").copy()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Ecuador Energy")
    st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-top:-8px">Anomaly Detection · Fase 1</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "",
        ["Overview", "Detector", "Modelos", "Cross-Country", "Metodología"],
        format_func=lambda x: {
            "Overview":      "📊  Resumen",
            "Detector":      "🔍  Detector por País",
            "Modelos":       "📈  Comparación de Modelos",
            "Cross-Country": "🌎  Validación Cruzada",
            "Metodología":   "📋  Metodología",
        }[x],
    )
    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem;color:#64748b;line-height:1.8">
<b style="color:#94a3b8">Datos</b><br>Ember CC BY 4.0<br>
<b style="color:#94a3b8">Cobertura</b><br>8 países · 784 meses<br>
<b style="color:#94a3b8">Técnicas</b><br>IF + STL + CUSUM<br>
<b style="color:#94a3b8">Consenso F1</b><br>0.750 · MCC 0.765
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem;color:#64748b">
Diego Fernando Lojan Tenesaca<br>
<a href="https://github.com/DiegoFernandoLojanTenesaca" style="color:#60a5fa">GitHub</a> ·
<a href="https://linkedin.com/in/diego-fernando-lojan" style="color:#60a5fa">LinkedIn</a>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    # Banner principal
    st.markdown("""
<div class="banner">
  <div class="banner-icon">⚡</div>
  <div>
    <div class="banner-title">Ecuador Energy — Detección de Anomalías en el Sector Eléctrico</div>
    <div class="banner-sub">
      Isolation Forest · STL Decomposition · CUSUM · Consenso multi-técnica &nbsp;|&nbsp;
      8 países · 784 meses · Datos reales Ember 2018–2025
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    strict = metrics.get("strict", {}).get("Consensus", {})

    # KPIs
    st.markdown("""
<div class="kpi-row">
  <div class="kpi kpi-red">
    <div class="kpi-label">Consenso F1</div>
    <div class="kpi-value">0.750</div>
    <div class="kpi-sub">GT estricto oct–dic 2024</div>
  </div>
  <div class="kpi kpi-green">
    <div class="kpi-label">Recall</div>
    <div class="kpi-value">100%</div>
    <div class="kpi-sub">3/3 meses de crisis detectados</div>
  </div>
  <div class="kpi kpi-blue">
    <div class="kpi-label">MCC</div>
    <div class="kpi-value">0.765</div>
    <div class="kpi-sub">Matthews Correlation Coefficient</div>
  </div>
  <div class="kpi kpi-orange">
    <div class="kpi-label">Países analizados</div>
    <div class="kpi-value">8</div>
    <div class="kpi-sub">784 meses · LATAM</div>
  </div>
  <div class="kpi kpi-purple">
    <div class="kpi-label">Modelos comparados</div>
    <div class="kpi-value">9</div>
    <div class="kpi-sub">IF · LOF · SVM · ARIMA · Prophet …</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Serie temporal + mix
    col_l, col_r = st.columns([3, 1], gap="large")

    with col_l:
        st.markdown('<div class="sec-title">Generación Hidroeléctrica — Ecuador (señal principal)</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_ec["fecha"], y=df_ec["gen_hydro"],
            fill="tozeroy", name="Hidro (%)",
            line=dict(color="#3b82f6", width=1.8),
            fillcolor="rgba(59,130,246,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=df_ec["fecha"], y=df_ec["gen_fossil"],
            fill="tozeroy", name="Fósil (%)",
            line=dict(color="#ef4444", width=1.4),
            fillcolor="rgba(239,68,68,0.08)",
        ))
        # Anomalías
        anom = df_ec[df_ec["consensus"] == 1]
        fig.add_trace(go.Scatter(
            x=anom["fecha"], y=anom["gen_hydro"],
            mode="markers", name="Anomalía detectada",
            marker=dict(color="#ef4444", size=9, symbol="x-thin-open", line=dict(width=2.5)),
        ))
        for b in CRISIS_BANDS:
            fig.add_vrect(x0=b["start"], x1=b["end"], fillcolor=b["color"], layer="below",
                          annotation_text=b["label"], annotation_position="top left",
                          annotation_font_size=9, annotation_font_color="#92400e")
        fig.update_layout(
            **PLOTLY_TEMPLATE, height=310,
            xaxis_title=None, yaxis_title="% de generación",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font_size=11),
            yaxis=dict(gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-title">Mix promedio</div>', unsafe_allow_html=True)
        avg = df_ec[["gen_hydro","gen_fossil","gen_solar","gen_wind","gen_bioenergy"]].mean()
        fig_pie = go.Figure(go.Pie(
            labels=["Hidro","Fósil","Solar","Eólica","Bioenergía"],
            values=avg.values,
            marker_colors=["#3b82f6","#ef4444","#f59e0b","#22c55e","#8b5cf6"],
            hole=0.50,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig_pie.update_layout(**PLOTLY_TEMPLATE, height=310, showlegend=False, **M(t=10,b=10,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Demanda + CO2
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sec-title">Demanda eléctrica (TWh)</div>', unsafe_allow_html=True)
        fig_d = go.Figure()
        normal = df_ec[df_ec["consensus"] == 0]
        anom_d = df_ec[df_ec["consensus"] == 1]
        fig_d.add_trace(go.Scatter(
            x=df_ec["fecha"], y=df_ec["demanda_twh"],
            mode="lines", name="Demanda",
            line=dict(color="#64748b", width=1.6),
        ))
        fig_d.add_trace(go.Scatter(
            x=anom_d["fecha"], y=anom_d["demanda_twh"],
            mode="markers", name="Anomalía",
            marker=dict(color="#ef4444", size=9, symbol="x-thin-open", line=dict(width=2.5)),
        ))
        for b in CRISIS_BANDS:
            fig_d.add_vrect(x0=b["start"], x1=b["end"], fillcolor=b["color"], layer="below")
        fig_d.update_layout(**PLOTLY_TEMPLATE, height=230, xaxis_title=None, yaxis_title="TWh",
            showlegend=False, yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_d, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">Intensidad de CO₂ (gCO₂/kWh)</div>', unsafe_allow_html=True)
        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(
            x=df_ec["fecha"], y=df_ec["co2_intensity"],
            fill="tozeroy", mode="lines",
            line=dict(color="#f59e0b", width=1.6),
            fillcolor="rgba(245,158,11,0.10)",
        ))
        anom_co2 = df_ec[df_ec["consensus"] == 1]
        fig_co2.add_trace(go.Scatter(
            x=anom_co2["fecha"], y=anom_co2["co2_intensity"],
            mode="markers",
            marker=dict(color="#ef4444", size=9, symbol="x-thin-open", line=dict(width=2.5)),
            showlegend=False,
        ))
        for b in CRISIS_BANDS:
            fig_co2.add_vrect(x0=b["start"], x1=b["end"], fillcolor=b["color"], layer="below")
        fig_co2.update_layout(**PLOTLY_TEMPLATE, height=230, xaxis_title=None, yaxis_title="gCO₂/kWh",
            showlegend=False, yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_co2, use_container_width=True)

    # Tabla anomalías
    st.markdown('<div class="sec-title">Meses anómalos detectados — Ecuador · Consenso ≥2 técnicas</div>', unsafe_allow_html=True)
    tbl = df_ec[df_ec["consensus"] == 1][[
        "fecha","if_anomaly","stl_anomaly","cusum_anomaly","gen_hydro","gen_fossil","co2_intensity","demanda_twh"
    ]].copy()
    tbl["fecha"] = tbl["fecha"].dt.strftime("%b %Y")
    tbl["if_anomaly"]   = tbl["if_anomaly"].map({1:"✓",0:"–"})
    tbl["stl_anomaly"]  = tbl["stl_anomaly"].map({1:"✓",0:"–"})
    tbl["cusum_anomaly"] = tbl["cusum_anomaly"].map({1:"✓",0:"–"})
    tbl.columns = ["Fecha","IF","STL","CUSUM","Hidro %","Fósil %","CO₂ g/kWh","Demanda TWh"]
    st.dataframe(tbl.set_index("Fecha"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# DETECTOR POR PAÍS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Detector":
    st.markdown('<h2 style="margin-bottom:4px">🔍 Detector por País</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-top:0;margin-bottom:20px">Resultados de Isolation Forest · STL · CUSUM · Consenso para cada país LATAM.</p>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.6], gap="medium")
    with col_a:
        pais = st.selectbox("País", sorted(df_all["pais"].unique()),
                            index=list(sorted(df_all["pais"].unique())).index("Ecuador"))
    with col_b:
        metodo_map = {
            "Consenso ≥2":      "consensus",
            "Isolation Forest": "if_anomaly",
            "STL":              "stl_anomaly",
            "CUSUM":            "cusum_anomaly",
        }
        metodo_lbl = st.selectbox("Método", list(metodo_map.keys()))
        metodo_col = metodo_map[metodo_lbl]
    with col_c:
        pass  # espacio

    df_p = df_all[df_all["pais"] == pais].sort_values("fecha").copy()

    # KPIs del país
    cross_p = df_cross[df_cross["country"] == pais]
    row_c = cross_p[cross_p["method"] == "Consensus"]
    if not row_c.empty:
        r = row_c.iloc[0]
        hydro_avg = df_p["gen_hydro"].mean() if "gen_hydro" in df_p.columns else 0
        ka, kb, kc, kd = st.columns(4)
        ka.metric("F1 Consenso",   f"{r['f1']:.3f}")
        kb.metric("Precision",     f"{r['precision']:.3f}")
        kc.metric("Recall",        f"{r['recall']:.3f}")
        kd.metric("Hidro promedio",f"{hydro_avg:.1f}%")

    # Serie temporal
    col_m = "gen_hydro" if "gen_hydro" in df_p.columns else "demanda_twh"
    lbl_m = "Generación Hidro (%)" if col_m == "gen_hydro" else "Demanda (TWh)"
    anom_p = df_p[df_p[metodo_col] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_p["fecha"], y=df_p[col_m], mode="lines",
        name=lbl_m, line=dict(color="#3b82f6", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=anom_p["fecha"], y=anom_p[col_m], mode="markers",
        name=metodo_lbl,
        marker=dict(color="#ef4444", size=10, symbol="x-thin-open", line=dict(width=2.5)),
    ))
    if pais == "Ecuador":
        for b in CRISIS_BANDS:
            fig.add_vrect(x0=b["start"], x1=b["end"], fillcolor=b["color"], layer="below",
                          annotation_text=b["label"], annotation_position="top left",
                          annotation_font_size=9)
    fig.update_layout(**PLOTLY_TEMPLATE, height=300, xaxis_title=None, yaxis_title=lbl_m,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
    st.plotly_chart(fig, use_container_width=True)

    # Comparación 4 técnicas
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="sec-title">Anomalías por técnica</div>', unsafe_allow_html=True)
        tecnicas = [("Isolation Forest","if_anomaly"), ("STL","stl_anomaly"), ("CUSUM","cusum_anomaly"), ("Consenso ≥2","consensus")]
        rows = [{"Técnica": nm, "Anomalías": int(df_p[col].sum()), "% meses": f"{df_p[col].mean()*100:.1f}%"}
                for nm, col in tecnicas if col in df_p.columns]
        st.dataframe(pd.DataFrame(rows).set_index("Técnica"), use_container_width=True)

    with col_right:
        st.markdown('<div class="sec-title">Acuerdo entre técnicas</div>', unsafe_allow_html=True)
        if "n_techniques" in df_p.columns:
            vc = df_p["n_techniques"].value_counts().sort_index()
            fig_bar = go.Figure(go.Bar(
                x=vc.index.astype(str), y=vc.values,
                marker_color=["#e2e8f0","#93c5fd","#60a5fa","#3b82f6"][:len(vc)],
                text=vc.values, textposition="outside",
            ))
            fig_bar.update_layout(**PLOTLY_TEMPLATE, height=200,
                xaxis_title="Técnicas de acuerdo", yaxis_title="Meses",
                showlegend=False, yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_bar, use_container_width=True)

    # IF score
    if "if_score" in df_p.columns:
        st.markdown('<div class="sec-title">Isolation Forest — Anomaly score (menor = más anómalo)</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_p["fecha"], y=df_p["if_score"],
            fill="tozeroy", mode="lines",
            line=dict(color="#94a3b8", width=1.4),
            fillcolor="rgba(148,163,184,0.12)",
        ))
        fig_sc.add_hline(y=0, line_dash="dot", line_color="#ef4444", line_width=1,
                         annotation_text="umbral", annotation_position="right",
                         annotation_font_size=10)
        fig_sc.update_layout(**PLOTLY_TEMPLATE, height=200, xaxis_title=None, yaxis_title="Score",
            showlegend=False, yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODELOS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Modelos":
    st.markdown('<h2 style="margin-bottom:4px">📈 Comparación de 9 Modelos</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-top:0;margin-bottom:20px">Ecuador · Ground Truth estricto: oct–dic 2024 (3 meses, crisis severa)</p>', unsafe_allow_html=True)

    ORDER = ["Consensus","IF","LOF","Prophet","LSTM-AE","Elliptic","DBSCAN","OC-SVM","ARIMA"]
    NAMES = {"Consensus":"Consenso ≥2","IF":"Isolation Forest","LOF":"LOF",
             "Prophet":"Prophet","LSTM-AE":"LSTM Autoencoder","Elliptic":"Elliptic Envelope",
             "DBSCAN":"DBSCAN","OC-SVM":"One-Class SVM","ARIMA":"ARIMA"}

    rows_bl = []
    for key in ORDER:
        if key in baselines:
            d = baselines[key]
            rows_bl.append({"Modelo": NAMES.get(key,key), "_key": key,
                "Precision": d.get("precision",0), "Recall": d.get("recall",0),
                "F1": d.get("f1",0), "MCC": d.get("mcc",0)})
    df_bl = pd.DataFrame(rows_bl)

    col_tbl, col_bar = st.columns([1.3, 1.7], gap="large")

    with col_tbl:
        st.markdown('<div class="sec-title">Métricas completas</div>', unsafe_allow_html=True)
        disp = df_bl[["Modelo","Precision","Recall","F1","MCC"]].set_index("Modelo")
        st.dataframe(
            disp.style
                .highlight_max(axis=0, color="#dcfce7")
                .highlight_min(axis=0, color="#fee2e2")
                .format("{:.3f}"),
            use_container_width=True, height=340,
        )

    with col_bar:
        st.markdown('<div class="sec-title">F1-Score</div>', unsafe_allow_html=True)
        df_sorted = df_bl.sort_values("F1")
        colors_bar = ["#3b82f6" if k == "Consensus" else "#cbd5e1" for k in df_sorted["_key"]]
        fig_f1 = go.Figure(go.Bar(
            x=df_sorted["F1"], y=df_sorted["Modelo"],
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in df_sorted["F1"]],
            textposition="outside",
        ))
        fig_f1.update_layout(**PLOTLY_TEMPLATE, **M(t=10, b=20, l=130, r=50), height=340,
            xaxis_range=[0, 1.1], xaxis_title="F1-Score",
            yaxis_title=None, showlegend=False,
            yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_f1, use_container_width=True)

    # Bootstrap CIs
    st.markdown('<div class="sec-title">Intervalos de confianza Bootstrap 95% — F1 · Ecuador</div>', unsafe_allow_html=True)

    ci_rows = []
    for method, data in ci_data.items():
        if "f1" in data:
            d = data["f1"]
            ci_rows.append({"Técnica": method,
                "F1 medio": d.get("mean",0), "CI bajo": d.get("ci_low",0), "CI alto": d.get("ci_high",0)})

    if ci_rows:
        col_ci_l, col_ci_r = st.columns(2, gap="large")
        with col_ci_l:
            df_ci = pd.DataFrame(ci_rows).set_index("Técnica")
            st.dataframe(df_ci.style.format("{:.3f}"), use_container_width=True)
            st.caption("⚠️ CIs amplios porque N=73 meses. Con datos diarios (Fase 2, N≈3.400) deberían estrechar a ±0.10–0.15.")

        with col_ci_r:
            fig_ci = go.Figure()
            for row in ci_rows:
                fig_ci.add_trace(go.Bar(
                    name=row["Técnica"], x=[row["Técnica"]], y=[row["F1 medio"]],
                    error_y=dict(type="data", symmetric=False,
                                 array=[row["CI alto"] - row["F1 medio"]],
                                 arrayminus=[row["F1 medio"] - row["CI bajo"]]),
                    text=[f"{row['F1 medio']:.3f}"], textposition="outside",
                    marker_color=METHOD_COLOR.get(row["Técnica"], "#cbd5e1"),
                ))
            fig_ci.update_layout(**PLOTLY_TEMPLATE, **M(t=10, b=20), height=260,
                yaxis_range=[0,1.2], yaxis_title="F1", showlegend=False,
                yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_ci, use_container_width=True)

    # Sensitivity
    st.markdown('<div class="sec-title">Análisis de sensibilidad — Top configuraciones por F1</div>', unsafe_allow_html=True)
    if not df_sens.empty and "f1" in df_sens.columns:
        cols_s = [c for c in ["contamination","stl_sigma","cusum_factor","f1","mcc","precision","recall"] if c in df_sens.columns]
        top = df_sens.nlargest(8,"f1")[cols_s].round(3)
        best = df_sens.nlargest(1,"f1").iloc[0]
        st.dataframe(top.set_index("contamination"), use_container_width=True)
        st.success(
            f"**Mejor:** contamination={best['contamination']} · stl_sigma={best['stl_sigma']} · "
            f"cusum_factor={best['cusum_factor']} → F1={best['f1']:.3f} · MCC={best['mcc']:.3f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-COUNTRY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Cross-Country":
    st.markdown('<h2 style="margin-bottom:4px">🌎 Validación Cruzada — 8 Países</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-top:0;margin-bottom:20px">La efectividad del consenso depende de la fracción de generación hidroeléctrica del país.</p>', unsafe_allow_html=True)

    # Hallazgo clave — banner
    st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:14px 20px;margin-bottom:18px">
  <b style="color:#166534">Hallazgo clave:</b>
  <span style="color:#14532d"> Países con hidro &gt; 30% (Ecuador, Brasil, Colombia): F1 ≥ 0.43.
  Países con hidro &lt; 20% (Chile, Argentina): F1 = 0.00 — sus crisis son térmicas, no hídricas.</span>
</div>
""", unsafe_allow_html=True)

    col_hl, col_hr = st.columns([1.6, 1.4], gap="large")

    with col_hl:
        st.markdown('<div class="sec-title">Hidro % vs F1 Consenso</div>', unsafe_allow_html=True)
        if not df_hydro.empty:
            dh = df_hydro[df_hydro["method"] == "consensus"].copy()
            dh["hydro_pct"] = (dh["hydro_dependency"] * 100).round(1)
            fig_sc = px.scatter(
                dh, x="hydro_pct", y="f1",
                color="hydro_driven_crisis",
                text="country" if "country" in dh.columns else None,
                color_discrete_map={True:"#3b82f6", False:"#ef4444"},
                labels={"hydro_pct":"Hidro %","f1":"F1 Consenso","hydro_driven_crisis":"Crisis hídrica"},
            )
            fig_sc.update_traces(textposition="top center", marker=dict(size=13))
            fig_sc.add_vline(x=30, line_dash="dot", line_color="#94a3b8", line_width=1,
                             annotation_text="30%", annotation_position="top",
                             annotation_font_size=10, annotation_font_color="#64748b")
            fig_sc.update_layout(**PLOTLY_TEMPLATE, height=320,
                yaxis=dict(gridcolor="#f1f5f9", range=[-0.12,1.0]),
                xaxis=dict(gridcolor="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, font_size=11))
            st.plotly_chart(fig_sc, use_container_width=True)

    with col_hr:
        st.markdown('<div class="sec-title">F1 por país — Consenso</div>', unsafe_allow_html=True)
        if "method" in df_cross.columns:
            cons = df_cross[df_cross["method"] == "Consensus"][["country","f1","precision","recall","mcc","n_crisis","crisis_detected"]].copy()
            cons = cons.sort_values("f1", ascending=False)
            cons.columns = ["País","F1","Prec","Recall","MCC","GT meses","Detectados"]
            st.dataframe(
                cons.set_index("País").style
                    .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=0.8)
                    .format({"F1":"{:.3f}","Prec":"{:.3f}","Recall":"{:.3f}","MCC":"{:.3f}"}),
                use_container_width=True, height=300,
            )

    # Barras por método y país
    st.markdown('<div class="sec-title">F1 por país y técnica</div>', unsafe_allow_html=True)
    if "method" in df_cross.columns:
        df_plot = df_cross[df_cross["method"].isin(["IF","STL","CUSUM","Consensus"])].copy()
        fig_grouped = px.bar(
            df_plot, x="country", y="f1", color="method",
            barmode="group",
            color_discrete_map=METHOD_COLOR,
            labels={"country":"País","f1":"F1-Score","method":"Técnica"},
            text_auto=".2f",
        )
        fig_grouped.update_traces(textfont_size=9)
        fig_grouped.update_layout(**PLOTLY_TEMPLATE, height=320,
            yaxis=dict(gridcolor="#f1f5f9", range=[0,1.0]),
            xaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11))
        st.plotly_chart(fig_grouped, use_container_width=True)

    # McNemar
    st.markdown('<div class="sec-title">Test de McNemar — Consenso vs Isolation Forest</div>', unsafe_allow_html=True)
    col_mn1, col_mn2 = st.columns(2, gap="large")
    with col_mn1:
        st.markdown("""
| | | |
|---|---|---|
| **H₀** | Ambos métodos tienen la misma tasa de error | |
| **N** | 73 meses mensuales (Ecuador) | |
| **b₀₁** | 1 (consenso correcto, IF incorrecto) | |
| **b₁₀** | 0 (consenso incorrecto, IF correcto) | |
| **p-valor** | **1.000** — no significativo | ⚠️ |

F1 consenso = 0.750 vs IF = 0.667. Diferencia real, pero insuficiente potencia estadística con N=73.
""")
    with col_mn2:
        st.markdown("""
**Colombia · XM API · N = 1,372 días diarios:**

| | |
|---|---|
| p-valor | **0.0000** ✅ significativo |
| Conclusión | El método funciona; el problema era el N |

Con la Fase 2 (Electricity Maps, ~3.400 días para Ecuador) se espera el mismo resultado.
""")


# ═════════════════════════════════════════════════════════════════════════════
# METODOLOGÍA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Metodología":
    st.markdown('<h2 style="margin-bottom:4px">📋 Metodología</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-top:0;margin-bottom:20px">Decisiones de diseño, fuentes oficiales y limitaciones del estudio.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Técnicas y modelo", "Datos y ground truth", "Limitaciones"])

    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
**3 técnicas complementarias + consenso**

| Técnica | Qué detecta |
|---------|-------------|
| **Isolation Forest** | Outliers multivariados (213 features) |
| **STL Decomposition** | Residuos > 2σ tras trend+seasonality |
| **CUSUM** | Cambios estructurales en generación hidro |

Un mes es anómalo si **≥ 2 técnicas coinciden**.

---

**Consenso ponderado por hidro**

```
alpha   = min(hydro_share × 2, 1.0)
w_IF    = 1.0 − 0.3 × alpha
w_STL   = 0.5 + 0.5 × alpha
w_CUSUM = 0.5 + 0.5 × alpha
```
Países con más hidro → más peso a STL/CUSUM.
""")
        with col2:
            st.markdown("""
**Feature engineering — 24 → 213 variables**

- Rolling mean y σ (6 y 12 meses)
- Lags 1 y 12 meses
- Z-scores vs media móvil
- Ratio hidro/fósil
- Cambio YoY
- Mes, trimestre

**Hiperparámetros optimizados**

| Param | Valor |
|-------|-------|
| n_estimators | 300 |
| contamination | 0.08 |
| stl_sigma | 2.0 |
| cusum_h | 4σ |
| warmup | 12 meses |
""")

    with tab2:
        st.markdown("""
**Dataset principal — Ember Global Electricity Data**
- Licencia: CC BY 4.0 · Cobertura: 8 países, 2018–2025, mensual

---

**Ground truth oficial por país**

| País | Crisis | Fuente |
|------|--------|--------|
| Ecuador | Abr–Dic 2024 | Decreto Ejecutivo No. 229 · CENACE Informe Anual 2024 |
| Brasil | Jun–Nov 2021 | Decreto 10.939/2021 · MP 1.055/2021 |
| Colombia | Ene–Jun 2024 | XM Colombia · precios mayoristas +22.68% |
| Chile | 2019 | U. de Chile · Biblioteca del Congreso Nacional |
| Argentina | Ene–Mar 2022 | SMN Argentina · FARN informe climático |

---

**Validación estadística**

| Variable | μ normal | μ anomalía | p-valor | Cohen's d |
|----------|----------|------------|---------|-----------|
| gen_hydro | 38.71% | 28.72% | 0.0004*** | 2.81 |
| gen_fossil | 11.87% | 21.47% | 0.0004*** | 3.14 |
| co2_intensity | 175.7 | 298.5 | 0.0003*** | 3.15 |
| importaciones | 0.04 | 0.16 | 0.0045** | 1.37 |
| demanda_twh | 2.86 | 2.87 | 0.9702 ns | 0.04 |
""")

    with tab3:
        st.markdown("""
| Limitación | Impacto | Solución prevista |
|------------|---------|-------------------|
| N=73 meses Ecuador | McNemar p=1.0 · CIs [0.40, 1.00] | Fase 2: ~3.400 días diarios |
| Granularidad mensual | No detecta cortes de días | Electricity Maps (diario) |
| Chile F1=0 · Argentina F1=0 | STL/CUSUM no detectan crisis térmicas | Usar solo IF si hidro < 20% |
| Consenso = IF numéricamente | Ponderación no se diferencia en estos datos | Más países · más variabilidad |
| GT Amplio recall=33% | Solo detecta pico de la crisis, no los 9 meses | Resolución diaria: más granularidad |

---

**Referencias**
- Liu et al. (2008). *Isolation Forest*. IEEE ICDM.
- Cleveland et al. (1990). *STL*. J. Official Statistics.
- Page (1954). *CUSUM*. Biometrika.
- CENACE (2025). *Informe Anual 2024*. cenace.gob.ec
- Ember (2026). *Global Electricity Data*. ember-energy.org
""")

    st.divider()
    st.markdown("""
<div style="font-size:0.82rem;color:#64748b">
<b>Diego Fernando Lojan Tenesaca</b> · Data & AI Engineer ·
<a href="https://github.com/DiegoFernandoLojanTenesaca">GitHub</a> ·
<a href="https://linkedin.com/in/diego-fernando-lojan">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
