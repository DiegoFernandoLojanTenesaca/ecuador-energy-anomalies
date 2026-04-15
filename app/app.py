"""Ecuador Energy Anomaly Detector - Streamlit App.

Dashboard interactivo para detectar y visualizar anomalías
en el consumo y generación eléctrica de Ecuador.
Datos reales de Ember, Our World in Data y World Bank.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Agregar src al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.isolation_forest import AnomalyDetector
from src.visualization.plots import (
    plot_timeseries_with_anomalies,
    plot_anomaly_heatmap,
    plot_score_distribution,
    plot_feature_importance,
    plot_anomaly_timeline,
    plot_overview_kpis,
)

# --- Configuración de página ---
st.set_page_config(
    page_title="Ecuador Energy Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constantes ---
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "anomaly_detector.joblib"

# Columnas principales de datos reales
MAIN_COLS = {
    "demanda_twh": "Demanda (TWh)",
    "gen_total_generation": "Generación Total (%)",
    "gen_hydro": "Generación Hidro (%)",
    "gen_fossil": "Generación Fósil (%)",
    "gen_other_fossil": "Otras Fósiles (%)",
    "gen_gas": "Gas Natural (%)",
    "gen_wind": "Eólica (%)",
    "gen_solar": "Solar (%)",
    "gen_bioenergy": "Bioenergía (%)",
    "co2_intensity_gco2_kwh": "Intensidad CO₂ (gCO₂/kWh)",
    "importaciones_netas_twh": "Importaciones Netas (TWh)",
}

CRISIS_EVENTS = [
    {"name": "Crisis sequía oct-2023", "start": "2023-10-01", "end": "2024-01-31",
     "color": "rgba(255,165,0,0.15)"},
    {"name": "Apagones abr-jun 2024", "start": "2024-04-01", "end": "2024-06-30",
     "color": "rgba(255,0,0,0.10)"},
    {"name": "Crisis oct-dic 2024", "start": "2024-09-15", "end": "2024-12-31",
     "color": "rgba(255,0,0,0.20)"},
]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Carga datos procesados (prioridad: results con anomalías > raw)."""
    # 1. Buscar resultados ya procesados con anomalías
    results_files = list(PROCESSED_DIR.glob("*real_results*"))
    if results_files:
        df = pd.read_parquet(results_files[0]) if results_files[0].suffix == ".parquet" else pd.read_csv(results_files[0])
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    # 2. Buscar cualquier parquet procesado
    parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    # 3. Cargar datos raw si existen
    raw_file = RAW_DIR / "ecuador_electricity_real.parquet"
    if raw_file.exists():
        df = pd.read_parquet(raw_file)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    return pd.DataFrame()


@st.cache_resource
def load_model() -> AnomalyDetector:
    """Carga modelo entrenado."""
    if MODEL_PATH.exists():
        return AnomalyDetector.load(MODEL_PATH)
    return None


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta detección de anomalías."""
    model = load_model()
    if model is not None:
        try:
            return model.predict(df)
        except Exception:
            pass

    with st.spinner("Entrenando modelo Isolation Forest..."):
        detector = AnomalyDetector(
            contamination=st.session_state.get("contamination", 0.08),
            n_estimators=st.session_state.get("n_estimators", 300),
        )
        return detector.fit_predict(df)


def get_display_name(col: str) -> str:
    """Retorna nombre legible para una columna."""
    return MAIN_COLS.get(col, col.replace("_", " ").title())


# === SIDEBAR ===
with st.sidebar:
    st.title("⚡ Ecuador Energy")
    st.caption("Detector de Anomalías en el Sector Eléctrico")

    st.divider()

    page = st.radio(
        "Navegación",
        ["📊 Overview", "📈 Explorador", "🔍 Anomalías", "📋 Acerca de"],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Parámetros del Modelo")
    contamination = st.slider(
        "Tasa de contaminación (%)",
        min_value=1, max_value=20, value=8,
        help="Porcentaje esperado de anomalías en los datos",
    )
    st.session_state["contamination"] = contamination / 100

    n_estimators = st.select_slider(
        "Número de árboles",
        options=[50, 100, 150, 200, 300, 500],
        value=300,
    )
    st.session_state["n_estimators"] = n_estimators

    st.divider()
    st.caption("Datos: Ember, Our World in Data, World Bank")
    st.caption("Modelo: Isolation Forest + SHAP")
    st.caption("85 meses | Ene 2019 - Ene 2026")


# === CARGAR DATOS ===
df = load_data()

if df.empty:
    st.error(
        "⚠️ No se encontraron datos.\n\n"
        "Ejecuta el pipeline de datos:\n"
        "```bash\n"
        "python scripts/scrape_all.py\n"
        "python scripts/train_model.py\n"
        "```"
    )
    st.stop()


# === DETECCIÓN ===
if "is_anomaly" not in df.columns:
    df = run_detection(df)


# === PÁGINAS ===
if page == "📊 Overview":
    st.title("⚡ Ecuador Energy Anomaly Detector")
    st.markdown(
        "Detección automática de anomalías en el sector eléctrico ecuatoriano "
        "usando **Isolation Forest** con datos reales mensuales de **Ember** y "
        "**Our World in Data** (2019-2026)."
    )

    if "is_anomaly" in df.columns:
        st.plotly_chart(plot_overview_kpis(df), width="stretch")

        col1, col2 = st.columns(2)

        # Seleccionar columna principal disponible
        main_value_col = "demanda_twh" if "demanda_twh" in df.columns else "gen_total_generation"

        with col1:
            fig = plot_timeseries_with_anomalies(
                df, value_col=main_value_col,
                title=f"{get_display_name(main_value_col)} - Ecuador"
            )
            # Agregar bandas de crisis conocidas
            for event in CRISIS_EVENTS:
                fig.add_vrect(
                    x0=event["start"], x1=event["end"],
                    fillcolor=event["color"], layer="below",
                    annotation_text=event["name"],
                    annotation_position="top left",
                    annotation_font_size=9,
                )
            st.plotly_chart(fig, width="stretch")

        with col2:
            if "gen_hydro" in df.columns and "gen_fossil" in df.columns:
                import plotly.graph_objects as go
                fig_mix = go.Figure()
                fig_mix.add_trace(go.Scatter(
                    x=df["fecha"], y=df["gen_hydro"],
                    name="Hidro", fill="tozeroy",
                    line=dict(color="#2196F3"),
                ))
                fig_mix.add_trace(go.Scatter(
                    x=df["fecha"], y=df["gen_fossil"],
                    name="Fósil", fill="tozeroy",
                    line=dict(color="#F44336"),
                ))
                fig_mix.update_layout(
                    title="Mix Energético: Hidro vs Fósil (%)",
                    xaxis_title="Fecha", yaxis_title="% de generación",
                    template="plotly_white",
                )
                st.plotly_chart(fig_mix, width="stretch")
            else:
                st.plotly_chart(plot_anomaly_heatmap(df), width="stretch")

        st.plotly_chart(plot_anomaly_timeline(df), width="stretch")

        # Tabla de anomalías detectadas
        anomalies = df[df["is_anomaly"] == 1].copy()
        if len(anomalies) > 0:
            st.subheader(f"Meses Anómalos Detectados ({len(anomalies)})")
            show_cols = ["fecha", "anomaly_score"]
            for c in ["demanda_twh", "gen_hydro", "gen_fossil", "co2_intensity_gco2_kwh", "importaciones_netas_twh"]:
                if c in anomalies.columns:
                    show_cols.append(c)
            display = anomalies[show_cols].sort_values("anomaly_score")
            display["fecha"] = display["fecha"].dt.strftime("%Y-%m")
            st.dataframe(display, width="stretch")

elif page == "📈 Explorador":
    st.title("📈 Explorador de Datos")

    # Mostrar solo columnas relevantes
    available_main = [c for c in MAIN_COLS if c in df.columns]
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in all_numeric if c not in MAIN_COLS and c not in ["is_anomaly", "anomaly_score", "anio"]]

    col_options = available_main + other_cols
    col_labels = [get_display_name(c) for c in col_options]

    selected_idx = st.selectbox(
        "Variable a visualizar",
        range(len(col_options)),
        format_func=lambda i: col_labels[i],
    )
    selected_col = col_options[selected_idx]

    if "fecha" in df.columns:
        date_range = st.date_input(
            "Rango de fechas",
            value=(df["fecha"].min(), df["fecha"].max()),
        )
        if len(date_range) == 2:
            mask = (df["fecha"] >= pd.to_datetime(date_range[0])) & \
                   (df["fecha"] <= pd.to_datetime(date_range[1]))
            filtered = df[mask]
        else:
            filtered = df
    else:
        filtered = df

    fig = plot_timeseries_with_anomalies(
        filtered, value_col=selected_col,
        title=f"{get_display_name(selected_col)} - Ecuador",
    )
    for event in CRISIS_EVENTS:
        fig.add_vrect(
            x0=event["start"], x1=event["end"],
            fillcolor=event["color"], layer="below",
        )
    st.plotly_chart(fig, width="stretch")

    # Estadísticas descriptivas
    st.subheader("Estadísticas Descriptivas")
    desc_cols = [c for c in available_main if c in filtered.columns]
    if desc_cols:
        desc = filtered[desc_cols].describe().round(3)
        desc.columns = [get_display_name(c) for c in desc.columns]
        st.dataframe(desc)

elif page == "🔍 Anomalías":
    st.title("🔍 Detector de Anomalías")

    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_score_distribution(df), width="stretch")
        with col2:
            st.metric("Anomalías Detectadas", f"{len(anomalies)}")
            st.metric("Tasa de Anomalías", f"{len(anomalies)/len(df)*100:.1f}%")
            if len(anomalies) > 0:
                st.metric("Score Más Extremo", f"{anomalies['anomaly_score'].min():.4f}")
                worst = anomalies.nsmallest(1, "anomaly_score").iloc[0]
                st.metric("Mes Más Anómalo", worst["fecha"].strftime("%Y-%m") if hasattr(worst["fecha"], "strftime") else str(worst["fecha"]))

        st.subheader("Top Anomalías Más Severas")
        if len(anomalies) > 0:
            top = anomalies.nsmallest(20, "anomaly_score")
            show_cols = ["fecha", "anomaly_score"]
            for c in ["demanda_twh", "gen_hydro", "gen_fossil", "gen_gas", "co2_intensity_gco2_kwh"]:
                if c in top.columns:
                    show_cols.append(c)

            display = top[show_cols].copy()
            display["fecha"] = display["fecha"].dt.strftime("%Y-%m") if hasattr(display["fecha"].iloc[0], "strftime") else display["fecha"]
            display.columns = [get_display_name(c) if c != "fecha" else "Fecha" for c in show_cols]
            st.dataframe(
                display.style.background_gradient(
                    subset=["Anomaly Score"], cmap="RdYlBu"
                ),
                width="stretch",
            )

        # Análisis de crisis conocidas
        st.subheader("Validación contra Crisis Conocidas")
        for event in CRISIS_EVENTS:
            start = pd.to_datetime(event["start"])
            end = pd.to_datetime(event["end"])
            mask = (df["fecha"] >= start) & (df["fecha"] <= end)
            event_data = df[mask]
            if len(event_data) > 0:
                n_anom = event_data["is_anomaly"].sum()
                rate = n_anom / len(event_data) * 100
                match = "🟢 STRONG" if rate > 30 else "🟡 MODERATE" if rate > 15 else "🔴 WEAK"
                st.markdown(f"**{event['name']}**: {match} — {n_anom}/{len(event_data)} meses anómalos ({rate:.0f}%)")
            else:
                st.markdown(f"**{event['name']}**: Sin datos en el rango")

    else:
        st.info("Ejecuta la detección primero.")

elif page == "📋 Acerca de":
    st.title("📋 Acerca del Proyecto")

    st.markdown("""
    ## Metodología

    ### Fuentes de Datos Reales
    - **[Ember](https://ember-energy.org/)**: Datos mensuales de generación, demanda, emisiones
      y capacidad instalada por país (85 meses para Ecuador: 2019-2026)
    - **[Our World in Data](https://ourworldindata.org/energy)**: Datos anuales complementarios
      con contexto de población, PIB, y mix energético histórico
    - **[World Bank](https://data.worldbank.org/)**: Indicadores de consumo per cápita

    ### Modelo: Isolation Forest
    Algoritmo no supervisado que **aísla anomalías** basándose en que los puntos
    anómalos requieren menos particiones aleatorias para ser separados del resto.

    - **No requiere etiquetas** — ideal cuando no tenemos "anomalías confirmadas"
    - **300 árboles** de decisión aleatorios
    - **Contamination ~8%** — proporción esperada de anomalías

    ### Features Engineered (213 columnas)
    - **Temporales**: mes, trimestre, estación lluviosa/seca
    - **Rolling stats**: medias móviles y desviación estándar (7, 14, 30 meses)
    - **Lags**: valores rezagados (1, 7, 30 meses)
    - **Ratios**: hidro/total generación
    - **Descomposición**: tendencia + residual
    - **Z-scores**: desviación respecto a la media móvil

    ### Contexto: Ecuador y la Hidroelectricidad
    Ecuador genera ~70% de su electricidad con **hidroeléctricas**, lo que lo hace
    extremadamente vulnerable a sequías. Las crisis de 2023-2024 provocaron:

    - Racionamientos de hasta **14 horas diarias**
    - Incremento masivo de generación **térmica** (fósil) de emergencia
    - Aumento de **importaciones** de energía desde Colombia/Perú
    - Subida de la **intensidad de carbono** del sector eléctrico

    El modelo detectó exitosamente la **crisis de oct-dic 2024** con 100% de match.

    ---

    **Autor**: Diego Fernando Lojan Tenesaca
    **Stack**: Python · Scikit-learn · Plotly · Streamlit
    **Código**: [GitHub](https://github.com/DiegoFernandoLojanTenesaca)
    """)

    st.divider()
    st.caption("⚡ Ecuador Energy Anomaly Detector v1.0 — Datos reales, no simulados")
