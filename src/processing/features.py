"""Feature engineering para detección de anomalías en consumo eléctrico."""

import logging

import numpy as np
import pandas as pd

from .cleaner import is_ecuador_holiday

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    """Agrega features temporales basadas en la columna de fecha.

    Features creadas:
    - hora, dia_semana, dia_mes, mes, trimestre, anio
    - es_fin_semana, es_festivo
    - estacion (seca/lluviosa para Ecuador)
    """
    if date_col not in df.columns:
        logger.warning(f"Columna {date_col} no encontrada")
        return df

    dt = df[date_col]

    # Features básicas
    if hasattr(dt.dt, "hour"):
        df["hora"] = dt.dt.hour
    df["dia_semana"] = dt.dt.dayofweek  # 0=Lunes, 6=Domingo
    df["dia_mes"] = dt.dt.day
    df["mes"] = dt.dt.month
    df["trimestre"] = dt.dt.quarter
    df["anio"] = dt.dt.year
    df["dia_anio"] = dt.dt.dayofyear

    # Indicadores
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_festivo"] = dt.apply(is_ecuador_holiday).astype(int)

    # Estación en Ecuador (Costa y Sierra tienen patrones diferentes)
    # Costa: lluviosa dic-may, seca jun-nov
    # Sierra: lluviosa oct-may, seca jun-sep
    # Usamos una aproximación general
    df["es_estacion_lluviosa"] = df["mes"].isin([10, 11, 12, 1, 2, 3, 4, 5]).astype(int)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: list[int] = None,
) -> pd.DataFrame:
    """Agrega estadísticas de ventanas móviles.

    Args:
        df: DataFrame con datos temporales ordenados.
        value_cols: Columnas numéricas sobre las que calcular.
        windows: Tamaños de ventana (en número de filas).
    """
    windows = windows or [7, 14, 30]

    for col in value_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_media_{w}d"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_std_{w}d"] = df[col].rolling(w, min_periods=1).std()
            df[f"{col}_min_{w}d"] = df[col].rolling(w, min_periods=1).min()
            df[f"{col}_max_{w}d"] = df[col].rolling(w, min_periods=1).max()

        # Cambio porcentual
        df[f"{col}_pct_change"] = df[col].pct_change()

        # Z-score respecto a ventana de 30 días
        media_30 = df[col].rolling(30, min_periods=1).mean()
        std_30 = df[col].rolling(30, min_periods=1).std()
        df[f"{col}_zscore_30d"] = (df[col] - media_30) / std_30.replace(0, np.nan)

    return df


def add_lag_features(
    df: pd.DataFrame,
    value_cols: list[str],
    lags: list[int] = None,
) -> pd.DataFrame:
    """Agrega features de rezago (lag).

    Args:
        df: DataFrame ordenado temporalmente.
        value_cols: Columnas sobre las que calcular lags.
        lags: Períodos de rezago.
    """
    lags = lags or [1, 7, 30]

    for col in value_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ratios específicos del sector eléctrico ecuatoriano.

    Ratios calculados (si las columnas existen):
    - Generación hidro / generación total
    - Generación térmica / generación total
    - Demanda / capacidad instalada (factor de carga)
    - Pérdidas / energía total
    """
    # Ratio hidro vs total
    hidro_cols = [c for c in df.columns if "hidro" in c or "hydro" in c]
    total_cols = [c for c in df.columns if "total" in c and "gener" in c.lower()]

    if hidro_cols and total_cols:
        df["ratio_hidro_total"] = df[hidro_cols[0]] / df[total_cols[0]].replace(0, np.nan)

    # Ratio térmico vs total
    term_cols = [c for c in df.columns if "term" in c or "therm" in c]
    if term_cols and total_cols:
        df["ratio_termico_total"] = df[term_cols[0]] / df[total_cols[0]].replace(0, np.nan)

    return df


def add_decomposition_features(
    df: pd.DataFrame,
    value_col: str,
    period: int = 30,
) -> pd.DataFrame:
    """Descomposición simple de serie temporal (trend + residual).

    Usa media móvil como proxy del trend. El residual es la desviación
    del trend, que es donde se esconden las anomalías.
    """
    if value_col not in df.columns:
        return df

    # Trend = media móvil
    df[f"{value_col}_trend"] = df[value_col].rolling(period, min_periods=1, center=True).mean()

    # Residual = valor - trend
    df[f"{value_col}_residual"] = df[value_col] - df[f"{value_col}_trend"]

    # Residual normalizado
    residual_std = df[f"{value_col}_residual"].std()
    if residual_std > 0:
        df[f"{value_col}_residual_norm"] = df[f"{value_col}_residual"] / residual_std

    return df


def engineer_features(
    df: pd.DataFrame,
    date_col: str = "fecha",
    value_cols: list[str] = None,
) -> pd.DataFrame:
    """Pipeline completo de feature engineering.

    Args:
        df: DataFrame limpio con datos temporales.
        date_col: Columna de fecha.
        value_cols: Columnas numéricas principales. Si None, auto-detecta.

    Returns:
        DataFrame con features adicionales.
    """
    logger.info(f"Feature engineering: {df.shape}")

    # Auto-detectar columnas numéricas si no se especifican
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir columnas que parecen IDs o años
        value_cols = [
            c for c in value_cols
            if not any(x in c for x in ["id", "codigo", "code", "anio", "year"])
        ]

    # Ordenar por fecha
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    df = add_temporal_features(df, date_col)
    df = add_rolling_features(df, value_cols)
    df = add_lag_features(df, value_cols)
    df = add_ratio_features(df)

    # Descomposición para las columnas principales (max 3)
    for col in value_cols[:3]:
        df = add_decomposition_features(df, col)

    # Reemplazar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Features generados: {df.shape} ({len(df.columns)} columnas)")
    return df
