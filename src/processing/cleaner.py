"""Limpieza y normalización de datos del sector eléctrico ecuatoriano."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Festivos de Ecuador (fechas fijas)
ECUADOR_HOLIDAYS_FIXED = [
    (1, 1),   # Año Nuevo
    (5, 1),   # Día del Trabajo
    (5, 24),  # Batalla de Pichincha
    (8, 10),  # Primer Grito de Independencia
    (10, 9),  # Independencia de Guayaquil
    (11, 2),  # Día de los Difuntos
    (11, 3),  # Independencia de Cuenca
    (12, 25), # Navidad
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas: minúsculas, sin acentos, underscores."""
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ñ": "n", "ü": "u",
    }
    new_cols = []
    for col in df.columns:
        c = str(col).strip().lower()
        for old, new in replacements.items():
            c = c.replace(old, new)
        c = c.replace(" ", "_").replace("-", "_").replace(".", "_")
        c = pd.io.common.stringify_path(c) if hasattr(pd.io.common, 'stringify_path') else c
        new_cols.append(c)
    df.columns = new_cols
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    """Intenta parsear la columna de fecha en múltiples formatos."""
    if date_col not in df.columns:
        # Buscar columna que contenga 'fecha' o 'date'
        candidates = [c for c in df.columns if "fecha" in c or "date" in c]
        if candidates:
            date_col = candidates[0]
        else:
            logger.warning("No se encontró columna de fecha")
            return df

    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=fmt)
            logger.info(f"Fecha parseada con formato {fmt}")
            return df
        except (ValueError, TypeError):
            continue

    # Fallback: parseo automático
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
    return df


def clean_numeric_columns(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    """Convierte columnas numéricas, reemplaza comas por puntos."""
    exclude = exclude or []
    for col in df.columns:
        if col in exclude or df[col].dtype == "datetime64[ns]":
            continue
        if df[col].dtype == object:
            # Intentar convertir: quitar comas de miles, cambiar coma decimal
            try:
                cleaned = df[col].astype(str).str.replace(",", ".", regex=False)
                cleaned = pd.to_numeric(cleaned, errors="coerce")
                if cleaned.notna().sum() > len(df) * 0.5:
                    df[col] = cleaned
            except (ValueError, TypeError):
                continue
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Eliminadas {removed} filas duplicadas")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
    """Maneja valores faltantes en columnas numéricas.

    Args:
        df: DataFrame con posibles NaN.
        strategy: "interpolate", "ffill", "drop", o "mean".
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)
    elif strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill(limit=5)
    elif strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "drop":
        df = df.dropna(subset=numeric_cols)

    return df


def is_ecuador_holiday(date: pd.Timestamp) -> bool:
    """Verifica si una fecha es festivo ecuatoriano (solo fechas fijas)."""
    return (date.month, date.day) in ECUADOR_HOLIDAYS_FIXED


def clean_dataframe(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    """Pipeline completo de limpieza.

    Args:
        df: DataFrame crudo.
        date_col: Nombre de la columna de fecha.

    Returns:
        DataFrame limpio.
    """
    logger.info(f"Limpiando DataFrame: {df.shape}")

    df = normalize_columns(df)
    df = remove_duplicates(df)
    df = parse_dates(df, date_col)
    df = clean_numeric_columns(df, exclude=[date_col])
    df = handle_missing(df, strategy="interpolate")

    logger.info(f"DataFrame limpio: {df.shape}")
    return df
