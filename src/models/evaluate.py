"""Evaluación del modelo de detección de anomalías.

Sin labels, usamos métricas de validación no supervisada y
correlación con eventos conocidos.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Eventos conocidos en el sector eléctrico ecuatoriano
# que deberían correlacionar con anomalías detectadas
KNOWN_EVENTS = [
    {
        "name": "Crisis energética por sequía",
        "start": "2023-10-01",
        "end": "2024-01-31",
        "description": "Reducción drástica de generación hidroeléctrica por sequía",
    },
    {
        "name": "Apagones programados 2024",
        "start": "2024-04-01",
        "end": "2024-06-30",
        "description": "Cortes de energía programados por déficit de generación",
    },
    {
        "name": "Crisis energética oct 2024",
        "start": "2024-09-15",
        "end": "2024-12-31",
        "description": "Segunda ola de crisis energética, racionamientos de hasta 14h",
    },
]


def evaluate_score_distribution(df: pd.DataFrame) -> dict:
    """Analiza la distribución de anomaly scores.

    Args:
        df: DataFrame con columna 'anomaly_score'.

    Returns:
        Diccionario con estadísticas de distribución.
    """
    scores = df["anomaly_score"]
    anomalies = df[df["is_anomaly"] == 1]

    stats = {
        "total_points": len(df),
        "n_anomalies": len(anomalies),
        "anomaly_rate": len(anomalies) / len(df) * 100,
        "score_mean": scores.mean(),
        "score_std": scores.std(),
        "score_min": scores.min(),
        "score_max": scores.max(),
        "score_q25": scores.quantile(0.25),
        "score_median": scores.median(),
        "score_q75": scores.quantile(0.75),
        "anomaly_score_mean": anomalies["anomaly_score"].mean() if len(anomalies) > 0 else None,
    }

    logger.info(
        f"Distribución: {stats['n_anomalies']}/{stats['total_points']} anomalías "
        f"({stats['anomaly_rate']:.1f}%)"
    )

    return stats


def validate_against_known_events(
    df: pd.DataFrame,
    date_col: str = "fecha",
) -> list[dict]:
    """Valida si las anomalías detectadas coinciden con eventos conocidos.

    Args:
        df: DataFrame con predicciones y columna de fecha.
        date_col: Columna de fecha.

    Returns:
        Lista de resultados de validación por evento.
    """
    if date_col not in df.columns:
        logger.warning(f"Columna {date_col} no encontrada para validación")
        return []

    results = []

    for event in KNOWN_EVENTS:
        start = pd.to_datetime(event["start"])
        end = pd.to_datetime(event["end"])

        mask = (df[date_col] >= start) & (df[date_col] <= end)
        event_data = df[mask]

        if len(event_data) == 0:
            results.append({
                **event,
                "data_points": 0,
                "anomalies_found": 0,
                "anomaly_rate": 0,
                "avg_score": None,
                "match": "NO DATA",
            })
            continue

        n_anomalies = event_data["is_anomaly"].sum()
        anomaly_rate = n_anomalies / len(event_data) * 100

        # Una tasa de anomalías > 20% durante un evento conocido es buena señal
        match = "STRONG" if anomaly_rate > 30 else "MODERATE" if anomaly_rate > 15 else "WEAK"

        results.append({
            **event,
            "data_points": len(event_data),
            "anomalies_found": int(n_anomalies),
            "anomaly_rate": anomaly_rate,
            "avg_score": event_data["anomaly_score"].mean(),
            "match": match,
        })

        logger.info(
            f"Evento '{event['name']}': {n_anomalies}/{len(event_data)} "
            f"anomalías ({anomaly_rate:.1f}%) - Match: {match}"
        )

    return results


def evaluate_temporal_clustering(df: pd.DataFrame, date_col: str = "fecha") -> dict:
    """Evalúa si las anomalías se agrupan temporalmente.

    Anomalías que se agrupan en períodos cortos son más creíbles
    que anomalías dispersas aleatoriamente.
    """
    if date_col not in df.columns or "is_anomaly" not in df.columns:
        return {}

    anomalies = df[df["is_anomaly"] == 1].copy()
    if len(anomalies) < 2:
        return {"clustering": "INSUFFICIENT_DATA"}

    anomalies = anomalies.sort_values(date_col)
    time_diffs = anomalies[date_col].diff().dt.days.dropna()

    return {
        "n_anomalies": len(anomalies),
        "mean_days_between": time_diffs.mean(),
        "median_days_between": time_diffs.median(),
        "max_consecutive_days": (time_diffs <= 1).astype(int).groupby(
            (time_diffs > 1).cumsum()
        ).sum().max() if len(time_diffs) > 0 else 0,
        "clustering": "GOOD" if time_diffs.median() < 7 else "MODERATE" if time_diffs.median() < 30 else "SPARSE",
    }


def full_evaluation(df: pd.DataFrame, date_col: str = "fecha") -> dict:
    """Evaluación completa del modelo."""
    logger.info("=== Evaluación completa ===")

    return {
        "score_distribution": evaluate_score_distribution(df),
        "known_events": validate_against_known_events(df, date_col),
        "temporal_clustering": evaluate_temporal_clustering(df, date_col),
    }
