"""Validación cruzada entre países y análisis estadístico formal.

- Leave-One-Country-Out: entrenar en 7 países, evaluar en 1
- Bootstrap confidence intervals para todas las métricas
- Sensitivity analysis formal con parameter sweeps
- Comparación de consenso vs individuales en TODOS los países
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

from .consensus import ConsensusDetector

logger = logging.getLogger(__name__)

# Crisis documentadas por país (ground truth parcial)
KNOWN_CRISES = {
    "Ecuador": [
        {"start": "2024-10-01", "end": "2024-12-31", "name": "Crisis severa oct-dic 2024"},
    ],
    "Colombia": [
        {"start": "2024-01-01", "end": "2024-06-30", "name": "El Nino impact 2024"},
    ],
    "Brazil": [
        {"start": "2021-06-01", "end": "2021-10-31", "name": "Crisis hidrica 2021"},
    ],
    "Chile": [
        {"start": "2021-01-01", "end": "2021-06-30", "name": "Mega-sequia central 2021"},
    ],
    "Argentina": [
        {"start": "2022-01-01", "end": "2022-03-31", "name": "Ola de calor + sequia 2022"},
    ],
}


def consensus_vs_individual_all_countries(
    results: pd.DataFrame,
    country_col: str = "pais",
) -> pd.DataFrame:
    """Compara consenso vs técnicas individuales en TODOS los países.

    Prueba la hipótesis: el consenso consistentemente supera
    a cada técnica individual, no solo en Ecuador.
    """
    rows = []
    for country in sorted(results[country_col].unique()):
        crises = KNOWN_CRISES.get(country, [])
        if not crises:
            continue

        sub = results[results[country_col] == country]
        crisis = crises[0]

        y_true = np.zeros(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.Timestamp(crisis["start"]) <= row["fecha"] <= pd.Timestamp(crisis["end"]):
                y_true[i] = 1

        if y_true.sum() == 0:
            continue

        for method, col in [
            ("IF", "if_anomaly"),
            ("STL", "stl_anomaly"),
            ("CUSUM", "cusum_anomaly"),
            ("Consensus", "consensus"),
        ]:
            if col not in sub.columns:
                continue
            y_pred = sub[col].values
            rows.append({
                "country": country,
                "method": method,
                "crisis": crisis["name"],
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "n_anomalies": int(y_pred.sum()),
                "n_crisis": int(y_true.sum()),
                "crisis_detected": int((y_true * y_pred).sum()),
            })

    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(
    results: pd.DataFrame,
    country: str,
    crisis_start: str,
    crisis_end: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    country_col: str = "pais",
) -> dict:
    """Calcula intervalos de confianza por bootstrap para las métricas.

    Resamplea con reemplazo y calcula métricas en cada iteración
    para obtener distribuciones empíricas de P, R, F1, MCC.
    """
    sub = results[results[country_col] == country].copy()
    y_true = np.zeros(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        if pd.Timestamp(crisis_start) <= row["fecha"] <= pd.Timestamp(crisis_end):
            y_true[i] = 1

    alpha = 1 - confidence
    ci = {}

    for method, col in [
        ("IF", "if_anomaly"),
        ("STL", "stl_anomaly"),
        ("CUSUM", "cusum_anomaly"),
        ("Consensus", "consensus"),
    ]:
        if col not in sub.columns:
            continue

        y_pred = sub[col].values
        boot_metrics = {"precision": [], "recall": [], "f1": [], "mcc": []}

        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            yt = y_true[idx]
            yp = y_pred[idx]

            if yt.sum() == 0 or yp.sum() == 0:
                continue

            boot_metrics["precision"].append(precision_score(yt, yp, zero_division=0))
            boot_metrics["recall"].append(recall_score(yt, yp, zero_division=0))
            boot_metrics["f1"].append(f1_score(yt, yp, zero_division=0))
            boot_metrics["mcc"].append(matthews_corrcoef(yt, yp))

        ci[method] = {}
        for metric, values in boot_metrics.items():
            if len(values) < 10:
                ci[method][metric] = {"mean": 0, "ci_low": 0, "ci_high": 0}
                continue
            values = np.array(values)
            ci[method][metric] = {
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 100 * alpha / 2)),
                "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
                "std": float(np.std(values)),
            }

    return ci


def sensitivity_analysis(
    df: pd.DataFrame,
    country: str,
    crisis_start: str,
    crisis_end: str,
    country_col: str = "pais",
) -> pd.DataFrame:
    """Análisis de sensibilidad: cómo cambian las métricas
    al variar contamination, stl_threshold y cusum_threshold.
    """
    rows = []

    param_grid = [
        {"contamination": c, "stl_sigma": s, "cusum_factor": f}
        for c in [0.05, 0.08, 0.10, 0.12]
        for s in [1.5, 2.0, 2.5]
        for f in [3.0, 4.0, 5.0]
    ]

    sub = df[df[country_col] == country].copy()

    for params in param_grid:
        detector = ConsensusDetector(
            if_params={"n_estimators": 200, "contamination": params["contamination"], "random_state": 42},
            stl_threshold_sigma=params["stl_sigma"],
            cusum_threshold_factor=params["cusum_factor"],
            min_agreement=2,
            warmup_months=12,
        )
        result = detector.fit_predict_country(sub)

        y_true = np.zeros(len(result))
        for i, (_, row) in enumerate(result.iterrows()):
            if pd.Timestamp(crisis_start) <= row["fecha"] <= pd.Timestamp(crisis_end):
                y_true[i] = 1

        y_pred = result["consensus"].values
        if y_true.sum() > 0:
            rows.append({
                **params,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "n_anomalies": int(y_pred.sum()),
            })

    return pd.DataFrame(rows)
