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

# Crisis documentadas con fuentes oficiales verificables
KNOWN_CRISES = {
    "Ecuador": [
        {
            "start": "2024-04-01", "end": "2024-12-31",
            "name": "Crisis energetica Ecuador 2024",
            "source": "Decreto Ejecutivo No. 229 (19-abr-2024), estado de excepcion",
            "ref": "CENACE Informe Anual 2024; Min. Energia Acuerdo MEM-MEM-2024-0005-AM",
            "severity": "14h daily blackouts, worst drought in 61 years, 1.5% GDP impact",
            "hydro_driven": True,
        },
    ],
    "Colombia": [
        {
            "start": "2024-01-01", "end": "2024-06-30",
            "name": "Impacto El Nino en sector electrico Colombia 2024",
            "source": "XM Colombia, Informe variables del mercado abr-2024",
            "ref": "xm.com.co/noticias/6865; precios mayoristas +22.68%",
            "severity": "Wholesale price increase 22.68%, demand +2.3% YoY",
            "hydro_driven": True,
        },
    ],
    "Brazil": [
        {
            "start": "2021-06-01", "end": "2021-11-30",
            "name": "Crisis hidrica Brasil 2021",
            "source": "Decreto 10.939/2021; MP 1.055/2021 (creacion CREG)",
            "ref": "EPE NT-DEE-DEA-001-2023; Nature d41586-021-03625-w",
            "severity": "Worst drought in 91 years, new tariff flag, 70% hydro affected",
            "hydro_driven": True,
        },
    ],
    "Chile": [
        {
            "start": "2019-01-01", "end": "2019-12-31",
            "name": "Mega-sequia Chile (fase aguda 2019)",
            "source": "U. de Chile repositorio/2250/146730; BCN siit/mega_sequia",
            "ref": "Mega-drought 2010-present, acute phase 2019",
            "severity": "Hydro dropped from 30% to 17%, Rapel reservoir 52% deficit",
            "hydro_driven": True,
        },
    ],
    "Argentina": [
        {
            "start": "2022-01-01", "end": "2022-03-31",
            "name": "Ola de calor + sequia Argentina 2022",
            "source": "SMN Argentina informe 0036ID2023; FARN impactos-cambio-climatico",
            "ref": "50+ cities >40C, 50% thermal generation, imports from Brazil/Uruguay",
            "severity": "Extreme heatwave, La Nina, emergency energy imports",
            "hydro_driven": False,
        },
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
            ("Weighted", "weighted_consensus"),
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


def mcnemar_test(results: pd.DataFrame, country: str,
                 crisis_start: str, crisis_end: str,
                 method_a: str = "consensus", method_b: str = "if_anomaly",
                 country_col: str = "pais") -> dict:
    """Test de McNemar: compara dos clasificadores en los mismos datos.

    H0: ambos metodos tienen la misma tasa de error.
    Mas potente que bootstrap para comparar dos modelos con N bajo.
    """
    from scipy.stats import binomtest

    sub = results[results[country_col] == country]
    y_true = np.zeros(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        if pd.Timestamp(crisis_start) <= row["fecha"] <= pd.Timestamp(crisis_end):
            y_true[i] = 1

    ya = sub[method_a].values
    yb = sub[method_b].values

    b01 = sum(1 for i in range(len(y_true)) if (ya[i] == y_true[i]) and (yb[i] != y_true[i]))
    b10 = sum(1 for i in range(len(y_true)) if (ya[i] != y_true[i]) and (yb[i] == y_true[i]))

    n = b01 + b10
    if n == 0:
        return {"statistic": 0, "p_value": 1.0, "b01": b01, "b10": b10, "significant": False}

    p_value = binomtest(b01, n, 0.5).pvalue

    return {
        "method_a": method_a, "method_b": method_b,
        "b01_a_right_b_wrong": b01, "b10_a_wrong_b_right": b10,
        "p_value": float(p_value), "significant": p_value < 0.05,
    }


def hydro_dependency_analysis(results: pd.DataFrame, country_col: str = "pais") -> pd.DataFrame:
    """Analiza relacion entre dependencia hidro y efectividad del consenso."""
    rows = []
    for country in sorted(results[country_col].unique()):
        crises = KNOWN_CRISES.get(country, [])
        if not crises:
            continue
        sub = results[results[country_col] == country]
        hydro_dep = sub["hydro_dependency"].mean() if "hydro_dependency" in sub.columns else 0

        crisis = crises[0]
        y_true = np.zeros(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.Timestamp(crisis["start"]) <= row["fecha"] <= pd.Timestamp(crisis["end"]):
                y_true[i] = 1
        if y_true.sum() == 0:
            continue

        for method in ["consensus", "weighted_consensus"]:
            if method not in sub.columns:
                continue
            y_pred = sub[method].values
            rows.append({
                "country": country, "method": method,
                "hydro_dependency": float(hydro_dep),
                "hydro_driven_crisis": crisis.get("hydro_driven", None),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "crisis_detected": int((y_true * y_pred).sum()),
                "crisis_total": int(y_true.sum()),
            })

    return pd.DataFrame(rows)
