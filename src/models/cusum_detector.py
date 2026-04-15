"""Detección de cambios estructurales con CUSUM (Cumulative Sum).

Detecta cambios sostenidos en la media de una serie temporal.
Cuando la desviación acumulada excede un umbral, se señala alarma.

Referencia: Page, E.S. (1954). Continuous Inspection Schemes. Biometrika.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CUSUMResult:
    cusum_pos: np.ndarray
    cusum_neg: np.ndarray
    threshold: float
    allowance: float
    mean: float
    alarm_dates: list


class CUSUMDetector:
    """Detector de cambios estructurales con CUSUM bilateral."""

    def __init__(self, threshold_factor: float = 4.0, allowance_factor: float = 0.5):
        """
        Args:
            threshold_factor: Multiplicador de sigma para el umbral h.
            allowance_factor: Multiplicador de sigma para la tolerancia k.
        """
        self.threshold_factor = threshold_factor
        self.allowance_factor = allowance_factor
        self.result: CUSUMResult | None = None

    def fit_detect(self, series: pd.Series) -> pd.Series:
        """Aplica CUSUM bilateral y detecta alarmas.

        Args:
            series: Serie temporal con DatetimeIndex.

        Returns:
            Serie binaria (1=alarma, 0=normal) con mismo índice.
        """
        ts = series.copy().dropna()
        values = ts.values.astype(float)
        n = len(values)

        mu = values.mean()
        sigma = values.std()
        k = self.allowance_factor * sigma  # tolerancia
        h = self.threshold_factor * sigma  # umbral

        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        alarms = np.zeros(n, dtype=int)

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (values[i] - mu) - k)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + (values[i] - mu) + k)
            if cusum_pos[i] > h or cusum_neg[i] < -h:
                alarms[i] = 1

        alarm_dates = ts.index[alarms == 1].tolist()

        self.result = CUSUMResult(
            cusum_pos=cusum_pos,
            cusum_neg=cusum_neg,
            threshold=h,
            allowance=k,
            mean=mu,
            alarm_dates=alarm_dates,
        )

        logger.info(
            f"CUSUM: {alarms.sum()} alarmas de {n} meses "
            f"(h={h:.2f}, k={k:.2f}, mu={mu:.2f})"
        )

        return pd.Series(alarms, index=ts.index, name="cusum_anomaly")