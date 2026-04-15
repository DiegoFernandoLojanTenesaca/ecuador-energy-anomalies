"""Detección de anomalías basada en STL Decomposition.

Descompone la serie temporal en tendencia + estacionalidad + residual.
Meses con residuales > threshold*sigma se marcan como anomalías.

Referencia: Cleveland et al. (1990). STL: A Seasonal-Trend Decomposition
Procedure Based on LOESS. Journal of Official Statistics.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


@dataclass
class STLResult:
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    resid_std: float
    anomaly_dates: list
    anomaly_values: dict


class STLDetector:
    """Detector de anomalías basado en STL Decomposition."""

    def __init__(self, period: int = 12, threshold_sigma: float = 2.0, robust: bool = True):
        self.period = period
        self.threshold_sigma = threshold_sigma
        self.robust = robust
        self.result: STLResult | None = None

    def fit_detect(self, series: pd.Series) -> pd.Series:
        """Ajusta STL y detecta anomalías en los residuales.

        Args:
            series: Serie temporal con DatetimeIndex y frecuencia regular.

        Returns:
            Serie binaria (1=anomalía, 0=normal) con mismo índice.
        """
        ts = series.copy().dropna()
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("La serie debe tener DatetimeIndex")

        ts = ts.asfreq("MS")
        ts = ts.interpolate(method="linear")

        if len(ts) < 2 * self.period:
            logger.warning(f"Serie muy corta ({len(ts)} < {2*self.period}). STL puede ser inestable.")

        stl = STL(ts, period=self.period, robust=self.robust)
        decomposition = stl.fit()

        resid = decomposition.resid
        resid_std = resid.std()
        threshold = self.threshold_sigma * resid_std

        is_anomaly = (resid.abs() > threshold).astype(int)

        anomaly_dates = resid[is_anomaly == 1].index.tolist()
        anomaly_values = {
            d.strftime("%Y-%m"): {
                "residual": float(resid[d]),
                "sigma": float(resid[d] / resid_std),
            }
            for d in anomaly_dates
        }

        self.result = STLResult(
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            resid=resid,
            resid_std=resid_std,
            anomaly_dates=anomaly_dates,
            anomaly_values=anomaly_values,
        )

        logger.info(
            f"STL: {is_anomaly.sum()} anomalías de {len(ts)} meses "
            f"(threshold={self.threshold_sigma}sigma={threshold:.2f})"
        )

        return is_anomaly