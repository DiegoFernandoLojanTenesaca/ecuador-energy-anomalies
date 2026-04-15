"""Detección de anomalías por consenso multi-técnica.

Combina Isolation Forest + STL Decomposition + CUSUM.
Un mes es anómalo si >= min_agreement técnicas coinciden.

Este enfoque reduce falsos positivos manteniendo alto recall,
ya que cada técnica captura diferentes aspectos de anomalías.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score, silhouette_score,
)

from .stl_detector import STLDetector
from .cusum_detector import CUSUMDetector

logger = logging.getLogger(__name__)


class ConsensusDetector:
    """Detector multi-técnica con consenso."""

    def __init__(
        self,
        if_params: dict = None,
        stl_threshold_sigma: float = 2.0,
        cusum_threshold_factor: float = 4.0,
        min_agreement: int = 2,
        warmup_months: int = 12,
    ):
        self.if_params = if_params or {
            "n_estimators": 300,
            "contamination": 0.08,
            "random_state": 42,
        }
        self.stl_threshold_sigma = stl_threshold_sigma
        self.cusum_threshold_factor = cusum_threshold_factor
        self.min_agreement = min_agreement
        self.warmup_months = warmup_months

        self.stl_detector = STLDetector(threshold_sigma=stl_threshold_sigma)
        self.cusum_detector = CUSUMDetector(threshold_factor=cusum_threshold_factor)

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Construye features por país para Isolation Forest."""
        feature_cols_out = []
        for col in ["gen_hydro", "gen_fossil", "demanda_twh", "co2_intensity"]:
            if col not in df.columns or df[col].isna().all():
                continue
            df[f"{col}_m6"] = df[col].rolling(6, min_periods=3).mean()
            df[f"{col}_s6"] = df[col].rolling(6, min_periods=3).std()
            df[f"{col}_m12"] = df[col].rolling(12, min_periods=6).mean()
            df[f"{col}_z"] = (df[col] - df[f"{col}_m12"]) / df[f"{col}_s6"].replace(0, np.nan)
            df[f"{col}_pct"] = df[col].pct_change()
            df[f"{col}_l1"] = df[col].shift(1)
            df[f"{col}_l12"] = df[col].shift(12)
            df[f"{col}_yoy"] = (df[col] - df[f"{col}_l12"]) / df[f"{col}_l12"].replace(0, np.nan)

        df["mes"] = df["fecha"].dt.month
        df["trimestre"] = df["fecha"].dt.quarter
        if "gen_hydro" in df.columns and "gen_fossil" in df.columns:
            df["ratio_hf"] = df["gen_hydro"] / df["gen_fossil"].replace(0, np.nan)

        excl = ["fecha", "pais", "_source"]
        feature_cols_out = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in excl
        ]
        return df, feature_cols_out

    def fit_predict_country(
        self, df: pd.DataFrame, hydro_col: str = "gen_hydro"
    ) -> pd.DataFrame:
        """Ejecuta las 3 técnicas en un DataFrame de un solo país.

        Args:
            df: DataFrame con columnas fecha, gen_hydro, gen_fossil, etc.
            hydro_col: Columna de generación hidro para STL y CUSUM.

        Returns:
            DataFrame con columnas adicionales: if_anomaly, stl_anomaly,
            cusum_anomaly, consensus, anomaly_score.
        """
        df = df.sort_values("fecha").copy()
        result = df.copy()

        # --- 1. Isolation Forest ---
        df_feat, feature_cols = self._build_features(df.copy())
        df_trimmed = df_feat.iloc[self.warmup_months:].reset_index(drop=True)

        X = df_trimmed[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X))

        iso = IsolationForest(**self.if_params, n_jobs=-1)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        # Mapear de vuelta al df original
        n_valid = min(len(preds), len(result) - self.warmup_months)
        result["if_anomaly"] = 0
        result.iloc[self.warmup_months:self.warmup_months + n_valid, result.columns.get_loc("if_anomaly")] = (
            (preds[:n_valid] == -1).astype(int)
        )

        result["if_score"] = 0.0
        result.iloc[self.warmup_months:self.warmup_months + n_valid, result.columns.get_loc("if_score")] = (
            scores[:n_valid]
        )

        # --- 2. STL ---
        if hydro_col in df.columns and df[hydro_col].notna().sum() >= 24:
            hydro_ts = df.set_index("fecha")[hydro_col].dropna()
            stl_anomalies = self.stl_detector.fit_detect(hydro_ts)
            stl_dates = set(stl_anomalies[stl_anomalies == 1].index)
            result["stl_anomaly"] = result["fecha"].isin(stl_dates).astype(int)
        else:
            result["stl_anomaly"] = 0

        # --- 3. CUSUM ---
        if hydro_col in df.columns and df[hydro_col].notna().sum() >= 12:
            hydro_ts = df.set_index("fecha")[hydro_col].dropna()
            cusum_anomalies = self.cusum_detector.fit_detect(hydro_ts)
            cusum_dates = set(cusum_anomalies[cusum_anomalies == 1].index)
            result["cusum_anomaly"] = result["fecha"].isin(cusum_dates).astype(int)
        else:
            result["cusum_anomaly"] = 0

        # --- 4. Consenso ---
        votes = result["if_anomaly"] + result["stl_anomaly"] + result["cusum_anomaly"]
        result["consensus"] = (votes >= self.min_agreement).astype(int)
        result["n_techniques"] = votes

        n_cons = result["consensus"].sum()
        logger.info(
            f"Consenso: {n_cons} anomalías "
            f"(IF={result['if_anomaly'].sum()}, "
            f"STL={result['stl_anomaly'].sum()}, "
            f"CUSUM={result['cusum_anomaly'].sum()})"
        )

        return result

    def fit_predict_multi(
        self, df: pd.DataFrame, country_col: str = "pais"
    ) -> pd.DataFrame:
        """Ejecuta las 3 técnicas por país en un dataset multi-país.

        Args:
            df: DataFrame con columna de país.
            country_col: Nombre de la columna de país.

        Returns:
            DataFrame combinado con resultados por país.
        """
        all_results = []
        for country in sorted(df[country_col].unique()):
            logger.info(f"--- Procesando {country} ---")
            sub = df[df[country_col] == country].copy()
            result = self.fit_predict_country(sub)
            result[country_col] = country
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        total = combined["consensus"].sum()
        logger.info(f"Total multi-país: {total} anomalías de consenso en {len(combined)} meses")
        return combined

    @staticmethod
    def compute_metrics(
        results: pd.DataFrame,
        country: str,
        crisis_start: str,
        crisis_end: str,
        country_col: str = "pais",
    ) -> dict:
        """Calcula métricas de clasificación para un país y período de crisis.

        Args:
            results: DataFrame con columnas consensus, if_anomaly, etc.
            country: Nombre del país.
            crisis_start: Fecha inicio del ground truth.
            crisis_end: Fecha fin del ground truth.

        Returns:
            Diccionario con métricas por técnica.
        """
        ec = results[results[country_col] == country].copy()
        y_true = np.zeros(len(ec))
        for i, (_, row) in enumerate(ec.iterrows()):
            if pd.Timestamp(crisis_start) <= row["fecha"] <= pd.Timestamp(crisis_end):
                y_true[i] = 1

        if y_true.sum() == 0:
            return {"error": "No ground truth in range"}

        metrics = {}
        for method, col in [
            ("isolation_forest", "if_anomaly"),
            ("stl", "stl_anomaly"),
            ("cusum", "cusum_anomaly"),
            ("consensus", "consensus"),
        ]:
            y_pred = ec[col].values
            if y_pred.sum() == 0:
                metrics[method] = {"precision": 0, "recall": 0, "f1": 0, "mcc": 0}
                continue

            metrics[method] = {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "n_anomalies": int(y_pred.sum()),
                "crisis_detected": int(sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)),
                "crisis_total": int(y_true.sum()),
            }

        return metrics