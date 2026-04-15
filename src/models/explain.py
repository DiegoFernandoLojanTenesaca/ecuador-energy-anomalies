"""Explicabilidad de anomalías usando SHAP."""

import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """Explica POR QUÉ un punto fue clasificado como anomalía."""

    def __init__(self, detector):
        """
        Args:
            detector: Instancia de AnomalyDetector ya entrenada.
        """
        self.detector = detector
        self.explainer = None
        self.shap_values = None

    def compute_shap(self, df: pd.DataFrame, max_samples: int = 500) -> np.ndarray:
        """Calcula SHAP values para el dataset.

        Args:
            df: DataFrame con features.
            max_samples: Máximo de muestras para background data.

        Returns:
            Array de SHAP values.
        """
        X = df[self.detector.feature_names].copy()
        X = X.fillna(X.median())
        X_scaled = self.detector.scaler.transform(X)

        # Usar submuestra como background para eficiencia
        n_bg = min(max_samples, len(X_scaled))
        bg_indices = np.random.choice(len(X_scaled), n_bg, replace=False)
        background = X_scaled[bg_indices]

        logger.info(f"Calculando SHAP values ({len(X_scaled)} muestras, {n_bg} background)...")
        self.explainer = shap.TreeExplainer(
            self.detector.model,
            data=background,
            feature_perturbation="interventional",
        )
        self.shap_values = self.explainer.shap_values(X_scaled)
        logger.info("SHAP values calculados")

        return self.shap_values

    def explain_anomaly(self, df: pd.DataFrame, idx: int) -> dict:
        """Explica una anomalía específica.

        Args:
            df: DataFrame con predicciones.
            idx: Índice de la fila a explicar.

        Returns:
            Diccionario con las features más importantes y su contribución.
        """
        if self.shap_values is None:
            self.compute_shap(df)

        row_shap = self.shap_values[idx]
        feature_importance = pd.Series(
            row_shap, index=self.detector.feature_names
        ).sort_values()

        # Las features con SHAP más negativo contribuyen más a la anomalía
        top_contributors = feature_importance.head(5)

        explanation = {
            "idx": idx,
            "anomaly_score": df.iloc[idx].get("anomaly_score", None),
            "top_contributors": {
                name: {
                    "shap_value": float(val),
                    "actual_value": float(df.iloc[idx].get(name, np.nan)),
                    "direction": "bajo" if val < 0 else "alto",
                }
                for name, val in top_contributors.items()
            },
            "summary": self._generate_summary(top_contributors, df.iloc[idx]),
        }

        return explanation

    def _generate_summary(self, contributors: pd.Series, row: pd.Series) -> str:
        """Genera resumen legible de la explicación."""
        parts = []
        for name, shap_val in contributors.items():
            direction = "inusualmente bajo" if shap_val < 0 else "inusualmente alto"
            val = row.get(name, "N/A")
            parts.append(f"- {name}: {val:.2f} ({direction})")

        return "Factores principales:\n" + "\n".join(parts)

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importancia global de features basada en SHAP.

        Returns:
            DataFrame con features ordenadas por importancia.
        """
        if self.shap_values is None:
            raise RuntimeError("Ejecutar compute_shap() primero")

        importance = pd.DataFrame({
            "feature": self.detector.feature_names,
            "mean_abs_shap": np.abs(self.shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return importance
