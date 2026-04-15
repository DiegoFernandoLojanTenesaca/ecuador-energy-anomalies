"""Modelo Isolation Forest para detección de anomalías."""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


class AnomalyDetector:
    """Detector de anomalías basado en Isolation Forest.

    Diseñado para datos del sector eléctrico ecuatoriano.
    """

    def __init__(
        self,
        contamination: str | float = "auto",
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        """Selecciona features numéricas relevantes, excluyendo metadatos."""
        exclude_patterns = [
            "fecha", "date", "source", "file", "id", "nombre", "name",
            "codigo", "code", "_lag_", "anio", "year",
        ]
        features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if not any(pat in col.lower() for pat in exclude_patterns):
                features.append(col)
        return features

    def fit(self, df: pd.DataFrame, feature_cols: list[str] = None) -> "AnomalyDetector":
        """Entrena el modelo con datos históricos.

        Args:
            df: DataFrame con features.
            feature_cols: Columnas a usar. Si None, auto-selecciona.

        Returns:
            self
        """
        if feature_cols is None:
            feature_cols = self._select_features(df)

        self.feature_names = feature_cols
        X = df[feature_cols].copy()

        # Manejar NaN
        X = X.fillna(X.median())

        # Escalar
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar
        logger.info(f"Entrenando Isolation Forest: {X_scaled.shape} ({len(feature_cols)} features)")
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Estadísticas del entrenamiento
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        n_anomalies = (predictions == -1).sum()

        logger.info(
            f"Entrenamiento completado. "
            f"Anomalías detectadas: {n_anomalies}/{len(df)} "
            f"({n_anomalies/len(df)*100:.1f}%)"
        )
        logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice anomalías en nuevos datos.

        Args:
            df: DataFrame con las mismas features usadas en fit.

        Returns:
            DataFrame original con columnas adicionales:
            - anomaly_score: Score de anomalía (más negativo = más anómalo)
            - is_anomaly: 1 si es anomalía, 0 si no
            - anomaly_label: "Normal" o "Anomalía"
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Ejecutar fit() primero.")

        X = df[self.feature_names].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)

        result = df.copy()
        result["anomaly_score"] = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        result["is_anomaly"] = (predictions == -1).astype(int)
        result["anomaly_label"] = np.where(
            result["is_anomaly"] == 1, "Anomalía", "Normal"
        )

        n_anomalies = result["is_anomaly"].sum()
        logger.info(f"Predicción: {n_anomalies}/{len(df)} anomalías")

        return result

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str] = None) -> pd.DataFrame:
        """Entrena y predice en un solo paso."""
        self.fit(df, feature_cols)
        return self.predict(df)

    def get_top_anomalies(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Retorna las N anomalías más extremas.

        Args:
            df: DataFrame con predicciones (debe tener anomaly_score).
            n: Número de anomalías a retornar.

        Returns:
            DataFrame con las top anomalías ordenadas por severidad.
        """
        if "anomaly_score" not in df.columns:
            df = self.predict(df)

        anomalies = df[df["is_anomaly"] == 1].copy()
        return anomalies.nsmallest(n, "anomaly_score")

    def save(self, path: Path = None) -> Path:
        """Guarda el modelo entrenado."""
        path = path or MODEL_DIR / "anomaly_detector.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "params": {
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
            },
        }
        joblib.dump(artifact, path)
        logger.info(f"Modelo guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path = None) -> "AnomalyDetector":
        """Carga un modelo guardado."""
        path = path or MODEL_DIR / "anomaly_detector.joblib"
        artifact = joblib.load(path)

        detector = cls(**artifact["params"])
        detector.model = artifact["model"]
        detector.scaler = artifact["scaler"]
        detector.feature_names = artifact["feature_names"]
        detector.is_fitted = True

        logger.info(f"Modelo cargado: {path} ({len(detector.feature_names)} features)")
        return detector
