"""Baselines adicionales para comparación con el enfoque de consenso.

Modelos implementados:
- Local Outlier Factor (LOF)
- One-Class SVM
- ARIMA residuals
- Prophet anomalies
- LSTM Autoencoder (si torch disponible)
- Elliptic Envelope
- DBSCAN-based

Cada baseline expone: fit_predict(X_or_series) -> array de 0/1
"""

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LOFBaseline:
    """Local Outlier Factor."""

    def __init__(self, contamination=0.08, n_neighbors=20):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.fit_predict(X)
        return (preds == -1).astype(int)


class SVMBaseline:
    """One-Class SVM."""

    def __init__(self, nu=0.08):
        self.model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.fit_predict(X)
        return (preds == -1).astype(int)


class EllipticBaseline:
    """Elliptic Envelope (Gaussian assumption)."""

    def __init__(self, contamination=0.08):
        self.model = EllipticEnvelope(contamination=contamination, random_state=42)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        try:
            preds = self.model.fit_predict(X)
            return (preds == -1).astype(int)
        except Exception as e:
            logger.warning(f"EllipticEnvelope failed: {e}")
            return np.zeros(len(X), dtype=int)


class DBSCANBaseline:
    """DBSCAN — puntos noise (-1) son anomalías."""

    def __init__(self, eps=2.0, min_samples=3):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        labels = self.model.fit_predict(X)
        return (labels == -1).astype(int)


class ARIMABaseline:
    """ARIMA residuals — anomalía si residual > threshold*sigma."""

    def __init__(self, order=(1, 1, 1), threshold_sigma=2.0):
        self.order = order
        self.threshold_sigma = threshold_sigma

    def fit_predict_series(self, series: pd.Series) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        try:
            ts = series.dropna().asfreq("MS").interpolate()
            model = ARIMA(ts, order=self.order)
            result = model.fit()
            resid = result.resid
            threshold = self.threshold_sigma * resid.std()
            anomalies = (resid.abs() > threshold).astype(int)
            return anomalies
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")
            return pd.Series(np.zeros(len(series)), index=series.index)


class ProphetBaseline:
    """Prophet — anomalía si valor real fuera del intervalo de predicción."""

    def __init__(self, interval_width=0.95):
        self.interval_width = interval_width

    def fit_predict_series(self, series: pd.Series) -> np.ndarray:
        from prophet import Prophet

        try:
            df = pd.DataFrame({"ds": series.index, "y": series.values})
            m = Prophet(
                interval_width=self.interval_width,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            m.fit(df)
            forecast = m.predict(df)
            anomalies = (
                (df["y"].values < forecast["yhat_lower"].values)
                | (df["y"].values > forecast["yhat_upper"].values)
            ).astype(int)
            return pd.Series(anomalies, index=series.index)
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
            return pd.Series(np.zeros(len(series)), index=series.index)


class LSTMAutoencoder:
    """LSTM Autoencoder — anomalía si reconstruction error > threshold."""

    def __init__(self, seq_len=6, threshold_percentile=92, epochs=50):
        self.seq_len = seq_len
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available, skipping LSTM-AE")
            return np.zeros(len(X), dtype=int)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        n_features = X_sc.shape[1]

        # Crear secuencias
        sequences = []
        for i in range(len(X_sc) - self.seq_len + 1):
            sequences.append(X_sc[i : i + self.seq_len])
        sequences = np.array(sequences)
        X_tensor = torch.FloatTensor(sequences)

        # Modelo
        class AE(nn.Module):
            def __init__(self, n_feat, hidden=16):
                super().__init__()
                self.encoder = nn.LSTM(n_feat, hidden, batch_first=True)
                self.decoder = nn.LSTM(hidden, n_feat, batch_first=True)

            def forward(self, x):
                _, (h, c) = self.encoder(x)
                repeated = h.permute(1, 0, 2).repeat(1, x.size(1), 1)
                out, _ = self.decoder(repeated)
                return out

        model = AE(n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = loss_fn(output, X_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            errors = ((X_tensor - reconstructed) ** 2).mean(dim=(1, 2)).numpy()

        threshold = np.percentile(errors, self.threshold_percentile)
        seq_anomalies = (errors > threshold).astype(int)

        # Mapear de secuencias a puntos originales
        point_anomalies = np.zeros(len(X), dtype=int)
        for i, is_anom in enumerate(seq_anomalies):
            if is_anom:
                point_anomalies[i : i + self.seq_len] = 1

        return point_anomalies
