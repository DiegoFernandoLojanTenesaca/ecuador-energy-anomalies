"""Comparacion completa: consenso vs 7 baselines en 8 paises.

Genera:
- data/processed/baselines_comparison.json
- data/processed/cross_country_validation.json
- data/processed/confidence_intervals.json
- data/processed/sensitivity_analysis.csv
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.consensus import ConsensusDetector
from src.models.baselines import (
    LOFBaseline, SVMBaseline, EllipticBaseline, DBSCANBaseline,
    ARIMABaseline, ProphetBaseline, LSTMAutoencoder,
)
from src.models.cross_validation import (
    consensus_vs_individual_all_countries,
    bootstrap_confidence_intervals,
    sensitivity_analysis,
    KNOWN_CRISES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def build_features_country(df_country):
    """Features para un pais."""
    s = df_country.sort_values("fecha").copy()
    for col in ["gen_hydro", "gen_fossil", "demanda_twh", "co2_intensity"]:
        if col not in s.columns or s[col].isna().all():
            continue
        s[f"{col}_m6"] = s[col].rolling(6, min_periods=3).mean()
        s[f"{col}_s6"] = s[col].rolling(6, min_periods=3).std()
        s[f"{col}_z"] = (s[col] - s[f"{col}_m6"]) / s[f"{col}_s6"].replace(0, np.nan)
        s[f"{col}_pct"] = s[col].pct_change()
        s[f"{col}_l12"] = s[col].shift(12)
    s["mes"] = s["fecha"].dt.month
    if "gen_hydro" in s.columns and "gen_fossil" in s.columns:
        s["ratio_hf"] = s["gen_hydro"] / s["gen_fossil"].replace(0, np.nan)
    return s.iloc[12:]  # warmup


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW / "latam_electricity.csv", parse_dates=["fecha"])
    df.columns = [c.replace(",", "_").replace(" ", "_") for c in df.columns]

    # ========================================
    # 1. CONSENSO (ya implementado)
    # ========================================
    logger.info("=== 1. Ejecutando consenso multi-tecnica ===")
    detector = ConsensusDetector(
        if_params={"n_estimators": 300, "contamination": 0.08, "random_state": 42},
        min_agreement=2, warmup_months=12,
    )
    consensus_results = detector.fit_predict_multi(df, country_col="pais")

    # ========================================
    # 2. BASELINES por pais (Ecuador)
    # ========================================
    logger.info("=== 2. Baselines para Ecuador ===")
    ec_raw = df[df["pais"] == "Ecuador"].copy()
    ec_feat = build_features_country(ec_raw)
    feature_cols = [c for c in ec_feat.select_dtypes(include=[np.number]).columns if c != "fecha"]
    X = np.nan_to_num(StandardScaler().fit_transform(
        ec_feat[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    ))

    # Ground truth Ecuador
    y_true = np.zeros(len(ec_feat))
    for i, (_, row) in enumerate(ec_feat.iterrows()):
        if pd.Timestamp("2024-10-01") <= row["fecha"] <= pd.Timestamp("2024-12-31"):
            y_true[i] = 1

    hydro_ts = ec_raw.set_index("fecha")["gen_hydro"].dropna().asfreq("MS").interpolate()

    baselines_results = {}

    # 2a. LOF
    lof = LOFBaseline(contamination=0.08)
    lof_pred = lof.fit_predict(X)
    baselines_results["LOF"] = lof_pred

    # 2b. One-Class SVM
    svm = SVMBaseline(nu=0.08)
    svm_pred = svm.fit_predict(X)
    baselines_results["OC-SVM"] = svm_pred

    # 2c. Elliptic Envelope
    ee = EllipticBaseline(contamination=0.08)
    ee_pred = ee.fit_predict(X)
    baselines_results["Elliptic"] = ee_pred

    # 2d. DBSCAN
    dbscan = DBSCANBaseline(eps=3.0, min_samples=3)
    dbscan_pred = dbscan.fit_predict(X)
    baselines_results["DBSCAN"] = dbscan_pred

    # 2e. ARIMA
    arima = ARIMABaseline(order=(1, 1, 1), threshold_sigma=2.0)
    arima_pred = arima.fit_predict_series(hydro_ts)
    arima_dates = set(arima_pred[arima_pred == 1].index)
    arima_mapped = ec_feat["fecha"].isin(arima_dates).astype(int).values
    baselines_results["ARIMA"] = arima_mapped

    # 2f. Prophet
    logger.info("  Prophet...")
    prophet = ProphetBaseline(interval_width=0.95)
    prophet_pred = prophet.fit_predict_series(hydro_ts)
    prophet_dates = set(prophet_pred[prophet_pred == 1].index)
    prophet_mapped = ec_feat["fecha"].isin(prophet_dates).astype(int).values
    baselines_results["Prophet"] = prophet_mapped

    # 2g. LSTM Autoencoder
    logger.info("  LSTM-AE...")
    lstm = LSTMAutoencoder(seq_len=6, threshold_percentile=92, epochs=50)
    lstm_pred = lstm.fit_predict(X)
    baselines_results["LSTM-AE"] = lstm_pred

    # 2h. Consensus (from consensus_results)
    ec_cons = consensus_results[consensus_results["pais"] == "Ecuador"].sort_values("fecha")
    # Align with ec_feat
    consensus_mapped = ec_feat["fecha"].isin(
        ec_cons[ec_cons["consensus"] == 1]["fecha"]
    ).astype(int).values
    baselines_results["Consensus"] = consensus_mapped

    # IF individual
    if_mapped = ec_feat["fecha"].isin(
        ec_cons[ec_cons["if_anomaly"] == 1]["fecha"]
    ).astype(int).values
    baselines_results["IF"] = if_mapped

    # Calcular metricas para cada baseline
    logger.info("\n=== COMPARACION DE BASELINES (Ecuador, GT: oct-dic 2024) ===")
    comparison = {}
    for name, preds in baselines_results.items():
        p = preds[:len(y_true)]  # Align
        if len(p) < len(y_true):
            p = np.concatenate([p, np.zeros(len(y_true) - len(p), dtype=int)])
        comparison[name] = {
            "precision": float(precision_score(y_true, p, zero_division=0)),
            "recall": float(recall_score(y_true, p, zero_division=0)),
            "f1": float(f1_score(y_true, p, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, p)),
            "n_anomalies": int(p.sum()),
        }
        m = comparison[name]
        logger.info(f"  {name:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} MCC={m['mcc']:.3f} n={m['n_anomalies']}")

    with open(OUT / "baselines_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ========================================
    # 3. CROSS-COUNTRY VALIDATION
    # ========================================
    logger.info("\n=== 3. Validacion cross-country ===")
    cross = consensus_vs_individual_all_countries(consensus_results)
    cross.to_csv(OUT / "cross_country_validation.csv", index=False)
    logger.info(f"  {len(cross)} filas guardadas")

    for country in cross["country"].unique():
        sub = cross[cross["country"] == country]
        cons = sub[sub["method"] == "Consensus"]
        if len(cons) > 0:
            c = cons.iloc[0]
            logger.info(f"  {country:12s} Consenso F1={c['f1']:.3f} MCC={c['mcc']:.3f} crisis={c['crisis_detected']}/{c['n_crisis']}")

    # ========================================
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================
    logger.info("\n=== 4. Intervalos de confianza (bootstrap) ===")
    ci = bootstrap_confidence_intervals(
        consensus_results, country="Ecuador",
        crisis_start="2024-10-01", crisis_end="2024-12-31",
        n_bootstrap=1000, confidence=0.95,
    )
    with open(OUT / "confidence_intervals.json", "w") as f:
        json.dump(ci, f, indent=2, default=str)

    for method, metrics in ci.items():
        f1_info = metrics.get("f1", {})
        logger.info(f"  {method:12s} F1={f1_info.get('mean',0):.3f} [{f1_info.get('ci_low',0):.3f}, {f1_info.get('ci_high',0):.3f}]")

    # ========================================
    # 5. SENSITIVITY ANALYSIS
    # ========================================
    logger.info("\n=== 5. Analisis de sensibilidad ===")
    ec_only = df[df["pais"] == "Ecuador"].copy()
    sens = sensitivity_analysis(
        ec_only, country="Ecuador",
        crisis_start="2024-10-01", crisis_end="2024-12-31",
    )
    sens.to_csv(OUT / "sensitivity_analysis.csv", index=False)
    best = sens.loc[sens["f1"].idxmax()]
    logger.info(f"  Mejor config: cont={best['contamination']}, stl={best['stl_sigma']}, cusum={best['cusum_factor']}")
    logger.info(f"  F1={best['f1']:.3f} MCC={best['mcc']:.3f}")
    logger.info(f"  {len(sens)} combinaciones evaluadas")

    logger.info("\n=== COMPARACION COMPLETA FINALIZADA ===")
    logger.info(f"Archivos en {OUT}/:")
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.csv")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
