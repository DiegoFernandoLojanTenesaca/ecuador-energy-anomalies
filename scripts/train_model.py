"""Pipeline completo: IF por pais + STL + CUSUM + consenso.

Este script reproduce TODAS las metricas del README.
Entrada: data/raw/latam_electricity.csv (8 paises, ~784 meses)
Salida: data/processed/latam_multitechnique_results.parquet + metricas
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.consensus import ConsensusDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos LATAM
    latam_path = RAW_DIR / "latam_electricity.csv"
    if not latam_path.exists():
        logger.error(f"No existe {latam_path}. Ejecuta scrape_all.py primero.")
        sys.exit(1)

    df = pd.read_csv(latam_path, parse_dates=["fecha"])
    df.columns = [c.replace(",", "_").replace(" ", "_") for c in df.columns]
    logger.info(f"Datos: {len(df)} filas, {df['pais'].nunique()} paises")

    # 2. Ejecutar multi-tecnica por pais
    detector = ConsensusDetector(
        if_params={"n_estimators": 300, "contamination": 0.08, "random_state": 42},
        stl_threshold_sigma=2.0,
        cusum_threshold_factor=4.0,
        min_agreement=2,
        warmup_months=12,
    )

    results = detector.fit_predict_multi(df, country_col="pais")

    # 3. Guardar resultados
    out_csv = PROCESSED_DIR / "latam_multitechnique_results.csv"
    out_parquet = PROCESSED_DIR / "latam_multitechnique_results.parquet"
    results.to_csv(out_csv, index=False)
    results.to_parquet(out_parquet, index=False)
    logger.info(f"Resultados: {out_csv}")

    # 4. Metricas Ecuador (ground truth: oct-dic 2024)
    metrics = ConsensusDetector.compute_metrics(
        results, country="Ecuador",
        crisis_start="2024-10-01", crisis_end="2024-12-31",
    )

    logger.info("=== METRICAS ECUADOR (GT: oct-dic 2024) ===")
    for method, m in metrics.items():
        if isinstance(m, dict) and "f1" in m:
            logger.info(
                f"  {method:20s} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} MCC={m['mcc']:.3f} "
                f"crisis={m['crisis_detected']}/{m['crisis_total']}"
            )

    # Guardar metricas como JSON
    metrics_path = PROCESSED_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metricas: {metrics_path}")

    # 5. Resumen por pais
    logger.info("=== ANOMALIAS POR PAIS ===")
    for pais in sorted(results["pais"].unique()):
        sub = results[results["pais"] == pais]
        n_if = sub["if_anomaly"].sum()
        n_stl = sub["stl_anomaly"].sum()
        n_cu = sub["cusum_anomaly"].sum()
        n_con = sub["consensus"].sum()
        logger.info(f"  {pais:12s} IF={n_if:>2} STL={n_stl:>2} CUSUM={n_cu:>2} Consenso={n_con:>2}")

    logger.info("=== Pipeline completado ===")


if __name__ == "__main__":
    main()
