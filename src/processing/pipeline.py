"""Pipeline ETL completo: Raw -> Clean -> Features -> ML-ready."""

import logging
from pathlib import Path

import pandas as pd

from .cleaner import clean_dataframe
from .features import engineer_features

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_all_raw_files(source_dir: Path) -> pd.DataFrame:
    """Carga y concatena todos los archivos Excel/CSV de un directorio.

    Args:
        source_dir: Directorio con archivos raw.

    Returns:
        DataFrame concatenado.
    """
    dfs = []

    for filepath in sorted(source_dir.rglob("*")):
        if filepath.suffix.lower() in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(filepath, engine="openpyxl")
                df["_source_file"] = filepath.name
                dfs.append(df)
                logger.info(f"Cargado: {filepath.name} ({len(df)} filas)")
            except Exception as e:
                logger.warning(f"Error cargando {filepath}: {e}")

        elif filepath.suffix.lower() == ".csv":
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    df["_source_file"] = filepath.name
                    dfs.append(df)
                    logger.info(f"Cargado: {filepath.name} ({len(df)} filas)")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

    if not dfs:
        logger.warning(f"No se encontraron archivos en {source_dir}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total combinado: {combined.shape}")
    return combined


def run_etl(
    source: str = "cenace",
    date_col: str = "fecha",
    value_cols: list[str] = None,
) -> pd.DataFrame:
    """Ejecuta el pipeline ETL completo.

    Args:
        source: "cenace", "arcernnr", o "all".
        date_col: Nombre de la columna de fecha.
        value_cols: Columnas numéricas principales.

    Returns:
        DataFrame procesado y listo para ML.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if source == "all":
        sources = ["cenace", "arcernnr"]
    else:
        sources = [source]

    all_dfs = []

    for src in sources:
        src_dir = RAW_DIR / src
        if not src_dir.exists():
            logger.warning(f"Directorio {src_dir} no existe")
            continue

        logger.info(f"--- Procesando {src} ---")

        # 1. Cargar datos raw
        raw_df = load_all_raw_files(src_dir)
        if raw_df.empty:
            continue

        # 2. Limpiar
        clean_df = clean_dataframe(raw_df, date_col)

        # 3. Feature engineering
        featured_df = engineer_features(clean_df, date_col, value_cols)

        # 4. Guardar intermedio
        output_path = PROCESSED_DIR / f"{src}_processed.parquet"
        featured_df.to_parquet(output_path, index=False)
        logger.info(f"Guardado: {output_path}")

        all_dfs.append(featured_df)

    if not all_dfs:
        logger.error("No se procesaron datos")
        return pd.DataFrame()

    # Combinar todas las fuentes
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = PROCESSED_DIR / "combined_processed.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"Combinado guardado: {combined_path}")
        return combined

    return all_dfs[0]
