"""Scraper para datos reales de Ember y Our World in Data.

Estas son las fuentes que realmente funcionan y tienen datos
mensuales de Ecuador actualizados.

Fuentes:
- Ember: Datos mensuales de generación, demanda, emisiones (2019-presente)
- OWID: Datos anuales complementarios con contexto histórico (1900-presente)
- World Bank: Indicadores de consumo per cápita
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import download_file

logger = logging.getLogger(__name__)

# URLs de descarga directa (verificadas y funcionales)
EMBER_MONTHLY_URL = (
    "https://storage.googleapis.com/emb-prod-bkt-publicdata/"
    "public-downloads/monthly_full_release_long_format.csv"
)
EMBER_YEARLY_URL = (
    "https://storage.googleapis.com/emb-prod-bkt-publicdata/"
    "public-downloads/yearly_full_release_long_format.csv"
)
OWID_ENERGY_URL = (
    "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
)
WORLD_BANK_URL_TEMPLATE = (
    "https://api.worldbank.org/v2/country/EC/indicator/{indicator}"
    "?date=1990:2030&format=json&per_page=100"
)

WB_INDICATORS = {
    "EG.USE.ELEC.KH.PC": "consumo_electrico_per_capita_kwh",
}

RAW_DIR = Path("data/raw")


class EmberOwidScraper:
    """Scraper para datos de Ember y Our World in Data."""

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_ember_monthly(self) -> Path:
        """Descarga datos mensuales de Ember (generación, demanda, emisiones)."""
        dest = self.raw_dir / "ember_monthly_full.csv"
        logger.info("Descargando Ember monthly data...")
        return download_file(EMBER_MONTHLY_URL, dest)

    def download_ember_yearly(self) -> Path:
        """Descarga datos anuales de Ember."""
        dest = self.raw_dir / "ember_yearly_full.csv"
        logger.info("Descargando Ember yearly data...")
        return download_file(EMBER_YEARLY_URL, dest)

    def download_owid(self) -> Path:
        """Descarga datos de Our World in Data."""
        dest = self.raw_dir / "owid_energy_full.csv"
        logger.info("Descargando Our World in Data energy dataset...")
        return download_file(OWID_ENERGY_URL, dest)

    def download_world_bank(self) -> list[Path]:
        """Descarga indicadores del World Bank para Ecuador."""
        downloaded = []
        for indicator, name in WB_INDICATORS.items():
            url = WORLD_BANK_URL_TEMPLATE.format(indicator=indicator)
            dest = self.raw_dir / f"worldbank_{name}.json"
            try:
                download_file(url, dest)
                downloaded.append(dest)
            except Exception as e:
                logger.warning(f"World Bank {indicator} falló: {e}")
        return downloaded

    def extract_ecuador_monthly(self, ember_path: Path = None) -> pd.DataFrame:
        """Extrae y pivota datos mensuales de Ecuador del dataset Ember.

        Returns:
            DataFrame con columnas: fecha, gen_hydro, gen_fossil, demanda_twh, etc.
        """
        ember_path = ember_path or self.raw_dir / "ember_monthly_full.csv"
        if not ember_path.exists():
            ember_path = self.download_ember_monthly()

        df = pd.read_csv(ember_path)
        ec = df[df["Area"] == "Ecuador"].copy()
        ec["Date"] = pd.to_datetime(ec["Date"])

        logger.info(f"Ecuador monthly: {len(ec)} registros, {ec['Date'].min()} a {ec['Date'].max()}")

        # Pivotar generación
        gen = ec[ec["Category"] == "Electricity generation"].pivot_table(
            index="Date", columns="Variable", values="Value"
        ).reset_index()
        gen.columns = ["fecha"] + [f"gen_{c.lower().replace(' ', '_')}" for c in gen.columns[1:]]

        # Demanda
        demand = ec[ec["Variable"] == "Demand"][["Date", "Value"]].copy()
        demand.columns = ["fecha", "demanda_twh"]

        # CO2 intensity
        co2 = ec[ec["Variable"] == "CO2 intensity"][["Date", "Value"]].copy()
        co2.columns = ["fecha", "co2_intensity_gco2_kwh"]

        # Importaciones netas
        imports = ec[ec["Variable"] == "Net Imports"][["Date", "Value"]].copy()
        imports.columns = ["fecha", "importaciones_netas_twh"]

        # Merge
        result = gen.merge(demand, on="fecha", how="outer")
        result = result.merge(co2, on="fecha", how="outer")
        result = result.merge(imports, on="fecha", how="outer")
        result = result.sort_values("fecha").reset_index(drop=True)

        return result

    def extract_ecuador_annual(self, owid_path: Path = None) -> pd.DataFrame:
        """Extrae datos anuales de Ecuador del dataset OWID."""
        owid_path = owid_path or self.raw_dir / "owid_energy_full.csv"
        if not owid_path.exists():
            owid_path = self.download_owid()

        df = pd.read_csv(owid_path)
        ec = df[df["country"] == "Ecuador"].copy()

        cols = ["year", "population", "gdp", "per_capita_electricity",
                "hydro_share_elec", "fossil_share_elec", "renewables_share_elec"]
        available = [c for c in cols if c in ec.columns]
        result = ec[available].copy()

        rename = {
            "year": "anio", "population": "poblacion", "gdp": "pib_usd",
            "per_capita_electricity": "consumo_per_capita_kwh",
            "hydro_share_elec": "share_hidro_pct",
            "fossil_share_elec": "share_fossil_pct",
            "renewables_share_elec": "share_renovable_pct",
        }
        result = result.rename(columns={k: v for k, v in rename.items() if k in result.columns})

        return result

    LATAM_COUNTRIES = [
        "Ecuador", "Colombia", "Peru", "Chile",
        "Brazil", "Argentina", "Bolivia", "Uruguay",
    ]

    def extract_latam_monthly(self, ember_path: Path = None) -> pd.DataFrame:
        """Extrae datos mensuales de 8 países LATAM del dataset Ember."""
        ember_path = ember_path or self.raw_dir / "ember_monthly_full.csv"
        if not ember_path.exists():
            ember_path = self.download_ember_monthly()

        df = pd.read_csv(ember_path)
        df["Date"] = pd.to_datetime(df["Date"])

        all_dfs = []
        for country in self.LATAM_COUNTRIES:
            c = df[df["Area"] == country].copy()
            if len(c) == 0:
                continue

            gen = c[c["Category"] == "Electricity generation"].pivot_table(
                index="Date", columns="Variable", values="Value"
            ).reset_index()
            gen.columns = ["fecha"] + [f"gen_{x.lower().replace(' ', '_')}" for x in gen.columns[1:]]

            dem = c[c["Variable"] == "Demand"][["Date", "Value"]].copy()
            dem.columns = ["fecha", "demanda_twh"]

            co2 = c[c["Variable"] == "CO2 intensity"][["Date", "Value"]].copy()
            co2.columns = ["fecha", "co2_intensity"]

            imp = c[c["Variable"] == "Net Imports"][["Date", "Value"]].copy()
            imp.columns = ["fecha", "importaciones_netas"]

            result = gen.merge(dem, on="fecha", how="outer")
            result = result.merge(co2, on="fecha", how="outer")
            result = result.merge(imp, on="fecha", how="outer")
            result["pais"] = country
            all_dfs.append(result)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.columns = [col.replace(",", "_").replace(" ", "_") for col in combined.columns]
        combined = combined.sort_values(["pais", "fecha"]).reset_index(drop=True)

        output = self.raw_dir / "latam_electricity.csv"
        combined.to_csv(output, index=False)
        combined.to_parquet(self.raw_dir / "latam_electricity.parquet", index=False)
        logger.info(f"LATAM dataset: {combined.shape} ({combined['pais'].nunique()} paises)")

        return combined

    def build_dataset(self) -> pd.DataFrame:
        """Construye el dataset completo de Ecuador combinando todas las fuentes.

        Returns:
            DataFrame con datos mensuales enriquecidos con datos anuales.
        """
        logger.info("=== Construyendo dataset Ecuador ===")

        # Datos mensuales (Ember)
        monthly = self.extract_ecuador_monthly()

        # Datos anuales (OWID)
        annual = self.extract_ecuador_annual()

        # Merge anual sobre mensual
        monthly["anio"] = monthly["fecha"].dt.year
        result = monthly.merge(annual, on="anio", how="left")

        # Limpiar nombres de columnas
        result.columns = [c.replace(",", "_").replace(" ", "_") for c in result.columns]

        # Guardar
        output_csv = self.raw_dir / "ecuador_electricity_real.csv"
        output_parquet = self.raw_dir / "ecuador_electricity_real.parquet"
        result.to_csv(output_csv, index=False)
        result.to_parquet(output_parquet, index=False)

        logger.info(f"Dataset construido: {result.shape}")
        logger.info(f"Guardado en {output_csv} y {output_parquet}")

        return result

    def scrape_all(self) -> dict[str, list[Path]]:
        """Ejecuta todos los scrapers y construye el dataset."""
        logger.info("=== Iniciando descarga de datos ===")

        results = {"ember_monthly": [], "ember_yearly": [], "owid": [], "worldbank": []}

        try:
            path = self.download_ember_monthly()
            results["ember_monthly"].append(path)
        except Exception as e:
            logger.error(f"Ember monthly falló: {e}")

        try:
            path = self.download_ember_yearly()
            results["ember_yearly"].append(path)
        except Exception as e:
            logger.error(f"Ember yearly falló: {e}")

        try:
            path = self.download_owid()
            results["owid"].append(path)
        except Exception as e:
            logger.error(f"OWID falló: {e}")

        results["worldbank"] = self.download_world_bank()

        # Construir datasets
        try:
            self.build_dataset()
        except Exception as e:
            logger.error(f"Error construyendo dataset Ecuador: {e}")

        try:
            self.extract_latam_monthly()
        except Exception as e:
            logger.error(f"Error construyendo dataset LATAM: {e}")

        total = sum(len(v) for v in results.values())
        logger.info(f"=== Descarga completada: {total} archivos ===")
        return results
