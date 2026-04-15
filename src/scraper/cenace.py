"""Scraper para datos de CENACE (Centro Nacional de Control de Energía).

Fuentes:
- datosabiertos.gob.ec: Producción de energía eléctrica del parque generador (XLS/CSV)
- cenace.gob.ec: Información operativa (HTML)
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from .utils import download_file, fetch_html

logger = logging.getLogger(__name__)

# URLs conocidas del portal de datos abiertos
DATOS_ABIERTOS_BASE = "https://www.datosabiertos.gob.ec"
CENACE_DATASETS_URL = f"{DATOS_ABIERTOS_BASE}/dataset/?organization=cenace"
CENACE_PRODUCCION_URL = (
    f"{DATOS_ABIERTOS_BASE}/dataset/"
    "produccion-de-energia-electrica-del-parque-generador1"
)

# Información operativa de CENACE
CENACE_INFO_OPERATIVA = "https://www.cenace.gob.ec/info-operativa/InformacionOperativa.htm"

RAW_DIR = Path("data/raw/cenace")


class CenaceScraper:
    """Scraper para datos del CENACE Ecuador."""

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self) -> list[dict]:
        """Descubre datasets disponibles en el portal de datos abiertos.

        Returns:
            Lista de diccionarios con info de cada dataset encontrado.
        """
        datasets = []
        try:
            html = fetch_html(CENACE_DATASETS_URL)
            soup = BeautifulSoup(html, "lxml")

            for item in soup.select(".dataset-item, .dataset-listing .dataset-content"):
                title_el = item.select_one("h3 a, .dataset-heading a")
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                if href and not href.startswith("http"):
                    href = DATOS_ABIERTOS_BASE + href

                desc_el = item.select_one(".notes, p")
                desc = desc_el.get_text(strip=True) if desc_el else ""

                formats = [
                    span.get_text(strip=True).upper()
                    for span in item.select(".format-label, .label")
                ]

                datasets.append({
                    "title": title,
                    "url": href,
                    "description": desc,
                    "formats": formats,
                })

            logger.info(f"Encontrados {len(datasets)} datasets de CENACE")
        except Exception as e:
            logger.error(f"Error descubriendo datasets: {e}")

        return datasets

    def download_production_data(self) -> list[Path]:
        """Descarga datos de producción de energía eléctrica.

        Busca archivos XLS/XLSX/CSV en el dataset de producción
        del parque generador.

        Returns:
            Lista de Paths a archivos descargados.
        """
        downloaded = []
        try:
            html = fetch_html(CENACE_PRODUCCION_URL)
            soup = BeautifulSoup(html, "lxml")

            # Buscar enlaces a recursos descargables
            resource_links = soup.select(
                'a[href$=".xls"], a[href$=".xlsx"], a[href$=".csv"], '
                'a.resource-url-analytics'
            )

            for link in resource_links:
                href = link.get("href", "")
                if not href:
                    continue
                if not href.startswith("http"):
                    href = DATOS_ABIERTOS_BASE + href

                # Determinar nombre del archivo
                name = link.get_text(strip=True) or href.split("/")[-1]
                name = re.sub(r'[^\w\s\-.]', '_', name)[:80]

                # Determinar extensión
                ext = ".xlsx"
                if href.endswith(".csv"):
                    ext = ".csv"
                elif href.endswith(".xls"):
                    ext = ".xls"

                if not name.endswith(ext):
                    name = f"{name}{ext}"

                dest = self.raw_dir / "produccion" / name
                try:
                    download_file(href, dest)
                    downloaded.append(dest)
                except Exception as e:
                    logger.warning(f"No se pudo descargar {href}: {e}")

            logger.info(f"Descargados {len(downloaded)} archivos de producción")
        except Exception as e:
            logger.error(f"Error descargando datos de producción: {e}")

        return downloaded

    def download_dispatched_power(self) -> list[Path]:
        """Descarga datos de potencia efectiva despachada.

        Returns:
            Lista de Paths a archivos descargados.
        """
        downloaded = []
        search_url = (
            f"{DATOS_ABIERTOS_BASE}/dataset/"
            "?organization=cenace&q=potencia+efectiva+despachada"
        )

        try:
            html = fetch_html(search_url)
            soup = BeautifulSoup(html, "lxml")

            dataset_links = soup.select(".dataset-heading a, h3 a")
            for ds_link in dataset_links:
                ds_url = ds_link.get("href", "")
                if not ds_url.startswith("http"):
                    ds_url = DATOS_ABIERTOS_BASE + ds_url

                ds_html = fetch_html(ds_url)
                ds_soup = BeautifulSoup(ds_html, "lxml")

                for res in ds_soup.select('a[href$=".xls"], a[href$=".xlsx"], a[href$=".csv"]'):
                    href = res.get("href", "")
                    if not href.startswith("http"):
                        href = DATOS_ABIERTOS_BASE + href

                    name = href.split("/")[-1] or "potencia_despachada.xlsx"
                    dest = self.raw_dir / "potencia" / name
                    try:
                        download_file(href, dest)
                        downloaded.append(dest)
                    except Exception as e:
                        logger.warning(f"No se pudo descargar {href}: {e}")

        except Exception as e:
            logger.error(f"Error descargando potencia despachada: {e}")

        return downloaded

    def load_production_excel(self, filepath: Path) -> pd.DataFrame:
        """Carga un archivo Excel de producción y lo normaliza.

        Args:
            filepath: Ruta al archivo Excel.

        Returns:
            DataFrame con columnas normalizadas.
        """
        try:
            if filepath.suffix == ".csv":
                df = pd.read_csv(filepath, encoding="latin-1")
            else:
                df = pd.read_excel(filepath, engine="openpyxl")

            # Normalizar nombres de columnas
            df.columns = [
                col.strip().lower().replace(" ", "_").replace("á", "a")
                .replace("é", "e").replace("í", "i").replace("ó", "o")
                .replace("ú", "u").replace("ñ", "n")
                for col in df.columns
            ]

            logger.info(f"Cargado {filepath.name}: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error cargando {filepath}: {e}")
            return pd.DataFrame()

    def scrape_all(self) -> dict[str, list[Path]]:
        """Ejecuta todos los scrapers disponibles.

        Returns:
            Diccionario con categoría -> lista de archivos descargados.
        """
        logger.info("=== Iniciando scraping CENACE ===")
        results = {
            "produccion": self.download_production_data(),
            "potencia": self.download_dispatched_power(),
        }
        total = sum(len(v) for v in results.values())
        logger.info(f"=== CENACE completado: {total} archivos ===")
        return results
