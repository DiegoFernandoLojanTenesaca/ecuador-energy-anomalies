"""Scraper para datos de ARCERNNR (ex-ARCONEL).

Fuentes:
- regulacionelectrica.gob.ec: Boletines estadísticos del sector eléctrico
- arconel.gob.ec: Estadísticas del sector eléctrico
- anda.inec.gob.ec: Catálogo de datos INEC (Estadística Sector Eléctrico)
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from .utils import download_file, fetch_html

logger = logging.getLogger(__name__)

# URLs de ARCERNNR / ARCONEL
ARCONEL_STATS_URL = "https://arconel.gob.ec/estadistica-del-sector-electrico/"
ARCONEL_PUBLICATIONS_URL = (
    "https://arconel.gob.ec/publicaciones-estadistica-del-sector-electrico-2/"
)
REGULACION_BOLETINES_URL = (
    "https://www.regulacionelectrica.gob.ec/boletines-estadisticos/"
)

# INEC ANDA - Catálogo de datos
INEC_CATALOG_BASE = "https://anda.inec.gob.ec/anda/index.php/catalog"
INEC_SECTOR_2023 = f"{INEC_CATALOG_BASE}/1080"
INEC_SECTOR_2022 = f"{INEC_CATALOG_BASE}/981"

# Balance Nacional de Energía Eléctrica
BNEE_URL = (
    "https://datosabiertos.gob.ec/dataset/"
    "https-www-controlrecursosyenergia-gob-ec-balance-nacional-de-energia-electrica"
)

RAW_DIR = Path("data/raw/arcernnr")


class ArcernnrScraper:
    """Scraper para datos de ARCERNNR/ARCONEL."""

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_statistical_bulletins(self) -> list[Path]:
        """Descarga boletines estadísticos de ARCONEL/ARCERNNR.

        Returns:
            Lista de Paths a archivos descargados.
        """
        downloaded = []

        for url in [ARCONEL_STATS_URL, ARCONEL_PUBLICATIONS_URL, REGULACION_BOLETINES_URL]:
            try:
                html = fetch_html(url)
                soup = BeautifulSoup(html, "lxml")

                # Buscar enlaces a archivos descargables
                file_patterns = [
                    'a[href$=".xlsx"]', 'a[href$=".xls"]',
                    'a[href$=".csv"]', 'a[href$=".pdf"]',
                ]
                for pattern in file_patterns:
                    for link in soup.select(pattern):
                        href = link.get("href", "")
                        if not href:
                            continue

                        name = href.split("/")[-1]
                        # Solo descargar Excel y CSV, no PDF (difícil de parsear)
                        if name.lower().endswith(".pdf"):
                            continue

                        dest = self.raw_dir / "boletines" / name
                        try:
                            download_file(href, dest)
                            downloaded.append(dest)
                        except Exception as e:
                            logger.warning(f"No se pudo descargar {href}: {e}")

            except Exception as e:
                logger.warning(f"Error accediendo {url}: {e}")

        logger.info(f"Descargados {len(downloaded)} boletines estadísticos")
        return downloaded

    def download_bnee(self) -> list[Path]:
        """Descarga Balance Nacional de Energía Eléctrica.

        Returns:
            Lista de Paths a archivos descargados.
        """
        downloaded = []
        try:
            html = fetch_html(BNEE_URL)
            soup = BeautifulSoup(html, "lxml")

            for link in soup.select('a[href$=".xlsx"], a[href$=".xls"], a[href$=".csv"]'):
                href = link.get("href", "")
                if not href:
                    continue
                if not href.startswith("http"):
                    href = "https://datosabiertos.gob.ec" + href

                name = href.split("/")[-1] or "bnee.xlsx"
                dest = self.raw_dir / "bnee" / name
                try:
                    download_file(href, dest)
                    downloaded.append(dest)
                except Exception as e:
                    logger.warning(f"No se pudo descargar {href}: {e}")

        except Exception as e:
            logger.error(f"Error descargando BNEE: {e}")

        logger.info(f"Descargados {len(downloaded)} archivos BNEE")
        return downloaded

    def download_inec_data(self) -> list[Path]:
        """Descarga datos del catálogo INEC ANDA.

        Returns:
            Lista de Paths a archivos descargados.
        """
        downloaded = []

        for catalog_url in [INEC_SECTOR_2023, INEC_SECTOR_2022]:
            try:
                # Intentar acceder a la página de materiales relacionados
                materials_url = f"{catalog_url}/related_materials"
                html = fetch_html(materials_url)
                soup = BeautifulSoup(html, "lxml")

                for link in soup.select('a[href*="download"], a[href$=".csv"], a[href$=".xlsx"]'):
                    href = link.get("href", "")
                    if not href.startswith("http"):
                        href = "https://anda.inec.gob.ec" + href

                    name = link.get_text(strip=True) or href.split("/")[-1]
                    name = re.sub(r'[^\w\s\-.]', '_', name)[:80]
                    if not any(name.endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
                        name += ".xlsx"

                    year = "2023" if "1080" in catalog_url else "2022"
                    dest = self.raw_dir / "inec" / year / name
                    try:
                        download_file(href, dest)
                        downloaded.append(dest)
                    except Exception as e:
                        logger.warning(f"No se pudo descargar {href}: {e}")

            except Exception as e:
                logger.warning(f"Error accediendo INEC {catalog_url}: {e}")

        logger.info(f"Descargados {len(downloaded)} archivos INEC")
        return downloaded

    def scrape_all(self) -> dict[str, list[Path]]:
        """Ejecuta todos los scrapers disponibles.

        Returns:
            Diccionario con categoría -> lista de archivos descargados.
        """
        logger.info("=== Iniciando scraping ARCERNNR ===")
        results = {
            "boletines": self.download_statistical_bulletins(),
            "bnee": self.download_bnee(),
            "inec": self.download_inec_data(),
        }
        total = sum(len(v) for v in results.values())
        logger.info(f"=== ARCERNNR completado: {total} archivos ===")
        return results
