"""Script para descargar datos reales del sector eléctrico de Ecuador.

Fuentes principales (funcionales):
- Ember: datos mensuales de generación, demanda, emisiones
- Our World in Data: datos anuales complementarios
- World Bank: indicadores de consumo per cápita

Fuentes secundarias (gobierno ecuatoriano, pueden fallar):
- CENACE: datosabiertos.gob.ec
- ARCERNNR: regulacionelectrica.gob.ec
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scraper.ember_owid import EmberOwidScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Descargando datos eléctricos de Ecuador ===")

    # Fuentes principales (Ember + OWID + World Bank)
    scraper = EmberOwidScraper()
    results = scraper.scrape_all()

    # Resumen
    logger.info("=== RESUMEN ===")
    for source, files in results.items():
        logger.info(f"  {source}: {len(files)} archivos")

    total = sum(len(v) for v in results.values())
    logger.info(f"TOTAL: {total} archivos descargados")

    # Intentar fuentes gubernamentales (pueden fallar)
    try:
        from src.scraper.cenace import CenaceScraper
        from src.scraper.arcernnr import ArcernnrScraper

        logger.info("--- Intentando fuentes gubernamentales (pueden fallar) ---")
        cenace = CenaceScraper()
        cenace_results = cenace.scrape_all()
        arcernnr = ArcernnrScraper()
        arcernnr_results = arcernnr.scrape_all()

        for cat, files in cenace_results.items():
            if files:
                logger.info(f"  CENACE/{cat}: {len(files)} archivos")
        for cat, files in arcernnr_results.items():
            if files:
                logger.info(f"  ARCERNNR/{cat}: {len(files)} archivos")
    except Exception as e:
        logger.warning(f"Fuentes gubernamentales no disponibles: {e}")

    logger.info("=== Scraping completado ===")


if __name__ == "__main__":
    main()
