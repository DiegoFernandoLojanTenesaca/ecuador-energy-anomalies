"""Utilidades compartidas para los scrapers."""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-EC,es;q=0.9,en;q=0.5",
}

CACHE_DIR = Path(os.getenv("SCRAPER_CACHE_DIR", "data/cache"))
DELAY = float(os.getenv("SCRAPER_DELAY_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))


def get_cache_path(url: str, ext: str = "") -> Path:
    """Genera ruta de cache basada en hash de la URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    filename = f"{url_hash}{ext}"
    return CACHE_DIR / filename


def download_file(
    url: str,
    dest: Path,
    use_cache: bool = True,
    headers: Optional[dict] = None,
) -> Path:
    """Descarga un archivo con retry, rate limiting y cache.

    Args:
        url: URL del archivo a descargar.
        dest: Ruta destino donde guardar el archivo.
        use_cache: Si True, usa cache local.
        headers: Headers HTTP adicionales.

    Returns:
        Path al archivo descargado.
    """
    if use_cache and dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Cache hit: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Descargando ({attempt}/{MAX_RETRIES}): {url}")
            resp = requests.get(
                url,
                headers=merged_headers,
                timeout=60,
                verify=False,  # Sitios .gob.ec tienen certificados problemáticos
                stream=True,
            )
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Descargado: {dest} ({dest.stat().st_size:,} bytes)")
            time.sleep(DELAY)
            return dest

        except requests.RequestException as e:
            logger.warning(f"Intento {attempt} falló: {e}")
            if attempt < MAX_RETRIES:
                wait = DELAY * attempt
                logger.info(f"Esperando {wait}s antes de reintentar...")
                time.sleep(wait)
            else:
                raise

    return dest  # unreachable, pero satisface el type checker


def fetch_html(url: str, use_cache: bool = True) -> str:
    """Descarga una página HTML con cache."""
    cache_path = get_cache_path(url, ".html")

    if use_cache and cache_path.exists():
        logger.info(f"Cache hit HTML: {cache_path}")
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            html = resp.text

            cache_path.write_text(html, encoding="utf-8")
            time.sleep(DELAY)
            return html

        except requests.RequestException as e:
            logger.warning(f"Intento {attempt} falló: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY * attempt)
            else:
                raise

    return ""
