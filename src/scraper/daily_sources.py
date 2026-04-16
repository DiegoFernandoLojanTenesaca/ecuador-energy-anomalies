"""Scrapers para datos DIARIOS/HORARIOS de Latinoamerica.

Fase 2: datos de alta frecuencia para mejorar poder estadistico.

Fuentes:
- XM Colombia: API publica sin key (generacion horaria desde 2018)
- Electricity Maps: API con free tier (5min-diario, 2021-2024, necesita token)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


class XMColombiaDaily:
    """Scraper para datos horarios/diarios de XM Colombia (gratis, sin key)."""

    API_URL = "https://servapibi.xm.com.co/hourly"
    HEADERS = {"Content-Type": "application/json"}

    def __init__(self, raw_dir: Path = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_generation(
        self, start: datetime, end: datetime, delay: float = 0.5
    ) -> pd.DataFrame:
        """Descarga generacion horaria del SIN Colombia.

        Args:
            start: Fecha inicio.
            end: Fecha fin.
            delay: Segundos entre requests (rate limiting).

        Returns:
            DataFrame con columnas: fecha, hora, generacion_kwh, pais.
        """
        all_rows = []
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=29), end)
            body = {
                "MetricId": "Gene",
                "StartDate": current.strftime("%Y-%m-%d"),
                "EndDate": chunk_end.strftime("%Y-%m-%d"),
                "Entity": "Sistema",
            }
            try:
                r = requests.post(self.API_URL, json=body, headers=self.HEADERS, timeout=30)
                if r.status_code == 200:
                    for item in r.json().get("Items", []):
                        date = item["Date"]
                        for entity in item.get("HourlyEntities", []):
                            vals = entity.get("Values", {})
                            for h in range(1, 25):
                                key = f"Hour{h:02d}"
                                if key in vals and vals[key]:
                                    all_rows.append({
                                        "fecha": date,
                                        "hora": h,
                                        "generacion_kwh": float(vals[key]),
                                        "pais": "Colombia",
                                    })
                else:
                    logger.warning(f"XM {r.status_code} en {current.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.warning(f"XM timeout {current.strftime('%Y-%m-%d')}: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(delay)

        df = pd.DataFrame(all_rows)
        if len(df) > 0:
            df["fecha"] = pd.to_datetime(df["fecha"])
            df["generacion_mwh"] = df["generacion_kwh"] / 1000
        logger.info(f"XM Colombia: {len(df)} registros horarios")
        return df

    def to_daily(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Agrega horario a diario."""
        if hourly.empty:
            return pd.DataFrame()
        return hourly.groupby(["fecha", "pais"]).agg(
            generacion_total_mwh=("generacion_mwh", "sum"),
            generacion_media_mwh=("generacion_mwh", "mean"),
            generacion_max_mwh=("generacion_mwh", "max"),
            generacion_min_mwh=("generacion_mwh", "min"),
        ).reset_index()

    def scrape_and_save(
        self, start: datetime = None, end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Descarga, agrega a diario y guarda."""
        start = start or datetime(2021, 1, 1)
        end = end or datetime(2025, 1, 1)

        hourly = self.download_generation(start, end)
        daily = self.to_daily(hourly)

        hourly.to_csv(self.raw_dir / "colombia_hourly_generation.csv", index=False)
        daily.to_csv(self.raw_dir / "colombia_daily_generation.csv", index=False)
        logger.info(f"Guardado: {len(hourly)} horarios, {len(daily)} diarios")

        return hourly, daily


class ElectricityMapsDaily:
    """Scraper para datos diarios de Electricity Maps (necesita token).

    Registro gratis en: https://app.electricitymaps.com
    Token gratis incluye 5 datasets historicos.
    """

    API_URL = "https://api.electricitymap.org/v3"
    LATAM_ZONES = {
        "EC": "Ecuador",
        "CO": "Colombia",
        "PE": "Peru",
        "CL-SEN": "Chile",
        "BR": "Brazil",
        "AR": "Argentina",
        "BO": "Bolivia",
        "UY": "Uruguay",
    }

    def __init__(self, token: str = None, raw_dir: Path = None):
        self.token = token
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        headers = {"auth-token": self.token} if self.token else {}
        r = requests.get(f"{self.API_URL}/{endpoint}", params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def download_power_breakdown(
        self, zone: str, start: datetime, end: datetime, granularity: str = "daily"
    ) -> pd.DataFrame:
        """Descarga power breakdown historico para una zona.

        Args:
            zone: Codigo de zona (EC, CO, BR, etc.).
            start: Fecha inicio.
            end: Fecha fin.
            granularity: 'hourly' o 'daily'.

        Returns:
            DataFrame con generacion por fuente.
        """
        if not self.token:
            logger.error("Electricity Maps necesita auth-token. Registrate en app.electricitymaps.com")
            return pd.DataFrame()

        all_data = []
        current = start

        # API limita a 10 dias por request para hourly, 100 para daily
        chunk_days = 100 if granularity == "daily" else 10

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            try:
                data = self._get("power-breakdown/past-range", {
                    "zone": zone,
                    "start": current.isoformat(),
                    "end": chunk_end.isoformat(),
                    "temporalGranularity": granularity,
                })
                if isinstance(data, list):
                    all_data.extend(data)
                logger.info(f"  {zone} {current.strftime('%Y-%m')}... {len(all_data)} records")
            except Exception as e:
                logger.warning(f"  {zone} {current.strftime('%Y-%m')}: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["zona"] = zone
        df["pais"] = self.LATAM_ZONES.get(zone, zone)
        logger.info(f"Electricity Maps {zone}: {len(df)} registros")
        return df

    def scrape_latam(
        self, start: datetime = None, end: datetime = None, granularity: str = "daily"
    ) -> pd.DataFrame:
        """Descarga datos de todos los paises LATAM."""
        start = start or datetime(2021, 1, 1)
        end = end or datetime(2025, 1, 1)

        all_dfs = []
        for zone, country in self.LATAM_ZONES.items():
            logger.info(f"Descargando {country} ({zone})...")
            df = self.download_power_breakdown(zone, start, end, granularity)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(self.raw_dir / "latam_daily_electricity.csv", index=False)
        logger.info(f"LATAM diario: {len(combined)} registros de {len(all_dfs)} paises")
        return combined
