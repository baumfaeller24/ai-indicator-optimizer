"""
Dukascopy Data Connector für Forex-Daten mit parallelem Download
"""

import asyncio
import aiohttp
import struct
import gzip
import lzma
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import logging
import time
import numpy as np
from dataclasses import dataclass

from .models import TickData, OHLCVData, MarketData


@dataclass
class DukascopyConfig:
    """Konfiguration für Dukascopy Connector"""
    base_url: str = "https://datafeed.dukascopy.com/datafeed"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    chunk_size: int = 8192
    cache_dir: str = "./data/cache"


class DukascopyDataValidator:
    """Validiert Dukascopy Forex-Daten"""
    
    @staticmethod
    def validate_tick_data(tick_data: List[TickData]) -> Dict[str, Any]:
        """Validiert Tick-Daten"""
        if not tick_data:
            return {"valid": False, "errors": ["No tick data provided"]}
        
        errors = []
        warnings = []
        
        # Zeitstempel-Validierung
        timestamps = [tick.timestamp for tick in tick_data]
        if len(timestamps) != len(set(timestamps)):
            errors.append("Duplicate timestamps found")
        
        # Sortierung prüfen
        if timestamps != sorted(timestamps):
            warnings.append("Timestamps not in chronological order")
        
        # Preis-Validierung
        for i, tick in enumerate(tick_data):
            if tick.bid <= 0 or tick.ask <= 0:
                errors.append(f"Invalid price at index {i}: bid={tick.bid}, ask={tick.ask}")
            
            if tick.ask <= tick.bid:
                errors.append(f"Ask price <= Bid price at index {i}")
            
            if tick.volume < 0:
                errors.append(f"Negative volume at index {i}: {tick.volume}")
        
        # Spread-Analyse
        spreads = [tick.spread for tick in tick_data]
        avg_spread = np.mean(spreads)
        max_spread = np.max(spreads)
        
        if max_spread > avg_spread * 10:
            warnings.append(f"Unusually large spread detected: {max_spread}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "count": len(tick_data),
                "avg_spread": avg_spread,
                "max_spread": max_spread,
                "time_range": (timestamps[0], timestamps[-1]) if timestamps else None
            }
        }
    
    @staticmethod
    def validate_ohlcv_data(ohlcv_data: List[OHLCVData]) -> Dict[str, Any]:
        """Validiert OHLCV-Daten"""
        if not ohlcv_data:
            return {"valid": False, "errors": ["No OHLCV data provided"]}
        
        errors = []
        warnings = []
        
        for i, candle in enumerate(ohlcv_data):
            # OHLC-Konsistenz
            if not (candle.low <= candle.open <= candle.high and 
                   candle.low <= candle.close <= candle.high):
                errors.append(f"OHLC inconsistency at index {i}")
            
            # Positive Werte
            if any(val <= 0 for val in [candle.open, candle.high, candle.low, candle.close]):
                errors.append(f"Non-positive OHLC values at index {i}")
            
            # Volume
            if candle.volume < 0:
                errors.append(f"Negative volume at index {i}")
        
        # Zeitstempel-Validierung
        timestamps = [candle.timestamp for candle in ohlcv_data]
        if timestamps != sorted(timestamps):
            warnings.append("Timestamps not in chronological order")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "count": len(ohlcv_data),
                "time_range": (timestamps[0], timestamps[-1]) if timestamps else None
            }
        }


class DukascopyConnector:
    """
    Dukascopy API Connector für Forex-Daten mit parallelem Download
    """
    
    def __init__(self, parallel_workers: int = 32, config: Optional[DukascopyConfig] = None):
        self.parallel_workers = parallel_workers
        self.config = config or DukascopyConfig()
        self.validator = DukascopyDataValidator()
        self.logger = logging.getLogger(__name__)
        
        # Cache-Verzeichnis erstellen
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Symbol-Mapping für Dukascopy
        self.symbol_mapping = {
            "EURUSD": "EURUSD",
            "GBPUSD": "GBPUSD", 
            "USDJPY": "USDJPY",
            "USDCHF": "USDCHF",
            "AUDUSD": "AUDUSD",
            "USDCAD": "USDCAD",
            "NZDUSD": "NZDUSD"
        }
    
    def fetch_tick_data(self, symbol: str, start: datetime, end: datetime) -> List[TickData]:
        """
        Lädt Tick-Daten für Symbol und Zeitraum mit parallelem Download
        """
        self.logger.info(f"Fetching tick data for {symbol} from {start} to {end}")
        
        if symbol not in self.symbol_mapping:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        # Zeitraum in Stunden aufteilen für parallelen Download
        hour_ranges = self._split_time_range_by_hours(start, end)
        
        # Paralleler Download mit ThreadPoolExecutor
        tick_data = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_range = {
                executor.submit(self._fetch_tick_data_hour, symbol, hour_start, hour_end): (hour_start, hour_end)
                for hour_start, hour_end in hour_ranges
            }
            
            for future in as_completed(future_to_range):
                hour_start, hour_end = future_to_range[future]
                try:
                    hour_data = future.result()
                    tick_data.extend(hour_data)
                    self.logger.debug(f"Downloaded {len(hour_data)} ticks for {hour_start}")
                except Exception as e:
                    self.logger.error(f"Failed to download data for {hour_start}: {e}")
        
        # Sortieren nach Zeitstempel
        tick_data.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Downloaded {len(tick_data)} total ticks")
        return tick_data
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str, period: int) -> List[OHLCVData]:
        """
        Lädt OHLCV-Daten für Symbol, Timeframe und Periode
        """
        self.logger.info(f"Fetching OHLCV data for {symbol}, {timeframe}, {period} days")
        
        # Berechne Zeitraum
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=period)
        
        # Lade Tick-Daten
        tick_data = self.fetch_tick_data(symbol, start_time, end_time)
        
        # Konvertiere zu OHLCV
        ohlcv_data = self._convert_ticks_to_ohlcv(tick_data, timeframe)
        
        self.logger.info(f"Generated {len(ohlcv_data)} OHLCV candles")
        return ohlcv_data
    
    def _fetch_tick_data_hour(self, symbol: str, start: datetime, end: datetime) -> List[TickData]:
        """
        Lädt Tick-Daten für eine Stunde (Dukascopy-spezifisches Format)
        """
        try:
            # Dukascopy verwendet UTC und spezielle URL-Struktur
            year = start.year
            month = start.month - 1  # Dukascopy: 0-basiert
            day = start.day
            hour = start.hour
            
            # URL für Dukascopy Tick-Daten
            url = f"{self.config.base_url}/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
            
            # Download mit Retry-Logic
            binary_data = self._download_with_retry(url)
            
            if not binary_data:
                return []
            
            # Dekomprimiere LZMA-komprimierte Daten
            try:
                decompressed_data = lzma.decompress(binary_data)
            except:
                # Fallback: versuche GZIP
                try:
                    decompressed_data = gzip.decompress(binary_data)
                except:
                    # Rohdaten verwenden
                    decompressed_data = binary_data
            
            # Parse Dukascopy .bi5 Format
            tick_data = self._parse_bi5_data(decompressed_data, start)
            
            return tick_data
            
        except Exception as e:
            self.logger.error(f"Error fetching tick data for {start}: {e}")
            return []
    
    def _download_with_retry(self, url: str) -> Optional[bytes]:
        """
        Download mit Retry-Logic
        """
        for attempt in range(self.config.max_retries):
            try:
                # Synchroner HTTP-Request (für ThreadPool)
                import requests
                
                response = requests.get(
                    url, 
                    timeout=self.config.timeout,
                    headers={'User-Agent': 'AI-Indicator-Optimizer/1.0'}
                )
                
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 404:
                    # Keine Daten für diesen Zeitraum
                    return None
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        return None
    
    def _parse_bi5_data(self, data: bytes, base_time: datetime) -> List[TickData]:
        """
        Parsed Dukascopy .bi5 Format (20 Bytes pro Tick)
        """
        tick_data = []
        
        if len(data) % 20 != 0:
            self.logger.warning(f"Invalid bi5 data length: {len(data)}")
            return tick_data
        
        # Basis-Zeitstempel (Stundenbeginn)
        base_timestamp = int(base_time.timestamp() * 1000)
        
        for i in range(0, len(data), 20):
            try:
                # Dukascopy .bi5 Format:
                # 4 bytes: time offset (ms from hour start)
                # 4 bytes: ask price (scaled)
                # 4 bytes: bid price (scaled)  
                # 4 bytes: ask volume
                # 4 bytes: bid volume
                
                chunk = data[i:i+20]
                if len(chunk) != 20:
                    break
                
                # Unpack binary data (big-endian)
                time_offset, ask_scaled, bid_scaled, ask_vol, bid_vol = struct.unpack('>IIIII', chunk)
                
                # Berechne Zeitstempel
                timestamp = datetime.fromtimestamp((base_timestamp + time_offset) / 1000, tz=timezone.utc)
                
                # Skaliere Preise (Dukascopy verwendet Point-Skalierung)
                # Für EUR/USD: 5 Dezimalstellen
                scale_factor = 100000.0
                ask_price = ask_scaled / scale_factor
                bid_price = bid_scaled / scale_factor
                
                # Volume (kombiniert)
                volume = (ask_vol + bid_vol) / 1000000.0  # Skalierung
                
                tick = TickData(
                    timestamp=timestamp,
                    bid=bid_price,
                    ask=ask_price,
                    volume=volume
                )
                
                tick_data.append(tick)
                
            except Exception as e:
                self.logger.warning(f"Error parsing tick at offset {i}: {e}")
                continue
        
        return tick_data
    
    def _convert_ticks_to_ohlcv(self, tick_data: List[TickData], timeframe: str) -> List[OHLCVData]:
        """
        Konvertiert Tick-Daten zu OHLCV-Candles
        """
        if not tick_data:
            return []
        
        # Timeframe zu Sekunden
        timeframe_seconds = self._parse_timeframe(timeframe)
        
        ohlcv_data = []
        current_candle_start = None
        current_candle_data = []
        
        for tick in tick_data:
            # Berechne Candle-Start-Zeit
            timestamp_seconds = int(tick.timestamp.timestamp())
            candle_start = timestamp_seconds - (timestamp_seconds % timeframe_seconds)
            candle_start_dt = datetime.fromtimestamp(candle_start, tz=timezone.utc)
            
            # Neue Candle beginnen?
            if current_candle_start != candle_start:
                # Vorherige Candle abschließen
                if current_candle_data:
                    candle = self._create_ohlcv_candle(current_candle_start, current_candle_data)
                    ohlcv_data.append(candle)
                
                # Neue Candle beginnen
                current_candle_start = candle_start_dt
                current_candle_data = []
            
            current_candle_data.append(tick)
        
        # Letzte Candle abschließen
        if current_candle_data:
            candle = self._create_ohlcv_candle(current_candle_start, current_candle_data)
            ohlcv_data.append(candle)
        
        return ohlcv_data
    
    def _create_ohlcv_candle(self, timestamp: datetime, ticks: List[TickData]) -> OHLCVData:
        """
        Erstellt OHLCV-Candle aus Tick-Daten
        """
        if not ticks:
            raise ValueError("No ticks provided for candle creation")
        
        # Verwende Mid-Preise für OHLC
        mid_prices = [tick.mid_price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        
        return OHLCVData(
            timestamp=timestamp,
            open=mid_prices[0],
            high=max(mid_prices),
            low=min(mid_prices),
            close=mid_prices[-1],
            volume=sum(volumes)
        )
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """
        Parsed Timeframe zu Sekunden
        """
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return timeframe_map[timeframe]
    
    def _split_time_range_by_hours(self, start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Teilt Zeitraum in Stunden-Chunks für parallelen Download
        """
        ranges = []
        current = start.replace(minute=0, second=0, microsecond=0)
        
        while current < end:
            next_hour = current + timedelta(hours=1)
            range_end = min(next_hour, end)
            ranges.append((current, range_end))
            current = next_hour
        
        return ranges
    
    def validate_data_integrity(self, data: MarketData) -> Dict[str, Any]:
        """
        Validiert Datenintegrität für MarketData
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "ohlcv_validation": None,
            "tick_validation": None
        }
        
        # OHLCV-Validierung
        if data.ohlcv_data:
            ohlcv_result = self.validator.validate_ohlcv_data(data.ohlcv_data)
            results["ohlcv_validation"] = ohlcv_result
            
            if not ohlcv_result["valid"]:
                results["valid"] = False
                results["errors"].extend([f"OHLCV: {err}" for err in ohlcv_result["errors"]])
            
            results["warnings"].extend([f"OHLCV: {warn}" for warn in ohlcv_result["warnings"]])
        
        # Tick-Validierung
        if data.tick_data:
            tick_result = self.validator.validate_tick_data(data.tick_data)
            results["tick_validation"] = tick_result
            
            if not tick_result["valid"]:
                results["valid"] = False
                results["errors"].extend([f"Tick: {err}" for err in tick_result["errors"]])
            
            results["warnings"].extend([f"Tick: {warn}" for warn in tick_result["warnings"]])
        
        return results
    
    def get_available_symbols(self) -> List[str]:
        """
        Gibt verfügbare Symbole zurück
        """
        return list(self.symbol_mapping.keys())
    
    def get_data_info(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """
        Gibt Informationen über verfügbare Daten zurück
        """
        return {
            "symbol": symbol,
            "start": start,
            "end": end,
            "estimated_hours": int((end - start).total_seconds() / 3600),
            "supported": symbol in self.symbol_mapping,
            "parallel_workers": self.parallel_workers
        }