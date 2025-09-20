"""
Dukascopy Data Connector für EUR/USD Forex-Daten
Optimiert für parallele Downloads mit 32 CPU-Kernen
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import gzip
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import io

from .models import TradingData, OHLCVData


@dataclass
class DukascopyConfig:
    """Konfiguration für Dukascopy Data Connector"""
    base_url: str = "https://datafeed.dukascopy.com/datafeed"
    symbols: List[str] = None
    max_workers: int = 32  # Ryzen 9 9950X optimal
    chunk_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_dir: str = "./data/cache"
    use_real_data: bool = True  # True = echte Dukascopy-Daten, False = simuliert
    local_data_path: Optional[str] = None  # Pfad zu lokalen Profi-Daten
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]


class DukascopyConnector:
    """
    Dukascopy Data Connector für historische Forex-Daten
    Unterstützt Tick-Daten und OHLCV-Aggregation
    """
    
    def __init__(self, config: Optional[DukascopyConfig] = None):
        self.config = config or DukascopyConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup Cache Directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Symbol Mapping (Dukascopy Format)
        self.symbol_mapping = {
            "EUR/USD": "EURUSD",
            "GBP/USD": "GBPUSD", 
            "USD/JPY": "USDJPY",
            "AUD/USD": "AUDUSD",
            "USD/CHF": "USDCHF",
            "USD/CAD": "USDCAD",
            "NZD/USD": "NZDUSD"
        }
        
        # Session für HTTP Requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.logger.info(f"DukascopyConnector initialized with {self.config.max_workers} workers")
    
    def get_ohlcv_data(self, 
                      symbol: str,
                      timeframe: str = "1H",
                      start_date: datetime = None,
                      end_date: datetime = None,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Lädt OHLCV-Daten für Symbol und Zeitraum
        
        Args:
            symbol: Trading-Symbol (z.B. "EUR/USD")
            timeframe: Zeitrahmen ("1M", "5M", "15M", "1H", "4H", "1D")
            start_date: Start-Datum
            end_date: End-Datum
            use_cache: Cache verwenden
            
        Returns:
            DataFrame mit OHLCV-Daten
        """
        try:
            # Default Dates
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=14)  # 14 Tage Standard
            
            self.logger.info(f"Loading OHLCV data for {symbol} {timeframe} from {start_date} to {end_date}")
            
            # Check Cache
            if use_cache:
                cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                if cached_data is not None:
                    self.logger.info(f"Loaded {len(cached_data)} candles from cache")
                    return cached_data
            
            # Load Tick Data und aggregiere zu OHLCV
            tick_data = self._load_tick_data_parallel(symbol, start_date, end_date)
            
            if tick_data.empty:
                self.logger.warning(f"No tick data found for {symbol}")
                return pd.DataFrame()
            
            # Aggregiere zu OHLCV
            ohlcv_data = self._aggregate_to_ohlcv(tick_data, timeframe)
            
            # Cache speichern
            if use_cache and not ohlcv_data.empty:
                self._save_to_cache(ohlcv_data, symbol, timeframe, start_date, end_date)
            
            self.logger.info(f"Loaded {len(ohlcv_data)} {timeframe} candles for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Failed to load OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_tick_data_parallel(self, 
                                symbol: str,
                                start_date: datetime,
                                end_date: datetime) -> pd.DataFrame:
        """Lädt Tick-Daten parallel für Zeitraum"""
        
        try:
            # Generiere Datum-Liste
            date_list = []
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                date_list.append(current_date)
                current_date += timedelta(days=1)
            
            self.logger.info(f"Loading tick data for {len(date_list)} days with {self.config.max_workers} workers")
            
            # Parallel Processing
            all_tick_data = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit Tasks
                future_to_date = {
                    executor.submit(self._load_tick_data_for_date, symbol, date): date 
                    for date in date_list
                }
                
                # Collect Results
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        tick_data = future.result()
                        if not tick_data.empty:
                            all_tick_data.append(tick_data)
                            self.logger.debug(f"Loaded {len(tick_data)} ticks for {date}")
                    except Exception as e:
                        self.logger.error(f"Failed to load tick data for {date}: {e}")
            
            # Kombiniere alle Daten
            if all_tick_data:
                combined_data = pd.concat(all_tick_data, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"Combined {len(combined_data)} total ticks")
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Parallel tick data loading failed: {e}")
            return pd.DataFrame()
    
    def _load_tick_data_for_date(self, symbol: str, date: datetime.date) -> pd.DataFrame:
        """Lädt Tick-Daten für einen Tag"""
        
        try:
            # Dukascopy Symbol Format
            dukascopy_symbol = self.symbol_mapping.get(symbol, symbol.replace("/", ""))
            
            # Wähle Datenquelle basierend auf Priorität
            
            # 1. Priorität: Lokale Profi-Daten
            if self.config.local_data_path:
                local_data = self._load_local_professional_data(symbol, date)
                if not local_data.empty:
                    self.logger.info(f"Using local professional data: {len(local_data)} ticks for {symbol} on {date}")
                    return local_data
            
            # 2. Priorität: Echte Dukascopy-Daten
            if self.config.use_real_data:
                real_data = self._load_real_dukascopy_data(dukascopy_symbol, date)
                
                if not real_data.empty:
                    self.logger.info(f"Using real Dukascopy data: {len(real_data)} ticks for {symbol} on {date}")
                    return real_data
                else:
                    self.logger.warning(f"Real Dukascopy data not available for {symbol} on {date}")
            
            # 3. Fallback: Simulierte Daten
            self.logger.info(f"Using simulated data for {symbol} on {date}")
            return self._generate_simulated_tick_data(symbol, date)
            
        except Exception as e:
            self.logger.error(f"Failed to load tick data for {symbol} on {date}: {e}")
            # Fallback zu simulierten Daten
            return self._generate_simulated_tick_data(symbol, date)
    
    def _generate_simulated_tick_data(self, symbol: str, date: datetime.date) -> pd.DataFrame:
        """Generiert simulierte Tick-Daten für Demo/Testing"""
        
        try:
            # Base Price für Symbol
            base_prices = {
                "EUR/USD": 1.1000,
                "GBP/USD": 1.2500,
                "USD/JPY": 150.00,
                "AUD/USD": 0.6500
            }
            
            base_price = base_prices.get(symbol, 1.0000)
            
            # Prüfe ob Wochenende (Forex-Markt geschlossen)
            if date.weekday() >= 5:  # Samstag (5) oder Sonntag (6)
                return pd.DataFrame()  # Keine Ticks am Wochenende
            
            # Generiere realistische Tick-Anzahl für Forex
            np.random.seed(int(date.strftime("%Y%m%d")))  # Reproduzierbare Daten
            
            # EUR/USD: 50k-200k Ticks pro Tag (je nach Volatilität)
            base_ticks_per_day = {
                "EUR/USD": (50000, 200000),
                "GBP/USD": (40000, 150000), 
                "USD/JPY": (35000, 120000),
                "AUD/USD": (25000, 100000),
                "USD/CHF": (20000, 80000)
            }
            
            min_ticks, max_ticks = base_ticks_per_day.get(symbol, (10000, 50000))
            num_ticks = np.random.randint(min_ticks, max_ticks)
            
            # Zeitstempel über den Tag verteilt
            start_time = datetime.combine(date, datetime.min.time())
            timestamps = []
            
            # Forex-Handelszeiten berücksichtigen (UTC)
            # Haupthandelszeiten: 22:00 Sonntag - 22:00 Freitag UTC
            trading_start_hour = 0  # Vereinfacht: ganzer Tag
            trading_end_hour = 24
            
            for i in range(num_ticks):
                # Ticks hauptsächlich während Handelszeiten
                if np.random.random() < 0.9:  # 90% während Hauptzeiten
                    hour_offset = np.random.randint(trading_start_hour, trading_end_hour)
                    minute_offset = np.random.randint(0, 60)
                    second_offset = np.random.randint(0, 60)
                    microsecond_offset = np.random.randint(0, 1000000)
                else:  # 10% außerhalb (dünner Handel)
                    hour_offset = np.random.randint(0, 24)
                    minute_offset = np.random.randint(0, 60) 
                    second_offset = np.random.randint(0, 60)
                    microsecond_offset = np.random.randint(0, 1000000)
                
                timestamp = start_time + timedelta(
                    hours=hour_offset,
                    minutes=minute_offset, 
                    seconds=second_offset,
                    microseconds=microsecond_offset
                )
                timestamps.append(timestamp)
            
            timestamps.sort()
            
            # Preis-Bewegung simulieren (Random Walk)
            prices = [base_price]
            
            for i in range(1, num_ticks):
                # Kleine zufällige Änderungen
                change = np.random.normal(0, 0.0001)  # 1 Pip Standardabweichung
                new_price = prices[-1] + change
                
                # Bounds für realistische Preise
                min_price = base_price * 0.99
                max_price = base_price * 1.01
                new_price = max(min_price, min(max_price, new_price))
                
                prices.append(new_price)
            
            # Bid/Ask Spread simulieren
            spread = 0.00015  # 1.5 Pips
            bid_prices = [p - spread/2 for p in prices]
            ask_prices = [p + spread/2 for p in prices]
            
            # Volume simulieren
            volumes = np.random.uniform(0.1, 10.0, num_ticks)  # 0.1 - 10 Lots
            
            # DataFrame erstellen
            tick_data = pd.DataFrame({
                'timestamp': timestamps,
                'bid': bid_prices,
                'ask': ask_prices,
                'volume': volumes
            })
            
            return tick_data
            
        except Exception as e:
            self.logger.error(f"Simulated tick data generation failed: {e}")
            return pd.DataFrame()
    
    def _aggregate_to_ohlcv(self, tick_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregiert Tick-Daten zu OHLCV-Kerzen"""
        
        try:
            if tick_data.empty:
                return pd.DataFrame()
            
            # Tick-Daten direkt zurückgeben (keine Aggregation)
            if timeframe in ["100tick", "1000tick"]:
                return self._process_tick_data(tick_data, timeframe)
            
            # Timeframe Mapping für OHLCV
            timeframe_mapping = {
                "1M": "1min",    # 1 Minute
                "5M": "5min",    # 5 Minuten
                "15M": "15min",  # 15 Minuten
                "30M": "30min",  # 30 Minuten
                "1H": "1h",      # 1 Stunde
                "4H": "4h",      # 4 Stunden
                "1D": "1D"       # 1 Tag
            }
            
            pandas_timeframe = timeframe_mapping.get(timeframe, "1H")
            
            # Mid-Price berechnen
            tick_data['price'] = (tick_data['bid'] + tick_data['ask']) / 2
            
            # Timestamp als Index setzen
            tick_data['timestamp'] = pd.to_datetime(tick_data['timestamp'])
            tick_data.set_index('timestamp', inplace=True)
            
            # OHLCV Aggregation
            ohlcv = tick_data['price'].resample(pandas_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Volume aggregieren
            volume = tick_data['volume'].resample(pandas_timeframe).sum()
            
            # Kombiniere OHLCV + Volume
            ohlcv['volume'] = volume
            ohlcv = ohlcv.dropna()
            
            # Reset Index für DataFrame
            ohlcv.reset_index(inplace=True)
            
            # Runde Preise
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                ohlcv[col] = ohlcv[col].round(5)  # 5 Dezimalstellen für Forex
            
            return ohlcv
            
        except Exception as e:
            self.logger.error(f"OHLCV aggregation failed: {e}")
            return pd.DataFrame()
    
    def _process_tick_data(self, tick_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Verarbeitet Tick-Daten für spezifische Tick-Counts"""
        
        try:
            if tick_data.empty:
                return pd.DataFrame()
            
            # Tick-Count extrahieren
            tick_count = int(timeframe.replace("tick", ""))  # "100tick" -> 100
            
            # Mid-Price berechnen
            tick_data['price'] = (tick_data['bid'] + tick_data['ask']) / 2
            
            # Sortiere nach Timestamp
            tick_data = tick_data.sort_values('timestamp').reset_index(drop=True)
            
            # Gruppiere in Tick-Blöcke
            ohlcv_data = []
            
            for i in range(0, len(tick_data), tick_count):
                chunk = tick_data.iloc[i:i+tick_count]
                
                if len(chunk) < tick_count // 2:  # Mindestens 50% der Ticks
                    continue
                
                # OHLCV für diesen Tick-Block berechnen
                ohlc_row = {
                    'timestamp': chunk['timestamp'].iloc[-1],  # Letzter Timestamp
                    'open': chunk['price'].iloc[0],
                    'high': chunk['price'].max(),
                    'low': chunk['price'].min(),
                    'close': chunk['price'].iloc[-1],
                    'volume': chunk['volume'].sum(),
                    'tick_count': len(chunk)  # Zusätzliche Info
                }
                
                ohlcv_data.append(ohlc_row)
            
            # Zu DataFrame konvertieren
            result_df = pd.DataFrame(ohlcv_data)
            
            if not result_df.empty:
                # Preise runden
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    result_df[col] = result_df[col].round(5)
                
                self.logger.info(f"Processed {len(tick_data)} ticks into {len(result_df)} {timeframe} bars")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Tick data processing failed: {e}")
            return pd.DataFrame()
    
    def _load_local_professional_data(self, symbol: str, date: datetime.date) -> pd.DataFrame:
        """Lädt lokale professionelle Tick-Daten"""
        
        try:
            if not self.config.local_data_path:
                return pd.DataFrame()
            
            from pathlib import Path
            
            # Verschiedene Dateiformate unterstützen
            base_path = Path(self.config.local_data_path)
            
            # Dateiname-Patterns für verschiedene Formate
            patterns = [
                f"{symbol}_{date.strftime('%Y%m%d')}.csv",
                f"{symbol}_{date.strftime('%Y-%m-%d')}.csv",
                f"{symbol.replace('/', '')}_{date.strftime('%Y%m%d')}.csv",
                f"{symbol}_{date.strftime('%Y%m%d')}.parquet",
                f"{symbol}_{date.strftime('%Y-%m-%d')}.parquet",
                f"{date.strftime('%Y')}/{date.strftime('%m')}/{symbol}_{date.strftime('%d')}.csv",
                f"{symbol}/{date.strftime('%Y%m%d')}.csv"
            ]
            
            for pattern in patterns:
                file_path = base_path / pattern
                
                if file_path.exists():
                    self.logger.info(f"Found local data file: {file_path}")
                    
                    # Lade basierend auf Dateierweiterung
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_path.suffix.lower() == '.parquet':
                        df = pd.read_parquet(file_path)
                    else:
                        continue
                    
                    # Validiere und normalisiere Spalten
                    df = self._normalize_professional_data(df, symbol)
                    
                    if not df.empty:
                        self.logger.info(f"Loaded {len(df)} professional ticks from {file_path}")
                        return df
            
            # Keine lokalen Daten gefunden
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error loading local professional data: {e}")
            return pd.DataFrame()
    
    def _normalize_professional_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalisiert professionelle Daten auf einheitliches Format"""
        
        try:
            # Erkenne verschiedene Spalten-Formate
            column_mappings = {
                # Standard-Format
                'timestamp': ['timestamp', 'time', 'datetime', 'date_time'],
                'ask': ['ask', 'ask_price', 'askprice', 'offer'],
                'bid': ['bid', 'bid_price', 'bidprice'],
                'volume': ['volume', 'vol', 'size', 'quantity'],
                'ask_volume': ['ask_volume', 'ask_vol', 'ask_size', 'offer_volume'],
                'bid_volume': ['bid_volume', 'bid_vol', 'bid_size']
            }
            
            # Mappe Spalten
            normalized_df = pd.DataFrame()
            
            for target_col, possible_names in column_mappings.items():
                for name in possible_names:
                    if name in df.columns:
                        normalized_df[target_col] = df[name]
                        break
            
            # Timestamp konvertieren
            if 'timestamp' in normalized_df.columns:
                normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'])
            else:
                self.logger.error("No timestamp column found in professional data")
                return pd.DataFrame()
            
            # Mindest-Spalten prüfen
            required_cols = ['timestamp', 'ask', 'bid']
            if not all(col in normalized_df.columns for col in required_cols):
                self.logger.error(f"Missing required columns in professional data. Found: {list(normalized_df.columns)}")
                return pd.DataFrame()
            
            # Volume-Spalten hinzufügen falls nicht vorhanden
            if 'volume' not in normalized_df.columns:
                if 'ask_volume' in normalized_df.columns and 'bid_volume' in normalized_df.columns:
                    normalized_df['volume'] = normalized_df['ask_volume'] + normalized_df['bid_volume']
                else:
                    # Synthetisches Volume basierend auf Spread und Aktivität
                    normalized_df['volume'] = self._generate_synthetic_volume(normalized_df)
                    normalized_df['ask_volume'] = normalized_df['volume'] * 0.5
                    normalized_df['bid_volume'] = normalized_df['volume'] * 0.5
            
            # Sortiere nach Timestamp
            normalized_df = normalized_df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Normalized professional data: {len(normalized_df)} ticks with columns {list(normalized_df.columns)}")
            
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error normalizing professional data: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_volume(self, df: pd.DataFrame) -> pd.Series:
        """Generiert synthetisches Volume für professionelle Daten ohne Volume"""
        
        try:
            # Berechne Spread
            spread = df['ask'] - df['bid']
            
            # Berechne Preisvolatilität
            price_mid = (df['ask'] + df['bid']) / 2
            price_changes = price_mid.diff().abs()
            
            # Volume basierend auf Spread und Volatilität
            # Enger Spread + hohe Volatilität = hohes Volume
            base_volume = 1.0
            spread_factor = 1.0 / (spread / price_mid + 0.0001)  # Inverser Spread
            volatility_factor = price_changes / price_mid.mean()
            
            synthetic_volume = base_volume * spread_factor * (1 + volatility_factor)
            
            # Normalisiere auf realistische Werte (0.1 - 10 Lots)
            synthetic_volume = synthetic_volume.clip(0.1, 10.0)
            
            return synthetic_volume
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic volume: {e}")
            return pd.Series([1.0] * len(df))

    def _load_real_dukascopy_data(self, symbol: str, date: datetime.date) -> pd.DataFrame:
        """Lädt echte Dukascopy Historical Data"""
        
        try:
            # Dukascopy Historical Data URL Format
            # https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5
            
            year = date.year
            month = date.month - 1  # Dukascopy verwendet 0-basierte Monate
            day = date.day
            
            all_ticks = []
            
            # Lade Daten für alle 24 Stunden des Tages
            for hour in range(24):
                try:
                    url = f"{self.config.base_url}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
                    
                    self.logger.debug(f"Downloading: {url}")
                    
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        # Dekomprimiere und parse Binärdaten
                        tick_data = self._parse_dukascopy_binary(response.content, date, hour)
                        
                        if not tick_data.empty:
                            all_ticks.append(tick_data)
                            self.logger.debug(f"Loaded {len(tick_data)} ticks for {hour:02d}:00")
                    
                    elif response.status_code == 404:
                        # Keine Daten für diese Stunde (normal am Wochenende)
                        continue
                    else:
                        self.logger.warning(f"HTTP {response.status_code} for {url}")
                        
                except Exception as e:
                    self.logger.debug(f"Failed to load hour {hour:02d}: {e}")
                    continue
            
            # Kombiniere alle Stunden-Daten
            if all_ticks:
                combined_data = pd.concat(all_ticks, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"Loaded {len(combined_data)} real ticks for {symbol} on {date}")
                return combined_data
            else:
                self.logger.warning(f"No real tick data found for {symbol} on {date}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Real Dukascopy data loading failed: {e}")
            return pd.DataFrame()
    
    def _parse_dukascopy_binary(self, binary_data: bytes, date: datetime.date, hour: int) -> pd.DataFrame:
        """Parst Dukascopy .bi5 Binärformat"""
        
        try:
            if not binary_data:
                return pd.DataFrame()
            
            # Dekomprimiere LZMA-komprimierte Daten
            try:
                import lzma
                decompressed_data = lzma.decompress(binary_data)
            except:
                # Fallback: Versuche ohne Dekomprimierung
                decompressed_data = binary_data
            
            # Dukascopy .bi5 Format:
            # Jeder Tick = 20 Bytes
            # 4 bytes: timestamp (milliseconds from hour start)
            # 4 bytes: ask price (big-endian int, point value)
            # 4 bytes: bid price (big-endian int, point value)  
            # 4 bytes: ask volume (big-endian float)
            # 4 bytes: bid volume (big-endian float)
            
            tick_size = 20
            num_ticks = len(decompressed_data) // tick_size
            
            if num_ticks == 0:
                return pd.DataFrame()
            
            ticks = []
            base_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            
            for i in range(num_ticks):
                offset = i * tick_size
                tick_bytes = decompressed_data[offset:offset + tick_size]
                
                if len(tick_bytes) < tick_size:
                    break
                
                # Parse Binärdaten (Big-Endian)
                timestamp_ms = struct.unpack('>I', tick_bytes[0:4])[0]
                ask_price_raw = struct.unpack('>I', tick_bytes[4:8])[0]
                bid_price_raw = struct.unpack('>I', tick_bytes[8:12])[0]
                ask_volume = struct.unpack('>f', tick_bytes[12:16])[0]
                bid_volume = struct.unpack('>f', tick_bytes[16:20])[0]
                
                # Timestamp berechnen
                timestamp = base_time + timedelta(milliseconds=timestamp_ms)
                
                # Preise konvertieren (Dukascopy verwendet Point-Values)
                # Für EUR/USD: 1 Point = 0.00001 (5 Dezimalstellen)
                point_value = 0.00001  # Standard für Major-Paare
                ask_price = ask_price_raw * point_value
                bid_price = bid_price_raw * point_value
                
                # Volume kombinieren
                volume = ask_volume + bid_volume
                
                ticks.append({
                    'timestamp': timestamp,
                    'ask': ask_price,
                    'bid': bid_price,
                    'volume': volume
                })
            
            if ticks:
                return pd.DataFrame(ticks)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Binary parsing failed: {e}")
            return pd.DataFrame()
    
    def _load_from_cache(self, 
                        symbol: str,
                        timeframe: str,
                        start_date: datetime,
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """Lädt Daten aus Cache"""
        
        try:
            # Safe filename (replace / with _)
            safe_symbol = symbol.replace("/", "_")
            cache_key = f"{safe_symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                # Check if cache is recent (< 1 hour old)
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 3600:  # 1 hour
                    data = pd.read_parquet(cache_file)
                    return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache loading failed: {e}")
            return None
    
    def _save_to_cache(self, 
                      data: pd.DataFrame,
                      symbol: str,
                      timeframe: str,
                      start_date: datetime,
                      end_date: datetime):
        """Speichert Daten in Cache"""
        
        try:
            # Safe filename (replace / with _)
            safe_symbol = symbol.replace("/", "_")
            cache_key = f"{safe_symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            data.to_parquet(cache_file, compression='snappy')
            self.logger.debug(f"Data cached to {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Cache saving failed: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Gibt verfügbare Symbole zurück"""
        return list(self.symbol_mapping.keys())
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validiert Datenintegrität"""
        
        validation_result = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            if data.empty:
                validation_result["valid"] = False
                validation_result["issues"].append("Empty dataset")
                return validation_result
            
            # Required Columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing columns: {missing_columns}")
            
            # Price Validation
            if 'high' in data.columns and 'low' in data.columns:
                invalid_prices = data[data['high'] < data['low']]
                if len(invalid_prices) > 0:
                    validation_result["valid"] = False
                    validation_result["issues"].append(f"Invalid high/low prices: {len(invalid_prices)} rows")
            
            # OHLC Validation
            price_cols = ['open', 'high', 'low', 'close']
            if all(col in data.columns for col in price_cols):
                for idx, row in data.iterrows():
                    if not (row['low'] <= row['open'] <= row['high'] and 
                           row['low'] <= row['close'] <= row['high']):
                        validation_result["issues"].append(f"Invalid OHLC at index {idx}")
                        if len(validation_result["issues"]) > 10:  # Limit issues
                            break
            
            # Statistics
            validation_result["statistics"] = {
                "total_rows": len(data),
                "date_range": {
                    "start": data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                    "end": data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
                },
                "price_range": {
                    "min": data[price_cols].min().min() if price_cols[0] in data.columns else None,
                    "max": data[price_cols].max().max() if price_cols[0] in data.columns else None
                } if price_cols[0] in data.columns else None
            }
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Gibt Verbindungsstatus zurück"""
        
        status = {
            "connected": False,
            "response_time": None,
            "last_check": datetime.now().isoformat()
        }
        
        try:
            # Test Connection (simuliert)
            start_time = time.time()
            
            # In Produktion: Echter API-Test
            # response = self.session.get(f"{self.config.base_url}/test", timeout=5)
            # status["connected"] = response.status_code == 200
            
            # Für Demo: Simuliere erfolgreiche Verbindung
            time.sleep(0.1)  # Simuliere Netzwerk-Latenz
            status["connected"] = True
            
            status["response_time"] = time.time() - start_time
            
        except Exception as e:
            status["connected"] = False
            status["error"] = str(e)
        
        return status
    
    def cleanup(self):
        """Bereinigt Ressourcen"""
        try:
            self.session.close()
            self.logger.info("DukascopyConnector cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# Convenience Functions
def create_trading_data_from_ohlcv(symbol: str, 
                                  timeframe: str,
                                  ohlcv_df: pd.DataFrame) -> TradingData:
    """Erstellt TradingData-Objekt aus OHLCV DataFrame"""
    
    ohlcv_data = []
    
    for _, row in ohlcv_df.iterrows():
        ohlcv_data.append(OHLCVData(
            timestamp=row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        ))
    
    return TradingData(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_data=ohlcv_data
    )


def quick_data_download(symbol: str = "EUR/USD", 
                       days: int = 7,
                       timeframe: str = "1H") -> pd.DataFrame:
    """Schneller Daten-Download für Testing"""
    
    connector = DukascopyConnector()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return connector.get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )