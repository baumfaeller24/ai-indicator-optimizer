"""
Training Dataset Builder für MiniCPM Fine-Tuning
Erstellt multimodale Trading-Datasets aus historischen Daten
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import random
from datetime import datetime, timedelta
import io
import base64
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# from ..data.dukascopy_connector import DukascopyConnector
# from ..data.models import TradingData, OHLCVData
# from ..indicators.indicator_calculator import IndicatorCalculator
# from ..visualization.chart_renderer import ChartRenderer


@dataclass
class DatasetSample:
    """Einzelnes Dataset-Sample für Training"""
    chart_image: Union[Image.Image, str]  # PIL Image oder Pfad
    numerical_data: Dict[str, Any]        # Indikator-Werte
    pattern_label: str                    # Pattern-Typ
    pattern_description: str              # Beschreibung
    market_context: Dict[str, Any]        # Market-Kontext
    confidence_score: float               # Label-Qualität
    metadata: Dict[str, Any]              # Zusätzliche Metadaten


@dataclass
class PatternTemplate:
    """Template für Pattern-Generierung"""
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    visual_markers: List[Dict[str, Any]]
    indicator_ranges: Dict[str, Tuple[float, float]]


class PatternDetector:
    """
    Automatische Pattern-Erkennung in historischen Daten
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern Templates
        self.pattern_templates = self._load_pattern_templates()
    
    def _load_pattern_templates(self) -> Dict[str, PatternTemplate]:
        """Lädt Pattern-Templates"""
        
        templates = {
            "double_top": PatternTemplate(
                pattern_type="double_top",
                description="Double top reversal pattern with two peaks at similar levels",
                conditions={
                    "min_peaks": 2,
                    "peak_similarity": 0.02,  # 2% Toleranz
                    "valley_depth": 0.01,     # 1% Mindest-Tal
                    "time_separation": 10     # Mindest-Kerzen zwischen Peaks
                },
                visual_markers=[
                    {"type": "peak", "color": "red", "size": 8},
                    {"type": "valley", "color": "blue", "size": 6}
                ],
                indicator_ranges={
                    "RSI": (60, 80),  # Überkauft bei Peaks
                    "MACD": (-0.01, 0.01)  # Neutral/Divergenz
                }
            ),
            
            "double_bottom": PatternTemplate(
                pattern_type="double_bottom",
                description="Double bottom reversal pattern with two lows at similar levels",
                conditions={
                    "min_lows": 2,
                    "low_similarity": 0.02,
                    "peak_height": 0.01,
                    "time_separation": 10
                },
                visual_markers=[
                    {"type": "low", "color": "green", "size": 8},
                    {"type": "peak", "color": "orange", "size": 6}
                ],
                indicator_ranges={
                    "RSI": (20, 40),  # Überverkauft bei Lows
                    "MACD": (-0.01, 0.01)
                }
            ),
            
            "head_shoulders": PatternTemplate(
                pattern_type="head_shoulders",
                description="Head and shoulders reversal pattern with three peaks",
                conditions={
                    "peaks_count": 3,
                    "head_prominence": 0.015,  # Head 1.5% höher
                    "shoulder_similarity": 0.01,
                    "neckline_slope": 0.005
                },
                visual_markers=[
                    {"type": "head", "color": "red", "size": 10},
                    {"type": "shoulder", "color": "orange", "size": 8},
                    {"type": "neckline", "color": "blue", "style": "dashed"}
                ],
                indicator_ranges={
                    "RSI": (50, 85),
                    "MACD": (-0.02, 0.02)
                }
            ),
            
            "triangle": PatternTemplate(
                pattern_type="triangle",
                description="Triangle consolidation pattern with converging trend lines",
                conditions={
                    "min_touches": 4,  # 2 pro Linie
                    "convergence_angle": (5, 45),  # Grad
                    "duration": (15, 50),  # Kerzen
                    "volume_decline": True
                },
                visual_markers=[
                    {"type": "trendline", "color": "purple", "style": "solid"},
                    {"type": "breakout", "color": "yellow", "size": 12}
                ],
                indicator_ranges={
                    "RSI": (40, 60),  # Neutral
                    "ATR": (0.0005, 0.002)  # Niedrige Volatilität
                }
            ),
            
            "support_resistance": PatternTemplate(
                pattern_type="support_resistance",
                description="Clear support or resistance level with multiple touches",
                conditions={
                    "min_touches": 3,
                    "level_tolerance": 0.001,  # 0.1% Toleranz
                    "bounce_strength": 0.005,  # 0.5% Mindest-Bounce
                    "time_span": (20, 100)
                },
                visual_markers=[
                    {"type": "level", "color": "red", "style": "solid", "width": 2},
                    {"type": "touch", "color": "blue", "size": 6}
                ],
                indicator_ranges={
                    "RSI": (25, 75),
                    "Volume": (0.8, 1.5)  # Relative zu Durchschnitt
                }
            ),
            
            "breakout": PatternTemplate(
                pattern_type="breakout",
                description="Price breakout from consolidation with volume confirmation",
                conditions={
                    "consolidation_duration": (10, 30),
                    "breakout_strength": 0.01,  # 1% Mindest-Breakout
                    "volume_increase": 1.5,     # 50% mehr Volume
                    "follow_through": 0.005     # 0.5% Follow-through
                },
                visual_markers=[
                    {"type": "consolidation", "color": "gray", "alpha": 0.3},
                    {"type": "breakout_candle", "color": "lime", "size": 10},
                    {"type": "volume_spike", "color": "orange"}
                ],
                indicator_ranges={
                    "RSI": (45, 75),
                    "MACD": (0, 0.02),  # Bullish momentum
                    "Volume": (1.2, 3.0)
                }
            )
        }
        
        return templates
    
    def detect_patterns(self, 
                       ohlcv_data: pd.DataFrame,
                       indicators: Dict[str, Any],
                       lookback_window: int = 100) -> List[Dict[str, Any]]:
        """Detektiert Patterns in OHLCV-Daten"""
        
        detected_patterns = []
        
        try:
            # Für jedes Pattern-Template
            for pattern_type, template in self.pattern_templates.items():
                
                # Suche Pattern in sliding window
                for i in range(lookback_window, len(ohlcv_data) - 10):
                    window_data = ohlcv_data.iloc[i-lookback_window:i+10]
                    
                    # Pattern-spezifische Detection
                    if pattern_type == "double_top":
                        pattern_info = self._detect_double_top(window_data, template)
                    elif pattern_type == "double_bottom":
                        pattern_info = self._detect_double_bottom(window_data, template)
                    elif pattern_type == "head_shoulders":
                        pattern_info = self._detect_head_shoulders(window_data, template)
                    elif pattern_type == "triangle":
                        pattern_info = self._detect_triangle(window_data, template)
                    elif pattern_type == "support_resistance":
                        pattern_info = self._detect_support_resistance(window_data, template)
                    elif pattern_type == "breakout":
                        pattern_info = self._detect_breakout(window_data, template)
                    else:
                        continue
                    
                    if pattern_info and pattern_info["confidence"] > 0.6:
                        pattern_info.update({
                            "pattern_type": pattern_type,
                            "start_index": i - lookback_window,
                            "end_index": i + 10,
                            "data_window": window_data,
                            "template": template
                        })
                        detected_patterns.append(pattern_info)
            
            self.logger.info(f"Detected {len(detected_patterns)} patterns")
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return []
    
    def _detect_double_top(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Double Top Pattern"""
        try:
            highs = data['high'].values
            
            # Finde Peaks
            peaks = []
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 2:
                return None
            
            # Prüfe auf ähnliche Peak-Höhen
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1_idx, peak1_price = peaks[i]
                    peak2_idx, peak2_price = peaks[j]
                    
                    # Zeit-Separation prüfen
                    if abs(peak2_idx - peak1_idx) < template.conditions["time_separation"]:
                        continue
                    
                    # Preis-Ähnlichkeit prüfen
                    price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                    if price_diff <= template.conditions["peak_similarity"]:
                        
                        # Tal zwischen Peaks prüfen
                        valley_start = min(peak1_idx, peak2_idx)
                        valley_end = max(peak1_idx, peak2_idx)
                        valley_low = data['low'].iloc[valley_start:valley_end].min()
                        
                        valley_depth = (min(peak1_price, peak2_price) - valley_low) / min(peak1_price, peak2_price)
                        
                        if valley_depth >= template.conditions["valley_depth"]:
                            return {
                                "confidence": min(0.9, 0.6 + (1 - price_diff) * 0.3),
                                "peaks": [(peak1_idx, peak1_price), (peak2_idx, peak2_price)],
                                "valley": (valley_start + np.argmin(data['low'].iloc[valley_start:valley_end]), valley_low),
                                "description": f"Double top at {peak1_price:.4f} and {peak2_price:.4f}"
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Double top detection failed: {e}")
            return None
    
    def _detect_double_bottom(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Double Bottom Pattern"""
        try:
            lows = data['low'].values
            
            # Finde Lows
            bottoms = []
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                    lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                    bottoms.append((i, lows[i]))
            
            if len(bottoms) < 2:
                return None
            
            # Ähnliche Bottom-Levels finden
            for i in range(len(bottoms) - 1):
                for j in range(i + 1, len(bottoms)):
                    bottom1_idx, bottom1_price = bottoms[i]
                    bottom2_idx, bottom2_price = bottoms[j]
                    
                    if abs(bottom2_idx - bottom1_idx) < template.conditions["time_separation"]:
                        continue
                    
                    price_diff = abs(bottom1_price - bottom2_price) / max(bottom1_price, bottom2_price)
                    if price_diff <= template.conditions["low_similarity"]:
                        
                        # Peak zwischen Bottoms
                        peak_start = min(bottom1_idx, bottom2_idx)
                        peak_end = max(bottom1_idx, bottom2_idx)
                        peak_high = data['high'].iloc[peak_start:peak_end].max()
                        
                        peak_height = (peak_high - max(bottom1_price, bottom2_price)) / max(bottom1_price, bottom2_price)
                        
                        if peak_height >= template.conditions["peak_height"]:
                            return {
                                "confidence": min(0.9, 0.6 + (1 - price_diff) * 0.3),
                                "bottoms": [(bottom1_idx, bottom1_price), (bottom2_idx, bottom2_price)],
                                "peak": (peak_start + np.argmax(data['high'].iloc[peak_start:peak_end]), peak_high),
                                "description": f"Double bottom at {bottom1_price:.4f} and {bottom2_price:.4f}"
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Double bottom detection failed: {e}")
            return None
    
    def _detect_head_shoulders(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Head and Shoulders Pattern"""
        try:
            highs = data['high'].values
            
            # Finde alle Peaks
            peaks = []
            for i in range(3, len(highs) - 3):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2] and
                    highs[i] > highs[i-3] and highs[i] > highs[i+3]):
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 3:
                return None
            
            # Suche Head-Shoulders Formation
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Head muss höchster Peak sein
                head_prominence = (head[1] - max(left_shoulder[1], right_shoulder[1])) / head[1]
                if head_prominence < template.conditions["head_prominence"]:
                    continue
                
                # Shoulders ähnliche Höhe
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1])
                if shoulder_diff <= template.conditions["shoulder_similarity"]:
                    
                    return {
                        "confidence": min(0.9, 0.7 + head_prominence * 2),
                        "left_shoulder": left_shoulder,
                        "head": head,
                        "right_shoulder": right_shoulder,
                        "description": f"Head and shoulders: {left_shoulder[1]:.4f} - {head[1]:.4f} - {right_shoulder[1]:.4f}"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Head and shoulders detection failed: {e}")
            return None
    
    def _detect_triangle(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Triangle Pattern"""
        try:
            # Vereinfachte Triangle-Detection
            highs = data['high'].values
            lows = data['low'].values
            
            # Prüfe auf konvergierende Trend-Linien
            if len(data) < 20:
                return None
            
            # Obere Trendlinie (fallend)
            upper_peaks = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    upper_peaks.append((i, highs[i]))
            
            # Untere Trendlinie (steigend)
            lower_peaks = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    lower_peaks.append((i, lows[i]))
            
            if len(upper_peaks) >= 2 and len(lower_peaks) >= 2:
                # Prüfe Konvergenz
                upper_slope = (upper_peaks[-1][1] - upper_peaks[0][1]) / (upper_peaks[-1][0] - upper_peaks[0][0])
                lower_slope = (lower_peaks[-1][1] - lower_peaks[0][1]) / (lower_peaks[-1][0] - lower_peaks[0][0])
                
                # Triangle: obere Linie fällt, untere steigt (oder beide konvergieren)
                if upper_slope < 0 and lower_slope > 0:
                    return {
                        "confidence": 0.7,
                        "upper_line": upper_peaks,
                        "lower_line": lower_peaks,
                        "description": f"Triangle pattern with {len(upper_peaks)} upper and {len(lower_peaks)} lower touches"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Triangle detection failed: {e}")
            return None
    
    def _detect_support_resistance(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Support/Resistance Levels"""
        try:
            # Finde häufige Preis-Levels
            all_prices = np.concatenate([data['high'].values, data['low'].values, data['close'].values])
            
            # Runde Preise für Level-Clustering
            price_precision = 4  # 4 Dezimalstellen für Forex
            rounded_prices = np.round(all_prices, price_precision)
            
            # Zähle Häufigkeiten
            unique_prices, counts = np.unique(rounded_prices, return_counts=True)
            
            # Finde Levels mit mindestens 3 Touches
            significant_levels = []
            for price, count in zip(unique_prices, counts):
                if count >= template.conditions["min_touches"]:
                    
                    # Prüfe Bounce-Stärke
                    touches = []
                    for i, row in data.iterrows():
                        if (abs(row['high'] - price) <= template.conditions["level_tolerance"] * price or
                            abs(row['low'] - price) <= template.conditions["level_tolerance"] * price):
                            touches.append(i)
                    
                    if len(touches) >= template.conditions["min_touches"]:
                        significant_levels.append({
                            "level": price,
                            "touches": len(touches),
                            "touch_indices": touches
                        })
            
            if significant_levels:
                # Wähle stärkstes Level
                best_level = max(significant_levels, key=lambda x: x["touches"])
                
                return {
                    "confidence": min(0.9, 0.5 + best_level["touches"] * 0.1),
                    "level": best_level["level"],
                    "touches": best_level["touches"],
                    "touch_indices": best_level["touch_indices"],
                    "description": f"Support/Resistance at {best_level['level']:.4f} with {best_level['touches']} touches"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Support/Resistance detection failed: {e}")
            return None
    
    def _detect_breakout(self, data: pd.DataFrame, template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """Detektiert Breakout Pattern"""
        try:
            if len(data) < 20:
                return None
            
            # Suche Konsolidierung (niedrige Volatilität)
            for i in range(10, len(data) - 5):
                consolidation_window = data.iloc[i-10:i]
                
                # Prüfe Volatilität in Konsolidierung
                price_range = (consolidation_window['high'].max() - consolidation_window['low'].min())
                avg_price = consolidation_window['close'].mean()
                volatility = price_range / avg_price
                
                if volatility < 0.02:  # Niedrige Volatilität (2%)
                    
                    # Prüfe Breakout
                    breakout_candles = data.iloc[i:i+5]
                    consolidation_high = consolidation_window['high'].max()
                    consolidation_low = consolidation_window['low'].min()
                    
                    for j, (_, candle) in enumerate(breakout_candles.iterrows()):
                        # Bullish Breakout
                        if candle['close'] > consolidation_high * (1 + template.conditions["breakout_strength"]):
                            
                            # Volume-Bestätigung (falls verfügbar)
                            volume_confirmed = True
                            if 'volume' in data.columns:
                                avg_volume = consolidation_window['volume'].mean()
                                volume_confirmed = candle['volume'] > avg_volume * template.conditions["volume_increase"]
                            
                            if volume_confirmed:
                                return {
                                    "confidence": 0.8,
                                    "breakout_type": "bullish",
                                    "consolidation_range": (consolidation_low, consolidation_high),
                                    "breakout_price": candle['close'],
                                    "breakout_index": i + j,
                                    "description": f"Bullish breakout above {consolidation_high:.4f} to {candle['close']:.4f}"
                                }
                        
                        # Bearish Breakout
                        elif candle['close'] < consolidation_low * (1 - template.conditions["breakout_strength"]):
                            
                            volume_confirmed = True
                            if 'volume' in data.columns:
                                avg_volume = consolidation_window['volume'].mean()
                                volume_confirmed = candle['volume'] > avg_volume * template.conditions["volume_increase"]
                            
                            if volume_confirmed:
                                return {
                                    "confidence": 0.8,
                                    "breakout_type": "bearish",
                                    "consolidation_range": (consolidation_low, consolidation_high),
                                    "breakout_price": candle['close'],
                                    "breakout_index": i + j,
                                    "description": f"Bearish breakout below {consolidation_low:.4f} to {candle['close']:.4f}"
                                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Breakout detection failed: {e}")
            return None


class TrainingDatasetBuilder:
    """
    Haupt-Klasse für Training Dataset Erstellung
    Kombiniert historische Daten, Pattern-Detection und Chart-Rendering
    """
    
    def __init__(self, 
                 dukascopy_connector: DukascopyConnector,
                 indicator_calculator: IndicatorCalculator,
                 chart_renderer: ChartRenderer):
        
        self.dukascopy_connector = dukascopy_connector
        self.indicator_calculator = indicator_calculator
        self.chart_renderer = chart_renderer
        self.pattern_detector = PatternDetector()
        self.logger = logging.getLogger(__name__)
        
        # Dataset Configuration
        self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        self.timeframes = ["1H", "4H", "1D"]
        self.lookback_days = 30  # Für Pattern-Detection
        
    def build_training_dataset(self, 
                              num_samples: int = 1000,
                              output_dir: str = "./datasets/training",
                              num_workers: int = 8) -> List[DatasetSample]:
        """Erstellt komplettes Training-Dataset"""
        
        self.logger.info(f"Building training dataset with {num_samples} samples")
        
        # Setup Output Directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Sammle historische Daten
        historical_data = self._collect_historical_data()
        
        # Generiere Dataset-Samples
        dataset_samples = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            # Erstelle Tasks für parallel processing
            samples_per_symbol = num_samples // len(self.symbols)
            
            for symbol in self.symbols:
                if symbol in historical_data:
                    future = executor.submit(
                        self._generate_samples_for_symbol,
                        symbol,
                        historical_data[symbol],
                        samples_per_symbol,
                        output_dir
                    )
                    futures.append(future)
            
            # Sammle Ergebnisse
            for future in tqdm(futures, desc="Processing symbols"):
                try:
                    symbol_samples = future.result()
                    dataset_samples.extend(symbol_samples)
                except Exception as e:
                    self.logger.error(f"Sample generation failed: {e}")
        
        # Shuffle Dataset
        random.shuffle(dataset_samples)
        
        # Speichere Dataset-Metadaten
        self._save_dataset_metadata(dataset_samples, output_dir)
        
        self.logger.info(f"Training dataset created with {len(dataset_samples)} samples")
        return dataset_samples
    
    def _collect_historical_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Sammelt historische Daten für alle Symbole und Timeframes"""
        
        historical_data = {}
        
        for symbol in self.symbols:
            historical_data[symbol] = {}
            
            for timeframe in self.timeframes:
                try:
                    # Lade Daten für letzten Monat
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)
                    
                    ohlcv_data = self.dukascopy_connector.get_ohlcv_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if len(ohlcv_data) > 50:  # Mindest-Daten
                        historical_data[symbol][timeframe] = ohlcv_data
                        self.logger.info(f"Loaded {len(ohlcv_data)} candles for {symbol} {timeframe}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load data for {symbol} {timeframe}: {e}")
        
        return historical_data
    
    def _generate_samples_for_symbol(self, 
                                   symbol: str,
                                   symbol_data: Dict[str, pd.DataFrame],
                                   num_samples: int,
                                   output_dir: str) -> List[DatasetSample]:
        """Generiert Samples für ein Symbol"""
        
        samples = []
        
        for timeframe, ohlcv_data in symbol_data.items():
            
            # Berechne Indikatoren
            indicators = self.indicator_calculator.calculate_all_indicators(
                TradingData(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_data=[
                        OHLCVData(
                            timestamp=row['timestamp'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row.get('volume', 0)
                        ) for _, row in ohlcv_data.iterrows()
                    ]
                )
            )
            
            # Detektiere Patterns
            detected_patterns = self.pattern_detector.detect_patterns(
                ohlcv_data, indicators, lookback_window=50
            )
            
            # Erstelle Samples aus detected patterns
            for pattern_info in detected_patterns:
                try:
                    sample = self._create_sample_from_pattern(
                        symbol, timeframe, pattern_info, indicators, output_dir
                    )
                    if sample:
                        samples.append(sample)
                        
                        if len(samples) >= num_samples // len(self.timeframes):
                            break
                            
                except Exception as e:
                    self.logger.error(f"Sample creation failed: {e}")
            
            if len(samples) >= num_samples // len(self.timeframes):
                break
        
        return samples
    
    def _create_sample_from_pattern(self, 
                                  symbol: str,
                                  timeframe: str,
                                  pattern_info: Dict[str, Any],
                                  indicators: Dict[str, Any],
                                  output_dir: str) -> Optional[DatasetSample]:
        """Erstellt Dataset-Sample aus Pattern-Info"""
        
        try:
            # Chart-Daten für Rendering
            chart_data = pattern_info["data_window"]
            
            # Erstelle Chart-Image
            chart_image = self.chart_renderer.render_trading_chart(
                TradingData(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_data=[
                        OHLCVData(
                            timestamp=row['timestamp'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row.get('volume', 0)
                        ) for _, row in chart_data.iterrows()
                    ]
                ),
                indicators=indicators,
                width=448,
                height=448
            )
            
            # Pattern-spezifische Marker hinzufügen
            chart_image = self._add_pattern_markers(chart_image, pattern_info)
            
            # Speichere Chart-Image
            image_filename = f"{symbol}_{timeframe}_{pattern_info['pattern_type']}_{int(time.time())}.png"
            image_path = Path(output_dir) / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            chart_image.save(image_path)
            
            # Extrahiere relevante Indikator-Werte
            numerical_data = self._extract_numerical_features(indicators, chart_data)
            
            # Market Context
            market_context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": chart_data.iloc[-1]['timestamp'].isoformat(),
                "trend": self._determine_trend(chart_data),
                "volatility": self._calculate_volatility(chart_data)
            }
            
            # Pattern Description
            pattern_description = self._generate_pattern_description(pattern_info, numerical_data)
            
            return DatasetSample(
                chart_image=str(image_path),
                numerical_data=numerical_data,
                pattern_label=pattern_info["pattern_type"],
                pattern_description=pattern_description,
                market_context=market_context,
                confidence_score=pattern_info["confidence"],
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "pattern_info": pattern_info,
                    "creation_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sample creation failed: {e}")
            return None
    
    def _add_pattern_markers(self, chart_image: Image.Image, pattern_info: Dict[str, Any]) -> Image.Image:
        """Fügt Pattern-spezifische Marker zum Chart hinzu"""
        
        try:
            draw = ImageDraw.Draw(chart_image)
            
            pattern_type = pattern_info["pattern_type"]
            
            # Pattern-spezifische Marker
            if pattern_type == "double_top" and "peaks" in pattern_info:
                for peak_idx, peak_price in pattern_info["peaks"]:
                    # Berechne Pixel-Position (vereinfacht)
                    x = int((peak_idx / 100) * chart_image.width)  # Approximation
                    y = int(chart_image.height * 0.2)  # Oberer Bereich
                    
                    # Zeichne Peak-Marker
                    draw.ellipse([x-5, y-5, x+5, y+5], fill="red", outline="darkred")
                    draw.text((x+10, y-10), "Peak", fill="red")
            
            elif pattern_type == "double_bottom" and "bottoms" in pattern_info:
                for bottom_idx, bottom_price in pattern_info["bottoms"]:
                    x = int((bottom_idx / 100) * chart_image.width)
                    y = int(chart_image.height * 0.8)  # Unterer Bereich
                    
                    draw.ellipse([x-5, y-5, x+5, y+5], fill="green", outline="darkgreen")
                    draw.text((x+10, y+10), "Bottom", fill="green")
            
            elif pattern_type == "support_resistance" and "level" in pattern_info:
                # Zeichne horizontale Linie für S/R Level
                y = int(chart_image.height * 0.5)  # Mitte (vereinfacht)
                draw.line([0, y, chart_image.width, y], fill="blue", width=2)
                draw.text((10, y-20), f"S/R: {pattern_info['level']:.4f}", fill="blue")
            
            return chart_image
            
        except Exception as e:
            self.logger.error(f"Adding pattern markers failed: {e}")
            return chart_image
    
    def _extract_numerical_features(self, indicators: Dict[str, Any], chart_data: pd.DataFrame) -> Dict[str, Any]:
        """Extrahiert numerische Features für Training"""
        
        features = {}
        
        try:
            # Letzte Indikator-Werte
            if 'RSI' in indicators and len(indicators['RSI']) > 0:
                features['RSI'] = indicators['RSI'][-1]
            
            if 'MACD' in indicators:
                macd_data = indicators['MACD']
                if 'macd' in macd_data and len(macd_data['macd']) > 0:
                    features['MACD'] = {
                        'macd': macd_data['macd'][-1],
                        'signal': macd_data['signal'][-1] if 'signal' in macd_data else 0,
                        'histogram': macd_data['histogram'][-1] if 'histogram' in macd_data else 0
                    }
            
            if 'SMA_20' in indicators and len(indicators['SMA_20']) > 0:
                features['SMA_20'] = indicators['SMA_20'][-1]
            
            if 'SMA_50' in indicators and len(indicators['SMA_50']) > 0:
                features['SMA_50'] = indicators['SMA_50'][-1]
            
            if 'BollingerBands' in indicators:
                bb_data = indicators['BollingerBands']
                if 'upper' in bb_data and len(bb_data['upper']) > 0:
                    features['BollingerBands'] = {
                        'upper': bb_data['upper'][-1],
                        'middle': bb_data['middle'][-1] if 'middle' in bb_data else 0,
                        'lower': bb_data['lower'][-1] if 'lower' in bb_data else 0
                    }
            
            # Preis-Features
            latest_candle = chart_data.iloc[-1]
            features['Price'] = {
                'open': latest_candle['open'],
                'high': latest_candle['high'],
                'low': latest_candle['low'],
                'close': latest_candle['close']
            }
            
            # Volatilität
            features['ATR'] = self._calculate_atr(chart_data)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
        
        return features
    
    def _determine_trend(self, chart_data: pd.DataFrame) -> str:
        """Bestimmt Trend-Richtung"""
        try:
            if len(chart_data) < 10:
                return "neutral"
            
            # Einfache Trend-Bestimmung über SMA
            short_sma = chart_data['close'].rolling(5).mean().iloc[-1]
            long_sma = chart_data['close'].rolling(10).mean().iloc[-1]
            
            if short_sma > long_sma * 1.001:
                return "bullish"
            elif short_sma < long_sma * 0.999:
                return "bearish"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"
    
    def _calculate_volatility(self, chart_data: pd.DataFrame) -> float:
        """Berechnet Volatilität"""
        try:
            if len(chart_data) < 5:
                return 0.0
            
            returns = chart_data['close'].pct_change().dropna()
            return float(returns.std())
            
        except Exception:
            return 0.0
    
    def _calculate_atr(self, chart_data: pd.DataFrame, period: int = 14) -> float:
        """Berechnet Average True Range"""
        try:
            if len(chart_data) < period:
                return 0.0
            
            high_low = chart_data['high'] - chart_data['low']
            high_close = np.abs(chart_data['high'] - chart_data['close'].shift())
            low_close = np.abs(chart_data['low'] - chart_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr) if not np.isnan(atr) else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_pattern_description(self, pattern_info: Dict[str, Any], numerical_data: Dict[str, Any]) -> str:
        """Generiert Pattern-Beschreibung für Training"""
        
        pattern_type = pattern_info["pattern_type"]
        confidence = pattern_info["confidence"]
        
        base_description = pattern_info.get("description", f"{pattern_type} pattern detected")
        
        # Füge Indikator-Kontext hinzu
        context_parts = [base_description]
        
        if "RSI" in numerical_data:
            rsi = numerical_data["RSI"]
            if rsi > 70:
                context_parts.append("RSI indicates overbought conditions")
            elif rsi < 30:
                context_parts.append("RSI indicates oversold conditions")
            else:
                context_parts.append(f"RSI at {rsi:.1f} shows neutral momentum")
        
        if "MACD" in numerical_data:
            macd_data = numerical_data["MACD"]
            if macd_data["macd"] > macd_data["signal"]:
                context_parts.append("MACD shows bullish momentum")
            else:
                context_parts.append("MACD shows bearish momentum")
        
        # Trading-Empfehlung basierend auf Pattern
        if pattern_type in ["double_bottom", "head_shoulders_inverse"]:
            context_parts.append("Consider bullish reversal opportunity")
        elif pattern_type in ["double_top", "head_shoulders"]:
            context_parts.append("Consider bearish reversal opportunity")
        elif pattern_type == "breakout":
            context_parts.append("Monitor for continuation or reversal")
        
        context_parts.append(f"Pattern confidence: {confidence:.1%}")
        
        return ". ".join(context_parts) + "."
    
    def _save_dataset_metadata(self, dataset_samples: List[DatasetSample], output_dir: str):
        """Speichert Dataset-Metadaten"""
        
        metadata = {
            "total_samples": len(dataset_samples),
            "creation_time": datetime.now().isoformat(),
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "pattern_distribution": {},
            "confidence_stats": {}
        }
        
        # Pattern-Verteilung
        pattern_counts = {}
        confidences = []
        
        for sample in dataset_samples:
            pattern = sample.pattern_label
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            confidences.append(sample.confidence_score)
        
        metadata["pattern_distribution"] = pattern_counts
        metadata["confidence_stats"] = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences)
        }
        
        # Speichere Metadaten
        metadata_file = Path(output_dir) / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Dataset metadata saved to {metadata_file}")
    
    def load_dataset_samples(self, dataset_dir: str) -> List[Dict[str, Any]]:
        """Lädt Dataset-Samples für Training"""
        
        try:
            # Lade Metadaten
            metadata_file = Path(dataset_dir) / "dataset_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Lade Sample-Daten
            samples = []
            images_dir = Path(dataset_dir) / "images"
            
            for image_file in images_dir.glob("*.png"):
                # Parse Filename für Metadaten
                filename_parts = image_file.stem.split("_")
                if len(filename_parts) >= 3:
                    symbol = filename_parts[0]
                    timeframe = filename_parts[1]
                    pattern_type = filename_parts[2]
                    
                    # Erstelle Sample-Dict für Training
                    sample = {
                        "chart_image": str(image_file),
                        "pattern_label": pattern_type,
                        "pattern_description": f"{pattern_type} pattern in {symbol} {timeframe}",
                        "numerical_data": {},  # Würde aus separater Datei geladen
                        "market_context": {
                            "symbol": symbol,
                            "timeframe": timeframe
                        }
                    }
                    samples.append(sample)
            
            self.logger.info(f"Loaded {len(samples)} dataset samples from {dataset_dir}")
            return samples
            
        except Exception as e:
            self.logger.error(f"Loading dataset samples failed: {e}")
            return []