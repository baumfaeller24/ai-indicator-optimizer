"""
Multimodal Data Processing Pipeline - Optimiert für High-End Hardware
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import logging
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import torch
import cv2

from .models import OHLCVData, IndicatorData, MarketData, TickData


@dataclass
class ProcessingConfig:
    """Konfiguration für Data Processing"""
    cpu_workers: int = 32
    gpu_acceleration: bool = True
    chart_width: int = 1024
    chart_height: int = 768
    chart_dpi: int = 100
    enable_caching: bool = True
    cache_size_mb: int = 1024


class IndicatorCalculator:
    """
    Hochperformante technische Indikator-Berechnung mit Parallelisierung
    """
    
    def __init__(self, cpu_workers: int = 32):
        self.cpu_workers = cpu_workers
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_indicators(self, ohlcv_data: List[OHLCVData]) -> IndicatorData:
        """
        Berechnet alle Standard-Indikatoren parallel
        """
        if not ohlcv_data:
            return IndicatorData()
        
        # Konvertiere zu Pandas für effiziente Berechnungen
        df = self._ohlcv_to_dataframe(ohlcv_data)
        
        # Parallele Indikator-Berechnung
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            futures = {
                'rsi': executor.submit(self._calculate_rsi, df, 14),
                'macd': executor.submit(self._calculate_macd, df, 12, 26, 9),
                'bollinger': executor.submit(self._calculate_bollinger_bands, df, 20, 2),
                'sma': executor.submit(self._calculate_sma_multiple, df, [20, 50, 200]),
                'ema': executor.submit(self._calculate_ema_multiple, df, [12, 26]),
                'stochastic': executor.submit(self._calculate_stochastic, df, 14, 3),
                'atr': executor.submit(self._calculate_atr, df, 14),
                'adx': executor.submit(self._calculate_adx, df, 14)
            }
            
            # Sammle Ergebnisse
            results = {}
            for indicator, future in futures.items():
                try:
                    results[indicator] = future.result()
                    self.logger.debug(f"Calculated {indicator}")
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator}: {e}")
                    results[indicator] = None
        
        return IndicatorData(
            rsi=results.get('rsi'),
            macd=results.get('macd'),
            bollinger=results.get('bollinger'),
            sma=results.get('sma'),
            ema=results.get('ema'),
            stochastic=results.get('stochastic'),
            atr=results.get('atr'),
            adx=results.get('adx')
        )
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List[OHLCVData]) -> pd.DataFrame:
        """Konvertiert OHLCV-Daten zu Pandas DataFrame"""
        data = []
        for candle in ohlcv_data:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Berechnet Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0).tolist()
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Berechnet MACD (Moving Average Convergence Divergence)"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.fillna(0.0).tolist(),
            'signal': signal_line.fillna(0.0).tolist(),
            'histogram': histogram.fillna(0.0).tolist()
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
        """Berechnet Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band.fillna(df['close']).tolist(),
            'middle': sma.fillna(df['close']).tolist(),
            'lower': lower_band.fillna(df['close']).tolist()
        }
    
    def _calculate_sma_multiple(self, df: pd.DataFrame, periods: List[int]) -> Dict[int, List[float]]:
        """Berechnet mehrere Simple Moving Averages"""
        result = {}
        for period in periods:
            sma = df['close'].rolling(window=period).mean()
            result[period] = sma.fillna(df['close']).tolist()
        return result
    
    def _calculate_ema_multiple(self, df: pd.DataFrame, periods: List[int]) -> Dict[int, List[float]]:
        """Berechnet mehrere Exponential Moving Averages"""
        result = {}
        for period in periods:
            ema = df['close'].ewm(span=period).mean()
            result[period] = ema.fillna(df['close']).tolist()
        return result
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
        """Berechnet Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent.fillna(50.0).tolist(),
            'd': d_percent.fillna(50.0).tolist()
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Berechnet Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = pd.Series(true_range).rolling(window=period).mean()
        
        return atr.fillna(0.0).tolist()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Berechnet Average Directional Index"""
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
        
        # True Range
        tr = self._calculate_atr(df, 1)  # ATR mit Periode 1 = True Range
        tr_series = pd.Series(tr)
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_series.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_series.rolling(window=period).mean())
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(25.0).tolist()


class ChartRenderer:
    """
    GPU-beschleunigte Chart-Generierung für multimodale AI-Eingabe
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # GPU-Beschleunigung prüfen
        self.use_gpu = config.gpu_acceleration and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU rendering")
        
        # Matplotlib Style Setup
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend für Threading
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def generate_candlestick_chart(self, ohlcv_data: List[OHLCVData], indicators: IndicatorData) -> Image.Image:
        """
        Generiert Candlestick-Chart mit Indikatoren
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.config.chart_width/100, self.config.chart_height/100), 
                                dpi=self.config.chart_dpi, facecolor='black')
        
        # Daten vorbereiten
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in ohlcv_data])
        
        # Hauptchart: Candlesticks + Moving Averages
        self._plot_candlesticks(axes[0], df)
        self._plot_moving_averages(axes[0], df, indicators)
        self._plot_bollinger_bands(axes[0], df, indicators)
        
        # Subchart 1: RSI + Stochastic
        self._plot_oscillators(axes[1], df, indicators)
        
        # Subchart 2: MACD + Volume
        self._plot_macd_volume(axes[2], df, indicators)
        
        # Styling
        for ax in axes:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='gray')
        
        plt.tight_layout()
        
        # Konvertiere zu PIL Image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        # Konvertiere RGBA zu RGB
        buf = buf[:, :, :3]
        
        plt.close(fig)
        
        # GPU-Beschleunigung für Post-Processing
        if self.use_gpu:
            buf = self._gpu_enhance_image(buf)
        
        return Image.fromarray(buf)
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plottet Candlestick-Chart"""
        for i, row in df.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            ax.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.8, width=0.8)
            
            # Wicks
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        
        ax.set_title('EUR/USD Candlestick Chart', color='white', fontsize=14)
        ax.set_ylabel('Price', color='white')
    
    def _plot_moving_averages(self, ax, df: pd.DataFrame, indicators: IndicatorData):
        """Plottet Moving Averages"""
        if indicators.sma:
            for period, values in indicators.sma.items():
                if len(values) == len(df):
                    ax.plot(range(len(values)), values, label=f'SMA {period}', linewidth=2)
        
        if indicators.ema:
            for period, values in indicators.ema.items():
                if len(values) == len(df):
                    ax.plot(range(len(values)), values, label=f'EMA {period}', linewidth=2, linestyle='--')
        
        ax.legend(loc='upper left')
    
    def _plot_bollinger_bands(self, ax, df: pd.DataFrame, indicators: IndicatorData):
        """Plottet Bollinger Bands"""
        if indicators.bollinger:
            upper = indicators.bollinger.get('upper', [])
            middle = indicators.bollinger.get('middle', [])
            lower = indicators.bollinger.get('lower', [])
            
            if len(upper) == len(df):
                x = range(len(upper))
                ax.plot(x, upper, color='cyan', alpha=0.7, linewidth=1)
                ax.plot(x, middle, color='yellow', alpha=0.7, linewidth=1)
                ax.plot(x, lower, color='cyan', alpha=0.7, linewidth=1)
                ax.fill_between(x, upper, lower, alpha=0.1, color='cyan')
    
    def _plot_oscillators(self, ax, df: pd.DataFrame, indicators: IndicatorData):
        """Plottet RSI und Stochastic"""
        ax2 = ax.twinx()
        
        # RSI
        if indicators.rsi and len(indicators.rsi) == len(df):
            ax.plot(range(len(indicators.rsi)), indicators.rsi, color='orange', label='RSI', linewidth=2)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_ylabel('RSI', color='white')
        
        # Stochastic
        if indicators.stochastic:
            k_values = indicators.stochastic.get('k', [])
            d_values = indicators.stochastic.get('d', [])
            
            if len(k_values) == len(df):
                ax2.plot(range(len(k_values)), k_values, color='purple', label='%K', linewidth=1)
                ax2.plot(range(len(d_values)), d_values, color='pink', label='%D', linewidth=1)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Stochastic', color='white')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_macd_volume(self, ax, df: pd.DataFrame, indicators: IndicatorData):
        """Plottet MACD und Volume"""
        ax2 = ax.twinx()
        
        # MACD
        if indicators.macd:
            macd_line = indicators.macd.get('macd', [])
            signal_line = indicators.macd.get('signal', [])
            histogram = indicators.macd.get('histogram', [])
            
            if len(macd_line) == len(df):
                x = range(len(macd_line))
                ax.plot(x, macd_line, color='blue', label='MACD', linewidth=2)
                ax.plot(x, signal_line, color='red', label='Signal', linewidth=2)
                ax.bar(x, histogram, color='gray', alpha=0.6, label='Histogram')
                ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
                ax.set_ylabel('MACD', color='white')
        
        # Volume
        volumes = [candle.volume for candle in df.itertuples()]
        ax2.bar(range(len(volumes)), volumes, color='lightblue', alpha=0.3, label='Volume')
        ax2.set_ylabel('Volume', color='white')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _gpu_enhance_image(self, image_array: np.ndarray) -> np.ndarray:
        """GPU-beschleunigte Bildverbesserung"""
        try:
            # Konvertiere zu PyTorch Tensor
            tensor = torch.from_numpy(image_array).float().to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # BHWC -> BCHW, normalize
            
            # Einfache Bildverbesserung (Kontrast, Schärfe)
            enhanced = torch.clamp(tensor * 1.1 + 0.05, 0, 1)  # Kontrast + Helligkeit
            
            # Zurück zu NumPy
            enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced = (enhanced * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"GPU enhancement failed, using original: {e}")
            return image_array
    
    def generate_multiple_timeframes(self, market_data: MarketData, indicators: IndicatorData) -> List[Image.Image]:
        """
        Generiert Charts für verschiedene Zeitrahmen parallel
        """
        timeframes = ['1m', '5m', '15m', '1h']
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for tf in timeframes:
                # Simuliere verschiedene Zeitrahmen (würde echte Daten-Aggregation nutzen)
                future = executor.submit(self.generate_candlestick_chart, market_data.ohlcv_data, indicators)
                futures.append(future)
            
            charts = []
            for future in futures:
                try:
                    chart = future.result()
                    charts.append(chart)
                except Exception as e:
                    self.logger.error(f"Chart generation failed: {e}")
        
        return charts


class MultimodalDatasetBuilder:
    """
    Erstellt multimodale Datasets für MiniCPM-4.1-8B Training
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_training_sample(self, market_data: MarketData, indicators: IndicatorData, 
                             chart_images: List[Image.Image]) -> Dict[str, Any]:
        """
        Erstellt einen multimodalen Training-Sample
        """
        # Numerische Features normalisieren
        numerical_features = self._extract_numerical_features(market_data, indicators)
        normalized_features = self._normalize_features(numerical_features)
        
        # Chart-Images preprocessing
        processed_images = self._preprocess_images(chart_images)
        
        # Text-Beschreibungen generieren
        text_descriptions = self._generate_text_descriptions(market_data, indicators)
        
        return {
            'numerical_features': normalized_features,
            'chart_images': processed_images,
            'text_descriptions': text_descriptions,
            'metadata': {
                'symbol': market_data.symbol,
                'timeframe': market_data.timeframe,
                'data_points': len(market_data.ohlcv_data),
                'timestamp_range': (
                    market_data.ohlcv_data[0].timestamp if market_data.ohlcv_data else None,
                    market_data.ohlcv_data[-1].timestamp if market_data.ohlcv_data else None
                )
            }
        }
    
    def _extract_numerical_features(self, market_data: MarketData, indicators: IndicatorData) -> np.ndarray:
        """Extrahiert numerische Features"""
        features = []
        
        # OHLCV Features
        for candle in market_data.ohlcv_data[-100:]:  # Letzte 100 Candles
            features.extend([
                candle.open, candle.high, candle.low, candle.close, candle.volume,
                candle.body_size, candle.upper_shadow, candle.lower_shadow
            ])
        
        # Indikator Features (letzte Werte)
        if indicators.rsi:
            features.extend(indicators.rsi[-10:])  # Letzte 10 RSI-Werte
        
        if indicators.macd and 'macd' in indicators.macd:
            features.extend(indicators.macd['macd'][-10:])
        
        # Weitere Indikatoren...
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalisiert Features für ML"""
        # Z-Score Normalisierung
        mean = np.mean(features)
        std = np.std(features)
        
        if std > 0:
            normalized = (features - mean) / std
        else:
            normalized = features
        
        return normalized
    
    def _preprocess_images(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Preprocessing für Chart-Images"""
        processed = []
        
        for img in images:
            # Resize für MiniCPM Input
            img_resized = img.resize((224, 224))  # Standard Vision Model Input
            
            # Zu NumPy Array
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            processed.append(img_array)
        
        return processed
    
    def _generate_text_descriptions(self, market_data: MarketData, indicators: IndicatorData) -> List[str]:
        """Generiert Text-Beschreibungen für multimodale Eingabe"""
        descriptions = []
        
        if not market_data.ohlcv_data:
            return descriptions
        
        latest_candle = market_data.ohlcv_data[-1]
        
        # Basis-Beschreibung
        trend = "bullish" if latest_candle.is_bullish else "bearish"
        descriptions.append(f"Latest {market_data.symbol} candle shows {trend} movement")
        
        # RSI-Beschreibung
        if indicators.rsi and len(indicators.rsi) > 0:
            rsi_value = indicators.rsi[-1]
            if rsi_value > 70:
                descriptions.append("RSI indicates overbought conditions")
            elif rsi_value < 30:
                descriptions.append("RSI indicates oversold conditions")
            else:
                descriptions.append("RSI shows neutral momentum")
        
        # MACD-Beschreibung
        if indicators.macd and 'macd' in indicators.macd:
            macd_values = indicators.macd['macd']
            signal_values = indicators.macd['signal']
            
            if len(macd_values) > 0 and len(signal_values) > 0:
                if macd_values[-1] > signal_values[-1]:
                    descriptions.append("MACD shows bullish crossover signal")
                else:
                    descriptions.append("MACD shows bearish crossover signal")
        
        return descriptions


class DataProcessor:
    """
    Hauptklasse für multimodale Datenverarbeitung
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.indicator_calculator = IndicatorCalculator(self.config.cpu_workers)
        self.chart_renderer = ChartRenderer(self.config)
        self.dataset_builder = MultimodalDatasetBuilder(self.config)
        self.logger = logging.getLogger(__name__)
    
    def process_market_data(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Vollständige multimodale Verarbeitung von Marktdaten
        """
        self.logger.info(f"Processing {len(market_data.ohlcv_data)} data points for {market_data.symbol}")
        
        # 1. Indikatoren berechnen
        indicators = self.indicator_calculator.calculate_all_indicators(market_data.ohlcv_data)
        
        # 2. Charts generieren
        chart_images = self.chart_renderer.generate_multiple_timeframes(market_data, indicators)
        
        # 3. Multimodales Dataset erstellen
        training_sample = self.dataset_builder.create_training_sample(
            market_data, indicators, chart_images
        )
        
        return {
            'indicators': indicators,
            'chart_images': chart_images,
            'training_sample': training_sample,
            'processing_stats': {
                'data_points': len(market_data.ohlcv_data),
                'indicators_calculated': 8,
                'charts_generated': len(chart_images),
                'features_extracted': len(training_sample['numerical_features'])
            }
        }
    
    def calculate_indicators(self, ohlcv: List[OHLCVData]) -> IndicatorData:
        """Backward compatibility"""
        return self.indicator_calculator.calculate_all_indicators(ohlcv)
    
    def generate_chart_images(self, ohlcv: List[OHLCVData], indicators: IndicatorData) -> List[Image.Image]:
        """Backward compatibility"""
        market_data = MarketData(symbol="TEMP", timeframe="1m", ohlcv_data=ohlcv)
        return self.chart_renderer.generate_multiple_timeframes(market_data, indicators)
    
    def create_multimodal_dataset(self, data: MarketData) -> Dict[str, Any]:
        """Backward compatibility"""
        return self.process_market_data(data)