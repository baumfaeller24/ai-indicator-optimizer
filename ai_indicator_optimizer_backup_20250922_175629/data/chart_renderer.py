#!/usr/bin/env python3
"""
🧩 BAUSTEIN A3: Chart Renderer
Chart-Generierung für Vision-Pipeline mit MEGA-DATASET Integration

Features:
- GPU-beschleunigte Chart-Generierung
- Multi-Timeframe Candlestick-Charts
- Integration mit 62.2M Ticks MEGA-DATASET
- Optimiert für Vision-Analyse
- Batch-Processing für 250+ Charts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import io
from PIL import Image
import logging
import time
from dataclasses import dataclass
import seaborn as sns

# GPU-Acceleration (falls verfügbar)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class ChartConfig:
    """Konfiguration für Chart-Rendering"""
    width: int = 12
    height: int = 8
    dpi: int = 150
    style: str = "dark_background"
    use_gpu: bool = True
    save_format: str = "PNG"
    compression_level: int = 6
    
    # Candlestick-Styling
    bullish_color: str = "#00ff88"
    bearish_color: str = "#ff4444"
    wick_color: str = "#ffffff"
    background_color: str = "#1a1a1a"
    grid_color: str = "#333333"
    text_color: str = "#ffffff"
    
    # Technical Indicators
    sma_colors: List[str] = None
    volume_color: str = "#666666"
    
    def __post_init__(self):
        if self.sma_colors is None:
            self.sma_colors = ["#ffaa00", "#00aaff", "#aa00ff"]


class ChartRenderer:
    """
    🧩 BAUSTEIN A3: Chart Renderer
    
    Generiert Candlestick-Charts für Vision-Analyse:
    - GPU-beschleunigte Rendering
    - Multi-Timeframe-Support
    - MEGA-DATASET-Integration
    - Vision-optimierte Ausgabe
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize Chart Renderer
        
        Args:
            config: Chart-Konfiguration
        """
        self.config = config or ChartConfig()
        
        # GPU-Verfügbarkeit prüfen
        self.use_gpu = self.config.use_gpu and GPU_AVAILABLE
        
        # Matplotlib-Setup
        self._setup_matplotlib()
        
        # Performance Tracking
        self.charts_generated = 0
        self.total_render_time = 0.0
        self.gpu_accelerated_operations = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ChartRenderer initialized: GPU={'✅' if self.use_gpu else '❌'}")
    
    def _setup_matplotlib(self):
        """Setup Matplotlib für optimale Chart-Qualität"""
        # Style setzen
        plt.style.use(self.config.style)
        
        # Font-Einstellungen
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.labelcolor': self.config.text_color,
            'axes.edgecolor': self.config.text_color,
            'xtick.color': self.config.text_color,
            'ytick.color': self.config.text_color,
            'text.color': self.config.text_color,
            'figure.facecolor': self.config.background_color,
            'axes.facecolor': self.config.background_color,
            'savefig.facecolor': self.config.background_color,
            'grid.color': self.config.grid_color,
            'grid.alpha': 0.3
        })
    
    def generate_candlestick_chart(
        self,
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]],
        indicators: Optional[Dict[str, Any]] = None,
        title: str = "EUR/USD Chart",
        timeframe: str = "1h",
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generiere Candlestick-Chart
        
        Args:
            ohlcv_data: OHLCV-Daten
            indicators: Technische Indikatoren
            title: Chart-Titel
            timeframe: Timeframe für Optimierung
            save_path: Optional Speicher-Pfad
            
        Returns:
            PIL Image des Charts
        """
        start_time = time.time()
        
        try:
            # Daten normalisieren
            df = self._normalize_ohlcv_data(ohlcv_data)
            
            if len(df) == 0:
                raise ValueError("No OHLCV data provided")
            
            # Chart erstellen
            fig, (ax_price, ax_volume) = plt.subplots(
                2, 1, 
                figsize=(self.config.width, self.config.height),
                gridspec_kw={'height_ratios': [3, 1]},
                dpi=self.config.dpi
            )
            
            # Candlesticks zeichnen
            self._draw_candlesticks(ax_price, df)
            
            # Technische Indikatoren hinzufügen
            if indicators:
                self._draw_indicators(ax_price, df, indicators)
            
            # Volume zeichnen
            self._draw_volume(ax_volume, df)
            
            # Chart-Formatierung
            self._format_chart(ax_price, ax_volume, title, timeframe)
            
            # Chart zu Image konvertieren
            chart_image = self._fig_to_image(fig)
            
            # Optional speichern
            if save_path:
                chart_image.save(save_path, format=self.config.save_format)
            
            # Cleanup
            plt.close(fig)
            
            # Performance Tracking
            render_time = time.time() - start_time
            self.total_render_time += render_time
            self.charts_generated += 1
            
            self.logger.debug(f"Chart generated in {render_time:.3f}s")
            
            return chart_image
            
        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")
            raise
    
    def _normalize_ohlcv_data(self, data: Union[pd.DataFrame, pl.DataFrame, List[Dict]]) -> pd.DataFrame:
        """Normalisiere OHLCV-Daten zu Pandas DataFrame"""
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Erforderliche Spalten prüfen
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Timestamp-Spalte behandeln
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Fallback: Index als Datetime
            df['datetime'] = pd.date_range(start='2025-01-01', periods=len(df), freq='1H')
        
        # Volume hinzufügen falls nicht vorhanden
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(100, 1000, len(df))  # Mock Volume
        
        # Sortieren nach Zeit
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def _draw_candlesticks(self, ax: plt.Axes, df: pd.DataFrame):
        """Zeichne Candlesticks"""
        if self.use_gpu and len(df) > 1000:
            # GPU-beschleunigte Berechnung für große Datasets
            self._draw_candlesticks_gpu(ax, df)
        else:
            # Standard CPU-Implementierung
            self._draw_candlesticks_cpu(ax, df)
    
    def _draw_candlesticks_cpu(self, ax: plt.Axes, df: pd.DataFrame):
        """CPU-basierte Candlestick-Zeichnung"""
        for i, row in df.iterrows():
            x = i
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Farbe bestimmen
            color = self.config.bullish_color if close_price >= open_price else self.config.bearish_color
            
            # Wick (High-Low Linie)
            ax.plot([x, x], [low_price, high_price], 
                   color=self.config.wick_color, linewidth=1, alpha=0.8)
            
            # Body (Open-Close Rechteck)
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                rect = Rectangle((x - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, alpha=0.8, edgecolor=color)
                ax.add_patch(rect)
            else:
                # Doji (gleicher Open/Close)
                ax.plot([x - 0.3, x + 0.3], [open_price, open_price],
                       color=self.config.wick_color, linewidth=2)
    
    def _draw_candlesticks_gpu(self, ax: plt.Axes, df: pd.DataFrame):
        """GPU-beschleunigte Candlestick-Zeichnung"""
        try:
            # Daten zu GPU übertragen
            opens = cp.array(df['open'].values)
            highs = cp.array(df['high'].values)
            lows = cp.array(df['low'].values)
            closes = cp.array(df['close'].values)
            
            # GPU-Berechnungen
            body_heights = cp.abs(closes - opens)
            body_bottoms = cp.minimum(opens, closes)
            is_bullish = closes >= opens
            
            # Zurück zu CPU für Matplotlib
            body_heights_cpu = cp.asnumpy(body_heights)
            body_bottoms_cpu = cp.asnumpy(body_bottoms)
            is_bullish_cpu = cp.asnumpy(is_bullish)
            
            self.gpu_accelerated_operations += 1
            
            # Zeichnen mit vorberechneten Werten
            for i in range(len(df)):
                x = i
                color = self.config.bullish_color if is_bullish_cpu[i] else self.config.bearish_color
                
                # Wick
                ax.plot([x, x], [df.iloc[i]['low'], df.iloc[i]['high']], 
                       color=self.config.wick_color, linewidth=1, alpha=0.8)
                
                # Body
                if body_heights_cpu[i] > 0:
                    rect = Rectangle((x - 0.3, body_bottoms_cpu[i]), 0.6, body_heights_cpu[i],
                                   facecolor=color, alpha=0.8, edgecolor=color)
                    ax.add_patch(rect)
                    
        except Exception as e:
            self.logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            self._draw_candlesticks_cpu(ax, df)
    
    def _draw_indicators(self, ax: plt.Axes, df: pd.DataFrame, indicators: Dict[str, Any]):
        """Zeichne technische Indikatoren"""
        try:
            # SMA-Linien
            sma_periods = [5, 20, 50]
            for i, period in enumerate(sma_periods):
                sma_col = f'sma_{period}'
                if sma_col in df.columns:
                    color = self.config.sma_colors[i % len(self.config.sma_colors)]
                    ax.plot(range(len(df)), df[sma_col], 
                           color=color, linewidth=2, alpha=0.8, label=f'SMA {period}')
            
            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                ax.fill_between(range(len(df)), df['bb_upper'], df['bb_lower'],
                               alpha=0.1, color='blue', label='Bollinger Bands')
                ax.plot(range(len(df)), df['bb_upper'], color='blue', alpha=0.5, linewidth=1)
                ax.plot(range(len(df)), df['bb_lower'], color='blue', alpha=0.5, linewidth=1)
            
            # Support/Resistance Levels
            if 'support' in indicators:
                for level in indicators['support']:
                    ax.axhline(y=level, color='green', linestyle='--', alpha=0.6, linewidth=1)
            
            if 'resistance' in indicators:
                for level in indicators['resistance']:
                    ax.axhline(y=level, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Legende hinzufügen
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
            
        except Exception as e:
            self.logger.warning(f"Indicator drawing failed: {e}")
    
    def _draw_volume(self, ax: plt.Axes, df: pd.DataFrame):
        """Zeichne Volume-Bars"""
        try:
            volumes = df['volume'].values
            colors = [self.config.bullish_color if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                     else self.config.bearish_color for i in range(len(df))]
            
            ax.bar(range(len(df)), volumes, color=colors, alpha=0.6, width=0.8)
            ax.set_ylabel('Volume', color=self.config.text_color)
            
        except Exception as e:
            self.logger.warning(f"Volume drawing failed: {e}")
    
    def _format_chart(self, ax_price: plt.Axes, ax_volume: plt.Axes, title: str, timeframe: str):
        """Formatiere Chart-Layout"""
        # Titel
        ax_price.set_title(f'{title} ({timeframe.upper()})', 
                          fontsize=14, color=self.config.text_color, pad=20)
        
        # Y-Achse Labels
        ax_price.set_ylabel('Price', color=self.config.text_color)
        
        # Grid
        ax_price.grid(True, alpha=0.3, color=self.config.grid_color)
        ax_volume.grid(True, alpha=0.3, color=self.config.grid_color)
        
        # X-Achse (nur für Volume-Chart)
        ax_volume.set_xlabel('Time', color=self.config.text_color)
        
        # Layout optimieren
        plt.tight_layout()
    
    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Konvertiere Matplotlib Figure zu PIL Image"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='PNG', dpi=self.config.dpi, 
                   bbox_inches='tight', facecolor=self.config.background_color)
        buffer.seek(0)
        
        image = Image.open(buffer)
        return image.copy()  # Copy um Buffer zu schließen
    
    def generate_multiple_timeframes(
        self,
        ohlcv_data: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
        indicators: Optional[Dict[str, Any]] = None,
        base_title: str = "EUR/USD",
        output_dir: Optional[str] = None
    ) -> Dict[str, Image.Image]:
        """
        Generiere Charts für mehrere Timeframes
        
        Args:
            ohlcv_data: Dictionary mit OHLCV-Daten pro Timeframe
            indicators: Technische Indikatoren
            base_title: Basis-Titel für Charts
            output_dir: Optional Output-Verzeichnis
            
        Returns:
            Dictionary mit Charts pro Timeframe
        """
        charts = {}
        
        for timeframe, data in ohlcv_data.items():
            try:
                title = f"{base_title} {timeframe.upper()}"
                save_path = None
                
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    save_path = str(output_path / f"chart_{timeframe}.png")
                
                chart = self.generate_candlestick_chart(
                    data, indicators, title, timeframe, save_path
                )
                
                charts[timeframe] = chart
                
            except Exception as e:
                self.logger.error(f"Failed to generate chart for {timeframe}: {e}")
        
        return charts
    
    def generate_mega_dataset_charts(
        self,
        mega_ohlcv_data: Dict[str, pl.DataFrame],
        charts_per_timeframe: int = 50,
        output_dir: str = "data/mega_charts_generated"
    ) -> List[str]:
        """
        Generiere Charts für MEGA-DATASET
        
        Args:
            mega_ohlcv_data: MEGA-DATASET OHLCV-Daten
            charts_per_timeframe: Anzahl Charts pro Timeframe
            output_dir: Output-Verzeichnis
            
        Returns:
            Liste der generierten Chart-Pfade
        """
        self.logger.info(f"🎨 Generating MEGA-DATASET charts: {charts_per_timeframe} per timeframe")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_chart_paths = []
        
        for timeframe, ohlcv_df in mega_ohlcv_data.items():
            if len(ohlcv_df) < 100:  # Mindestens 100 Bars benötigt
                continue
            
            self.logger.info(f"  📊 Generating {charts_per_timeframe} charts for {timeframe}")
            
            pandas_df = ohlcv_df.to_pandas()
            window_size = 100
            total_bars = len(pandas_df)
            
            # Gleichmäßig verteilte Chart-Positionen
            step_size = max(1, (total_bars - window_size) // charts_per_timeframe)
            
            for i in range(charts_per_timeframe):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                if end_idx >= total_bars:
                    break
                
                try:
                    # Chart-Daten extrahieren
                    chart_data = pandas_df.iloc[start_idx:end_idx].copy()
                    
                    # Chart generieren
                    chart_path = output_path / f"mega_chart_{timeframe}_{i+1:03d}.png"
                    
                    chart_image = self.generate_candlestick_chart(
                        chart_data,
                        indicators=None,  # Keine Indikatoren für saubere Vision-Analyse
                        title=f"EUR/USD {timeframe.upper()} - MEGA Dataset Chart {i+1}",
                        timeframe=timeframe,
                        save_path=str(chart_path)
                    )
                    
                    all_chart_paths.append(str(chart_path))
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ Chart generation failed for {timeframe} chart {i+1}: {e}")
                    continue
            
            self.logger.info(f"    ✅ Generated charts for {timeframe}")
        
        self.logger.info(f"✅ Total charts generated: {len(all_chart_paths)}")
        return all_chart_paths
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "charts_generated": self.charts_generated,
            "total_render_time": self.total_render_time,
            "average_render_time": self.total_render_time / max(1, self.charts_generated),
            "gpu_accelerated_operations": self.gpu_accelerated_operations,
            "gpu_available": GPU_AVAILABLE,
            "gpu_enabled": self.use_gpu,
            "charts_per_minute": (self.charts_generated / self.total_render_time * 60) if self.total_render_time > 0 else 0
        }


def demo_chart_renderer():
    """
    🎨 Demo für Chart Renderer
    """
    print("🎨 BAUSTEIN A3: CHART RENDERER DEMO")
    print("=" * 60)
    
    # Erstelle Chart Renderer
    config = ChartConfig(
        width=12,
        height=8,
        dpi=150,
        use_gpu=True
    )
    
    renderer = ChartRenderer(config)
    
    # Test-Daten generieren
    print("\n📊 Generating test OHLCV data...")
    
    dates = pd.date_range(start='2025-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Simuliere realistische Forex-Daten
    base_price = 1.0950
    price_changes = np.random.normal(0, 0.0005, len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 0.002, len(dates)),
        'low': prices - np.random.uniform(0, 0.002, len(dates)),
        'close': prices + np.random.normal(0, 0.0003, len(dates)),
        'volume': np.random.randint(500, 2000, len(dates))
    })
    
    # Technische Indikatoren hinzufügen
    test_data['sma_5'] = test_data['close'].rolling(5).mean()
    test_data['sma_20'] = test_data['close'].rolling(20).mean()
    
    print(f"✅ Generated {len(test_data)} bars of test data")
    
    # Test 1: Einzelner Chart
    print(f"\n🎨 TEST 1: Single Chart Generation")
    print("-" * 40)
    
    start_time = time.time()
    
    chart_image = renderer.generate_candlestick_chart(
        test_data,
        indicators={'support': [1.0920], 'resistance': [1.0980]},
        title="EUR/USD Test Chart",
        timeframe="1h",
        save_path="test_chart_single.png"
    )
    
    single_time = time.time() - start_time
    
    print(f"✅ Single chart generated in {single_time:.3f}s")
    print(f"📊 Chart size: {chart_image.size}")
    print(f"💾 Saved to: test_chart_single.png")
    
    # Test 2: Multi-Timeframe Charts
    print(f"\n🎨 TEST 2: Multi-Timeframe Charts")
    print("-" * 40)
    
    # Simuliere verschiedene Timeframes
    timeframes_data = {
        "1m": test_data.iloc[:100],  # Erste 100 Bars
        "5m": test_data.iloc[::5],   # Jede 5. Bar
        "1h": test_data.iloc[::20],  # Jede 20. Bar
    }
    
    start_time = time.time()
    
    multi_charts = renderer.generate_multiple_timeframes(
        timeframes_data,
        indicators={'support': [1.0920], 'resistance': [1.0980]},
        base_title="EUR/USD Multi-TF",
        output_dir="test_charts_multi"
    )
    
    multi_time = time.time() - start_time
    
    print(f"✅ {len(multi_charts)} charts generated in {multi_time:.3f}s")
    print(f"⚡ Average: {multi_time/len(multi_charts):.3f}s per chart")
    
    for tf, chart in multi_charts.items():
        print(f"  📊 {tf}: {chart.size}")
    
    # Test 3: Performance Stats
    print(f"\n📈 TEST 3: Performance Statistics")
    print("-" * 40)
    
    stats = renderer.get_performance_stats()
    
    print(f"📊 Renderer Performance:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    print(f"\n🎉 CHART RENDERER DEMO COMPLETED!")
    
    return True


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    success = demo_chart_renderer()
    exit(0 if success else 1)