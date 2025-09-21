#!/usr/bin/env python3
"""
🚀 PROFESSIONAL TICKDATA PROCESSOR
Verarbeitung der echten EURUSD Tickdaten für ML-Training

Features:
- Verarbeitung von 5 Parquet-Dateien mit Millionen von Ticks
- Konvertierung zu OHLCV-Bars (1m, 5m, 15m, 1h)
- Integration mit bestehender Pipeline
- Chart-Generierung für Vision-Training
- Optimiert für RTX 5090 + 32 Cores
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Import bestehende Komponenten
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient


class ProfessionalTickDataProcessor:
    """
    🚀 Professional Tick Data Processor
    
    Verarbeitet echte EURUSD Tickdaten für ML-Training:
    - 2.88M+ Ticks pro Datei
    - Bid/Ask-Spread-Analyse
    - OHLCV-Bar-Generierung
    - Multimodale Feature-Extraktion
    """
    
    def __init__(self, output_dir: str = "data/professional"):
        """
        Initialize Professional Tick Data Processor
        
        Args:
            output_dir: Output-Verzeichnis für verarbeitete Daten
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten
        self.feature_extractor = EnhancedFeatureExtractor()
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        self.vision_client = OllamaVisionClient()
        
        # Performance Tracking
        self.total_ticks_processed = 0
        self.total_bars_generated = 0
        self.processing_start_time = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Hardware Info
        self.cpu_cores = psutil.cpu_count()
        self.logger.info(f"Initialized with {self.cpu_cores} CPU cores")
    
    def load_all_tickdata(self) -> pl.DataFrame:
        """
        Lade alle 5 EURUSD Tickdaten-Dateien
        
        Returns:
            Kombinierter DataFrame mit allen Ticks
        """
        self.logger.info("🔄 Loading all EURUSD tickdata files...")
        
        tick_files = [
            "EURUSD-2025-07_part1.parquet",
            "EURUSD-2025-07_part2.parquet", 
            "EURUSD-2025-07_part3.parquet",
            "EURUSD-2025-07_part4.parquet",
            "EURUSD-2025-07_part5.parquet"
        ]
        
        dataframes = []
        total_rows = 0
        
        for file in tick_files:
            if Path(file).exists():
                self.logger.info(f"Loading {file}...")
                df = pl.read_parquet(file)
                
                # Standardisiere Spaltennamen
                df = df.rename({
                    df.columns[0]: "symbol",
                    df.columns[1]: "timestamp", 
                    df.columns[2]: "bid",
                    df.columns[3]: "ask"
                })
                
                dataframes.append(df)
                total_rows += len(df)
                self.logger.info(f"  - Loaded {len(df):,} ticks from {file}")
            else:
                self.logger.warning(f"File not found: {file}")
        
        if not dataframes:
            raise FileNotFoundError("No tickdata files found!")
        
        # Kombiniere alle DataFrames
        combined_df = pl.concat(dataframes)
        self.logger.info(f"✅ Total ticks loaded: {len(combined_df):,}")
        
        return combined_df
    
    def parse_timestamp(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Timestamp-Spalte zu datetime
        
        Args:
            df: DataFrame mit Timestamp-Spalte
            
        Returns:
            DataFrame mit geparsten Timestamps
        """
        self.logger.info("🔄 Parsing timestamps...")
        
        # Konvertiere Timestamp-String zu datetime
        df = df.with_columns([
            pl.col("timestamp").str.strptime(
                pl.Datetime, 
                format="%Y%m%d %H:%M:%S%.3f",
                strict=False
            ).alias("datetime")
        ])
        
        # Sortiere nach Zeit
        df = df.sort("datetime")
        
        self.logger.info("✅ Timestamps parsed and sorted")
        return df
    
    def calculate_mid_price(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Berechne Mid-Price aus Bid/Ask
        
        Args:
            df: DataFrame mit Bid/Ask-Spalten
            
        Returns:
            DataFrame mit Mid-Price
        """
        df = df.with_columns([
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price"),
            (pl.col("ask") - pl.col("bid")).alias("spread")
        ])
        
        return df
    
    def resample_to_bars(
        self, 
        df: pl.DataFrame, 
        timeframe: str = "1m"
    ) -> pl.DataFrame:
        """
        Resample Tickdaten zu OHLCV-Bars
        
        Args:
            df: Tick DataFrame
            timeframe: Zeitrahmen (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            OHLCV DataFrame
        """
        self.logger.info(f"🔄 Resampling to {timeframe} bars...")
        
        # Timeframe zu Polars-Regel
        timeframe_map = {
            "1m": "1m",
            "5m": "5m", 
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        rule = timeframe_map.get(timeframe, "1m")
        
        # Resample zu OHLCV
        ohlcv_df = df.group_by_dynamic(
            "datetime",
            every=rule,
            closed="left"
        ).agg([
            pl.col("mid_price").first().alias("open"),
            pl.col("mid_price").max().alias("high"),
            pl.col("mid_price").min().alias("low"),
            pl.col("mid_price").last().alias("close"),
            pl.col("mid_price").count().alias("volume"),  # Tick-Count als Volume
            pl.col("spread").mean().alias("avg_spread"),
            pl.col("bid").first().alias("open_bid"),
            pl.col("ask").first().alias("open_ask")
        ])
        
        # Filtere leere Bars
        ohlcv_df = ohlcv_df.filter(pl.col("volume") > 0)
        
        self.logger.info(f"✅ Generated {len(ohlcv_df):,} {timeframe} bars")
        return ohlcv_df
    
    def extract_features_from_bars(self, ohlcv_df: pl.DataFrame) -> pl.DataFrame:
        """
        Extrahiere technische Features aus OHLCV-Bars
        
        Args:
            ohlcv_df: OHLCV DataFrame
            
        Returns:
            DataFrame mit Features
        """
        self.logger.info("🔄 Extracting technical features...")
        
        # Konvertiere zu Pandas für Feature-Extraktion
        pandas_df = ohlcv_df.to_pandas()
        
        features_list = []
        
        for i, row in pandas_df.iterrows():
            if i < 50:  # Brauche mindestens 50 Bars für Indikatoren
                continue
                
            # Erstelle Mock-Bar für Feature-Extraktor
            mock_bar = type('MockBar', (), {
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'ts_event': int(row['datetime'].timestamp() * 1e9),
                'ts_init': int(row['datetime'].timestamp() * 1e9)
            })()
            
            # Extrahiere Features
            try:
                features = self.feature_extractor.extract_features(mock_bar)
                features['timestamp'] = row['datetime']
                features['symbol'] = 'EUR/USD'
                features['timeframe'] = '1m'
                features['avg_spread'] = row['avg_spread']
                
                features_list.append(features)
                
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for bar {i}: {e}")
                continue
        
        if features_list:
            features_df = pl.DataFrame(features_list)
            self.logger.info(f"✅ Extracted features for {len(features_df):,} bars")
            return features_df
        else:
            self.logger.warning("No features extracted!")
            return pl.DataFrame()
    
    def generate_charts_for_vision(
        self, 
        ohlcv_df: pl.DataFrame, 
        window_size: int = 100,
        num_charts: int = 50
    ) -> List[str]:
        """
        Generiere Charts für Vision-Training
        
        Args:
            ohlcv_df: OHLCV DataFrame
            window_size: Anzahl Bars pro Chart
            num_charts: Anzahl Charts zu generieren
            
        Returns:
            Liste der generierten Chart-Pfade
        """
        self.logger.info(f"🔄 Generating {num_charts} charts for vision training...")
        
        chart_paths = []
        pandas_df = ohlcv_df.to_pandas()
        
        # Gleichmäßig verteilte Chart-Positionen
        total_bars = len(pandas_df)
        step_size = max(1, (total_bars - window_size) // num_charts)
        
        for i in range(num_charts):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx >= total_bars:
                break
                
            # Chart-Daten extrahieren
            chart_data = pandas_df.iloc[start_idx:end_idx].copy()
            
            # Chart erstellen
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Candlestick-Chart
            for j, row in chart_data.iterrows():
                x = j - start_idx
                
                # Farbe bestimmen
                color = 'green' if row['close'] >= row['open'] else 'red'
                
                # Wick (High-Low Linie)
                ax.plot([x, x], [row['low'], row['high']], color='black', linewidth=1)
                
                # Body (Open-Close Rechteck)
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                rect = plt.Rectangle((x - 0.3, body_bottom), 0.6, body_height,
                                   facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
            
            # Chart-Formatierung
            ax.set_title(f'EUR/USD Professional Data - Chart {i+1}', fontsize=14)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_xlabel('Time', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Speichere Chart
            chart_path = self.output_dir / f"professional_chart_{i+1:03d}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_paths.append(str(chart_path))
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"  Generated {i+1}/{num_charts} charts")
        
        self.logger.info(f"✅ Generated {len(chart_paths)} charts")
        return chart_paths
    
    def analyze_charts_with_vision(self, chart_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analysiere Charts mit Vision-Client
        
        Args:
            chart_paths: Liste der Chart-Pfade
            
        Returns:
            Liste der Vision-Analysen
        """
        self.logger.info(f"🔄 Analyzing {len(chart_paths)} charts with vision...")
        
        vision_analyses = []
        
        for i, chart_path in enumerate(chart_paths):
            try:
                analysis = self.vision_client.analyze_chart_image(
                    chart_path, 
                    analysis_type="comprehensive"
                )
                
                analysis['chart_path'] = chart_path
                analysis['chart_index'] = i
                vision_analyses.append(analysis)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"  Analyzed {i+1}/{len(chart_paths)} charts")
                    
            except Exception as e:
                self.logger.warning(f"Vision analysis failed for {chart_path}: {e}")
                continue
        
        self.logger.info(f"✅ Completed vision analysis for {len(vision_analyses)} charts")
        return vision_analyses
    
    def save_processed_data(
        self, 
        ohlcv_df: pl.DataFrame,
        features_df: pl.DataFrame,
        vision_analyses: List[Dict[str, Any]]
    ):
        """
        Speichere verarbeitete Daten in Schema-konformen Streams
        
        Args:
            ohlcv_df: OHLCV DataFrame
            features_df: Features DataFrame  
            vision_analyses: Vision-Analysen
        """
        self.logger.info("💾 Saving processed data...")
        
        # 1. OHLCV-Daten speichern
        ohlcv_path = self.output_dir / "eurusd_professional_ohlcv.parquet"
        ohlcv_df.write_parquet(ohlcv_path, compression="zstd")
        self.logger.info(f"  Saved OHLCV data: {ohlcv_path}")
        
        # 2. Features zu Technical Features Stream
        if len(features_df) > 0:
            self.schema_manager.write_to_stream(
                features_df.to_dicts(),
                DataStreamType.TECHNICAL_FEATURES
            )
            self.logger.info(f"  Saved {len(features_df)} feature records to technical_features stream")
        
        # 3. Vision-Analysen zu AI Predictions Stream
        if vision_analyses:
            self.schema_manager.write_to_stream(
                vision_analyses,
                DataStreamType.AI_PREDICTIONS
            )
            self.logger.info(f"  Saved {len(vision_analyses)} vision analyses to ai_predictions stream")
        
        # 4. Performance-Metriken
        performance_data = {
            "timestamp": datetime.now(),
            "component": "ProfessionalTickDataProcessor",
            "operation": "full_processing",
            "bars_processed": len(ohlcv_df),
            "features_extracted": len(features_df),
            "charts_generated": len(vision_analyses),
            "total_ticks": self.total_ticks_processed,
            "processing_time_minutes": (time.time() - self.processing_start_time) / 60
        }
        
        self.schema_manager.write_to_stream(
            performance_data,
            DataStreamType.PERFORMANCE_METRICS
        )
        
        self.logger.info("✅ All data saved successfully")
    
    def run_full_processing(
        self,
        timeframes: List[str] = ["1m", "5m", "15m"],
        num_charts: int = 50,
        enable_vision: bool = True
    ) -> Dict[str, Any]:
        """
        Führe vollständige Verarbeitung der Tickdaten durch
        
        Args:
            timeframes: Liste der zu generierenden Timeframes
            num_charts: Anzahl Charts für Vision-Training
            enable_vision: Ob Vision-Analyse durchgeführt werden soll
            
        Returns:
            Verarbeitungs-Ergebnisse
        """
        self.processing_start_time = time.time()
        self.logger.info("🚀 Starting full professional tickdata processing...")
        
        results = {
            "start_time": datetime.now(),
            "timeframes_processed": [],
            "total_bars_generated": 0,
            "charts_generated": 0,
            "vision_analyses": 0,
            "processing_time_minutes": 0
        }
        
        try:
            # 1. Lade alle Tickdaten
            tick_df = self.load_all_tickdata()
            self.total_ticks_processed = len(tick_df)
            
            # 2. Parse Timestamps und berechne Mid-Price
            tick_df = self.parse_timestamp(tick_df)
            tick_df = self.calculate_mid_price(tick_df)
            
            # 3. Verarbeite jeden Timeframe
            all_features = []
            main_ohlcv_df = None
            
            for timeframe in timeframes:
                self.logger.info(f"\n📊 Processing timeframe: {timeframe}")
                
                # Resample zu Bars
                ohlcv_df = self.resample_to_bars(tick_df, timeframe)
                
                if main_ohlcv_df is None:  # Verwende ersten Timeframe für Charts
                    main_ohlcv_df = ohlcv_df
                
                # Extrahiere Features
                features_df = self.extract_features_from_bars(ohlcv_df)
                if len(features_df) > 0:
                    all_features.append(features_df)
                
                results["timeframes_processed"].append(timeframe)
                results["total_bars_generated"] += len(ohlcv_df)
                self.total_bars_generated += len(ohlcv_df)
            
            # 4. Kombiniere alle Features
            if all_features:
                combined_features = pl.concat(all_features)
            else:
                combined_features = pl.DataFrame()
            
            # 5. Generiere Charts für Vision-Training
            chart_paths = []
            vision_analyses = []
            
            if main_ohlcv_df is not None and len(main_ohlcv_df) > 100:
                chart_paths = self.generate_charts_for_vision(
                    main_ohlcv_df, 
                    num_charts=num_charts
                )
                results["charts_generated"] = len(chart_paths)
                
                # 6. Vision-Analyse (optional)
                if enable_vision and chart_paths:
                    vision_analyses = self.analyze_charts_with_vision(chart_paths)
                    results["vision_analyses"] = len(vision_analyses)
            
            # 7. Speichere alle Daten
            if main_ohlcv_df is not None:
                self.save_processed_data(main_ohlcv_df, combined_features, vision_analyses)
            
            # 8. Finale Statistiken
            processing_time = time.time() - self.processing_start_time
            results["processing_time_minutes"] = processing_time / 60
            results["end_time"] = datetime.now()
            
            self.logger.info(f"\n🎉 PROCESSING COMPLETED!")
            self.logger.info(f"  - Total ticks processed: {self.total_ticks_processed:,}")
            self.logger.info(f"  - Total bars generated: {results['total_bars_generated']:,}")
            self.logger.info(f"  - Charts generated: {results['charts_generated']}")
            self.logger.info(f"  - Vision analyses: {results['vision_analyses']}")
            self.logger.info(f"  - Processing time: {results['processing_time_minutes']:.1f} minutes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {e}")
            results["error"] = str(e)
            return results


def run_professional_tickdata_processing():
    """
    🚀 Hauptfunktion für Professional Tickdata Processing
    """
    
    print("🚀 PROFESSIONAL TICKDATA PROCESSING")
    print("=" * 60)
    print("Processing EURUSD-2025-07 tickdata files...")
    print("Expected: ~14M+ ticks across 5 files")
    print("Output: OHLCV bars, features, charts, vision analysis")
    print("=" * 60)
    
    # Erstelle Processor
    processor = ProfessionalTickDataProcessor()
    
    # Führe vollständige Verarbeitung durch
    results = processor.run_full_processing(
        timeframes=["1m", "5m", "15m"],
        num_charts=100,  # Mehr Charts für besseres Training
        enable_vision=True
    )
    
    # Speichere Ergebnisse
    results_file = "professional_tickdata_processing_results.json"
    import json
    
    # Serialisierbare Version
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, datetime):
            serializable_results[key] = value.isoformat()
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  - Processing time: {results.get('processing_time_minutes', 0):.1f} minutes")
    print(f"  - Total bars: {results.get('total_bars_generated', 0):,}")
    print(f"  - Charts generated: {results.get('charts_generated', 0)}")
    print(f"  - Vision analyses: {results.get('vision_analyses', 0)}")
    print(f"  - Results saved to: {results_file}")
    
    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        return False
    else:
        print(f"\n🎉 SUCCESS: Professional tickdata processing completed!")
        return True


if __name__ == "__main__":
    success = run_professional_tickdata_processing()
    exit(0 if success else 1)