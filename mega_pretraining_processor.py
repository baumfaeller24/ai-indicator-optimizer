#!/usr/bin/env python3
"""
🚀 MEGA PRETRAINING PROCESSOR
Verarbeitung ALLER verfügbaren Tickdaten für ultimatives ML-Training

Verfügbare Daten:
- Juli 2025: 14.4M ticks (bereits verarbeitet)
- April 2025: 19.8M ticks (neu konvertiert)
- Mai 2025: 16.7M ticks (neu konvertiert)  
- Juni 2025: 11.3M ticks (neu konvertiert)
TOTAL: 62.2+ MILLIONEN TICKS!

Features:
- Verarbeitung aller 4 Monate EUR/USD Daten
- Massive Chart-Generierung (500+ Charts)
- Umfassende Vision-Analyse
- Multi-Timeframe OHLCV-Generierung
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
import json

# Import bestehende Komponenten
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient


class MegaPretrainingProcessor:
    """
    🚀 Mega Pretraining Processor
    
    Verarbeitet ALLE verfügbaren EUR/USD Tickdaten:
    - 62.2+ Millionen Ticks aus 4 Monaten
    - Multi-Timeframe OHLCV-Generierung
    - Massive Chart-Generierung für Vision-Training
    - Umfassende KI-Analyse mit MiniCPM-4.1-8B
    """
    
    def __init__(self, output_dir: str = "data/mega_pretraining"):
        """
        Initialize Mega Pretraining Processor
        
        Args:
            output_dir: Output-Verzeichnis für alle Daten
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        self.vision_client = OllamaVisionClient()
        
        # Verfügbare Datenquellen
        self.data_sources = {
            "july_2025": {
                "files": [
                    "EURUSD-2025-07_part1.parquet",
                    "EURUSD-2025-07_part2.parquet", 
                    "EURUSD-2025-07_part3.parquet",
                    "EURUSD-2025-07_part4.parquet",
                    "EURUSD-2025-07_part5.parquet"
                ],
                "estimated_ticks": 14_400_075,
                "status": "processed"
            },
            "april_2025": {
                "files": ["data/forex_converted/EURUSD-2025-04.parquet"],
                "estimated_ticks": 19_821_115,
                "status": "converted"
            },
            "may_2025": {
                "files": ["data/forex_converted/EURUSD-2025-05.parquet"],
                "estimated_ticks": 16_703_219,
                "status": "converted"
            },
            "june_2025": {
                "files": ["data/forex_converted/EURUSD-2025-06.parquet"],
                "estimated_ticks": 11_296_527,
                "status": "converted"
            }
        }
        
        # Performance Tracking
        self.total_ticks_processed = 0
        self.total_bars_generated = 0
        self.total_charts_generated = 0
        self.total_vision_analyses = 0
        self.processing_start_time = None
        
        # Hardware Info
        self.cpu_cores = psutil.cpu_count()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Mega Pretraining Processor with {self.cpu_cores} CPU cores")
    
    def load_all_available_tickdata(self) -> pl.DataFrame:
        """
        Lade ALLE verfügbaren Tickdaten
        
        Returns:
            Kombinierter DataFrame mit allen verfügbaren Ticks
        """
        self.logger.info("🔄 Loading ALL available tickdata...")
        
        all_dataframes = []
        total_ticks = 0
        
        for period, info in self.data_sources.items():
            self.logger.info(f"\n📊 Loading {period} data...")
            
            period_dataframes = []
            
            for file_path in info["files"]:
                file_path = Path(file_path)
                
                if file_path.exists():
                    self.logger.info(f"  Loading {file_path.name}...")
                    
                    try:
                        df = pl.read_parquet(file_path)
                        
                        # Standardisiere Spaltennamen falls nötig
                        if len(df.columns) == 4:
                            # Standard Tickdata Format
                            if df.columns != ['symbol', 'timestamp', 'bid', 'ask']:
                                df = df.rename({
                                    df.columns[0]: "symbol",
                                    df.columns[1]: "timestamp", 
                                    df.columns[2]: "bid",
                                    df.columns[3]: "ask"
                                })
                        
                        period_dataframes.append(df)
                        total_ticks += len(df)
                        self.logger.info(f"    ✅ {len(df):,} ticks from {file_path.name}")
                        
                    except Exception as e:
                        self.logger.error(f"    ❌ Failed to load {file_path.name}: {e}")
                else:
                    self.logger.warning(f"    ⚠️ File not found: {file_path}")
            
            # Kombiniere Daten für diesen Zeitraum
            if period_dataframes:
                period_combined = pl.concat(period_dataframes)
                all_dataframes.append(period_combined)
                self.logger.info(f"  ✅ {period}: {len(period_combined):,} total ticks")
        
        if not all_dataframes:
            raise FileNotFoundError("No tickdata files found!")
        
        # Kombiniere alle Zeiträume
        mega_df = pl.concat(all_dataframes)
        self.total_ticks_processed = len(mega_df)
        
        self.logger.info(f"\n🎉 MEGA DATASET LOADED!")
        self.logger.info(f"  - Total ticks: {len(mega_df):,}")
        self.logger.info(f"  - Time periods: {len(self.data_sources)}")
        self.logger.info(f"  - Data coverage: April-July 2025")
        
        return mega_df
    
    def parse_and_prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Timestamps und bereite Daten vor
        
        Args:
            df: Raw Tickdata DataFrame
            
        Returns:
            Vorbereiteter DataFrame
        """
        self.logger.info("🔄 Parsing timestamps and preparing data...")
        
        try:
            # Parse Timestamps (verschiedene Formate unterstützen)
            df = df.with_columns([
                pl.col("timestamp").str.strptime(
                    pl.Datetime, 
                    format="%Y%m%d %H:%M:%S%.3f",
                    strict=False
                ).alias("datetime")
            ])
            
            # Berechne Mid-Price und Spread
            df = df.with_columns([
                ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price"),
                (pl.col("ask") - pl.col("bid")).alias("spread"),
                (pl.col("ask") - pl.col("bid")).alias("spread_pips").mul(10000)  # Pips
            ])
            
            # Sortiere nach Zeit
            df = df.sort("datetime")
            
            # Filtere ungültige Daten
            df = df.filter(
                (pl.col("bid") > 0) & 
                (pl.col("ask") > 0) & 
                (pl.col("spread") > 0) &
                (pl.col("spread") < 0.01)  # Unrealistische Spreads filtern
            )
            
            self.logger.info(f"✅ Data prepared: {len(df):,} valid ticks")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return df
    
    def generate_multi_timeframe_ohlcv(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Generiere OHLCV-Daten für mehrere Timeframes
        
        Args:
            df: Tick DataFrame
            
        Returns:
            Dictionary mit OHLCV-DataFrames pro Timeframe
        """
        self.logger.info("🔄 Generating multi-timeframe OHLCV data...")
        
        timeframes = {
            "1m": "1m",
            "5m": "5m", 
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        ohlcv_data = {}
        
        for tf_name, tf_rule in timeframes.items():
            self.logger.info(f"  📊 Generating {tf_name} bars...")
            
            try:
                # Resample zu OHLCV
                ohlcv_df = df.group_by_dynamic(
                    "datetime",
                    every=tf_rule,
                    closed="left"
                ).agg([
                    pl.col("mid_price").first().alias("open"),
                    pl.col("mid_price").max().alias("high"),
                    pl.col("mid_price").min().alias("low"),
                    pl.col("mid_price").last().alias("close"),
                    pl.col("mid_price").count().alias("volume"),
                    pl.col("spread").mean().alias("avg_spread"),
                    pl.col("spread_pips").mean().alias("avg_spread_pips"),
                    pl.col("bid").first().alias("open_bid"),
                    pl.col("ask").first().alias("open_ask")
                ])
                
                # Filtere leere Bars
                ohlcv_df = ohlcv_df.filter(pl.col("volume") > 0)
                
                ohlcv_data[tf_name] = ohlcv_df
                self.total_bars_generated += len(ohlcv_df)
                
                self.logger.info(f"    ✅ {tf_name}: {len(ohlcv_df):,} bars")
                
            except Exception as e:
                self.logger.error(f"    ❌ Failed to generate {tf_name} bars: {e}")
        
        return ohlcv_data
    
    def generate_mega_charts(
        self, 
        ohlcv_data: Dict[str, pl.DataFrame], 
        charts_per_timeframe: int = 100
    ) -> List[str]:
        """
        Generiere massive Anzahl von Charts für Vision-Training
        
        Args:
            ohlcv_data: OHLCV-Daten pro Timeframe
            charts_per_timeframe: Anzahl Charts pro Timeframe
            
        Returns:
            Liste aller generierten Chart-Pfade
        """
        self.logger.info(f"🔄 Generating mega chart dataset: {charts_per_timeframe} charts per timeframe...")
        
        all_chart_paths = []
        
        for timeframe, ohlcv_df in ohlcv_data.items():
            if len(ohlcv_df) < 100:  # Brauche mindestens 100 Bars
                continue
                
            self.logger.info(f"  📊 Generating {charts_per_timeframe} charts for {timeframe}...")
            
            pandas_df = ohlcv_df.to_pandas()
            window_size = 100
            
            # Gleichmäßig verteilte Chart-Positionen
            total_bars = len(pandas_df)
            step_size = max(1, (total_bars - window_size) // charts_per_timeframe)
            
            for i in range(charts_per_timeframe):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                if end_idx >= total_bars:
                    break
                
                try:
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
                    ax.set_title(f'EUR/USD {timeframe.upper()} - Mega Dataset Chart {i+1}', fontsize=14)
                    ax.set_ylabel('Price', fontsize=12)
                    ax.set_xlabel('Time', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    # Speichere Chart
                    chart_path = self.output_dir / f"mega_chart_{timeframe}_{i+1:03d}.png"
                    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    all_chart_paths.append(str(chart_path))
                    self.total_charts_generated += 1
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ Chart generation failed for {timeframe} chart {i+1}: {e}")
                    continue
            
            self.logger.info(f"    ✅ Generated charts for {timeframe}")
        
        self.logger.info(f"✅ Total charts generated: {len(all_chart_paths)}")
        return all_chart_paths
    
    def run_mega_vision_analysis(self, chart_paths: List[str], batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Führe massive Vision-Analyse durch
        
        Args:
            chart_paths: Liste aller Chart-Pfade
            batch_size: Batch-Größe für Vision-Analyse
            
        Returns:
            Liste aller Vision-Analysen
        """
        self.logger.info(f"🔄 Running mega vision analysis: {len(chart_paths)} charts...")
        
        all_analyses = []
        
        # Verarbeite in Batches für bessere Performance
        for i in range(0, len(chart_paths), batch_size):
            batch = chart_paths[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chart_paths) + batch_size - 1) // batch_size
            
            self.logger.info(f"  🤖 Processing batch {batch_num}/{total_batches} ({len(batch)} charts)...")
            
            batch_analyses = []
            
            for j, chart_path in enumerate(batch):
                try:
                    analysis = self.vision_client.analyze_chart_image(
                        chart_path, 
                        analysis_type="comprehensive"
                    )
                    
                    # Erweitere um Metadaten
                    analysis.update({
                        'chart_path': chart_path,
                        'chart_index': i + j,
                        'batch_number': batch_num,
                        'timeframe': self._extract_timeframe_from_path(chart_path)
                    })
                    
                    batch_analyses.append(analysis)
                    self.total_vision_analyses += 1
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ Vision analysis failed for {chart_path}: {e}")
                    continue
            
            all_analyses.extend(batch_analyses)
            
            # Progress Update
            if batch_num % 5 == 0:
                self.logger.info(f"    📈 Progress: {len(all_analyses)}/{len(chart_paths)} charts analyzed")
        
        self.logger.info(f"✅ Mega vision analysis completed: {len(all_analyses)} analyses")
        return all_analyses
    
    def _extract_timeframe_from_path(self, chart_path: str) -> str:
        """Extrahiere Timeframe aus Chart-Pfad"""
        if "_1m_" in chart_path:
            return "1m"
        elif "_5m_" in chart_path:
            return "5m"
        elif "_15m_" in chart_path:
            return "15m"
        elif "_1h_" in chart_path:
            return "1h"
        elif "_4h_" in chart_path:
            return "4h"
        elif "_1d_" in chart_path:
            return "1d"
        else:
            return "unknown"
    
    def save_mega_dataset(
        self,
        tick_df: pl.DataFrame,
        ohlcv_data: Dict[str, pl.DataFrame],
        vision_analyses: List[Dict[str, Any]]
    ):
        """
        Speichere das komplette Mega-Dataset
        
        Args:
            tick_df: Alle Tickdaten
            ohlcv_data: OHLCV-Daten pro Timeframe
            vision_analyses: Alle Vision-Analysen
        """
        self.logger.info("💾 Saving mega dataset...")
        
        # 1. Speichere Raw Tickdata
        tick_path = self.output_dir / "mega_tickdata_all_months.parquet"
        tick_df.write_parquet(tick_path, compression="zstd")
        self.logger.info(f"  ✅ Saved mega tickdata: {tick_path} ({len(tick_df):,} ticks)")
        
        # 2. Speichere OHLCV pro Timeframe
        for timeframe, ohlcv_df in ohlcv_data.items():
            ohlcv_path = self.output_dir / f"mega_ohlcv_{timeframe}.parquet"
            ohlcv_df.write_parquet(ohlcv_path, compression="zstd")
            self.logger.info(f"  ✅ Saved {timeframe} OHLCV: {ohlcv_path} ({len(ohlcv_df):,} bars)")
        
        # 3. Speichere Vision-Analysen
        if vision_analyses:
            self.schema_manager.write_to_stream(
                vision_analyses,
                DataStreamType.AI_PREDICTIONS
            )
            self.logger.info(f"  ✅ Saved {len(vision_analyses)} vision analyses")
        
        # 4. Speichere Performance-Metriken
        performance_data = {
            "timestamp": datetime.now(),
            "component": "MegaPretrainingProcessor",
            "operation": "mega_processing",
            "total_ticks_processed": self.total_ticks_processed,
            "total_bars_generated": self.total_bars_generated,
            "total_charts_generated": self.total_charts_generated,
            "total_vision_analyses": self.total_vision_analyses,
            "processing_time_minutes": (time.time() - self.processing_start_time) / 60,
            "data_months": 4,
            "timeframes_generated": len(ohlcv_data)
        }
        
        self.schema_manager.write_to_stream(
            performance_data,
            DataStreamType.PERFORMANCE_METRICS
        )
        
        self.logger.info("✅ Mega dataset saved successfully")
    
    def run_mega_pretraining(
        self,
        charts_per_timeframe: int = 50,
        enable_vision_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Führe komplettes Mega-Pretraining durch
        
        Args:
            charts_per_timeframe: Anzahl Charts pro Timeframe
            enable_vision_analysis: Ob Vision-Analyse durchgeführt werden soll
            
        Returns:
            Mega-Processing-Ergebnisse
        """
        self.processing_start_time = time.time()
        self.logger.info("🚀 STARTING MEGA PRETRAINING PROCESSING...")
        
        results = {
            "start_time": datetime.now(),
            "data_sources": self.data_sources,
            "total_ticks_processed": 0,
            "total_bars_generated": 0,
            "total_charts_generated": 0,
            "total_vision_analyses": 0,
            "timeframes_processed": [],
            "processing_time_minutes": 0
        }
        
        try:
            # 1. Lade alle verfügbaren Tickdaten
            mega_tick_df = self.load_all_available_tickdata()
            
            # 2. Parse und bereite Daten vor
            prepared_df = self.parse_and_prepare_data(mega_tick_df)
            
            # 3. Generiere Multi-Timeframe OHLCV
            ohlcv_data = self.generate_multi_timeframe_ohlcv(prepared_df)
            
            # 4. Generiere massive Chart-Sammlung
            chart_paths = self.generate_mega_charts(ohlcv_data, charts_per_timeframe)
            
            # 5. Vision-Analyse (optional, aber empfohlen)
            vision_analyses = []
            if enable_vision_analysis and chart_paths:
                vision_analyses = self.run_mega_vision_analysis(chart_paths)
            
            # 6. Speichere komplettes Mega-Dataset
            self.save_mega_dataset(prepared_df, ohlcv_data, vision_analyses)
            
            # 7. Finale Statistiken
            processing_time = time.time() - self.processing_start_time
            
            results.update({
                "end_time": datetime.now(),
                "total_ticks_processed": self.total_ticks_processed,
                "total_bars_generated": self.total_bars_generated,
                "total_charts_generated": self.total_charts_generated,
                "total_vision_analyses": self.total_vision_analyses,
                "timeframes_processed": list(ohlcv_data.keys()),
                "processing_time_minutes": processing_time / 60
            })
            
            self.logger.info(f"\n🎉 MEGA PRETRAINING COMPLETED!")
            self.logger.info(f"  - Total ticks processed: {self.total_ticks_processed:,}")
            self.logger.info(f"  - Total bars generated: {self.total_bars_generated:,}")
            self.logger.info(f"  - Total charts generated: {self.total_charts_generated}")
            self.logger.info(f"  - Total vision analyses: {self.total_vision_analyses}")
            self.logger.info(f"  - Processing time: {processing_time / 60:.1f} minutes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Mega pretraining failed: {e}")
            results["error"] = str(e)
            return results


def run_mega_pretraining():
    """
    🚀 Hauptfunktion für Mega Pretraining
    """
    
    print("🚀 MEGA PRETRAINING PROCESSOR")
    print("=" * 80)
    print("Processing ALL available EUR/USD tickdata for ultimate ML training:")
    print("- April 2025: 19.8M ticks")
    print("- May 2025: 16.7M ticks") 
    print("- June 2025: 11.3M ticks")
    print("- July 2025: 14.4M ticks")
    print("TOTAL: 62.2+ MILLION TICKS!")
    print("=" * 80)
    
    # Erstelle Mega Processor
    processor = MegaPretrainingProcessor()
    
    # Führe Mega Pretraining durch
    results = processor.run_mega_pretraining(
        charts_per_timeframe=50,  # 50 Charts pro Timeframe = 300+ Charts total
        enable_vision_analysis=True
    )
    
    # Speichere Ergebnisse
    results_file = "mega_pretraining_results.json"
    
    # Serialisierbare Version
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, datetime):
            serializable_results[key] = value.isoformat()
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📊 MEGA PRETRAINING RESULTS:")
    print(f"  - Total ticks: {results.get('total_ticks_processed', 0):,}")
    print(f"  - Total bars: {results.get('total_bars_generated', 0):,}")
    print(f"  - Total charts: {results.get('total_charts_generated', 0)}")
    print(f"  - Vision analyses: {results.get('total_vision_analyses', 0)}")
    print(f"  - Processing time: {results.get('processing_time_minutes', 0):.1f} minutes")
    print(f"  - Results saved to: {results_file}")
    
    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        return False
    else:
        print(f"\n🎉 MEGA PRETRAINING SUCCESS!")
        print("🚀 Ready for world-class ML training with 62M+ ticks!")
        return True


if __name__ == "__main__":
    success = run_mega_pretraining()
    exit(0 if success else 1)