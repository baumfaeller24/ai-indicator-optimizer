#!/usr/bin/env python3
"""
🧩 BAUSTEIN A3: Enhanced Chart Processor
Verbindet Chart-Generierung mit Vision-Analyse für multimodale Pipeline

Features:
- Integration von ChartRenderer und OllamaVisionClient
- Automatische Chart-zu-Vision-Pipeline
- MEGA-DATASET-optimierte Verarbeitung
- Batch-Processing für 250+ Charts
- Performance-optimierte multimodale Analyse
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
from datetime import datetime
import json
import pandas as pd
import polars as pl
from PIL import Image
from dataclasses import dataclass

# Import der Komponenten
from ai_indicator_optimizer.data.chart_renderer import ChartRenderer, ChartConfig
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType


@dataclass
class ChartVisionResult:
    """Ergebnis der Chart-Vision-Analyse"""
    chart_path: str
    chart_image: Image.Image
    vision_analysis: Dict[str, Any]
    timeframe: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class EnhancedChartProcessor:
    """
    🧩 BAUSTEIN A3: Enhanced Chart Processor
    
    Verbindet bestehende Chart-Generierung mit neuer Vision-Analyse:
    - Automatische Chart-zu-Vision-Pipeline
    - MEGA-DATASET-Integration
    - Performance-optimierte Verarbeitung
    - Strukturierte multimodale Ausgabe
    """
    
    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        vision_base_url: str = "http://localhost:11434",
        vision_model: str = "openbmb/minicpm4.1:latest",
        output_dir: str = "data/chart_vision_pipeline"
    ):
        """
        Initialize Enhanced Chart Processor
        
        Args:
            chart_config: Konfiguration für Chart-Rendering
            vision_base_url: Ollama Server URL
            vision_model: Vision Model Name
            output_dir: Output-Verzeichnis für Pipeline-Daten
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten initialisieren
        self.chart_renderer = ChartRenderer(chart_config)
        self.vision_client = OllamaVisionClient(vision_base_url, vision_model)
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        
        # Performance Tracking
        self.total_charts_processed = 0
        self.total_vision_analyses = 0
        self.total_processing_time = 0.0
        self.successful_analyses = 0
        self.failed_analyses = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Chart Processor initialized with Chart+Vision pipeline")
    
    def process_chart_with_vision(
        self,
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]],
        timeframe: str = "1h",
        title: str = "EUR/USD Chart",
        indicators: Optional[Dict[str, Any]] = None,
        analysis_type: str = "comprehensive",
        save_chart: bool = True
    ) -> ChartVisionResult:
        """
        Verarbeite OHLCV-Daten zu Chart und führe Vision-Analyse durch
        
        Args:
            ohlcv_data: OHLCV-Daten
            timeframe: Timeframe für Chart
            title: Chart-Titel
            indicators: Technische Indikatoren
            analysis_type: Art der Vision-Analyse
            save_chart: Ob Chart gespeichert werden soll
            
        Returns:
            ChartVisionResult mit Chart und Analyse
        """
        start_time = time.time()
        
        try:
            # 1. Chart generieren
            self.logger.debug(f"Generating chart for {timeframe} timeframe...")
            
            chart_path = None
            if save_chart:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chart_path = str(self.output_dir / f"chart_{timeframe}_{timestamp}.png")
            
            chart_image = self.chart_renderer.generate_candlestick_chart(
                ohlcv_data=ohlcv_data,
                indicators=indicators,
                title=title,
                timeframe=timeframe,
                save_path=chart_path
            )
            
            self.total_charts_processed += 1
            
            # 2. Vision-Analyse durchführen
            self.logger.debug(f"Performing vision analysis...")
            
            vision_analysis = self.vision_client.analyze_chart_image(
                chart_image,
                analysis_type=analysis_type
            )
            
            self.total_vision_analyses += 1
            
            # 3. Erfolg bewerten
            success = "error" not in vision_analysis
            if success:
                self.successful_analyses += 1
            else:
                self.failed_analyses += 1
            
            # 4. Ergebnis zusammenstellen
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            result = ChartVisionResult(
                chart_path=chart_path or "in_memory",
                chart_image=chart_image,
                vision_analysis=vision_analysis,
                timeframe=timeframe,
                processing_time=processing_time,
                success=success,
                error_message=vision_analysis.get("error") if not success else None
            )
            
            # 5. Daten in Schema Manager speichern
            self._save_to_schema_manager(result, ohlcv_data)
            
            self.logger.debug(f"Chart+Vision processing completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.failed_analyses += 1
            
            error_msg = f"Chart+Vision processing failed: {e}"
            self.logger.error(error_msg)
            
            return ChartVisionResult(
                chart_path="",
                chart_image=None,
                vision_analysis={"error": str(e)},
                timeframe=timeframe,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def _save_to_schema_manager(
        self, 
        result: ChartVisionResult, 
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]]
    ):
        """Speichere Ergebnisse in Schema Manager"""
        try:
            # AI Predictions speichern
            if result.success and result.vision_analysis:
                prediction_data = {
                    "timestamp": datetime.now(),
                    "symbol": "EUR/USD",  # Default, kann erweitert werden
                    "timeframe": result.timeframe,
                    "model_name": "MiniCPM-4.1-8B",
                    "prediction_class": result.vision_analysis.get("trading_signals", {}).get("recommendation", "hold"),
                    "confidence_score": result.vision_analysis.get("confidence_score", 0.0),
                    "buy_probability": 0.6 if result.vision_analysis.get("trading_signals", {}).get("trend") == "bullish" else 0.3,
                    "sell_probability": 0.6 if result.vision_analysis.get("trading_signals", {}).get("trend") == "bearish" else 0.3,
                    "hold_probability": 0.4,
                    "processing_time_ms": result.processing_time * 1000,
                    "chart_path": result.chart_path,
                    "patterns_identified": len(result.vision_analysis.get("patterns_identified", [])),
                    "analysis_quality": result.vision_analysis.get("analysis_quality", "medium")
                }
                
                self.schema_manager.write_to_stream(prediction_data, DataStreamType.AI_PREDICTIONS)
            
            # Performance Metrics speichern
            performance_data = {
                "timestamp": datetime.now(),
                "component": "EnhancedChartProcessor",
                "operation": "chart_vision_processing",
                "duration_ms": result.processing_time * 1000,
                "success_rate": 1.0 if result.success else 0.0,
                "timeframe": result.timeframe,
                "chart_generated": 1,
                "vision_analysis_completed": 1 if result.success else 0
            }
            
            self.schema_manager.write_to_stream(performance_data, DataStreamType.PERFORMANCE_METRICS)
            
        except Exception as e:
            self.logger.warning(f"Failed to save to schema manager: {e}")
    
    def process_multiple_timeframes(
        self,
        ohlcv_data_dict: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
        base_title: str = "EUR/USD",
        indicators: Optional[Dict[str, Any]] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, ChartVisionResult]:
        """
        Verarbeite mehrere Timeframes mit Chart+Vision-Pipeline
        
        Args:
            ohlcv_data_dict: Dictionary mit OHLCV-Daten pro Timeframe
            base_title: Basis-Titel für Charts
            indicators: Technische Indikatoren
            analysis_type: Art der Vision-Analyse
            
        Returns:
            Dictionary mit ChartVisionResult pro Timeframe
        """
        self.logger.info(f"Processing {len(ohlcv_data_dict)} timeframes with Chart+Vision pipeline")
        
        results = {}
        
        for timeframe, ohlcv_data in ohlcv_data_dict.items():
            try:
                title = f"{base_title} {timeframe.upper()}"
                
                result = self.process_chart_with_vision(
                    ohlcv_data=ohlcv_data,
                    timeframe=timeframe,
                    title=title,
                    indicators=indicators,
                    analysis_type=analysis_type,
                    save_chart=True
                )
                
                results[timeframe] = result
                
                status = "✅" if result.success else "❌"
                self.logger.info(f"  {status} {timeframe}: {result.processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"  ❌ {timeframe}: Failed - {e}")
                results[timeframe] = ChartVisionResult(
                    chart_path="",
                    chart_image=None,
                    vision_analysis={"error": str(e)},
                    timeframe=timeframe,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def process_mega_dataset_batch(
        self,
        mega_ohlcv_data: Dict[str, pl.DataFrame],
        charts_per_timeframe: int = 10,
        analysis_type: str = "comprehensive"
    ) -> List[ChartVisionResult]:
        """
        Batch-Verarbeitung von MEGA-DATASET mit Chart+Vision-Pipeline
        
        Args:
            mega_ohlcv_data: MEGA-DATASET OHLCV-Daten
            charts_per_timeframe: Anzahl Charts pro Timeframe
            analysis_type: Art der Vision-Analyse
            
        Returns:
            Liste aller ChartVisionResult
        """
        self.logger.info(f"🔄 Processing MEGA-DATASET batch: {charts_per_timeframe} charts per timeframe")
        
        all_results = []
        
        for timeframe, ohlcv_df in mega_ohlcv_data.items():
            if len(ohlcv_df) < 100:  # Mindestens 100 Bars
                continue
            
            self.logger.info(f"  📊 Processing {timeframe} timeframe...")
            
            pandas_df = ohlcv_df.to_pandas()
            window_size = 100
            total_bars = len(pandas_df)
            
            # Gleichmäßig verteilte Positionen
            step_size = max(1, (total_bars - window_size) // charts_per_timeframe)
            
            timeframe_results = []
            
            for i in range(charts_per_timeframe):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                if end_idx >= total_bars:
                    break
                
                try:
                    # Chart-Daten extrahieren
                    chart_data = pandas_df.iloc[start_idx:end_idx].copy()
                    
                    # Chart+Vision-Verarbeitung
                    result = self.process_chart_with_vision(
                        ohlcv_data=chart_data,
                        timeframe=timeframe,
                        title=f"EUR/USD {timeframe.upper()} - MEGA Dataset {i+1}",
                        indicators=None,  # Saubere Charts für Vision
                        analysis_type=analysis_type,
                        save_chart=True
                    )
                    
                    timeframe_results.append(result)
                    all_results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ Chart {i+1} failed: {e}")
            
            # Timeframe-Statistiken
            successful = [r for r in timeframe_results if r.success]
            success_rate = len(successful) / len(timeframe_results) if timeframe_results else 0
            avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
            
            self.logger.info(f"    ✅ {timeframe}: {len(successful)}/{len(timeframe_results)} success ({success_rate:.1%}), {avg_time:.2f}s avg")
        
        return all_results
    
    def analyze_existing_mega_charts(
        self,
        charts_directory: str = "data/mega_pretraining",
        max_charts: int = 50,
        analysis_type: str = "comprehensive"
    ) -> List[ChartVisionResult]:
        """
        Analysiere bestehende MEGA-DATASET Charts mit Vision
        
        Args:
            charts_directory: Verzeichnis mit bestehenden Charts
            max_charts: Maximale Anzahl zu analysierender Charts
            analysis_type: Art der Vision-Analyse
            
        Returns:
            Liste der Vision-Analyse-Ergebnisse
        """
        self.logger.info(f"🔍 Analyzing existing MEGA-DATASET charts: max {max_charts}")
        
        charts_dir = Path(charts_directory)
        if not charts_dir.exists():
            self.logger.error(f"Charts directory not found: {charts_dir}")
            return []
        
        chart_files = list(charts_dir.glob("mega_chart_*.png"))[:max_charts]
        self.logger.info(f"Found {len(chart_files)} chart files")
        
        results = []
        
        for i, chart_file in enumerate(chart_files):
            try:
                # Timeframe aus Dateiname extrahieren
                timeframe_match = chart_file.name.split('_')
                timeframe = timeframe_match[2] if len(timeframe_match) > 2 else "1h"
                
                self.logger.debug(f"  🔍 Analyzing {chart_file.name} ({timeframe})...")
                
                start_time = time.time()
                
                # Vision-Analyse durchführen
                vision_analysis = self.vision_client.analyze_chart_image(
                    str(chart_file),
                    analysis_type=analysis_type
                )
                
                processing_time = time.time() - start_time
                success = "error" not in vision_analysis
                
                # Chart-Bild laden
                try:
                    chart_image = Image.open(chart_file)
                except Exception:
                    chart_image = None
                
                result = ChartVisionResult(
                    chart_path=str(chart_file),
                    chart_image=chart_image,
                    vision_analysis=vision_analysis,
                    timeframe=timeframe,
                    processing_time=processing_time,
                    success=success,
                    error_message=vision_analysis.get("error") if not success else None
                )
                
                results.append(result)
                
                # Performance Tracking
                self.total_vision_analyses += 1
                self.total_processing_time += processing_time
                
                if success:
                    self.successful_analyses += 1
                else:
                    self.failed_analyses += 1
                
                # Speichere in Schema Manager
                self._save_existing_chart_analysis(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"    📈 Progress: {i + 1}/{len(chart_files)} charts analyzed")
                
            except Exception as e:
                self.logger.warning(f"  ❌ Analysis failed for {chart_file.name}: {e}")
        
        # Finale Statistiken
        successful = [r for r in results if r.success]
        success_rate = len(successful) / len(results) if results else 0
        avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
        
        self.logger.info(f"✅ Analysis completed: {len(successful)}/{len(results)} success ({success_rate:.1%}), {avg_time:.2f}s avg")
        
        return results
    
    def _save_existing_chart_analysis(self, result: ChartVisionResult):
        """Speichere Analyse bestehender Charts"""
        try:
            if result.success:
                prediction_data = {
                    "timestamp": datetime.now(),
                    "symbol": "EUR/USD",
                    "timeframe": result.timeframe,
                    "model_name": "MiniCPM-4.1-8B",
                    "prediction_class": result.vision_analysis.get("trading_signals", {}).get("recommendation", "hold"),
                    "confidence_score": result.vision_analysis.get("confidence_score", 0.0),
                    "chart_path": result.chart_path,
                    "analysis_source": "existing_mega_chart",
                    "processing_time_ms": result.processing_time * 1000
                }
                
                self.schema_manager.write_to_stream(prediction_data, DataStreamType.AI_PREDICTIONS)
                
        except Exception as e:
            self.logger.warning(f"Failed to save existing chart analysis: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        chart_stats = self.chart_renderer.get_performance_stats()
        vision_stats = self.vision_client.get_performance_stats()
        
        return {
            "enhanced_processor": {
                "total_charts_processed": self.total_charts_processed,
                "total_vision_analyses": self.total_vision_analyses,
                "successful_analyses": self.successful_analyses,
                "failed_analyses": self.failed_analyses,
                "success_rate": self.successful_analyses / max(1, self.total_vision_analyses),
                "total_processing_time": self.total_processing_time,
                "average_processing_time": self.total_processing_time / max(1, self.total_vision_analyses),
                "charts_per_minute": (self.total_charts_processed / self.total_processing_time * 60) if self.total_processing_time > 0 else 0
            },
            "chart_renderer": chart_stats,
            "vision_client": vision_stats
        }
    
    def export_results_summary(self, output_file: str = "chart_vision_pipeline_results.json") -> Dict[str, Any]:
        """Exportiere Ergebnisse-Zusammenfassung"""
        stats = self.get_performance_stats()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "baustein_a3_v1.0",
            "performance_stats": stats,
            "mega_dataset_integration": True,
            "components": {
                "chart_renderer": "✅ Active",
                "vision_client": "✅ Active", 
                "schema_manager": "✅ Active"
            }
        }
        
        # Speichere Summary
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results summary exported to: {output_path}")
        
        return summary


def demo_enhanced_chart_processor():
    """
    🧩 Demo für Enhanced Chart Processor
    """
    print("🧩 BAUSTEIN A3: ENHANCED CHART PROCESSOR DEMO")
    print("=" * 70)
    
    # Erstelle Enhanced Chart Processor
    processor = EnhancedChartProcessor()
    
    # Test-Daten generieren
    print("\n📊 Generating test OHLCV data...")
    
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=150, freq='1H')
    np.random.seed(42)
    
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
    
    print(f"✅ Generated {len(test_data)} bars of test data")
    
    # Test 1: Einzelne Chart+Vision-Verarbeitung
    print(f"\n🎨 TEST 1: Single Chart+Vision Processing")
    print("-" * 50)
    
    result = processor.process_chart_with_vision(
        ohlcv_data=test_data,
        timeframe="1h",
        title="EUR/USD Test Chart",
        indicators={'support': [1.0920], 'resistance': [1.0980]},
        analysis_type="comprehensive"
    )
    
    print(f"✅ Chart+Vision Result:")
    print(f"  - Success: {result.success}")
    print(f"  - Processing time: {result.processing_time:.3f}s")
    print(f"  - Chart saved: {result.chart_path}")
    print(f"  - Vision confidence: {result.vision_analysis.get('confidence_score', 0.0):.2f}")
    print(f"  - Patterns found: {len(result.vision_analysis.get('patterns_identified', []))}")
    
    # Test 2: Multi-Timeframe Processing
    print(f"\n🎨 TEST 2: Multi-Timeframe Processing")
    print("-" * 50)
    
    timeframes_data = {
        "1m": test_data.iloc[:100],
        "5m": test_data.iloc[::5],
        "1h": test_data.iloc[::20],
    }
    
    multi_results = processor.process_multiple_timeframes(
        ohlcv_data_dict=timeframes_data,
        base_title="EUR/USD Multi-TF Test",
        analysis_type="patterns"
    )
    
    print(f"✅ Multi-Timeframe Results:")
    for tf, result in multi_results.items():
        status = "✅" if result.success else "❌"
        confidence = result.vision_analysis.get('confidence_score', 0.0)
        print(f"  {status} {tf}: {result.processing_time:.2f}s, confidence: {confidence:.2f}")
    
    # Test 3: Bestehende MEGA-Charts analysieren (falls vorhanden)
    print(f"\n🔍 TEST 3: Existing MEGA-Charts Analysis")
    print("-" * 50)
    
    existing_results = processor.analyze_existing_mega_charts(
        charts_directory="data/mega_pretraining",
        max_charts=5,  # Nur 5 für Demo
        analysis_type="comprehensive"
    )
    
    if existing_results:
        successful = [r for r in existing_results if r.success]
        print(f"✅ Existing Charts Analysis:")
        print(f"  - Charts analyzed: {len(existing_results)}")
        print(f"  - Success rate: {len(successful)/len(existing_results):.1%}")
        print(f"  - Average time: {sum(r.processing_time for r in successful)/len(successful):.2f}s")
    else:
        print("⚠️ No existing MEGA-charts found (expected if not generated yet)")
    
    # Test 4: Performance Statistics
    print(f"\n📈 TEST 4: Performance Statistics")
    print("-" * 50)
    
    stats = processor.get_performance_stats()
    
    print(f"📊 Enhanced Processor Performance:")
    enhanced_stats = stats["enhanced_processor"]
    for key, value in enhanced_stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Test 5: Results Export
    print(f"\n💾 TEST 5: Results Export")
    print("-" * 50)
    
    summary = processor.export_results_summary("demo_chart_vision_results.json")
    
    print(f"📄 Results Summary:")
    print(f"  - Pipeline version: {summary['pipeline_version']}")
    print(f"  - Components active: {len([c for c in summary['components'].values() if '✅' in c])}/3")
    print(f"  - Export file: demo_chart_vision_results.json")
    
    print(f"\n🎉 ENHANCED CHART PROCESSOR DEMO COMPLETED!")
    
    return True


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    success = demo_enhanced_chart_processor()
    exit(0 if success else 1)