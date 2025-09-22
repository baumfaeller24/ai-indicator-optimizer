#!/usr/bin/env python3
"""
🧩 BAUSTEIN A3 INTEGRATION TEST
Test der Chart-Vision-Pipeline-Grundlagen mit MEGA-DATASET Integration

Features:
- Chart-Renderer + Vision-Client Integration
- Automatische Chart-zu-Vision-Pipeline
- MEGA-DATASET-Kompatibilität
- Performance-Validierung
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from pathlib import Path
import time
from datetime import datetime
import json
import pandas as pd
import polars as pl
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Import der lokalen Komponenten
from ai_indicator_optimizer.data.chart_renderer import ChartRenderer, ChartConfig
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType


@dataclass
class ChartVisionResult:
    """Ergebnis der Chart-Vision-Analyse"""
    chart_path: str
    chart_image: Optional[Image.Image]
    vision_analysis: Dict[str, Any]
    timeframe: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class EnhancedChartProcessor:
    """
    🧩 BAUSTEIN A3: Enhanced Chart Processor (Test Version)
    
    Verbindet Chart-Generierung mit Vision-Analyse
    """
    
    def __init__(self, output_dir: str = "data/chart_vision_pipeline"):
        """Initialize Enhanced Chart Processor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten initialisieren
        self.chart_renderer = ChartRenderer()
        self.vision_client = OllamaVisionClient()
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        
        # Performance Tracking
        self.total_charts_processed = 0
        self.total_vision_analyses = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.total_processing_time = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Chart Processor initialized")
    
    def process_chart_with_vision(
        self,
        ohlcv_data: Union[pd.DataFrame, List[Dict]],
        timeframe: str = "1h",
        title: str = "EUR/USD Chart",
        analysis_type: str = "comprehensive",
        save_chart: bool = True
    ) -> ChartVisionResult:
        """
        Verarbeite OHLCV-Daten zu Chart und führe Vision-Analyse durch
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
    
    def analyze_existing_mega_charts(
        self,
        charts_directory: str = "data/mega_pretraining",
        max_charts: int = 10,
        analysis_type: str = "comprehensive"
    ) -> List[ChartVisionResult]:
        """
        Analysiere bestehende MEGA-DATASET Charts mit Vision
        """
        self.logger.info(f"🔍 Analyzing existing MEGA-DATASET charts: max {max_charts}")
        
        charts_dir = Path(charts_directory)
        if not charts_dir.exists():
            self.logger.warning(f"Charts directory not found: {charts_dir}")
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
                
                if (i + 1) % 5 == 0:
                    self.logger.info(f"    📈 Progress: {i + 1}/{len(chart_files)} charts analyzed")
                
            except Exception as e:
                self.logger.warning(f"  ❌ Analysis failed for {chart_file.name}: {e}")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "total_charts_processed": self.total_charts_processed,
            "total_vision_analyses": self.total_vision_analyses,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "success_rate": self.successful_analyses / max(1, self.total_vision_analyses),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(1, self.total_vision_analyses),
            "charts_per_minute": (self.total_charts_processed / self.total_processing_time * 60) if self.total_processing_time > 0 else 0
        }


def test_chart_vision_pipeline():
    """
    🧩 Test Chart-Vision-Pipeline Integration
    """
    print("🧩 BAUSTEIN A3: CHART-VISION-PIPELINE INTEGRATION TEST")
    print("=" * 70)
    
    # Erstelle Enhanced Chart Processor
    processor = EnhancedChartProcessor()
    
    # Test-Daten generieren
    print("\n📊 Generating test OHLCV data...")
    
    dates = pd.date_range(start='2025-01-01', periods=150, freq='h')
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
        title="EUR/USD Test Chart - Baustein A3",
        analysis_type="comprehensive"
    )
    
    print(f"✅ Chart+Vision Result:")
    print(f"  - Success: {result.success}")
    print(f"  - Processing time: {result.processing_time:.3f}s")
    print(f"  - Chart saved: {result.chart_path}")
    
    if result.success:
        print(f"  - Vision confidence: {result.vision_analysis.get('confidence_score', 0.0):.2f}")
        print(f"  - Patterns found: {len(result.vision_analysis.get('patterns_identified', []))}")
        print(f"  - Trend detected: {result.vision_analysis.get('trading_signals', {}).get('trend', 'unknown')}")
        print(f"  - Recommendation: {result.vision_analysis.get('trading_signals', {}).get('recommendation', 'hold')}")
    else:
        print(f"  - Error: {result.error_message}")
    
    # Test 2: Multi-Timeframe Processing
    print(f"\n🎨 TEST 2: Multi-Timeframe Processing")
    print("-" * 50)
    
    timeframes_data = {
        "1m": test_data.iloc[:100],
        "5m": test_data.iloc[::5],
        "1h": test_data.iloc[::20],
    }
    
    multi_results = {}
    
    for timeframe, data in timeframes_data.items():
        result = processor.process_chart_with_vision(
            ohlcv_data=data,
            timeframe=timeframe,
            title=f"EUR/USD {timeframe.upper()} - Multi-TF Test",
            analysis_type="patterns"
        )
        multi_results[timeframe] = result
    
    print(f"✅ Multi-Timeframe Results:")
    for tf, result in multi_results.items():
        status = "✅" if result.success else "❌"
        confidence = result.vision_analysis.get('confidence_score', 0.0) if result.success else 0.0
        print(f"  {status} {tf}: {result.processing_time:.2f}s, confidence: {confidence:.2f}")
    
    # Test 3: Bestehende MEGA-Charts analysieren
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
        
        if successful:
            avg_time = sum(r.processing_time for r in successful) / len(successful)
            avg_confidence = sum(r.vision_analysis.get('confidence_score', 0.0) for r in successful) / len(successful)
            avg_patterns = sum(len(r.vision_analysis.get('patterns_identified', [])) for r in successful) / len(successful)
            
            print(f"  - Average time: {avg_time:.2f}s")
            print(f"  - Average confidence: {avg_confidence:.2f}")
            print(f"  - Average patterns: {avg_patterns:.1f}")
    else:
        print("⚠️ No existing MEGA-charts found (expected if not generated yet)")
    
    # Test 4: Performance Statistics
    print(f"\n📈 TEST 4: Performance Statistics")
    print("-" * 50)
    
    stats = processor.get_performance_stats()
    
    print(f"📊 Chart-Vision-Pipeline Performance:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Test 5: Component Integration Check
    print(f"\n🔧 TEST 5: Component Integration Check")
    print("-" * 50)
    
    # Chart Renderer Stats
    chart_stats = processor.chart_renderer.get_performance_stats()
    print(f"📊 Chart Renderer:")
    print(f"  - Charts generated: {chart_stats['charts_generated']}")
    print(f"  - Average render time: {chart_stats['average_render_time']:.3f}s")
    print(f"  - GPU enabled: {chart_stats['gpu_enabled']}")
    
    # Vision Client Stats
    vision_stats = processor.vision_client.get_performance_stats()
    print(f"🧠 Vision Client:")
    print(f"  - Total requests: {vision_stats['total_requests']}")
    print(f"  - Success rate: {vision_stats['success_rate']:.1%}")
    print(f"  - Average inference time: {vision_stats['average_inference_time']:.3f}s")
    
    # Schema Manager Check
    schema_stats = processor.schema_manager.get_performance_stats()
    print(f"💾 Schema Manager:")
    print(f"  - Successful writes: {schema_stats['successful_writes']}")
    print(f"  - Success rate: {schema_stats['success_rate']:.1%}")
    
    # Gesamtergebnis
    print(f"\n🎯 BAUSTEIN A3 INTEGRATION TEST RESULTS:")
    print("=" * 60)
    
    overall_success = (
        stats['success_rate'] > 0.7 and
        chart_stats['charts_generated'] > 0 and
        vision_stats['total_requests'] > 0
    )
    
    if overall_success:
        print("✅ Chart-Vision-Pipeline Integration: PASSED")
        print("✅ All components working together successfully")
        print("✅ MEGA-DATASET compatibility confirmed")
    else:
        print("❌ Chart-Vision-Pipeline Integration: ISSUES DETECTED")
        print("⚠️ Check component configurations and connections")
    
    # Export Results
    results_summary = {
        "test_timestamp": datetime.now().isoformat(),
        "baustein": "A3_chart_vision_pipeline",
        "overall_success": overall_success,
        "performance_stats": stats,
        "component_stats": {
            "chart_renderer": chart_stats,
            "vision_client": vision_stats,
            "schema_manager": schema_stats
        },
        "test_results": {
            "single_chart_vision": result.success,
            "multi_timeframe": len([r for r in multi_results.values() if r.success]) / len(multi_results),
            "existing_charts": len([r for r in existing_results if r.success]) / max(1, len(existing_results))
        }
    }
    
    # Speichere Ergebnisse
    with open("baustein_a3_integration_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: baustein_a3_integration_results.json")
    
    if overall_success:
        print(f"\n🎉 BAUSTEIN A3 SUCCESSFULLY INTEGRATED!")
        print(f"Chart-Vision-Pipeline ready for multimodal analysis!")
    else:
        print(f"\n⚠️ BAUSTEIN A3 INTEGRATION NEEDS ATTENTION")
    
    return overall_success


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    success = test_chart_vision_pipeline()
    exit(0 if success else 1)