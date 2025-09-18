#!/usr/bin/env python3
"""
Test-Script für die Multimodal Pattern Recognition Engine.
Testet alle vier Hauptkomponenten von Task 8.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import logging
from datetime import datetime, timedelta
from typing import List

# Projekt-Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
from ai_indicator_optimizer.data.models import OHLCVData
from ai_indicator_optimizer.ai.pattern_recognition_engine import (
    MultimodalPatternRecognitionEngine, 
    PatternRecognitionConfig
)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_chart_image(width: int = 800, height: int = 600) -> Image.Image:
    """Erstellt ein Test-Chart-Bild mit simulierten Candlesticks"""
    try:
        # Weißer Hintergrund
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Chart-Bereich definieren
        margin = 50
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        # Simulierte Preisdaten für Candlesticks
        num_candles = 50
        candle_width = chart_width // num_candles
        
        # Preis-Range
        min_price = 1.0800
        max_price = 1.0900
        price_range = max_price - min_price
        
        # Candlesticks zeichnen
        current_price = 1.0850
        for i in range(num_candles):
            # Simuliere Preisbewegung
            open_price = current_price
            close_price = current_price + np.random.normal(0, 0.0005)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0003))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0003))
            
            # Koordinaten berechnen
            x = margin + i * candle_width
            
            # Y-Koordinaten (invertiert für Chart)
            y_open = margin + chart_height - ((open_price - min_price) / price_range) * chart_height
            y_close = margin + chart_height - ((close_price - min_price) / price_range) * chart_height
            y_high = margin + chart_height - ((high_price - min_price) / price_range) * chart_height
            y_low = margin + chart_height - ((low_price - min_price) / price_range) * chart_height
            
            # Candlestick zeichnen
            # High-Low Linie
            draw.line([(x + candle_width//2, y_high), (x + candle_width//2, y_low)], fill='black', width=1)
            
            # Body
            body_color = 'green' if close_price > open_price else 'red'
            draw.rectangle([x + 1, min(y_open, y_close), x + candle_width - 1, max(y_open, y_close)], 
                         fill=body_color, outline='black')
            
            current_price = close_price
        
        # Chart-Rahmen
        draw.rectangle([margin, margin, width - margin, height - margin], outline='black', width=2)
        
        logger.info(f"Test-Chart-Bild erstellt: {width}x{height}")
        return image
        
    except Exception as e:
        logger.exception(f"Fehler beim Erstellen des Test-Chart-Bildes: {e}")
        # Fallback: Einfaches weißes Bild
        return Image.new('RGB', (width, height), 'white')

def create_test_ohlcv_data(num_periods: int = 100) -> OHLCVData:
    """Erstellt Test-OHLCV-Daten"""
    try:
        # Simuliere EUR/USD Daten
        base_price = 1.0850
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_time = datetime.now() - timedelta(hours=num_periods)
        current_price = base_price
        
        for i in range(num_periods):
            # Timestamp
            timestamps.append(current_time + timedelta(hours=i))
            
            # OHLC
            open_price = current_price
            close_price = current_price + np.random.normal(0, 0.0008)  # 8 Pips Standardabweichung
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0005))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0005))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            
            # Volume (simuliert)
            volumes.append(np.random.randint(1000, 10000))
            
            current_price = close_price
        
        ohlcv_data = OHLCVData(
            timestamp=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes
        )
        
        logger.info(f"Test-OHLCV-Daten erstellt: {num_periods} Perioden")
        return ohlcv_data
        
    except Exception as e:
        logger.exception(f"Fehler beim Erstellen der Test-OHLCV-Daten: {e}")
        raise

def test_pattern_recognition_engine():
    """Testet die Multimodal Pattern Recognition Engine"""
    try:
        logger.info("🚀 Starte Test der Multimodal Pattern Recognition Engine")
        
        # 1. Hardware-Detektor initialisieren
        logger.info("1. Hardware-Detektor initialisieren...")
        hardware_detector = HardwareDetector()
        
        # Hardware-Info ausgeben
        logger.info(f"   CPU: {hardware_detector.cpu_info.model if hardware_detector.cpu_info else 'Unknown'}")
        logger.info(f"   CPU Cores: {hardware_detector.cpu_info.cores_logical if hardware_detector.cpu_info else 0}")
        logger.info(f"   GPU: {hardware_detector.gpu_info[0].name if hardware_detector.gpu_info else 'No GPU'}")
        logger.info(f"   RAM: {hardware_detector.memory_info.total // (1024**3) if hardware_detector.memory_info else 0} GB")
        
        # 2. Test-Konfiguration erstellen
        logger.info("2. Test-Konfiguration erstellen...")
        config = PatternRecognitionConfig(
            enable_visual_analysis=True,
            enable_numerical_optimization=True,
            enable_strategy_generation=True,
            enable_confidence_scoring=True,
            optimization_method="random",  # Schneller für Tests
            max_optimization_iterations=20,  # Reduziert für Tests
            parallel_optimization_jobs=4,
            use_gpu_acceleration=True,
            enable_caching=True
        )
        
        # 3. Pattern Recognition Engine initialisieren
        logger.info("3. Pattern Recognition Engine initialisieren...")
        engine = MultimodalPatternRecognitionEngine(hardware_detector, config)
        
        # 4. Test-Daten erstellen
        logger.info("4. Test-Daten erstellen...")
        chart_image = create_test_chart_image(1024, 768)
        ohlcv_data = create_test_ohlcv_data(200)  # 200 Stunden Daten
        
        # 5. Multimodale Analyse durchführen
        logger.info("5. Multimodale Analyse durchführen...")
        logger.info("   Dies kann einige Minuten dauern...")
        
        result = engine.analyze_market_data(
            chart_image=chart_image,
            ohlcv_data=ohlcv_data,
            timeframe="1h",
            market_context={
                "volatility": "medium",
                "liquidity": "high",
                "regime": "normal"
            }
        )
        
        # 6. Ergebnisse ausgeben
        logger.info("6. Ergebnisse analysieren...")
        
        print("\n" + "="*80)
        print("🎯 MULTIMODAL PATTERN RECOGNITION RESULTS")
        print("="*80)
        
        # Executive Summary
        print(f"\n📊 EXECUTIVE SUMMARY:")
        summary = result.executive_summary
        print(f"   Overall Sentiment: {summary.get('overall_sentiment', 'N/A')}")
        print(f"   Confidence Level: {summary.get('confidence_level', 'N/A')}")
        print(f"   Primary Strategy: {summary.get('primary_strategy', 'N/A')}")
        print(f"   Risk Level: {summary.get('risk_level', 'N/A')}")
        
        # Visual Analysis
        print(f"\n👁️ VISUAL PATTERN ANALYSIS:")
        visual = result.visual_analysis
        print(f"   Patterns Detected: {len(visual.patterns)}")
        print(f"   Overall Sentiment: {visual.overall_sentiment}")
        print(f"   Confidence Score: {visual.confidence_score:.3f}")
        
        if visual.patterns:
            print(f"   Top Patterns:")
            for i, pattern in enumerate(visual.patterns[:3], 1):
                print(f"     {i}. {pattern.pattern_type.value} (confidence: {pattern.confidence:.3f})")
        
        # Indicator Optimization
        print(f"\n📈 INDICATOR OPTIMIZATION:")
        indicators = result.indicator_optimization
        print(f"   Indicators Optimized: {len(indicators)}")
        
        if indicators:
            print(f"   Top Performers:")
            sorted_indicators = sorted(indicators.items(), key=lambda x: x[1].performance_score, reverse=True)
            for i, (indicator, opt_result) in enumerate(sorted_indicators[:3], 1):
                print(f"     {i}. {indicator.value}: {opt_result.performance_score:.3f}")
                print(f"        Parameters: {opt_result.optimal_parameters}")
        
        # Strategy Generation
        print(f"\n🎯 STRATEGY GENERATION:")
        strategy = result.strategy_generation
        print(f"   Primary Strategy: {strategy.primary_strategy.name}")
        print(f"   Strategy Type: {strategy.primary_strategy.strategy_type.value}")
        print(f"   Confidence: {strategy.primary_strategy.confidence_score:.3f}")
        print(f"   Alternative Strategies: {len(strategy.alternative_strategies)}")
        
        # Current Signal
        signal = strategy.current_signal
        print(f"\n📡 CURRENT TRADING SIGNAL:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Strength: {signal.strength.value}")
        print(f"   Confidence: {signal.confidence:.3f}")
        if signal.reasoning:
            print(f"   Reasoning: {signal.reasoning}")
        
        # Confidence Metrics
        print(f"\n🎲 CONFIDENCE METRICS:")
        confidence = result.confidence_metrics
        print(f"   Overall Confidence: {confidence.overall_confidence:.3f}")
        print(f"   Calibrated Confidence: {confidence.calibrated_confidence:.3f}")
        print(f"   Confidence Level: {confidence.confidence_level.value}")
        print(f"   Reliability Score: {confidence.reliability_score:.3f}")
        
        # Performance Stats
        print(f"\n⚡ PERFORMANCE STATISTICS:")
        print(f"   Processing Time: {result.processing_time:.2f} seconds")
        print(f"   Cache Hit Rate: {result.cache_hit_rate:.1%}")
        
        # Component Performance
        print(f"\n🔧 COMPONENT PERFORMANCE:")
        for component, perf in result.component_performance.items():
            if perf.get("success", False):
                duration = perf.get("duration", 0)
                print(f"   {component}: {duration:.2f}s ✅")
            else:
                print(f"   {component}: FAILED ❌")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Risk Assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        risk = result.risk_assessment
        print(f"   Overall Risk: {risk.get('overall_risk', 'N/A')}")
        print(f"   Confidence Risk: {risk.get('confidence_risk', 'N/A')}")
        print(f"   Market Risk: {risk.get('market_risk', 'N/A')}")
        
        # Errors (falls vorhanden)
        if result.error_log:
            print(f"\n❌ ERRORS:")
            for error in result.error_log:
                print(f"   - {error}")
        
        print("\n" + "="*80)
        print("✅ TEST ERFOLGREICH ABGESCHLOSSEN!")
        print("="*80)
        
        # Performance-Statistiken der Engine
        perf_stats = engine.get_performance_stats()
        print(f"\n📊 ENGINE PERFORMANCE STATS:")
        print(f"   Total Analyses: {perf_stats['total_analyses']}")
        print(f"   Successful Analyses: {perf_stats['successful_analyses']}")
        print(f"   Average Processing Time: {perf_stats['average_processing_time']:.2f}s")
        print(f"   Cache Hits: {perf_stats['cache_hits']}")
        print(f"   Cache Misses: {perf_stats['cache_misses']}")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Test fehlgeschlagen: {e}")
        print(f"\n❌ TEST FEHLGESCHLAGEN: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🧪 Multimodal Pattern Recognition Engine Test")
    print("=" * 50)
    
    success = test_pattern_recognition_engine()
    
    if success:
        print("\n🎉 Alle Tests erfolgreich!")
        return 0
    else:
        print("\n💥 Tests fehlgeschlagen!")
        return 1

if __name__ == "__main__":
    sys.exit(main())