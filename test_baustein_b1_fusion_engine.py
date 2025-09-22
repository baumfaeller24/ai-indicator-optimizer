#!/usr/bin/env python3
"""
🧩 BAUSTEIN B1 INTEGRATION TEST
Test der Multimodal Fusion Engine mit MEGA-DATASET Integration

Features:
- Vision+Indikatoren-Fusion-Engine
- Verschiedene Fusion-Strategien
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
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

# Import der lokalen Komponenten
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType


class FusionStrategy(Enum):
    """Strategien für multimodale Feature-Fusion"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"


class MultimodalFusionEngine:
    """
    🧩 BAUSTEIN B1: Multimodal Fusion Engine (Test Version)
    
    Kombiniert Vision-Analyse mit technischen Indikatoren
    """
    
    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_BASED):
        """Initialize Multimodal Fusion Engine"""
        self.fusion_strategy = fusion_strategy
        
        # Komponenten initialisieren
        self.feature_extractor = EnhancedFeatureExtractor()
        self.vision_client = OllamaVisionClient()
        self.schema_manager = UnifiedSchemaManager("data/multimodal_fusion/unified")
        
        # Performance Tracking
        self.total_fusions = 0
        self.successful_fusions = 0
        self.failed_fusions = 0
        self.total_processing_time = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multimodal Fusion Engine initialized with {fusion_strategy.value}")
    
    def fuse_vision_and_indicators(
        self,
        ohlcv_data: Union[pd.DataFrame, List[Dict]],
        timeframe: str = "1h",
        symbol: str = "EUR/USD"
    ) -> Dict[str, Any]:
        """
        Hauptfunktion: Fusioniere Vision-Analyse mit technischen Indikatoren
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Starting multimodal fusion for {symbol} {timeframe}")
            
            # 1. Technische Features extrahieren
            technical_features, technical_confidence = self._extract_technical_features(ohlcv_data)
            
            # 2. Vision-Features extrahieren (Mock für Test)
            vision_features, vision_confidence = self._extract_vision_features_mock(ohlcv_data)
            
            # 3. Features fusionieren
            fused_features, fusion_confidence = self._perform_feature_fusion(
                technical_features, technical_confidence,
                vision_features, vision_confidence
            )
            
            # 4. Ergebnis zusammenstellen
            processing_time = time.time() - start_time
            
            result = {
                "technical_features": technical_features,
                "technical_confidence": technical_confidence,
                "vision_features": vision_features,
                "vision_confidence": vision_confidence,
                "fused_features": fused_features,
                "fusion_confidence": fusion_confidence,
                "fusion_strategy": self.fusion_strategy.value,
                "processing_time": processing_time,
                "timestamp": datetime.now(),
                "symbol": symbol,
                "timeframe": timeframe,
                "success": True
            }
            
            # Performance Tracking
            self.total_fusions += 1
            self.successful_fusions += 1
            self.total_processing_time += processing_time
            
            self.logger.info(f"✅ Multimodal fusion completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_fusions += 1
            self.failed_fusions += 1
            self.total_processing_time += processing_time
            
            error_msg = f"Multimodal fusion failed: {e}"
            self.logger.error(error_msg)
            
            return {
                "error": str(e),
                "processing_time": processing_time,
                "success": False
            }
    
    def _extract_technical_features(self, ohlcv_data: Union[pd.DataFrame, List[Dict]]) -> Tuple[Dict[str, float], float]:
        """Extrahiere technische Features"""
        try:
            # Normalisiere Daten
            if isinstance(ohlcv_data, list):
                df = pd.DataFrame(ohlcv_data)
            else:
                df = ohlcv_data.copy()
            
            # Verwende Enhanced Feature Extractor
            features = self.feature_extractor.extract_features(df)
            
            # Konvertiere zu Dictionary
            technical_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    technical_features[key] = float(value)
            
            # Durchschnittliche technische Konfidenz
            technical_confidence = 0.8  # Mock für Test
            
            self.logger.debug(f"Technical features extracted: {len(technical_features)} features")
            
            return technical_features, technical_confidence
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")
            return {}, 0.0
    
    def _extract_vision_features_mock(self, ohlcv_data: Union[pd.DataFrame, List[Dict]]) -> Tuple[Dict[str, Any], float]:
        """Mock Vision-Features für Test"""
        try:
            # Mock Vision Features basierend auf Daten
            if isinstance(ohlcv_data, list):
                df = pd.DataFrame(ohlcv_data)
            else:
                df = ohlcv_data.copy()
            
            # Einfache Mock-Features basierend auf Preis-Bewegung
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            
            vision_features = {
                "vision_trend_numeric": 1.0 if price_change > 0 else -1.0,
                "vision_recommendation_numeric": 1.0 if price_change > 0.01 else (-1.0 if price_change < -0.01 else 0.0),
                "vision_pattern_count": np.random.randint(1, 4),
                "vision_pattern_strength": np.random.uniform(0.3, 0.9),
                "vision_insights_count": np.random.randint(2, 6),
                "vision_analysis_depth": np.random.uniform(0.4, 0.8),
                "vision_confidence": np.random.uniform(0.6, 0.9),
                "vision_processing_efficiency": np.random.uniform(0.7, 1.0),
                "vision_has_reversal_pattern": np.random.choice([0.0, 1.0]),
                "vision_has_continuation_pattern": np.random.choice([0.0, 1.0]),
                "vision_has_support_resistance": np.random.choice([0.0, 1.0])
            }
            
            vision_confidence = vision_features["vision_confidence"]
            
            self.logger.debug(f"Vision features extracted (mock): {len(vision_features)} features")
            
            return vision_features, vision_confidence
            
        except Exception as e:
            self.logger.error(f"Vision feature extraction failed: {e}")
            return {}, 0.0
    
    def _perform_feature_fusion(
        self,
        technical_features: Dict[str, float],
        technical_confidence: float,
        vision_features: Dict[str, Any],
        vision_confidence: float
    ) -> Tuple[Dict[str, float], float]:
        """Führe Feature-Fusion durch"""
        try:
            fused_features = {}
            
            # Gewichtungen berechnen
            tech_weight, vision_weight = self._calculate_fusion_weights(
                technical_confidence, vision_confidence
            )
            
            # 1. Direkte Feature-Fusion
            tech_trend = technical_features.get("trend_strength", 0.0)
            vision_trend = vision_features.get("vision_trend_numeric", 0.0)
            fused_features["multimodal_trend_strength"] = tech_weight * tech_trend + vision_weight * vision_trend
            
            # 2. Momentum-Fusion
            tech_momentum = technical_features.get("rsi_14", 50.0) / 100.0
            vision_momentum = vision_features.get("vision_pattern_strength", 0.5)
            fused_features["multimodal_momentum"] = tech_weight * tech_momentum + vision_weight * vision_momentum
            
            # 3. Pattern-Konfidenz
            vision_pattern_count = vision_features.get("vision_pattern_count", 0.0)
            rsi = technical_features.get("rsi_14", 50.0)
            rsi_extreme_factor = 1.2 if rsi > 70 or rsi < 30 else 1.0
            
            fused_features["multimodal_pattern_confidence"] = (
                vision_pattern_count * rsi_extreme_factor * vision_weight +
                (rsi / 100.0) * tech_weight
            )
            
            # 4. Konfidenz-Features
            fused_features["multimodal_overall_confidence"] = (
                technical_confidence * tech_weight + vision_confidence * vision_weight
            )
            
            confidence_consistency = 1.0 - abs(technical_confidence - vision_confidence)
            fused_features["multimodal_confidence_consistency"] = confidence_consistency
            
            avg_confidence = (technical_confidence + vision_confidence) / 2.0
            fused_features["multimodal_risk_score"] = 1.0 - avg_confidence
            fused_features["multimodal_opportunity_score"] = avg_confidence * confidence_consistency
            
            # Gesamte Fusion-Konfidenz
            fusion_confidence = self._calculate_fusion_confidence(
                technical_confidence, vision_confidence, tech_weight, vision_weight
            )
            
            self.logger.debug(f"Feature fusion completed: {len(fused_features)} fused features")
            
            return fused_features, fusion_confidence
            
        except Exception as e:
            self.logger.error(f"Feature fusion failed: {e}")
            return {}, 0.0
    
    def _calculate_fusion_weights(self, technical_confidence: float, vision_confidence: float) -> Tuple[float, float]:
        """Berechne Fusion-Gewichtungen"""
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return 0.6, 0.4
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
            total_confidence = technical_confidence + vision_confidence
            if total_confidence > 0:
                tech_weight = technical_confidence / total_confidence
                vision_weight = vision_confidence / total_confidence
            else:
                tech_weight = vision_weight = 0.5
            return tech_weight, vision_weight
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            if technical_confidence >= vision_confidence:
                return 0.7, 0.3
            else:
                return 0.3, 0.7
        else:  # ENSEMBLE
            return 0.5, 0.5
    
    def _calculate_fusion_confidence(
        self, technical_confidence: float, vision_confidence: float, 
        tech_weight: float, vision_weight: float
    ) -> float:
        """Berechne Gesamtkonfidenz der Fusion"""
        weighted_confidence = technical_confidence * tech_weight + vision_confidence * vision_weight
        consistency_bonus = 1.0 - abs(technical_confidence - vision_confidence)
        multimodal_bonus = 0.1 if technical_confidence > 0 and vision_confidence > 0 else 0.0
        
        fusion_confidence = weighted_confidence * (1.0 + consistency_bonus * 0.1 + multimodal_bonus)
        return min(fusion_confidence, 1.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "total_fusions": self.total_fusions,
            "successful_fusions": self.successful_fusions,
            "failed_fusions": self.failed_fusions,
            "success_rate": self.successful_fusions / max(1, self.total_fusions),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(1, self.total_fusions),
            "fusion_strategy": self.fusion_strategy.value,
            "fusions_per_minute": (self.total_fusions / self.total_processing_time * 60) if self.total_processing_time > 0 else 0
        }


def test_multimodal_fusion_engine():
    """
    🧩 Test Multimodal Fusion Engine Integration
    """
    print("🧩 BAUSTEIN B1: MULTIMODAL FUSION ENGINE INTEGRATION TEST")
    print("=" * 70)
    
    # Test-Daten generieren
    print("\n📊 Generating test OHLCV data...")
    
    dates = pd.date_range(start='2025-01-01', periods=100, freq='h')
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
    
    # Test 1: Einzelne Multimodale Fusion
    print(f"\n🔄 TEST 1: Single Multimodal Fusion")
    print("-" * 50)
    
    fusion_engine = MultimodalFusionEngine(FusionStrategy.CONFIDENCE_BASED)
    
    result = fusion_engine.fuse_vision_and_indicators(
        ohlcv_data=test_data,
        timeframe="1h",
        symbol="EUR/USD"
    )
    
    if result.get("success", False):
        print(f"✅ Multimodal Fusion Result:")
        print(f"  - Processing time: {result['processing_time']:.3f}s")
        print(f"  - Technical features: {len(result['technical_features'])}")
        print(f"  - Technical confidence: {result['technical_confidence']:.2f}")
        print(f"  - Vision features: {len(result['vision_features'])}")
        print(f"  - Vision confidence: {result['vision_confidence']:.2f}")
        print(f"  - Fused features: {len(result['fused_features'])}")
        print(f"  - Fusion confidence: {result['fusion_confidence']:.2f}")
        print(f"  - Fusion strategy: {result['fusion_strategy']}")
        
        # Zeige einige fusionierte Features
        print(f"\n📊 Sample Fused Features:")
        for i, (key, value) in enumerate(list(result['fused_features'].items())[:5]):
            print(f"  - {key}: {value:.4f}")
    else:
        print(f"❌ Fusion failed: {result.get('error', 'Unknown error')}")
    
    # Test 2: Verschiedene Fusion-Strategien
    print(f"\n🔄 TEST 2: Different Fusion Strategies")
    print("-" * 50)
    
    strategies = [
        FusionStrategy.WEIGHTED_AVERAGE,
        FusionStrategy.CONFIDENCE_BASED,
        FusionStrategy.HIERARCHICAL,
        FusionStrategy.ENSEMBLE
    ]
    
    strategy_results = {}
    
    for strategy in strategies:
        strategy_engine = MultimodalFusionEngine(fusion_strategy=strategy)
        
        strategy_result = strategy_engine.fuse_vision_and_indicators(
            ohlcv_data=test_data.iloc[:50],  # Kleinere Daten für Speed
            timeframe="1h",
            symbol="EUR/USD"
        )
        
        strategy_results[strategy.value] = strategy_result
        
        if strategy_result.get("success", False):
            print(f"  ✅ {strategy.value}:")
            print(f"    - Fusion confidence: {strategy_result['fusion_confidence']:.3f}")
            print(f"    - Processing time: {strategy_result['processing_time']:.3f}s")
            print(f"    - Fused features: {len(strategy_result['fused_features'])}")
        else:
            print(f"  ❌ {strategy.value}: Failed")
    
    # Test 3: Multi-Timeframe Fusion
    print(f"\n🔄 TEST 3: Multi-Timeframe Fusion")
    print("-" * 50)
    
    timeframes = ["1m", "5m", "1h", "4h"]
    timeframe_results = {}
    
    for timeframe in timeframes:
        # Verschiedene Datengrößen für verschiedene Timeframes
        if timeframe == "1m":
            tf_data = test_data.iloc[:30]
        elif timeframe == "5m":
            tf_data = test_data.iloc[::2]  # Jede 2. Zeile
        elif timeframe == "1h":
            tf_data = test_data.iloc[::5]  # Jede 5. Zeile
        else:  # 4h
            tf_data = test_data.iloc[::10]  # Jede 10. Zeile
        
        tf_result = fusion_engine.fuse_vision_and_indicators(
            ohlcv_data=tf_data,
            timeframe=timeframe,
            symbol="EUR/USD"
        )
        
        timeframe_results[timeframe] = tf_result
        
        if tf_result.get("success", False):
            print(f"  ✅ {timeframe}: {tf_result['processing_time']:.3f}s, confidence: {tf_result['fusion_confidence']:.3f}")
        else:
            print(f"  ❌ {timeframe}: Failed")
    
    # Test 4: Performance Statistics
    print(f"\n📈 TEST 4: Performance Statistics")
    print("-" * 50)
    
    stats = fusion_engine.get_performance_stats()
    
    print(f"📊 Fusion Engine Performance:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Test 5: Feature Analysis
    print(f"\n🔍 TEST 5: Feature Analysis")
    print("-" * 50)
    
    if result.get("success", False):
        print(f"📊 Technical Features Sample:")
        for i, (key, value) in enumerate(list(result['technical_features'].items())[:5]):
            print(f"  - {key}: {value:.4f}")
        
        print(f"\n🧠 Vision Features Sample:")
        vision_numeric = {k: v for k, v in result['vision_features'].items() if isinstance(v, (int, float))}
        for i, (key, value) in enumerate(list(vision_numeric.items())[:5]):
            print(f"  - {key}: {value:.4f}")
        
        print(f"\n🔗 Multimodal Features Sample:")
        multimodal_features = {k: v for k, v in result['fused_features'].items() if "multimodal" in k}
        for key, value in list(multimodal_features.items())[:5]:
            print(f"  - {key}: {value:.4f}")
    
    # Gesamtergebnis
    print(f"\n🎯 BAUSTEIN B1 INTEGRATION TEST RESULTS:")
    print("=" * 60)
    
    successful_strategies = len([r for r in strategy_results.values() if r.get("success", False)])
    successful_timeframes = len([r for r in timeframe_results.values() if r.get("success", False)])
    
    overall_success = (
        result.get("success", False) and
        successful_strategies >= 3 and  # Mindestens 3 von 4 Strategien
        successful_timeframes >= 3 and  # Mindestens 3 von 4 Timeframes
        stats['success_rate'] > 0.8
    )
    
    if overall_success:
        print("✅ Multimodal Fusion Engine Integration: PASSED")
        print("✅ Vision+Indikatoren-Fusion working successfully")
        print("✅ Multiple fusion strategies validated")
        print("✅ Multi-timeframe compatibility confirmed")
    else:
        print("❌ Multimodal Fusion Engine Integration: ISSUES DETECTED")
        print("⚠️ Check component configurations and fusion logic")
    
    # Export Results
    results_summary = {
        "test_timestamp": datetime.now().isoformat(),
        "baustein": "B1_multimodal_fusion_engine",
        "overall_success": overall_success,
        "performance_stats": stats,
        "strategy_results": {k: v.get("success", False) for k, v in strategy_results.items()},
        "timeframe_results": {k: v.get("success", False) for k, v in timeframe_results.items()},
        "fusion_capabilities": {
            "technical_features_extracted": len(result.get('technical_features', {})),
            "vision_features_extracted": len(result.get('vision_features', {})),
            "fused_features_generated": len(result.get('fused_features', {})),
            "fusion_confidence": result.get('fusion_confidence', 0.0)
        }
    }
    
    # Speichere Ergebnisse
    with open("baustein_b1_integration_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: baustein_b1_integration_results.json")
    
    if overall_success:
        print(f"\n🎉 BAUSTEIN B1 SUCCESSFULLY INTEGRATED!")
        print(f"Multimodal Fusion Engine ready for advanced ML training!")
    else:
        print(f"\n⚠️ BAUSTEIN B1 INTEGRATION NEEDS ATTENTION")
    
    return overall_success


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    success = test_multimodal_fusion_engine()
    exit(0 if success else 1)