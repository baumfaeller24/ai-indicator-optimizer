#!/usr/bin/env python3
"""
🧩 BAUSTEIN B1: Multimodal Fusion Engine
Vision+Indikatoren-Fusion für multimodale KI-Eingabe

Features:
- Kombination von Vision-Analyse und technischen Indikatoren
- Intelligente Feature-Fusion für ML-Training
- MEGA-DATASET-optimierte multimodale Verarbeitung
- Konfidenz-basierte Feature-Gewichtung
- Integration mit bestehender Enhanced Feature Extraction
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Import bestehender Komponenten
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.data.enhanced_chart_processor import EnhancedChartProcessor
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType


class FusionStrategy(Enum):
    """Strategien für multimodale Feature-Fusion"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"


@dataclass
class MultimodalFeatures:
    """Multimodale Feature-Struktur"""
    # Technische Features
    technical_features: Dict[str, float]
    technical_confidence: float
    
    # Vision Features
    vision_features: Dict[str, Any]
    vision_confidence: float
    
    # Fusionierte Features
    fused_features: Dict[str, float]
    fusion_confidence: float
    fusion_strategy: FusionStrategy
    
    # Metadaten
    timestamp: datetime
    symbol: str
    timeframe: str
    processing_time: float


class MultimodalFusionEngine:
    """
    🧩 BAUSTEIN B1: Multimodal Fusion Engine
    
    Kombiniert Vision-Analyse mit technischen Indikatoren:
    - Intelligente Feature-Fusion
    - Konfidenz-basierte Gewichtung
    - MEGA-DATASET-optimierte Verarbeitung
    - Multimodale ML-Eingabe-Generierung
    """
    
    def __init__(
        self,
        fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_BASED,
        output_dir: str = "data/multimodal_fusion"
    ):
        """
        Initialize Multimodal Fusion Engine
        
        Args:
            fusion_strategy: Strategie für Feature-Fusion
            output_dir: Output-Verzeichnis für Fusion-Daten
        """
        self.fusion_strategy = fusion_strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten initialisieren
        self.feature_extractor = EnhancedFeatureExtractor()
        self.vision_client = OllamaVisionClient()
        self.chart_processor = EnhancedChartProcessor()
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        
        # Feature-Mapping-Konfiguration
        self.feature_mappings = self._initialize_feature_mappings()
        self.fusion_weights = self._initialize_fusion_weights()
        
        # Performance Tracking
        self.total_fusions = 0
        self.successful_fusions = 0
        self.failed_fusions = 0
        self.total_processing_time = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multimodal Fusion Engine initialized with {fusion_strategy.value} strategy")
    
    def _initialize_feature_mappings(self) -> Dict[str, Any]:
        """Initialisiere Feature-Mappings zwischen Vision und Technical"""
        return {
            # Vision → Technical Mappings
            "vision_to_technical": {
                "trend_direction": {
                    "bullish": {"rsi_signal": 1.0, "macd_signal": 1.0, "trend_strength": 0.8},
                    "bearish": {"rsi_signal": -1.0, "macd_signal": -1.0, "trend_strength": 0.8},
                    "neutral": {"rsi_signal": 0.0, "macd_signal": 0.0, "trend_strength": 0.2}
                },
                "pattern_strength": {
                    "high": {"volatility_factor": 1.2, "momentum_factor": 1.1},
                    "medium": {"volatility_factor": 1.0, "momentum_factor": 1.0},
                    "low": {"volatility_factor": 0.8, "momentum_factor": 0.9}
                }
            },
            
            # Technical → Vision Validation
            "technical_to_vision": {
                "rsi_overbought": {"expected_pattern": "reversal", "confidence_boost": 0.1},
                "rsi_oversold": {"expected_pattern": "reversal", "confidence_boost": 0.1},
                "macd_bullish_cross": {"expected_trend": "bullish", "confidence_boost": 0.15},
                "macd_bearish_cross": {"expected_trend": "bearish", "confidence_boost": 0.15}
            },
            
            # Fusion Feature Names
            "fused_feature_names": [
                "multimodal_trend_strength",
                "multimodal_momentum",
                "multimodal_volatility",
                "multimodal_pattern_confidence",
                "multimodal_reversal_probability",
                "multimodal_breakout_probability",
                "multimodal_support_resistance_strength",
                "multimodal_volume_confirmation",
                "multimodal_risk_score",
                "multimodal_opportunity_score"
            ]
        }
    
    def _initialize_fusion_weights(self) -> Dict[str, float]:
        """Initialisiere Gewichtungen für verschiedene Fusion-Strategien"""
        return {
            FusionStrategy.WEIGHTED_AVERAGE.value: {
                "technical_weight": 0.6,
                "vision_weight": 0.4
            },
            FusionStrategy.CONFIDENCE_BASED.value: {
                "min_technical_weight": 0.3,
                "max_technical_weight": 0.8,
                "min_vision_weight": 0.2,
                "max_vision_weight": 0.7
            },
            FusionStrategy.HIERARCHICAL.value: {
                "primary_weight": 0.7,
                "secondary_weight": 0.3
            },
            FusionStrategy.ENSEMBLE.value: {
                "equal_weight": 0.5
            }
        }
    
    def fuse_vision_and_indicators(
        self,
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]],
        timeframe: str = "1h",
        symbol: str = "EUR/USD",
        chart_analysis_type: str = "comprehensive"
    ) -> MultimodalFeatures:
        """
        Hauptfunktion: Fusioniere Vision-Analyse mit technischen Indikatoren
        
        Args:
            ohlcv_data: OHLCV-Daten
            timeframe: Timeframe für Analyse
            symbol: Trading-Symbol
            chart_analysis_type: Art der Chart-Analyse
            
        Returns:
            MultimodalFeatures mit fusionierten Daten
        """
        start_time = datetime.now()
        processing_start = start_time.timestamp()
        
        try:
            self.logger.info(f"🔄 Starting multimodal fusion for {symbol} {timeframe}")
            
            # 1. Technische Features extrahieren
            technical_features, technical_confidence = self._extract_technical_features(ohlcv_data)
            
            # 2. Vision-Features extrahieren (via Chart-Processing)
            vision_features, vision_confidence = self._extract_vision_features(
                ohlcv_data, timeframe, symbol, chart_analysis_type
            )
            
            # 3. Features fusionieren
            fused_features, fusion_confidence = self._perform_feature_fusion(
                technical_features, technical_confidence,
                vision_features, vision_confidence
            )
            
            # 4. Multimodale Features zusammenstellen
            processing_time = datetime.now().timestamp() - processing_start
            
            multimodal_result = MultimodalFeatures(
                technical_features=technical_features,
                technical_confidence=technical_confidence,
                vision_features=vision_features,
                vision_confidence=vision_confidence,
                fused_features=fused_features,
                fusion_confidence=fusion_confidence,
                fusion_strategy=self.fusion_strategy,
                timestamp=start_time,
                symbol=symbol,
                timeframe=timeframe,
                processing_time=processing_time
            )
            
            # 5. Performance Tracking
            self.total_fusions += 1
            self.successful_fusions += 1
            self.total_processing_time += processing_time
            
            # 6. Daten speichern
            self._save_multimodal_features(multimodal_result)
            
            self.logger.info(f"✅ Multimodal fusion completed in {processing_time:.3f}s")
            
            return multimodal_result
            
        except Exception as e:
            processing_time = datetime.now().timestamp() - processing_start
            self.total_fusions += 1
            self.failed_fusions += 1
            self.total_processing_time += processing_time
            
            error_msg = f"Multimodal fusion failed: {e}"
            self.logger.error(error_msg)
            
            # Fallback-Ergebnis
            return MultimodalFeatures(
                technical_features={},
                technical_confidence=0.0,
                vision_features={"error": str(e)},
                vision_confidence=0.0,
                fused_features={},
                fusion_confidence=0.0,
                fusion_strategy=self.fusion_strategy,
                timestamp=start_time,
                symbol=symbol,
                timeframe=timeframe,
                processing_time=processing_time
            )
    
    def _extract_technical_features(
        self, 
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]]
    ) -> Tuple[Dict[str, float], float]:
        """Extrahiere technische Features"""
        try:
            # Normalisiere Daten
            if isinstance(ohlcv_data, pl.DataFrame):
                df = ohlcv_data.to_pandas()
            elif isinstance(ohlcv_data, list):
                df = pd.DataFrame(ohlcv_data)
            else:
                df = ohlcv_data.copy()
            
            # Verwende Enhanced Feature Extractor
            features = self.feature_extractor.extract_features(df)
            
            # Konvertiere zu Dictionary mit Konfidenz-Bewertung
            technical_features = {}
            confidence_factors = []
            
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    technical_features[key] = float(value)
                    
                    # Konfidenz basierend auf Feature-Typ
                    if "sma" in key or "ema" in key:
                        confidence_factors.append(0.9)  # Hohe Konfidenz für MA
                    elif "rsi" in key:
                        confidence_factors.append(0.8)  # Gute Konfidenz für RSI
                    elif "macd" in key:
                        confidence_factors.append(0.85)  # Sehr gute Konfidenz für MACD
                    else:
                        confidence_factors.append(0.7)  # Standard-Konfidenz
            
            # Durchschnittliche technische Konfidenz
            technical_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            self.logger.debug(f"Technical features extracted: {len(technical_features)} features, {technical_confidence:.2f} confidence")
            
            return technical_features, technical_confidence
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")
            return {}, 0.0
    
    def _extract_vision_features(
        self,
        ohlcv_data: Union[pd.DataFrame, pl.DataFrame, List[Dict]],
        timeframe: str,
        symbol: str,
        analysis_type: str
    ) -> Tuple[Dict[str, Any], float]:
        """Extrahiere Vision-Features via Chart-Processing"""
        try:
            # Chart+Vision-Processing
            chart_vision_result = self.chart_processor.process_chart_with_vision(
                ohlcv_data=ohlcv_data,
                timeframe=timeframe,
                title=f"{symbol} {timeframe.upper()} - Multimodal Analysis",
                analysis_type=analysis_type,
                save_chart=False  # Für Performance
            )
            
            if not chart_vision_result.success:
                return {"error": chart_vision_result.error_message}, 0.0
            
            # Vision-Analyse zu strukturierten Features konvertieren
            vision_analysis = chart_vision_result.vision_analysis
            vision_features = self._convert_vision_to_features(vision_analysis)
            
            # Vision-Konfidenz
            vision_confidence = vision_analysis.get("confidence_score", 0.5)
            
            self.logger.debug(f"Vision features extracted: {len(vision_features)} features, {vision_confidence:.2f} confidence")
            
            return vision_features, vision_confidence
            
        except Exception as e:
            self.logger.error(f"Vision feature extraction failed: {e}")
            return {"error": str(e)}, 0.0
    
    def _convert_vision_to_features(self, vision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Konvertiere Vision-Analyse zu strukturierten Features"""
        vision_features = {}
        
        try:
            # Trading Signals
            trading_signals = vision_analysis.get("trading_signals", {})
            
            # Trend Direction (numerisch)
            trend = trading_signals.get("trend", "neutral")
            if trend == "bullish":
                vision_features["vision_trend_numeric"] = 1.0
            elif trend == "bearish":
                vision_features["vision_trend_numeric"] = -1.0
            else:
                vision_features["vision_trend_numeric"] = 0.0
            
            # Recommendation (numerisch)
            recommendation = trading_signals.get("recommendation", "hold")
            if recommendation == "buy":
                vision_features["vision_recommendation_numeric"] = 1.0
            elif recommendation == "sell":
                vision_features["vision_recommendation_numeric"] = -1.0
            else:
                vision_features["vision_recommendation_numeric"] = 0.0
            
            # Pattern Strength
            patterns_count = len(vision_analysis.get("patterns_identified", []))
            vision_features["vision_pattern_count"] = float(patterns_count)
            vision_features["vision_pattern_strength"] = min(patterns_count / 5.0, 1.0)  # Normalisiert
            
            # Analysis Quality
            insights_count = len(vision_analysis.get("key_insights", []))
            vision_features["vision_insights_count"] = float(insights_count)
            vision_features["vision_analysis_depth"] = min(insights_count / 10.0, 1.0)  # Normalisiert
            
            # Confidence Score
            vision_features["vision_confidence"] = vision_analysis.get("confidence_score", 0.5)
            
            # Processing Quality
            processing_time = vision_analysis.get("processing_time", 0.0)
            vision_features["vision_processing_efficiency"] = max(0.0, 1.0 - (processing_time / 10.0))  # Normalisiert
            
            # Pattern-spezifische Features
            patterns = vision_analysis.get("patterns_identified", [])
            vision_features["vision_has_reversal_pattern"] = 1.0 if any("reversal" in p.lower() for p in patterns) else 0.0
            vision_features["vision_has_continuation_pattern"] = 1.0 if any("continuation" in p.lower() for p in patterns) else 0.0
            vision_features["vision_has_support_resistance"] = 1.0 if any("support" in p.lower() or "resistance" in p.lower() for p in patterns) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Vision feature conversion failed: {e}")
            vision_features["conversion_error"] = str(e)
        
        return vision_features
    
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
            
            # Gewichtungen basierend auf Fusion-Strategie berechnen
            tech_weight, vision_weight = self._calculate_fusion_weights(
                technical_confidence, vision_confidence
            )
            
            # 1. Direkte Feature-Fusion (wo möglich)
            fused_features.update(self._fuse_direct_features(
                technical_features, vision_features, tech_weight, vision_weight
            ))
            
            # 2. Cross-Modal Feature-Generierung
            fused_features.update(self._generate_cross_modal_features(
                technical_features, vision_features, tech_weight, vision_weight
            ))
            
            # 3. Multimodale Konfidenz-Features
            fused_features.update(self._generate_confidence_features(
                technical_confidence, vision_confidence, tech_weight, vision_weight
            ))
            
            # 4. Gesamte Fusion-Konfidenz berechnen
            fusion_confidence = self._calculate_fusion_confidence(
                technical_confidence, vision_confidence, tech_weight, vision_weight
            )
            
            self.logger.debug(f"Feature fusion completed: {len(fused_features)} fused features, {fusion_confidence:.2f} confidence")
            
            return fused_features, fusion_confidence
            
        except Exception as e:
            self.logger.error(f"Feature fusion failed: {e}")
            return {}, 0.0
    
    def _calculate_fusion_weights(
        self, 
        technical_confidence: float, 
        vision_confidence: float
    ) -> Tuple[float, float]:
        """Berechne Fusion-Gewichtungen basierend auf Strategie"""
        
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            weights = self.fusion_weights[FusionStrategy.WEIGHTED_AVERAGE.value]
            return weights["technical_weight"], weights["vision_weight"]
        
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
            # Gewichtung basierend auf Konfidenz
            total_confidence = technical_confidence + vision_confidence
            if total_confidence > 0:
                tech_weight = technical_confidence / total_confidence
                vision_weight = vision_confidence / total_confidence
            else:
                tech_weight = vision_weight = 0.5
            
            # Grenzen einhalten
            weights = self.fusion_weights[FusionStrategy.CONFIDENCE_BASED.value]
            tech_weight = np.clip(tech_weight, weights["min_technical_weight"], weights["max_technical_weight"])
            vision_weight = np.clip(vision_weight, weights["min_vision_weight"], weights["max_vision_weight"])
            
            # Normalisieren
            total_weight = tech_weight + vision_weight
            tech_weight /= total_weight
            vision_weight /= total_weight
            
            return tech_weight, vision_weight
        
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            # Primäre Modalität basierend auf höherer Konfidenz
            if technical_confidence >= vision_confidence:
                return 0.7, 0.3
            else:
                return 0.3, 0.7
        
        else:  # ENSEMBLE
            return 0.5, 0.5
    
    def _fuse_direct_features(
        self,
        technical_features: Dict[str, float],
        vision_features: Dict[str, Any],
        tech_weight: float,
        vision_weight: float
    ) -> Dict[str, float]:
        """Direkte Feature-Fusion für ähnliche Features"""
        fused = {}
        
        try:
            # Trend-Features fusionieren
            tech_trend = technical_features.get("trend_strength", 0.0)
            vision_trend = vision_features.get("vision_trend_numeric", 0.0)
            fused["multimodal_trend_strength"] = tech_weight * tech_trend + vision_weight * vision_trend
            
            # Momentum-Features fusionieren
            tech_momentum = technical_features.get("rsi_14", 50.0) / 100.0  # Normalisiert
            vision_momentum = vision_features.get("vision_pattern_strength", 0.5)
            fused["multimodal_momentum"] = tech_weight * tech_momentum + vision_weight * vision_momentum
            
            # Volatility-Features fusionieren
            tech_volatility = technical_features.get("atr_14", 0.001) * 10000  # Normalisiert zu Pips
            vision_volatility = vision_features.get("vision_analysis_depth", 0.5)
            fused["multimodal_volatility"] = tech_weight * tech_volatility + vision_weight * vision_volatility
            
        except Exception as e:
            self.logger.warning(f"Direct feature fusion failed: {e}")
        
        return fused
    
    def _generate_cross_modal_features(
        self,
        technical_features: Dict[str, float],
        vision_features: Dict[str, Any],
        tech_weight: float,
        vision_weight: float
    ) -> Dict[str, float]:
        """Generiere Cross-Modal Features"""
        cross_modal = {}
        
        try:
            # Pattern-Konfidenz basierend auf technischer Bestätigung
            rsi = technical_features.get("rsi_14", 50.0)
            vision_pattern_count = vision_features.get("vision_pattern_count", 0.0)
            
            # RSI-Extremwerte verstärken Pattern-Konfidenz
            rsi_extreme_factor = 1.0
            if rsi > 70 or rsi < 30:
                rsi_extreme_factor = 1.2
            
            cross_modal["multimodal_pattern_confidence"] = (
                vision_pattern_count * rsi_extreme_factor * vision_weight +
                (rsi / 100.0) * tech_weight
            )
            
            # Reversal-Wahrscheinlichkeit
            macd_signal = technical_features.get("macd_signal", 0.0)
            vision_reversal = vision_features.get("vision_has_reversal_pattern", 0.0)
            
            cross_modal["multimodal_reversal_probability"] = (
                abs(macd_signal) * tech_weight +
                vision_reversal * vision_weight
            )
            
            # Breakout-Wahrscheinlichkeit
            bb_width = technical_features.get("bb_width", 0.001) * 10000  # Normalisiert
            vision_continuation = vision_features.get("vision_has_continuation_pattern", 0.0)
            
            cross_modal["multimodal_breakout_probability"] = (
                (1.0 / max(bb_width, 0.1)) * tech_weight +  # Niedrige BB-Width = höhere Breakout-Chance
                vision_continuation * vision_weight
            )
            
            # Support/Resistance-Stärke
            volume_ratio = technical_features.get("volume_ratio", 1.0)
            vision_sr = vision_features.get("vision_has_support_resistance", 0.0)
            
            cross_modal["multimodal_support_resistance_strength"] = (
                volume_ratio * tech_weight +
                vision_sr * vision_weight
            )
            
        except Exception as e:
            self.logger.warning(f"Cross-modal feature generation failed: {e}")
        
        return cross_modal
    
    def _generate_confidence_features(
        self,
        technical_confidence: float,
        vision_confidence: float,
        tech_weight: float,
        vision_weight: float
    ) -> Dict[str, float]:
        """Generiere Konfidenz-basierte Features"""
        confidence_features = {}
        
        try:
            # Multimodale Gesamtkonfidenz
            confidence_features["multimodal_overall_confidence"] = (
                technical_confidence * tech_weight + vision_confidence * vision_weight
            )
            
            # Konfidenz-Konsistenz (wie ähnlich sind die Konfidenzen?)
            confidence_consistency = 1.0 - abs(technical_confidence - vision_confidence)
            confidence_features["multimodal_confidence_consistency"] = confidence_consistency
            
            # Risk Score (niedrigere Konfidenz = höheres Risiko)
            avg_confidence = (technical_confidence + vision_confidence) / 2.0
            confidence_features["multimodal_risk_score"] = 1.0 - avg_confidence
            
            # Opportunity Score (hohe Konfidenz + Konsistenz = hohe Opportunity)
            confidence_features["multimodal_opportunity_score"] = avg_confidence * confidence_consistency
            
            # Modalitäts-Dominanz
            if technical_confidence > vision_confidence:
                confidence_features["multimodal_technical_dominance"] = 1.0
                confidence_features["multimodal_vision_dominance"] = 0.0
            else:
                confidence_features["multimodal_technical_dominance"] = 0.0
                confidence_features["multimodal_vision_dominance"] = 1.0
            
        except Exception as e:
            self.logger.warning(f"Confidence feature generation failed: {e}")
        
        return confidence_features
    
    def _calculate_fusion_confidence(
        self,
        technical_confidence: float,
        vision_confidence: float,
        tech_weight: float,
        vision_weight: float
    ) -> float:
        """Berechne Gesamtkonfidenz der Fusion"""
        try:
            # Gewichtete Konfidenz
            weighted_confidence = technical_confidence * tech_weight + vision_confidence * vision_weight
            
            # Konsistenz-Bonus
            consistency_bonus = 1.0 - abs(technical_confidence - vision_confidence)
            
            # Multimodalitäts-Bonus (beide Modalitäten verfügbar)
            multimodal_bonus = 0.1 if technical_confidence > 0 and vision_confidence > 0 else 0.0
            
            # Gesamtkonfidenz
            fusion_confidence = weighted_confidence * (1.0 + consistency_bonus * 0.1 + multimodal_bonus)
            
            return min(fusion_confidence, 1.0)  # Cap bei 1.0
            
        except Exception:
            return 0.5  # Fallback
    
    def _save_multimodal_features(self, multimodal_result: MultimodalFeatures):
        """Speichere multimodale Features in Schema Manager"""
        try:
            # Technical Features speichern
            if multimodal_result.technical_features:
                technical_data = {
                    "timestamp": multimodal_result.timestamp,
                    "symbol": multimodal_result.symbol,
                    "timeframe": multimodal_result.timeframe,
                    **multimodal_result.technical_features
                }
                self.schema_manager.write_to_stream(technical_data, DataStreamType.TECHNICAL_FEATURES)
            
            # AI Predictions (aus Vision) speichern
            if multimodal_result.vision_features and "error" not in multimodal_result.vision_features:
                prediction_data = {
                    "timestamp": multimodal_result.timestamp,
                    "symbol": multimodal_result.symbol,
                    "timeframe": multimodal_result.timeframe,
                    "model_name": "MultimodalFusionEngine",
                    "prediction_class": "multimodal_analysis",
                    "confidence_score": multimodal_result.fusion_confidence,
                    "processing_time_ms": multimodal_result.processing_time * 1000,
                    "fusion_strategy": multimodal_result.fusion_strategy.value,
                    "technical_confidence": multimodal_result.technical_confidence,
                    "vision_confidence": multimodal_result.vision_confidence
                }
                self.schema_manager.write_to_stream(prediction_data, DataStreamType.AI_PREDICTIONS)
            
            # Performance Metrics speichern
            performance_data = {
                "timestamp": multimodal_result.timestamp,
                "component": "MultimodalFusionEngine",
                "operation": "multimodal_fusion",
                "duration_ms": multimodal_result.processing_time * 1000,
                "success_rate": 1.0,
                "fusion_strategy": multimodal_result.fusion_strategy.value,
                "features_fused": len(multimodal_result.fused_features)
            }
            self.schema_manager.write_to_stream(performance_data, DataStreamType.PERFORMANCE_METRICS)
            
        except Exception as e:
            self.logger.warning(f"Failed to save multimodal features: {e}")
    
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


def demo_multimodal_fusion_engine():
    """
    🧩 Demo für Multimodal Fusion Engine
    """
    print("🧩 BAUSTEIN B1: MULTIMODAL FUSION ENGINE DEMO")
    print("=" * 70)
    
    # Erstelle Fusion Engine
    fusion_engine = MultimodalFusionEngine(
        fusion_strategy=FusionStrategy.CONFIDENCE_BASED
    )
    
    # Test-Daten generieren
    print("\n📊 Generating test OHLCV data...")
    
    import numpy as np
    
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
    
    result = fusion_engine.fuse_vision_and_indicators(
        ohlcv_data=test_data,
        timeframe="1h",
        symbol="EUR/USD",
        chart_analysis_type="comprehensive"
    )
    
    print(f"✅ Multimodal Fusion Result:")
    print(f"  - Processing time: {result.processing_time:.3f}s")
    print(f"  - Technical features: {len(result.technical_features)}")
    print(f"  - Technical confidence: {result.technical_confidence:.2f}")
    print(f"  - Vision features: {len(result.vision_features)}")
    print(f"  - Vision confidence: {result.vision_confidence:.2f}")
    print(f"  - Fused features: {len(result.fused_features)}")
    print(f"  - Fusion confidence: {result.fusion_confidence:.2f}")
    print(f"  - Fusion strategy: {result.fusion_strategy.value}")
    
    # Zeige einige fusionierte Features
    print(f"\n📊 Sample Fused Features:")
    for i, (key, value) in enumerate(list(result.fused_features.items())[:5]):
        print(f"  - {key}: {value:.4f}")
    
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
        
        print(f"  ✅ {strategy.value}:")
        print(f"    - Fusion confidence: {strategy_result.fusion_confidence:.3f}")
        print(f"    - Processing time: {strategy_result.processing_time:.3f}s")
        print(f"    - Fused features: {len(strategy_result.fused_features)}")
    
    # Test 3: Performance Statistics
    print(f"\n📈 TEST 3: Performance Statistics")
    print("-" * 50)
    
    stats = fusion_engine.get_performance_stats()
    
    print(f"📊 Fusion Engine Performance:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Test 4: Feature Analysis
    print(f"\n🔍 TEST 4: Feature Analysis")
    print("-" * 50)
    
    print(f"📊 Technical Features Sample:")
    for i, (key, value) in enumerate(list(result.technical_features.items())[:5]):
        print(f"  - {key}: {value:.4f}")
    
    print(f"\n🧠 Vision Features Sample:")
    vision_numeric = {k: v for k, v in result.vision_features.items() if isinstance(v, (int, float))}
    for i, (key, value) in enumerate(list(vision_numeric.items())[:5]):
        print(f"  - {key}: {value:.4f}")
    
    print(f"\n🔗 Multimodal Features Sample:")
    multimodal_features = {k: v for k, v in result.fused_features.items() if "multimodal" in k}
    for key, value in list(multimodal_features.items())[:5]:
        print(f"  - {key}: {value:.4f}")
    
    print(f"\n🎉 MULTIMODAL FUSION ENGINE DEMO COMPLETED!")
    
    return True


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    success = demo_multimodal_fusion_engine()
    exit(0 if success else 1)