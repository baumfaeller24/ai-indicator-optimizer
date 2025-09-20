"""
Confidence Scoring System für multimodale Predictions.
Bewertet und kalibriert Confidence-Scores aus verschiedenen Analysemethoden.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# Import mit Try-Catch für robuste Imports
try:
    from .visual_pattern_analyzer import VisualPatternAnalyzer
except ImportError:
    VisualPatternAnalyzer = None

try:
    from .numerical_indicator_optimizer import OptimizationResult, IndicatorType
except ImportError:
    OptimizationResult = None
    IndicatorType = None

try:
    from .multimodal_strategy_generator import StrategyGenerationResult, TradingStrategy
except ImportError:
    StrategyGenerationResult = None
    TradingStrategy = None

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence-Level-Kategorien"""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0

class UncertaintySource(Enum):
    """Quellen der Unsicherheit"""
    MODEL_UNCERTAINTY = "model_uncertainty"
    DATA_QUALITY = "data_quality"
    MARKET_VOLATILITY = "market_volatility"
    PATTERN_AMBIGUITY = "pattern_ambiguity"
    INDICATOR_CONFLICT = "indicator_conflict"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class ConfidenceMetrics:
    """Detaillierte Confidence-Metriken"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    calibrated_confidence: float
    uncertainty_sources: Dict[UncertaintySource, float]
    component_confidences: Dict[str, float]
    reliability_score: float
    prediction_interval: Tuple[float, float]
    validation_factors: Dict[str, float] = field(default_factory=dict)
    risk_adjusted_confidence: float = 0.0
    temporal_stability: float = 0.0


class EnhancedConfidenceScorer:
    """
    Enhanced Confidence Scoring System mit Multi-Factor-Validation
    
    Phase 2 Core AI Enhancement:
    - Multi-Factor-Confidence-Scoring
    - Uncertainty Quantification
    - Confidence Calibration
    - Risk-Adjusted Confidence
    - Temporal Stability Analysis
    - Cross-Validation-basierte Reliability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Enhanced Confidence Scorer
        
        Args:
            config: Konfiguration für Confidence-Scoring
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scoring-Parameter
        self.calibration_method = self.config.get("calibration_method", "isotonic")
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.3)
        self.min_samples_for_calibration = self.config.get("min_samples_for_calibration", 50)
        
        # Multi-Factor-Weights
        self.factor_weights = self.config.get("factor_weights", {
            "ai_prediction": 0.4,
            "pattern_analysis": 0.2,
            "technical_indicators": 0.2,
            "market_regime": 0.1,
            "volatility": 0.1
        })
        
        # Calibration Models
        self.calibration_models = {}
        self.historical_predictions = []
        self.historical_outcomes = []
        
        # Statistics
        self.scores_calculated = 0
        self.calibration_updates = 0
        
        self.logger.info(f"EnhancedConfidenceScorer initialized: method={self.calibration_method}")
    
    def calculate_enhanced_confidence(
        self,
        ai_prediction: Dict[str, Any],
        pattern_features: Dict[str, float],
        technical_indicators: Dict[str, float],
        market_context: Dict[str, Any],
        additional_factors: Optional[Dict[str, float]] = None
    ) -> ConfidenceMetrics:
        """
        Berechne Enhanced Confidence Score mit Multi-Factor-Validation
        
        Args:
            ai_prediction: AI-Prediction mit Confidence
            pattern_features: Visual Pattern Features
            technical_indicators: Technische Indikatoren
            market_context: Market-Context (Regime, Volatility, etc.)
            additional_factors: Zusätzliche Faktoren
            
        Returns:
            ConfidenceMetrics mit detaillierter Analyse
        """
        try:
            # 1. Component Confidences berechnen
            component_confidences = self._calculate_component_confidences(
                ai_prediction, pattern_features, technical_indicators, market_context
            )
            
            # 2. Multi-Factor-Confidence berechnen
            overall_confidence = self._calculate_multi_factor_confidence(component_confidences)
            
            # 3. Uncertainty Sources analysieren
            uncertainty_sources = self._analyze_uncertainty_sources(
                ai_prediction, pattern_features, technical_indicators, market_context
            )
            
            # 4. Confidence Calibration
            calibrated_confidence = self._calibrate_confidence(overall_confidence, component_confidences)
            
            # 5. Risk-Adjusted Confidence
            risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
                calibrated_confidence, market_context, uncertainty_sources
            )
            
            # 6. Validation Factors
            validation_factors = self._calculate_validation_factors(
                component_confidences, uncertainty_sources, additional_factors
            )
            
            # 7. Temporal Stability
            temporal_stability = self._calculate_temporal_stability(component_confidences)
            
            # 8. Reliability Score
            reliability_score = self._calculate_reliability_score(
                component_confidences, uncertainty_sources, validation_factors
            )
            
            # 9. Prediction Interval
            prediction_interval = self._calculate_prediction_interval(
                calibrated_confidence, uncertainty_sources
            )
            
            # 10. Confidence Level
            confidence_level = self._determine_confidence_level(risk_adjusted_confidence)
            
            # Create ConfidenceMetrics
            metrics = ConfidenceMetrics(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                calibrated_confidence=calibrated_confidence,
                uncertainty_sources=uncertainty_sources,
                component_confidences=component_confidences,
                reliability_score=reliability_score,
                prediction_interval=prediction_interval,
                validation_factors=validation_factors,
                risk_adjusted_confidence=risk_adjusted_confidence,
                temporal_stability=temporal_stability
            )
            
            self.scores_calculated += 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {e}")
            return self._get_fallback_confidence_metrics()
    
    def _calculate_component_confidences(
        self,
        ai_prediction: Dict[str, Any],
        pattern_features: Dict[str, float],
        technical_indicators: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Berechne Confidence für einzelne Komponenten"""
        component_confidences = {}
        
        # AI Prediction Confidence
        ai_confidence = ai_prediction.get("confidence", 0.5)
        component_confidences["ai_prediction"] = float(ai_confidence)
        
        # Pattern Analysis Confidence
        pattern_confidence = self._calculate_pattern_confidence(pattern_features)
        component_confidences["pattern_analysis"] = pattern_confidence
        
        # Technical Indicators Confidence
        technical_confidence = self._calculate_technical_confidence(technical_indicators)
        component_confidences["technical_indicators"] = technical_confidence
        
        # Market Regime Confidence
        regime_confidence = self._calculate_regime_confidence(market_context)
        component_confidences["market_regime"] = regime_confidence
        
        # Volatility Confidence
        volatility_confidence = self._calculate_volatility_confidence(market_context)
        component_confidences["volatility"] = volatility_confidence
        
        return component_confidences
    
    def _calculate_pattern_confidence(self, pattern_features: Dict[str, float]) -> float:
        """Berechne Pattern-basierte Confidence"""
        try:
            # Pattern Strength
            pattern_strength = pattern_features.get("pattern_confidence_max", 0.5)
            
            # Pattern Count (mehr Patterns = höhere Confidence)
            pattern_count = pattern_features.get("pattern_count", 0)
            count_factor = min(1.0, pattern_count / 3.0)  # Max bei 3+ Patterns
            
            # Pattern Type Consistency
            bullish_patterns = pattern_features.get("pattern_bullish", 0)
            bearish_patterns = pattern_features.get("pattern_bearish", 0)
            neutral_patterns = pattern_features.get("pattern_neutral", 0)
            
            total_patterns = bullish_patterns + bearish_patterns + neutral_patterns
            if total_patterns > 0:
                # Consistency = dominante Pattern-Richtung
                max_direction = max(bullish_patterns, bearish_patterns, neutral_patterns)
                consistency = max_direction / total_patterns
            else:
                consistency = 0.5
            
            # Kombiniere Faktoren
            pattern_confidence = (pattern_strength * 0.5 + count_factor * 0.3 + consistency * 0.2)
            
            return min(1.0, max(0.0, pattern_confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_technical_confidence(self, technical_indicators: Dict[str, float]) -> float:
        """Berechne Technical-Indicator-basierte Confidence"""
        try:
            confidence_factors = []
            
            # RSI Confidence
            rsi = technical_indicators.get("rsi_14", 50)
            if rsi < 30 or rsi > 70:  # Oversold/Overbought
                rsi_confidence = 0.8
            elif 40 <= rsi <= 60:  # Neutral
                rsi_confidence = 0.4
            else:
                rsi_confidence = 0.6
            confidence_factors.append(rsi_confidence)
            
            # Bollinger Bands Confidence
            bb_position = technical_indicators.get("bb_position", 0.5)
            if bb_position < 0.1 or bb_position > 0.9:  # Near bands
                bb_confidence = 0.7
            else:
                bb_confidence = 0.5
            confidence_factors.append(bb_confidence)
            
            # Volume Confidence
            volume_ratio = technical_indicators.get("volume_ratio", 1.0)
            if volume_ratio > 1.5:  # High volume
                volume_confidence = 0.8
            elif volume_ratio < 0.5:  # Low volume
                volume_confidence = 0.3
            else:
                volume_confidence = 0.6
            confidence_factors.append(volume_confidence)
            
            # Moving Average Confidence
            sma_5 = technical_indicators.get("sma_5", 0)
            sma_20 = technical_indicators.get("sma_20", 0)
            if sma_5 > 0 and sma_20 > 0:
                ma_diff = abs(sma_5 - sma_20) / max(sma_20, 1e-6)
                ma_confidence = min(0.9, ma_diff * 100)  # Higher diff = higher confidence
                confidence_factors.append(ma_confidence)
            
            # Durchschnitt der Faktoren
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_regime_confidence(self, market_context: Dict[str, Any]) -> float:
        """Berechne Market-Regime-basierte Confidence"""
        try:
            regime = market_context.get("market_regime", "unknown")
            
            # Regime-spezifische Confidence
            regime_confidences = {
                "trending": 0.8,    # Hohe Confidence in Trends
                "ranging": 0.6,     # Mittlere Confidence in Ranges
                "volatile": 0.4,    # Niedrige Confidence in volatilen Markets
                "quiet": 0.7,       # Gute Confidence in ruhigen Markets
                "unknown": 0.5      # Neutrale Confidence
            }
            
            base_confidence = regime_confidences.get(regime, 0.5)
            
            # Trend Strength Adjustment
            trend_strength = market_context.get("trend_strength", 0.0)
            trend_adjustment = trend_strength * 0.2  # Max 20% Boost
            
            return min(1.0, base_confidence + trend_adjustment)
            
        except Exception:
            return 0.5
    
    def _calculate_volatility_confidence(self, market_context: Dict[str, Any]) -> float:
        """Berechne Volatility-basierte Confidence"""
        try:
            volatility = market_context.get("volatility", 0.01)
            
            # Optimal Volatility Range für Trading
            if 0.005 <= volatility <= 0.02:  # Optimal range
                return 0.8
            elif volatility < 0.002:  # Too quiet
                return 0.4
            elif volatility > 0.05:  # Too volatile
                return 0.3
            else:  # Moderate volatility
                return 0.6
                
        except Exception:
            return 0.5
    
    def _calculate_multi_factor_confidence(self, component_confidences: Dict[str, float]) -> float:
        """Berechne gewichtete Multi-Factor-Confidence"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component, confidence in component_confidences.items():
                weight = self.factor_weights.get(component, 0.1)
                weighted_sum += confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _analyze_uncertainty_sources(
        self,
        ai_prediction: Dict[str, Any],
        pattern_features: Dict[str, float],
        technical_indicators: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> Dict[UncertaintySource, float]:
        """Analysiere Uncertainty Sources"""
        uncertainty_sources = {}
        
        # Model Uncertainty
        ai_confidence = ai_prediction.get("confidence", 0.5)
        model_uncertainty = 1.0 - ai_confidence
        uncertainty_sources[UncertaintySource.MODEL_UNCERTAINTY] = model_uncertainty
        
        # Data Quality Uncertainty
        data_completeness = market_context.get("data_completeness", 1.0)
        data_uncertainty = 1.0 - data_completeness
        uncertainty_sources[UncertaintySource.DATA_QUALITY] = data_uncertainty
        
        # Market Volatility Uncertainty
        volatility = market_context.get("volatility", 0.01)
        volatility_uncertainty = min(1.0, volatility * 50)  # Scale volatility to [0,1]
        uncertainty_sources[UncertaintySource.MARKET_VOLATILITY] = volatility_uncertainty
        
        # Pattern Ambiguity
        pattern_count = pattern_features.get("pattern_count", 0)
        if pattern_count == 0:
            pattern_uncertainty = 1.0
        else:
            # Mehr widersprüchliche Patterns = höhere Uncertainty
            bullish = pattern_features.get("pattern_bullish", 0)
            bearish = pattern_features.get("pattern_bearish", 0)
            if bullish > 0 and bearish > 0:
                pattern_uncertainty = min(bullish, bearish) / max(bullish, bearish)
            else:
                pattern_uncertainty = 0.2
        uncertainty_sources[UncertaintySource.PATTERN_AMBIGUITY] = pattern_uncertainty
        
        # Indicator Conflict
        rsi = technical_indicators.get("rsi_14", 50)
        bb_position = technical_indicators.get("bb_position", 0.5)
        
        # RSI vs BB Position Conflict
        rsi_signal = "bullish" if rsi < 30 else ("bearish" if rsi > 70 else "neutral")
        bb_signal = "bullish" if bb_position < 0.2 else ("bearish" if bb_position > 0.8 else "neutral")
        
        if rsi_signal != bb_signal and rsi_signal != "neutral" and bb_signal != "neutral":
            indicator_uncertainty = 0.8
        else:
            indicator_uncertainty = 0.2
        uncertainty_sources[UncertaintySource.INDICATOR_CONFLICT] = indicator_uncertainty
        
        # Insufficient Data
        history_length = market_context.get("history_length", 50)
        if history_length < 20:
            data_insufficiency = 0.8
        elif history_length < 50:
            data_insufficiency = 0.4
        else:
            data_insufficiency = 0.1
        uncertainty_sources[UncertaintySource.INSUFFICIENT_DATA] = data_insufficiency
        
        return uncertainty_sources
    
    def _calibrate_confidence(self, raw_confidence: float, component_confidences: Dict[str, float]) -> float:
        """Kalibriere Confidence basierend auf historischen Daten"""
        try:
            if len(self.historical_predictions) < self.min_samples_for_calibration:
                # Nicht genügend Daten für Kalibrierung
                return raw_confidence
            
            # Einfache Isotonic Regression Calibration
            if self.calibration_method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(self.historical_predictions, self.historical_outcomes)
                calibrated = calibrator.predict([raw_confidence])[0]
                return float(calibrated)
            else:
                # Fallback: Einfache lineare Kalibrierung
                historical_mean = np.mean(self.historical_predictions)
                outcome_mean = np.mean(self.historical_outcomes)
                
                if historical_mean > 0:
                    calibration_factor = outcome_mean / historical_mean
                    return min(1.0, max(0.0, raw_confidence * calibration_factor))
                else:
                    return raw_confidence
                    
        except Exception:
            return raw_confidence
    
    def _calculate_risk_adjusted_confidence(
        self,
        calibrated_confidence: float,
        market_context: Dict[str, Any],
        uncertainty_sources: Dict[UncertaintySource, float]
    ) -> float:
        """Berechne Risk-Adjusted Confidence"""
        try:
            # Base Risk Adjustment
            volatility = market_context.get("volatility", 0.01)
            volatility_penalty = min(0.3, volatility * 15)  # Max 30% penalty
            
            # Uncertainty Penalty
            avg_uncertainty = sum(uncertainty_sources.values()) / len(uncertainty_sources)
            uncertainty_penalty = avg_uncertainty * 0.2  # Max 20% penalty
            
            # Market Regime Adjustment
            regime = market_context.get("market_regime", "unknown")
            regime_adjustments = {
                "trending": 0.0,     # No penalty
                "ranging": -0.1,     # 10% penalty
                "volatile": -0.2,    # 20% penalty
                "quiet": 0.05,       # 5% bonus
                "unknown": -0.15     # 15% penalty
            }
            regime_adjustment = regime_adjustments.get(regime, -0.1)
            
            # Apply Adjustments
            risk_adjusted = calibrated_confidence - volatility_penalty - uncertainty_penalty + regime_adjustment
            
            return min(1.0, max(0.0, risk_adjusted))
            
        except Exception:
            return calibrated_confidence * 0.8  # Conservative fallback
    
    def _calculate_validation_factors(
        self,
        component_confidences: Dict[str, float],
        uncertainty_sources: Dict[UncertaintySource, float],
        additional_factors: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Berechne Validation Factors"""
        validation_factors = {}
        
        # Component Consistency
        confidences = list(component_confidences.values())
        if len(confidences) > 1:
            consistency = 1.0 - (np.std(confidences) / max(np.mean(confidences), 1e-6))
            validation_factors["component_consistency"] = max(0.0, consistency)
        
        # Uncertainty Level
        avg_uncertainty = sum(uncertainty_sources.values()) / len(uncertainty_sources)
        validation_factors["uncertainty_level"] = 1.0 - avg_uncertainty
        
        # Signal Strength
        max_confidence = max(component_confidences.values())
        min_confidence = min(component_confidences.values())
        signal_strength = max_confidence - min_confidence
        validation_factors["signal_strength"] = signal_strength
        
        # Additional Factors
        if additional_factors:
            validation_factors.update(additional_factors)
        
        return validation_factors
    
    def _calculate_temporal_stability(self, component_confidences: Dict[str, float]) -> float:
        """Berechne Temporal Stability (vereinfacht)"""
        # Placeholder für Temporal Stability
        # In echter Implementation würde hier historische Confidence-Entwicklung analysiert
        avg_confidence = sum(component_confidences.values()) / len(component_confidences)
        return min(1.0, avg_confidence + 0.1)  # Leicht optimistisch
    
    def _calculate_reliability_score(
        self,
        component_confidences: Dict[str, float],
        uncertainty_sources: Dict[UncertaintySource, float],
        validation_factors: Dict[str, float]
    ) -> float:
        """Berechne Reliability Score"""
        try:
            # Component Reliability
            avg_component_confidence = sum(component_confidences.values()) / len(component_confidences)
            
            # Uncertainty Penalty
            avg_uncertainty = sum(uncertainty_sources.values()) / len(uncertainty_sources)
            uncertainty_penalty = avg_uncertainty * 0.3
            
            # Validation Bonus
            consistency = validation_factors.get("component_consistency", 0.5)
            validation_bonus = consistency * 0.2
            
            # Calculate Reliability
            reliability = avg_component_confidence - uncertainty_penalty + validation_bonus
            
            return min(1.0, max(0.0, reliability))
            
        except Exception:
            return 0.5
    
    def _calculate_prediction_interval(
        self,
        calibrated_confidence: float,
        uncertainty_sources: Dict[UncertaintySource, float]
    ) -> Tuple[float, float]:
        """Berechne Prediction Interval"""
        try:
            # Base Interval Width basierend auf Uncertainty
            avg_uncertainty = sum(uncertainty_sources.values()) / len(uncertainty_sources)
            interval_width = avg_uncertainty * 0.4  # Max 40% interval
            
            # Confidence-basierte Anpassung
            confidence_adjustment = (1.0 - calibrated_confidence) * 0.2
            total_width = interval_width + confidence_adjustment
            
            # Calculate Interval
            lower_bound = max(0.0, calibrated_confidence - total_width / 2)
            upper_bound = min(1.0, calibrated_confidence + total_width / 2)
            
            return (lower_bound, upper_bound)
            
        except Exception:
            return (max(0.0, calibrated_confidence - 0.2), min(1.0, calibrated_confidence + 0.2))
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Bestimme Confidence Level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _get_fallback_confidence_metrics(self) -> ConfidenceMetrics:
        """Fallback Confidence Metrics bei Fehlern"""
        return ConfidenceMetrics(
            overall_confidence=0.5,
            confidence_level=ConfidenceLevel.MODERATE,
            calibrated_confidence=0.5,
            uncertainty_sources={source: 0.5 for source in UncertaintySource},
            component_confidences={"fallback": 0.5},
            reliability_score=0.5,
            prediction_interval=(0.3, 0.7),
            validation_factors={"fallback": True},
            risk_adjusted_confidence=0.4,
            temporal_stability=0.5
        )
    
    def update_calibration(self, prediction_confidence: float, actual_outcome: float):
        """Update Calibration mit neuen Daten"""
        try:
            self.historical_predictions.append(prediction_confidence)
            self.historical_outcomes.append(actual_outcome)
            
            # Limit History Size
            max_history = 1000
            if len(self.historical_predictions) > max_history:
                self.historical_predictions = self.historical_predictions[-max_history:]
                self.historical_outcomes = self.historical_outcomes[-max_history:]
            
            self.calibration_updates += 1
            
        except Exception as e:
            self.logger.error(f"Error updating calibration: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Confidence Scorer Statistiken"""
        return {
            "scores_calculated": self.scores_calculated,
            "calibration_updates": self.calibration_updates,
            "historical_samples": len(self.historical_predictions),
            "calibration_method": self.calibration_method,
            "factor_weights": self.factor_weights,
            "uncertainty_threshold": self.uncertainty_threshold,
            "calibration_ready": len(self.historical_predictions) >= self.min_samples_for_calibration
        }


# Factory Function
def create_enhanced_confidence_scorer(config: Optional[Dict] = None) -> EnhancedConfidenceScorer:
    """
    Factory Function für Enhanced Confidence Scorer
    
    Args:
        config: Konfiguration für Confidence-Scoring
    
    Returns:
        EnhancedConfidenceScorer Instance
    """
    return EnhancedConfidenceScorer(config=config)
    confidence_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfidenceCalibration:
    """Kalibrierungs-Parameter für Confidence-Scores"""
    calibration_method: str = "isotonic"  # "isotonic", "platt", "beta"
    historical_accuracy: List[float] = field(default_factory=list)
    historical_confidences: List[float] = field(default_factory=list)
    calibration_curve: Optional[Any] = None
    last_calibration: Optional[float] = None

class ConfidenceScoring:
    """
    Bewertet und kalibriert Confidence-Scores für multimodale Trading-Predictions.
    Kombiniert verschiedene Unsicherheitsquellen und kalibriert Scores basierend auf historischer Performance.
    """
    
    def __init__(self):
        # Gewichtungen für verschiedene Komponenten
        self.component_weights = {
            "visual_patterns": 0.35,
            "numerical_indicators": 0.30,
            "ai_reasoning": 0.20,
            "market_context": 0.15
        }
        
        # Kalibrierungs-Objekte
        self.calibration_models = {
            "visual": ConfidenceCalibration(),
            "numerical": ConfidenceCalibration(),
            "multimodal": ConfidenceCalibration()
        }
        
        # Unsicherheits-Gewichtungen
        self.uncertainty_weights = {
            UncertaintySource.MODEL_UNCERTAINTY: 0.25,
            UncertaintySource.DATA_QUALITY: 0.20,
            UncertaintySource.MARKET_VOLATILITY: 0.20,
            UncertaintySource.PATTERN_AMBIGUITY: 0.15,
            UncertaintySource.INDICATOR_CONFLICT: 0.15,
            UncertaintySource.INSUFFICIENT_DATA: 0.05
        }
        
        # Historische Performance für Kalibrierung
        self.performance_history = []
        
        logger.info("ConfidenceScoring System initialisiert")
    
    def calculate_multimodal_confidence(self,
                                      visual_analysis: Any,
                                      indicator_analysis: Optional[Dict[Any, Any]],
                                      strategy_result: Any,
                                      market_data: Optional[Dict[str, Any]] = None) -> ConfidenceMetrics:
        """
        Berechnet umfassende Confidence-Metriken für multimodale Analyse.
        
        Args:
            visual_analysis: Ergebnis der visuellen Pattern-Analyse
            indicator_analysis: Ergebnisse der Indikator-Optimierung
            strategy_result: Generierte Trading-Strategien
            market_data: Zusätzliche Marktdaten
            
        Returns:
            ConfidenceMetrics mit detaillierten Confidence-Bewertungen
        """
        try:
            logger.info("Berechne multimodale Confidence-Metriken")
            
            # 1. Komponenten-Confidences berechnen
            component_confidences = self._calculate_component_confidences(
                visual_analysis, indicator_analysis, strategy_result, market_data
            )
            
            # 2. Unsicherheitsquellen analysieren
            uncertainty_sources = self._analyze_uncertainty_sources(
                visual_analysis, indicator_analysis, strategy_result, market_data
            )
            
            # 3. Gesamt-Confidence berechnen
            overall_confidence = self._calculate_overall_confidence(
                component_confidences, uncertainty_sources
            )
            
            # 4. Confidence kalibrieren
            calibrated_confidence = self._calibrate_confidence(
                overall_confidence, component_confidences
            )
            
            # 5. Reliability Score berechnen
            reliability_score = self._calculate_reliability_score(
                component_confidences, uncertainty_sources
            )
            
            # 6. Prediction Interval schätzen
            prediction_interval = self._estimate_prediction_interval(
                calibrated_confidence, uncertainty_sources
            )
            
            # 7. Confidence Level bestimmen
            confidence_level = self._determine_confidence_level(calibrated_confidence)
            
            result = ConfidenceMetrics(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                calibrated_confidence=calibrated_confidence,
                uncertainty_sources=uncertainty_sources,
                component_confidences=component_confidences,
                reliability_score=reliability_score,
                prediction_interval=prediction_interval,
                confidence_metadata={
                    "calculation_method": "multimodal_fusion",
                    "calibration_applied": True,
                    "uncertainty_analysis": True,
                    "component_count": len(component_confidences)
                }
            )
            
            logger.info(f"Confidence-Berechnung abgeschlossen: {calibrated_confidence:.3f} ({confidence_level.value})")
            return result
            
        except Exception as e:
            logger.exception(f"Fehler bei Confidence-Berechnung: {e}")
            return self._create_fallback_confidence_metrics(e)
    
    def update_calibration(self, 
                          predicted_confidence: float,
                          actual_outcome: bool,
                          analysis_type: str = "multimodal") -> None:
        """
        Aktualisiert Kalibrierungs-Modelle basierend auf tatsächlichen Ergebnissen.
        
        Args:
            predicted_confidence: Vorhergesagte Confidence
            actual_outcome: Tatsächliches Ergebnis (True/False)
            analysis_type: Typ der Analyse ("visual", "numerical", "multimodal")
        """
        try:
            if analysis_type in self.calibration_models:
                calibration = self.calibration_models[analysis_type]
                
                # Historische Daten aktualisieren
                calibration.historical_confidences.append(predicted_confidence)
                calibration.historical_accuracy.append(1.0 if actual_outcome else 0.0)
                
                # Limitiere Historie auf letzte 1000 Einträge
                if len(calibration.historical_confidences) > 1000:
                    calibration.historical_confidences = calibration.historical_confidences[-1000:]
                    calibration.historical_accuracy = calibration.historical_accuracy[-1000:]
                
                # Re-kalibriere wenn genügend Daten vorhanden
                if len(calibration.historical_confidences) >= 50:
                    self._recalibrate_model(analysis_type)
                
                logger.info(f"Kalibrierung für {analysis_type} aktualisiert")
                
        except Exception as e:
            logger.exception(f"Fehler bei Kalibrierungs-Update: {e}")
    
    def _calculate_component_confidences(self,
                                       visual_analysis: Any,
                                       indicator_analysis: Optional[Dict[Any, Any]],
                                       strategy_result: Any,
                                       market_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Berechnet Confidence für einzelne Komponenten"""
        try:
            confidences = {}
            
            # Visual Pattern Confidence
            confidences["visual_patterns"] = self._calculate_visual_confidence(visual_analysis)
            
            # Numerical Indicator Confidence
            confidences["numerical_indicators"] = self._calculate_numerical_confidence(indicator_analysis)
            
            # AI Reasoning Confidence
            confidences["ai_reasoning"] = self._calculate_ai_confidence(strategy_result)
            
            # Market Context Confidence
            confidences["market_context"] = self._calculate_market_confidence(market_data)
            
            return confidences
            
        except Exception as e:
            logger.exception(f"Komponenten-Confidence-Berechnung fehlgeschlagen: {e}")
            return {"error": 0.0}
    
    def _calculate_visual_confidence(self, visual_analysis: Any) -> float:
        """Berechnet Confidence für visuelle Pattern-Analyse"""
        try:
            if not visual_analysis.patterns:
                return 0.1  # Minimale Confidence ohne Patterns
            
            # Basis-Confidence aus Pattern-Analyse
            base_confidence = visual_analysis.confidence_score
            
            # Adjustierung basierend auf Pattern-Qualität
            pattern_quality_bonus = 0.0
            for pattern in visual_analysis.patterns[:3]:  # Top 3 Patterns
                if pattern.confidence > 0.8:
                    pattern_quality_bonus += 0.1
                elif pattern.confidence > 0.6:
                    pattern_quality_bonus += 0.05
            
            # Adjustierung basierend auf Pattern-Konsistenz
            consistency_bonus = 0.0
            if len(visual_analysis.patterns) >= 2:
                directions = [p.direction for p in visual_analysis.patterns if p.direction]
                if directions and len(set(directions)) == 1:  # Alle Patterns zeigen in gleiche Richtung
                    consistency_bonus = 0.1
            
            # Adjustierung basierend auf Marktstruktur-Klarheit
            structure_bonus = 0.0
            market_structure = visual_analysis.market_structure
            if market_structure.get("trend") in ["bullish", "bearish"]:
                structure_bonus = 0.05
            
            total_confidence = base_confidence + pattern_quality_bonus + consistency_bonus + structure_bonus
            return min(total_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Visuelle Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.3  # Default
    
    def _calculate_numerical_confidence(self, indicator_analysis: Optional[Dict[Any, Any]]) -> float:
        """Berechnet Confidence für numerische Indikator-Analyse"""
        try:
            if not indicator_analysis:
                return 0.2  # Niedrige Confidence ohne Indikator-Analyse
            
            # Performance-Scores der Indikatoren
            performance_scores = [result.performance_score for result in indicator_analysis.values()]
            
            if not performance_scores:
                return 0.2
            
            # Basis-Confidence aus durchschnittlicher Performance
            avg_performance = np.mean(performance_scores)
            base_confidence = min(avg_performance, 1.0)
            
            # Bonus für konsistente Performance
            consistency_bonus = 0.0
            if len(performance_scores) > 1:
                performance_std = np.std(performance_scores)
                if performance_std < 0.1:  # Niedrige Standardabweichung = hohe Konsistenz
                    consistency_bonus = 0.1
            
            # Bonus für hohe absolute Performance
            excellence_bonus = 0.0
            if max(performance_scores) > 0.5:
                excellence_bonus = 0.05
            
            # Malus für negative Performance
            negative_penalty = 0.0
            negative_count = sum(1 for score in performance_scores if score < 0)
            if negative_count > 0:
                negative_penalty = negative_count * 0.1
            
            total_confidence = base_confidence + consistency_bonus + excellence_bonus - negative_penalty
            return max(min(total_confidence, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Numerische Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.3  # Default
    
    def _calculate_ai_confidence(self, strategy_result: StrategyGenerationResult) -> float:
        """Berechnet Confidence für AI-Reasoning"""
        try:
            # Basis-Confidence aus primärer Strategie
            base_confidence = strategy_result.primary_strategy.confidence_score
            
            # Bonus für multiple konsistente Strategien
            consistency_bonus = 0.0
            if len(strategy_result.alternative_strategies) >= 2:
                strategy_types = [s.strategy_type for s in strategy_result.alternative_strategies]
                if len(set(strategy_types)) <= 2:  # Ähnliche Strategie-Typen
                    consistency_bonus = 0.05
            
            # Bonus für starkes aktuelles Signal
            signal_bonus = 0.0
            current_signal = strategy_result.current_signal
            if current_signal.confidence > 0.7:
                signal_bonus = 0.1
            elif current_signal.confidence > 0.5:
                signal_bonus = 0.05
            
            # Adjustierung basierend auf Confidence-Breakdown
            breakdown_adjustment = 0.0
            confidence_breakdown = strategy_result.confidence_breakdown
            if isinstance(confidence_breakdown, dict):
                avg_component_confidence = np.mean(list(confidence_breakdown.values()))
                if avg_component_confidence > 0.6:
                    breakdown_adjustment = 0.05
            
            total_confidence = base_confidence + consistency_bonus + signal_bonus + breakdown_adjustment
            return min(total_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"AI-Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.4  # Default
    
    def _calculate_market_confidence(self, market_data: Optional[Dict[str, Any]]) -> float:
        """Berechnet Confidence basierend auf Marktkontext"""
        try:
            if not market_data:
                return 0.5  # Neutrale Confidence ohne Marktdaten
            
            base_confidence = 0.5
            
            # Adjustierung basierend auf Volatilität
            volatility = market_data.get("volatility", "medium")
            if volatility == "low":
                base_confidence += 0.1  # Niedrige Volatilität = höhere Vorhersagbarkeit
            elif volatility == "high":
                base_confidence -= 0.1  # Hohe Volatilität = niedrigere Vorhersagbarkeit
            
            # Adjustierung basierend auf Liquidität
            liquidity = market_data.get("liquidity", "normal")
            if liquidity == "high":
                base_confidence += 0.05
            elif liquidity == "low":
                base_confidence -= 0.1
            
            # Adjustierung basierend auf Markt-Regime
            market_regime = market_data.get("regime", "normal")
            if market_regime == "trending":
                base_confidence += 0.05
            elif market_regime == "crisis":
                base_confidence -= 0.2
            
            return max(min(base_confidence, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Markt-Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.5  # Default
    
    def _analyze_uncertainty_sources(self,
                                   visual_analysis: Any,
                                   indicator_analysis: Optional[Dict[Any, Any]],
                                   strategy_result: Any,
                                   market_data: Optional[Dict[str, Any]]) -> Dict[UncertaintySource, float]:
        """Analysiert verschiedene Unsicherheitsquellen"""
        try:
            uncertainties = {}
            
            # Model Uncertainty
            uncertainties[UncertaintySource.MODEL_UNCERTAINTY] = self._assess_model_uncertainty(strategy_result)
            
            # Data Quality Uncertainty
            uncertainties[UncertaintySource.DATA_QUALITY] = self._assess_data_quality_uncertainty(visual_analysis, indicator_analysis)
            
            # Market Volatility Uncertainty
            uncertainties[UncertaintySource.MARKET_VOLATILITY] = self._assess_volatility_uncertainty(market_data)
            
            # Pattern Ambiguity Uncertainty
            uncertainties[UncertaintySource.PATTERN_AMBIGUITY] = self._assess_pattern_ambiguity(visual_analysis)
            
            # Indicator Conflict Uncertainty
            uncertainties[UncertaintySource.INDICATOR_CONFLICT] = self._assess_indicator_conflict(indicator_analysis)
            
            # Insufficient Data Uncertainty
            uncertainties[UncertaintySource.INSUFFICIENT_DATA] = self._assess_data_sufficiency(visual_analysis, indicator_analysis)
            
            return uncertainties
            
        except Exception as e:
            logger.exception(f"Unsicherheits-Analyse fehlgeschlagen: {e}")
            return {source: 0.5 for source in UncertaintySource}
    
    def _assess_model_uncertainty(self, strategy_result: StrategyGenerationResult) -> float:
        """Bewertet Model-Unsicherheit"""
        try:
            # Niedrige Unsicherheit wenn Strategien konsistent sind
            if len(strategy_result.alternative_strategies) >= 2:
                confidence_scores = [s.confidence_score for s in strategy_result.alternative_strategies]
                confidence_std = np.std(confidence_scores)
                return min(confidence_std * 2, 1.0)  # Hohe Standardabweichung = hohe Unsicherheit
            
            return 0.3  # Default moderate Unsicherheit
            
        except Exception:
            return 0.5
    
    def _assess_data_quality_uncertainty(self, visual_analysis, indicator_analysis) -> float:
        """Bewertet Datenqualitäts-Unsicherheit"""
        try:
            uncertainty = 0.0
            
            # Visuelle Datenqualität
            if "error" in visual_analysis.analysis_metadata:
                uncertainty += 0.3
            
            # Indikator-Datenqualität
            if indicator_analysis:
                error_count = sum(1 for result in indicator_analysis.values() 
                                if "error" in result.optimization_metadata)
                uncertainty += (error_count / len(indicator_analysis)) * 0.3
            
            return min(uncertainty, 1.0)
            
        except Exception:
            return 0.2
    
    def _assess_volatility_uncertainty(self, market_data) -> float:
        """Bewertet Volatilitäts-Unsicherheit"""
        try:
            if not market_data:
                return 0.4
            
            volatility = market_data.get("volatility", "medium")
            if volatility == "very_high":
                return 0.8
            elif volatility == "high":
                return 0.6
            elif volatility == "medium":
                return 0.4
            elif volatility == "low":
                return 0.2
            else:
                return 0.1  # very_low
                
        except Exception:
            return 0.4
    
    def _assess_pattern_ambiguity(self, visual_analysis) -> float:
        """Bewertet Pattern-Ambiguität"""
        try:
            if not visual_analysis.patterns:
                return 0.8  # Hohe Unsicherheit ohne Patterns
            
            # Niedrige Ambiguität wenn Patterns konsistent sind
            directions = [p.direction for p in visual_analysis.patterns if p.direction]
            if directions:
                unique_directions = len(set(directions))
                if unique_directions == 1:
                    return 0.2  # Alle Patterns zeigen in gleiche Richtung
                elif unique_directions == 2:
                    return 0.5  # Gemischte Signale
                else:
                    return 0.8  # Sehr gemischte Signale
            
            return 0.6  # Default
            
        except Exception:
            return 0.5
    
    def _assess_indicator_conflict(self, indicator_analysis) -> float:
        """Bewertet Indikator-Konflikte"""
        try:
            if not indicator_analysis or len(indicator_analysis) < 2:
                return 0.3  # Niedrige Unsicherheit mit wenigen Indikatoren
            
            # Prüfe Performance-Konsistenz
            performance_scores = [result.performance_score for result in indicator_analysis.values()]
            performance_range = max(performance_scores) - min(performance_scores)
            
            # Hohe Range = hohe Konflikte
            return min(performance_range, 1.0)
            
        except Exception:
            return 0.4
    
    def _assess_data_sufficiency(self, visual_analysis, indicator_analysis) -> float:
        """Bewertet Datensuffizienz"""
        try:
            insufficiency = 0.0
            
            # Visuelle Daten
            data_points = visual_analysis.analysis_metadata.get("data_points", 0)
            if data_points < 100:
                insufficiency += 0.3
            elif data_points < 50:
                insufficiency += 0.5
            
            # Indikator-Daten
            if indicator_analysis:
                for result in indicator_analysis.values():
                    if result.optimization_metadata.get("trials", 0) < 50:
                        insufficiency += 0.1
            
            return min(insufficiency, 1.0)
            
        except Exception:
            return 0.3
    
    def _calculate_overall_confidence(self, 
                                    component_confidences: Dict[str, float],
                                    uncertainty_sources: Dict[UncertaintySource, float]) -> float:
        """Berechnet Gesamt-Confidence"""
        try:
            # Gewichtete Kombination der Komponenten-Confidences
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for component, confidence in component_confidences.items():
                if component in self.component_weights:
                    weight = self.component_weights[component]
                    weighted_confidence += confidence * weight
                    total_weight += weight
            
            if total_weight > 0:
                base_confidence = weighted_confidence / total_weight
            else:
                base_confidence = 0.5
            
            # Unsicherheits-Adjustierung
            uncertainty_penalty = 0.0
            for source, uncertainty in uncertainty_sources.items():
                weight = self.uncertainty_weights.get(source, 0.1)
                uncertainty_penalty += uncertainty * weight
            
            # Finale Confidence
            final_confidence = base_confidence * (1 - uncertainty_penalty)
            return max(min(final_confidence, 1.0), 0.0)
            
        except Exception as e:
            logger.exception(f"Gesamt-Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.5
    
    def _calibrate_confidence(self, 
                            raw_confidence: float,
                            component_confidences: Dict[str, float]) -> float:
        """Kalibriert Confidence basierend auf historischer Performance"""
        try:
            calibration = self.calibration_models["multimodal"]
            
            if (calibration.calibration_curve is not None and 
                len(calibration.historical_confidences) >= 50):
                
                # Verwende kalibriertes Modell
                calibrated = calibration.calibration_curve.predict([raw_confidence])[0]
                return max(min(calibrated, 1.0), 0.0)
            else:
                # Einfache Kalibrierung ohne historische Daten
                # Konservative Adjustierung
                if raw_confidence > 0.8:
                    return raw_confidence * 0.9  # Reduziere sehr hohe Confidence
                elif raw_confidence < 0.2:
                    return raw_confidence * 1.1  # Erhöhe sehr niedrige Confidence leicht
                else:
                    return raw_confidence
                    
        except Exception as e:
            logger.warning(f"Confidence-Kalibrierung fehlgeschlagen: {e}")
            return raw_confidence
    
    def _calculate_reliability_score(self,
                                   component_confidences: Dict[str, float],
                                   uncertainty_sources: Dict[UncertaintySource, float]) -> float:
        """Berechnet Reliability Score"""
        try:
            # Basis-Reliability aus Komponenten-Konsistenz
            confidences = list(component_confidences.values())
            if len(confidences) > 1:
                confidence_std = np.std(confidences)
                consistency_score = 1.0 - min(confidence_std, 1.0)
            else:
                consistency_score = 0.5
            
            # Adjustierung basierend auf Unsicherheiten
            avg_uncertainty = np.mean(list(uncertainty_sources.values()))
            uncertainty_penalty = avg_uncertainty * 0.5
            
            reliability = consistency_score - uncertainty_penalty
            return max(min(reliability, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Reliability-Score-Berechnung fehlgeschlagen: {e}")
            return 0.5
    
    def _estimate_prediction_interval(self,
                                    calibrated_confidence: float,
                                    uncertainty_sources: Dict[UncertaintySource, float]) -> Tuple[float, float]:
        """Schätzt Prediction Interval"""
        try:
            # Basis-Interval basierend auf Confidence
            base_width = (1.0 - calibrated_confidence) * 0.5
            
            # Adjustierung basierend auf Unsicherheiten
            avg_uncertainty = np.mean(list(uncertainty_sources.values()))
            uncertainty_adjustment = avg_uncertainty * 0.3
            
            total_width = base_width + uncertainty_adjustment
            
            # Symmetrisches Interval um Confidence
            lower_bound = max(calibrated_confidence - total_width, 0.0)
            upper_bound = min(calibrated_confidence + total_width, 1.0)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.warning(f"Prediction-Interval-Schätzung fehlgeschlagen: {e}")
            return (0.0, 1.0)
    
    def _determine_confidence_level(self, calibrated_confidence: float) -> ConfidenceLevel:
        """Bestimmt Confidence-Level-Kategorie"""
        if calibrated_confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif calibrated_confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif calibrated_confidence >= 0.4:
            return ConfidenceLevel.MODERATE
        elif calibrated_confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _recalibrate_model(self, analysis_type: str) -> None:
        """Re-kalibriert Confidence-Modell"""
        try:
            calibration = self.calibration_models[analysis_type]
            
            if len(calibration.historical_confidences) < 50:
                return
            
            X = np.array(calibration.historical_confidences).reshape(-1, 1)
            y = np.array(calibration.historical_accuracy)
            
            # Isotonic Regression für Kalibrierung
            if calibration.calibration_method == "isotonic":
                calibration.calibration_curve = IsotonicRegression(out_of_bounds='clip')
                calibration.calibration_curve.fit(X.flatten(), y)
            
            calibration.last_calibration = len(calibration.historical_confidences)
            
            logger.info(f"Modell für {analysis_type} re-kalibriert mit {len(y)} Datenpunkten")
            
        except Exception as e:
            logger.exception(f"Re-Kalibrierung für {analysis_type} fehlgeschlagen: {e}")
    
    def _create_fallback_confidence_metrics(self, error: Exception) -> ConfidenceMetrics:
        """Erstellt Fallback-Confidence-Metriken bei Fehlern"""
        return ConfidenceMetrics(
            overall_confidence=0.3,
            confidence_level=ConfidenceLevel.LOW,
            calibrated_confidence=0.3,
            uncertainty_sources={source: 0.7 for source in UncertaintySource},
            component_confidences={"error": 0.0},
            reliability_score=0.1,
            prediction_interval=(0.0, 0.6),
            confidence_metadata={"error": str(error), "fallback": True}
        )
    
    def get_confidence_explanation(self, confidence_metrics: ConfidenceMetrics) -> str:
        """Erstellt menschenlesbare Erklärung der Confidence-Bewertung"""
        try:
            explanation_parts = []
            
            # Gesamt-Bewertung
            level = confidence_metrics.confidence_level.value.replace('_', ' ').title()
            explanation_parts.append(f"Overall confidence: {level} ({confidence_metrics.calibrated_confidence:.1%})")
            
            # Stärkste Komponente
            if confidence_metrics.component_confidences:
                best_component = max(confidence_metrics.component_confidences.items(), key=lambda x: x[1])
                explanation_parts.append(f"Strongest signal from: {best_component[0]} ({best_component[1]:.1%})")
            
            # Hauptunsicherheitsquelle
            if confidence_metrics.uncertainty_sources:
                main_uncertainty = max(confidence_metrics.uncertainty_sources.items(), key=lambda x: x[1])
                explanation_parts.append(f"Main uncertainty: {main_uncertainty[0].value.replace('_', ' ')} ({main_uncertainty[1]:.1%})")
            
            # Reliability
            reliability_desc = "High" if confidence_metrics.reliability_score > 0.7 else "Moderate" if confidence_metrics.reliability_score > 0.4 else "Low"
            explanation_parts.append(f"Reliability: {reliability_desc} ({confidence_metrics.reliability_score:.1%})")
            
            return ". ".join(explanation_parts)
            
        except Exception as e:
            logger.warning(f"Confidence-Erklärung-Erstellung fehlgeschlagen: {e}")
            return f"Confidence: {confidence_metrics.calibrated_confidence:.1%} (explanation unavailable)"