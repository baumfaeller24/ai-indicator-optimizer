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

from .visual_pattern_analyzer import PatternAnalysisResult, VisualPattern
from .numerical_indicator_optimizer import OptimizationResult, IndicatorType
from .multimodal_strategy_generator import StrategyGenerationResult, TradingStrategy

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
                                      visual_analysis: PatternAnalysisResult,
                                      indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                      strategy_result: StrategyGenerationResult,
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
                                       visual_analysis: PatternAnalysisResult,
                                       indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                       strategy_result: StrategyGenerationResult,
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
    
    def _calculate_visual_confidence(self, visual_analysis: PatternAnalysisResult) -> float:
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
    
    def _calculate_numerical_confidence(self, indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]]) -> float:
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
                                   visual_analysis: PatternAnalysisResult,
                                   indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                   strategy_result: StrategyGenerationResult,
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