#!/usr/bin/env python3
"""
Multimodal Confidence Scorer für kombinierte Vision+Text-Analyse
U3 - Unified Multimodal Flow Integration - Day 3

Features:
- Cross-Modal Validation zwischen Vision und Text
- Uncertainty Quantification für Confidence-Scores
- Adaptive Weighting basierend auf Kontext
- Confidence Calibration für bessere Accuracy
- Multi-Modal Agreement Scoring
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Import existing components
from .dynamic_fusion_agent import AdaptivePrompt, FusionStrategy
from .chart_to_strategy_pipeline import StrategyResult, StrategySignal, RecognizedPattern


class ConfidenceLevel(Enum):
    """Confidence-Level-Kategorien"""
    VERY_LOW = "very_low"      # 0.0 - 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # 0.9 - 1.0


class UncertaintyType(Enum):
    """Typen von Uncertainty"""
    ALEATORIC = "aleatoric"        # Inherent data uncertainty
    EPISTEMIC = "epistemic"        # Model uncertainty
    CROSS_MODAL = "cross_modal"    # Disagreement between modalities
    TEMPORAL = "temporal"          # Time-based uncertainty


@dataclass
class VisionResult:
    """Vision-Analyse-Ergebnis"""
    confidence: float
    patterns: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    support_resistance: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    processing_time: float
    model_uncertainty: float
    metadata: Dict[str, Any]


@dataclass
class TextResult:
    """Text-Feature-Analyse-Ergebnis"""
    confidence: float
    signal_strength: float
    technical_scores: Dict[str, float]
    market_context: Dict[str, Any]
    risk_assessment: Dict[str, float]
    processing_time: float
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class FusionContext:
    """Kontext für Multimodal-Fusion"""
    timeframe: str
    market_volatility: float
    data_quality: float
    historical_performance: Dict[str, float]
    fusion_strategy: FusionStrategy
    adaptive_weights: Dict[str, float]
    context_complexity: float


@dataclass
class UncertaintyQuantification:
    """Uncertainty-Quantifizierung"""
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    cross_modal_uncertainty: float
    temporal_uncertainty: float
    uncertainty_breakdown: Dict[str, float]
    confidence_interval: Tuple[float, float]
    reliability_score: float


@dataclass
class ConfidenceScore:
    """Comprehensive Confidence-Score"""
    combined_confidence: float
    vision_confidence: float
    text_confidence: float
    cross_modal_agreement: float
    uncertainty: UncertaintyQuantification
    adaptive_weights: Dict[str, float]
    calibrated_confidence: float
    confidence_level: ConfidenceLevel
    reliability_metrics: Dict[str, float]
    score_breakdown: Dict[str, float]
    generation_timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ConfidenceConfig:
    """Konfiguration für Confidence Scorer"""
    vision_weight_default: float = 0.6
    text_weight_default: float = 0.4
    agreement_weight: float = 0.2
    uncertainty_penalty: float = 0.1
    calibration_samples: int = 1000
    confidence_smoothing: float = 0.05
    temporal_decay: float = 0.95
    cross_modal_threshold: float = 0.3
    reliability_threshold: float = 0.7


class ConfidenceCalibrator:
    """Kalibriert Confidence-Scores für bessere Accuracy"""
    
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        self.calibration_data = []
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def add_calibration_sample(self, predicted_confidence: float, actual_outcome: float):
        """Fügt Kalibrierungs-Sample hinzu"""
        self.calibration_data.append((predicted_confidence, actual_outcome))
        
        # Keep only recent samples
        if len(self.calibration_data) > 10000:
            self.calibration_data = self.calibration_data[-5000:]
    
    def fit_calibrator(self):
        """Trainiert Confidence-Kalibrator"""
        if len(self.calibration_data) < 50:
            self.logger.warning("Insufficient calibration data")
            return False
        
        try:
            X = np.array([sample[0] for sample in self.calibration_data]).reshape(-1, 1)
            y = np.array([sample[1] for sample in self.calibration_data])
            
            if self.method == "isotonic":
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(X.flatten(), y)
            else:
                # Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression
                self.calibrator = LogisticRegression()
                self.calibrator.fit(X, y)
            
            self.is_fitted = True
            self.logger.info(f"Calibrator fitted with {len(self.calibration_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibrator fitting failed: {e}")
            return False
    
    def calibrate_confidence(self, confidence: float) -> float:
        """Kalibriert Confidence-Score"""
        if not self.is_fitted or self.calibrator is None:
            return confidence
        
        try:
            if self.method == "isotonic":
                calibrated = self.calibrator.predict([confidence])[0]
            else:
                calibrated = self.calibrator.predict_proba([[confidence]])[0][1]
            
            # Ensure valid range
            return max(0.0, min(1.0, calibrated))
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return confidence
    
    def get_calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Erstellt Kalibrierungs-Kurve für Analyse"""
        if len(self.calibration_data) < 20:
            return np.array([]), np.array([])
        
        confidences = np.array([sample[0] for sample in self.calibration_data])
        outcomes = np.array([sample[1] for sample in self.calibration_data])
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = outcomes[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        return np.array(bin_centers), np.array(bin_accuracies)


class UncertaintyEstimator:
    """Schätzt verschiedene Typen von Uncertainty"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.historical_uncertainties = []
    
    def estimate_aleatoric_uncertainty(self, 
                                     vision_result: VisionResult,
                                     text_result: TextResult) -> float:
        """Schätzt Aleatoric (Data) Uncertainty"""
        try:
            # Data quality indicators
            vision_quality = 1.0 - vision_result.model_uncertainty
            text_quality = len(text_result.technical_scores) / 10.0  # Normalize by expected features
            
            # Processing time as quality indicator (faster = more certain)
            vision_time_quality = max(0, 1 - vision_result.processing_time / 10.0)
            text_time_quality = max(0, 1 - text_result.processing_time / 5.0)
            
            # Combine quality indicators
            overall_quality = np.mean([vision_quality, text_quality, vision_time_quality, text_time_quality])
            
            # Convert quality to uncertainty (inverse relationship)
            aleatoric_uncertainty = 1.0 - overall_quality
            
            return max(0.0, min(1.0, aleatoric_uncertainty))
            
        except Exception as e:
            self.logger.error(f"Aleatoric uncertainty estimation failed: {e}")
            return 0.5  # Default moderate uncertainty
    
    def estimate_epistemic_uncertainty(self,
                                     vision_result: VisionResult,
                                     text_result: TextResult) -> float:
        """Schätzt Epistemic (Model) Uncertainty"""
        try:
            # Model confidence variance
            vision_conf_var = vision_result.model_uncertainty
            
            # Feature importance variance as proxy for model uncertainty
            if text_result.feature_importance:
                importance_values = list(text_result.feature_importance.values())
                text_conf_var = np.var(importance_values) if len(importance_values) > 1 else 0.0
            else:
                text_conf_var = 0.5
            
            # Pattern confidence variance
            if vision_result.patterns:
                pattern_confidences = [p.get("confidence", 0.5) for p in vision_result.patterns]
                pattern_var = np.var(pattern_confidences) if len(pattern_confidences) > 1 else 0.0
            else:
                pattern_var = 0.5
            
            # Combine uncertainties
            epistemic_uncertainty = np.mean([vision_conf_var, text_conf_var, pattern_var])
            
            return max(0.0, min(1.0, epistemic_uncertainty))
            
        except Exception as e:
            self.logger.error(f"Epistemic uncertainty estimation failed: {e}")
            return 0.3  # Default low-moderate uncertainty
    
    def estimate_cross_modal_uncertainty(self,
                                       vision_result: VisionResult,
                                       text_result: TextResult,
                                       cross_modal_agreement: float) -> float:
        """Schätzt Cross-Modal Uncertainty basierend auf Disagreement"""
        try:
            # Direct disagreement measure
            disagreement = 1.0 - cross_modal_agreement
            
            # Confidence difference as additional indicator
            conf_diff = abs(vision_result.confidence - text_result.confidence)
            
            # Signal strength consistency
            vision_signal_strength = np.mean([p.get("confidence", 0.5) for p in vision_result.patterns]) if vision_result.patterns else 0.5
            text_signal_strength = text_result.signal_strength
            signal_diff = abs(vision_signal_strength - text_signal_strength)
            
            # Combine disagreement indicators
            cross_modal_uncertainty = np.mean([disagreement, conf_diff, signal_diff])
            
            return max(0.0, min(1.0, cross_modal_uncertainty))
            
        except Exception as e:
            self.logger.error(f"Cross-modal uncertainty estimation failed: {e}")
            return 0.4  # Default moderate uncertainty
    
    def estimate_temporal_uncertainty(self,
                                    current_confidence: float,
                                    fusion_context: FusionContext) -> float:
        """Schätzt Temporal Uncertainty basierend auf historischer Performance"""
        try:
            # Historical performance variance
            if fusion_context.historical_performance:
                hist_values = list(fusion_context.historical_performance.values())
                if len(hist_values) > 1:
                    hist_variance = np.var(hist_values)
                    hist_mean = np.mean(hist_values)
                    
                    # Deviation from historical mean
                    deviation = abs(current_confidence - hist_mean)
                    
                    # Combine variance and deviation
                    temporal_uncertainty = (hist_variance + deviation) / 2.0
                else:
                    temporal_uncertainty = 0.3  # Moderate uncertainty for single sample
            else:
                temporal_uncertainty = 0.5  # High uncertainty without history
            
            # Market volatility impact
            volatility_impact = fusion_context.market_volatility * 0.3
            temporal_uncertainty += volatility_impact
            
            return max(0.0, min(1.0, temporal_uncertainty))
            
        except Exception as e:
            self.logger.error(f"Temporal uncertainty estimation failed: {e}")
            return 0.4  # Default moderate uncertainty
    
    def quantify_uncertainty(self,
                           vision_result: VisionResult,
                           text_result: TextResult,
                           cross_modal_agreement: float,
                           fusion_context: FusionContext,
                           combined_confidence: float) -> UncertaintyQuantification:
        """Vollständige Uncertainty-Quantifizierung"""
        try:
            # Estimate individual uncertainty types
            aleatoric = self.estimate_aleatoric_uncertainty(vision_result, text_result)
            epistemic = self.estimate_epistemic_uncertainty(vision_result, text_result)
            cross_modal = self.estimate_cross_modal_uncertainty(vision_result, text_result, cross_modal_agreement)
            temporal = self.estimate_temporal_uncertainty(combined_confidence, fusion_context)
            
            # Total uncertainty (not simple sum due to correlations)
            uncertainty_components = [aleatoric, epistemic, cross_modal, temporal]
            total_uncertainty = np.sqrt(np.mean(np.square(uncertainty_components)))
            
            # Confidence interval estimation
            uncertainty_std = np.std(uncertainty_components)
            conf_lower = max(0.0, combined_confidence - 1.96 * uncertainty_std)
            conf_upper = min(1.0, combined_confidence + 1.96 * uncertainty_std)
            
            # Reliability score (inverse of total uncertainty)
            reliability_score = 1.0 - total_uncertainty
            
            # Uncertainty breakdown
            uncertainty_breakdown = {
                "aleatoric_pct": aleatoric / max(total_uncertainty, 1e-6) * 100,
                "epistemic_pct": epistemic / max(total_uncertainty, 1e-6) * 100,
                "cross_modal_pct": cross_modal / max(total_uncertainty, 1e-6) * 100,
                "temporal_pct": temporal / max(total_uncertainty, 1e-6) * 100
            }
            
            return UncertaintyQuantification(
                total_uncertainty=total_uncertainty,
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
                cross_modal_uncertainty=cross_modal,
                temporal_uncertainty=temporal,
                uncertainty_breakdown=uncertainty_breakdown,
                confidence_interval=(conf_lower, conf_upper),
                reliability_score=reliability_score
            )
            
        except Exception as e:
            self.logger.error(f"Uncertainty quantification failed: {e}")
            # Return default uncertainty
            return UncertaintyQuantification(
                total_uncertainty=0.5,
                aleatoric_uncertainty=0.3,
                epistemic_uncertainty=0.3,
                cross_modal_uncertainty=0.4,
                temporal_uncertainty=0.4,
                uncertainty_breakdown={"aleatoric_pct": 25, "epistemic_pct": 25, "cross_modal_pct": 25, "temporal_pct": 25},
                confidence_interval=(0.3, 0.7),
                reliability_score=0.5
            )


class CrossModalValidator:
    """Validiert Agreement zwischen Vision und Text Modalitäten"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_cross_modal_agreement(self,
                                      vision_result: VisionResult,
                                      text_result: TextResult) -> float:
        """Berechnet Cross-Modal Agreement Score"""
        try:
            agreement_factors = []
            
            # 1. Confidence Agreement
            conf_agreement = 1.0 - abs(vision_result.confidence - text_result.confidence)
            agreement_factors.append(conf_agreement)
            
            # 2. Trend Direction Agreement
            vision_trend = self.extract_trend_from_vision(vision_result)
            text_trend = self.extract_trend_from_text(text_result)
            trend_agreement = self.calculate_trend_agreement(vision_trend, text_trend)
            agreement_factors.append(trend_agreement)
            
            # 3. Signal Strength Agreement
            vision_strength = self.extract_signal_strength_from_vision(vision_result)
            text_strength = text_result.signal_strength
            strength_agreement = 1.0 - abs(vision_strength - text_strength)
            agreement_factors.append(strength_agreement)
            
            # 4. Risk Assessment Agreement
            vision_risk = self.extract_risk_from_vision(vision_result)
            text_risk = np.mean(list(text_result.risk_assessment.values())) if text_result.risk_assessment else 0.5
            risk_agreement = 1.0 - abs(vision_risk - text_risk)
            agreement_factors.append(risk_agreement)
            
            # 5. Technical Indicator Agreement
            tech_agreement = self.calculate_technical_agreement(vision_result, text_result)
            agreement_factors.append(tech_agreement)
            
            # Weighted average of agreement factors
            weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Confidence and trend are most important
            cross_modal_agreement = np.average(agreement_factors, weights=weights)
            
            return max(0.0, min(1.0, cross_modal_agreement))
            
        except Exception as e:
            self.logger.error(f"Cross-modal agreement calculation failed: {e}")
            return 0.5  # Default moderate agreement
    
    def extract_trend_from_vision(self, vision_result: VisionResult) -> str:
        """Extrahiert Trend-Richtung aus Vision-Analyse"""
        try:
            if vision_result.trend_analysis:
                return vision_result.trend_analysis.get("direction", "neutral")
            
            # Fallback: analyze patterns for trend
            if vision_result.patterns:
                bullish_patterns = sum(1 for p in vision_result.patterns if "bullish" in p.get("description", "").lower())
                bearish_patterns = sum(1 for p in vision_result.patterns if "bearish" in p.get("description", "").lower())
                
                if bullish_patterns > bearish_patterns:
                    return "bullish"
                elif bearish_patterns > bullish_patterns:
                    return "bearish"
            
            return "neutral"
            
        except Exception:
            return "neutral"
    
    def extract_trend_from_text(self, text_result: TextResult) -> str:
        """Extrahiert Trend-Richtung aus Text-Features"""
        try:
            # Analyze technical scores for trend
            if text_result.technical_scores:
                bullish_indicators = 0
                bearish_indicators = 0
                
                for indicator, score in text_result.technical_scores.items():
                    if score > 0.6:
                        bullish_indicators += 1
                    elif score < 0.4:
                        bearish_indicators += 1
                
                if bullish_indicators > bearish_indicators:
                    return "bullish"
                elif bearish_indicators > bullish_indicators:
                    return "bearish"
            
            return "neutral"
            
        except Exception:
            return "neutral"
    
    def calculate_trend_agreement(self, vision_trend: str, text_trend: str) -> float:
        """Berechnet Trend-Agreement zwischen Vision und Text"""
        if vision_trend == text_trend:
            return 1.0
        elif (vision_trend == "neutral") or (text_trend == "neutral"):
            return 0.5  # Partial agreement with neutral
        else:
            return 0.0  # Complete disagreement (bullish vs bearish)
    
    def extract_signal_strength_from_vision(self, vision_result: VisionResult) -> float:
        """Extrahiert Signal-Stärke aus Vision-Analyse"""
        try:
            if vision_result.patterns:
                pattern_strengths = [p.get("confidence", 0.5) for p in vision_result.patterns]
                return np.mean(pattern_strengths)
            
            return vision_result.confidence
            
        except Exception:
            return 0.5
    
    def extract_risk_from_vision(self, vision_result: VisionResult) -> float:
        """Extrahiert Risk-Assessment aus Vision-Analyse"""
        try:
            # Use model uncertainty as risk proxy
            return vision_result.model_uncertainty
            
        except Exception:
            return 0.5
    
    def calculate_technical_agreement(self,
                                    vision_result: VisionResult,
                                    text_result: TextResult) -> float:
        """Berechnet Technical Indicator Agreement"""
        try:
            if not vision_result.technical_indicators or not text_result.technical_scores:
                return 0.5  # Default if no technical data
            
            # Compare overlapping technical indicators
            vision_tech = vision_result.technical_indicators
            text_tech = text_result.technical_scores
            
            common_indicators = set(vision_tech.keys()) & set(text_tech.keys())
            
            if not common_indicators:
                return 0.5  # No common indicators
            
            agreements = []
            for indicator in common_indicators:
                vision_val = vision_tech[indicator]
                text_val = text_tech[indicator]
                
                # Normalize values to 0-1 range if needed
                if isinstance(vision_val, (int, float)) and isinstance(text_val, (int, float)):
                    agreement = 1.0 - abs(vision_val - text_val)
                    agreements.append(agreement)
            
            return np.mean(agreements) if agreements else 0.5
            
        except Exception as e:
            self.logger.error(f"Technical agreement calculation failed: {e}")
            return 0.5


class MultimodalConfidenceScorer:
    """
    Hauptklasse für kombinierte Confidence-Bewertung von Vision+Text-Analyse
    """
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core Components
        self.calibrator = ConfidenceCalibrator()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.cross_modal_validator = CrossModalValidator()
        
        # Performance Tracking
        self.scoring_history = []
        self.calibration_performance = []
        
        # Statistics
        self.stats = {
            "total_scores_calculated": 0,
            "average_confidence": 0.0,
            "average_uncertainty": 0.0,
            "average_agreement": 0.0,
            "calibration_samples": 0
        }
        
        self.logger.info("Multimodal Confidence Scorer initialized")
    
    def calculate_adaptive_weights(self,
                                 fusion_context: FusionContext,
                                 cross_modal_agreement: float) -> Dict[str, float]:
        """Berechnet adaptive Gewichtungen basierend auf Kontext"""
        try:
            # Base weights from config
            vision_weight = self.config.vision_weight_default
            text_weight = self.config.text_weight_default
            agreement_weight = self.config.agreement_weight
            
            # Adjust based on fusion strategy
            if fusion_context.fusion_strategy == FusionStrategy.VISION_DOMINANT:
                vision_weight = 0.8
                text_weight = 0.2
            elif fusion_context.fusion_strategy == FusionStrategy.TEXT_DOMINANT:
                vision_weight = 0.2
                text_weight = 0.8
            elif fusion_context.fusion_strategy == FusionStrategy.ADAPTIVE:
                # Adjust based on data quality and historical performance
                if fusion_context.data_quality > 0.8:
                    vision_weight = 0.7  # High quality data favors vision
                else:
                    text_weight = 0.7   # Low quality data favors text features
            
            # Adjust based on cross-modal agreement
            if cross_modal_agreement > 0.8:
                # High agreement - increase agreement weight
                agreement_weight = min(0.3, agreement_weight * 1.5)
            elif cross_modal_agreement < 0.4:
                # Low agreement - decrease agreement weight
                agreement_weight = max(0.1, agreement_weight * 0.5)
            
            # Adjust based on market volatility
            if fusion_context.market_volatility > 0.8:
                # High volatility - favor text features (more stable)
                text_weight = min(0.8, text_weight * 1.2)
                vision_weight = max(0.2, 1.0 - text_weight - agreement_weight)
            
            # Normalize weights
            total_weight = vision_weight + text_weight + agreement_weight
            if total_weight > 0:
                vision_weight /= total_weight
                text_weight /= total_weight
                agreement_weight /= total_weight
            
            return {
                "vision": vision_weight,
                "text": text_weight,
                "agreement": agreement_weight
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive weight calculation failed: {e}")
            return {
                "vision": self.config.vision_weight_default,
                "text": self.config.text_weight_default,
                "agreement": self.config.agreement_weight
            }
    
    def determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Bestimmt Confidence-Level basierend auf Score"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def calculate_reliability_metrics(self,
                                    uncertainty: UncertaintyQuantification,
                                    cross_modal_agreement: float,
                                    fusion_context: FusionContext) -> Dict[str, float]:
        """Berechnet Reliability-Metriken"""
        try:
            # Base reliability from uncertainty
            uncertainty_reliability = uncertainty.reliability_score
            
            # Agreement reliability
            agreement_reliability = cross_modal_agreement
            
            # Context reliability (data quality, historical performance)
            context_reliability = fusion_context.data_quality
            if fusion_context.historical_performance:
                hist_values = list(fusion_context.historical_performance.values())
                hist_consistency = 1.0 - np.std(hist_values) if len(hist_values) > 1 else 0.8
                context_reliability = (context_reliability + hist_consistency) / 2.0
            
            # Temporal reliability (based on market stability)
            temporal_reliability = 1.0 - fusion_context.market_volatility * 0.5
            
            # Overall reliability
            overall_reliability = np.mean([
                uncertainty_reliability,
                agreement_reliability,
                context_reliability,
                temporal_reliability
            ])
            
            return {
                "uncertainty_reliability": uncertainty_reliability,
                "agreement_reliability": agreement_reliability,
                "context_reliability": context_reliability,
                "temporal_reliability": temporal_reliability,
                "overall_reliability": overall_reliability
            }
            
        except Exception as e:
            self.logger.error(f"Reliability metrics calculation failed: {e}")
            return {
                "uncertainty_reliability": 0.5,
                "agreement_reliability": 0.5,
                "context_reliability": 0.5,
                "temporal_reliability": 0.5,
                "overall_reliability": 0.5
            }
    
    async def calculate_multimodal_confidence(self,
                                            vision_result: VisionResult,
                                            text_result: TextResult,
                                            fusion_context: FusionContext) -> ConfidenceScore:
        """
        Hauptmethode: Berechnet kombinierte Confidence aus Vision- und Text-Analyse
        
        Args:
            vision_result: Ollama Vision Analysis Ergebnis
            text_result: TorchServe Feature Analysis Ergebnis
            fusion_context: Kontext für Fusion-Gewichtung
            
        Returns:
            ConfidenceScore mit detaillierter Aufschlüsselung
        """
        start_time = time.time()
        
        try:
            # Extract individual confidences
            vision_conf = vision_result.confidence
            text_conf = text_result.confidence
            
            # Calculate cross-modal agreement
            cross_modal_agreement = self.cross_modal_validator.calculate_cross_modal_agreement(
                vision_result, text_result
            )
            
            # Calculate adaptive weights
            adaptive_weights = self.calculate_adaptive_weights(fusion_context, cross_modal_agreement)
            
            # Calculate combined confidence
            combined_confidence = (
                adaptive_weights["vision"] * vision_conf +
                adaptive_weights["text"] * text_conf +
                adaptive_weights["agreement"] * cross_modal_agreement
            )
            
            # Apply uncertainty penalty
            uncertainty = self.uncertainty_estimator.quantify_uncertainty(
                vision_result, text_result, cross_modal_agreement, fusion_context, combined_confidence
            )
            
            # Apply uncertainty penalty to combined confidence
            uncertainty_penalty = uncertainty.total_uncertainty * self.config.uncertainty_penalty
            combined_confidence = max(0.0, combined_confidence - uncertainty_penalty)
            
            # Apply confidence smoothing
            if self.stats["total_scores_calculated"] > 0:
                smoothing = self.config.confidence_smoothing
                combined_confidence = (
                    (1 - smoothing) * combined_confidence +
                    smoothing * self.stats["average_confidence"]
                )
            
            # Calibrate confidence
            calibrated_confidence = self.calibrator.calibrate_confidence(combined_confidence)
            
            # Determine confidence level
            confidence_level = self.determine_confidence_level(calibrated_confidence)
            
            # Calculate reliability metrics
            reliability_metrics = self.calculate_reliability_metrics(
                uncertainty, cross_modal_agreement, fusion_context
            )
            
            # Score breakdown for transparency
            score_breakdown = {
                "vision_contribution": adaptive_weights["vision"] * vision_conf,
                "text_contribution": adaptive_weights["text"] * text_conf,
                "agreement_contribution": adaptive_weights["agreement"] * cross_modal_agreement,
                "uncertainty_penalty": uncertainty_penalty,
                "smoothing_adjustment": combined_confidence - (
                    adaptive_weights["vision"] * vision_conf +
                    adaptive_weights["text"] * text_conf +
                    adaptive_weights["agreement"] * cross_modal_agreement -
                    uncertainty_penalty
                )
            }
            
            # Create confidence score
            confidence_score = ConfidenceScore(
                combined_confidence=combined_confidence,
                vision_confidence=vision_conf,
                text_confidence=text_conf,
                cross_modal_agreement=cross_modal_agreement,
                uncertainty=uncertainty,
                adaptive_weights=adaptive_weights,
                calibrated_confidence=calibrated_confidence,
                confidence_level=confidence_level,
                reliability_metrics=reliability_metrics,
                score_breakdown=score_breakdown,
                generation_timestamp=time.time(),
                metadata={
                    "processing_time": time.time() - start_time,
                    "fusion_strategy": fusion_context.fusion_strategy.value,
                    "timeframe": fusion_context.timeframe,
                    "market_volatility": fusion_context.market_volatility
                }
            )
            
            # Update statistics
            self.update_statistics(confidence_score)
            
            # Store for history
            self.scoring_history.append({
                "timestamp": time.time(),
                "combined_confidence": combined_confidence,
                "calibrated_confidence": calibrated_confidence,
                "cross_modal_agreement": cross_modal_agreement,
                "total_uncertainty": uncertainty.total_uncertainty,
                "reliability": reliability_metrics["overall_reliability"]
            })
            
            # Keep history manageable
            if len(self.scoring_history) > 10000:
                self.scoring_history = self.scoring_history[-5000:]
            
            self.logger.info(f"Calculated multimodal confidence: {calibrated_confidence:.3f} ({confidence_level.value})")
            
            return confidence_score
            
        except Exception as e:
            self.logger.error(f"Multimodal confidence calculation failed: {e}")
            
            # Return default confidence score
            return ConfidenceScore(
                combined_confidence=0.5,
                vision_confidence=vision_result.confidence if vision_result else 0.5,
                text_confidence=text_result.confidence if text_result else 0.5,
                cross_modal_agreement=0.5,
                uncertainty=UncertaintyQuantification(
                    total_uncertainty=0.5, aleatoric_uncertainty=0.3, epistemic_uncertainty=0.3,
                    cross_modal_uncertainty=0.4, temporal_uncertainty=0.4,
                    uncertainty_breakdown={"aleatoric_pct": 25, "epistemic_pct": 25, "cross_modal_pct": 25, "temporal_pct": 25},
                    confidence_interval=(0.3, 0.7), reliability_score=0.5
                ),
                adaptive_weights={"vision": 0.6, "text": 0.4, "agreement": 0.2},
                calibrated_confidence=0.5,
                confidence_level=ConfidenceLevel.MEDIUM,
                reliability_metrics={"overall_reliability": 0.5},
                score_breakdown={"error": "calculation_failed"},
                generation_timestamp=time.time(),
                metadata={"error": str(e)}
            )
    
    def update_statistics(self, confidence_score: ConfidenceScore):
        """Aktualisiert interne Statistiken"""
        try:
            self.stats["total_scores_calculated"] += 1
            n = self.stats["total_scores_calculated"]
            
            # Update running averages
            self.stats["average_confidence"] = (
                (self.stats["average_confidence"] * (n - 1) + confidence_score.combined_confidence) / n
            )
            
            self.stats["average_uncertainty"] = (
                (self.stats["average_uncertainty"] * (n - 1) + confidence_score.uncertainty.total_uncertainty) / n
            )
            
            self.stats["average_agreement"] = (
                (self.stats["average_agreement"] * (n - 1) + confidence_score.cross_modal_agreement) / n
            )
            
        except Exception as e:
            self.logger.error(f"Statistics update failed: {e}")
    
    def update_calibration(self, 
                          predicted_confidence: float, 
                          actual_outcome: float):
        """Aktualisiert Confidence-Kalibrierung basierend auf tatsächlichen Ergebnissen"""
        try:
            self.calibrator.add_calibration_sample(predicted_confidence, actual_outcome)
            self.stats["calibration_samples"] += 1
            
            # Refit calibrator periodically
            if self.stats["calibration_samples"] % 100 == 0:
                success = self.calibrator.fit_calibrator()
                if success:
                    self.logger.info(f"Calibrator refitted with {self.stats['calibration_samples']} samples")
            
        except Exception as e:
            self.logger.error(f"Calibration update failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        calibration_curve = self.calibrator.get_calibration_curve()
        
        return {
            **self.stats,
            "scoring_history_size": len(self.scoring_history),
            "calibrator_fitted": self.calibrator.is_fitted,
            "calibration_curve_points": len(calibration_curve[0]) if len(calibration_curve) > 0 else 0,
            "recent_performance": self.scoring_history[-10:] if len(self.scoring_history) >= 10 else self.scoring_history
        }
    
    def export_calibration_data(self, filepath: Path):
        """Exportiert Kalibrierungs-Daten für Analyse"""
        try:
            calibration_curve = self.calibrator.get_calibration_curve()
            
            export_data = {
                "stats": self.stats,
                "scoring_history": self.scoring_history,
                "calibration_data": self.calibrator.calibration_data,
                "calibration_curve": {
                    "bin_centers": calibration_curve[0].tolist() if len(calibration_curve) > 0 else [],
                    "bin_accuracies": calibration_curve[1].tolist() if len(calibration_curve) > 1 else []
                },
                "export_timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Calibration data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Calibration data export failed: {e}")


# Factory Function
def create_multimodal_confidence_scorer(config: Optional[ConfidenceConfig] = None) -> MultimodalConfidenceScorer:
    """Factory function für Multimodal Confidence Scorer"""
    return MultimodalConfidenceScorer(config)


# Testing Function
async def test_multimodal_confidence_scorer():
    """Test function für Multimodal Confidence Scorer"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    vision_result = VisionResult(
        confidence=0.8,
        patterns=[{"name": "bullish_flag", "confidence": 0.85}],
        trend_analysis={"direction": "bullish", "strength": 0.7},
        support_resistance={"support": [1.1000], "resistance": [1.1050]},
        technical_indicators={"rsi": 0.65, "macd": 0.02},
        processing_time=1.5,
        model_uncertainty=0.2,
        metadata={"source": "test"}
    )
    
    text_result = TextResult(
        confidence=0.75,
        signal_strength=0.8,
        technical_scores={"rsi": 0.7, "macd": 0.6, "bollinger": 0.65},
        market_context={"volatility": "normal", "volume": "high"},
        risk_assessment={"drawdown_risk": 0.3, "volatility_risk": 0.4},
        processing_time=0.8,
        feature_importance={"rsi": 0.3, "macd": 0.25, "volume": 0.2},
        metadata={"source": "test"}
    )
    
    fusion_context = FusionContext(
        timeframe="1h",
        market_volatility=0.6,
        data_quality=0.9,
        historical_performance={"last_week": 0.75, "last_month": 0.8},
        fusion_strategy=FusionStrategy.ADAPTIVE,
        adaptive_weights={"vision": 0.6, "text": 0.4},
        context_complexity=0.7
    )
    
    # Test Multimodal Confidence Scorer
    scorer = create_multimodal_confidence_scorer()
    
    confidence_score = await scorer.calculate_multimodal_confidence(
        vision_result, text_result, fusion_context
    )
    
    print(f"Combined Confidence: {confidence_score.combined_confidence:.3f}")
    print(f"Calibrated Confidence: {confidence_score.calibrated_confidence:.3f}")
    print(f"Confidence Level: {confidence_score.confidence_level.value}")
    print(f"Cross-Modal Agreement: {confidence_score.cross_modal_agreement:.3f}")
    print(f"Total Uncertainty: {confidence_score.uncertainty.total_uncertainty:.3f}")
    print(f"Reliability: {confidence_score.reliability_metrics['overall_reliability']:.3f}")
    
    # Test calibration update
    scorer.update_calibration(confidence_score.combined_confidence, 0.9)  # Simulate good outcome
    
    stats = scorer.get_performance_stats()
    print(f"Performance Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_multimodal_confidence_scorer())