#!/usr/bin/env python3
"""
Enhanced Confidence Scoring mit Multi-Factor-Validation
Phase 2 Implementation - Core AI Enhancement

Features:
- Multi-Factor-Confidence-Scoring
- Pattern-Consistency-Validation
- Market-Regime-Awareness
- Historical-Performance-Integration
- Ensemble-Confidence-Scoring
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import math

from nautilus_trader.model.data import Bar


class EnhancedConfidenceScorer:
    """
    Enhanced Confidence Scorer für AI Trading System
    
    Phase 2 Core AI Enhancement:
    - Multi-Factor-Confidence-Scoring
    - Pattern-Consistency-Validation
    - Market-Regime-Awareness
    - Historical-Performance-Integration
    - Ensemble-Confidence-Scoring
    - Temporal-Confidence-Decay
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Enhanced Confidence Scorer
        
        Args:
            config: Konfiguration für Confidence-Scoring
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Confidence-Scoring-Parameter
        self.base_confidence_weight = self.config.get("base_confidence_weight", 0.4)
        self.pattern_consistency_weight = self.config.get("pattern_consistency_weight", 0.2)
        self.market_regime_weight = self.config.get("market_regime_weight", 0.15)
        self.historical_performance_weight = self.config.get("historical_performance_weight", 0.15)
        self.ensemble_weight = self.config.get("ensemble_weight", 0.1)
        
        # Pattern-Consistency-Parameter
        self.pattern_lookback = self.config.get("pattern_lookback", 10)
        self.consistency_threshold = self.config.get("consistency_threshold", 0.7)
        
        # Market-Regime-Parameter
        self.regime_confidence_multipliers = self.config.get("regime_confidence_multipliers", {
            "trending": 1.1,
            "ranging": 0.9,
            "volatile": 0.8,
            "quiet": 1.05,
            "mixed": 0.95
        })
        
        # Historical-Performance-Parameter
        self.performance_lookback_days = self.config.get("performance_lookback_days", 30)
        self.min_historical_samples = self.config.get("min_historical_samples", 10)
        
        # Temporal-Decay-Parameter
        self.confidence_decay_rate = self.config.get("confidence_decay_rate", 0.1)  # 10% per hour
        self.max_confidence_age_hours = self.config.get("max_confidence_age_hours", 4)
        
        # Performance-History
        self.prediction_history: List[Dict] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        self.logger.info("EnhancedConfidenceScorer initialized")
    
    def calculate_enhanced_confidence(
        self,
        *,
        base_ai_confidence: float,
        prediction: Dict[str, Any],
        features: Dict[str, Any],
        market_regime: str,
        pattern_features: Optional[Dict[str, Any]] = None,
        bar_history: Optional[List[Bar]] = None,
        prediction_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Berechne Enhanced Confidence mit Multi-Factor-Validation
        
        Args:
            base_ai_confidence: Basis AI-Confidence-Score
            prediction: AI-Prediction-Dictionary
            features: Feature-Dictionary
            market_regime: Market-Regime-Classification
            pattern_features: Pattern-Features (optional)
            bar_history: Bar-Historie für Konsistenz-Check
            prediction_timestamp: Timestamp der Prediction
            
        Returns:
            Dictionary mit Enhanced Confidence-Informationen
        """
        confidence_info = {
            "base_confidence": base_ai_confidence,
            "enhanced_confidence": base_ai_confidence,
            "confidence_factors": {},
            "confidence_breakdown": {},
            "validation_results": {},
            "risk_adjusted_confidence": base_ai_confidence,
            "final_confidence": base_ai_confidence,
            "confidence_reasoning": []
        }
        
        try:
            # 1. Pattern-Consistency-Validation
            pattern_consistency_score = self._calculate_pattern_consistency(
                prediction, pattern_features, bar_history
            )
            confidence_info["confidence_factors"]["pattern_consistency"] = pattern_consistency_score
            
            # 2. Market-Regime-Confidence-Adjustment
            regime_adjustment = self._calculate_regime_confidence_adjustment(
                market_regime, features
            )
            confidence_info["confidence_factors"]["market_regime"] = regime_adjustment
            
            # 3. Historical-Performance-Integration
            historical_performance_score = self._calculate_historical_performance_score(
                prediction, market_regime
            )
            confidence_info["confidence_factors"]["historical_performance"] = historical_performance_score
            
            # 4. Ensemble-Confidence-Scoring
            ensemble_score = self._calculate_ensemble_confidence(
                prediction, features, market_regime
            )
            confidence_info["confidence_factors"]["ensemble"] = ensemble_score
            
            # 5. Temporal-Confidence-Decay
            temporal_adjustment = self._calculate_temporal_adjustment(prediction_timestamp)
            confidence_info["confidence_factors"]["temporal_adjustment"] = temporal_adjustment
            
            # 6. Kombiniere alle Faktoren
            enhanced_confidence = self._combine_confidence_factors(
                base_ai_confidence,
                pattern_consistency_score,
                regime_adjustment,
                historical_performance_score,
                ensemble_score,
                temporal_adjustment
            )
            
            confidence_info["enhanced_confidence"] = enhanced_confidence
            
            # 7. Risk-Adjusted-Confidence
            risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
                enhanced_confidence, features, market_regime
            )
            confidence_info["risk_adjusted_confidence"] = risk_adjusted_confidence
            
            # 8. Final-Confidence mit Bounds
            final_confidence = max(0.0, min(1.0, risk_adjusted_confidence))
            confidence_info["final_confidence"] = final_confidence
            
            # 9. Confidence-Breakdown
            confidence_info["confidence_breakdown"] = {
                "base_weight": self.base_confidence_weight,
                "pattern_weight": self.pattern_consistency_weight,
                "regime_weight": self.market_regime_weight,
                "historical_weight": self.historical_performance_weight,
                "ensemble_weight": self.ensemble_weight
            }
            
            # 10. Validation-Results
            confidence_info["validation_results"] = self._validate_confidence_score(
                final_confidence, confidence_info["confidence_factors"]
            )
            
            # 11. Reasoning
            confidence_info["confidence_reasoning"] = self._generate_confidence_reasoning(
                confidence_info
            )
            
            self.logger.debug(f"Enhanced confidence calculated: {final_confidence:.3f} (base: {base_ai_confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {e}")
            confidence_info["confidence_reasoning"].append(f"Error in calculation: {e}")
        
        return confidence_info
    
    def _calculate_pattern_consistency(
        self, 
        prediction: Dict[str, Any], 
        pattern_features: Optional[Dict[str, Any]], 
        bar_history: Optional[List[Bar]]
    ) -> float:
        """
        Berechne Pattern-Consistency-Score
        
        Args:
            prediction: AI-Prediction
            pattern_features: Pattern-Features
            bar_history: Bar-Historie
            
        Returns:
            Pattern-Consistency-Score (0.0 - 1.0)
        """
        try:
            if not pattern_features or not bar_history:
                return 0.5  # Neutral score wenn keine Pattern-Daten
            
            consistency_score = 0.5
            
            # 1. Pattern-Count-Consistency
            pattern_count = pattern_features.get("pattern_count", 0)
            if pattern_count > 0:
                pattern_confidence = pattern_features.get("pattern_confidence", 0.5)
                consistency_score += (pattern_confidence - 0.5) * 0.3
            
            # 2. Trend-Consistency
            trend_direction = pattern_features.get("trend_direction", "sideways")
            prediction_action = prediction.get("action", "HOLD")
            
            if (trend_direction == "uptrend" and prediction_action == "BUY") or \
               (trend_direction == "downtrend" and prediction_action == "SELL"):
                consistency_score += 0.2
            elif trend_direction == "sideways" and prediction_action == "HOLD":
                consistency_score += 0.1
            
            # 3. Support/Resistance-Consistency
            support_count = pattern_features.get("support_levels_count", 0)
            resistance_count = pattern_features.get("resistance_levels_count", 0)
            
            if support_count > 0 and prediction_action == "BUY":
                consistency_score += 0.1
            if resistance_count > 0 and prediction_action == "SELL":
                consistency_score += 0.1
            
            # 4. Reversal-Pattern-Consistency
            has_reversal = pattern_features.get("has_reversal", False)
            if has_reversal:
                # Reversal-Patterns sollten gegen aktuellen Trend gehen
                if (trend_direction == "uptrend" and prediction_action == "SELL") or \
                   (trend_direction == "downtrend" and prediction_action == "BUY"):
                    consistency_score += 0.15
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern consistency: {e}")
            return 0.5
    
    def _calculate_regime_confidence_adjustment(
        self, 
        market_regime: str, 
        features: Dict[str, Any]
    ) -> float:
        """
        Berechne Market-Regime-basierte Confidence-Adjustment
        
        Args:
            market_regime: Market-Regime
            features: Feature-Dictionary
            
        Returns:
            Regime-Confidence-Adjustment (0.0 - 2.0)
        """
        try:
            # Basis-Multiplier für Regime
            base_multiplier = self.regime_confidence_multipliers.get(market_regime, 1.0)
            
            # Zusätzliche Adjustments basierend auf Features
            volatility = features.get("volatility_level", 0.01)
            volume_ratio = features.get("volume_regime", 1.0)
            
            # Volatility-Adjustment
            if market_regime == "volatile" and volatility > 0.02:
                base_multiplier *= 0.9  # Reduziere Confidence bei hoher Volatility
            elif market_regime == "quiet" and volatility < 0.005:
                base_multiplier *= 1.05  # Erhöhe Confidence bei niedriger Volatility
            
            # Volume-Adjustment
            if volume_ratio > 1.5:  # Hohe Volume
                base_multiplier *= 1.02
            elif volume_ratio < 0.5:  # Niedrige Volume
                base_multiplier *= 0.98
            
            return max(0.5, min(1.5, base_multiplier))
            
        except Exception as e:
            self.logger.error(f"Error calculating regime confidence adjustment: {e}")
            return 1.0
    
    def _calculate_historical_performance_score(
        self, 
        prediction: Dict[str, Any], 
        market_regime: str
    ) -> float:
        """
        Berechne Historical-Performance-Score
        
        Args:
            prediction: AI-Prediction
            market_regime: Market-Regime
            
        Returns:
            Historical-Performance-Score (0.0 - 1.0)
        """
        try:
            if len(self.prediction_history) < self.min_historical_samples:
                return 0.5  # Neutral score bei wenig Historie
            
            # Filtere relevante historische Predictions
            cutoff_date = datetime.now() - timedelta(days=self.performance_lookback_days)
            relevant_predictions = [
                p for p in self.prediction_history 
                if p.get("timestamp", datetime.min) > cutoff_date
                and p.get("market_regime") == market_regime
                and p.get("action") == prediction.get("action")
            ]
            
            if not relevant_predictions:
                return 0.5
            
            # Berechne Performance-Metriken
            successful_predictions = [
                p for p in relevant_predictions 
                if p.get("outcome", "unknown") == "success"
            ]
            
            success_rate = len(successful_predictions) / len(relevant_predictions)
            
            # Berechne durchschnittliche Confidence vs. Outcome
            confidence_accuracy = 0.5
            if relevant_predictions:
                confidence_outcomes = []
                for p in relevant_predictions:
                    pred_confidence = p.get("confidence", 0.5)
                    actual_outcome = 1.0 if p.get("outcome") == "success" else 0.0
                    confidence_outcomes.append(abs(pred_confidence - actual_outcome))
                
                # Niedrigere Abweichung = bessere Confidence-Accuracy
                avg_deviation = np.mean(confidence_outcomes)
                confidence_accuracy = max(0.0, 1.0 - avg_deviation)
            
            # Kombiniere Success-Rate und Confidence-Accuracy
            performance_score = (success_rate * 0.7) + (confidence_accuracy * 0.3)
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating historical performance score: {e}")
            return 0.5
    
    def _calculate_ensemble_confidence(
        self, 
        prediction: Dict[str, Any], 
        features: Dict[str, Any], 
        market_regime: str
    ) -> float:
        """
        Berechne Ensemble-Confidence-Score
        
        Args:
            prediction: AI-Prediction
            features: Feature-Dictionary
            market_regime: Market-Regime
            
        Returns:
            Ensemble-Confidence-Score (0.0 - 1.0)
        """
        try:
            ensemble_scores = []
            
            # 1. Technical-Indicator-Ensemble
            technical_score = self._calculate_technical_indicator_consensus(
                prediction, features
            )
            ensemble_scores.append(technical_score)
            
            # 2. Pattern-Recognition-Ensemble
            pattern_score = self._calculate_pattern_recognition_consensus(
                prediction, features
            )
            ensemble_scores.append(pattern_score)
            
            # 3. Market-Regime-Ensemble
            regime_score = self._calculate_market_regime_consensus(
                prediction, market_regime, features
            )
            ensemble_scores.append(regime_score)
            
            # 4. Volatility-Ensemble
            volatility_score = self._calculate_volatility_consensus(
                prediction, features
            )
            ensemble_scores.append(volatility_score)
            
            # Berechne gewichteten Durchschnitt
            if ensemble_scores:
                ensemble_confidence = np.mean(ensemble_scores)
            else:
                ensemble_confidence = 0.5
            
            return max(0.0, min(1.0, ensemble_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def _calculate_temporal_adjustment(self, prediction_timestamp: Optional[datetime]) -> float:
        """
        Berechne Temporal-Confidence-Decay
        
        Args:
            prediction_timestamp: Timestamp der Prediction
            
        Returns:
            Temporal-Adjustment-Factor (0.0 - 1.0)
        """
        try:
            if not prediction_timestamp:
                return 1.0  # Keine Decay wenn kein Timestamp
            
            # Berechne Alter der Prediction
            age = datetime.now() - prediction_timestamp
            age_hours = age.total_seconds() / 3600
            
            if age_hours <= 0:
                return 1.0
            
            if age_hours >= self.max_confidence_age_hours:
                return 0.1  # Minimale Confidence nach max age
            
            # Exponential Decay
            decay_factor = math.exp(-self.confidence_decay_rate * age_hours)
            
            return max(0.1, decay_factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal adjustment: {e}")
            return 1.0
    
    def _combine_confidence_factors(
        self,
        base_confidence: float,
        pattern_consistency: float,
        regime_adjustment: float,
        historical_performance: float,
        ensemble_score: float,
        temporal_adjustment: float
    ) -> float:
        """
        Kombiniere alle Confidence-Faktoren
        
        Returns:
            Combined Enhanced Confidence
        """
        try:
            # Gewichtete Kombination
            enhanced_confidence = (
                base_confidence * self.base_confidence_weight +
                pattern_consistency * self.pattern_consistency_weight +
                (base_confidence * regime_adjustment) * self.market_regime_weight +
                historical_performance * self.historical_performance_weight +
                ensemble_score * self.ensemble_weight
            )
            
            # Temporal-Adjustment anwenden
            enhanced_confidence *= temporal_adjustment
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"Error combining confidence factors: {e}")
            return base_confidence
    
    def _calculate_risk_adjusted_confidence(
        self, 
        enhanced_confidence: float, 
        features: Dict[str, Any], 
        market_regime: str
    ) -> float:
        """
        Berechne Risk-Adjusted-Confidence
        
        Args:
            enhanced_confidence: Enhanced Confidence
            features: Feature-Dictionary
            market_regime: Market-Regime
            
        Returns:
            Risk-Adjusted-Confidence
        """
        try:
            risk_adjustment = 1.0
            
            # Volatility-Risk-Adjustment
            volatility = features.get("volatility_level", 0.01)
            if volatility > 0.02:  # Hohe Volatility
                risk_adjustment *= 0.9
            elif volatility > 0.015:  # Mittlere Volatility
                risk_adjustment *= 0.95
            
            # Market-Regime-Risk-Adjustment
            if market_regime == "volatile":
                risk_adjustment *= 0.9
            elif market_regime == "mixed":
                risk_adjustment *= 0.95
            
            # Trend-Strength-Risk-Adjustment
            trend_strength = features.get("trend_strength", 0.0)
            if trend_strength < 0.001:  # Schwacher Trend
                risk_adjustment *= 0.95
            
            # News-Event-Risk-Adjustment (falls verfügbar)
            if features.get("near_news_event", False):
                risk_adjustment *= 0.85
            
            return enhanced_confidence * risk_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating risk adjusted confidence: {e}")
            return enhanced_confidence
    
    def _validate_confidence_score(
        self, 
        final_confidence: float, 
        confidence_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validiere Final-Confidence-Score
        
        Returns:
            Validation-Results
        """
        validation = {
            "is_valid": True,
            "warnings": [],
            "quality_score": 0.0
        }
        
        try:
            # 1. Range-Validation
            if final_confidence < 0 or final_confidence > 1:
                validation["is_valid"] = False
                validation["warnings"].append(f"Confidence out of range: {final_confidence}")
            
            # 2. Factor-Consistency-Validation
            factor_values = list(confidence_factors.values())
            if factor_values:
                factor_std = np.std(factor_values)
                if factor_std > 0.3:  # Hohe Varianz zwischen Faktoren
                    validation["warnings"].append("High variance between confidence factors")
            
            # 3. Quality-Score-Berechnung
            quality_factors = []
            
            # Anzahl verfügbarer Faktoren
            available_factors = len([f for f in confidence_factors.values() if f != 0.5])
            quality_factors.append(min(1.0, available_factors / 5))  # Max 5 Faktoren
            
            # Konsistenz der Faktoren
            if factor_values:
                consistency = 1.0 - (np.std(factor_values) / np.mean(factor_values))
                quality_factors.append(max(0.0, consistency))
            
            validation["quality_score"] = np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error validating confidence score: {e}")
            validation["warnings"].append(f"Validation error: {e}")
        
        return validation
    
    def _generate_confidence_reasoning(self, confidence_info: Dict[str, Any]) -> List[str]:
        """
        Generiere Confidence-Reasoning
        
        Returns:
            Liste mit Reasoning-Strings
        """
        reasoning = []
        
        try:
            base_conf = confidence_info["base_confidence"]
            final_conf = confidence_info["final_confidence"]
            factors = confidence_info["confidence_factors"]
            
            reasoning.append(f"Base AI confidence: {base_conf:.3f}")
            reasoning.append(f"Final enhanced confidence: {final_conf:.3f}")
            
            # Faktor-Reasoning
            for factor_name, factor_value in factors.items():
                if factor_value != 0.5:  # Nur nicht-neutrale Faktoren
                    if factor_value > 0.6:
                        reasoning.append(f"{factor_name}: positive impact ({factor_value:.3f})")
                    elif factor_value < 0.4:
                        reasoning.append(f"{factor_name}: negative impact ({factor_value:.3f})")
            
            # Quality-Reasoning
            quality_score = confidence_info["validation_results"].get("quality_score", 0.5)
            if quality_score > 0.7:
                reasoning.append("High confidence quality")
            elif quality_score < 0.3:
                reasoning.append("Low confidence quality - use with caution")
            
        except Exception as e:
            reasoning.append(f"Error generating reasoning: {e}")
        
        return reasoning
    
    def update_prediction_outcome(
        self, 
        prediction_id: str, 
        outcome: str, 
        actual_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update Prediction-Outcome für Historical-Performance-Tracking
        
        Args:
            prediction_id: ID der Prediction
            outcome: Outcome ("success", "failure", "partial")
            actual_result: Tatsächliches Ergebnis
        """
        try:
            # Finde Prediction in Historie
            for prediction in self.prediction_history:
                if prediction.get("id") == prediction_id:
                    prediction["outcome"] = outcome
                    prediction["actual_result"] = actual_result
                    prediction["outcome_timestamp"] = datetime.now()
                    break
            
            self.logger.debug(f"Updated prediction outcome: {prediction_id} -> {outcome}")
            
        except Exception as e:
            self.logger.error(f"Error updating prediction outcome: {e}")
    
    def add_prediction_to_history(
        self, 
        prediction: Dict[str, Any], 
        confidence_info: Dict[str, Any]
    ) -> None:
        """
        Füge Prediction zur Historie hinzu
        
        Args:
            prediction: Prediction-Dictionary
            confidence_info: Confidence-Informationen
        """
        try:
            history_entry = {
                "id": prediction.get("id", f"pred_{len(self.prediction_history)}"),
                "timestamp": datetime.now(),
                "action": prediction.get("action"),
                "confidence": confidence_info.get("final_confidence"),
                "base_confidence": confidence_info.get("base_confidence"),
                "market_regime": prediction.get("market_regime"),
                "confidence_factors": confidence_info.get("confidence_factors", {}),
                "outcome": "pending"
            }
            
            self.prediction_history.append(history_entry)
            
            # Begrenze Historie-Größe
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history = self.prediction_history[-self.max_history_size:]
            
        except Exception as e:
            self.logger.error(f"Error adding prediction to history: {e}")
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """
        Erhalte Confidence-Scoring-Statistiken
        
        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            "total_predictions": len(self.prediction_history),
            "confidence_weights": {
                "base_confidence": self.base_confidence_weight,
                "pattern_consistency": self.pattern_consistency_weight,
                "market_regime": self.market_regime_weight,
                "historical_performance": self.historical_performance_weight,
                "ensemble": self.ensemble_weight
            }
        }
        
        if self.prediction_history:
            recent_predictions = self.prediction_history[-50:]  # Letzte 50
            
            confidences = [p.get("confidence", 0.5) for p in recent_predictions]
            base_confidences = [p.get("base_confidence", 0.5) for p in recent_predictions]
            
            stats.update({
                "avg_enhanced_confidence": np.mean(confidences),
                "avg_base_confidence": np.mean(base_confidences),
                "confidence_improvement": np.mean(confidences) - np.mean(base_confidences),
                "confidence_std": np.std(confidences),
                "successful_predictions": len([p for p in recent_predictions if p.get("outcome") == "success"]),
                "pending_predictions": len([p for p in recent_predictions if p.get("outcome") == "pending"])
            })
        
        return stats
    
    # Helper Methods für Ensemble-Scoring
    def _calculate_technical_indicator_consensus(
        self, 
        prediction: Dict[str, Any], 
        features: Dict[str, Any]
    ) -> float:
        """Berechne Technical-Indicator-Consensus"""
        try:
            action = prediction.get("action", "HOLD")
            consensus_score = 0.5
            
            # RSI-Consensus
            rsi = features.get("rsi", 50)
            if action == "BUY" and rsi < 30:
                consensus_score += 0.1
            elif action == "SELL" and rsi > 70:
                consensus_score += 0.1
            
            # MACD-Consensus
            macd_bullish = features.get("macd_bullish", False)
            if action == "BUY" and macd_bullish:
                consensus_score += 0.1
            elif action == "SELL" and not macd_bullish:
                consensus_score += 0.1
            
            # Moving-Average-Consensus
            ma_bullish_cross = features.get("ma_bullish_cross", False)
            if action == "BUY" and ma_bullish_cross:
                consensus_score += 0.1
            elif action == "SELL" and not ma_bullish_cross:
                consensus_score += 0.1
            
            return max(0.0, min(1.0, consensus_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicator consensus: {e}")
            return 0.5
    
    def _calculate_pattern_recognition_consensus(
        self, 
        prediction: Dict[str, Any], 
        features: Dict[str, Any]
    ) -> float:
        """Berechne Pattern-Recognition-Consensus"""
        try:
            action = prediction.get("action", "HOLD")
            consensus_score = 0.5
            
            # Candlestick-Pattern-Consensus
            has_doji = features.get("has_doji", False)
            has_hammer = features.get("has_hammer", False)
            has_engulfing = features.get("has_engulfing", False)
            
            if action == "BUY" and (has_hammer or has_engulfing):
                consensus_score += 0.15
            elif action == "SELL" and has_engulfing:
                consensus_score += 0.15
            elif action == "HOLD" and has_doji:
                consensus_score += 0.1
            
            return max(0.0, min(1.0, consensus_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern recognition consensus: {e}")
            return 0.5
    
    def _calculate_market_regime_consensus(
        self, 
        prediction: Dict[str, Any], 
        market_regime: str, 
        features: Dict[str, Any]
    ) -> float:
        """Berechne Market-Regime-Consensus"""
        try:
            action = prediction.get("action", "HOLD")
            consensus_score = 0.5
            
            # Regime-Action-Alignment
            if market_regime == "trending":
                trend_direction = features.get("trend_direction", "sideways")
                if (trend_direction == "uptrend" and action == "BUY") or \
                   (trend_direction == "downtrend" and action == "SELL"):
                    consensus_score += 0.2
            elif market_regime == "ranging" and action == "HOLD":
                consensus_score += 0.1
            
            return max(0.0, min(1.0, consensus_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime consensus: {e}")
            return 0.5
    
    def _calculate_volatility_consensus(
        self, 
        prediction: Dict[str, Any], 
        features: Dict[str, Any]
    ) -> float:
        """Berechne Volatility-Consensus"""
        try:
            action = prediction.get("action", "HOLD")
            volatility = features.get("volatility_level", 0.01)
            consensus_score = 0.5
            
            # Volatility-Action-Alignment
            if volatility < 0.005 and action != "HOLD":  # Niedrige Volatility
                consensus_score -= 0.1  # Weniger Confidence für Trades bei niedriger Volatility
            elif volatility > 0.02 and action != "HOLD":  # Hohe Volatility
                consensus_score -= 0.05  # Etwas weniger Confidence bei hoher Volatility
            elif 0.005 <= volatility <= 0.015:  # Optimale Volatility
                consensus_score += 0.1
            
            return max(0.0, min(1.0, consensus_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility consensus: {e}")
            return 0.5


# Factory Function
def create_enhanced_confidence_scorer(config: Optional[Dict] = None) -> EnhancedConfidenceScorer:
    """
    Factory Function für Enhanced Confidence Scorer
    
    Args:
        config: Konfiguration für Confidence-Scoring
        
    Returns:
        EnhancedConfidenceScorer Instance
    """
    return EnhancedConfidenceScorer(config)


if __name__ == "__main__":
    # Test des Enhanced Confidence Scorers
    print("🧪 Testing EnhancedConfidenceScorer...")
    
    scorer = create_enhanced_confidence_scorer()
    
    # Test-Daten
    test_scenarios = [
        {
            "base_ai_confidence": 0.8,
            "prediction": {"action": "BUY", "confidence": 0.8, "reasoning": "test"},
            "features": {
                "rsi": 25, "macd_bullish": True, "ma_bullish_cross": True,
                "volatility_level": 0.01, "trend_strength": 0.003,
                "has_hammer": True, "trend_direction": "uptrend"
            },
            "market_regime": "trending",
            "pattern_features": {
                "pattern_count": 2, "pattern_confidence": 0.75,
                "trend_direction": "uptrend", "has_reversal": False
            }
        },
        {
            "base_ai_confidence": 0.6,
            "prediction": {"action": "SELL", "confidence": 0.6, "reasoning": "test"},
            "features": {
                "rsi": 75, "macd_bullish": False, "ma_bullish_cross": False,
                "volatility_level": 0.02, "trend_strength": 0.001,
                "has_engulfing": True, "trend_direction": "downtrend"
            },
            "market_regime": "volatile",
            "pattern_features": {
                "pattern_count": 1, "pattern_confidence": 0.6,
                "trend_direction": "downtrend", "has_reversal": True
            }
        }
    ]
    
    print(f"📊 Enhanced Confidence Scoring Test Results:")
    
    for i, scenario in enumerate(test_scenarios, 1):
        confidence_info = scorer.calculate_enhanced_confidence(**scenario)
        
        print(f"\\n   Scenario {i}:")
        print(f"     Base Confidence: {confidence_info['base_confidence']:.3f}")
        print(f"     Enhanced Confidence: {confidence_info['enhanced_confidence']:.3f}")
        print(f"     Final Confidence: {confidence_info['final_confidence']:.3f}")
        print(f"     Quality Score: {confidence_info['validation_results']['quality_score']:.3f}")
        
        # Zeige Top-Faktoren
        factors = confidence_info['confidence_factors']
        top_factors = sorted(factors.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:3]
        print(f"     Top Factors: {', '.join([f'{k}={v:.3f}' for k, v in top_factors])}")
        
        # Füge zur Historie hinzu
        scorer.add_prediction_to_history(scenario["prediction"], confidence_info)
    
    # Test Statistics
    stats = scorer.get_confidence_statistics()
    print(f"\\n📊 Confidence Scoring Statistics:")
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Avg Enhanced Confidence: {stats.get('avg_enhanced_confidence', 0):.3f}")
    print(f"   Confidence Improvement: {stats.get('confidence_improvement', 0):.3f}")
    
    print("✅ EnhancedConfidenceScorer Test completed!")