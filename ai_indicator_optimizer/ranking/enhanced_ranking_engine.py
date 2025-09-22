"""
Enhanced Ranking Engine Implementation
Task 5: Enhanced Ranking Engine Implementation

This module implements the advanced ranking system with multiple criteria,
portfolio optimization, and risk-adjusted scoring for the Top-5 strategies system.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class StrategyRankingCriteria(Enum):
    """Strategy ranking criteria enumeration"""
    SIGNAL_CONFIDENCE = "signal_confidence"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    OPPORTUNITY_SCORE = "opportunity_score"
    FUSION_CONFIDENCE = "fusion_confidence"
    CONSISTENCY_SCORE = "consistency_score"
    PROFIT_POTENTIAL = "profit_potential"
    DRAWDOWN_RISK = "drawdown_risk"

@dataclass
class StrategyScore:
    """Base strategy score from Baustein B3"""
    name: str
    signal_confidence: float
    risk_reward_ratio: float
    opportunity_score: float
    fusion_confidence: float
    consistency_score: float
    profit_potential: float
    drawdown_risk: float
    composite_score: float = 0.0
    
    def __post_init__(self):
        if self.composite_score == 0.0:
            self.composite_score = self._calculate_composite_score()
    
    def _calculate_composite_score(self) -> float:
        """Calculate composite score from individual metrics"""
        return (
            self.signal_confidence * 0.2 +
            (self.risk_reward_ratio / 4.0) * 0.15 +
            self.opportunity_score * 0.15 +
            self.fusion_confidence * 0.15 +
            self.consistency_score * 0.15 +
            self.profit_potential * 0.15 +
            (1.0 - self.drawdown_risk) * 0.05
        )

@dataclass
class EnhancedStrategyRanking:
    """Enhanced strategy ranking with portfolio optimization"""
    strategy: StrategyScore
    portfolio_fit: float
    diversification_score: float
    risk_adjusted_score: float
    final_ranking_score: float
    rank_position: int
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    performance_projections: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class RankingConfig:
    """Configuration for enhanced ranking engine"""
    # Ranking criteria weights
    criteria_weights: Dict[StrategyRankingCriteria, float] = field(default_factory=lambda: {
        StrategyRankingCriteria.SIGNAL_CONFIDENCE: 0.20,
        StrategyRankingCriteria.RISK_REWARD_RATIO: 0.15,
        StrategyRankingCriteria.OPPORTUNITY_SCORE: 0.15,
        StrategyRankingCriteria.FUSION_CONFIDENCE: 0.15,
        StrategyRankingCriteria.CONSISTENCY_SCORE: 0.15,
        StrategyRankingCriteria.PROFIT_POTENTIAL: 0.15,
        StrategyRankingCriteria.DRAWDOWN_RISK: 0.05
    })
    
    # Portfolio optimization weights
    portfolio_fit_weight: float = 0.20
    diversification_weight: float = 0.10
    risk_adjustment_weight: float = 0.10
    base_score_weight: float = 0.60
    
    # Risk-free rate for Sharpe-like calculations
    risk_free_rate: float = 0.02
    
    # Confidence interval settings
    confidence_level: float = 0.95
    monte_carlo_simulations: int = 1000
    
    # Quality thresholds
    min_confidence_threshold: float = 0.5
    min_composite_score: float = 0.4
    max_drawdown_threshold: float = 0.25

class MultiCriteriaEvaluator:
    """Multi-criteria evaluator with 7+ ranking factors"""
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.logger = logging.getLogger(f"{__name__}.MultiCriteriaEvaluator")
        
    def evaluate_strategies(self, strategies: List[StrategyScore]) -> List[StrategyScore]:
        """
        Evaluate strategies using multi-criteria analysis
        
        Args:
            strategies: List of strategy scores from Baustein B3
            
        Returns:
            List of evaluated strategies with enhanced metrics
        """
        self.logger.info(f"🔍 Evaluating {len(strategies)} strategies with multi-criteria analysis")
        
        evaluated_strategies = []
        
        for strategy in strategies:
            # Apply quality filters
            if not self._meets_quality_thresholds(strategy):
                self.logger.warning(f"⚠️ Strategy {strategy.name} filtered out due to quality thresholds")
                continue
            
            # Calculate weighted composite score
            weighted_score = self._calculate_weighted_score(strategy)
            
            # Update strategy with enhanced metrics
            enhanced_strategy = StrategyScore(
                name=strategy.name,
                signal_confidence=strategy.signal_confidence,
                risk_reward_ratio=strategy.risk_reward_ratio,
                opportunity_score=strategy.opportunity_score,
                fusion_confidence=strategy.fusion_confidence,
                consistency_score=strategy.consistency_score,
                profit_potential=strategy.profit_potential,
                drawdown_risk=strategy.drawdown_risk,
                composite_score=weighted_score
            )
            
            evaluated_strategies.append(enhanced_strategy)
        
        # Sort by composite score
        evaluated_strategies.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.logger.info(f"✅ Multi-criteria evaluation complete: {len(evaluated_strategies)} strategies passed filters")
        
        return evaluated_strategies
    
    def _meets_quality_thresholds(self, strategy: StrategyScore) -> bool:
        """Check if strategy meets minimum quality thresholds"""
        return (
            strategy.signal_confidence >= self.config.min_confidence_threshold and
            strategy.composite_score >= self.config.min_composite_score and
            strategy.drawdown_risk <= self.config.max_drawdown_threshold
        )
    
    def _calculate_weighted_score(self, strategy: StrategyScore) -> float:
        """Calculate weighted composite score using configured weights"""
        weights = self.config.criteria_weights
        
        score = (
            strategy.signal_confidence * weights[StrategyRankingCriteria.SIGNAL_CONFIDENCE] +
            (strategy.risk_reward_ratio / 4.0) * weights[StrategyRankingCriteria.RISK_REWARD_RATIO] +
            strategy.opportunity_score * weights[StrategyRankingCriteria.OPPORTUNITY_SCORE] +
            strategy.fusion_confidence * weights[StrategyRankingCriteria.FUSION_CONFIDENCE] +
            strategy.consistency_score * weights[StrategyRankingCriteria.CONSISTENCY_SCORE] +
            strategy.profit_potential * weights[StrategyRankingCriteria.PROFIT_POTENTIAL] +
            (1.0 - strategy.drawdown_risk) * weights[StrategyRankingCriteria.DRAWDOWN_RISK]
        )
        
        return score

class PortfolioFitCalculator:
    """Portfolio fit calculator for diversification scoring"""
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.logger = logging.getLogger(f"{__name__}.PortfolioFitCalculator")
    
    def calculate_portfolio_fit(self, strategy: StrategyScore, all_strategies: List[StrategyScore]) -> float:
        """
        Calculate portfolio fit based on diversification benefits
        
        Args:
            strategy: Strategy to evaluate
            all_strategies: All available strategies for comparison
            
        Returns:
            Portfolio fit score (0.0 to 1.0)
        """
        if len(all_strategies) <= 1:
            return 1.0
        
        # Calculate correlation with other strategies
        correlations = []
        
        for other_strategy in all_strategies:
            if other_strategy.name != strategy.name:
                correlation = self._calculate_strategy_correlation(strategy, other_strategy)
                correlations.append(correlation)
        
        # Portfolio fit is inversely related to average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        portfolio_fit = 1.0 - min(abs(avg_correlation), 1.0)
        
        self.logger.debug(f"📊 Portfolio fit for {strategy.name}: {portfolio_fit:.3f} (avg correlation: {avg_correlation:.3f})")
        
        return portfolio_fit
    
    def calculate_diversification_score(self, strategy: StrategyScore, all_strategies: List[StrategyScore]) -> float:
        """
        Calculate diversification score for strategy
        
        Args:
            strategy: Strategy to evaluate
            all_strategies: All available strategies for comparison
            
        Returns:
            Diversification score (0.0 to 1.0)
        """
        if len(all_strategies) <= 1:
            return 1.0
        
        # Calculate uniqueness metrics
        uniqueness_factors = []
        
        # Risk-reward uniqueness
        rr_ratios = [s.risk_reward_ratio for s in all_strategies]
        rr_uniqueness = self._calculate_uniqueness(strategy.risk_reward_ratio, rr_ratios)
        uniqueness_factors.append(rr_uniqueness)
        
        # Signal confidence uniqueness
        conf_scores = [s.signal_confidence for s in all_strategies]
        conf_uniqueness = self._calculate_uniqueness(strategy.signal_confidence, conf_scores)
        uniqueness_factors.append(conf_uniqueness)
        
        # Drawdown risk uniqueness
        dd_risks = [s.drawdown_risk for s in all_strategies]
        dd_uniqueness = self._calculate_uniqueness(strategy.drawdown_risk, dd_risks)
        uniqueness_factors.append(dd_uniqueness)
        
        # Overall diversification score
        diversification_score = np.mean(uniqueness_factors)
        
        self.logger.debug(f"🎯 Diversification score for {strategy.name}: {diversification_score:.3f}")
        
        return diversification_score
    
    def _calculate_strategy_correlation(self, strategy1: StrategyScore, strategy2: StrategyScore) -> float:
        """Calculate correlation between two strategies based on their metrics"""
        # Create feature vectors for correlation calculation
        features1 = np.array([
            strategy1.signal_confidence,
            strategy1.risk_reward_ratio / 4.0,  # Normalize
            strategy1.opportunity_score,
            strategy1.fusion_confidence,
            strategy1.consistency_score,
            strategy1.profit_potential,
            1.0 - strategy1.drawdown_risk  # Invert for positive correlation
        ])
        
        features2 = np.array([
            strategy2.signal_confidence,
            strategy2.risk_reward_ratio / 4.0,
            strategy2.opportunity_score,
            strategy2.fusion_confidence,
            strategy2.consistency_score,
            strategy2.profit_potential,
            1.0 - strategy2.drawdown_risk
        ])
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(features1, features2)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            correlation = 0.0
        
        return correlation
    
    def _calculate_uniqueness(self, value: float, all_values: List[float]) -> float:
        """Calculate how unique a value is within a distribution"""
        if len(all_values) <= 1:
            return 1.0
        
        # Calculate standard deviation from mean
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        if std_val == 0:
            return 0.0
        
        # Uniqueness based on distance from mean in standard deviations
        z_score = abs(value - mean_val) / std_val
        uniqueness = min(z_score / 2.0, 1.0)  # Cap at 1.0
        
        return uniqueness

class RiskAdjustedScorer:
    """Risk-adjusted scorer with Sharpe-like calculation"""
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.logger = logging.getLogger(f"{__name__}.RiskAdjustedScorer")
    
    def calculate_risk_adjusted_score(self, strategy: StrategyScore) -> float:
        """
        Calculate Sharpe-like risk-adjusted performance
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Risk-adjusted score
        """
        # Expected return proxy (profit potential adjusted by confidence)
        expected_return = strategy.profit_potential * strategy.signal_confidence
        
        # Risk proxy (drawdown risk + volatility estimate)
        volatility_estimate = self._estimate_volatility(strategy)
        total_risk = strategy.drawdown_risk + volatility_estimate
        
        # Sharpe-like ratio
        if total_risk > 0:
            risk_adjusted_score = (expected_return - self.config.risk_free_rate) / total_risk
        else:
            risk_adjusted_score = expected_return - self.config.risk_free_rate
        
        # Normalize to 0-10 scale
        normalized_score = max(0, min(10, risk_adjusted_score * 2 + 5))
        
        self.logger.debug(f"📈 Risk-adjusted score for {strategy.name}: {normalized_score:.3f}")
        
        return normalized_score
    
    def calculate_confidence_intervals(self, strategy: StrategyScore) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for strategy performance projections
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Dictionary of confidence intervals for different metrics
        """
        confidence_intervals = {}
        
        # Monte Carlo simulation for confidence intervals
        n_simulations = self.config.monte_carlo_simulations
        
        # Simulate expected returns
        return_simulations = self._simulate_returns(strategy, n_simulations)
        confidence_intervals["expected_return"] = self._calculate_confidence_interval(
            return_simulations, self.config.confidence_level
        )
        
        # Simulate drawdowns
        drawdown_simulations = self._simulate_drawdowns(strategy, n_simulations)
        confidence_intervals["max_drawdown"] = self._calculate_confidence_interval(
            drawdown_simulations, self.config.confidence_level
        )
        
        # Simulate Sharpe ratios
        sharpe_simulations = self._simulate_sharpe_ratios(strategy, n_simulations)
        confidence_intervals["sharpe_ratio"] = self._calculate_confidence_interval(
            sharpe_simulations, self.config.confidence_level
        )
        
        self.logger.debug(f"📊 Confidence intervals calculated for {strategy.name}")
        
        return confidence_intervals
    
    def calculate_performance_projections(self, strategy: StrategyScore) -> Dict[str, float]:
        """
        Calculate performance projections for different time horizons
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Dictionary of performance projections
        """
        projections = {}
        
        # Base annual return estimate
        base_annual_return = strategy.profit_potential * strategy.signal_confidence
        
        # Time horizon projections
        projections["1_month"] = base_annual_return / 12
        projections["3_months"] = base_annual_return / 4
        projections["6_months"] = base_annual_return / 2
        projections["1_year"] = base_annual_return
        
        # Risk-adjusted projections
        risk_adjustment = 1.0 - strategy.drawdown_risk
        projections["risk_adjusted_1_year"] = base_annual_return * risk_adjustment
        
        # Compound growth estimate
        monthly_return = base_annual_return / 12
        projections["compound_1_year"] = (1 + monthly_return) ** 12 - 1
        
        self.logger.debug(f"📈 Performance projections calculated for {strategy.name}")
        
        return projections
    
    def _estimate_volatility(self, strategy: StrategyScore) -> float:
        """Estimate strategy volatility from available metrics"""
        # Volatility proxy based on consistency and confidence
        consistency_factor = 1.0 - strategy.consistency_score
        confidence_factor = 1.0 - strategy.fusion_confidence
        
        volatility_estimate = (consistency_factor + confidence_factor) / 2
        
        return volatility_estimate
    
    def _simulate_returns(self, strategy: StrategyScore, n_simulations: int) -> np.ndarray:
        """Simulate returns using Monte Carlo"""
        mean_return = strategy.profit_potential * strategy.signal_confidence
        volatility = self._estimate_volatility(strategy)
        
        returns = np.random.normal(mean_return, volatility, n_simulations)
        
        return returns
    
    def _simulate_drawdowns(self, strategy: StrategyScore, n_simulations: int) -> np.ndarray:
        """Simulate drawdowns using Monte Carlo"""
        mean_drawdown = strategy.drawdown_risk
        volatility = self._estimate_volatility(strategy) * 0.5  # Lower volatility for drawdowns
        
        drawdowns = np.random.normal(mean_drawdown, volatility, n_simulations)
        drawdowns = np.clip(drawdowns, 0, 1)  # Clip to valid range
        
        return drawdowns
    
    def _simulate_sharpe_ratios(self, strategy: StrategyScore, n_simulations: int) -> np.ndarray:
        """Simulate Sharpe ratios using Monte Carlo"""
        returns = self._simulate_returns(strategy, n_simulations)
        volatilities = np.random.normal(
            self._estimate_volatility(strategy), 
            self._estimate_volatility(strategy) * 0.2, 
            n_simulations
        )
        volatilities = np.clip(volatilities, 0.01, 1.0)  # Avoid division by zero
        
        sharpe_ratios = (returns - self.config.risk_free_rate) / volatilities
        
        return sharpe_ratios
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval from data"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        
        return (lower_bound, upper_bound)

class EnhancedRankingEngine:
    """
    Advanced ranking system with multiple criteria, portfolio optimization,
    and risk-adjusted scoring
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.logger = logging.getLogger(f"{__name__}.EnhancedRankingEngine")
        
        # Initialize components
        self.multi_criteria_evaluator = MultiCriteriaEvaluator(self.config)
        self.portfolio_calculator = PortfolioFitCalculator(self.config)
        self.risk_scorer = RiskAdjustedScorer(self.config)
        
        self.logger.info("🏆 Enhanced Ranking Engine initialized")
    
    def compute_final_ranking(self, strategies: List[StrategyScore]) -> List[EnhancedStrategyRanking]:
        """
        Compute final enhanced ranking with all factors
        
        Args:
            strategies: List of strategy scores from Baustein B3
            
        Returns:
            List of enhanced strategy rankings
        """
        self.logger.info(f"🚀 Computing final ranking for {len(strategies)} strategies")
        
        if not strategies:
            self.logger.warning("⚠️ No strategies provided for ranking")
            return []
        
        # Step 1: Multi-criteria evaluation
        evaluated_strategies = self.multi_criteria_evaluator.evaluate_strategies(strategies)
        
        if not evaluated_strategies:
            self.logger.warning("⚠️ No strategies passed quality filters")
            return []
        
        # Step 2: Calculate enhanced rankings
        enhanced_rankings = []
        
        for strategy in evaluated_strategies:
            # Portfolio fit calculation
            portfolio_fit = self.portfolio_calculator.calculate_portfolio_fit(
                strategy, evaluated_strategies
            )
            
            # Diversification score
            diversification_score = self.portfolio_calculator.calculate_diversification_score(
                strategy, evaluated_strategies
            )
            
            # Risk-adjusted score
            risk_adjusted_score = self.risk_scorer.calculate_risk_adjusted_score(strategy)
            
            # Confidence intervals
            confidence_intervals = self.risk_scorer.calculate_confidence_intervals(strategy)
            
            # Performance projections
            performance_projections = self.risk_scorer.calculate_performance_projections(strategy)
            
            # Risk metrics
            risk_metrics = {
                "volatility_estimate": self.risk_scorer._estimate_volatility(strategy),
                "max_drawdown_estimate": strategy.drawdown_risk,
                "risk_adjusted_return": risk_adjusted_score,
                "sharpe_estimate": risk_adjusted_score / 2.0  # Rough Sharpe estimate
            }
            
            # Final ranking score calculation
            final_ranking_score = (
                strategy.composite_score * self.config.base_score_weight +
                portfolio_fit * self.config.portfolio_fit_weight +
                diversification_score * self.config.diversification_weight +
                (risk_adjusted_score / 10.0) * self.config.risk_adjustment_weight
            )
            
            # Create enhanced ranking
            enhanced_ranking = EnhancedStrategyRanking(
                strategy=strategy,
                portfolio_fit=portfolio_fit,
                diversification_score=diversification_score,
                risk_adjusted_score=risk_adjusted_score,
                final_ranking_score=final_ranking_score,
                rank_position=0,  # Will be set after sorting
                confidence_intervals=confidence_intervals,
                performance_projections=performance_projections,
                risk_metrics=risk_metrics
            )
            
            enhanced_rankings.append(enhanced_ranking)
        
        # Step 3: Sort by final ranking score and assign positions
        enhanced_rankings.sort(key=lambda x: x.final_ranking_score, reverse=True)
        
        for i, ranking in enumerate(enhanced_rankings):
            ranking.rank_position = i + 1
        
        self.logger.info(f"🏆 Final ranking computed: {len(enhanced_rankings)} strategies ranked")
        
        # Log top strategies
        for i, ranking in enumerate(enhanced_rankings[:5]):
            self.logger.info(
                f"  #{i+1}: {ranking.strategy.name} "
                f"(Score: {ranking.final_ranking_score:.3f}, "
                f"Portfolio Fit: {ranking.portfolio_fit:.3f}, "
                f"Risk-Adj: {ranking.risk_adjusted_score:.3f})"
            )
        
        return enhanced_rankings
    
    def get_ranking_summary(self, enhanced_rankings: List[EnhancedStrategyRanking]) -> Dict[str, Any]:
        """
        Get comprehensive ranking summary
        
        Args:
            enhanced_rankings: List of enhanced strategy rankings
            
        Returns:
            Dictionary with ranking summary statistics
        """
        if not enhanced_rankings:
            return {"total_strategies": 0, "summary": "No strategies ranked"}
        
        summary = {
            "total_strategies": len(enhanced_rankings),
            "ranking_timestamp": datetime.now().isoformat(),
            "config": {
                "criteria_weights": {k.value: v for k, v in self.config.criteria_weights.items()},
                "portfolio_fit_weight": self.config.portfolio_fit_weight,
                "diversification_weight": self.config.diversification_weight,
                "risk_adjustment_weight": self.config.risk_adjustment_weight,
                "base_score_weight": self.config.base_score_weight
            },
            "top_strategy": {
                "name": enhanced_rankings[0].strategy.name,
                "final_score": enhanced_rankings[0].final_ranking_score,
                "portfolio_fit": enhanced_rankings[0].portfolio_fit,
                "risk_adjusted_score": enhanced_rankings[0].risk_adjusted_score
            },
            "score_statistics": {
                "mean_final_score": np.mean([r.final_ranking_score for r in enhanced_rankings]),
                "std_final_score": np.std([r.final_ranking_score for r in enhanced_rankings]),
                "min_final_score": min(r.final_ranking_score for r in enhanced_rankings),
                "max_final_score": max(r.final_ranking_score for r in enhanced_rankings)
            },
            "portfolio_statistics": {
                "mean_portfolio_fit": np.mean([r.portfolio_fit for r in enhanced_rankings]),
                "mean_diversification": np.mean([r.diversification_score for r in enhanced_rankings]),
                "mean_risk_adjusted": np.mean([r.risk_adjusted_score for r in enhanced_rankings])
            }
        }
        
        return summary

# Factory function for easy instantiation
def create_enhanced_ranking_engine(
    criteria_weights: Optional[Dict[str, float]] = None,
    portfolio_weights: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.02
) -> EnhancedRankingEngine:
    """
    Factory function for Enhanced Ranking Engine
    
    Args:
        criteria_weights: Custom criteria weights
        portfolio_weights: Custom portfolio optimization weights
        risk_free_rate: Risk-free rate for Sharpe calculations
        
    Returns:
        Configured EnhancedRankingEngine instance
    """
    config = RankingConfig()
    
    # Update criteria weights if provided
    if criteria_weights:
        for criteria_name, weight in criteria_weights.items():
            try:
                criteria = StrategyRankingCriteria(criteria_name)
                config.criteria_weights[criteria] = weight
            except ValueError:
                logger.warning(f"⚠️ Unknown criteria: {criteria_name}")
    
    # Update portfolio weights if provided
    if portfolio_weights:
        config.portfolio_fit_weight = portfolio_weights.get("portfolio_fit", config.portfolio_fit_weight)
        config.diversification_weight = portfolio_weights.get("diversification", config.diversification_weight)
        config.risk_adjustment_weight = portfolio_weights.get("risk_adjustment", config.risk_adjustment_weight)
        config.base_score_weight = portfolio_weights.get("base_score", config.base_score_weight)
    
    # Update risk-free rate
    config.risk_free_rate = risk_free_rate
    
    return EnhancedRankingEngine(config)

# Example usage and testing
def main():
    """Example usage of Enhanced Ranking Engine"""
    
    # Create sample strategies
    sample_strategies = [
        StrategyScore(
            name="Professional_Tickdata_Strategy_1",
            signal_confidence=0.85,
            risk_reward_ratio=2.5,
            opportunity_score=0.78,
            fusion_confidence=0.82,
            consistency_score=0.75,
            profit_potential=0.88,
            drawdown_risk=0.15
        ),
        StrategyScore(
            name="Professional_Tickdata_Strategy_2",
            signal_confidence=0.92,
            risk_reward_ratio=3.2,
            opportunity_score=0.86,
            fusion_confidence=0.89,
            consistency_score=0.83,
            profit_potential=0.94,
            drawdown_risk=0.12
        ),
        StrategyScore(
            name="Professional_Tickdata_Strategy_3",
            signal_confidence=0.79,
            risk_reward_ratio=2.1,
            opportunity_score=0.73,
            fusion_confidence=0.77,
            consistency_score=0.71,
            profit_potential=0.81,
            drawdown_risk=0.18
        )
    ]
    
    # Create enhanced ranking engine
    ranking_engine = create_enhanced_ranking_engine()
    
    # Compute final rankings
    enhanced_rankings = ranking_engine.compute_final_ranking(sample_strategies)
    
    # Get summary
    summary = ranking_engine.get_ranking_summary(enhanced_rankings)
    
    print("🏆 Enhanced Ranking Results:")
    print(json.dumps(summary, indent=2, default=str))
    
    print("\n📊 Top Strategies:")
    for ranking in enhanced_rankings:
        print(f"#{ranking.rank_position}: {ranking.strategy.name}")
        print(f"  Final Score: {ranking.final_ranking_score:.3f}")
        print(f"  Portfolio Fit: {ranking.portfolio_fit:.3f}")
        print(f"  Risk-Adjusted: {ranking.risk_adjusted_score:.3f}")
        print(f"  Diversification: {ranking.diversification_score:.3f}")
        print()

if __name__ == "__main__":
    main()