#!/usr/bin/env python3
"""
Enhanced Ranking Engine - Task 5 Implementation

Implementiert Multi-Kriterien Evaluator mit 7+ Ranking-Faktoren für das
Top-5-Strategien-Ranking-System (Baustein C2).

Features:
- Multi-Kriterien Evaluator mit 7+ Ranking-Faktoren
- Portfolio-Fit-Calculator für Diversifikations-Scoring
- Risk-Adjusted-Scorer mit Sharpe-ähnlicher Berechnung
- Enhanced Ranking mit Final-Score-Computation
- Confidence-Intervals und Performance-Projections

Author: AI Indicator Optimizer Team
Date: 2025-09-22
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import json
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# ==================== ENUMS & CONSTANTS ====================

class StrategyRankingCriteria(Enum):
    """Ranking criteria for strategy evaluation"""
    SIGNAL_CONFIDENCE = "signal_confidence"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    OPPORTUNITY_SCORE = "opportunity_score"
    FUSION_CONFIDENCE = "fusion_confidence"
    CONSISTENCY_SCORE = "consistency_score"
    PROFIT_POTENTIAL = "profit_potential"
    DRAWDOWN_RISK = "drawdown_risk"
    PORTFOLIO_FIT = "portfolio_fit"
    DIVERSIFICATION_SCORE = "diversification_score"
    MARKET_ADAPTABILITY = "market_adaptability"


class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ==================== DATA MODELS ====================

@dataclass
class StrategyScore:
    """Strategy score from Baustein B3 AI Strategy Evaluator"""
    strategy_id: str
    name: str
    signal_confidence: float
    risk_reward_ratio: float
    opportunity_score: float
    fusion_confidence: float
    consistency_score: float
    profit_potential: float
    drawdown_risk: float
    composite_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "signal_confidence": self.signal_confidence,
            "risk_reward_ratio": self.risk_reward_ratio,
            "opportunity_score": self.opportunity_score,
            "fusion_confidence": self.fusion_confidence,
            "consistency_score": self.consistency_score,
            "profit_potential": self.profit_potential,
            "drawdown_risk": self.drawdown_risk,
            "composite_score": self.composite_score,
            "metadata": self.metadata
        }


@dataclass
class EnhancedStrategyRanking:
    """Enhanced strategy ranking with portfolio optimization"""
    strategy: StrategyScore
    portfolio_fit: float
    diversification_score: float
    risk_adjusted_score: float
    market_adaptability: float
    final_ranking_score: float
    rank_position: int
    confidence_intervals: Dict[str, Tuple[float, float]]
    performance_projections: Dict[str, float]
    risk_metrics: Dict[str, float]
    risk_level: RiskLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy": self.strategy.to_dict(),
            "portfolio_fit": self.portfolio_fit,
            "diversification_score": self.diversification_score,
            "risk_adjusted_score": self.risk_adjusted_score,
            "market_adaptability": self.market_adaptability,
            "final_ranking_score": self.final_ranking_score,
            "rank_position": self.rank_position,
            "confidence_intervals": self.confidence_intervals,
            "performance_projections": self.performance_projections,
            "risk_metrics": self.risk_metrics,
            "risk_level": self.risk_level.value
        }


@dataclass
class RankingConfig:
    """Configuration for enhanced ranking engine"""
    criteria_weights: Dict[StrategyRankingCriteria, float] = field(default_factory=lambda: {
        StrategyRankingCriteria.SIGNAL_CONFIDENCE: 0.20,
        StrategyRankingCriteria.RISK_REWARD_RATIO: 0.18,
        StrategyRankingCriteria.OPPORTUNITY_SCORE: 0.15,
        StrategyRankingCriteria.FUSION_CONFIDENCE: 0.12,
        StrategyRankingCriteria.CONSISTENCY_SCORE: 0.10,
        StrategyRankingCriteria.PROFIT_POTENTIAL: 0.10,
        StrategyRankingCriteria.DRAWDOWN_RISK: 0.08,
        StrategyRankingCriteria.PORTFOLIO_FIT: 0.04,
        StrategyRankingCriteria.DIVERSIFICATION_SCORE: 0.02,
        StrategyRankingCriteria.MARKET_ADAPTABILITY: 0.01
    })
    min_confidence_threshold: float = 0.5
    min_composite_score: float = 0.4
    max_strategies: int = 5
    enable_risk_adjustment: bool = True
    enable_portfolio_optimization: bool = True
    confidence_level: float = 0.95
    
    def __post_init__(self):
        """Validate configuration"""
        total_weight = sum(self.criteria_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")


# ==================== ENHANCED RANKING ENGINE ====================

class EnhancedRankingEngine:
    """
    Advanced ranking system with multiple criteria and portfolio optimization
    
    Features:
    - Multi-criteria evaluation with configurable weights
    - Portfolio fit calculation for diversification
    - Risk-adjusted scoring with Sharpe-like metrics
    - Confidence intervals and performance projections
    - Market adaptability assessment
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.logger = self._setup_logging()
        self.scaler = MinMaxScaler()
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        
        self.logger.info("🏆 Enhanced Ranking Engine initialized")
        self.logger.info(f"📊 Criteria weights: {len(self.config.criteria_weights)} factors")
        self.logger.info(f"🎯 Max strategies: {self.config.max_strategies}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ranking engine"""
        logger = logging.getLogger(f"{__name__}.EnhancedRankingEngine")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    async def calculate_enhanced_rankings(
        self, 
        strategies: List[StrategyScore]
    ) -> List[EnhancedStrategyRanking]:
        """
        Calculate enhanced rankings with all criteria
        
        Args:
            strategies: List of strategy scores from Baustein B3
            
        Returns:
            List of enhanced strategy rankings (Top-5)
        """
        start_time = time.time()
        
        self.logger.info(f"🏆 Starting enhanced ranking calculation for {len(strategies)} strategies")
        
        # Filter strategies by minimum thresholds
        filtered_strategies = self._filter_strategies(strategies)
        self.logger.info(f"📊 Filtered to {len(filtered_strategies)} strategies above thresholds")
        
        if len(filtered_strategies) == 0:
            self.logger.warning("⚠️ No strategies meet minimum criteria")
            return []
        
        # Calculate enhanced metrics for all strategies
        enhanced_rankings = []
        
        for strategy in filtered_strategies:
            # Calculate portfolio fit
            portfolio_fit = self.calculate_portfolio_fit(strategy, filtered_strategies)
            
            # Calculate diversification score
            diversification_score = self.calculate_diversification_score(strategy, filtered_strategies)
            
            # Calculate risk-adjusted score
            risk_adjusted_score = self.calculate_risk_adjusted_score(strategy)
            
            # Calculate market adaptability
            market_adaptability = self.calculate_market_adaptability(strategy)
            
            # Calculate final ranking score
            final_score = self.compute_final_ranking_score(
                strategy, portfolio_fit, diversification_score, 
                risk_adjusted_score, market_adaptability
            )
            
            # Calculate confidence intervals
            confidence_intervals = self.calculate_confidence_intervals(strategy)
            
            # Calculate performance projections
            performance_projections = self.calculate_performance_projections(strategy)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(strategy)
            
            # Determine risk level
            risk_level = self.determine_risk_level(strategy)
            
            enhanced_ranking = EnhancedStrategyRanking(
                strategy=strategy,
                portfolio_fit=portfolio_fit,
                diversification_score=diversification_score,
                risk_adjusted_score=risk_adjusted_score,
                market_adaptability=market_adaptability,
                final_ranking_score=final_score,
                rank_position=0,  # Will be set after sorting
                confidence_intervals=confidence_intervals,
                performance_projections=performance_projections,
                risk_metrics=risk_metrics,
                risk_level=risk_level
            )
            
            enhanced_rankings.append(enhanced_ranking)
        
        # Sort by final ranking score (descending)
        enhanced_rankings.sort(key=lambda x: x.final_ranking_score, reverse=True)
        
        # Assign rank positions and limit to top strategies
        top_rankings = []
        for i, ranking in enumerate(enhanced_rankings[:self.config.max_strategies]):
            ranking.rank_position = i + 1
            top_rankings.append(ranking)
        
        execution_time = time.time() - start_time
        self.evaluation_count += len(strategies)
        self.total_evaluation_time += execution_time
        
        self.logger.info(f"🎯 Enhanced ranking completed in {execution_time:.3f}s")
        self.logger.info(f"🏆 Top-{len(top_rankings)} strategies selected")
        
        if top_rankings:
            avg_score = np.mean([r.final_ranking_score for r in top_rankings])
            self.logger.info(f"📊 Average final score: {avg_score:.3f}")
        
        return top_rankings
    
    def _filter_strategies(self, strategies: List[StrategyScore]) -> List[StrategyScore]:
        """Filter strategies by minimum thresholds"""
        filtered = []
        
        for strategy in strategies:
            # Check minimum confidence threshold
            if strategy.signal_confidence < self.config.min_confidence_threshold:
                continue
            
            # Check minimum composite score
            if strategy.composite_score < self.config.min_composite_score:
                continue
            
            # Check for valid scores (no NaN or infinite values)
            scores = [
                strategy.signal_confidence,
                strategy.risk_reward_ratio,
                strategy.opportunity_score,
                strategy.fusion_confidence,
                strategy.consistency_score,
                strategy.profit_potential,
                strategy.drawdown_risk,
                strategy.composite_score
            ]
            
            if any(not np.isfinite(score) for score in scores):
                continue
            
            filtered.append(strategy)
        
        return filtered
    
    def calculate_portfolio_fit(
        self, 
        strategy: StrategyScore, 
        all_strategies: List[StrategyScore]
    ) -> float:
        """
        Calculate portfolio fit based on diversification potential
        
        Args:
            strategy: Target strategy
            all_strategies: All available strategies
            
        Returns:
            Portfolio fit score (0.0 to 1.0)
        """
        if len(all_strategies) <= 1:
            return 1.0
        
        # Create feature vector for the strategy
        strategy_features = np.array([
            strategy.signal_confidence,
            strategy.risk_reward_ratio,
            strategy.opportunity_score,
            strategy.fusion_confidence,
            strategy.consistency_score,
            strategy.profit_potential,
            1.0 - strategy.drawdown_risk  # Invert drawdown risk
        ])
        
        # Calculate correlation with other strategies
        correlations = []
        
        for other_strategy in all_strategies:
            if other_strategy.strategy_id == strategy.strategy_id:
                continue
            
            other_features = np.array([
                other_strategy.signal_confidence,
                other_strategy.risk_reward_ratio,
                other_strategy.opportunity_score,
                other_strategy.fusion_confidence,
                other_strategy.consistency_score,
                other_strategy.profit_potential,
                1.0 - other_strategy.drawdown_risk
            ])
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(strategy_features, other_features)[0, 1]
            if np.isfinite(correlation):
                correlations.append(abs(correlation))
        
        if not correlations:
            return 1.0
        
        # Portfolio fit is inversely related to average correlation
        avg_correlation = np.mean(correlations)
        portfolio_fit = 1.0 - avg_correlation
        
        return max(0.0, min(1.0, portfolio_fit))
    
    def calculate_diversification_score(
        self, 
        strategy: StrategyScore, 
        all_strategies: List[StrategyScore]
    ) -> float:
        """
        Calculate diversification score using clustering analysis
        
        Args:
            strategy: Target strategy
            all_strategies: All available strategies
            
        Returns:
            Diversification score (0.0 to 1.0)
        """
        if len(all_strategies) <= 2:
            return 1.0
        
        # Create feature matrix
        features = []
        strategy_index = -1
        
        for i, strat in enumerate(all_strategies):
            if strat.strategy_id == strategy.strategy_id:
                strategy_index = i
            
            features.append([
                strat.signal_confidence,
                strat.risk_reward_ratio,
                strat.opportunity_score,
                strat.fusion_confidence,
                strat.consistency_score,
                strat.profit_potential,
                1.0 - strat.drawdown_risk
            ])
        
        features_array = np.array(features)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Perform clustering
        n_clusters = min(3, len(all_strategies))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Calculate diversification based on cluster distribution
        strategy_cluster = cluster_labels[strategy_index]
        cluster_counts = np.bincount(cluster_labels)
        
        # Diversification is higher if strategy is in a smaller cluster
        cluster_size = cluster_counts[strategy_cluster]
        total_strategies = len(all_strategies)
        
        diversification_score = 1.0 - (cluster_size / total_strategies)
        
        return max(0.0, min(1.0, diversification_score))
    
    def calculate_risk_adjusted_score(self, strategy: StrategyScore) -> float:
        """
        Calculate Sharpe-like risk-adjusted performance score
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Risk-adjusted score (0.0 to 1.0)
        """
        # Calculate risk-adjusted return using Sharpe-like formula
        expected_return = strategy.profit_potential
        risk = strategy.drawdown_risk
        
        # Avoid division by zero
        if risk <= 0.001:
            risk = 0.001
        
        # Sharpe-like ratio
        sharpe_ratio = expected_return / risk
        
        # Normalize to 0-1 range using sigmoid function
        risk_adjusted_score = 1.0 / (1.0 + np.exp(-sharpe_ratio))
        
        # Apply consistency factor
        consistency_factor = strategy.consistency_score
        risk_adjusted_score *= consistency_factor
        
        return max(0.0, min(1.0, risk_adjusted_score))
    
    def calculate_market_adaptability(self, strategy: StrategyScore) -> float:
        """
        Calculate market adaptability score
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Market adaptability score (0.0 to 1.0)
        """
        # Market adaptability based on fusion confidence and opportunity score
        fusion_weight = 0.6
        opportunity_weight = 0.4
        
        adaptability = (
            fusion_weight * strategy.fusion_confidence +
            opportunity_weight * strategy.opportunity_score
        )
        
        # Apply signal confidence as a multiplier
        adaptability *= strategy.signal_confidence
        
        return max(0.0, min(1.0, adaptability))
    
    def compute_final_ranking_score(
        self,
        strategy: StrategyScore,
        portfolio_fit: float,
        diversification_score: float,
        risk_adjusted_score: float,
        market_adaptability: float
    ) -> float:
        """
        Compute final ranking score using weighted criteria
        
        Args:
            strategy: Base strategy score
            portfolio_fit: Portfolio fit score
            diversification_score: Diversification score
            risk_adjusted_score: Risk-adjusted score
            market_adaptability: Market adaptability score
            
        Returns:
            Final ranking score (0.0 to 1.0)
        """
        # Get criteria weights
        weights = self.config.criteria_weights
        
        # Calculate weighted score
        final_score = (
            weights[StrategyRankingCriteria.SIGNAL_CONFIDENCE] * strategy.signal_confidence +
            weights[StrategyRankingCriteria.RISK_REWARD_RATIO] * strategy.risk_reward_ratio +
            weights[StrategyRankingCriteria.OPPORTUNITY_SCORE] * strategy.opportunity_score +
            weights[StrategyRankingCriteria.FUSION_CONFIDENCE] * strategy.fusion_confidence +
            weights[StrategyRankingCriteria.CONSISTENCY_SCORE] * strategy.consistency_score +
            weights[StrategyRankingCriteria.PROFIT_POTENTIAL] * strategy.profit_potential +
            weights[StrategyRankingCriteria.DRAWDOWN_RISK] * (1.0 - strategy.drawdown_risk) +
            weights[StrategyRankingCriteria.PORTFOLIO_FIT] * portfolio_fit +
            weights[StrategyRankingCriteria.DIVERSIFICATION_SCORE] * diversification_score +
            weights[StrategyRankingCriteria.MARKET_ADAPTABILITY] * market_adaptability
        )
        
        return max(0.0, min(1.0, final_score))
    
    def calculate_confidence_intervals(
        self, 
        strategy: StrategyScore
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for key metrics
        
        Args:
            strategy: Strategy to analyze
            
        Returns:
            Dictionary of confidence intervals
        """
        confidence_level = self.config.confidence_level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Estimate standard errors (simplified approach)
        base_std = 0.05  # 5% base standard error
        
        intervals = {}
        
        # Key metrics with confidence intervals
        metrics = {
            "signal_confidence": strategy.signal_confidence,
            "profit_potential": strategy.profit_potential,
            "risk_reward_ratio": strategy.risk_reward_ratio,
            "composite_score": strategy.composite_score
        }
        
        for metric_name, value in metrics.items():
            # Estimate standard error based on value and consistency
            std_error = base_std * (1.0 - strategy.consistency_score * 0.5)
            margin_of_error = z_score * std_error
            
            lower_bound = max(0.0, value - margin_of_error)
            upper_bound = min(1.0, value + margin_of_error)
            
            intervals[metric_name] = (lower_bound, upper_bound)
        
        return intervals
    
    def calculate_performance_projections(
        self, 
        strategy: StrategyScore
    ) -> Dict[str, float]:
        """
        Calculate performance projections
        
        Args:
            strategy: Strategy to analyze
            
        Returns:
            Dictionary of performance projections
        """
        projections = {}
        
        # Expected monthly return
        monthly_return = strategy.profit_potential * 0.1  # 10% of profit potential per month
        projections["expected_monthly_return"] = monthly_return
        
        # Expected annual return
        annual_return = monthly_return * 12 * (1 + strategy.consistency_score * 0.2)
        projections["expected_annual_return"] = annual_return
        
        # Maximum drawdown projection
        max_drawdown = strategy.drawdown_risk * 1.5  # 150% of base drawdown risk
        projections["projected_max_drawdown"] = max_drawdown
        
        # Win rate projection
        win_rate = 0.5 + (strategy.signal_confidence - 0.5) * 0.6
        projections["projected_win_rate"] = max(0.3, min(0.8, win_rate))
        
        # Volatility projection
        volatility = strategy.drawdown_risk * 2.0
        projections["projected_volatility"] = volatility
        
        return projections
    
    def calculate_risk_metrics(self, strategy: StrategyScore) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            strategy: Strategy to analyze
            
        Returns:
            Dictionary of risk metrics
        """
        risk_metrics = {}
        
        # Value at Risk (simplified)
        var_95 = strategy.drawdown_risk * 1.65  # 95% VaR approximation
        risk_metrics["var_95"] = var_95
        
        # Expected Shortfall
        expected_shortfall = var_95 * 1.3
        risk_metrics["expected_shortfall"] = expected_shortfall
        
        # Risk-Return Ratio
        risk_return_ratio = strategy.profit_potential / max(0.001, strategy.drawdown_risk)
        risk_metrics["risk_return_ratio"] = risk_return_ratio
        
        # Stability Score
        stability_score = strategy.consistency_score * (1.0 - strategy.drawdown_risk)
        risk_metrics["stability_score"] = stability_score
        
        # Stress Test Score (how well strategy handles market stress)
        stress_test_score = strategy.fusion_confidence * strategy.signal_confidence
        risk_metrics["stress_test_score"] = stress_test_score
        
        return risk_metrics
    
    def determine_risk_level(self, strategy: StrategyScore) -> RiskLevel:
        """
        Determine risk level classification
        
        Args:
            strategy: Strategy to classify
            
        Returns:
            Risk level classification
        """
        # Calculate composite risk score
        risk_score = (
            strategy.drawdown_risk * 0.4 +
            (1.0 - strategy.consistency_score) * 0.3 +
            (1.0 - strategy.signal_confidence) * 0.2 +
            (1.0 - strategy.risk_reward_ratio) * 0.1
        )
        
        # Classify risk level
        if risk_score <= 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score <= 0.4:
            return RiskLevel.LOW
        elif risk_score <= 0.6:
            return RiskLevel.MEDIUM
        elif risk_score <= 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_evaluation_time = (
            self.total_evaluation_time / max(1, self.evaluation_count)
        )
        
        return {
            "total_evaluations": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_evaluation_time,
            "evaluations_per_second": 1.0 / max(0.001, avg_evaluation_time),
            "criteria_count": len(self.config.criteria_weights),
            "max_strategies": self.config.max_strategies
        }


# ==================== MOCK DATA GENERATOR ====================

def generate_mock_strategy_scores(count: int = 20) -> List[StrategyScore]:
    """Generate mock strategy scores for testing"""
    import random
    
    strategies = []
    
    for i in range(count):
        # Generate realistic but varied scores
        signal_confidence = random.uniform(0.4, 0.95)
        risk_reward_ratio = random.uniform(0.3, 0.9)
        opportunity_score = random.uniform(0.2, 0.85)
        fusion_confidence = random.uniform(0.5, 0.9)
        consistency_score = random.uniform(0.4, 0.8)
        profit_potential = random.uniform(0.3, 0.8)
        drawdown_risk = random.uniform(0.1, 0.6)
        
        # Calculate composite score
        composite_score = (
            signal_confidence * 0.25 +
            risk_reward_ratio * 0.20 +
            opportunity_score * 0.15 +
            fusion_confidence * 0.15 +
            consistency_score * 0.10 +
            profit_potential * 0.10 +
            (1.0 - drawdown_risk) * 0.05
        )
        
        strategy = StrategyScore(
            strategy_id=f"strategy_{i+1:03d}",
            name=f"AI_Strategy_{i+1}",
            signal_confidence=signal_confidence,
            risk_reward_ratio=risk_reward_ratio,
            opportunity_score=opportunity_score,
            fusion_confidence=fusion_confidence,
            consistency_score=consistency_score,
            profit_potential=profit_potential,
            drawdown_risk=drawdown_risk,
            composite_score=composite_score,
            metadata={
                "timeframe": random.choice(["1m", "5m", "15m", "1h", "4h"]),
                "symbol": "EUR/USD",
                "created_at": datetime.now().isoformat()
            }
        )
        
        strategies.append(strategy)
    
    return strategies


# ==================== MAIN EXECUTION ====================

async def main():
    """Test the Enhanced Ranking Engine"""
    print("🏆 Testing Enhanced Ranking Engine")
    print("=" * 60)
    
    # Create ranking engine
    config = RankingConfig(max_strategies=5)
    engine = EnhancedRankingEngine(config)
    
    # Generate mock strategies
    print("📊 Generating mock strategy scores...")
    strategies = generate_mock_strategy_scores(20)
    print(f"✅ Generated {len(strategies)} mock strategies")
    
    # Calculate enhanced rankings
    print("\n🔄 Calculating enhanced rankings...")
    start_time = time.time()
    
    rankings = await engine.calculate_enhanced_rankings(strategies)
    
    execution_time = time.time() - start_time
    
    # Display results
    print(f"\n🎯 Enhanced Ranking Results ({execution_time:.3f}s)")
    print("=" * 60)
    
    for ranking in rankings:
        print(f"#{ranking.rank_position}: {ranking.strategy.name}")
        print(f"  Final Score: {ranking.final_ranking_score:.3f}")
        print(f"  Portfolio Fit: {ranking.portfolio_fit:.3f}")
        print(f"  Risk Level: {ranking.risk_level.value}")
        print(f"  Expected Annual Return: {ranking.performance_projections['expected_annual_return']:.1%}")
        print()
    
    # Performance stats
    stats = engine.get_performance_stats()
    print("📊 Performance Statistics:")
    print(f"  Evaluations/sec: {stats['evaluations_per_second']:.1f}")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Criteria count: {stats['criteria_count']}")


if __name__ == "__main__":
    asyncio.run(main())