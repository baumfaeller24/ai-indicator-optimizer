#!/usr/bin/env python3
"""
🧩 BAUSTEIN B3: AI Strategy Evaluator
KI-basierte Strategien-Bewertung und Top-5-Ranking-System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import statistics

class StrategyRankingCriteria(Enum):
    """Kriterien für Strategien-Ranking"""
    SIGNAL_CONFIDENCE = "signal_confidence"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    OPPORTUNITY_SCORE = "opportunity_score"
    FUSION_CONFIDENCE = "fusion_confidence"
    CONSISTENCY_SCORE = "consistency_score"
    PROFIT_POTENTIAL = "profit_potential"
    DRAWDOWN_RISK = "drawdown_risk"
    SHARPE_RATIO = "sharpe_ratio"


@dataclass
class StrategyScore:
    """Strategien-Score-Details"""
    strategy_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    signal_confidence_score: float
    risk_reward_score: float
    opportunity_score: float
    fusion_confidence_score: float
    consistency_score: float
    profit_potential_score: float
    drawdown_risk_score: float
    composite_score: float
    weighted_score: float
    rank_position: int = 0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    expected_sharpe: float = 0.0
    market_fit_score: float = 0.0
    execution_difficulty: float = 0.0
    liquidity_requirement: float = 0.0
    evaluation_notes: List[str] = field(default_factory=list)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class Top5StrategiesResult:
    """Top-5-Strategien Ergebnis"""
    timestamp: datetime
    symbol: str
    timeframe: str
    evaluation_mode: str
    top_strategies: List[StrategyScore]
    avg_composite_score: float
    score_distribution: Dict[str, float]
    category_distribution: Dict[str, int]
    evaluation_time: float
    total_strategies_evaluated: int
    market_conditions: Dict[str, Any]
    recommended_allocation: Dict[str, float]
    evaluation_quality: str
    confidence_level: float
    key_insights: List[str]
    risk_warnings: List[str]


class AIStrategyEvaluator:
    """🧩 BAUSTEIN B3: AI Strategy Evaluator"""
    
    def __init__(self, ranking_criteria: List[StrategyRankingCriteria] = None, output_dir: str = "data/strategy_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ranking_criteria = ranking_criteria or [
            StrategyRankingCriteria.SIGNAL_CONFIDENCE,
            StrategyRankingCriteria.RISK_REWARD_RATIO,
            StrategyRankingCriteria.OPPORTUNITY_SCORE
        ]
        
        self.evaluation_weights = {
            StrategyRankingCriteria.SIGNAL_CONFIDENCE: 0.25,
            StrategyRankingCriteria.RISK_REWARD_RATIO: 0.20,
            StrategyRankingCriteria.OPPORTUNITY_SCORE: 0.15,
            StrategyRankingCriteria.FUSION_CONFIDENCE: 0.15,
            StrategyRankingCriteria.CONSISTENCY_SCORE: 0.10,
            StrategyRankingCriteria.PROFIT_POTENTIAL: 0.10,
            StrategyRankingCriteria.DRAWDOWN_RISK: 0.05
        }
        
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.total_evaluation_time = 0.0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AI Strategy Evaluator initialized")
    
    def evaluate_and_rank_strategies(self, symbols: List[str] = None, timeframes: List[str] = None, max_strategies: int = 5, evaluation_mode: str = "comprehensive") -> Top5StrategiesResult:
        """Hauptfunktion: Evaluiere und ranke Strategien für Top-5-Liste"""
        start_time = datetime.now()
        evaluation_start = start_time.timestamp()
        
        try:
            self.logger.info(f"🔄 Starting strategy evaluation and ranking")
            
            symbols = symbols or ["EUR/USD"]  # Fokus auf EUR/USD wie in der Spec definiert
            timeframes = timeframes or ["1h", "4h"]
            
            strategy_scores = self._create_mock_strategy_scores(symbols, timeframes)
            ranked_strategies = self._rank_strategies(strategy_scores, max_strategies)
            market_conditions = {"dominant_market_regime": "trending", "average_signal_confidence": 0.75}
            recommended_allocation = self._calculate_recommended_allocation(ranked_strategies)
            evaluation_quality, confidence_level = self._assess_evaluation_quality(ranked_strategies)
            key_insights, risk_warnings = self._generate_insights_and_warnings(ranked_strategies, market_conditions)
            stats = self._calculate_evaluation_statistics(strategy_scores, ranked_strategies)
            
            evaluation_time = datetime.now().timestamp() - evaluation_start
            
            result = Top5StrategiesResult(
                timestamp=start_time,
                symbol="MULTI" if len(symbols) > 1 else symbols[0],
                timeframe="MULTI" if len(timeframes) > 1 else timeframes[0],
                evaluation_mode=evaluation_mode,
                top_strategies=ranked_strategies,
                avg_composite_score=stats["avg_composite_score"],
                score_distribution=stats["score_distribution"],
                category_distribution=stats["category_distribution"],
                evaluation_time=evaluation_time,
                total_strategies_evaluated=len(strategy_scores),
                market_conditions=market_conditions,
                recommended_allocation=recommended_allocation,
                evaluation_quality=evaluation_quality,
                confidence_level=confidence_level,
                key_insights=key_insights,
                risk_warnings=risk_warnings
            )
            
            self.total_evaluations += 1
            self.successful_evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            self.logger.info(f"✅ Strategy evaluation completed in {evaluation_time:.3f}s")
            return result
            
        except Exception as e:
            evaluation_time = datetime.now().timestamp() - evaluation_start
            self.total_evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            error_msg = f"Strategy evaluation failed: {e}"
            self.logger.error(error_msg)
            
            return Top5StrategiesResult(
                timestamp=start_time, symbol="ERROR", timeframe="ERROR", evaluation_mode=evaluation_mode,
                top_strategies=[], avg_composite_score=0.0, score_distribution={}, category_distribution={},
                evaluation_time=evaluation_time, total_strategies_evaluated=0, market_conditions={"error": str(e)},
                recommended_allocation={}, evaluation_quality="poor", confidence_level=0.0,
                key_insights=[], risk_warnings=[error_msg]
            )  
  
    def _create_mock_strategy_scores(self, symbols: List[str], timeframes: List[str]) -> List[StrategyScore]:
        """Erstelle Mock-Strategien-Scores für Demo"""
        strategy_scores = []
        
        for i, symbol in enumerate(symbols):
            for j, timeframe in enumerate(timeframes):
                base_score = 0.5 + (i * 0.1) + (j * 0.05)
                
                strategy_score = StrategyScore(
                    strategy_id=f"strategy_{i+1}_{j+1}_{symbol}_{timeframe}",
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    signal_confidence_score=base_score + np.random.uniform(-0.1, 0.1),
                    risk_reward_score=base_score + np.random.uniform(-0.1, 0.1),
                    opportunity_score=base_score + np.random.uniform(-0.1, 0.1),
                    fusion_confidence_score=base_score + np.random.uniform(-0.1, 0.1),
                    consistency_score=base_score + np.random.uniform(-0.1, 0.1),
                    profit_potential_score=base_score + np.random.uniform(-0.1, 0.1),
                    drawdown_risk_score=base_score + np.random.uniform(-0.1, 0.1),
                    composite_score=base_score,
                    weighted_score=base_score + np.random.uniform(-0.05, 0.05),
                    expected_return=base_score * 0.15,
                    expected_risk=0.10 + (1.0 - base_score) * 0.10,
                    expected_sharpe=base_score * 2.0,
                    market_fit_score=base_score,
                    execution_difficulty=1.0 - base_score,
                    liquidity_requirement=base_score * 0.5,
                    evaluation_notes=[f"Mock strategy for {symbol} {timeframe}"],
                    confidence_breakdown={"mock_confidence": base_score}
                )
                strategy_scores.append(strategy_score)
        
        return strategy_scores
    
    def _rank_strategies(self, strategy_scores: List[StrategyScore], max_strategies: int) -> List[StrategyScore]:
        """Ranke Strategien und gebe Top-N zurück"""
        try:
            sorted_strategies = sorted(strategy_scores, key=lambda x: x.weighted_score, reverse=True)
            
            for i, strategy in enumerate(sorted_strategies):
                strategy.rank_position = i + 1
            
            top_strategies = sorted_strategies[:max_strategies]
            self.logger.info(f"Ranked {len(strategy_scores)} strategies, returning top {len(top_strategies)}")
            return top_strategies
            
        except Exception as e:
            self.logger.error(f"Strategy ranking failed: {e}")
            return strategy_scores[:max_strategies] if strategy_scores else []
    
    def _calculate_recommended_allocation(self, ranked_strategies: List[StrategyScore]) -> Dict[str, float]:
        """Berechne empfohlene Portfolio-Allokation"""
        try:
            if not ranked_strategies:
                return {}
            
            total_score = sum(strategy.weighted_score for strategy in ranked_strategies)
            
            if total_score == 0:
                equal_weight = 1.0 / len(ranked_strategies)
                return {strategy.strategy_id: equal_weight for strategy in ranked_strategies}
            
            allocation = {}
            for strategy in ranked_strategies:
                weight = strategy.weighted_score / total_score
                allocation[strategy.strategy_id] = weight
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Allocation calculation failed: {e}")
            return {}
    
    def _assess_evaluation_quality(self, ranked_strategies: List[StrategyScore]) -> Tuple[str, float]:
        """Bewerte Qualität der Evaluation"""
        try:
            if not ranked_strategies:
                return "poor", 0.0
            
            avg_composite = statistics.mean([s.composite_score for s in ranked_strategies])
            avg_weighted = statistics.mean([s.weighted_score for s in ranked_strategies])
            
            scores = [s.weighted_score for s in ranked_strategies]
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            quality_score = (avg_composite + avg_weighted) / 2.0
            diversity_bonus = min(score_std * 2.0, 0.2)
            final_quality_score = min(quality_score + diversity_bonus, 1.0)
            
            if final_quality_score > 0.8:
                quality_label = "excellent"
            elif final_quality_score > 0.6:
                quality_label = "good"
            elif final_quality_score > 0.4:
                quality_label = "fair"
            else:
                quality_label = "poor"
            
            return quality_label, final_quality_score
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return "poor", 0.0
    
    def _generate_insights_and_warnings(self, ranked_strategies: List[StrategyScore], market_conditions: Dict) -> Tuple[List[str], List[str]]:
        """Generiere Key Insights und Risk Warnings"""
        insights = []
        warnings = []
        
        try:
            if not ranked_strategies:
                warnings.append("No strategies could be evaluated")
                return insights, warnings
            
            top_strategy = ranked_strategies[0]
            
            insights.append(f"Top strategy: {top_strategy.symbol} {top_strategy.timeframe}")
            insights.append(f"Best composite score: {top_strategy.composite_score:.1%}")
            insights.append(f"Expected return: {top_strategy.expected_return:.1%} with {top_strategy.expected_risk:.1%} risk")
            
            dominant_regime = market_conditions.get("dominant_market_regime", "unknown")
            if dominant_regime != "unknown":
                insights.append(f"Market regime: {dominant_regime}")
            
            avg_risk = statistics.mean([s.expected_risk for s in ranked_strategies])
            if avg_risk > 0.15:
                warnings.append(f"High average risk detected: {avg_risk:.1%}")
            
            difficult_strategies = [s for s in ranked_strategies if s.execution_difficulty > 0.7]
            if difficult_strategies:
                warnings.append(f"{len(difficult_strategies)} strategies may be difficult to execute")
            
            return insights, warnings
            
        except Exception as e:
            self.logger.error(f"Insights generation failed: {e}")
            return ["Insights generation failed"], [str(e)]
    
    def _calculate_evaluation_statistics(self, all_scores: List[StrategyScore], top_strategies: List[StrategyScore]) -> Dict[str, Any]:
        """Berechne Evaluierungs-Statistiken"""
        try:
            if not all_scores:
                return {"avg_composite_score": 0.0, "score_distribution": {}, "category_distribution": {}}
            
            avg_composite_score = statistics.mean([s.composite_score for s in all_scores])
            
            composite_scores = [s.composite_score for s in all_scores]
            score_distribution = {
                "min": min(composite_scores),
                "q1": np.percentile(composite_scores, 25),
                "median": np.percentile(composite_scores, 50),
                "q3": np.percentile(composite_scores, 75),
                "max": max(composite_scores),
                "std": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0.0
            }
            
            category_distribution = {"trend_following": 2, "mean_reversion": 1, "breakout": 1}
            
            return {
                "avg_composite_score": avg_composite_score,
                "score_distribution": score_distribution,
                "category_distribution": category_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {e}")
            return {"avg_composite_score": 0.0, "score_distribution": {}, "category_distribution": {}}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "evaluator_stats": {
                "total_evaluations": self.total_evaluations,
                "successful_evaluations": self.successful_evaluations,
                "success_rate": self.successful_evaluations / max(1, self.total_evaluations),
                "total_evaluation_time": self.total_evaluation_time,
                "average_evaluation_time": self.total_evaluation_time / max(1, self.total_evaluations),
                "evaluations_per_minute": (self.total_evaluations / self.total_evaluation_time * 60) if self.total_evaluation_time > 0 else 0,
                "ranking_criteria_count": len(self.ranking_criteria)
            }
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def demo_ai_strategy_evaluator():
    """🧩 Demo für AI Strategy Evaluator (Baustein B3)"""
    print("🧩 BAUSTEIN B3: AI STRATEGY EVALUATOR DEMO")
    print("=" * 70)
    
    evaluator = AIStrategyEvaluator(
        ranking_criteria=[
            StrategyRankingCriteria.SIGNAL_CONFIDENCE,
            StrategyRankingCriteria.RISK_REWARD_RATIO,
            StrategyRankingCriteria.OPPORTUNITY_SCORE,
            StrategyRankingCriteria.FUSION_CONFIDENCE
        ]
    )
    
    try:
        print("🔄 Evaluating and ranking strategies...")
        
        top5_result = evaluator.evaluate_and_rank_strategies(
            symbols=["EUR/USD", "GBP/USD"],
            timeframes=["1h", "4h"],
            max_strategies=5,
            evaluation_mode="comprehensive"
        )
        
        print(f"\n📊 TOP-5 STRATEGIES RESULTS:")
        print(f"Evaluation Time: {top5_result.evaluation_time:.3f}s")
        print(f"Total Strategies Evaluated: {top5_result.total_strategies_evaluated}")
        print(f"Evaluation Quality: {top5_result.evaluation_quality}")
        print(f"Confidence Level: {top5_result.confidence_level:.1%}")
        
        print(f"\n🏆 TOP-5 RANKED STRATEGIES:")
        for i, strategy in enumerate(top5_result.top_strategies, 1):
            print(f"  {i}. {strategy.symbol} {strategy.timeframe}")
            print(f"     Score: {strategy.weighted_score:.3f}")
            print(f"     Expected Return: {strategy.expected_return:.1%} | Risk: {strategy.expected_risk:.1%}")
        
        print(f"\n💡 KEY INSIGHTS:")
        for insight in top5_result.key_insights:
            print(f"  • {insight}")
        
        if top5_result.risk_warnings:
            print(f"\n⚠️  RISK WARNINGS:")
            for warning in top5_result.risk_warnings:
                print(f"  • {warning}")
        
        print(f"\n📈 RECOMMENDED ALLOCATION:")
        for strategy_id, allocation in top5_result.recommended_allocation.items():
            print(f"  • {strategy_id}: {allocation:.1%}")
        
        stats = evaluator.get_performance_stats()
        evaluator_stats = stats["evaluator_stats"]
        print(f"\n📊 PERFORMANCE STATS:")
        print(f"  • Total Evaluations: {evaluator_stats['total_evaluations']}")
        print(f"  • Success Rate: {evaluator_stats['success_rate']:.1%}")
        print(f"  • Avg Evaluation Time: {evaluator_stats['average_evaluation_time']:.3f}s")
        
        print(f"\n✅ BAUSTEIN B3 DEMO COMPLETED SUCCESSFULLY!")
        return top5_result
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    demo_ai_strategy_evaluator()