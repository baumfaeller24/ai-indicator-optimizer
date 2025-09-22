#!/usr/bin/env python3
"""
🧩 BAUSTEIN B3 DEMO: AI Strategy Evaluator
Vollständige Demo für KI-basierte Strategien-Bewertung

Demonstriert:
- Top-5-Strategien automatisches Ranking
- Multi-Kriterien-Bewertungsalgorithmus
- Performance-basierte Strategien-Bewertung
- Integration mit Baustein B2
- Konfidenz-gewichtete Scores
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

print("🧩 BAUSTEIN B3: AI STRATEGY EVALUATOR DEMO")
print("=" * 70)
print(f"Start Time: {datetime.now()}")
print()

# Mock-Enums und Klassen für Demo
class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class StrategyRankingCriteria(Enum):
    PROFITABILITY = "profitability"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    CONFIDENCE_SCORE = "confidence_score"
    MULTIMODAL_CONSISTENCY = "multimodal_consistency"

@dataclass
class MockStrategyAnalysis:
    symbol: str
    timeframe: str
    trading_signal: TradingSignal
    signal_confidence: float
    risk_reward_ratio: float
    opportunity_score: float
    risk_score: float
    timestamp: datetime
    processing_time: float
    multimodal_features: Dict[str, Any]
    confidence_breakdown: Dict[str, float]

@dataclass
class MockStrategyPerformance:
    strategy_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    volatility: float
    trades_count: int
    avg_trade_duration: float
    confidence_score: float
    multimodal_score: float

@dataclass
class MockRankedStrategy:
    rank: int
    strategy_analysis: MockStrategyAnalysis
    performance: MockStrategyPerformance
    ranking_score: float
    ranking_criteria: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    confidence_level: str

def create_mock_strategy_analysis(symbol: str, timeframe: str, signal: TradingSignal) -> MockStrategyAnalysis:
    """Erstelle Mock-Strategien-Analyse"""
    
    # Simuliere realistische Werte basierend auf Signal
    if signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY]:
        confidence = 0.7 + (0.2 * (1 if signal == TradingSignal.STRONG_BUY else 0.5))
        risk_reward = 2.0 + (1.0 * (1 if signal == TradingSignal.STRONG_BUY else 0.5))
        opportunity = 0.8 + (0.15 * (1 if signal == TradingSignal.STRONG_BUY else 0.5))
        risk = 0.3 - (0.1 * (1 if signal == TradingSignal.STRONG_BUY else 0.5))
    elif signal in [TradingSignal.STRONG_SELL, TradingSignal.SELL]:
        confidence = 0.6 + (0.2 * (1 if signal == TradingSignal.STRONG_SELL else 0.5))
        risk_reward = 1.8 + (0.7 * (1 if signal == TradingSignal.STRONG_SELL else 0.5))
        opportunity = 0.7 + (0.15 * (1 if signal == TradingSignal.STRONG_SELL else 0.5))
        risk = 0.4 - (0.1 * (1 if signal == TradingSignal.STRONG_SELL else 0.5))
    else:  # HOLD
        confidence = 0.5
        risk_reward = 1.2
        opportunity = 0.4
        risk = 0.6
    
    return MockStrategyAnalysis(
        symbol=symbol,
        timeframe=timeframe,
        trading_signal=signal,
        signal_confidence=confidence,
        risk_reward_ratio=risk_reward,
        opportunity_score=opportunity,
        risk_score=risk,
        timestamp=datetime.now(),
        processing_time=0.15,
        multimodal_features={
            "fusion_confidence": confidence * 0.9,
            "technical_confidence": confidence * 0.95,
            "vision_confidence": confidence * 0.85,
            "fused_features": {
                "trend_strength": 0.7 if signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY] else -0.7 if signal in [TradingSignal.STRONG_SELL, TradingSignal.SELL] else 0.0,
                "momentum": opportunity,
                "volatility": risk
            }
        },
        confidence_breakdown={
            "technical_analysis": confidence * 0.9,
            "pattern_recognition": confidence * 0.8,
            "multimodal_fusion": confidence * 0.85,
            "confidence_consistency": confidence * 0.9
        }
    )

def create_mock_performance(analysis: MockStrategyAnalysis) -> MockStrategyPerformance:
    """Erstelle Mock-Performance basierend auf Analyse"""
    
    # Performance basierend auf Signal-Stärke und Konfidenz
    base_performance = analysis.signal_confidence * analysis.opportunity_score
    
    # Realistische Performance-Metriken
    total_return = base_performance * 0.20  # Max 20% Return
    sharpe_ratio = base_performance * 2.5   # Max 2.5 Sharpe
    max_drawdown = analysis.risk_score * 0.15  # Max 15% Drawdown
    win_rate = 0.45 + (base_performance * 0.25)  # 45-70% Win Rate
    profit_factor = 1.0 + (analysis.risk_reward_ratio * 0.3)
    calmar_ratio = total_return / max(max_drawdown, 0.01)
    volatility = 0.12 + (analysis.risk_score * 0.08)
    
    multimodal_score = analysis.multimodal_features["fusion_confidence"]
    
    return MockStrategyPerformance(
        strategy_id=f"{analysis.symbol}_{analysis.timeframe}_{analysis.timestamp.strftime('%H%M%S')}",
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        calmar_ratio=calmar_ratio,
        volatility=volatility,
        trades_count=100,
        avg_trade_duration=24.0,
        confidence_score=analysis.signal_confidence,
        multimodal_score=multimodal_score
    )

def calculate_ranking_score(performance: MockStrategyPerformance, criteria: StrategyRankingCriteria) -> float:
    """Berechne Ranking-Score"""
    
    # Normalisierte Metriken (0-1)
    normalized = {
        "total_return": min(max(performance.total_return / 0.20, 0), 1),
        "sharpe_ratio": min(max(performance.sharpe_ratio / 2.5, 0), 1),
        "max_drawdown": min(max(1.0 - performance.max_drawdown / 0.15, 0), 1),
        "win_rate": performance.win_rate,
        "profit_factor": min(max((performance.profit_factor - 1.0) / 2.0, 0), 1),
        "confidence_score": performance.confidence_score,
        "multimodal_score": performance.multimodal_score,
        "calmar_ratio": min(max(performance.calmar_ratio / 3.0, 0), 1)
    }
    
    # Gewichtungen je nach Kriterium
    if criteria == StrategyRankingCriteria.PROFITABILITY:
        weights = {"total_return": 0.4, "profit_factor": 0.3, "win_rate": 0.2, "confidence_score": 0.1}
    elif criteria == StrategyRankingCriteria.RISK_ADJUSTED_RETURN:
        weights = {"sharpe_ratio": 0.35, "calmar_ratio": 0.25, "max_drawdown": 0.20, "multimodal_score": 0.20}
    elif criteria == StrategyRankingCriteria.CONFIDENCE_SCORE:
        weights = {"confidence_score": 0.4, "multimodal_score": 0.3, "sharpe_ratio": 0.2, "total_return": 0.1}
    else:  # MULTIMODAL_CONSISTENCY
        weights = {"multimodal_score": 0.5, "confidence_score": 0.3, "sharpe_ratio": 0.2}
    
    score = sum(normalized.get(metric, 0) * weight for metric, weight in weights.items())
    return min(score, 1.0)

def analyze_strengths_weaknesses(performance: MockStrategyPerformance) -> tuple[List[str], List[str]]:
    """Analysiere Stärken und Schwächen"""
    
    strengths = []
    weaknesses = []
    
    # Stärken
    if performance.sharpe_ratio > 1.8:
        strengths.append(f"Excellent risk-adjusted returns (Sharpe: {performance.sharpe_ratio:.2f})")
    if performance.total_return > 0.15:
        strengths.append(f"High profitability ({performance.total_return:.1%} return)")
    if performance.max_drawdown < 0.05:
        strengths.append(f"Low drawdown risk ({performance.max_drawdown:.1%})")
    if performance.win_rate > 0.65:
        strengths.append(f"High win rate ({performance.win_rate:.1%})")
    if performance.multimodal_score > 0.8:
        strengths.append(f"Strong multimodal consistency ({performance.multimodal_score:.1%})")
    
    # Schwächen
    if performance.sharpe_ratio < 1.2:
        weaknesses.append(f"Below-average risk-adjusted returns (Sharpe: {performance.sharpe_ratio:.2f})")
    if performance.max_drawdown > 0.08:
        weaknesses.append(f"Elevated drawdown risk ({performance.max_drawdown:.1%})")
    if performance.win_rate < 0.55:
        weaknesses.append(f"Low win rate ({performance.win_rate:.1%})")
    if performance.multimodal_score < 0.7:
        weaknesses.append(f"Inconsistent multimodal signals ({performance.multimodal_score:.1%})")
    
    # Fallbacks
    if not strengths:
        strengths.append("Balanced performance profile")
    if not weaknesses:
        weaknesses.append("No significant weaknesses identified")
    
    return strengths, weaknesses

def generate_recommendation(ranking_score: float) -> str:
    """Generiere Empfehlung"""
    
    if ranking_score > 0.8:
        return "STRONG BUY - Excellent strategy with high confidence"
    elif ranking_score > 0.7:
        return "BUY - Good strategy with solid fundamentals"
    elif ranking_score > 0.6:
        return "MODERATE BUY - Acceptable strategy with some limitations"
    elif ranking_score > 0.4:
        return "HOLD - Average strategy, monitor closely"
    else:
        return "AVOID - Below-average strategy with significant risks"

def determine_confidence_level(performance: MockStrategyPerformance) -> str:
    """Bestimme Konfidenz-Level"""
    
    overall_confidence = (performance.confidence_score + performance.multimodal_score) / 2.0
    
    if overall_confidence > 0.8:
        return "HIGH"
    elif overall_confidence > 0.6:
        return "MEDIUM"
    else:
        return "LOW"

def demo_ai_strategy_evaluator():
    """Hauptdemo für Baustein B3"""
    
    print("🔄 Initializing AI Strategy Evaluator...")
    
    # Simuliere verschiedene Markt-Szenarien
    market_scenarios = [
        ("EUR/USD", "1h", TradingSignal.STRONG_BUY),
        ("EUR/USD", "4h", TradingSignal.BUY),
        ("GBP/USD", "1h", TradingSignal.HOLD),
        ("GBP/USD", "4h", TradingSignal.SELL),
        ("USD/JPY", "1h", TradingSignal.STRONG_SELL),
        ("USD/JPY", "4h", TradingSignal.BUY),
        ("EUR/GBP", "1h", TradingSignal.STRONG_BUY),
        ("AUD/USD", "4h", TradingSignal.HOLD)
    ]
    
    print(f"📊 Evaluating {len(market_scenarios)} strategy configurations...")
    
    # Generiere Strategien-Analysen
    strategy_analyses = []
    for symbol, timeframe, signal in market_scenarios:
        analysis = create_mock_strategy_analysis(symbol, timeframe, signal)
        strategy_analyses.append(analysis)
    
    # Evaluiere Performance
    strategy_performances = []
    for analysis in strategy_analyses:
        performance = create_mock_performance(analysis)
        strategy_performances.append(performance)
    
    # Teste verschiedene Ranking-Kriterien
    ranking_criteria = StrategyRankingCriteria.RISK_ADJUSTED_RETURN
    
    print(f"🎯 Ranking strategies by: {ranking_criteria.value}")
    
    # Erstelle gerankte Strategien
    ranked_strategies = []
    for analysis, performance in zip(strategy_analyses, strategy_performances):
        ranking_score = calculate_ranking_score(performance, ranking_criteria)
        strengths, weaknesses = analyze_strengths_weaknesses(performance)
        recommendation = generate_recommendation(ranking_score)
        confidence_level = determine_confidence_level(performance)
        
        ranked_strategy = MockRankedStrategy(
            rank=0,  # Wird später gesetzt
            strategy_analysis=analysis,
            performance=performance,
            ranking_score=ranking_score,
            ranking_criteria={
                "total_return": performance.total_return,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "confidence_score": performance.confidence_score,
                "multimodal_score": performance.multimodal_score
            },
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
            confidence_level=confidence_level
        )
        
        ranked_strategies.append(ranked_strategy)
    
    # Sortiere und setze Ranks
    ranked_strategies.sort(key=lambda x: x.ranking_score, reverse=True)
    for i, strategy in enumerate(ranked_strategies):
        strategy.rank = i + 1
    
    # Zeige Top-5 Ergebnisse
    top_5 = ranked_strategies[:5]
    
    print(f"\n🏆 TOP-5 STRATEGIES EVALUATION RESULTS:")
    print(f"Ranking Criteria: {ranking_criteria.value}")
    print(f"Total Strategies Evaluated: {len(ranked_strategies)}")
    
    for strategy in top_5:
        print(f"\n🥇 RANK #{strategy.rank}")
        print(f"   📊 Strategy: {strategy.strategy_analysis.symbol} {strategy.strategy_analysis.timeframe}")
        print(f"   📈 Signal: {strategy.strategy_analysis.trading_signal.value.upper()}")
        print(f"   ⭐ Ranking Score: {strategy.ranking_score:.3f}")
        print(f"   🎯 Confidence Level: {strategy.confidence_level}")
        print(f"   💡 Recommendation: {strategy.recommendation}")
        
        print(f"\n   📈 Performance Metrics:")
        print(f"      Total Return: {strategy.performance.total_return:.1%}")
        print(f"      Sharpe Ratio: {strategy.performance.sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {strategy.performance.max_drawdown:.1%}")
        print(f"      Win Rate: {strategy.performance.win_rate:.1%}")
        print(f"      Profit Factor: {strategy.performance.profit_factor:.2f}")
        print(f"      Multimodal Score: {strategy.performance.multimodal_score:.1%}")
        
        print(f"\n   💪 Key Strengths:")
        for strength in strategy.strengths[:2]:
            print(f"      • {strength}")
        
        if strategy.weaknesses and strategy.weaknesses[0] != "No significant weaknesses identified":
            print(f"\n   ⚠️  Areas for Improvement:")
            for weakness in strategy.weaknesses[:2]:
                print(f"      • {weakness}")
    
    # Performance-Statistiken
    print(f"\n📊 EVALUATION STATISTICS:")
    
    avg_ranking_score = statistics.mean([s.ranking_score for s in ranked_strategies])
    avg_confidence = statistics.mean([s.performance.confidence_score for s in ranked_strategies])
    avg_multimodal = statistics.mean([s.performance.multimodal_score for s in ranked_strategies])
    
    print(f"   Average Ranking Score: {avg_ranking_score:.3f}")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    print(f"   Average Multimodal Score: {avg_multimodal:.1%}")
    
    # Signal-Verteilung
    signal_distribution = {}
    for strategy in ranked_strategies:
        signal = strategy.strategy_analysis.trading_signal.value
        signal_distribution[signal] = signal_distribution.get(signal, 0) + 1
    
    print(f"\n   📊 Signal Distribution:")
    for signal, count in signal_distribution.items():
        print(f"      {signal.upper()}: {count} strategies")
    
    # Konfidenz-Level-Verteilung
    confidence_distribution = {}
    for strategy in ranked_strategies:
        level = strategy.confidence_level
        confidence_distribution[level] = confidence_distribution.get(level, 0) + 1
    
    print(f"\n   🎯 Confidence Level Distribution:")
    for level, count in confidence_distribution.items():
        print(f"      {level}: {count} strategies")
    
    print(f"\n✅ BAUSTEIN B3 DEMO COMPLETED SUCCESSFULLY!")
    print(f"🎉 AI Strategy Evaluator demonstrated comprehensive ranking capabilities!")
    
    return top_5

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        # Run Demo
        top_strategies = demo_ai_strategy_evaluator()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️  Total Processing Time: {processing_time:.3f} seconds")
        print(f"📊 Strategies per Second: {8/processing_time:.1f}")
        
        print(f"\n🎯 BAUSTEIN B3 READY FOR INTEGRATION!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()