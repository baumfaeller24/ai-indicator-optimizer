#!/usr/bin/env python3
"""
🧩 BAUSTEIN B3 DEMO: AI Strategy Evaluator
Funktionierende Demo für KI-basierte Strategien-Bewertung
"""

import sys
import os
from datetime import datetime

print("🧩 BAUSTEIN B3: AI STRATEGY EVALUATOR DEMO")
print("=" * 70)
print(f"Start Time: {datetime.now()}")
print()

# Add current directory to Python path
sys.path.append('.')

try:
    # Import the AI Strategy Evaluator
    from ai_indicator_optimizer.ai.ai_strategy_evaluator import (
        AIStrategyEvaluator, 
        StrategyRankingCriteria,
        StrategyScore,
        Top5StrategiesResult
    )
    
    print("✅ Successfully imported AI Strategy Evaluator components")
    
    # Initialize the evaluator
    print("\n🔧 Initializing AI Strategy Evaluator...")
    
    evaluator = AIStrategyEvaluator(
        ranking_criteria=[
            StrategyRankingCriteria.SIGNAL_CONFIDENCE,
            StrategyRankingCriteria.RISK_REWARD_RATIO,
            StrategyRankingCriteria.OPPORTUNITY_SCORE,
            StrategyRankingCriteria.FUSION_CONFIDENCE
        ],
        output_dir="data/strategy_evaluation_demo"
    )
    
    print("✅ AI Strategy Evaluator initialized successfully")
    
    # Run strategy evaluation
    print("\n🔄 Running strategy evaluation and ranking...")
    
    top5_result = evaluator.evaluate_and_rank_strategies(
        symbols=["EUR/USD"],  # Fokus auf EUR/USD wie in der Spec definiert
        timeframes=["1h", "4h", "1d"],
        max_strategies=5,
        evaluation_mode="comprehensive"
    )
    
    print("✅ Strategy evaluation completed successfully")
    
    # Display results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"  • Evaluation Time: {top5_result.evaluation_time:.3f} seconds")
    print(f"  • Total Strategies Evaluated: {top5_result.total_strategies_evaluated}")
    print(f"  • Evaluation Quality: {top5_result.evaluation_quality}")
    print(f"  • Confidence Level: {top5_result.confidence_level:.1%}")
    print(f"  • Average Composite Score: {top5_result.avg_composite_score:.3f}")
    
    print(f"\n🏆 TOP-5 RANKED STRATEGIES:")
    for i, strategy in enumerate(top5_result.top_strategies, 1):
        print(f"  {i}. Strategy: {strategy.strategy_id}")
        print(f"     Symbol: {strategy.symbol} | Timeframe: {strategy.timeframe}")
        print(f"     Composite Score: {strategy.composite_score:.3f}")
        print(f"     Weighted Score: {strategy.weighted_score:.3f}")
        print(f"     Expected Return: {strategy.expected_return:.1%}")
        print(f"     Expected Risk: {strategy.expected_risk:.1%}")
        print(f"     Sharpe Ratio: {strategy.expected_sharpe:.2f}")
        print()
    
    print(f"💡 KEY INSIGHTS:")
    for insight in top5_result.key_insights:
        print(f"  • {insight}")
    
    if top5_result.risk_warnings:
        print(f"\n⚠️  RISK WARNINGS:")
        for warning in top5_result.risk_warnings:
            print(f"  • {warning}")
    
    print(f"\n📈 RECOMMENDED PORTFOLIO ALLOCATION:")
    for strategy_id, allocation in top5_result.recommended_allocation.items():
        short_id = strategy_id.split('_')[-2] + '_' + strategy_id.split('_')[-1]  # Shorten for display
        print(f"  • {short_id}: {allocation:.1%}")
    
    print(f"\n📊 MARKET CONDITIONS:")
    for key, value in top5_result.market_conditions.items():
        print(f"  • {key}: {value}")
    
    # Performance statistics
    print(f"\n📈 EVALUATOR PERFORMANCE STATS:")
    stats = evaluator.get_performance_stats()
    evaluator_stats = stats["evaluator_stats"]
    print(f"  • Total Evaluations: {evaluator_stats['total_evaluations']}")
    print(f"  • Success Rate: {evaluator_stats['success_rate']:.1%}")
    print(f"  • Average Evaluation Time: {evaluator_stats['average_evaluation_time']:.3f}s")
    print(f"  • Evaluations per Minute: {evaluator_stats['evaluations_per_minute']:.1f}")
    print(f"  • Ranking Criteria Count: {evaluator_stats['ranking_criteria_count']}")
    
    print(f"\n✅ BAUSTEIN B3 DEMO COMPLETED SUCCESSFULLY!")
    print(f"🎉 AI Strategy Evaluator is fully operational!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required modules are installed and paths are correct.")
    
except Exception as e:
    print(f"❌ Demo failed with error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nDemo completed at: {datetime.now()}")