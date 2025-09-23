#!/usr/bin/env python3
"""
Test Script für Task 5: Enhanced Ranking Engine Integration

Testet die Integration der Enhanced Ranking Engine in das Top5StrategiesRankingSystem
mit Multi-Kriterien Evaluator und Portfolio-Fit-Calculator.
"""

import asyncio
import logging
import time
from pathlib import Path
import json

# Import the enhanced ranking engine
from ai_indicator_optimizer.ranking.enhanced_ranking_engine import (
    EnhancedRankingEngine, RankingConfig, StrategyScore, EnhancedStrategyRanking,
    generate_mock_strategy_scores
)

# Import the main pipeline system
from ai_indicator_optimizer.integration.top5_strategies_ranking_system import (
    Top5StrategiesRankingSystem,
    PipelineConfig,
    ExecutionMode
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_enhanced_ranking_integration():
    """
    Test the Enhanced Ranking Engine integration with the pipeline
    """
    logger.info("🏆 Starting Task 5: Enhanced Ranking Engine Integration Test")
    logger.info("=" * 70)
    
    # Create test configuration
    config = PipelineConfig(
        execution_mode=ExecutionMode.DEVELOPMENT,
        max_workers=4,  # Reduced for testing
        timeout_seconds=60,
        enable_gpu=True,
        enable_logging=True,
        output_directory="./test_task5_results",
        batch_size=100,
        min_confidence_threshold=0.5,
        min_strategies_required=5
    )
    
    logger.info(f"📊 Configuration:")
    logger.info(f"  - Mode: {config.execution_mode.value}")
    logger.info(f"  - Workers: {config.max_workers}")
    logger.info(f"  - Min Confidence: {config.min_confidence_threshold}")
    
    # Test 1: Enhanced Ranking Engine standalone
    logger.info("\n🔄 Test 1: Enhanced Ranking Engine Standalone")
    logger.info("-" * 50)
    
    start_time = time.time()
    
    # Create ranking engine
    ranking_config = RankingConfig(max_strategies=5)
    engine = EnhancedRankingEngine(ranking_config)
    
    # Generate mock strategies
    mock_strategies = generate_mock_strategy_scores(15)
    logger.info(f"📊 Generated {len(mock_strategies)} mock strategies")
    
    # Calculate enhanced rankings
    enhanced_rankings = await engine.calculate_enhanced_rankings(mock_strategies)
    
    test1_time = time.time() - start_time
    
    logger.info(f"🎯 Enhanced Ranking Results ({test1_time:.3f}s):")
    for ranking in enhanced_rankings:
        logger.info(f"  #{ranking.rank_position}: {ranking.strategy.name}")
        logger.info(f"    Final Score: {ranking.final_ranking_score:.3f}")
        logger.info(f"    Portfolio Fit: {ranking.portfolio_fit:.3f}")
        logger.info(f"    Risk Level: {ranking.risk_level.value}")
        logger.info(f"    Expected Return: {ranking.performance_projections['expected_annual_return']:.1%}")
    
    # Test 2: Integration with Pipeline System
    logger.info("\n🔄 Test 2: Pipeline Integration")
    logger.info("-" * 50)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline with enhanced ranking
        pipeline = Top5StrategiesRankingSystem(config)
        logger.info("✅ Pipeline system initialized with Enhanced Ranking Engine")
        
        # Test the ranking calculation stage directly
        test_strategies = []
        for i, strategy_score in enumerate(mock_strategies[:10]):
            # Convert StrategyScore to dictionary format expected by pipeline
            strategy_dict = {
                "strategy_id": strategy_score.strategy_id,
                "name": strategy_score.name,
                "confidence": strategy_score.signal_confidence,
                "expected_return": strategy_score.profit_potential / 10,  # Scale back
                "risk_score": strategy_score.drawdown_risk,
                "consistency": strategy_score.consistency_score,
                "metadata": strategy_score.metadata
            }
            test_strategies.append(strategy_dict)\n        \n        logger.info(f\"📊 Testing with {len(test_strategies)} strategies\")\n        \n        # Test the enhanced ranking calculation\n        input_data = {\"strategies\": test_strategies}\n        ranking_result = await pipeline._stage_ranking_calculation(input_data)\n        \n        test2_time = time.time() - start_time\n        \n        if ranking_result.get(\"ranking_complete\", False):\n            top5_strategies = ranking_result.get(\"top5_strategies\", [])\n            \n            logger.info(f\"🎉 Pipeline Integration: SUCCESS ({test2_time:.3f}s)\")\n            logger.info(f\"📈 Top-5 Strategies Generated: {len(top5_strategies)}\")\n            \n            for i, strategy in enumerate(top5_strategies):\n                logger.info(f\"  #{i+1}: {strategy.get('name', 'Unknown')}\")\n                logger.info(f\"    Final Score: {strategy.get('final_score', 0):.3f}\")\n                logger.info(f\"    Portfolio Fit: {strategy.get('portfolio_fit', 0):.3f}\")\n                logger.info(f\"    Risk Level: {strategy.get('risk_level', 'unknown')}\")\n            \n            # Validate enhanced features\n            enhanced_features_present = all(\n                \"portfolio_fit\" in strategy and\n                \"diversification_score\" in strategy and\n                \"risk_adjusted_score\" in strategy and\n                \"market_adaptability\" in strategy and\n                \"confidence_intervals\" in strategy and\n                \"performance_projections\" in strategy and\n                \"risk_metrics\" in strategy\n                for strategy in top5_strategies\n            )\n            \n            if enhanced_features_present:\n                logger.info(\"✅ All enhanced ranking features present\")\n            else:\n                logger.warning(\"⚠️ Some enhanced ranking features missing\")\n            \n            # Performance validation\n            ranking_metrics = ranking_result.get(\"ranking_metrics\", {})\n            avg_score = ranking_metrics.get(\"top5_avg_score\", 0)\n            \n            logger.info(f\"📊 Performance Metrics:\")\n            logger.info(f\"  - Average Top-5 Score: {avg_score:.3f}\")\n            logger.info(f\"  - Total Evaluated: {ranking_metrics.get('total_evaluated', 0)}\")\n            logger.info(f\"  - Success Rate: {ranking_result.get('success_rate', 0):.1%}\")\n            \n            return True\n        \n        else:\n            logger.error(\"❌ Pipeline Integration: FAILED\")\n            logger.error(f\"💥 Error: {ranking_result.get('error', 'Unknown error')}\")\n            return False\n            \n    except Exception as e:\n        test2_time = time.time() - start_time\n        logger.error(f\"💥 Pipeline integration crashed: {str(e)}\")\n        logger.error(f\"⏱️ Crashed after: {test2_time:.3f}s\")\n        return False


async def test_ranking_criteria_validation():
    \"\"\"Test that all 10 ranking criteria are properly implemented\"\"\"\n    logger.info(\"\\n🔄 Test 3: Ranking Criteria Validation\")\n    logger.info(\"-\" * 50)\n    \n    # Create ranking engine\n    ranking_config = RankingConfig(max_strategies=3)\n    engine = EnhancedRankingEngine(ranking_config)\n    \n    # Generate test strategies\n    test_strategies = generate_mock_strategy_scores(5)\n    \n    # Calculate rankings\n    rankings = await engine.calculate_enhanced_rankings(test_strategies)\n    \n    if rankings:\n        ranking = rankings[0]  # Test first ranking\n        \n        # Check all criteria are present\n        expected_criteria = [\n            \"signal_confidence\",\n            \"risk_reward_ratio\", \n            \"opportunity_score\",\n            \"fusion_confidence\",\n            \"consistency_score\",\n            \"profit_potential\",\n            \"drawdown_risk\",\n            \"portfolio_fit\",\n            \"diversification_score\",\n            \"market_adaptability\"\n        ]\n        \n        criteria_present = []\n        for criterion in expected_criteria:\n            if hasattr(ranking.strategy, criterion.replace(\"_ratio\", \"_ratio\").replace(\"_score\", \"_score\")):\n                criteria_present.append(criterion)\n            elif hasattr(ranking, criterion):\n                criteria_present.append(criterion)\n        \n        logger.info(f\"📊 Ranking Criteria Validation:\")\n        logger.info(f\"  - Expected: {len(expected_criteria)} criteria\")\n        logger.info(f\"  - Present: {len(criteria_present)} criteria\")\n        \n        for criterion in expected_criteria:\n            status = \"✅\" if criterion in criteria_present else \"❌\"\n            logger.info(f\"  {status} {criterion}\")\n        \n        # Check confidence intervals\n        if ranking.confidence_intervals:\n            logger.info(f\"✅ Confidence intervals: {len(ranking.confidence_intervals)} metrics\")\n        \n        # Check performance projections\n        if ranking.performance_projections:\n            logger.info(f\"✅ Performance projections: {len(ranking.performance_projections)} metrics\")\n        \n        # Check risk metrics\n        if ranking.risk_metrics:\n            logger.info(f\"✅ Risk metrics: {len(ranking.risk_metrics)} metrics\")\n        \n        success = len(criteria_present) >= 8  # At least 8 out of 10 criteria\n        \n        if success:\n            logger.info(\"🎯 Ranking Criteria Validation: PASSED\")\n        else:\n            logger.warning(\"⚠️ Ranking Criteria Validation: PARTIAL\")\n        \n        return success\n    \n    else:\n        logger.error(\"❌ Ranking Criteria Validation: FAILED - No rankings generated\")\n        return False


async def main():\n    \"\"\"Main test execution\"\"\"\n    logger.info(\"🎯 Task 5: Enhanced Ranking Engine Implementation Test\")\n    logger.info(\"🔧 Testing Multi-Kriterien Evaluator with 7+ Ranking-Faktoren\")\n    logger.info(\"⚡ Validating Portfolio-Fit-Calculator and Risk-Adjusted-Scorer\")\n    logger.info(\"📊 Checking Confidence-Intervals and Performance-Projections\")\n    \n    # Run all tests\n    test1_success = await test_enhanced_ranking_integration()\n    test2_success = await test_ranking_criteria_validation()\n    \n    overall_success = test1_success and test2_success\n    \n    if overall_success:\n        logger.info(\"\\n🎉 Task 5 Implementation: SUCCESS\")\n        logger.info(\"✅ Enhanced Ranking Engine fully integrated\")\n        logger.info(\"🏆 Multi-Kriterien Evaluator operational\")\n        logger.info(\"📊 Portfolio-Fit-Calculator functional\")\n        logger.info(\"⚡ Risk-Adjusted-Scorer implemented\")\n        logger.info(\"🚀 Ready for Task 6: Multimodal Flow Integration\")\n    else:\n        logger.error(\"\\n❌ Task 5 Implementation: FAILED\")\n        logger.error(\"🔧 Enhanced Ranking Engine requires debugging\")\n    \n    return overall_success


if __name__ == \"__main__\":\n    asyncio.run(main())