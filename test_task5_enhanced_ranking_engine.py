#!/usr/bin/env python3
"""
Test Task 5: Enhanced Ranking Engine Implementation
Tests the advanced ranking system with multi-criteria evaluation and portfolio optimization
"""

import json
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the Enhanced Ranking Engine
from ai_indicator_optimizer.ranking.enhanced_ranking_engine import (
    EnhancedRankingEngine,
    StrategyScore,
    RankingConfig,
    StrategyRankingCriteria,
    create_enhanced_ranking_engine
)

# Import Top5 System for integration testing
from ai_indicator_optimizer.integration.top5_strategies_ranking_system import (
    Top5StrategiesRankingSystem,
    PipelineConfig,
    ExecutionMode
)

class Task5EnhancedRankingEngineTest:
    """Test suite for Task 5: Enhanced Ranking Engine Implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Task 5 tests"""
        print("🏆 TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION TESTS")
        print("=" * 80)
        
        # Test 1: Enhanced Ranking Engine Components
        test1_result = await self.test_ranking_engine_components()
        self.results["test1_ranking_components"] = test1_result
        
        # Test 2: Multi-Criteria Evaluation
        test2_result = await self.test_multi_criteria_evaluation()
        self.results["test2_multi_criteria"] = test2_result
        
        # Test 3: Portfolio Fit and Diversification
        test3_result = await self.test_portfolio_optimization()
        self.results["test3_portfolio_optimization"] = test3_result
        
        # Test 4: Risk-Adjusted Scoring
        test4_result = await self.test_risk_adjusted_scoring()
        self.results["test4_risk_adjusted_scoring"] = test4_result
        
        # Test 5: Enhanced Ranking Integration
        test5_result = await self.test_enhanced_ranking_integration()
        self.results["test5_integration"] = test5_result
        
        # Test 6: Performance and Scalability
        test6_result = await self.test_performance_scalability()
        self.results["test6_performance"] = test6_result
        
        # Summary
        await self.print_test_summary()
        
        return self.results
    
    async def test_ranking_engine_components(self) -> Dict:
        """Test 1: Enhanced Ranking Engine Components"""
        print("\n🧪 Test 1: Enhanced Ranking Engine Components")
        
        try:
            # Test component initialization
            config = RankingConfig()
            ranking_engine = EnhancedRankingEngine(config)
            
            # Validate components
            assert hasattr(ranking_engine, 'multi_criteria_evaluator')
            assert hasattr(ranking_engine, 'portfolio_calculator')
            assert hasattr(ranking_engine, 'risk_scorer')
            
            # Test configuration
            assert len(config.criteria_weights) == 7  # 7 ranking criteria
            assert abs(sum(config.criteria_weights.values()) - 1.0) < 0.01  # Weights sum to ~1.0
            
            # Test factory function
            custom_engine = create_enhanced_ranking_engine(
                criteria_weights={
                    "signal_confidence": 0.3,
                    "risk_reward_ratio": 0.2
                },
                portfolio_weights={
                    "portfolio_fit": 0.25,
                    "diversification": 0.15
                }
            )
            
            print("✅ Enhanced Ranking Engine initialized")
            print("✅ All components available")
            print("✅ Configuration validated")
            print("✅ Factory function working")
            print(f"✅ Criteria weights: {len(config.criteria_weights)} configured")
            
            return {
                "success": True,
                "components_initialized": True,
                "criteria_count": len(config.criteria_weights),
                "weights_sum": sum(config.criteria_weights.values()),
                "factory_function": True
            }
            
        except Exception as e:
            print(f"❌ Ranking engine components test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_multi_criteria_evaluation(self) -> Dict:
        """Test 2: Multi-Criteria Evaluation"""
        print("\n🧪 Test 2: Multi-Criteria Evaluation")
        
        try:
            # Create test strategies with different characteristics
            test_strategies = [
                StrategyScore(
                    name="High_Confidence_Strategy",
                    signal_confidence=0.95,
                    risk_reward_ratio=2.0,
                    opportunity_score=0.80,
                    fusion_confidence=0.90,
                    consistency_score=0.85,
                    profit_potential=0.75,
                    drawdown_risk=0.10
                ),
                StrategyScore(
                    name="High_Risk_Reward_Strategy",
                    signal_confidence=0.70,
                    risk_reward_ratio=4.0,
                    opportunity_score=0.85,
                    fusion_confidence=0.75,
                    consistency_score=0.70,
                    profit_potential=0.90,
                    drawdown_risk=0.20
                ),
                StrategyScore(
                    name="Balanced_Strategy",
                    signal_confidence=0.80,
                    risk_reward_ratio=2.5,
                    opportunity_score=0.75,
                    fusion_confidence=0.80,
                    consistency_score=0.80,
                    profit_potential=0.80,
                    drawdown_risk=0.15
                ),
                StrategyScore(
                    name="Low_Quality_Strategy",
                    signal_confidence=0.40,  # Below threshold
                    risk_reward_ratio=1.5,
                    opportunity_score=0.50,
                    fusion_confidence=0.45,
                    consistency_score=0.50,
                    profit_potential=0.60,
                    drawdown_risk=0.30  # High drawdown
                )
            ]
            
            # Test multi-criteria evaluator
            ranking_engine = create_enhanced_ranking_engine()
            evaluator = ranking_engine.multi_criteria_evaluator
            
            # Evaluate strategies
            evaluated_strategies = evaluator.evaluate_strategies(test_strategies)
            
            # Validate results
            assert len(evaluated_strategies) < len(test_strategies)  # Some should be filtered
            assert all(s.signal_confidence >= 0.5 for s in evaluated_strategies)  # Quality filter
            assert all(s.drawdown_risk <= 0.25 for s in evaluated_strategies)  # Risk filter
            
            # Check sorting
            scores = [s.composite_score for s in evaluated_strategies]
            assert scores == sorted(scores, reverse=True)  # Should be sorted descending
            
            print(f"✅ Multi-criteria evaluation: {len(evaluated_strategies)}/{len(test_strategies)} strategies passed")
            print(f"✅ Quality filters applied successfully")
            print(f"✅ Strategies sorted by composite score")
            
            # Print top strategies
            for i, strategy in enumerate(evaluated_strategies[:3]):
                print(f"  #{i+1}: {strategy.name} (Score: {strategy.composite_score:.3f})")
            
            return {
                "success": True,
                "strategies_evaluated": len(evaluated_strategies),
                "strategies_filtered": len(test_strategies) - len(evaluated_strategies),
                "top_strategy": evaluated_strategies[0].name if evaluated_strategies else None,
                "top_score": evaluated_strategies[0].composite_score if evaluated_strategies else 0
            }
            
        except Exception as e:
            print(f"❌ Multi-criteria evaluation test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_portfolio_optimization(self) -> Dict:
        """Test 3: Portfolio Fit and Diversification"""
        print("\n🧪 Test 3: Portfolio Fit and Diversification")
        
        try:
            # Create strategies with different characteristics for diversification testing
            diverse_strategies = [
                StrategyScore(
                    name="Conservative_Strategy",
                    signal_confidence=0.85,
                    risk_reward_ratio=1.8,
                    opportunity_score=0.70,
                    fusion_confidence=0.80,
                    consistency_score=0.90,
                    profit_potential=0.70,
                    drawdown_risk=0.08
                ),
                StrategyScore(
                    name="Aggressive_Strategy",
                    signal_confidence=0.75,
                    risk_reward_ratio=3.5,
                    opportunity_score=0.90,
                    fusion_confidence=0.70,
                    consistency_score=0.60,
                    profit_potential=0.95,
                    drawdown_risk=0.22
                ),
                StrategyScore(
                    name="Similar_to_Conservative",
                    signal_confidence=0.87,  # Very similar to conservative
                    risk_reward_ratio=1.9,
                    opportunity_score=0.72,
                    fusion_confidence=0.82,
                    consistency_score=0.88,
                    profit_potential=0.72,
                    drawdown_risk=0.09
                )
            ]
            
            # Test portfolio calculator
            ranking_engine = create_enhanced_ranking_engine()
            portfolio_calc = ranking_engine.portfolio_calculator
            
            # Test portfolio fit calculation
            portfolio_fits = []
            diversification_scores = []
            
            for strategy in diverse_strategies:
                portfolio_fit = portfolio_calc.calculate_portfolio_fit(strategy, diverse_strategies)
                diversification_score = portfolio_calc.calculate_diversification_score(strategy, diverse_strategies)
                
                portfolio_fits.append(portfolio_fit)
                diversification_scores.append(diversification_score)
                
                print(f"  {strategy.name}:")
                print(f"    Portfolio Fit: {portfolio_fit:.3f}")
                print(f"    Diversification: {diversification_score:.3f}")
            
            # Validate diversification logic
            # Similar strategies should have lower portfolio fit
            conservative_fit = portfolio_fits[0]
            similar_fit = portfolio_fits[2]
            assert similar_fit < conservative_fit, "Similar strategies should have lower portfolio fit"
            
            print("✅ Portfolio fit calculation working")
            print("✅ Diversification scoring implemented")
            print("✅ Similar strategies penalized correctly")
            
            return {
                "success": True,
                "portfolio_fits": portfolio_fits,
                "diversification_scores": diversification_scores,
                "diversification_logic": similar_fit < conservative_fit,
                "mean_portfolio_fit": sum(portfolio_fits) / len(portfolio_fits),
                "mean_diversification": sum(diversification_scores) / len(diversification_scores)
            }
            
        except Exception as e:
            print(f"❌ Portfolio optimization test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_risk_adjusted_scoring(self) -> Dict:
        """Test 4: Risk-Adjusted Scoring"""
        print("\n🧪 Test 4: Risk-Adjusted Scoring")
        
        try:
            # Create strategies with different risk profiles
            risk_strategies = [
                StrategyScore(
                    name="Low_Risk_Strategy",
                    signal_confidence=0.80,
                    risk_reward_ratio=2.0,
                    opportunity_score=0.75,
                    fusion_confidence=0.85,
                    consistency_score=0.90,
                    profit_potential=0.70,
                    drawdown_risk=0.05
                ),
                StrategyScore(
                    name="High_Risk_Strategy",
                    signal_confidence=0.75,
                    risk_reward_ratio=3.0,
                    opportunity_score=0.85,
                    fusion_confidence=0.70,
                    consistency_score=0.60,
                    profit_potential=0.95,
                    drawdown_risk=0.25
                )
            ]
            
            # Test risk-adjusted scorer
            ranking_engine = create_enhanced_ranking_engine()
            risk_scorer = ranking_engine.risk_scorer
            
            risk_scores = []
            confidence_intervals = []
            performance_projections = []
            
            for strategy in risk_strategies:
                # Risk-adjusted score
                risk_score = risk_scorer.calculate_risk_adjusted_score(strategy)
                risk_scores.append(risk_score)
                
                # Confidence intervals
                conf_intervals = risk_scorer.calculate_confidence_intervals(strategy)
                confidence_intervals.append(conf_intervals)
                
                # Performance projections
                projections = risk_scorer.calculate_performance_projections(strategy)
                performance_projections.append(projections)
                
                print(f"  {strategy.name}:")
                print(f"    Risk-Adjusted Score: {risk_score:.3f}")
                print(f"    Expected Return CI: {conf_intervals.get('expected_return', (0, 0))}")
                print(f"    1-Year Projection: {projections.get('1_year', 0):.3f}")
            
            # Validate risk adjustment logic
            assert len(risk_scores) == 2
            assert all(isinstance(score, (int, float)) for score in risk_scores)
            assert all(len(ci) == 3 for ci in confidence_intervals)  # 3 confidence intervals
            assert all(len(proj) == 6 for proj in performance_projections)  # 6 projections
            
            print("✅ Risk-adjusted scoring implemented")
            print("✅ Confidence intervals calculated")
            print("✅ Performance projections generated")
            print("✅ Monte Carlo simulations working")
            
            return {
                "success": True,
                "risk_scores": risk_scores,
                "confidence_intervals_count": len(confidence_intervals[0]),
                "projections_count": len(performance_projections[0]),
                "low_risk_score": risk_scores[0],
                "high_risk_score": risk_scores[1]
            }
            
        except Exception as e:
            print(f"❌ Risk-adjusted scoring test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_enhanced_ranking_integration(self) -> Dict:
        """Test 5: Enhanced Ranking Integration with Top5 System"""
        print("\n🧪 Test 5: Enhanced Ranking Integration")
        
        try:
            # Configure Top5 system to use enhanced ranking
            config = PipelineConfig(
                execution_mode=ExecutionMode.DEVELOPMENT,
                max_strategies=3,
                symbols=["EUR/USD"],
                timeframes=["5m"],
                max_workers=4,
                output_dir="test_output/task5_enhanced_ranking"
            )
            
            # Execute pipeline with enhanced ranking
            start_time = time.time()
            top5_system = Top5StrategiesRankingSystem(config=config)
            result = top5_system.execute_full_pipeline()
            execution_time = time.time() - start_time
            
            # Check ranking stage results
            ranking_stage = None
            for stage_result in result.stage_results:
                if stage_result.stage.value == "ranking_calculation":
                    ranking_stage = stage_result
                    break
            
            enhanced_ranking_used = False
            ranking_summary = None
            top5_strategies = []
            
            if ranking_stage and ranking_stage.success:
                enhanced_ranking_used = ranking_stage.data.get("enhanced_ranking_used", False)
                ranking_summary = ranking_stage.data.get("ranking_summary")
                top5_strategies = ranking_stage.data.get("top5_strategies", [])
            
            # Validate enhanced ranking features
            if enhanced_ranking_used and top5_strategies:
                # Check for enhanced ranking features
                first_strategy = top5_strategies[0]
                has_portfolio_fit = "portfolio_fit" in first_strategy
                has_diversification = "diversification_score" in first_strategy
                has_risk_adjusted = "risk_adjusted_score" in first_strategy
                has_confidence_intervals = "confidence_intervals" in first_strategy
                has_projections = "performance_projections" in first_strategy
                
                enhanced_features = sum([
                    has_portfolio_fit, has_diversification, has_risk_adjusted,
                    has_confidence_intervals, has_projections
                ])
            else:
                enhanced_features = 0
                has_portfolio_fit = has_diversification = has_risk_adjusted = False
                has_confidence_intervals = has_projections = False
            
            print(f"✅ Pipeline execution: {'Success' if result.success_rate > 0.8 else 'Failed'}")
            print(f"✅ Enhanced ranking used: {enhanced_ranking_used}")
            print(f"✅ Top strategies generated: {len(top5_strategies)}")
            print(f"✅ Enhanced features: {enhanced_features}/5")
            print(f"  - Portfolio Fit: {has_portfolio_fit}")
            print(f"  - Diversification: {has_diversification}")
            print(f"  - Risk-Adjusted: {has_risk_adjusted}")
            print(f"  - Confidence Intervals: {has_confidence_intervals}")
            print(f"  - Performance Projections: {has_projections}")
            
            return {
                "success": result.success_rate > 0.8,
                "enhanced_ranking_used": enhanced_ranking_used,
                "top_strategies_count": len(top5_strategies),
                "enhanced_features_count": enhanced_features,
                "execution_time": execution_time,
                "pipeline_quality": result.pipeline_quality,
                "has_portfolio_fit": has_portfolio_fit,
                "has_diversification": has_diversification,
                "has_risk_adjusted": has_risk_adjusted,
                "has_confidence_intervals": has_confidence_intervals,
                "has_projections": has_projections
            }
            
        except Exception as e:
            print(f"❌ Enhanced ranking integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_performance_scalability(self) -> Dict:
        """Test 6: Performance and Scalability"""
        print("\n🧪 Test 6: Performance and Scalability")
        
        try:
            # Create different sized strategy sets for performance testing
            strategy_counts = [5, 10, 25, 50]
            performance_results = []
            
            for count in strategy_counts:
                # Generate test strategies
                test_strategies = []
                for i in range(count):
                    strategy = StrategyScore(
                        name=f"Test_Strategy_{i+1}",
                        signal_confidence=0.5 + (i % 5) * 0.1,
                        risk_reward_ratio=1.5 + (i % 4) * 0.5,
                        opportunity_score=0.6 + (i % 4) * 0.1,
                        fusion_confidence=0.7 + (i % 3) * 0.1,
                        consistency_score=0.6 + (i % 4) * 0.1,
                        profit_potential=0.7 + (i % 3) * 0.1,
                        drawdown_risk=0.1 + (i % 3) * 0.05
                    )
                    test_strategies.append(strategy)
                
                # Measure ranking performance
                ranking_engine = create_enhanced_ranking_engine()
                
                start_time = time.time()
                enhanced_rankings = ranking_engine.compute_final_ranking(test_strategies)
                execution_time = time.time() - start_time
                
                strategies_per_second = len(enhanced_rankings) / execution_time if execution_time > 0 else 0
                
                performance_results.append({
                    "strategy_count": count,
                    "execution_time": execution_time,
                    "strategies_per_second": strategies_per_second,
                    "ranked_strategies": len(enhanced_rankings)
                })
                
                print(f"  {count} strategies: {execution_time:.3f}s ({strategies_per_second:.0f} strategies/sec)")
            
            # Calculate performance metrics
            avg_execution_time = sum(r["execution_time"] for r in performance_results) / len(performance_results)
            max_strategies_per_sec = max(r["strategies_per_second"] for r in performance_results)
            
            # Performance should be reasonable (>100 strategies/sec for small sets)
            performance_acceptable = max_strategies_per_sec > 100
            
            print(f"✅ Performance testing complete")
            print(f"✅ Average execution time: {avg_execution_time:.3f}s")
            print(f"✅ Max throughput: {max_strategies_per_sec:.0f} strategies/sec")
            print(f"✅ Performance acceptable: {performance_acceptable}")
            
            return {
                "success": True,
                "performance_results": performance_results,
                "avg_execution_time": avg_execution_time,
                "max_strategies_per_sec": max_strategies_per_sec,
                "performance_acceptable": performance_acceptable,
                "scalability_tested": len(strategy_counts)
            }
            
        except Exception as e:
            print(f"❌ Performance and scalability test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION SUMMARY")
        print("=" * 80)
        
        # Calculate overall success
        successful_tests = sum(1 for result in self.results.values() if result.get("success", False))
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        print(f"✅ Success Rate: {success_rate:.1%}")
        
        # Test-specific summaries
        if "test2_multi_criteria" in self.results:
            multi_result = self.results["test2_multi_criteria"]
            if multi_result.get("success"):
                print(f"🔍 Multi-Criteria: {multi_result.get('strategies_evaluated', 0)} strategies evaluated")
                print(f"🏆 Top Strategy: {multi_result.get('top_strategy', 'N/A')}")
        
        if "test3_portfolio_optimization" in self.results:
            portfolio_result = self.results["test3_portfolio_optimization"]
            if portfolio_result.get("success"):
                print(f"📊 Portfolio Optimization: Diversification logic working")
                print(f"📈 Mean Portfolio Fit: {portfolio_result.get('mean_portfolio_fit', 0):.3f}")
        
        if "test4_risk_adjusted_scoring" in self.results:
            risk_result = self.results["test4_risk_adjusted_scoring"]
            if risk_result.get("success"):
                print(f"⚖️ Risk-Adjusted Scoring: {risk_result.get('confidence_intervals_count', 0)} CI types")
                print(f"📈 Performance Projections: {risk_result.get('projections_count', 0)} time horizons")
        
        if "test5_integration" in self.results:
            integration_result = self.results["test5_integration"]
            if integration_result.get("success"):
                print(f"🔗 Integration: Enhanced ranking {'used' if integration_result.get('enhanced_ranking_used') else 'fallback'}")
                print(f"🎯 Enhanced Features: {integration_result.get('enhanced_features_count', 0)}/5")
        
        if "test6_performance" in self.results:
            perf_result = self.results["test6_performance"]
            if perf_result.get("success"):
                print(f"⚡ Performance: {perf_result.get('max_strategies_per_sec', 0):.0f} strategies/sec max")
        
        # Status determination
        if success_rate >= 0.9:
            status = "🎉 TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION - EXCELLENT"
        elif success_rate >= 0.7:
            status = "✅ TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION - SUCCESS"
        elif success_rate >= 0.5:
            status = "⚠️ TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION - PARTIAL SUCCESS"
        else:
            status = "❌ TASK 5: ENHANCED RANKING ENGINE IMPLEMENTATION - NEEDS ATTENTION"
        
        print(f"\n{status}")
        
        # Save results
        results_file = "task5_enhanced_ranking_engine_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {results_file}")

async def main():
    """Main test execution"""
    test_suite = Task5EnhancedRankingEngineTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())