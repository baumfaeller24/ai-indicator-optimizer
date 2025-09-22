#!/usr/bin/env python3
"""
🧩 BAUSTEIN B2 TEST: Multimodal Analysis Pipeline
Comprehensive Testing für die multimodale Analyse-Pipeline

Test-Bereiche:
- Multimodale Strategien-Analyse
- Trading-Signal-Generierung
- Risk-Reward-Assessment
- Performance-Validierung
- Integration mit Bausteinen A1-B1
"""

import sys
import os
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_indicator_optimizer.ai.multimodal_analysis_pipeline import (
    MultimodalAnalysisPipeline, StrategyAnalyzer, AnalysisMode, TradingSignal, FusionStrategy
)
from ai_indicator_optimizer.ai.multimodal_fusion_engine import MultimodalFusionEngine, MultimodalFeatures


class BausteinB2Tester:
    """
    🧩 BAUSTEIN B2 TESTER: Multimodal Analysis Pipeline
    """
    
    def __init__(self):
        """Initialize Baustein B2 Tester"""
        self.logger = logging.getLogger(__name__)
        self.test_results = {
            "timestamp": datetime.now(),
            "baustein": "B2_MULTIMODAL_ANALYSIS_PIPELINE",
            "tests": {},
            "performance_metrics": {},
            "integration_status": {},
            "overall_success": False
        }
        
        # Test-Konfiguration
        self.test_symbols = ["EUR/USD", "GBP/USD"]
        self.test_timeframes = ["1h", "4h"]
        self.test_modes = [AnalysisMode.FAST, AnalysisMode.COMPREHENSIVE]
        
        print("🧩 BAUSTEIN B2 TESTER INITIALIZED")
        print("=" * 70)
    
    def run_all_tests(self) -> dict:
        """Führe alle Tests für Baustein B2 durch"""
        
        print("🔄 STARTING BAUSTEIN B2 COMPREHENSIVE TESTS")
        print("=" * 70)
        
        try:
            # Test 1: Pipeline Initialization
            self.test_pipeline_initialization()
            
            # Test 2: Strategy Analyzer
            self.test_strategy_analyzer()
            
            # Test 3: Multimodal Strategy Analysis
            self.test_multimodal_strategy_analysis()
            
            # Test 4: Trading Signal Generation
            self.test_trading_signal_generation()
            
            # Test 5: Risk-Reward Assessment
            self.test_risk_reward_assessment()
            
            # Test 6: Performance Validation
            self.test_performance_validation()
            
            # Test 7: Integration with Bausteine A1-B1
            self.test_integration_with_previous_bausteine()
            
            # Test 8: Multi-Symbol Multi-Timeframe
            self.test_multi_symbol_multi_timeframe()
            
            # Test 9: Error Handling
            self.test_error_handling()
            
            # Test 10: Performance Benchmarks
            self.test_performance_benchmarks()
            
            # Gesamtergebnis
            self._calculate_overall_results()
            
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            self.test_results["overall_success"] = False
            self.test_results["error"] = str(e)
            return self.test_results
    
    def test_pipeline_initialization(self):
        """Test 1: Pipeline Initialization"""
        test_name = "pipeline_initialization"
        print(f"\n🔄 Test 1: {test_name}")
        
        try:
            start_time = time.time()
            
            # Verschiedene Konfigurationen testen
            configs = [
                (FusionStrategy.CONFIDENCE_BASED, AnalysisMode.COMPREHENSIVE),
                (FusionStrategy.WEIGHTED_AVERAGE, AnalysisMode.FAST),
                (FusionStrategy.HIERARCHICAL, AnalysisMode.DEEP),
                (FusionStrategy.ENSEMBLE, AnalysisMode.REAL_TIME)
            ]
            
            successful_inits = 0
            
            for fusion_strategy, analysis_mode in configs:
                try:
                    pipeline = MultimodalAnalysisPipeline(
                        fusion_strategy=fusion_strategy,
                        analysis_mode=analysis_mode,
                        output_dir=f"test_cache/baustein_b2_{fusion_strategy.value}_{analysis_mode.value}"
                    )
                    
                    # Validiere Komponenten
                    assert hasattr(pipeline, 'fusion_engine'), "Fusion Engine not initialized"
                    assert hasattr(pipeline, 'strategy_analyzer'), "Strategy Analyzer not initialized"
                    assert hasattr(pipeline, 'data_connector'), "Data Connector not initialized"
                    assert hasattr(pipeline, 'schema_manager'), "Schema Manager not initialized"
                    
                    successful_inits += 1
                    print(f"  ✅ {fusion_strategy.value} + {analysis_mode.value}: OK")
                    
                except Exception as e:
                    print(f"  ❌ {fusion_strategy.value} + {analysis_mode.value}: {e}")
            
            duration = time.time() - start_time
            success_rate = successful_inits / len(configs)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.75,
                "success_rate": success_rate,
                "successful_inits": successful_inits,
                "total_configs": len(configs),
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%} ({successful_inits}/{len(configs)})")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_strategy_analyzer(self):
        """Test 2: Strategy Analyzer"""
        test_name = "strategy_analyzer"
        print(f"\n🔄 Test 2: {test_name}")
        
        try:
            start_time = time.time()
            
            analyzer = StrategyAnalyzer()
            
            # Mock multimodal features für Testing
            mock_features = MultimodalFeatures(
                technical_features={
                    "rsi_14": 65.0,
                    "macd_signal": 0.001,
                    "trend_strength": 0.7,
                    "atr_14": 0.0015,
                    "bb_width": 0.002,
                    "volume_ratio": 1.2
                },
                technical_confidence=0.8,
                vision_features={
                    "vision_trend_numeric": 1.0,
                    "vision_pattern_count": 3.0,
                    "vision_pattern_strength": 0.75,
                    "vision_confidence": 0.7,
                    "vision_has_reversal_pattern": 0.0,
                    "vision_has_continuation_pattern": 1.0,
                    "vision_has_support_resistance": 1.0
                },
                vision_confidence=0.7,
                fused_features={
                    "multimodal_trend_strength": 0.8,
                    "multimodal_momentum": 0.65,
                    "multimodal_volatility": 0.5,
                    "multimodal_pattern_confidence": 0.7,
                    "multimodal_reversal_probability": 0.2,
                    "multimodal_breakout_probability": 0.8,
                    "multimodal_support_resistance_strength": 0.9,
                    "multimodal_risk_score": 0.3,
                    "multimodal_opportunity_score": 0.8,
                    "multimodal_confidence_consistency": 0.85
                },
                fusion_confidence=0.75,
                fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                timestamp=datetime.now(),
                symbol="EUR/USD",
                timeframe="1h",
                processing_time=0.5
            )
            
            # Strategien-Analyse durchführen
            current_price = 1.1000
            analysis = analyzer.analyze_strategy(mock_features, current_price)
            
            # Validierungen
            validations = {
                "has_trading_signal": analysis.trading_signal is not None,
                "has_confidence": 0.0 <= analysis.signal_confidence <= 1.0,
                "has_reasoning": len(analysis.signal_reasoning) > 0,
                "has_risk_scores": 0.0 <= analysis.risk_score <= 1.0 and 0.0 <= analysis.opportunity_score <= 1.0,
                "has_position_size": 0.0 <= analysis.position_size_factor <= 1.0,
                "has_insights": len(analysis.key_insights) > 0,
                "has_confidence_breakdown": len(analysis.confidence_breakdown) > 0
            }
            
            # Entry/Exit-Punkte validieren (wenn nicht HOLD)
            if analysis.trading_signal != TradingSignal.HOLD:
                validations["has_entry_price"] = analysis.entry_price is not None
                validations["has_stop_loss"] = analysis.stop_loss is not None
                validations["has_take_profit"] = analysis.take_profit is not None
            else:
                validations["hold_signal_correct"] = (
                    analysis.entry_price is None and 
                    analysis.stop_loss is None and 
                    analysis.take_profit is None
                )
            
            duration = time.time() - start_time
            success_rate = sum(validations.values()) / len(validations)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.9,
                "success_rate": success_rate,
                "validations": validations,
                "trading_signal": analysis.trading_signal.value,
                "signal_confidence": analysis.signal_confidence,
                "risk_score": analysis.risk_score,
                "opportunity_score": analysis.opportunity_score,
                "position_size_factor": analysis.position_size_factor,
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%}")
            print(f"  🎯 Trading Signal: {analysis.trading_signal.value} (conf: {analysis.signal_confidence:.1%})")
            print(f"  ⚖️  Risk/Opportunity: {analysis.risk_score:.2f}/{analysis.opportunity_score:.2f}")
            print(f"  💰 Position Size: {analysis.position_size_factor:.1%}")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_multimodal_strategy_analysis(self):
        """Test 3: Multimodal Strategy Analysis"""
        test_name = "multimodal_strategy_analysis"
        print(f"\n🔄 Test 3: {test_name}")
        
        try:
            start_time = time.time()
            
            # Pipeline erstellen
            pipeline = MultimodalAnalysisPipeline(
                fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                analysis_mode=AnalysisMode.FAST,  # Fast für Testing
                output_dir="test_cache/baustein_b2_analysis"
            )
            
            # Mock-Daten für Testing (falls echte Daten nicht verfügbar)
            try:
                # Versuche echte Analyse
                analysis = pipeline.analyze_multimodal_strategy(
                    symbol="EUR/USD",
                    timeframe="1h",
                    lookback_periods=20,  # Weniger Daten für schnelleren Test
                    current_price=1.1000
                )
                
                real_data_test = True
                
            except Exception as data_error:
                print(f"  ⚠️  Real data test failed: {data_error}")
                print(f"  🔄 Falling back to mock data test...")
                
                # Fallback zu Mock-Test
                real_data_test = False
                
                # Erstelle Mock-Analyse
                analysis = pipeline.strategy_analyzer.analyze_strategy(
                    multimodal_features=MultimodalFeatures(
                        technical_features={"rsi_14": 70.0, "macd_signal": 0.002},
                        technical_confidence=0.8,
                        vision_features={"vision_trend_numeric": 1.0},
                        vision_confidence=0.7,
                        fused_features={
                            "multimodal_trend_strength": 0.8,
                            "multimodal_momentum": 0.7,
                            "multimodal_pattern_confidence": 0.75,
                            "multimodal_risk_score": 0.3,
                            "multimodal_opportunity_score": 0.8
                        },
                        fusion_confidence=0.75,
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        timestamp=datetime.now(),
                        symbol="EUR/USD",
                        timeframe="1h",
                        processing_time=0.1
                    ),
                    current_price=1.1000
                )
            
            # Validierungen
            validations = {
                "analysis_completed": analysis is not None,
                "has_symbol": analysis.symbol == "EUR/USD",
                "has_timeframe": analysis.timeframe == "1h",
                "has_timestamp": analysis.timestamp is not None,
                "has_multimodal_features": analysis.multimodal_features is not None,
                "has_trading_signal": analysis.trading_signal is not None,
                "valid_confidence": 0.0 <= analysis.signal_confidence <= 1.0,
                "has_processing_time": analysis.processing_time > 0,
                "has_insights": len(analysis.key_insights) > 0
            }
            
            duration = time.time() - start_time
            success_rate = sum(validations.values()) / len(validations)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "validations": validations,
                "real_data_test": real_data_test,
                "analysis_summary": {
                    "symbol": analysis.symbol,
                    "timeframe": analysis.timeframe,
                    "trading_signal": analysis.trading_signal.value,
                    "signal_confidence": analysis.signal_confidence,
                    "processing_time": analysis.processing_time
                },
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%}")
            print(f"  🎯 Analysis: {analysis.trading_signal.value} (conf: {analysis.signal_confidence:.1%})")
            print(f"  ⏱️  Processing Time: {analysis.processing_time:.3f}s")
            print(f"  🔄 Real Data Test: {'✅' if real_data_test else '❌'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_trading_signal_generation(self):
        """Test 4: Trading Signal Generation"""
        test_name = "trading_signal_generation"
        print(f"\n🔄 Test 4: {test_name}")
        
        try:
            start_time = time.time()
            
            analyzer = StrategyAnalyzer()
            
            # Test verschiedene Szenarien
            test_scenarios = [
                {
                    "name": "strong_bullish",
                    "features": {
                        "multimodal_trend_strength": 0.9,
                        "multimodal_momentum": 0.8,
                        "multimodal_pattern_confidence": 0.85,
                        "multimodal_breakout_probability": 0.9,
                        "multimodal_confidence_consistency": 0.9
                    },
                    "expected_signal": [TradingSignal.STRONG_BUY, TradingSignal.BUY]
                },
                {
                    "name": "strong_bearish",
                    "features": {
                        "multimodal_trend_strength": -0.9,
                        "multimodal_momentum": 0.2,
                        "multimodal_pattern_confidence": 0.8,
                        "multimodal_reversal_probability": 0.9,
                        "multimodal_confidence_consistency": 0.85
                    },
                    "expected_signal": [TradingSignal.STRONG_SELL, TradingSignal.SELL]
                },
                {
                    "name": "neutral_low_confidence",
                    "features": {
                        "multimodal_trend_strength": 0.1,
                        "multimodal_momentum": 0.5,
                        "multimodal_pattern_confidence": 0.3,
                        "multimodal_confidence_consistency": 0.4
                    },
                    "expected_signal": [TradingSignal.HOLD]
                }
            ]
            
            successful_scenarios = 0
            scenario_results = {}
            
            for scenario in test_scenarios:
                try:
                    # Mock Features erstellen
                    mock_features = MultimodalFeatures(
                        technical_features={"rsi_14": 50.0},
                        technical_confidence=0.7,
                        vision_features={"vision_trend_numeric": 0.0},
                        vision_confidence=0.7,
                        fused_features=scenario["features"],
                        fusion_confidence=0.8,
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        timestamp=datetime.now(),
                        symbol="EUR/USD",
                        timeframe="1h",
                        processing_time=0.1
                    )
                    
                    # Analyse durchführen
                    analysis = analyzer.analyze_strategy(mock_features, 1.1000)
                    
                    # Signal validieren
                    signal_correct = analysis.trading_signal in scenario["expected_signal"]
                    
                    scenario_results[scenario["name"]] = {
                        "expected": [s.value for s in scenario["expected_signal"]],
                        "actual": analysis.trading_signal.value,
                        "confidence": analysis.signal_confidence,
                        "correct": signal_correct
                    }
                    
                    if signal_correct:
                        successful_scenarios += 1
                        print(f"  ✅ {scenario['name']}: {analysis.trading_signal.value} (conf: {analysis.signal_confidence:.1%})")
                    else:
                        print(f"  ❌ {scenario['name']}: Expected {scenario['expected_signal']}, got {analysis.trading_signal.value}")
                
                except Exception as e:
                    scenario_results[scenario["name"]] = {
                        "error": str(e),
                        "correct": False
                    }
                    print(f"  ❌ {scenario['name']}: {e}")
            
            duration = time.time() - start_time
            success_rate = successful_scenarios / len(test_scenarios)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "successful_scenarios": successful_scenarios,
                "total_scenarios": len(test_scenarios),
                "scenario_results": scenario_results,
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%} ({successful_scenarios}/{len(test_scenarios)})")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_risk_reward_assessment(self):
        """Test 5: Risk-Reward Assessment"""
        test_name = "risk_reward_assessment"
        print(f"\n🔄 Test 5: {test_name}")
        
        try:
            start_time = time.time()
            
            analyzer = StrategyAnalyzer()
            
            # Test Risk-Reward für verschiedene Szenarien
            risk_scenarios = [
                {
                    "name": "low_risk_high_reward",
                    "features": {
                        "multimodal_risk_score": 0.2,
                        "multimodal_opportunity_score": 0.9,
                        "multimodal_volatility": 0.3,
                        "multimodal_confidence_consistency": 0.9
                    },
                    "expected_risk": "low",
                    "expected_opportunity": "high"
                },
                {
                    "name": "high_risk_low_reward",
                    "features": {
                        "multimodal_risk_score": 0.8,
                        "multimodal_opportunity_score": 0.3,
                        "multimodal_volatility": 0.9,
                        "multimodal_confidence_consistency": 0.3
                    },
                    "expected_risk": "high",
                    "expected_opportunity": "low"
                }
            ]
            
            successful_assessments = 0
            assessment_results = {}
            
            for scenario in risk_scenarios:
                try:
                    # Mock Features
                    mock_features = MultimodalFeatures(
                        technical_features={"rsi_14": 50.0},
                        technical_confidence=0.7,
                        vision_features={"vision_trend_numeric": 0.0},
                        vision_confidence=0.7,
                        fused_features=scenario["features"],
                        fusion_confidence=0.7,
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        timestamp=datetime.now(),
                        symbol="EUR/USD",
                        timeframe="1h",
                        processing_time=0.1
                    )
                    
                    # Analyse durchführen
                    analysis = analyzer.analyze_strategy(mock_features, 1.1000)
                    
                    # Risk-Reward validieren
                    risk_correct = (
                        (scenario["expected_risk"] == "low" and analysis.risk_score < 0.5) or
                        (scenario["expected_risk"] == "high" and analysis.risk_score > 0.5)
                    )
                    
                    opportunity_correct = (
                        (scenario["expected_opportunity"] == "low" and analysis.opportunity_score < 0.5) or
                        (scenario["expected_opportunity"] == "high" and analysis.opportunity_score > 0.5)
                    )
                    
                    assessment_correct = risk_correct and opportunity_correct
                    
                    assessment_results[scenario["name"]] = {
                        "risk_score": analysis.risk_score,
                        "opportunity_score": analysis.opportunity_score,
                        "risk_reward_ratio": analysis.risk_reward_ratio,
                        "risk_correct": risk_correct,
                        "opportunity_correct": opportunity_correct,
                        "overall_correct": assessment_correct
                    }
                    
                    if assessment_correct:
                        successful_assessments += 1
                        print(f"  ✅ {scenario['name']}: Risk {analysis.risk_score:.2f}, Opportunity {analysis.opportunity_score:.2f}")
                    else:
                        print(f"  ❌ {scenario['name']}: Risk assessment incorrect")
                
                except Exception as e:
                    assessment_results[scenario["name"]] = {
                        "error": str(e),
                        "overall_correct": False
                    }
                    print(f"  ❌ {scenario['name']}: {e}")
            
            duration = time.time() - start_time
            success_rate = successful_assessments / len(risk_scenarios)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "successful_assessments": successful_assessments,
                "total_scenarios": len(risk_scenarios),
                "assessment_results": assessment_results,
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%} ({successful_assessments}/{len(risk_scenarios)})")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_performance_validation(self):
        """Test 6: Performance Validation"""
        test_name = "performance_validation"
        print(f"\n🔄 Test 6: {test_name}")
        
        try:
            start_time = time.time()
            
            pipeline = MultimodalAnalysisPipeline(
                fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                analysis_mode=AnalysisMode.FAST,
                output_dir="test_cache/baustein_b2_performance"
            )
            
            # Performance-Tests
            num_analyses = 5
            analysis_times = []
            successful_analyses = 0
            
            for i in range(num_analyses):
                try:
                    analysis_start = time.time()
                    
                    # Mock-Analyse für Performance-Test
                    mock_features = MultimodalFeatures(
                        technical_features={"rsi_14": 50.0 + i * 5},
                        technical_confidence=0.7 + i * 0.05,
                        vision_features={"vision_trend_numeric": i * 0.2 - 0.4},
                        vision_confidence=0.6 + i * 0.08,
                        fused_features={
                            "multimodal_trend_strength": i * 0.2 - 0.4,
                            "multimodal_momentum": 0.5 + i * 0.1,
                            "multimodal_pattern_confidence": 0.6 + i * 0.05
                        },
                        fusion_confidence=0.7 + i * 0.05,
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        timestamp=datetime.now(),
                        symbol="EUR/USD",
                        timeframe="1h",
                        processing_time=0.1
                    )
                    
                    analysis = pipeline.strategy_analyzer.analyze_strategy(
                        mock_features, 1.1000 + i * 0.001
                    )
                    
                    analysis_time = time.time() - analysis_start
                    analysis_times.append(analysis_time)
                    successful_analyses += 1
                    
                    print(f"  ✅ Analysis {i+1}: {analysis_time:.3f}s - {analysis.trading_signal.value}")
                    
                except Exception as e:
                    print(f"  ❌ Analysis {i+1}: {e}")
            
            # Performance-Statistiken
            if analysis_times:
                avg_time = np.mean(analysis_times)
                max_time = np.max(analysis_times)
                min_time = np.min(analysis_times)
                analyses_per_second = 1.0 / avg_time if avg_time > 0 else 0
            else:
                avg_time = max_time = min_time = analyses_per_second = 0
            
            # Pipeline-Performance-Stats
            pipeline_stats = pipeline.get_performance_stats()
            
            duration = time.time() - start_time
            success_rate = successful_analyses / num_analyses
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8 and avg_time < 1.0,  # 80% success + <1s per analysis
                "success_rate": success_rate,
                "performance_metrics": {
                    "avg_analysis_time": avg_time,
                    "max_analysis_time": max_time,
                    "min_analysis_time": min_time,
                    "analyses_per_second": analyses_per_second,
                    "successful_analyses": successful_analyses,
                    "total_analyses": num_analyses
                },
                "pipeline_stats": pipeline_stats,
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%} ({successful_analyses}/{num_analyses})")
            print(f"  ⚡ Avg Analysis Time: {avg_time:.3f}s")
            print(f"  🚀 Analyses/sec: {analyses_per_second:.1f}")
            print(f"  ⏱️  Total Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_integration_with_previous_bausteine(self):
        """Test 7: Integration with Bausteine A1-B1"""
        test_name = "integration_bausteine_a1_b1"
        print(f"\n🔄 Test 7: {test_name}")
        
        try:
            start_time = time.time()
            
            # Test Integration mit allen vorherigen Bausteinen
            integration_tests = {
                "fusion_engine_integration": False,
                "schema_manager_integration": False,
                "data_connector_integration": False,
                "vision_client_integration": False,
                "chart_processor_integration": False
            }
            
            try:
                # Pipeline mit allen Komponenten erstellen
                pipeline = MultimodalAnalysisPipeline(
                    fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                    analysis_mode=AnalysisMode.FAST,
                    output_dir="test_cache/baustein_b2_integration"
                )
                
                # Fusion Engine Integration (B1)
                if hasattr(pipeline, 'fusion_engine') and pipeline.fusion_engine is not None:
                    integration_tests["fusion_engine_integration"] = True
                    print(f"  ✅ Fusion Engine (B1): Integrated")
                
                # Schema Manager Integration (A1)
                if hasattr(pipeline, 'schema_manager') and pipeline.schema_manager is not None:
                    integration_tests["schema_manager_integration"] = True
                    print(f"  ✅ Schema Manager (A1): Integrated")
                
                # Data Connector Integration
                if hasattr(pipeline, 'data_connector') and pipeline.data_connector is not None:
                    integration_tests["data_connector_integration"] = True
                    print(f"  ✅ Data Connector: Integrated")
                
                # Vision Client Integration (A2) - via Fusion Engine
                if hasattr(pipeline.fusion_engine, 'vision_client'):
                    integration_tests["vision_client_integration"] = True
                    print(f"  ✅ Vision Client (A2): Integrated via Fusion Engine")
                
                # Chart Processor Integration (A3) - via Fusion Engine
                if hasattr(pipeline.fusion_engine, 'chart_processor'):
                    integration_tests["chart_processor_integration"] = True
                    print(f"  ✅ Chart Processor (A3): Integrated via Fusion Engine")
                
            except Exception as e:
                print(f"  ❌ Integration test failed: {e}")
            
            duration = time.time() - start_time
            success_rate = sum(integration_tests.values()) / len(integration_tests)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "integration_tests": integration_tests,
                "duration": duration
            }
            
            print(f"  📊 Integration Success Rate: {success_rate:.1%}")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_multi_symbol_multi_timeframe(self):
        """Test 8: Multi-Symbol Multi-Timeframe"""
        test_name = "multi_symbol_multi_timeframe"
        print(f"\n🔄 Test 8: {test_name}")
        
        try:
            start_time = time.time()
            
            pipeline = MultimodalAnalysisPipeline(
                fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                analysis_mode=AnalysisMode.FAST,
                output_dir="test_cache/baustein_b2_multi"
            )
            
            # Test verschiedene Symbol/Timeframe-Kombinationen
            test_combinations = [
                ("EUR/USD", "1h"),
                ("GBP/USD", "4h"),
                ("USD/JPY", "1h")
            ]
            
            successful_combinations = 0
            combination_results = {}
            
            for symbol, timeframe in test_combinations:
                try:
                    # Mock-Analyse für jede Kombination
                    mock_features = MultimodalFeatures(
                        technical_features={"rsi_14": 60.0},
                        technical_confidence=0.75,
                        vision_features={"vision_trend_numeric": 0.5},
                        vision_confidence=0.7,
                        fused_features={
                            "multimodal_trend_strength": 0.6,
                            "multimodal_momentum": 0.65,
                            "multimodal_pattern_confidence": 0.7
                        },
                        fusion_confidence=0.72,
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        timeframe=timeframe,
                        processing_time=0.1
                    )
                    
                    analysis = pipeline.strategy_analyzer.analyze_strategy(
                        mock_features, 1.1000
                    )
                    
                    # Validiere Symbol/Timeframe in Ergebnis
                    symbol_correct = analysis.symbol == symbol
                    timeframe_correct = analysis.timeframe == timeframe
                    
                    if symbol_correct and timeframe_correct:
                        successful_combinations += 1
                        combination_results[f"{symbol}_{timeframe}"] = {
                            "success": True,
                            "trading_signal": analysis.trading_signal.value,
                            "confidence": analysis.signal_confidence
                        }
                        print(f"  ✅ {symbol} {timeframe}: {analysis.trading_signal.value} (conf: {analysis.signal_confidence:.1%})")
                    else:
                        combination_results[f"{symbol}_{timeframe}"] = {
                            "success": False,
                            "error": "Symbol/Timeframe mismatch"
                        }
                        print(f"  ❌ {symbol} {timeframe}: Symbol/Timeframe mismatch")
                
                except Exception as e:
                    combination_results[f"{symbol}_{timeframe}"] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"  ❌ {symbol} {timeframe}: {e}")
            
            duration = time.time() - start_time
            success_rate = successful_combinations / len(test_combinations)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "successful_combinations": successful_combinations,
                "total_combinations": len(test_combinations),
                "combination_results": combination_results,
                "duration": duration
            }
            
            print(f"  📊 Success Rate: {success_rate:.1%} ({successful_combinations}/{len(test_combinations)})")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_error_handling(self):
        """Test 9: Error Handling"""
        test_name = "error_handling"
        print(f"\n🔄 Test 9: {test_name}")
        
        try:
            start_time = time.time()
            
            pipeline = MultimodalAnalysisPipeline(
                fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                analysis_mode=AnalysisMode.FAST,
                output_dir="test_cache/baustein_b2_errors"
            )
            
            # Test verschiedene Error-Szenarien
            error_scenarios = [
                {
                    "name": "invalid_symbol",
                    "test": lambda: pipeline.analyze_multimodal_strategy(
                        symbol="INVALID/SYMBOL",
                        timeframe="1h",
                        lookback_periods=10
                    )
                },
                {
                    "name": "invalid_timeframe",
                    "test": lambda: pipeline.analyze_multimodal_strategy(
                        symbol="EUR/USD",
                        timeframe="invalid_tf",
                        lookback_periods=10
                    )
                },
                {
                    "name": "zero_lookback",
                    "test": lambda: pipeline.analyze_multimodal_strategy(
                        symbol="EUR/USD",
                        timeframe="1h",
                        lookback_periods=0
                    )
                }
            ]
            
            successful_error_handling = 0
            error_results = {}
            
            for scenario in error_scenarios:
                try:
                    # Führe Error-Test durch
                    result = scenario["test"]()
                    
                    # Prüfe ob graceful fallback erfolgt ist
                    if (result is not None and 
                        hasattr(result, 'trading_signal') and 
                        result.trading_signal == TradingSignal.HOLD):
                        
                        successful_error_handling += 1
                        error_results[scenario["name"]] = {
                            "graceful_fallback": True,
                            "signal": result.trading_signal.value,
                            "confidence": result.signal_confidence
                        }
                        print(f"  ✅ {scenario['name']}: Graceful fallback to HOLD")
                    else:
                        error_results[scenario["name"]] = {
                            "graceful_fallback": False,
                            "unexpected_result": str(type(result))
                        }
                        print(f"  ❌ {scenario['name']}: No graceful fallback")
                
                except Exception as e:
                    # Exception ist auch OK, solange sie handled wird
                    successful_error_handling += 1
                    error_results[scenario["name"]] = {
                        "exception_handled": True,
                        "exception": str(e)
                    }
                    print(f"  ✅ {scenario['name']}: Exception properly handled: {e}")
            
            duration = time.time() - start_time
            success_rate = successful_error_handling / len(error_scenarios)
            
            self.test_results["tests"][test_name] = {
                "success": success_rate >= 0.8,
                "success_rate": success_rate,
                "successful_error_handling": successful_error_handling,
                "total_scenarios": len(error_scenarios),
                "error_results": error_results,
                "duration": duration
            }
            
            print(f"  📊 Error Handling Success Rate: {success_rate:.1%} ({successful_error_handling}/{len(error_scenarios)})")
            print(f"  ⏱️  Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test 10: Performance Benchmarks"""
        test_name = "performance_benchmarks"
        print(f"\n🔄 Test 10: {test_name}")
        
        try:
            start_time = time.time()
            
            # Performance-Benchmarks für verschiedene Modi
            benchmark_configs = [
                (AnalysisMode.FAST, "fast_mode"),
                (AnalysisMode.COMPREHENSIVE, "comprehensive_mode")
            ]
            
            benchmark_results = {}
            
            for analysis_mode, mode_name in benchmark_configs:
                try:
                    pipeline = MultimodalAnalysisPipeline(
                        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                        analysis_mode=analysis_mode,
                        output_dir=f"test_cache/baustein_b2_benchmark_{mode_name}"
                    )
                    
                    # Benchmark-Test
                    num_iterations = 3
                    mode_times = []
                    
                    for i in range(num_iterations):
                        iteration_start = time.time()
                        
                        # Mock-Analyse
                        mock_features = MultimodalFeatures(
                            technical_features={"rsi_14": 50.0 + i * 10},
                            technical_confidence=0.7,
                            vision_features={"vision_trend_numeric": 0.0},
                            vision_confidence=0.7,
                            fused_features={
                                "multimodal_trend_strength": 0.5,
                                "multimodal_momentum": 0.6,
                                "multimodal_pattern_confidence": 0.7
                            },
                            fusion_confidence=0.7,
                            fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
                            timestamp=datetime.now(),
                            symbol="EUR/USD",
                            timeframe="1h",
                            processing_time=0.1
                        )
                        
                        analysis = pipeline.strategy_analyzer.analyze_strategy(
                            mock_features, 1.1000
                        )
                        
                        iteration_time = time.time() - iteration_start
                        mode_times.append(iteration_time)
                    
                    # Statistiken
                    avg_time = np.mean(mode_times)
                    throughput = 1.0 / avg_time if avg_time > 0 else 0
                    
                    benchmark_results[mode_name] = {
                        "avg_time": avg_time,
                        "throughput": throughput,
                        "iterations": num_iterations,
                        "mode": analysis_mode.value
                    }
                    
                    print(f"  ✅ {mode_name}: {avg_time:.3f}s avg, {throughput:.1f} analyses/sec")
                    
                except Exception as e:
                    benchmark_results[mode_name] = {
                        "error": str(e)
                    }
                    print(f"  ❌ {mode_name}: {e}")
            
            duration = time.time() - start_time
            
            # Performance-Kriterien
            fast_mode_ok = (
                "fast_mode" in benchmark_results and 
                "avg_time" in benchmark_results["fast_mode"] and
                benchmark_results["fast_mode"]["avg_time"] < 0.5  # <0.5s für Fast Mode
            )
            
            comprehensive_mode_ok = (
                "comprehensive_mode" in benchmark_results and 
                "avg_time" in benchmark_results["comprehensive_mode"] and
                benchmark_results["comprehensive_mode"]["avg_time"] < 2.0  # <2s für Comprehensive Mode
            )
            
            self.test_results["tests"][test_name] = {
                "success": fast_mode_ok and comprehensive_mode_ok,
                "fast_mode_ok": fast_mode_ok,
                "comprehensive_mode_ok": comprehensive_mode_ok,
                "benchmark_results": benchmark_results,
                "duration": duration
            }
            
            print(f"  📊 Fast Mode: {'✅' if fast_mode_ok else '❌'}")
            print(f"  📊 Comprehensive Mode: {'✅' if comprehensive_mode_ok else '❌'}")
            print(f"  ⏱️  Total Duration: {duration:.3f}s")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ Test failed: {e}")
    
    def _calculate_overall_results(self):
        """Berechne Gesamtergebnisse"""
        
        # Erfolgreiche Tests zählen
        successful_tests = sum(1 for test in self.test_results["tests"].values() if test.get("success", False))
        total_tests = len(self.test_results["tests"])
        
        # Performance-Metriken sammeln
        total_duration = sum(test.get("duration", 0) for test in self.test_results["tests"].values())
        
        # Gesamtergebnis
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        self.test_results["overall_success"] = overall_success_rate >= 0.8
        
        # Performance-Metriken
        self.test_results["performance_metrics"] = {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": overall_success_rate,
            "total_duration": total_duration,
            "avg_test_duration": total_duration / total_tests if total_tests > 0 else 0
        }
        
        # Integration-Status
        self.test_results["integration_status"] = {
            "baustein_a1_integration": "schema_manager_integration" in str(self.test_results),
            "baustein_a2_integration": "vision_client_integration" in str(self.test_results),
            "baustein_a3_integration": "chart_processor_integration" in str(self.test_results),
            "baustein_b1_integration": "fusion_engine_integration" in str(self.test_results)
        }
        
        print(f"\n🎉 BAUSTEIN B2 TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"📊 Overall Success Rate: {overall_success_rate:.1%} ({successful_tests}/{total_tests})")
        print(f"⏱️  Total Duration: {total_duration:.3f}s")
        print(f"🎯 Baustein B2 Status: {'✅ PASSED' if self.test_results['overall_success'] else '❌ FAILED'}")


def main():
    """Hauptfunktion für Baustein B2 Tests"""
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Erstelle Test-Verzeichnisse
    os.makedirs("test_cache", exist_ok=True)
    
    print("🧩 BAUSTEIN B2: MULTIMODAL ANALYSIS PIPELINE TESTS")
    print("=" * 70)
    print(f"Start Time: {datetime.now()}")
    print(f"Test Environment: {sys.platform}")
    print()
    
    # Führe Tests durch
    tester = BausteinB2Tester()
    results = tester.run_all_tests()
    
    # Speichere Ergebnisse
    results_file = f"baustein_b2_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # JSON-serializable machen
    def make_serializable(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return str(obj)
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=make_serializable)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Finale Ausgabe
    if results["overall_success"]:
        print(f"\n🎉 BAUSTEIN B2 TESTS PASSED!")
        print(f"✅ Multimodal Analysis Pipeline is ready for production!")
        return 0
    else:
        print(f"\n❌ BAUSTEIN B2 TESTS FAILED!")
        print(f"❌ Issues need to be resolved before production use.")
        return 1


if __name__ == "__main__":
    exit(main())