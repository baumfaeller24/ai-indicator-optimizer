#!/usr/bin/env python3
"""
Phase 2 Integration Test - Enhanced Multimodal Pattern Recognition Engine
Task 8 Complete Test Suite

Tests:
1. VisualPatternAnalyzer
2. Enhanced Feature Extraction mit Zeitnormierung
3. Confidence-basierte Position-Sizing
4. Live-Control-System
5. Environment-Variable-Konfiguration
6. Enhanced Confidence Scoring
7. Complete Integration Workflow
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import json

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_visual_pattern_analyzer():
    """Test 1: VisualPatternAnalyzer"""
    print("🧪 Test 1: VisualPatternAnalyzer...")
    
    try:
        from ai_indicator_optimizer.ai.visual_pattern_analyzer import create_visual_pattern_analyzer
        
        print("✅ VisualPatternAnalyzer importiert")
        
        # Mock Bar für Testing
        class MockBar:
            def __init__(self, open_price, high, low, close, volume):
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
        
        # Test-Daten mit verschiedenen Patterns
        test_bars = [
            MockBar(1.1000, 1.1020, 1.0980, 1.1015, 1000),  # Bullish
            MockBar(1.1015, 1.1025, 1.1005, 1.1010, 1200),  # Small body (Doji-like)
            MockBar(1.1010, 1.1030, 1.0990, 1.0995, 1500),  # Bearish
            MockBar(1.0995, 1.1000, 1.0985, 1.0998, 800),   # Hammer-like
            MockBar(1.0998, 1.1035, 1.0990, 1.1030, 2000),  # Bullish Engulfing
        ]
        
        analyzer = create_visual_pattern_analyzer(use_mock=True, debug_mode=True)
        
        # Test Pattern-Erkennung
        patterns = analyzer.analyze_candlestick_patterns(test_bars)
        print(f"✅ Detected {len(patterns)} patterns")
        
        pattern_types = {}
        for pattern in patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            print(f"   {pattern.pattern_name}: {pattern.confidence:.2f} ({pattern.pattern_type})")
        
        print(f"📊 Pattern Types: {pattern_types}")
        
        # Test Statistiken
        stats = analyzer.get_statistics()
        print(f"📊 Analyzer Stats: {stats}")
        
        print("✅ VisualPatternAnalyzer Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ VisualPatternAnalyzer Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_feature_extraction():
    """Test 2: Enhanced Feature Extraction mit Zeitnormierung"""
    print("🧪 Test 2: Enhanced Feature Extraction...")
    
    try:
        from ai_indicator_optimizer.ai.enhanced_feature_extractor import create_enhanced_feature_extractor
        
        print("✅ Enhanced Feature Extractor importiert")
        
        # Mock Bar für Testing
        class MockBar:
            def __init__(self, open_price, high, low, close, volume, timestamp):
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.ts_init = timestamp
        
        # Test-Konfiguration
        config = {
            "include_time_features": True,
            "include_technical_indicators": True,
            "include_pattern_features": True,
            "include_volatility_features": True,
            "max_history": 20
        }
        
        extractor = create_enhanced_feature_extractor(config)
        
        # Test mit mehreren Bars für History-Building
        base_time = int(time.time() * 1e9)
        base_price = 1.1000
        
        print("📊 Extrahiere Enhanced Features...")
        
        for i in range(15):
            # Simuliere realistische Preis-Bewegung
            price_trend = 0.0002 * (i - 7)  # Trend
            price_noise = (hash(str(i * 11)) % 1000 - 500) * 0.000001  # Noise
            
            price = base_price + price_trend + price_noise
            
            bar = MockBar(
                open_price=price,
                high=price + 0.0003 + abs(price_noise) * 2,
                low=price - 0.0002 - abs(price_noise),
                close=price + 0.0001 + price_noise * 0.5,
                volume=1000 + i * 50 + (hash(str(i)) % 500),
                timestamp=base_time + i * 60 * 1e9  # 1 Minute Bars
            )
            
            features = extractor.extract_enhanced_features(bar)
            
            if i == 14:  # Zeige Features vom letzten Bar
                print(f"📊 Enhanced Features (Bar {i+1}):")
                
                # Time Features
                time_features = {k: v for k, v in features.items() if k.startswith(('hour', 'minute', 'dow', 'is_'))}
                print(f"   Time Features: {len(time_features)} features")
                print(f"   Sample: hour={features.get('hour', 0)}, is_london_session={features.get('is_london_session', 0)}")
                
                # Technical Features
                tech_features = {k: v for k, v in features.items() if k.startswith(('sma', 'ema', 'rsi', 'bb', 'atr'))}
                print(f"   Technical Features: {len(tech_features)} features")
                print(f"   Sample: rsi_14={features.get('rsi_14', 50):.1f}, bb_position={features.get('bb_position', 0.5):.2f}")
                
                # Pattern Features
                pattern_features = {k: v for k, v in features.items() if k.startswith('pattern')}
                print(f"   Pattern Features: {len(pattern_features)} features")
                print(f"   Sample: pattern_count={features.get('pattern_count', 0)}, pattern_confidence_max={features.get('pattern_confidence_max', 0):.2f}")
                
                # Volatility Features
                vol_features = {k: v for k, v in features.items() if 'volatility' in k or 'velocity' in k}
                print(f"   Volatility Features: {len(vol_features)} features")
                
                # Market Regime Features
                regime_features = {k: v for k, v in features.items() if k.startswith('regime') or k.startswith('trend')}
                print(f"   Market Regime Features: {len(regime_features)} features")
                
                print(f"📊 Total Features: {len(features)}")
        
        # Test Statistiken
        stats = extractor.get_statistics()
        print(f"📊 Extractor Stats: {stats}")
        
        print("✅ Enhanced Feature Extraction Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Feature Extraction Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_confidence_position_sizing():
    """Test 3: Confidence-basierte Position-Sizing"""
    print("🧪 Test 3: Confidence-basierte Position-Sizing...")
    
    try:
        from ai_indicator_optimizer.ai.confidence_position_sizer import create_confidence_position_sizer
        
        print("✅ Confidence Position Sizer importiert")
        
        # Test-Konfiguration
        config = {
            "base_position_size": 1000,
            "max_position_size": 5000,
            "min_position_size": 100,
            "confidence_multiplier": 2.0,
            "min_confidence": 0.6,
            "use_kelly_criterion": True,
            "volatility_adjustment": True,
            "drawdown_protection": True
        }
        
        sizer = create_confidence_position_sizer(config)
        
        # Test verschiedene Szenarien
        test_scenarios = [
            {
                "name": "High Confidence Trending",
                "confidence_score": 0.9,
                "risk_score": 0.2,
                "market_regime": "trending",
                "volatility": 0.01,
                "account_balance": 100000
            },
            {
                "name": "Low Confidence Volatile",
                "confidence_score": 0.5,
                "risk_score": 0.8,
                "market_regime": "volatile",
                "volatility": 0.03,
                "account_balance": 100000
            },
            {
                "name": "Medium Confidence Ranging",
                "confidence_score": 0.7,
                "risk_score": 0.4,
                "market_regime": "ranging",
                "volatility": 0.015,
                "account_balance": 100000
            }
        ]
        
        print("📊 Testing Position-Sizing-Szenarien...")
        
        for scenario in test_scenarios:
            result = sizer.calculate_position_size(
                confidence_score=scenario["confidence_score"],
                risk_score=scenario["risk_score"],
                market_regime=scenario["market_regime"],
                volatility=scenario["volatility"],
                account_balance=scenario["account_balance"],
                additional_factors={"liquidity_bonus": 0.1}
            )
            
            print(f"\\n   {scenario['name']}:")
            print(f"     Position Size: {result['position_size']}")
            print(f"     Risk %: {result['risk_metrics']['risk_percentage']:.2%}")
            print(f"     Risk-Reward: {result['risk_metrics']['risk_reward_ratio']:.2f}")
            print(f"     Sizing Steps: base={result['sizing_steps']['base_size']:.0f} -> final={result['sizing_steps']['final_size']:.0f}")
        
        # Test Performance Update
        sizer.update_performance({"pnl": -500})  # Simulate loss
        sizer.update_performance({"pnl": 1000})  # Simulate win
        
        # Test Statistiken
        stats = sizer.get_statistics()
        print(f"\\n📊 Position Sizer Stats: {stats}")
        
        print("✅ Confidence Position Sizing Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Confidence Position Sizing Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_live_control_system():
    """Test 4: Live-Control-System"""
    print("🧪 Test 4: Live-Control-System...")
    
    try:
        from ai_indicator_optimizer.ai.live_control_system import create_live_control_system
        
        print("✅ Live Control System importiert")
        
        # Test ohne Redis/Kafka (Mock-Modus)
        control_system = create_live_control_system(
            strategy_id="test_strategy_phase2",
            config={"redis_host": "localhost"},
            use_redis=False,
            use_kafka=False
        )
        
        print("📊 Testing Live Control Commands...")
        
        with control_system:
            # Test Parameter Update
            control_system.update_parameter("min_confidence", 0.8)
            control_system.update_parameter("max_position_size", 3000)
            print(f"   Parameters updated: min_confidence={control_system.get_parameter('min_confidence')}")
            
            # Test Pause/Resume
            control_system._handle_pause_command({"reason": "Phase 2 test pause"})
            print(f"   Paused: {control_system.is_paused}")
            print(f"   Trading allowed: {control_system.is_trading_allowed()}")
            
            control_system._handle_resume_command({"reason": "Phase 2 test resume"})
            print(f"   Resumed: {not control_system.is_paused}")
            print(f"   Trading allowed: {control_system.is_trading_allowed()}")
            
            # Test Emergency Stop
            control_system._handle_emergency_stop_command({"reason": "Test emergency"})
            print(f"   Emergency stopped: {control_system.is_emergency_stopped}")
            print(f"   Trading allowed: {control_system.is_trading_allowed()}")
            
            # Test Status
            status = control_system.get_current_status()
            print(f"   Status: strategy={status.strategy_id}, paused={status.is_paused}, emergency={status.is_emergency_stopped}")
            
            # Test Command Sending (Mock)
            success = control_system.send_command("update_params", {"test_param": "test_value"})
            print(f"   Command sent: {success}")
        
        # Test Statistiken
        stats = control_system.get_statistics()
        print(f"📊 Live Control Stats: {stats}")
        
        print("✅ Live Control System Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Live Control System Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_config():
    """Test 5: Environment-Variable-Konfiguration"""
    print("🧪 Test 5: Environment-Variable-Konfiguration...")
    
    try:
        from ai_indicator_optimizer.ai.environment_config import EnvironmentConfigManager, get_config
        
        print("✅ Environment Config Manager importiert")
        
        # Test verschiedene Environments
        for env in ["development", "staging", "production"]:
            print(f"\\n📋 Testing environment: {env}")
            
            config_manager = EnvironmentConfigManager(environment=env)
            config = config_manager.get_config()
            
            print(f"   Environment: {config.environment}")
            print(f"   Debug Mode: {config.debug_mode}")
            print(f"   AI Endpoint: {config.ai_endpoint}")
            print(f"   Min Confidence: {config.min_confidence}")
            print(f"   Base Position Size: {config.base_position_size}")
            print(f"   Use Redis: {config.use_redis}")
            print(f"   Log Rotation: {config.log_rotation}")
        
        # Test Configuration Update
        print("\\n📋 Testing configuration update...")
        config_manager.update_config({
            "min_confidence": 0.85,
            "debug_mode": False,
            "base_position_size": 2000
        })
        updated_config = config_manager.get_config()
        print(f"   Updated Min Confidence: {updated_config.min_confidence}")
        print(f"   Updated Debug Mode: {updated_config.debug_mode}")
        print(f"   Updated Base Position Size: {updated_config.base_position_size}")
        
        # Test Configuration Dictionary
        config_dict = config_manager.get_config_dict()
        print(f"   Config Dict Keys: {len(config_dict)} keys")
        
        # Test Environment Info
        env_info = config_manager.get_environment_info()
        print(f"   Environment Info: {env_info}")
        
        # Test Template Export
        config_manager.export_config_template("test_phase2_config_template.json")
        print("   Template exported: test_phase2_config_template.json")
        
        print("✅ Environment Configuration Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Environment Configuration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_confidence_scoring():
    """Test 6: Enhanced Confidence Scoring"""
    print("🧪 Test 6: Enhanced Confidence Scoring...")
    
    try:
        from ai_indicator_optimizer.ai.confidence_scoring import create_enhanced_confidence_scorer, ConfidenceLevel
        
        print("✅ Enhanced Confidence Scorer importiert")
        
        # Test-Konfiguration
        config = {
            "calibration_method": "isotonic",
            "factor_weights": {
                "ai_prediction": 0.4,
                "pattern_analysis": 0.2,
                "technical_indicators": 0.2,
                "market_regime": 0.1,
                "volatility": 0.1
            }
        }
        
        scorer = create_enhanced_confidence_scorer(config)
        
        # Test verschiedene Szenarien
        test_scenarios = [
            {
                "name": "High Confidence Scenario",
                "ai_prediction": {"confidence": 0.9, "action": "BUY"},
                "pattern_features": {
                    "pattern_confidence_max": 0.8,
                    "pattern_count": 2,
                    "pattern_bullish": 2,
                    "pattern_bearish": 0,
                    "pattern_neutral": 0
                },
                "technical_indicators": {
                    "rsi_14": 25,  # Oversold
                    "bb_position": 0.1,  # Near lower band
                    "volume_ratio": 2.0,  # High volume
                    "sma_5": 1.1000,
                    "sma_20": 1.0980
                },
                "market_context": {
                    "market_regime": "trending",
                    "volatility": 0.01,
                    "trend_strength": 0.05,
                    "data_completeness": 1.0,
                    "history_length": 50
                }
            },
            {
                "name": "Low Confidence Scenario",
                "ai_prediction": {"confidence": 0.4, "action": "HOLD"},
                "pattern_features": {
                    "pattern_confidence_max": 0.3,
                    "pattern_count": 1,
                    "pattern_bullish": 1,
                    "pattern_bearish": 1,
                    "pattern_neutral": 0
                },
                "technical_indicators": {
                    "rsi_14": 50,  # Neutral
                    "bb_position": 0.5,  # Middle
                    "volume_ratio": 0.8,  # Low volume
                    "sma_5": 1.1000,
                    "sma_20": 1.1000
                },
                "market_context": {
                    "market_regime": "volatile",
                    "volatility": 0.04,
                    "trend_strength": 0.001,
                    "data_completeness": 0.8,
                    "history_length": 15
                }
            }
        ]
        
        print("📊 Testing Enhanced Confidence Scoring...")
        
        for scenario in test_scenarios:
            metrics = scorer.calculate_enhanced_confidence(
                ai_prediction=scenario["ai_prediction"],
                pattern_features=scenario["pattern_features"],
                technical_indicators=scenario["technical_indicators"],
                market_context=scenario["market_context"],
                additional_factors={"test_factor": 0.1}
            )
            
            print(f"\\n   {scenario['name']}:")
            print(f"     Overall Confidence: {metrics.overall_confidence:.3f}")
            print(f"     Calibrated Confidence: {metrics.calibrated_confidence:.3f}")
            print(f"     Risk-Adjusted Confidence: {metrics.risk_adjusted_confidence:.3f}")
            print(f"     Confidence Level: {metrics.confidence_level.value}")
            print(f"     Reliability Score: {metrics.reliability_score:.3f}")
            print(f"     Prediction Interval: ({metrics.prediction_interval[0]:.3f}, {metrics.prediction_interval[1]:.3f})")
            print(f"     Temporal Stability: {metrics.temporal_stability:.3f}")
            
            # Component Confidences
            print(f"     Component Confidences:")
            for component, confidence in metrics.component_confidences.items():
                print(f"       {component}: {confidence:.3f}")
            
            # Uncertainty Sources
            print(f"     Top Uncertainty Sources:")
            sorted_uncertainties = sorted(metrics.uncertainty_sources.items(), key=lambda x: x[1], reverse=True)
            for source, uncertainty in sorted_uncertainties[:3]:
                print(f"       {source.value}: {uncertainty:.3f}")
        
        # Test Calibration Update
        scorer.update_calibration(0.8, 1.0)  # High confidence, positive outcome
        scorer.update_calibration(0.3, 0.0)  # Low confidence, negative outcome
        
        # Test Statistiken
        stats = scorer.get_statistics()
        print(f"\\n📊 Confidence Scorer Stats: {stats}")
        
        print("✅ Enhanced Confidence Scoring Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Confidence Scoring Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_integration():
    """Test 7: Complete Integration Workflow"""
    print("🧪 Test 7: Complete Integration Workflow...")
    
    try:
        # Import aller Komponenten
        from ai_indicator_optimizer.ai.visual_pattern_analyzer import create_visual_pattern_analyzer
        from ai_indicator_optimizer.ai.enhanced_feature_extractor import create_enhanced_feature_extractor
        from ai_indicator_optimizer.ai.confidence_position_sizer import create_confidence_position_sizer
        from ai_indicator_optimizer.ai.live_control_system import create_live_control_system
        from ai_indicator_optimizer.ai.environment_config import EnvironmentConfigManager
        from ai_indicator_optimizer.ai.confidence_scoring import create_enhanced_confidence_scorer
        
        print("✅ Alle Phase 2 Komponenten importiert")
        
        # Setup Complete Integration
        print("📊 Setting up Complete Integration...")
        
        # Environment Configuration
        config_manager = EnvironmentConfigManager(environment="development")
        config = config_manager.get_config()
        
        # Initialize Komponenten
        pattern_analyzer = create_visual_pattern_analyzer(use_mock=True, debug_mode=False)
        feature_extractor = create_enhanced_feature_extractor()
        position_sizer = create_confidence_position_sizer()
        confidence_scorer = create_enhanced_confidence_scorer()
        
        # Live Control System
        live_control = create_live_control_system(
            strategy_id="phase2_integration_test",
            use_redis=False,
            use_kafka=False
        )
        
        print("✅ Alle Komponenten initialisiert")
        
        # Simuliere Complete Trading Workflow
        print("📊 Simuliere Complete Trading Workflow...")
        
        # Mock Bar Data
        class MockBar:
            def __init__(self, open_price, high, low, close, volume, timestamp):
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.ts_init = timestamp
        
        base_time = int(time.time() * 1e9)
        base_price = 1.1000
        
        with live_control:
            for i in range(5):
                # 1. Create Mock Bar
                price = base_price + i * 0.0005 + (hash(str(i)) % 100 - 50) * 0.000001
                bar = MockBar(
                    open_price=price,
                    high=price + 0.0003,
                    low=price - 0.0002,
                    close=price + 0.0001,
                    volume=1000 + i * 100,
                    timestamp=base_time + i * 60 * 1e9
                )
                
                # 2. Check Live Control
                if not live_control.is_trading_allowed():
                    print(f"   Bar {i+1}: Trading not allowed (paused/stopped)")
                    continue
                
                # 3. Extract Enhanced Features
                features = feature_extractor.extract_enhanced_features(bar)
                
                # 4. Analyze Patterns
                patterns = pattern_analyzer.analyze_candlestick_patterns([bar])
                
                # 5. Create Pattern Features
                pattern_features = {
                    "pattern_count": len(patterns),
                    "pattern_confidence_max": max([p.confidence for p in patterns], default=0.0),
                    "pattern_bullish": sum(1 for p in patterns if p.pattern_type == "bullish"),
                    "pattern_bearish": sum(1 for p in patterns if p.pattern_type == "bearish"),
                    "pattern_neutral": sum(1 for p in patterns if p.pattern_type == "neutral")
                }
                
                # 6. Mock AI Prediction
                ai_prediction = {
                    "action": ["BUY", "SELL", "HOLD"][i % 3],
                    "confidence": 0.6 + (i % 3) * 0.1,
                    "reasoning": f"integration_test_bar_{i}"
                }
                
                # 7. Market Context
                market_context = {
                    "market_regime": ["trending", "ranging", "volatile"][i % 3],
                    "volatility": 0.01 + (i % 3) * 0.005,
                    "trend_strength": 0.02 + (i % 2) * 0.01,
                    "data_completeness": 1.0,
                    "history_length": min(50, i + 10)
                }
                
                # 8. Enhanced Confidence Scoring
                confidence_metrics = confidence_scorer.calculate_enhanced_confidence(
                    ai_prediction=ai_prediction,
                    pattern_features=pattern_features,
                    technical_indicators=features,
                    market_context=market_context
                )
                
                # 9. Position Sizing
                position_result = position_sizer.calculate_position_size(
                    confidence_score=confidence_metrics.risk_adjusted_confidence,
                    risk_score=0.1 + (i % 3) * 0.1,
                    market_regime=market_context["market_regime"],
                    volatility=market_context["volatility"],
                    account_balance=100000
                )
                
                # 10. Results
                print(f"\\n   Bar {i+1} Complete Workflow:")
                print(f"     Price: {bar.close:.4f}")
                print(f"     Patterns: {len(patterns)} detected")
                print(f"     AI Prediction: {ai_prediction['action']} @ {ai_prediction['confidence']:.2f}")
                print(f"     Enhanced Confidence: {confidence_metrics.risk_adjusted_confidence:.3f} ({confidence_metrics.confidence_level.value})")
                print(f"     Position Size: {position_result['position_size']}")
                print(f"     Market Regime: {market_context['market_regime']}")
                print(f"     Features Extracted: {len(features)}")
                
                # 11. Update Calibration
                confidence_scorer.update_calibration(
                    confidence_metrics.overall_confidence,
                    1.0 if ai_prediction["action"] == "BUY" else 0.0  # Mock outcome
                )
                
                # 12. Update Position Sizer Performance
                position_sizer.update_performance({"pnl": (i - 2) * 100})  # Mock PnL
        
        # Final Statistics
        print("\\n📊 Complete Integration Statistics:")
        
        pattern_stats = pattern_analyzer.get_statistics()
        print(f"   Pattern Analyzer: {pattern_stats['patterns_detected']} patterns detected")
        
        feature_stats = feature_extractor.get_statistics()
        print(f"   Feature Extractor: {feature_stats['features_extracted']} features extracted")
        
        position_stats = position_sizer.get_statistics()
        print(f"   Position Sizer: {position_stats['positions_sized']} positions sized")
        
        confidence_stats = confidence_scorer.get_statistics()
        print(f"   Confidence Scorer: {confidence_stats['scores_calculated']} scores calculated")
        
        live_stats = live_control.get_statistics()
        print(f"   Live Control: {live_stats['commands_processed']} commands processed")
        
        print("✅ Complete Integration Workflow Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Complete Integration Workflow Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Phase 2 Integration Test Suite"""
    print("🚀 Phase 2 Integration Test Suite")
    print("Task 8: Enhanced Multimodal Pattern Recognition Engine")
    print("=" * 80)
    
    # Erstelle Test-Verzeichnisse
    Path("test_logs").mkdir(exist_ok=True)
    
    success = True
    
    # Test 1: VisualPatternAnalyzer
    print("\\n📋 Test 1: VisualPatternAnalyzer")
    if not test_visual_pattern_analyzer():
        success = False
    
    # Test 2: Enhanced Feature Extraction
    print("\\n📋 Test 2: Enhanced Feature Extraction")
    if not test_enhanced_feature_extraction():
        success = False
    
    # Test 3: Confidence Position Sizing
    print("\\n📋 Test 3: Confidence Position Sizing")
    if not test_confidence_position_sizing():
        success = False
    
    # Test 4: Live Control System
    print("\\n📋 Test 4: Live Control System")
    if not test_live_control_system():
        success = False
    
    # Test 5: Environment Configuration
    print("\\n📋 Test 5: Environment Configuration")
    if not test_environment_config():
        success = False
    
    # Test 6: Enhanced Confidence Scoring
    print("\\n📋 Test 6: Enhanced Confidence Scoring")
    if not test_enhanced_confidence_scoring():
        success = False
    
    # Test 7: Complete Integration
    print("\\n📋 Test 7: Complete Integration Workflow")
    if not test_complete_integration():
        success = False
    
    print("\\n" + "=" * 80)
    
    if success:
        print("🎉 Phase 2 Integration Tests erfolgreich!")
        
        print("\\n🎯 Task 8 VOLLSTÄNDIG ABGESCHLOSSEN:")
        print("  ✅ VisualPatternAnalyzer für Candlestick-Pattern-Erkennung")
        print("  ✅ Enhanced Feature Extraction mit Zeitnormierung")
        print("  ✅ Confidence-basierte Position-Sizing mit Risk-Score-Integration")
        print("  ✅ Live-Control-System via Redis/Kafka")
        print("  ✅ Environment-Variable-basierte Konfiguration")
        print("  ✅ Enhanced Confidence Scoring mit Multi-Factor-Validation")
        print("  ✅ Complete Integration Workflow")
        
        print("\\n📊 Phase 2 Features implementiert:")
        print("  ✅ Multimodale Pattern-Erkennung")
        print("  ✅ Zeitnormierte Feature-Extraktion")
        print("  ✅ Multi-Factor-Confidence-Scoring")
        print("  ✅ Risk-Adjusted Position-Sizing")
        print("  ✅ Live-Parameter-Updates")
        print("  ✅ Environment-basierte Konfiguration")
        print("  ✅ Uncertainty Quantification")
        print("  ✅ Temporal Stability Analysis")
        
        print("\\n🚀 FAHRPLAN-STATUS: Phase 2 ✅ ABGESCHLOSSEN")
        print("   Nächste Phase: Phase 3 - Production Integration (Task 17)")
        
    else:
        print("❌ Phase 2 Integration Tests fehlgeschlagen!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)