#!/usr/bin/env python3
"""
Test für Automated Library Population System
Einfache Tests ohne komplexe Dependencies
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_synthetic_pattern_generator():
    """Test SyntheticPatternGenerator"""
    print("🎨 Testing SyntheticPatternGenerator...")
    
    try:
        from ai_indicator_optimizer.library.synthetic_pattern_generator import (
            SyntheticConfig, PatternTemplate, SyntheticPatternGenerator
        )
        
        # Test Config
        config = SyntheticConfig(
            base_patterns=["double_top"],
            variations_per_pattern=2,
            use_ai_generation=False
        )
        
        print(f"✅ SyntheticConfig created: {len(config.base_patterns)} pattern types")
        
        # Test Pattern Template
        template = PatternTemplate("double_top")
        ohlcv_df = template.generate(config)
        
        print(f"✅ Pattern template generated: {len(ohlcv_df)} candles")
        print(f"   Columns: {list(ohlcv_df.columns)}")
        print(f"   Price range: {ohlcv_df['close'].min():.5f} - {ohlcv_df['close'].max():.5f}")
        
        # Test Generator
        generator = SyntheticPatternGenerator(config)
        variations = generator.generate_pattern_variations("double_top")
        
        print(f"✅ Generated {len(variations)} pattern variations")
        
        if variations:
            pattern = variations[0]
            print(f"   Sample pattern: {pattern.pattern_type}")
            print(f"   Confidence: {pattern.confidence:.2f}")
            print(f"   Synthetic: {pattern.market_context.get('synthetic_generation', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ SyntheticPatternGenerator test failed: {e}")
        return False

def test_community_strategy_importer():
    """Test CommunityStrategyImporter"""
    print("\n🌐 Testing CommunityStrategyImporter...")
    
    try:
        from ai_indicator_optimizer.library.community_strategy_importer import (
            CommunityStrategyImporter, PineScriptParser, ImportedStrategy
        )
        
        # Test Pine Script Parser
        parser = PineScriptParser()
        
        sample_pine = '''
//@version=5
strategy("Test Strategy", overlay=true)

rsi = ta.rsi(close, 14)
longCondition = rsi < 30
shortCondition = rsi > 70

if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.entry("Short", strategy.short)
        '''
        
        strategy = parser.parse_pine_script(sample_pine)
        
        print(f"✅ Pine Script parsed: {strategy.name}")
        print(f"   Entry conditions: {len(strategy.entry_conditions)}")
        print(f"   Exit conditions: {len(strategy.exit_conditions)}")
        print(f"   Risk management: {strategy.risk_management}")
        
        # Test Importer
        importer = CommunityStrategyImporter()
        
        # Test sample strategies
        strategies = importer._get_sample_pine_scripts()
        print(f"✅ Sample strategies available: {len(strategies)}")
        
        # Test JSON strategies
        json_strategies = importer._get_sample_json_strategies()
        print(f"✅ Sample JSON strategies: {len(json_strategies)}")
        
        if json_strategies:
            json_strategy = importer._parse_json_strategy(json_strategies[0], importer.strategy_sources[0])
            print(f"   JSON strategy: {json_strategy.name}")
            print(f"   Win rate: {json_strategy.win_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ CommunityStrategyImporter test failed: {e}")
        return False

def test_pattern_validator():
    """Test PatternValidator"""
    print("\n🔍 Testing PatternValidator...")
    
    try:
        from ai_indicator_optimizer.library.pattern_validator import (
            PatternValidator, ValidationLevel, ValidationResult,
            TechnicalValidator, StatisticalValidator, TradingValidator
        )
        
        # Create mock pattern for testing
        from ai_indicator_optimizer.library.historical_pattern_miner import MinedPattern
        
        # Generate mock OHLCV data
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        np.random.seed(42)
        
        base_price = 1.1000
        prices = [base_price]
        
        for i in range(49):
            change = np.random.normal(0, 0.0005)
            new_price = prices[-1] + change
            prices.append(new_price)
        
        ohlcv_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price + np.random.uniform(0, 0.001)
            low = price - np.random.uniform(0, 0.001)
            open_price = prices[i-1] if i > 0 else price
            
            ohlcv_data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, price, high),
                'low': min(open_price, price, low),
                'close': price,
                'volume': np.random.uniform(1000000, 5000000)
            })
        
        # Create mock pattern
        mock_pattern = MinedPattern(
            pattern_id="test_pattern_001",
            symbol="EUR/USD",
            timeframe="1H",
            pattern_type="double_top",
            confidence=0.75,
            start_time=dates[0],
            end_time=dates[-1],
            price_data={"ohlcv": ohlcv_data},
            indicators={
                "RSI": [np.random.uniform(20, 80) for _ in range(50)],
                "SMA_20": [np.random.uniform(1.095, 1.105) for _ in range(50)]
            },
            pattern_features={
                "pattern_type": "double_top",
                "confidence": 0.75
            },
            market_context={
                "symbol": "EUR/USD",
                "timeframe": "1H",
                "volatility": 0.002
            },
            mining_timestamp=datetime.now()
        )
        
        print(f"✅ Mock pattern created: {mock_pattern.pattern_id}")
        
        # Test Validators
        technical_validator = TechnicalValidator()
        price_score, price_issues, price_metrics = technical_validator.validate_price_data(mock_pattern)
        
        print(f"✅ Technical validation: score={price_score:.2f}, issues={len(price_issues)}")
        
        trading_validator = TradingValidator()
        trading_score, trading_issues, trading_metrics = trading_validator.validate_trading_relevance(mock_pattern)
        
        print(f"✅ Trading validation: score={trading_score:.2f}, issues={len(trading_issues)}")
        
        # Test Full Validator
        validator = PatternValidator(ValidationLevel.STANDARD)
        result = validator.validate_pattern(mock_pattern)
        
        print(f"✅ Full validation completed:")
        print(f"   Valid: {result.is_valid}")
        print(f"   Quality Score: {result.quality_score:.2f}")
        print(f"   Technical Score: {result.technical_score:.2f}")
        print(f"   Trading Score: {result.trading_score:.2f}")
        print(f"   Critical Issues: {len(result.critical_issues)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ PatternValidator test failed: {e}")
        return False

def test_integration():
    """Test Integration zwischen Komponenten"""
    print("\n🔄 Testing Component Integration...")
    
    try:
        # Test 1: Synthetic → Validation
        from ai_indicator_optimizer.library.synthetic_pattern_generator import quick_synthetic_generation
        from ai_indicator_optimizer.library.pattern_validator import quick_pattern_validation, ValidationLevel
        
        # Generate synthetic pattern
        synthetic_patterns = quick_synthetic_generation("triangle", 1)
        
        if synthetic_patterns:
            pattern = synthetic_patterns[0]
            
            # Validate synthetic pattern
            validation_result = quick_pattern_validation(pattern, ValidationLevel.BASIC)
            
            print(f"✅ Synthetic → Validation integration:")
            print(f"   Pattern: {pattern.pattern_type}")
            print(f"   Valid: {validation_result.is_valid}")
            print(f"   Quality: {validation_result.quality_score:.2f}")
        
        # Test 2: Strategy Import → Pattern Conversion
        from ai_indicator_optimizer.library.community_strategy_importer import CommunityStrategyImporter
        
        importer = CommunityStrategyImporter()
        json_strategies = importer._get_sample_json_strategies()
        
        if json_strategies:
            parsed_strategy = importer._parse_json_strategy(json_strategies[0], importer.strategy_sources[0])
            
            # Convert to pattern
            converted_patterns = importer.convert_to_mined_patterns([parsed_strategy])
            
            print(f"✅ Strategy → Pattern conversion:")
            print(f"   Strategy: {parsed_strategy.name}")
            print(f"   Converted patterns: {len(converted_patterns)}")
            
            if converted_patterns:
                pattern = converted_patterns[0]
                print(f"   Pattern type: {pattern.pattern_type}")
                print(f"   Features: {len(pattern.pattern_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Haupttest-Funktion"""
    print("🚀 Automated Library Population System Tests")
    print("=" * 50)
    
    tests = [
        ("SyntheticPatternGenerator", test_synthetic_pattern_generator),
        ("CommunityStrategyImporter", test_community_strategy_importer),
        ("PatternValidator", test_pattern_validator),
        ("Component Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Automated Library Population System is ready!")
    else:
        print(f"\n⚠️ {failed} tests failed. Check implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)