#!/usr/bin/env python3
"""
Test Script für Enhanced AI Pattern Strategy
ChatGPT-Verbesserungen: Feature Logging, Dataset Builder, Enhanced Confidence
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_enhanced_logging_components():
    """Test der Enhanced Logging-Komponenten"""
    print("🧪 Testing Enhanced Logging Components...")
    
    try:
        # Test FeaturePredictionLogger
        from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger
        
        print("✅ FeaturePredictionLogger importiert")
        
        # Erstelle Test-Logger
        with FeaturePredictionLogger("test_logs/test_features.parquet", buffer_size=3) as logger:
            # Test-Daten loggen
            for i in range(5):
                logger.log(
                    ts_ns=int(time.time() * 1e9) + i * 1000000000,
                    instrument="EUR/USD",
                    features={
                        "open": 1.1000 + i * 0.0001,
                        "close": 1.1005 + i * 0.0001,
                        "rsi": 50 + i * 2,
                        "volatility": 0.001 + i * 0.0001
                    },
                    prediction={
                        "action": "BUY" if i % 2 == 0 else "SELL",
                        "confidence": 0.7 + i * 0.05,
                        "reasoning": f"test_prediction_{i}"
                    },
                    confidence_score=0.8 + i * 0.02,
                    risk_score=0.1 + i * 0.01,
                    market_regime="trending"
                )
        
        # Zeige Stats
        stats = logger.get_stats()
        print(f"📊 Logger Stats: {stats}")
        
        print("✅ FeaturePredictionLogger Test erfolgreich!")
        
    except Exception as e:
        print(f"❌ FeaturePredictionLogger Test fehlgeschlagen: {e}")
        return False
    
    try:
        # Test BarDatasetBuilder
        from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.enums import BarType
        from nautilus_trader.model.data import Bar
        
        print("✅ BarDatasetBuilder importiert")
        
        # Erstelle Test-Builder
        builder = BarDatasetBuilder(horizon=3, min_bars=5)
        
        # Generiere Test-Bars
        instrument_id = InstrumentId.from_str("EUR/USD.SIM")
        
        for i in range(10):
            # Simuliere Preis-Bewegung
            base_price = 1.1000
            price_change = (i - 5) * 0.0001
            
            bar = Bar(
                bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
                open=base_price + price_change,
                high=base_price + price_change + 0.0002,
                low=base_price + price_change - 0.0001,
                close=base_price + price_change + 0.0001,
                volume=1000 + i * 100,
                ts_event=int(time.time() * 1e9) + i * 60 * 1e9,
                ts_init=int(time.time() * 1e9) + i * 60 * 1e9
            )
            
            builder.on_bar(bar)
        
        # Zeige Stats
        stats = builder.get_stats()
        print(f"📊 Dataset Builder Stats: {stats}")
        
        # Export Test
        if builder.to_parquet("test_logs/test_dataset.parquet"):
            print("✅ BarDatasetBuilder Test erfolgreich!")
        else:
            print("❌ BarDatasetBuilder Export fehlgeschlagen!")
            return False
            
    except Exception as e:
        print(f"❌ BarDatasetBuilder Test fehlgeschlagen: {e}")
        return False
    
    return True

def test_enhanced_ai_strategy():
    """Test der Enhanced AI Pattern Strategy"""
    print("🧪 Testing Enhanced AI Pattern Strategy...")
    
    try:
        # Import Strategy
        sys.path.append('strategies/ai_strategies')
        from ai_pattern_strategy import AIPatternStrategy
        
        print("✅ Enhanced AI Pattern Strategy importiert")
        
        # Test-Konfiguration
        config = {
            "ai_endpoint": "http://localhost:8080/predictions/pattern_model",
            "min_confidence": 0.6,
            "position_size": 1000,
            "use_mock": True,
            "debug_mode": True,
            "feature_log_path": "test_logs/strategy_features.parquet",
            "log_buffer_size": 5,
            "dataset_horizon": 3,
            "min_dataset_bars": 5,
            "confidence_multiplier": 1.5,
            "max_position_multiplier": 2.0
        }
        
        # Erstelle Strategy-Instanz
        strategy = AIPatternStrategy(config)
        
        print("✅ Enhanced AI Pattern Strategy erstellt")
        
        # Simuliere Strategy-Lifecycle
        strategy.on_start()
        
        # Simuliere Bar-Processing
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.enums import BarType
        from nautilus_trader.model.data import Bar
        
        for i in range(8):  # Genug Bars für Forward-Return-Labels
            base_price = 1.1000
            price_change = (i - 4) * 0.0002
            
            bar = Bar(
                bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
                open=base_price + price_change,
                high=base_price + price_change + 0.0003,
                low=base_price + price_change - 0.0002,
                close=base_price + price_change + 0.0001,
                volume=1000 + i * 200,
                ts_event=int(time.time() * 1e9) + i * 60 * 1e9,
                ts_init=int(time.time() * 1e9) + i * 60 * 1e9
            )
            
            strategy.on_bar(bar)
            time.sleep(0.1)  # Kurze Pause
        
        # Strategy beenden
        strategy.on_stop()
        
        print("✅ Enhanced AI Pattern Strategy Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced AI Pattern Strategy Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Haupttest-Funktion"""
    print("🚀 Starting Enhanced AI Strategy Tests...")
    print("=" * 60)
    
    # Erstelle Test-Verzeichnisse
    Path("test_logs").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)
    
    success = True
    
    # Test 1: Enhanced Logging Components
    print("\n📋 Test 1: Enhanced Logging Components")
    if not test_enhanced_logging_components():
        success = False
    
    # Test 2: Enhanced AI Strategy
    print("\n📋 Test 2: Enhanced AI Pattern Strategy")
    if not test_enhanced_ai_strategy():
        success = False
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Alle Tests erfolgreich!")
        print("\n📊 Generierte Test-Dateien:")
        
        # Zeige generierte Dateien
        test_files = [
            "test_logs/test_features.parquet",
            "test_logs/test_dataset.parquet", 
            "test_logs/strategy_features.parquet"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                print(f"  ✅ {file_path} ({size} bytes)")
            else:
                print(f"  ❌ {file_path} (nicht gefunden)")
        
        print("\n🎯 Nächste Schritte:")
        print("  1. Prüfe die generierten Parquet-Dateien")
        print("  2. Teste mit echten Nautilus-Daten")
        print("  3. Implementiere TorchServe-Integration")
        
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)