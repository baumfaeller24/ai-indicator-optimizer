#!/usr/bin/env python3
"""
Vereinfachter Test für Enhanced Logging Components
Ohne Nautilus-Abhängigkeiten - fokussiert auf ChatGPT-Verbesserungen
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

def test_feature_prediction_logger():
    """Test des FeaturePredictionLoggers"""
    print("🧪 Testing FeaturePredictionLogger...")
    
    try:
        from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger, create_feature_logger
        
        print("✅ FeaturePredictionLogger importiert")
        
        # Test 1: Standard Logger
        print("\n📋 Test 1: Standard FeaturePredictionLogger")
        with FeaturePredictionLogger("test_logs/standard_features.parquet", buffer_size=3) as logger:
            # Simuliere AI-Trading-Daten
            for i in range(7):
                logger.log(
                    ts_ns=int(time.time() * 1e9) + i * 60 * 1000000000,  # 1 Minute Abstand
                    instrument="EUR/USD",
                    features={
                        "open": 1.1000 + i * 0.0001,
                        "high": 1.1000 + i * 0.0001 + 0.0003,
                        "low": 1.1000 + i * 0.0001 - 0.0002,
                        "close": 1.1005 + i * 0.0001,
                        "volume": 1000 + i * 100,
                        "rsi_14": 50 + i * 3,
                        "sma_20": 1.1002 + i * 0.0001,
                        "volatility_20": 0.001 + i * 0.0001,
                        "momentum_5": (i - 3) * 0.0001,
                        "body_ratio": 0.3 + i * 0.1,
                        "is_bullish": i % 2 == 0,
                        "hour": 9 + (i % 8),
                        "day_of_week": i % 5
                    },
                    prediction={
                        "action": "BUY" if i % 3 == 0 else ("SELL" if i % 3 == 1 else "HOLD"),
                        "confidence": 0.6 + i * 0.05,
                        "reasoning": f"ai_prediction_{i}",
                        "risk_score": 0.1 + i * 0.02
                    },
                    confidence_score=0.7 + i * 0.03,
                    risk_score=0.05 + i * 0.01,
                    market_regime=["trending", "ranging", "volatile", "quiet"][i % 4]
                )
                
                if i == 2:  # Test Auto-Flush bei Buffer-Überlauf
                    print(f"   Buffer-Flush bei Entry {i+1}")
        
        stats = logger.get_stats()
        print(f"📊 Standard Logger Stats: {stats}")
        
        # Test 2: Rotating Logger
        print("\n📋 Test 2: Rotating FeaturePredictionLogger")
        rotating_logger = create_feature_logger(
            "test_logs/rotating_features",
            buffer_size=5,
            rotating=True
        )
        
        # Simuliere Daten über mehrere "Tage"
        base_time = int(time.time() * 1e9)
        for day in range(2):
            for hour in range(3):
                ts = base_time + day * 24 * 3600 * 1e9 + hour * 3600 * 1e9
                
                rotating_logger.log(
                    ts_ns=int(ts),
                    instrument="GBP/USD",
                    features={
                        "open": 1.2500 + day * 0.001 + hour * 0.0001,
                        "close": 1.2505 + day * 0.001 + hour * 0.0001,
                        "rsi": 45 + hour * 5,
                        "day": day,
                        "hour": hour
                    },
                    prediction={
                        "action": ["BUY", "SELL", "HOLD"][hour % 3],
                        "confidence": 0.65 + hour * 0.1
                    }
                )
        
        rotating_logger.close()
        rotating_stats = rotating_logger.get_stats()
        print(f"📊 Rotating Logger Stats: {rotating_stats}")
        
        print("✅ FeaturePredictionLogger Tests erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ FeaturePredictionLogger Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bar_dataset_builder_standalone():
    """Test des BarDatasetBuilders ohne Nautilus-Abhängigkeiten"""
    print("🧪 Testing BarDatasetBuilder (Standalone)...")
    
    try:
        # Simuliere Bar-Daten ohne Nautilus
        class MockBar:
            def __init__(self, open_price, high, low, close, volume, timestamp):
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.ts_init = timestamp
                self.bar_type = MockBarType()
        
        class MockBarType:
            def __init__(self):
                self.instrument_id = "EUR/USD.SIM"
        
        # Erstelle vereinfachten Dataset Builder
        from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
        
        print("✅ BarDatasetBuilder importiert")
        
        builder = BarDatasetBuilder(horizon=3, min_bars=5, include_technical_indicators=True)
        
        # Generiere realistische Forex-Daten
        base_price = 1.1000
        base_time = int(time.time() * 1e9)
        
        print("📊 Generiere Test-Bars mit Forward-Return-Labels...")
        
        for i in range(12):  # Genug für Forward-Returns
            # Simuliere realistische Preis-Bewegung
            trend = 0.0002 * (i - 6)  # Trend-Komponente
            noise = (hash(str(i)) % 1000 - 500) * 0.000001  # Zufälliges Rauschen
            
            price = base_price + trend + noise
            
            bar = MockBar(
                open_price=price,
                high=price + 0.0002 + abs(noise),
                low=price - 0.0001 - abs(noise),
                close=price + 0.0001,
                volume=1000 + i * 50 + (hash(str(i)) % 500),
                timestamp=base_time + i * 60 * 1e9  # 1 Minute Bars
            )
            
            builder.on_bar(bar)
            
            if i % 3 == 0:
                print(f"   Processed {i+1} bars...")
        
        # Zeige Statistiken
        stats = builder.get_stats()
        print(f"📊 Dataset Builder Stats:")
        print(f"   Total Entries: {stats['total_entries']}")
        print(f"   Label Distribution: {stats['label_distribution']}")
        print(f"   Return Statistics: {stats['return_statistics']}")
        
        # Export Dataset
        export_path = "test_logs/standalone_dataset.parquet"
        if builder.to_parquet(export_path, include_metadata=True):
            print(f"✅ Dataset erfolgreich exportiert: {export_path}")
            
            # Lade und zeige Dataset
            import polars as pl
            df = pl.read_parquet(export_path)
            print(f"📊 Exported Dataset Shape: {df.shape}")
            print(f"📊 Columns: {df.columns[:10]}...")  # Erste 10 Spalten
            
            # Zeige Metadata
            metadata_path = Path(export_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"📊 Metadata: {metadata}")
        
        print("✅ BarDatasetBuilder Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ BarDatasetBuilder Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test der Integration zwischen Logger und Dataset Builder"""
    print("🧪 Testing Integration: Logger + Dataset Builder...")
    
    try:
        from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger
        from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
        
        # Simuliere integrierte Trading-Session
        with FeaturePredictionLogger("test_logs/integrated_features.parquet", buffer_size=10) as logger:
            builder = BarDatasetBuilder(horizon=2, min_bars=3)
            
            print("📊 Simuliere integrierte Trading-Session...")
            
            base_time = int(time.time() * 1e9)
            
            for i in range(8):
                # Simuliere Bar-Daten
                price = 1.1000 + i * 0.0001
                
                # Mock Bar für Dataset Builder
                class MockBar:
                    def __init__(self):
                        self.open = price
                        self.high = price + 0.0002
                        self.low = price - 0.0001
                        self.close = price + 0.0001
                        self.volume = 1000 + i * 100
                        self.ts_init = base_time + i * 60 * 1e9
                        self.bar_type = type('MockBarType', (), {'instrument_id': 'EUR/USD'})()
                
                bar = MockBar()
                
                # Dataset Builder Update
                builder.on_bar(bar)
                
                # Simuliere AI-Prediction
                features = {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "price_change": bar.close - bar.open,
                    "rsi": 50 + i * 2,
                    "session": i
                }
                
                prediction = {
                    "action": ["BUY", "SELL", "HOLD"][i % 3],
                    "confidence": 0.6 + i * 0.05,
                    "reasoning": f"integrated_test_{i}"
                }
                
                # Feature Logger Update
                logger.log(
                    ts_ns=bar.ts_init,
                    instrument="EUR/USD",
                    features=features,
                    prediction=prediction,
                    confidence_score=0.7 + i * 0.02,
                    market_regime="testing"
                )
                
                print(f"   Session {i+1}: {prediction['action']} @ {prediction['confidence']:.2f}")
            
            # Export Dataset
            if builder.to_parquet("test_logs/integrated_dataset.parquet"):
                print("✅ Integriertes Dataset exportiert")
            
            # Zeige finale Statistiken
            logger_stats = logger.get_stats()
            builder_stats = builder.get_stats()
            
            print(f"📊 Integration Stats:")
            print(f"   Logger Entries: {logger_stats['total_entries_logged']}")
            print(f"   Dataset Entries: {builder_stats['total_entries']}")
            print(f"   Label Distribution: {builder_stats['label_distribution']}")
        
        print("✅ Integration Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Haupttest-Funktion für Enhanced Logging"""
    print("🚀 Enhanced Logging Components Test Suite")
    print("ChatGPT-Verbesserungen: Feature Logging + Dataset Builder")
    print("=" * 70)
    
    # Erstelle Test-Verzeichnisse
    Path("test_logs").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)
    
    success = True
    
    # Test 1: FeaturePredictionLogger
    print("\n📋 Test 1: FeaturePredictionLogger")
    if not test_feature_prediction_logger():
        success = False
    
    # Test 2: BarDatasetBuilder (Standalone)
    print("\n📋 Test 2: BarDatasetBuilder (Standalone)")
    if not test_bar_dataset_builder_standalone():
        success = False
    
    # Test 3: Integration
    print("\n📋 Test 3: Integration Test")
    if not test_integration():
        success = False
    
    print("\n" + "=" * 70)
    
    if success:
        print("🎉 Alle Enhanced Logging Tests erfolgreich!")
        print("\n📊 Generierte Dateien:")
        
        # Zeige generierte Dateien
        test_files = [
            "test_logs/standard_features.parquet",
            "test_logs/standalone_dataset.parquet",
            "test_logs/integrated_features.parquet",
            "test_logs/integrated_dataset.parquet"
        ]
        
        total_size = 0
        for file_path in test_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                total_size += size
                print(f"  ✅ {file_path} ({size:,} bytes)")
            else:
                print(f"  ❌ {file_path} (nicht gefunden)")
        
        print(f"\n📊 Gesamt-Dateigröße: {total_size:,} bytes")
        
        print("\n🎯 ChatGPT-Verbesserungen erfolgreich implementiert:")
        print("  ✅ FeaturePredictionLogger mit Parquet-Export")
        print("  ✅ Buffer-System für Performance-Optimierung")
        print("  ✅ BarDatasetBuilder mit Forward-Return-Labeling")
        print("  ✅ Polars-Integration für große Datasets")
        print("  ✅ Automatische Metadata-Generierung")
        print("  ✅ Rotating Logger für tägliche Dateien")
        
        print("\n🚀 Bereit für Phase 2: Enhanced Pattern Recognition!")
        
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)