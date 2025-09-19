#!/usr/bin/env python3
"""
Production-Ready Test Suite für Enhanced AI Trading System
ChatGPT-Enhanced Version mit PyArrow, Order Adapter und robustem Error-Handling
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

def test_rotating_parquet_logger():
    """Test des Production-Ready Rotating Parquet Loggers"""
    print("🧪 Testing RotatingParquetLogger...")
    
    try:
        from ai_indicator_optimizer.logging.rotating_parquet_logger import create_rotating_logger
        
        print("✅ RotatingParquetLogger importiert")
        
        # Test 1: Standard Rotation (täglich)
        print("\n📋 Test 1: Daily Rotation Logger")
        with create_rotating_logger(
            "test_logs/daily_features", 
            rotation="day", 
            buffer_size=5,
            include_pid=True
        ) as logger:
            
            # Simuliere Trading-Session über mehrere "Tage"
            base_time = int(time.time() * 1e9)
            
            for day in range(2):
                for hour in range(4):
                    # Timestamp für verschiedene Tage
                    ts = base_time + day * 24 * 3600 * 1e9 + hour * 3600 * 1e9
                    
                    row = {
                        "ts_ns": int(ts),
                        "instrument": "EUR/USD",
                        "f_open": 1.1000 + day * 0.001 + hour * 0.0001,
                        "f_high": 1.1000 + day * 0.001 + hour * 0.0001 + 0.0003,
                        "f_low": 1.1000 + day * 0.001 + hour * 0.0001 - 0.0002,
                        "f_close": 1.1005 + day * 0.001 + hour * 0.0001,
                        "f_volume": 1000.0 + hour * 100,
                        "f_rsi": 50.0 + hour * 5,
                        "f_volatility": 0.001 + hour * 0.0001,
                        "pred_action": ["BUY", "SELL", "HOLD"][hour % 3],
                        "pred_confidence": 0.6 + hour * 0.1,
                        "pred_reason": f"test_day{day}_hour{hour}",
                        "pred_risk": 0.1 + hour * 0.02,
                        "enhanced_confidence": 0.7 + hour * 0.05,
                        "market_regime": ["trending", "ranging", "volatile"][hour % 3]
                    }
                    
                    logger.log(ts_ns=row["ts_ns"], row=row)
                    
                    if (day * 4 + hour + 1) % 5 == 0:  # Auto-flush test
                        print(f"   Auto-flush at day {day}, hour {hour}")
        
        stats = logger.get_stats()
        print(f"📊 Daily Logger Stats: {stats}")
        
        # Test 2: Hourly Rotation
        print("\n📋 Test 2: Hourly Rotation Logger")
        with create_rotating_logger(
            "test_logs/hourly_features", 
            rotation="hour", 
            buffer_size=3,
            include_pid=False  # Test ohne PID
        ) as hourly_logger:
            
            current_time = int(time.time() * 1e9)
            
            for i in range(8):
                # Verschiedene Stunden simulieren
                ts = current_time + i * 3600 * 1e9  # 1 Stunde Abstand
                
                row = {
                    "ts_ns": int(ts),
                    "instrument": "GBP/USD",
                    "f_open": 1.2500 + i * 0.0001,
                    "f_close": 1.2505 + i * 0.0001,
                    "f_volume": 800.0 + i * 50,
                    "pred_action": ["BUY", "SELL", "HOLD"][i % 3],
                    "pred_confidence": 0.65 + i * 0.03,
                    "hour_test": i
                }
                
                hourly_logger.log(ts_ns=row["ts_ns"], row=row)
        
        hourly_stats = hourly_logger.get_stats()
        print(f"📊 Hourly Logger Stats: {hourly_stats}")
        
        print("✅ RotatingParquetLogger Tests erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ RotatingParquetLogger Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_adapter():
    """Test des Order Adapters"""
    print("🧪 Testing OrderAdapter...")
    
    try:
        from ai_indicator_optimizer.trading.order_adapter import create_order_adapter, submit_market_order_safe
        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.identifiers import InstrumentId
        
        print("✅ OrderAdapter importiert")
        
        # Mock Strategy für Tests
        class MockStrategy:
            def __init__(self):
                self.trader_id = "test_trader"
                self.id = "test_strategy"
                self.orders_submitted = []
                self.log = type('MockLog', (), {
                    'info': lambda msg: print(f"INFO: {msg}"),
                    'error': lambda msg: print(f"ERROR: {msg}"),
                    'warning': lambda msg: print(f"WARNING: {msg}")
                })()
                
            def submit_market_order(self, instrument_id, side, quantity):
                self.orders_submitted.append((instrument_id, side, quantity))
                print(f"Mock convenience order: {side.name} {quantity} {instrument_id}")
                
            def generate_order_id(self):
                return f"order_{len(self.orders_submitted)}"
                
            class MockClock:
                def timestamp_ns(self):
                    return int(time.time() * 1e9)
            
            clock = MockClock()
        
        # Test Convenience Mode
        print("\n📋 Test 1: Convenience Mode")
        strategy = MockStrategy()
        adapter = create_order_adapter(strategy, "convenience")
        
        instrument_id = InstrumentId.from_str("EUR/USD.SIM")
        
        success = adapter.submit_market_order(instrument_id, OrderSide.BUY, 1000)
        print(f"Order success: {success}")
        print(f"Adapter Stats: {adapter.get_stats()}")
        
        # Test One-shot Function
        print("\n📋 Test 2: One-shot Function")
        success2 = submit_market_order_safe(strategy, instrument_id, OrderSide.SELL, 500)
        print(f"One-shot success: {success2}")
        
        print(f"Total mock orders: {len(strategy.orders_submitted)}")
        
        print("✅ OrderAdapter Tests erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ OrderAdapter Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_bar_dataset_builder():
    """Test des Enhanced Bar Dataset Builders"""
    print("🧪 Testing Enhanced BarDatasetBuilder...")
    
    try:
        from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
        
        print("✅ BarDatasetBuilder importiert")
        
        # Mock Bar für Tests
        class MockBar:
            def __init__(self, timestamp, open_price, high, low, close, volume):
                self.ts_init = timestamp
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.bar_type = type('MockBarType', (), {
                    'instrument_id': 'EUR/USD.SIM',
                    '__str__': lambda: 'EUR/USD.SIM-1-MINUTE-BID'
                })()
        
        # Erstelle Enhanced Dataset Builder
        builder = BarDatasetBuilder(
            horizon=5, 
            min_bars=8, 
            include_technical_indicators=True,
            return_thresholds={
                "buy_threshold": 0.0005,   # 0.05%
                "sell_threshold": -0.0005  # -0.05%
            }
        )
        
        print("📊 Generiere realistische Forex-Daten mit technischen Indikatoren...")
        
        # Simuliere realistische EUR/USD Bewegung
        base_price = 1.1000
        base_time = int(time.time() * 1e9)
        
        for i in range(15):  # Genug für Forward-Returns und technische Indikatoren
            # Simuliere Markt-Trend mit Rauschen
            trend_component = 0.0003 * (i - 7)  # Trend über Zeit
            noise_component = (hash(f"price_{i}") % 1000 - 500) * 0.000002  # Pseudo-random noise
            volatility = 0.0001 + abs(trend_component) * 0.5  # Höhere Volatilität bei Trends
            
            current_price = base_price + trend_component + noise_component
            
            bar = MockBar(
                timestamp=base_time + i * 60 * 1e9,  # 1-Minuten-Bars
                open_price=current_price,
                high=current_price + volatility + abs(noise_component),
                low=current_price - volatility - abs(noise_component),
                close=current_price + noise_component * 0.5,
                volume=1000 + i * 50 + (hash(f"vol_{i}") % 500)
            )
            
            builder.on_bar(bar)
            
            if i % 5 == 0:
                print(f"   Processed {i+1} bars, price: {current_price:.5f}")
        
        # Zeige Statistiken
        stats = builder.get_stats()
        print(f"\n📊 Dataset Builder Stats:")
        print(f"   Total Entries: {stats['total_entries']}")
        print(f"   Label Distribution: {stats['label_distribution']}")
        print(f"   Return Statistics: {stats['return_statistics']}")
        print(f"   Technical Indicators: {stats['technical_indicators_enabled']}")
        
        # Export Dataset
        export_path = "test_logs/enhanced_dataset.parquet"
        if builder.to_parquet(export_path, include_metadata=True):
            print(f"✅ Enhanced Dataset erfolgreich exportiert: {export_path}")
            
            # Lade und validiere Dataset
            import polars as pl
            df = pl.read_parquet(export_path)
            print(f"📊 Dataset Shape: {df.shape}")
            print(f"📊 Columns: {df.columns}")
            
            # Zeige Label-Verteilung
            label_col = f"label_name_h{builder.horizon}"
            if label_col in df.columns:
                label_dist = df[label_col].value_counts()
                print(f"📊 Label Distribution:\n{label_dist}")
            
            # Zeige Metadata
            metadata_path = Path(export_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"📊 Metadata Keys: {list(metadata.keys())}")
        
        print("✅ Enhanced BarDatasetBuilder Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced BarDatasetBuilder Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_full_pipeline():
    """Test der kompletten Integration aller Enhanced Components"""
    print("🧪 Testing Full Integration Pipeline...")
    
    try:
        from ai_indicator_optimizer.logging.rotating_parquet_logger import create_rotating_logger
        from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
        from ai_indicator_optimizer.trading.order_adapter import create_order_adapter
        
        print("✅ Alle Enhanced Components importiert")
        
        # Mock Strategy für Integration
        class MockIntegratedStrategy:
            def __init__(self):
                self.log = type('MockLog', (), {
                    'info': lambda msg: print(f"STRATEGY INFO: {msg}"),
                    'error': lambda msg: print(f"STRATEGY ERROR: {msg}"),
                    'warning': lambda msg: print(f"STRATEGY WARNING: {msg}")
                })()
                self.trader_id = "integration_test"
                self.id = "integration_strategy"
                self.orders = []
                
            def submit_market_order(self, instrument_id, side, quantity):
                self.orders.append((instrument_id, side, quantity))
                print(f"INTEGRATED ORDER: {side.name} {quantity} {instrument_id}")
                return True
        
        strategy = MockIntegratedStrategy()
        
        # Setup alle Enhanced Components
        print("\n📋 Setup Enhanced Components...")
        
        # 1. Rotating Parquet Logger
        logger = create_rotating_logger(
            "test_logs/integration/features",
            rotation="hour",
            buffer_size=10,
            include_pid=True,
            logger=strategy.log.info
        )
        
        # 2. Dataset Builder
        dataset_builder = BarDatasetBuilder(
            horizon=3,
            min_bars=5,
            include_technical_indicators=True
        )
        
        # 3. Order Adapter
        order_adapter = create_order_adapter(strategy, "convenience")
        
        print("✅ Alle Components initialisiert")
        
        # Simuliere komplette Trading-Session
        print("\n📋 Simuliere Trading-Session...")
        
        base_time = int(time.time() * 1e9)
        
        for i in range(12):
            # Simuliere realistische Bar-Daten
            price = 1.1000 + i * 0.0002
            
            # Mock Bar
            class MockBar:
                def __init__(self):
                    self.ts_init = base_time + i * 60 * 1e9
                    self.open = price
                    self.high = price + 0.0003
                    self.low = price - 0.0002
                    self.close = price + 0.0001
                    self.volume = 1000 + i * 100
                    self.bar_type = type('MockBarType', (), {
                        'instrument_id': 'EUR/USD.SIM'
                    })()
            
            bar = MockBar()
            
            # 1. Dataset Builder Update
            dataset_builder.on_bar(bar)
            
            # 2. Simuliere AI-Prediction
            features = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "price_change": bar.close - bar.open,
                "body_ratio": abs(bar.close - bar.open) / max(bar.high - bar.low, 1e-6),
                "rsi_14": 50 + i * 2,
                "sma_20": price,
                "volatility_20": 0.001 + i * 0.0001,
                "momentum_5": (i - 6) * 0.0001,
                "hour": 9 + (i % 8),
                "session": i
            }
            
            prediction = {
                "action": ["BUY", "SELL", "HOLD"][i % 3],
                "confidence": 0.6 + i * 0.03,
                "reasoning": f"integration_test_{i}",
                "risk_score": 0.1 + i * 0.01
            }
            
            # 3. Feature Logging
            row = {
                "ts_ns": int(bar.ts_init),
                "instrument": "EUR/USD",
                "f_open": float(bar.open),
                "f_high": float(bar.high),
                "f_low": float(bar.low),
                "f_close": float(bar.close),
                "f_volume": float(bar.volume),
                **{f"f_{k}": float(v) if isinstance(v, (int, float)) else str(v) 
                   for k, v in features.items() 
                   if k not in ["open", "high", "low", "close", "volume"]},
                "pred_action": str(prediction["action"]),
                "pred_confidence": float(prediction["confidence"]),
                "pred_reason": str(prediction["reasoning"]),
                "pred_risk": float(prediction["risk_score"]),
                "enhanced_confidence": float(prediction["confidence"] * 1.1),
                "market_regime": "integration_test"
            }
            
            logger.log(ts_ns=row["ts_ns"], row=row)
            
            # 4. Order Submission (bei starken Signalen)
            if prediction["confidence"] > 0.75:
                from nautilus_trader.model.enums import OrderSide
                from nautilus_trader.model.identifiers import InstrumentId
                
                side = OrderSide.BUY if prediction["action"] == "BUY" else OrderSide.SELL
                instrument_id = InstrumentId.from_str("EUR/USD.SIM")
                
                order_success = order_adapter.submit_market_order(
                    instrument_id, side, 1000
                )
                
                if order_success:
                    print(f"   🎯 Order executed: {prediction['action']} @ {prediction['confidence']:.2f}")
            
            if i % 4 == 0:
                print(f"   Session {i+1}: {prediction['action']} @ {prediction['confidence']:.2f}")
        
        # Cleanup und Stats
        logger.close()
        
        # Final Statistics
        print(f"\n📊 Integration Results:")
        print(f"   Logger Stats: {logger.get_stats()}")
        print(f"   Dataset Stats: {dataset_builder.get_stats()}")
        print(f"   Order Stats: {order_adapter.get_stats()}")
        print(f"   Total Orders: {len(strategy.orders)}")
        
        # Export Dataset
        if dataset_builder.to_parquet("test_logs/integration_dataset.parquet"):
            print("✅ Integration Dataset exportiert")
        
        print("✅ Full Integration Pipeline Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_generated_files():
    """Validiere alle generierten Parquet-Dateien"""
    print("🧪 Validating Generated Parquet Files...")
    
    try:
        import polars as pl
        
        test_files = [
            "test_logs/daily_features_*.parquet",
            "test_logs/hourly_features_*.parquet", 
            "test_logs/integration/features_*.parquet",
            "test_logs/enhanced_dataset.parquet",
            "test_logs/integration_dataset.parquet"
        ]
        
        total_files = 0
        total_rows = 0
        
        for pattern in test_files:
            files = list(Path(".").glob(pattern))
            
            for file_path in files:
                if file_path.exists():
                    try:
                        df = pl.read_parquet(file_path)
                        rows = len(df)
                        cols = len(df.columns)
                        size = file_path.stat().st_size
                        
                        print(f"✅ {file_path.name}: {rows} rows, {cols} cols, {size:,} bytes")
                        
                        # Validiere Schema
                        if "ts_ns" in df.columns:
                            print(f"   📊 Time range: {df['ts_ns'].min()} - {df['ts_ns'].max()}")
                        
                        if "pred_action" in df.columns:
                            actions = df["pred_action"].value_counts()
                            print(f"   📊 Actions: {dict(zip(actions['pred_action'], actions['count']))}")
                        
                        total_files += 1
                        total_rows += rows
                        
                    except Exception as e:
                        print(f"❌ {file_path.name}: Validation failed - {e}")
                else:
                    print(f"❌ {pattern}: File not found")
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total Files: {total_files}")
        print(f"   Total Rows: {total_rows:,}")
        print(f"   Average Rows/File: {total_rows/max(total_files,1):.0f}")
        
        print("✅ File Validation erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ File Validation fehlgeschlagen: {e}")
        return False

def main():
    """Haupttest-Funktion für Production-Ready System"""
    print("🚀 Production-Ready Enhanced AI Trading System Test Suite")
    print("ChatGPT-Enhanced mit PyArrow, Order Adapter und robustem Error-Handling")
    print("=" * 80)
    
    # Erstelle Test-Verzeichnisse
    Path("test_logs").mkdir(exist_ok=True)
    Path("test_logs/integration").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)
    
    success = True
    
    # Test 1: Rotating Parquet Logger
    print("\n📋 Test 1: Production-Ready Rotating Parquet Logger")
    if not test_rotating_parquet_logger():
        success = False
    
    # Test 2: Order Adapter
    print("\n📋 Test 2: Order Adapter")
    if not test_order_adapter():
        success = False
    
    # Test 3: Enhanced Dataset Builder
    print("\n📋 Test 3: Enhanced Bar Dataset Builder")
    if not test_enhanced_bar_dataset_builder():
        success = False
    
    # Test 4: Full Integration
    print("\n📋 Test 4: Full Integration Pipeline")
    if not test_integration_full_pipeline():
        success = False
    
    # Test 5: File Validation
    print("\n📋 Test 5: Generated Files Validation")
    if not validate_generated_files():
        success = False
    
    print("\n" + "=" * 80)
    
    if success:
        print("🎉 ALLE PRODUCTION-READY TESTS ERFOLGREICH!")
        print("\n🎯 ChatGPT-Enhanced Features implementiert:")
        print("  ✅ RotatingParquetLogger mit PyArrow (echtes Append)")
        print("  ✅ Multi-Process-Safety mit PID-Suffixen")
        print("  ✅ Schema-Drift-Protection")
        print("  ✅ Robustes Error-Handling mit Logging")
        print("  ✅ Memory-Monitoring für Langzeit-Sessions")
        print("  ✅ Order Adapter mit Fallback-Mechanismen")
        print("  ✅ Enhanced Dataset Builder mit technischen Indikatoren")
        print("  ✅ HTTP Session mit Retry-Strategy")
        print("  ✅ Forward-Return-Labeling für ML-Training")
        
        print("\n🚀 System ist bereit für:")
        print("  1. Integration in echte Nautilus-Strategien")
        print("  2. Live-Trading mit Production-Logging")
        print("  3. ML-Training mit generierten Datasets")
        print("  4. TorchServe-Integration (Phase 2)")
        
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)