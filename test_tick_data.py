#!/usr/bin/env python3
"""
Test für Tick-Daten Generierung
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

def test_tick_generation():
    """Testet Tick-Daten Generierung"""
    
    print("🧪 Testing Tick Data Generation")
    print("=" * 50)
    
    # Connector erstellen
    config = DukascopyConfig(max_workers=4)
    connector = DukascopyConnector(config)
    
    # 2 Tage EUR/USD Daten
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    print(f"📅 Loading data from {start_date.date()} to {end_date.date()}")
    
    # OHLCV für 100-Tick Bars
    ohlcv_df = connector.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=start_date,
        end_date=end_date,
        use_cache=False
    )
    
    print(f"📊 Results:")
    print(f"   Total Bars: {len(ohlcv_df)}")
    
    if not ohlcv_df.empty:
        print(f"   Date Range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
        print(f"   Price Range: {ohlcv_df['close'].min():.5f} - {ohlcv_df['close'].max():.5f}")
        
        # Zeige erste paar Bars
        print(f"\n📈 First 5 Bars:")
        print(ohlcv_df.head().to_string())
        
        # Erwartete vs. tatsächliche Anzahl
        expected_ticks_per_day = 100000  # Mittelwert
        expected_bars_per_day = expected_ticks_per_day // 100  # 100-Tick Bars
        expected_total_bars = expected_bars_per_day * 2  # 2 Tage
        
        print(f"\n🎯 Analysis:")
        print(f"   Expected ~{expected_total_bars} bars (2 days × ~{expected_bars_per_day} bars/day)")
        print(f"   Actual: {len(ohlcv_df)} bars")
        print(f"   Ratio: {len(ohlcv_df) / expected_total_bars * 100:.1f}% of expected")
        
        if 'tick_count' in ohlcv_df.columns:
            avg_ticks = ohlcv_df['tick_count'].mean()
            print(f"   Avg ticks per bar: {avg_ticks:.1f}")
    
    else:
        print("❌ No data generated!")

if __name__ == "__main__":
    test_tick_generation()