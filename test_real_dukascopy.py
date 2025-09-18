#!/usr/bin/env python3
"""
Test für echte Dukascopy-Daten
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

def test_real_dukascopy():
    """Testet echte Dukascopy-Daten"""
    
    print("🌐 Testing Real Dukascopy Data")
    print("=" * 50)
    
    # Connector mit echten Daten
    config = DukascopyConfig(
        max_workers=4,
        use_real_data=True
    )
    connector = DukascopyConnector(config)
    
    # Teste mit einem Werktag (nicht Wochenende)
    test_date = datetime(2025, 9, 16)  # Montag
    
    print(f"📅 Testing with date: {test_date.date()}")
    print("🔄 Attempting to download real Dukascopy data...")
    
    # Versuche echte Daten zu laden
    ohlcv_df = connector.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=test_date,
        end_date=test_date + timedelta(hours=2),  # Nur 2 Stunden für Test
        use_cache=False
    )
    
    print(f"📊 Results:")
    print(f"   Total Bars: {len(ohlcv_df)}")
    
    if not ohlcv_df.empty:
        print(f"   ✅ Real data loaded successfully!")
        print(f"   Date Range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
        print(f"   Price Range: {ohlcv_df['close'].min():.5f} - {ohlcv_df['close'].max():.5f}")
        
        # Zeige erste paar Bars
        print(f"\n📈 First 3 Bars:")
        print(ohlcv_df.head(3).to_string())
        
    else:
        print("❌ No real data available (likely fell back to simulated)")
        print("💡 This is normal - Dukascopy data may not be available for all dates")

if __name__ == "__main__":
    test_real_dukascopy()