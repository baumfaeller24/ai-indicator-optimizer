#!/usr/bin/env python3
"""
Script zum Aktivieren von echtem Tick-Volume
Zeigt wie man echte Dukascopy-Daten mit realem Volumen lädt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig
from datetime import datetime, timedelta

def enable_real_volume_demo():
    """Demo: Echtes Volumen aktivieren"""
    
    print("🎯 ECHTES TICK-VOLUME AKTIVIEREN")
    print("="*40)
    
    # Konfiguration für echte Daten
    config = DukascopyConfig(
        max_workers=8,
        cache_dir="./data/cache",
        use_real_data=True,  # ← ECHTE DATEN AKTIVIEREN!
        base_url="https://datafeed.dukascopy.com/datafeed"
    )
    
    connector = DukascopyConnector(config)
    
    print("✅ Dukascopy-Connector konfiguriert für echte Daten")
    print(f"📡 URL: {config.base_url}")
    print(f"🔄 Workers: {config.max_workers}")
    
    # Versuche echte Daten zu laden
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=2)  # Nur 2 Stunden für Test
    
    print(f"\n📊 Lade echte Tick-Daten...")
    print(f"   Zeitraum: {start_date} bis {end_date}")
    print(f"   Symbol: EUR/USD")
    print(f"   Timeframe: 100tick")
    
    try:
        # Versuche echte Daten zu laden
        ohlcv_df = connector.get_ohlcv_data(
            symbol="EUR/USD",
            timeframe="100tick",
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if not ohlcv_df.empty:
            print(f"✅ {len(ohlcv_df)} Bars mit echtem Volumen geladen!")
            
            # Analysiere echtes Volumen
            print(f"\n📈 ECHTES VOLUME-ANALYSE:")
            print(f"   Gesamt-Volumen: {ohlcv_df['volume'].sum():.2f} Lots")
            print(f"   Ø Volume/Bar: {ohlcv_df['volume'].mean():.2f} Lots")
            print(f"   Volume-Range: {ohlcv_df['volume'].min():.2f} - {ohlcv_df['volume'].max():.2f} Lots")
            
            # Volume-Variabilität (Indikator für echte Daten)
            volume_std = ohlcv_df['volume'].std()
            volume_mean = ohlcv_df['volume'].mean()
            variability = volume_std / volume_mean
            
            print(f"   Volume-Variabilität: {variability:.3f}")
            
            if variability > 0.1:
                print("✅ Hohe Volume-Variabilität → Echte Marktdaten!")
            else:
                print("⚠️ Niedrige Volume-Variabilität → Möglicherweise simuliert")
            
            # Erste 5 Bars anzeigen
            print(f"\n📊 ERSTE 5 BARS:")
            display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if 'tick_count' in ohlcv_df.columns:
                display_cols.append('tick_count')
            
            for i, row in ohlcv_df[display_cols].head().iterrows():
                print(f"   {i+1}. {row['timestamp']} | OHLC: {row['open']:.5f}-{row['high']:.5f}-{row['low']:.5f}-{row['close']:.5f} | Vol: {row['volume']:.1f}")
        
        else:
            print("❌ Keine echten Daten verfügbar - Fallback zu simulierten Daten")
            print("💡 Mögliche Gründe:")
            print("   - Dukascopy-Server nicht erreichbar")
            print("   - Keine Daten für gewählten Zeitraum")
            print("   - API-Limits erreicht")
    
    except Exception as e:
        print(f"❌ Fehler beim Laden echter Daten: {e}")
        print("💡 Fallback zu simulierten Daten aktiv")
    
    print(f"\n🔧 KONFIGURATION FÜR ECHTE DATEN:")
    print(f"""
# In der demo_gui.py oder anderen Scripts:
config = DukascopyConfig(
    max_workers=8,
    cache_dir="./data/cache",
    use_real_data=True,  # ← Echte Daten aktivieren
    base_url="https://datafeed.dukascopy.com/datafeed"
)

# Dann normale Verwendung:
connector = DukascopyConnector(config)
ohlcv_df = connector.get_ohlcv_data(
    symbol="EUR/USD",
    timeframe="100tick",
    start_date=start_date,
    end_date=end_date,
    use_cache=True
)

# Echtes Volumen ist dann in ohlcv_df['volume'] verfügbar!
""")

if __name__ == "__main__":
    enable_real_volume_demo()