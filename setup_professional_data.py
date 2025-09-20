#!/usr/bin/env python3
"""
Setup-Script für professionelle Datenquellen
Konfiguriert verschiedene Datenquellen für das AI-System
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

def setup_data_directories():
    """Erstelle Datenverzeichnisse"""
    
    print("📁 DATENVERZEICHNISSE ERSTELLEN")
    print("="*40)
    
    directories = [
        "./data/professional",
        "./data/professional/csv",
        "./data/professional/parquet",
        "./data/professional/EURUSD",
        "./data/professional/GBPUSD",
        "./data/professional/USDJPY",
        "./data/cache",
        "./data/backtest",
        "./data/live"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    print(f"\n✅ Alle Verzeichnisse erstellt!")

def create_sample_professional_data():
    """Erstelle Beispiel-Daten in verschiedenen Formaten"""
    
    print(f"\n📊 BEISPIEL-DATEN ERSTELLEN")
    print("="*30)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    base_prices = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 150.00}
    
    for symbol in symbols:
        print(f"\n📈 {symbol}:")
        
        # Generiere 3 Tage Daten
        for days_ago in range(3):
            date = datetime.now() - timedelta(days=days_ago)
            
            # Generiere Tick-Daten
            ticks = generate_realistic_ticks(
                symbol=symbol,
                date=date,
                base_price=base_prices[symbol],
                num_ticks=5000
            )
            
            df = pd.DataFrame(ticks)
            
            # Speichere in verschiedenen Formaten
            date_str = date.strftime('%Y%m%d')
            
            # Format 1: Standard CSV
            csv_file = f"./data/professional/{symbol}_{date_str}.csv"
            df.to_csv(csv_file, index=False)
            
            # Format 2: Parquet
            parquet_file = f"./data/professional/{symbol}_{date_str}.parquet"
            df.to_parquet(parquet_file, index=False)
            
            # Format 3: Symbol-Unterverzeichnis
            symbol_dir = Path(f"./data/professional/{symbol}")
            symbol_dir.mkdir(exist_ok=True)
            symbol_csv = symbol_dir / f"{date_str}.csv"
            df.to_csv(symbol_csv, index=False)
            
            print(f"   {date.strftime('%Y-%m-%d')}: {len(df)} Ticks → CSV, Parquet, Symbol-Dir")
    
    print(f"\n✅ Beispiel-Daten für {len(symbols)} Symbole erstellt!")

def generate_realistic_ticks(symbol: str, date: datetime, base_price: float, num_ticks: int = 5000):
    """Generiere realistische Tick-Daten"""
    
    ticks = []
    current_price = base_price
    
    # Basis-Zeitpunkt (Handelstag)
    base_time = date.replace(hour=8, minute=0, second=0, microsecond=0)
    
    for i in range(num_ticks):
        # Unregelmäßige Zeitabstände (exponential)
        time_offset = np.random.exponential(2.0)  # Ø 2 Sekunden
        timestamp = base_time + timedelta(seconds=time_offset * i)
        
        # Preis-Bewegung (Random Walk mit Trend)
        trend = 0.000001 * np.sin(i / 1000)  # Langfristiger Trend
        volatility = np.random.normal(0, 0.00005)  # Kurzfristige Volatilität
        
        price_change = trend + volatility
        current_price += price_change
        
        # Spread (abhängig von Volatilität)
        base_spread = 0.00015 if symbol == "EURUSD" else 0.00020
        volatility_factor = abs(volatility) * 10000
        spread = base_spread * (1 + volatility_factor)
        
        ask = current_price + spread/2
        bid = current_price - spread/2
        
        # Realistische Volume-Verteilung
        if np.random.random() < 0.6:  # 60% kleine Trades
            volume = np.random.uniform(0.1, 2.0)
        elif np.random.random() < 0.9:  # 30% mittlere Trades
            volume = np.random.uniform(2.0, 10.0)
        else:  # 10% große Trades
            volume = np.random.uniform(10.0, 100.0)
        
        # Ask/Bid Volume-Split
        ask_ratio = np.random.beta(2, 2)  # Beta-Verteilung für realistischen Split
        ask_volume = volume * ask_ratio
        bid_volume = volume * (1 - ask_ratio)
        
        ticks.append({
            'timestamp': timestamp,
            'ask': round(ask, 5),
            'bid': round(bid, 5),
            'volume': round(volume, 3),
            'ask_volume': round(ask_volume, 3),
            'bid_volume': round(bid_volume, 3),
            'spread': round(spread, 5)
        })
    
    return ticks

def test_data_loading():
    """Teste das Laden verschiedener Datenquellen"""
    
    print(f"\n🧪 DATENQUELLEN-TEST")
    print("="*25)
    
    # Test 1: Simulierte Daten
    print(f"\n1. Simulierte Daten:")
    config_sim = DukascopyConfig(
        use_real_data=False,
        local_data_path=None
    )
    
    connector_sim = DukascopyConnector(config_sim)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=2)
    
    df_sim = connector_sim.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=start_date,
        end_date=end_date,
        use_cache=False
    )
    
    print(f"   ✅ {len(df_sim)} Bars geladen (simuliert)")
    
    # Test 2: Lokale Profi-Daten
    print(f"\n2. Lokale Profi-Daten:")
    config_local = DukascopyConfig(
        use_real_data=False,
        local_data_path="./data/professional"
    )
    
    connector_local = DukascopyConnector(config_local)
    
    df_local = connector_local.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=start_date,
        end_date=end_date,
        use_cache=False
    )
    
    if not df_local.empty:
        print(f"   ✅ {len(df_local)} Bars geladen (lokal)")
    else:
        print(f"   ⚠️ Keine lokalen Daten gefunden")
    
    # Test 3: Echte Dukascopy-Daten
    print(f"\n3. Echte Dukascopy-Daten:")
    config_real = DukascopyConfig(
        use_real_data=True,
        local_data_path=None
    )
    
    connector_real = DukascopyConnector(config_real)
    
    # Teste mit älteren Daten (höhere Erfolgswahrscheinlichkeit)
    test_start = datetime.now() - timedelta(days=2)
    test_end = test_start + timedelta(hours=1)
    
    df_real = connector_real.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=test_start,
        end_date=test_end,
        use_cache=True
    )
    
    if not df_real.empty:
        print(f"   ✅ {len(df_real)} Bars geladen (Dukascopy)")
    else:
        print(f"   ⚠️ Keine Dukascopy-Daten verfügbar")
    
    # Vergleiche Datenquellen
    print(f"\n📊 VERGLEICH:")
    sources = [
        ("Simuliert", df_sim),
        ("Lokal", df_local),
        ("Dukascopy", df_real)
    ]
    
    for name, df in sources:
        if not df.empty:
            avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
            print(f"   {name:10}: {len(df):4} Bars, Ø Volume: {avg_volume:.2f}")
        else:
            print(f"   {name:10}: Keine Daten")

def create_configuration_examples():
    """Erstelle Konfigurationsbeispiele"""
    
    print(f"\n⚙️ KONFIGURATIONSBEISPIELE")
    print("="*30)
    
    examples = {
        "demo_gui_real_dukascopy.py": """
# Echte Dukascopy-Daten in der Demo-GUI aktivieren
config = DukascopyConfig(
    max_workers=8,
    cache_dir="./data/cache",
    use_real_data=True,  # ← Echte Dukascopy-Daten
    local_data_path=None
)
""",
        
        "demo_gui_professional.py": """
# Lokale professionelle Daten verwenden
config = DukascopyConfig(
    max_workers=4,
    cache_dir="./data/cache",
    use_real_data=False,
    local_data_path="./data/professional"  # ← Lokale Profi-Daten
)
""",
        
        "demo_gui_hybrid.py": """
# Hybrid: Lokale Daten mit Dukascopy-Fallback
config = DukascopyConfig(
    max_workers=8,
    cache_dir="./data/cache",
    use_real_data=True,  # Fallback zu Dukascopy
    local_data_path="./data/professional"  # Priorität: Lokale Daten
)
""",
        
        "production_config.py": """
# Produktions-Konfiguration
config = DukascopyConfig(
    max_workers=16,
    cache_dir="/data/market_data/cache",
    use_real_data=True,
    local_data_path="/data/market_data/professional",
    base_url="https://datafeed.dukascopy.com/datafeed"
)
"""
    }
    
    for filename, code in examples.items():
        print(f"\n📄 {filename}:")
        print(code)

def main():
    """Hauptfunktion"""
    
    print("🚀 PROFESSIONELLE DATENQUELLEN SETUP")
    print("="*50)
    
    # 1. Verzeichnisse erstellen
    setup_data_directories()
    
    # 2. Beispiel-Daten generieren
    create_sample_professional_data()
    
    # 3. Datenquellen testen
    test_data_loading()
    
    # 4. Konfigurationsbeispiele
    create_configuration_examples()
    
    print(f"\n🎉 SETUP ABGESCHLOSSEN!")
    print("="*25)
    
    print(f"""
✅ Datenverzeichnisse erstellt
✅ Beispiel-Daten generiert (3 Symbole, 3 Tage)
✅ Datenquellen getestet
✅ Konfigurationsbeispiele erstellt

🚀 NÄCHSTE SCHRITTE:

1. **Demo starten:**
   streamlit run demo_real_dukascopy.py --server.port 8504

2. **Eigene Profi-Daten hinzufügen:**
   - Kopiere CSV/Parquet-Dateien nach ./data/professional/
   - Unterstützte Formate: siehe demo_real_dukascopy.py

3. **In bestehender Demo-GUI aktivieren:**
   - Öffne demo_gui.py
   - Ändere DukascopyConfig zu use_real_data=True

4. **Produktions-Setup:**
   - Konfiguriere lokalen Daten-Pfad
   - Aktiviere echte Dukascopy-Daten
   - Optimiere Cache-Einstellungen
""")

if __name__ == "__main__":
    main()