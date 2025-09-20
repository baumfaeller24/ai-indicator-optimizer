#!/usr/bin/env python3
"""
Debug Script für OHLC Chart-Darstellung
Vergleicht Sample Data vs. Dukascopy Data
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig
from datetime import datetime, timedelta

def create_debug_chart(df, title):
    """Erstellt Debug-Chart für OHLC-Daten"""
    
    print(f"\n🔍 DEBUG: {title}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n📊 Erste 3 Zeilen:")
    print(df.head(3))
    print("\n📊 OHLC-Statistiken:")
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        print(f"Open Range: {df['open'].min():.5f} - {df['open'].max():.5f}")
        print(f"High Range: {df['high'].min():.5f} - {df['high'].max():.5f}")
        print(f"Low Range: {df['low'].min():.5f} - {df['low'].max():.5f}")
        print(f"Close Range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        
        # Prüfe OHLC-Logik
        invalid_ohlc = 0
        for i, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                invalid_ohlc += 1
        
        print(f"Invalid OHLC Bars: {invalid_ohlc}/{len(df)}")
        
        # Docht-Längen
        upper_wicks = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wicks = df[['open', 'close']].min(axis=1) - df['low']
        
        print(f"Upper Wicks: Avg={upper_wicks.mean():.6f}, Max={upper_wicks.max():.6f}")
        print(f"Lower Wicks: Avg={lower_wicks.mean():.6f}, Max={lower_wicks.max():.6f}")
    
    # Erstelle Chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=title,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_line_width=2,
            decreasing_line_width=2
        )
    )
    
    fig.update_layout(
        title=f"{title} - OHLC Debug Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig

def main():
    """Hauptfunktion für OHLC Debug"""
    
    print("🔍 OHLC CHART DEBUG ANALYSIS")
    print("="*50)
    
    # 1. Sample Data (funktioniert)
    print("\n📊 1. SAMPLE DATA (Generate Sample Data)")
    
    # Simuliere Sample Data wie in demo_gui.py
    import numpy as np
    
    base_price = 1.1000
    sample_data = []
    
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        
        # Simuliere Marktbewegung
        trend = 0.00001 * np.sin(i / 10)
        volatility = np.random.normal(0, 0.0005)
        
        price_change = trend + volatility
        open_price = base_price + price_change
        
        # OHLC mit echten Dochten
        high_offset = abs(np.random.normal(0, 0.0003))
        low_offset = abs(np.random.normal(0, 0.0003))
        close_change = np.random.normal(0, 0.0002)
        
        high_price = open_price + high_offset
        low_price = open_price - low_offset
        close_price = open_price + close_change
        
        # Stelle sicher dass OHLC-Logik stimmt
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        sample_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(800, 1200)
        })
        
        base_price = close_price
    
    sample_df = pd.DataFrame(sample_data)
    sample_fig = create_debug_chart(sample_df, "Sample Data")
    
    # 2. Dukascopy Data (Problem)
    print("\n📊 2. DUKASCOPY DATA (Download from Dukascopy)")
    
    try:
        # Lade echte Dukascopy-Daten
        config = DukascopyConfig(
            max_workers=4,
            cache_dir="./data/cache",
            use_real_data=False
        )
        
        connector = DukascopyConnector(config)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        dukascopy_df = connector.get_ohlcv_data(
            symbol="EUR/USD",
            timeframe="100tick",
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if not dukascopy_df.empty:
            dukascopy_fig = create_debug_chart(dukascopy_df, "Dukascopy Data")
            
            # Speichere Charts
            sample_fig.write_html("debug_sample_chart.html")
            dukascopy_fig.write_html("debug_dukascopy_chart.html")
            
            print(f"\n✅ Charts gespeichert:")
            print(f"   - debug_sample_chart.html")
            print(f"   - debug_dukascopy_chart.html")
            
            # Vergleiche die Daten
            print(f"\n🔍 VERGLEICH:")
            print(f"Sample Data: {len(sample_df)} bars")
            print(f"Dukascopy Data: {len(dukascopy_df)} bars")
            
            # Prüfe ob Dukascopy-Daten echte Dochte haben
            if 'high' in dukascopy_df.columns and 'low' in dukascopy_df.columns:
                duka_upper_wicks = dukascopy_df['high'] - dukascopy_df[['open', 'close']].max(axis=1)
                duka_lower_wicks = dukascopy_df[['open', 'close']].min(axis=1) - dukascopy_df['low']
                
                print(f"\n📊 DUKASCOPY DOCHT-ANALYSE:")
                print(f"Upper Wicks > 0: {(duka_upper_wicks > 0.00001).sum()}/{len(dukascopy_df)}")
                print(f"Lower Wicks > 0: {(duka_lower_wicks > 0.00001).sum()}/{len(dukascopy_df)}")
                
                if (duka_upper_wicks > 0.00001).sum() == 0 and (duka_lower_wicks > 0.00001).sum() == 0:
                    print("❌ PROBLEM GEFUNDEN: Dukascopy-Daten haben KEINE Dochte!")
                    print("   Alle High/Low Werte sind identisch mit Open/Close")
                else:
                    print("✅ Dukascopy-Daten haben korrekte Dochte")
        else:
            print("❌ Keine Dukascopy-Daten erhalten")
            
    except Exception as e:
        print(f"❌ Dukascopy-Test fehlgeschlagen: {e}")
    
    print(f"\n🎯 FAZIT:")
    print(f"Öffne die HTML-Dateien im Browser um die Charts zu vergleichen!")

if __name__ == "__main__":
    main()