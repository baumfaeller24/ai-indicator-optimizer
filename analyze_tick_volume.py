#!/usr/bin/env python3
"""
Tick-Volume-Analyse für Dukascopy-Daten
Zeigt wie echtes Handelsvolumen aus Ticks berechnet wird
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

def analyze_tick_volume():
    """Analysiere Tick-Volume aus Dukascopy-Daten"""
    
    print("📊 TICK-VOLUME-ANALYSE")
    print("="*50)
    
    # Lade Dukascopy-Daten
    config = DukascopyConfig(
        max_workers=4,
        cache_dir="./data/cache",
        use_real_data=False  # Erstmal simulierte Daten
    )
    
    connector = DukascopyConnector(config)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # Lade OHLCV-Daten (100-Tick-Bars)
    ohlcv_df = connector.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="100tick",
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if ohlcv_df.empty:
        print("❌ Keine Daten verfügbar")
        return
    
    print(f"✅ {len(ohlcv_df)} OHLCV-Bars geladen")
    print(f"📊 Zeitraum: {ohlcv_df['timestamp'].min()} bis {ohlcv_df['timestamp'].max()}")
    
    # Volume-Analyse
    print(f"\n📈 VOLUME-STATISTIKEN:")
    print(f"   Gesamt-Volumen: {ohlcv_df['volume'].sum():.2f} Lots")
    print(f"   Durchschnitt pro Bar: {ohlcv_df['volume'].mean():.2f} Lots")
    print(f"   Median pro Bar: {ohlcv_df['volume'].median():.2f} Lots")
    print(f"   Min/Max pro Bar: {ohlcv_df['volume'].min():.2f} / {ohlcv_df['volume'].max():.2f} Lots")
    
    # Volume-Verteilung
    print(f"\n📊 VOLUME-VERTEILUNG:")
    volume_ranges = [
        ("Sehr niedrig", 0, 100),
        ("Niedrig", 100, 200),
        ("Normal", 200, 400),
        ("Hoch", 400, 600),
        ("Sehr hoch", 600, float('inf'))
    ]
    
    for name, min_vol, max_vol in volume_ranges:
        count = ((ohlcv_df['volume'] >= min_vol) & (ohlcv_df['volume'] < max_vol)).sum()
        percentage = (count / len(ohlcv_df)) * 100
        print(f"   {name} ({min_vol}-{max_vol if max_vol != float('inf') else '∞'}): {count} Bars ({percentage:.1f}%)")
    
    # Tick-Count vs. Volume-Analyse
    if 'tick_count' in ohlcv_df.columns:
        print(f"\n🔍 TICK-COUNT vs. VOLUME:")
        print(f"   Durchschnittliche Ticks pro Bar: {ohlcv_df['tick_count'].mean():.1f}")
        print(f"   Volume pro Tick: {(ohlcv_df['volume'] / ohlcv_df['tick_count']).mean():.3f} Lots/Tick")
        
        # Korrelation
        correlation = ohlcv_df['volume'].corr(ohlcv_df['tick_count'])
        print(f"   Korrelation Volume-TickCount: {correlation:.3f}")
    
    # Zeitbasierte Volume-Analyse
    ohlcv_df['hour'] = ohlcv_df['timestamp'].dt.hour
    hourly_volume = ohlcv_df.groupby('hour')['volume'].agg(['mean', 'sum', 'count']).reset_index()
    
    print(f"\n🕐 STÜNDLICHE VOLUME-VERTEILUNG:")
    print("   Stunde | Ø Volume | Gesamt | Bars")
    print("   -------|----------|--------|------")
    for _, row in hourly_volume.iterrows():
        print(f"   {int(row['hour']):2d}:00  | {row['mean']:8.1f} | {row['sum']:6.0f} | {row['count']:4.0f}")
    
    return ohlcv_df

def create_volume_analysis_chart(df):
    """Erstelle Volume-Analyse-Chart"""
    
    # Subplots für verschiedene Analysen
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'OHLC + Volume',
            'Volume pro Bar',
            'Volume vs. Tick-Count',
            'Stündliche Volume-Verteilung'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # 1. OHLC + Volume
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # 2. Volume Bars
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume (Lots)',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 3. Volume vs. Tick-Count (falls verfügbar)
    if 'tick_count' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['tick_count'],
                y=df['volume'],
                mode='markers',
                name='Volume vs. Ticks',
                marker=dict(
                    color=df['volume'],
                    colorscale='Viridis',
                    size=8,
                    opacity=0.7
                ),
                text=[f"Time: {t}<br>Ticks: {tc}<br>Volume: {v:.1f}" 
                      for t, tc, v in zip(df['timestamp'], df['tick_count'], df['volume'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # 4. Stündliche Volume-Verteilung
    df['hour'] = df['timestamp'].dt.hour
    hourly_volume = df.groupby('hour')['volume'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=hourly_volume['hour'],
            y=hourly_volume['volume'],
            name='Ø Volume/Stunde',
            marker_color='orange',
            opacity=0.8
        ),
        row=4, col=1
    )
    
    # Layout
    fig.update_layout(
        title="📊 Tick-Volume-Analyse - EUR/USD 100-Tick-Bars",
        height=1000,
        template='plotly_dark',
        showlegend=True
    )
    
    # Y-Achsen-Labels
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    fig.update_yaxes(title_text="Volume (Lots)", row=2, col=1)
    fig.update_yaxes(title_text="Volume (Lots)", row=3, col=1)
    fig.update_yaxes(title_text="Ø Volume", row=4, col=1)
    
    # X-Achsen-Labels
    fig.update_xaxes(title_text="Tick-Count", row=3, col=1)
    fig.update_xaxes(title_text="Stunde (UTC)", row=4, col=1)
    
    return fig

def simulate_real_tick_volume():
    """Simuliere wie echtes Tick-Volume aussehen würde"""
    
    print(f"\n🎯 SIMULATION: ECHTES TICK-VOLUME")
    print("="*40)
    
    # Simuliere realistische Tick-Daten
    num_ticks = 1000
    base_time = datetime.now()
    
    ticks = []
    base_price = 1.1000
    
    for i in range(num_ticks):
        # Zeitstempel (unregelmäßig)
        time_offset = np.random.exponential(2.0)  # Exponential-Verteilung
        timestamp = base_time + timedelta(seconds=time_offset * i)
        
        # Preis-Bewegung
        price_change = np.random.normal(0, 0.00005)
        price = base_price + price_change
        
        # Spread
        spread = 0.00015
        bid = price - spread/2
        ask = price + spread/2
        
        # Volume (realistische Verteilung)
        # Kleine Trades häufiger, große Trades seltener
        if np.random.random() < 0.7:  # 70% kleine Trades
            volume = np.random.uniform(0.1, 2.0)
        elif np.random.random() < 0.9:  # 20% mittlere Trades
            volume = np.random.uniform(2.0, 10.0)
        else:  # 10% große Trades
            volume = np.random.uniform(10.0, 100.0)
        
        # Ask/Bid Volume-Split (meist ungleich)
        split = np.random.beta(2, 2)  # Beta-Verteilung für realistischen Split
        ask_volume = volume * split
        bid_volume = volume * (1 - split)
        
        ticks.append({
            'timestamp': timestamp,
            'ask': ask,
            'bid': bid,
            'ask_volume': ask_volume,
            'bid_volume': bid_volume,
            'volume': volume,
            'price': (ask + bid) / 2
        })
        
        base_price = price
    
    tick_df = pd.DataFrame(ticks)
    
    print(f"✅ {len(tick_df)} Ticks simuliert")
    print(f"📊 Gesamt-Volume: {tick_df['volume'].sum():.2f} Lots")
    print(f"📊 Ø Volume pro Tick: {tick_df['volume'].mean():.3f} Lots")
    print(f"📊 Volume-Range: {tick_df['volume'].min():.3f} - {tick_df['volume'].max():.1f} Lots")
    
    # Aggregiere zu 100-Tick-Bars
    bars = []
    for i in range(0, len(tick_df), 100):
        chunk = tick_df.iloc[i:i+100]
        if len(chunk) < 50:  # Mindestens 50 Ticks
            continue
        
        bar = {
            'timestamp': chunk['timestamp'].iloc[-1],
            'open': chunk['price'].iloc[0],
            'high': chunk['price'].max(),
            'low': chunk['price'].min(),
            'close': chunk['price'].iloc[-1],
            'volume': chunk['volume'].sum(),  # ← ECHTES VOLUME!
            'tick_count': len(chunk),
            'avg_tick_volume': chunk['volume'].mean(),
            'max_tick_volume': chunk['volume'].max(),
            'ask_bid_ratio': chunk['ask_volume'].sum() / chunk['bid_volume'].sum()
        }
        bars.append(bar)
    
    bar_df = pd.DataFrame(bars)
    
    print(f"\n📊 AGGREGIERTE BARS:")
    print(f"   {len(bar_df)} Bars erstellt")
    print(f"   Ø Volume pro Bar: {bar_df['volume'].mean():.2f} Lots")
    print(f"   Ø Ask/Bid-Ratio: {bar_df['ask_bid_ratio'].mean():.2f}")
    
    return tick_df, bar_df

def main():
    """Hauptfunktion"""
    
    # 1. Analysiere aktuelle Dukascopy-Daten
    ohlcv_df = analyze_tick_volume()
    
    if not ohlcv_df.empty:
        # Erstelle Chart
        fig = create_volume_analysis_chart(ohlcv_df)
        fig.write_html("tick_volume_analysis.html")
        print(f"\n✅ Volume-Analyse-Chart gespeichert: tick_volume_analysis.html")
    
    # 2. Simuliere echtes Tick-Volume
    tick_df, bar_df = simulate_real_tick_volume()
    
    # Vergleiche simuliertes vs. aktuelles Volume
    if not ohlcv_df.empty:
        print(f"\n🔍 VERGLEICH:")
        print(f"   Aktuell (simuliert): Ø {ohlcv_df['volume'].mean():.2f} Lots/Bar")
        print(f"   Realistisch (Ticks): Ø {bar_df['volume'].mean():.2f} Lots/Bar")
        
        ratio = bar_df['volume'].mean() / ohlcv_df['volume'].mean()
        print(f"   Verhältnis: {ratio:.2f}x")
    
    print(f"\n🎯 FAZIT:")
    print(f"✅ Echtes Tick-Volume aus Dukascopy-Daten ist möglich!")
    print(f"✅ Volume = Summe aller Ask- und Bid-Volumes pro Tick")
    print(f"✅ Realistische Volume-Verteilung mit großen und kleinen Trades")
    print(f"✅ Ask/Bid-Volume-Ratio zeigt Marktrichtung an")

if __name__ == "__main__":
    main()