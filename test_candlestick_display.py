#!/usr/bin/env python3
"""
Test Script für Candlestick-Darstellung
Vergleicht verschiedene Datenquellen
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

st.set_page_config(page_title="Candlestick Test", layout="wide")

st.title("🔍 Candlestick Display Test")

def create_sample_data():
    """Erstelle Sample-Daten mit echten Dochten"""
    data = []
    base_price = 1.1000
    
    for i in range(50):
        timestamp = datetime.now() - timedelta(minutes=50-i)
        
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
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(800, 1200)
        })
        
        base_price = close_price
    
    return pd.DataFrame(data)

def load_dukascopy_data():
    """Lade Dukascopy-Daten"""
    try:
        config = DukascopyConfig(
            max_workers=4,
            cache_dir="./data/cache",
            use_real_data=False
        )
        
        connector = DukascopyConnector(config)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        df = connector.get_ohlcv_data(
            symbol="EUR/USD",
            timeframe="100tick",
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        return df.head(50)  # Nur erste 50 für Vergleich
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Dukascopy-Daten: {e}")
        return pd.DataFrame()

def create_candlestick_chart(df, title):
    """Erstelle Candlestick-Chart"""
    
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
            decreasing_line_width=2,
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        )
    )
    
    fig.update_layout(
        title=f"{title} - Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        showlegend=True,
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def analyze_ohlc_data(df, name):
    """Analysiere OHLC-Daten"""
    
    st.subheader(f"📊 {name} - Datenanalyse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anzahl Bars", len(df))
        st.metric("Zeitspanne", f"{(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f}h")
    
    with col2:
        price_range = df['high'].max() - df['low'].min()
        st.metric("Preisspanne", f"{price_range:.5f}")
        avg_range = (df['high'] - df['low']).mean()
        st.metric("Ø Bar-Range", f"{avg_range:.6f}")
    
    with col3:
        # Docht-Analyse
        upper_wicks = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wicks = df[['open', 'close']].min(axis=1) - df['low']
        
        bars_with_upper_wicks = (upper_wicks > 0.00001).sum()
        bars_with_lower_wicks = (lower_wicks > 0.00001).sum()
        
        st.metric("Bars mit oberen Dochten", f"{bars_with_upper_wicks}/{len(df)}")
        st.metric("Bars mit unteren Dochten", f"{bars_with_lower_wicks}/{len(df)}")
    
    # Detailanalyse
    st.write("**Docht-Statistiken:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"- Obere Dochte Ø: {upper_wicks.mean():.6f}")
        st.write(f"- Obere Dochte Max: {upper_wicks.max():.6f}")
        st.write(f"- Obere Dochte > 0: {(upper_wicks > 0.00001).sum()}")
    
    with col2:
        st.write(f"- Untere Dochte Ø: {lower_wicks.mean():.6f}")
        st.write(f"- Untere Dochte Max: {lower_wicks.max():.6f}")
        st.write(f"- Untere Dochte > 0: {(lower_wicks > 0.00001).sum()}")
    
    # Erste 5 Zeilen anzeigen
    st.write("**Erste 5 Bars:**")
    display_df = df[['timestamp', 'open', 'high', 'low', 'close']].head()
    st.dataframe(display_df, use_container_width=True)
    
    return fig

# Main App
st.sidebar.header("⚙️ Test Configuration")

test_sample = st.sidebar.checkbox("Test Sample Data", value=True)
test_dukascopy = st.sidebar.checkbox("Test Dukascopy Data", value=True)

if test_sample:
    st.header("1. 📊 Sample Data (Generate Sample Data)")
    
    sample_df = create_sample_data()
    sample_fig = create_candlestick_chart(sample_df, "Sample Data")
    
    analyze_ohlc_data(sample_df, "Sample Data")
    st.plotly_chart(sample_fig, use_container_width=True)

if test_dukascopy:
    st.header("2. 🌐 Dukascopy Data (Download from Dukascopy)")
    
    dukascopy_df = load_dukascopy_data()
    
    if not dukascopy_df.empty:
        dukascopy_fig = create_candlestick_chart(dukascopy_df, "Dukascopy Data")
        
        analyze_ohlc_data(dukascopy_df, "Dukascopy Data")
        st.plotly_chart(dukascopy_fig, use_container_width=True)
    else:
        st.error("❌ Keine Dukascopy-Daten verfügbar")

# Vergleich
if test_sample and test_dukascopy and not dukascopy_df.empty:
    st.header("🔍 Vergleichsanalyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        sample_upper_wicks = sample_df['high'] - sample_df[['open', 'close']].max(axis=1)
        sample_lower_wicks = sample_df[['open', 'close']].min(axis=1) - sample_df['low']
        
        st.write(f"Bars mit Dochten: {(sample_upper_wicks > 0.00001).sum() + (sample_lower_wicks > 0.00001).sum()}/{len(sample_df)}")
        st.write(f"Ø Docht-Länge: {(sample_upper_wicks.mean() + sample_lower_wicks.mean()):.6f}")
    
    with col2:
        st.subheader("Dukascopy Data")
        duka_upper_wicks = dukascopy_df['high'] - dukascopy_df[['open', 'close']].max(axis=1)
        duka_lower_wicks = dukascopy_df[['open', 'close']].min(axis=1) - dukascopy_df['low']
        
        st.write(f"Bars mit Dochten: {(duka_upper_wicks > 0.00001).sum() + (duka_lower_wicks > 0.00001).sum()}/{len(dukascopy_df)}")
        st.write(f"Ø Docht-Länge: {(duka_upper_wicks.mean() + duka_lower_wicks.mean()):.6f}")
    
    # Fazit
    sample_wick_ratio = ((sample_upper_wicks > 0.00001).sum() + (sample_lower_wicks > 0.00001).sum()) / len(sample_df)
    duka_wick_ratio = ((duka_upper_wicks > 0.00001).sum() + (duka_lower_wicks > 0.00001).sum()) / len(dukascopy_df)
    
    if sample_wick_ratio > duka_wick_ratio * 2:
        st.warning("⚠️ Sample Data hat deutlich mehr Dochte als Dukascopy Data!")
    elif duka_wick_ratio > sample_wick_ratio * 2:
        st.warning("⚠️ Dukascopy Data hat deutlich mehr Dochte als Sample Data!")
    else:
        st.success("✅ Beide Datenquellen haben ähnliche Docht-Verteilungen")

st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Ziel:** Vergleiche Candlestick-Darstellung zwischen Sample- und Dukascopy-Daten")