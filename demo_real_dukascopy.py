#!/usr/bin/env python3
"""
Demo: Echte Dukascopy-Daten + Profi-Daten-Support
Zeigt Integration verschiedener Datenquellen
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig

st.set_page_config(page_title="🌐 Real Dukascopy + Pro Data", layout="wide")

st.title("🌐 Echte Dukascopy-Daten + Professionelle Datenquellen")

def create_data_source_config():
    """Erstelle Datenquellen-Konfiguration"""
    
    st.sidebar.header("📊 Datenquellen-Konfiguration")
    
    # Datenquelle auswählen
    data_source = st.sidebar.selectbox(
        "Primäre Datenquelle",
        ["Simulierte Daten", "Echte Dukascopy-Daten", "Lokale Profi-Daten", "Auto (Priorität)"]
    )
    
    # Lokale Daten-Pfad
    local_data_path = st.sidebar.text_input(
        "Lokaler Daten-Pfad",
        value="./data/professional",
        help="Pfad zu professionellen Tick-Daten (CSV/Parquet)"
    )
    
    # Dukascopy-Einstellungen
    st.sidebar.subheader("🌐 Dukascopy-Einstellungen")
    use_real_dukascopy = st.sidebar.checkbox("Echte Dukascopy-Daten", value=True)
    max_workers = st.sidebar.slider("Download-Workers", 1, 16, 8)
    
    # Symbol und Zeitraum
    st.sidebar.subheader("📈 Marktdaten")
    symbol = st.sidebar.selectbox("Symbol", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
    timeframe = st.sidebar.selectbox("Timeframe", ["100tick", "1000tick", "1M", "5M", "15M", "1H"])
    days = st.sidebar.slider("Tage", 1, 7, 2)
    
    return {
        "data_source": data_source,
        "local_data_path": local_data_path if local_data_path.strip() else None,
        "use_real_dukascopy": use_real_dukascopy,
        "max_workers": max_workers,
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days
    }

def create_connector_config(settings):
    """Erstelle Connector-Konfiguration"""
    
    # Bestimme Einstellungen basierend auf Datenquelle
    if settings["data_source"] == "Simulierte Daten":
        use_real_data = False
        local_path = None
    elif settings["data_source"] == "Echte Dukascopy-Daten":
        use_real_data = True
        local_path = None
    elif settings["data_source"] == "Lokale Profi-Daten":
        use_real_data = False
        local_path = settings["local_data_path"]
    else:  # Auto (Priorität)
        use_real_data = settings["use_real_dukascopy"]
        local_path = settings["local_data_path"]
    
    return DukascopyConfig(
        max_workers=settings["max_workers"],
        cache_dir="./data/cache",
        use_real_data=use_real_data,
        local_data_path=local_path
    )

def analyze_data_source(df, source_name):
    """Analysiere Datenquelle"""
    
    st.subheader(f"📊 {source_name} - Analyse")
    
    if df.empty:
        st.error("❌ Keine Daten verfügbar")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Anzahl Ticks/Bars", len(df))
        
    with col2:
        time_span = df['timestamp'].max() - df['timestamp'].min()
        st.metric("Zeitspanne", f"{time_span.total_seconds() / 3600:.1f}h")
    
    with col3:
        if 'volume' in df.columns:
            st.metric("Gesamt-Volume", f"{df['volume'].sum():.1f}")
        else:
            st.metric("Volume", "N/A")
    
    with col4:
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
            st.metric("Ø Zeitabstand", f"{time_diffs.mean():.1f}s")
        else:
            st.metric("Zeitabstand", "N/A")
    
    # Datenqualitäts-Analyse
    st.write("**Datenqualität:**")
    
    quality_checks = []
    
    # Timestamp-Kontinuität
    if len(df) > 1:
        time_gaps = df['timestamp'].diff().dt.total_seconds().dropna()
        large_gaps = (time_gaps > 300).sum()  # > 5 Minuten
        quality_checks.append(f"Große Zeitlücken: {large_gaps}")
    
    # Spread-Analyse
    if 'ask' in df.columns and 'bid' in df.columns:
        spreads = df['ask'] - df['bid']
        avg_spread = spreads.mean()
        quality_checks.append(f"Ø Spread: {avg_spread:.5f}")
        
        # Unrealistische Spreads
        weird_spreads = ((spreads < 0) | (spreads > 0.01)).sum()
        quality_checks.append(f"Unrealistische Spreads: {weird_spreads}")
    
    # Volume-Analyse
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        quality_checks.append(f"Zero-Volume Ticks: {zero_volume}")
        
        if 'ask_volume' in df.columns and 'bid_volume' in df.columns:
            quality_checks.append("✅ Ask/Bid-Volume verfügbar")
        else:
            quality_checks.append("⚠️ Nur Gesamt-Volume")
    
    for check in quality_checks:
        st.write(f"- {check}")
    
    # Erste 5 Zeilen anzeigen
    st.write("**Erste 5 Datenpunkte:**")
    display_cols = ['timestamp', 'ask', 'bid']
    if 'volume' in df.columns:
        display_cols.append('volume')
    if 'tick_count' in df.columns:
        display_cols.append('tick_count')
    
    st.dataframe(df[display_cols].head(), use_container_width=True)

def create_comparison_chart(dfs, names):
    """Erstelle Vergleichs-Chart verschiedener Datenquellen"""
    
    fig = make_subplots(
        rows=len(dfs), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{name} - OHLC + Volume" for name in names]
    )
    
    colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00']
    
    for i, (df, name) in enumerate(zip(dfs, names)):
        if df.empty:
            continue
        
        row = i + 1
        color = colors[i % len(colors)]
        
        # OHLC-Chart (falls verfügbar)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=f"{name} OHLC",
                    increasing_line_color=color,
                    decreasing_line_color=color
                ),
                row=row, col=1
            )
        elif 'ask' in df.columns and 'bid' in df.columns:
            # Tick-Daten als Linien
            mid_price = (df['ask'] + df['bid']) / 2
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=mid_price,
                    mode='lines',
                    name=f"{name} Mid-Price",
                    line=dict(color=color, width=1)
                ),
                row=row, col=1
            )
        
        # Volume (falls verfügbar)
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name=f"{name} Volume",
                    marker_color=color,
                    opacity=0.3,
                    yaxis=f'y{row+len(dfs)}'
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        title="📊 Datenquellen-Vergleich",
        height=300 * len(dfs),
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def setup_professional_data_demo():
    """Setup für Profi-Daten-Demo"""
    
    st.header("💼 Professionelle Datenquellen-Setup")
    
    st.markdown("""
    ### 📁 **Unterstützte Formate:**
    
    **CSV-Dateien:**
    ```
    timestamp,ask,bid,volume
    2025-09-20 10:00:00.123,1.10015,1.10000,2.5
    2025-09-20 10:00:00.456,1.10016,1.10001,1.8
    ```
    
    **Parquet-Dateien:**
    - Optimiert für große Datenmengen
    - Schnellere Ladezeiten
    - Komprimierung
    
    **Dateiname-Patterns:**
    - `EURUSD_20250920.csv`
    - `EUR/USD_2025-09-20.csv`
    - `2025/09/EURUSD_20.csv`
    - `EURUSD/20250920.csv`
    
    **Spalten-Mapping:**
    - `timestamp`, `time`, `datetime` → Zeitstempel
    - `ask`, `ask_price`, `offer` → Ask-Preis
    - `bid`, `bid_price` → Bid-Preis
    - `volume`, `vol`, `size` → Volumen
    - `ask_volume`, `bid_volume` → Separate Volumina
    """)
    
    # Beispiel-Daten erstellen
    if st.button("📝 Beispiel-Daten erstellen"):
        create_sample_professional_data()

def create_sample_professional_data():
    """Erstelle Beispiel-Daten für Demo"""
    
    try:
        # Erstelle Verzeichnis
        data_dir = Path("./data/professional")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generiere Beispiel-Tick-Daten
        base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        
        ticks = []
        base_price = 1.1000
        
        for i in range(1000):
            timestamp = base_time + timedelta(seconds=i * 2.5)  # Alle 2.5 Sekunden
            
            # Preis-Bewegung
            price_change = np.random.normal(0, 0.00005)
            price = base_price + price_change
            
            # Spread
            spread = np.random.uniform(0.00010, 0.00020)
            ask = price + spread/2
            bid = price - spread/2
            
            # Volume
            volume = np.random.uniform(0.5, 5.0)
            ask_volume = volume * np.random.uniform(0.3, 0.7)
            bid_volume = volume - ask_volume
            
            ticks.append({
                'timestamp': timestamp,
                'ask': ask,
                'bid': bid,
                'volume': volume,
                'ask_volume': ask_volume,
                'bid_volume': bid_volume
            })
            
            base_price = price
        
        # Speichere als CSV
        df = pd.DataFrame(ticks)
        today = datetime.now().strftime('%Y%m%d')
        
        csv_file = data_dir / f"EURUSD_{today}.csv"
        df.to_csv(csv_file, index=False)
        
        # Speichere als Parquet
        parquet_file = data_dir / f"EURUSD_{today}.parquet"
        df.to_parquet(parquet_file, index=False)
        
        st.success(f"✅ Beispiel-Daten erstellt:")
        st.write(f"- {csv_file}")
        st.write(f"- {parquet_file}")
        st.write(f"- {len(df)} Ticks generiert")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen der Beispiel-Daten: {e}")

def main():
    """Hauptfunktion"""
    
    # Konfiguration
    settings = create_data_source_config()
    
    # Setup-Bereich
    setup_professional_data_demo()
    
    # Daten laden
    if st.button("🚀 Daten laden und analysieren", type="primary"):
        
        with st.spinner("Lade Daten..."):
            
            # Connector erstellen
            config = create_connector_config(settings)
            connector = DukascopyConnector(config)
            
            # Zeitraum
            end_date = datetime.now()
            start_date = end_date - timedelta(days=settings["days"])
            
            # Daten laden
            try:
                df = connector.get_ohlcv_data(
                    symbol=settings["symbol"],
                    timeframe=settings["timeframe"],
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                
                if not df.empty:
                    # Analysiere Datenquelle
                    source_info = "Unbekannt"
                    if config.local_data_path and Path(config.local_data_path).exists():
                        source_info = f"Lokale Profi-Daten ({config.local_data_path})"
                    elif config.use_real_data:
                        source_info = "Echte Dukascopy-Daten"
                    else:
                        source_info = "Simulierte Daten"
                    
                    analyze_data_source(df, source_info)
                    
                    # Chart erstellen
                    if 'open' in df.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Candlestick(
                                x=df['timestamp'],
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=settings["symbol"],
                                increasing_line_color='#00ff88',
                                decreasing_line_color='#ff4444'
                            )
                        )
                        
                        fig.update_layout(
                            title=f"{source_info} - {settings['symbol']} {settings['timeframe']}",
                            height=600,
                            template='plotly_dark',
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume-Analyse
                    if 'volume' in df.columns:
                        st.subheader("📊 Volume-Analyse")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Volume-Verteilung
                            fig_vol = go.Figure()
                            fig_vol.add_trace(
                                go.Histogram(
                                    x=df['volume'],
                                    nbinsx=50,
                                    name='Volume-Verteilung',
                                    marker_color='lightblue'
                                )
                            )
                            fig_vol.update_layout(
                                title="Volume-Verteilung",
                                height=400,
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig_vol, use_container_width=True)
                        
                        with col2:
                            # Volume über Zeit
                            fig_vol_time = go.Figure()
                            fig_vol_time.add_trace(
                                go.Scatter(
                                    x=df['timestamp'],
                                    y=df['volume'],
                                    mode='lines',
                                    name='Volume',
                                    line=dict(color='orange')
                                )
                            )
                            fig_vol_time.update_layout(
                                title="Volume über Zeit",
                                height=400,
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig_vol_time, use_container_width=True)
                
                else:
                    st.error("❌ Keine Daten erhalten")
                    
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Daten: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()