#!/usr/bin/env python3
"""
🚀 AI-Indicator-Optimizer Live Demo GUI
Interaktive Streamlit-Demo der Data Processing Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_indicator_optimizer.data.models import OHLCVData, MarketData
from ai_indicator_optimizer.data.processor import DataProcessor, ProcessingConfig
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector, DukascopyConfig
from ai_indicator_optimizer.core import HardwareDetector, ResourceManager


# Streamlit Page Config
st.set_page_config(
    page_title="AI-Indicator-Optimizer Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .hardware-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def download_dukascopy_data(symbol: str, timeframe: str, days: int, use_cache: bool, max_workers: int, use_real_data: bool = False) -> MarketData:
    """Lädt Daten von Dukascopy"""
    try:
        # Dukascopy Connector konfigurieren
        config = DukascopyConfig(
            max_workers=max_workers,
            cache_dir="./data/cache",
            use_real_data=use_real_data
        )
        
        connector = DukascopyConnector(config)
        
        # Zeitraum berechnen
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # OHLCV-Daten laden
        ohlcv_df = connector.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if ohlcv_df.empty:
            st.error(f"❌ No data received for {symbol}")
            return None
        
        # Zu OHLCVData konvertieren
        ohlcv_data = []
        for _, row in ohlcv_df.iterrows():
            ohlcv = OHLCVData(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            ohlcv_data.append(ohlcv)
        
        # MarketData erstellen
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            ohlcv_data=ohlcv_data
        )
        
        st.success(f"✅ Downloaded {len(ohlcv_data)} candles from Dukascopy")
        return market_data
        
    except Exception as e:
        st.error(f"❌ Dukascopy download failed: {e}")
        return None


def load_uploaded_data(uploaded_file) -> MarketData:
    """Lädt OHLCV-Daten aus hochgeladener CSV-Datei"""
    try:
        # CSV einlesen
        df = pd.read_csv(uploaded_file)
        
        # Spalten validieren
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            st.info("Required columns: timestamp, open, high, low, close, volume")
            return None
        
        # Timestamp konvertieren
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # OHLCV Data erstellen
        ohlcv_data = []
        for _, row in df.iterrows():
            ohlcv = OHLCVData(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            ohlcv_data.append(ohlcv)
        
        # MarketData erstellen
        market_data = MarketData(
            symbol=uploaded_file.name.split('.')[0].upper(),  # Dateiname als Symbol
            timeframe="uploaded",
            ohlcv_data=ohlcv_data
        )
        
        return market_data
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None


def generate_sample_data(symbol: str, days: int, timeframe: str) -> MarketData:
    """Generiert Sample-Marktdaten"""
    
    # Zeitraum berechnen
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    # Timeframe zu Minuten oder Ticks
    if 'tick' in timeframe:
        # Tick-basierte Timeframes
        tick_count = int(timeframe.replace('tick', ''))
        total_candles = days * 1000  # Annahme: ~1000 Ticks pro Tag
        minutes_per_candle = (days * 24 * 60) / total_candles
    else:
        # Zeit-basierte Timeframes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }
        minutes_per_candle = timeframe_minutes.get(timeframe, 1)
        total_candles = int((days * 24 * 60) / minutes_per_candle)
    
    # Generiere realistische OHLCV-Daten
    ohlcv_data = []
    base_price = 1.1000 if symbol == "EURUSD" else 1.2500
    
    for i in range(total_candles):
        if 'tick' in timeframe:
            # Für Tick-Charts: unregelmäßige Zeitstempel
            timestamp = start_time + timedelta(seconds=i * (days * 24 * 3600) / total_candles)
        else:
            # Für Zeit-Charts: regelmäßige Intervalle
            timestamp = start_time + timedelta(minutes=i * minutes_per_candle)
        
        # Simuliere Marktbewegung mit Trend und Volatilität
        trend = 0.00001 * np.sin(i / 100)  # Langfristiger Trend
        volatility = np.random.normal(0, 0.0005)  # Kurzfristige Volatilität
        
        price_change = trend + volatility
        open_price = base_price + price_change
        
        # OHLC basierend auf Open
        high_offset = abs(np.random.normal(0, 0.0003))
        low_offset = abs(np.random.normal(0, 0.0003))
        close_change = np.random.normal(0, 0.0002)
        
        high_price = open_price + high_offset
        low_price = open_price - low_offset
        close_price = open_price + close_change
        
        # Stelle sicher dass OHLC-Logik stimmt
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = np.random.randint(800, 1200)
        
        ohlcv_data.append(OHLCVData(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        ))
        
        base_price = close_price
    
    return MarketData(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_data=ohlcv_data
    )


@st.cache_resource
def get_hardware_info():
    """Hardware-Informationen laden"""
    detector = HardwareDetector()
    resource_manager = ResourceManager(detector)
    return detector, resource_manager


def create_candlestick_chart(market_data: MarketData, indicators=None):
    """Erstellt interaktiven Candlestick-Chart mit Plotly - optimiert für Tick-Daten"""
    
    df = pd.DataFrame([{
        'timestamp': candle.timestamp,
        'open': candle.open,
        'high': candle.high,
        'low': candle.low,
        'close': candle.close,
        'volume': candle.volume
    } for candle in market_data.ohlcv_data])
    
    # Analysiere Datentyp für optimale Darstellung
    is_tick_data = 'tick' in market_data.timeframe.lower()
    
    # Für Tick-Daten: Analysiere Zeitabstände
    if is_tick_data and len(df) > 1:
        time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
        avg_time_diff = time_diffs.median()
        min_time_diff = time_diffs.min()
        max_time_diff = time_diffs.max()
        
        st.info(f"📊 Tick-Daten erkannt: Ø Zeitabstand {avg_time_diff:.1f}s (Min: {min_time_diff:.1f}s, Max: {max_time_diff:.1f}s)")
    
    # Analysiere Docht-Qualität
    if len(df) > 0:
        upper_wicks = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wicks = df[['open', 'close']].min(axis=1) - df['low']
        
        bars_with_wicks = ((upper_wicks > 0.00001) | (lower_wicks > 0.00001)).sum()
        wick_percentage = (bars_with_wicks / len(df)) * 100
        
        if wick_percentage < 50:
            st.warning(f"⚠️ Nur {wick_percentage:.1f}% der Kerzen haben sichtbare Dochte - möglicherweise Darstellungsproblem")
        else:
            st.success(f"✅ {wick_percentage:.1f}% der Kerzen haben Dochte - gute Datenqualität")
    
    # Subplots erstellen
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Indicators', 'RSI & Stochastic', 'MACD & Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick Chart mit optimierter Darstellung
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=market_data.symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_line_width=2,
            decreasing_line_width=2,
            # Optimiere Kerzenbreite für Tick-Daten
            line=dict(width=1),
            whiskerwidth=0.8,
            # Verbessere Sichtbarkeit der Dochte
            increasing_fillcolor='rgba(0, 255, 136, 0.7)',
            decreasing_fillcolor='rgba(255, 68, 68, 0.7)'
        ),
        row=1, col=1
    )
    
    # Indikatoren hinzufügen falls vorhanden
    if indicators:
        # Moving Averages
        if indicators.sma:
            for period, values in indicators.sma.items():
                if len(values) == len(df):
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=values,
                            name=f'SMA {period}',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
        
        # Bollinger Bands
        if indicators.bollinger:
            upper = indicators.bollinger.get('upper', [])
            middle = indicators.bollinger.get('middle', [])
            lower = indicators.bollinger.get('lower', [])
            
            if len(upper) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=upper,
                        name='BB Upper',
                        line=dict(color='cyan', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=lower,
                        name='BB Lower',
                        line=dict(color='cyan', width=1),
                        fill='tonexty',
                        fillcolor='rgba(0,255,255,0.1)',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # RSI
        if indicators.rsi and len(indicators.rsi) == len(df):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=indicators.rsi,
                    name='RSI',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # RSI Levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        
        # MACD
        if indicators.macd:
            macd_line = indicators.macd.get('macd', [])
            signal_line = indicators.macd.get('signal', [])
            histogram = indicators.macd.get('histogram', [])
            
            if len(macd_line) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=macd_line,
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=signal_line,
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=histogram,
                        name='Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ),
                    row=3, col=1
                )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.3,
            yaxis='y4'
        ),
        row=3, col=1
    )
    
    # Berechne optimale Kerzenbreite für Tick-Daten
    if 'tick' in market_data.timeframe and len(df) > 1:
        # Für Tick-Daten: Berechne durchschnittlichen Zeitabstand
        time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
        avg_time_diff = time_diffs.median()  # Median ist robuster als Mean
        
        # Setze Kerzenbreite basierend auf Zeitabstand
        if avg_time_diff > 0:
            # Konvertiere zu Millisekunden für Plotly
            candle_width = avg_time_diff * 1000 * 0.8  # 80% des Zeitabstands
            
            fig.update_layout(
                xaxis=dict(
                    type='date',
                    # Optimiere für Tick-Charts
                    tickmode='auto',
                    nticks=20
                )
            )
    
    # Layout anpassen
    fig.update_layout(
        title=f"{market_data.symbol} - {market_data.timeframe} Chart",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark',
        # Verbessere Zoom-Verhalten für Tick-Daten
        xaxis_fixedrange=False,
        yaxis_fixedrange=False
    )
    
    # Y-Achsen anpassen
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def main():
    """Hauptfunktion der Demo-GUI"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI-Indicator-Optimizer Demo</h1>', unsafe_allow_html=True)
    st.markdown("**Multimodal KI-gesteuerte Trading-Indikator-Optimierung**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Hardware Info
    with st.sidebar.expander("🖥️ Hardware Status", expanded=True):
        try:
            detector, resource_manager = get_hardware_info()
            
            # CPU Info
            if detector.cpu_info:
                st.markdown(f"**CPU:** {detector.cpu_info.model[:30]}...")
                st.markdown(f"**Cores:** {detector.cpu_info.cores_physical} physical, {detector.cpu_info.cores_logical} logical")
                st.markdown(f"**Frequency:** {detector.cpu_info.frequency_current:.0f} MHz")
            
            # GPU Info
            if detector.gpu_info:
                for i, gpu in enumerate(detector.gpu_info):
                    st.markdown(f"**GPU {i}:** {gpu.name}")
                    st.markdown(f"**VRAM:** {gpu.memory_total // (1024**3)} GB")
            
            # RAM Info
            if detector.memory_info:
                st.markdown(f"**RAM:** {detector.memory_info.total // (1024**3)} GB")
                st.markdown(f"**Available:** {detector.memory_info.available // (1024**3)} GB")
            
        except Exception as e:
            st.error(f"Hardware detection failed: {e}")
    
    # Data Configuration
    st.sidebar.subheader("📊 Data Settings")
    
    # Data Source Selection
    data_source = st.sidebar.radio("Data Source", ["Generate Sample Data", "Download from Dukascopy", "Upload CSV File"])
    
    if data_source == "Generate Sample Data":
        symbol = st.sidebar.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"])
        timeframe = st.sidebar.selectbox("Timeframe", ["100tick", "1000tick", "1m", "5m", "15m", "30m", "1h", "4h"])
        days = st.sidebar.slider("Days of Data", 1, 30, 7)
    elif data_source == "Download from Dukascopy":
        symbol = st.sidebar.selectbox("Symbol", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF"])
        timeframe = st.sidebar.selectbox("Timeframe", ["100tick", "1000tick", "1M", "5M", "15M", "30M", "1H", "4H", "1D"])
        days = st.sidebar.slider("Days of Data", 1, 14, 7)
        
        # Dukascopy-spezifische Optionen
        use_real_data = st.sidebar.checkbox("Use Real Dukascopy Data", value=False, help="Download real tick data from Dukascopy (slower)")
        use_cache = st.sidebar.checkbox("Use Cache", value=True, help="Cache downloaded data for faster access")
        max_workers = st.sidebar.slider("Download Workers", 1, 32, 16, help="Number of parallel download threads")
        
        # Tick-Daten Info
        if "tick" in timeframe:
            tick_count = timeframe.replace("tick", "")
            st.sidebar.info(f"📊 Tick Data: {tick_count} ticks per bar")
            st.sidebar.warning("⚡ Tick data requires more processing power")
        
        st.sidebar.info("🌐 Real-time data from Dukascopy")
    else:
        # File Upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload OHLCV CSV File",
            type=['csv'],
            help="Upload a CSV file with columns: timestamp, open, high, low, close, volume"
        )
        
        if uploaded_file:
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        else:
            st.sidebar.info("📁 Please upload a CSV file")
    
    # Processing Configuration
    st.sidebar.subheader("⚡ Processing Settings")
    cpu_workers = st.sidebar.slider("CPU Workers", 1, 32, 8)
    enable_gpu = st.sidebar.checkbox("GPU Acceleration", value=True)
    
    # Main Content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>📈 Market Data</h3><p>Real-time processing</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 AI Analysis</h3><p>Multimodal processing</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>📊 Indicators</h3><p>8 technical indicators</p></div>', unsafe_allow_html=True)
    
    # Process Data Button
    if data_source == "Generate Sample Data":
        button_text = "🚀 Generate & Process Data"
        button_disabled = False
    elif data_source == "Download from Dukascopy":
        button_text = "🌐 Download & Process Data"
        button_disabled = False
    else:
        button_text = "📊 Process Uploaded Data"
        button_disabled = 'uploaded_file' not in locals() or uploaded_file is None
    
    if st.button(button_text, type="primary", disabled=button_disabled):
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load/Generate Data
            if data_source == "Generate Sample Data":
                status_text.text("📊 Generating market data...")
                progress_bar.progress(20)
                market_data = generate_sample_data(symbol, days, timeframe)
            elif data_source == "Download from Dukascopy":
                status_text.text("🌐 Downloading from Dukascopy...")
                progress_bar.progress(20)
                market_data = download_dukascopy_data(symbol, timeframe, days, use_cache, max_workers, use_real_data)
            else:
                status_text.text("📁 Loading uploaded data...")
                progress_bar.progress(20)
                market_data = load_uploaded_data(uploaded_file)
            
            # Step 2: Initialize Processor
            status_text.text("⚙️ Initializing data processor...")
            progress_bar.progress(40)
            
            config = ProcessingConfig(
                cpu_workers=cpu_workers,
                gpu_acceleration=enable_gpu,
                chart_width=1024,
                chart_height=768
            )
            
            processor = DataProcessor(config)
            
            # Step 3: Process Data
            status_text.text("🧮 Calculating indicators...")
            progress_bar.progress(60)
            
            start_time = time.time()
            result = processor.process_market_data(market_data)
            processing_time = time.time() - start_time
            
            progress_bar.progress(80)
            status_text.text("📈 Creating charts...")
            
            # Step 4: Create Charts
            fig = create_candlestick_chart(market_data, result['indicators'])
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Display Results
            data_type = "tick bars" if "tick" in timeframe else "candles"
            st.success(f"✅ Processed {len(market_data.ohlcv_data)} {data_type} in {processing_time:.2f} seconds")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                data_label = f"{timeframe} Bars" if "tick" in timeframe else "Candles"
                st.metric(data_label, len(market_data.ohlcv_data))
            
            with col2:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            with col3:
                st.metric("Indicators", result['processing_stats']['indicators_calculated'])
            
            with col4:
                st.metric("Charts Generated", result['processing_stats']['charts_generated'])
            
            # Interactive Chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicator Details
            st.subheader("📊 Indicator Analysis")
            
            indicators = result['indicators']
            
            # RSI Analysis
            if indicators.rsi:
                latest_rsi = indicators.rsi[-1]
                rsi_status = "🔴 Overbought" if latest_rsi > 70 else "🟢 Oversold" if latest_rsi < 30 else "🟡 Neutral"
                st.write(f"**RSI:** {latest_rsi:.2f} - {rsi_status}")
            
            # MACD Analysis
            if indicators.macd:
                macd_line = indicators.macd['macd'][-1]
                signal_line = indicators.macd['signal'][-1]
                macd_signal = "🟢 Bullish" if macd_line > signal_line else "🔴 Bearish"
                st.write(f"**MACD:** {macd_line:.6f} - {macd_signal}")
            
            # Training Sample Info
            st.subheader("🤖 Multimodal Training Sample")
            training_sample = result['training_sample']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numerical Features:**")
                st.write(f"- Feature Count: {len(training_sample['numerical_features'])}")
                st.write(f"- Normalization: Z-Score")
                st.write(f"- Data Type: {training_sample['numerical_features'].dtype}")
            
            with col2:
                st.write("**Text Descriptions:**")
                for desc in training_sample['text_descriptions'][:3]:
                    st.write(f"- {desc}")
            
            # Chart Images
            if result['chart_images']:
                st.subheader("🖼️ Generated Chart Images")
                cols = st.columns(min(len(result['chart_images']), 3))
                
                for i, chart_img in enumerate(result['chart_images'][:3]):
                    with cols[i]:
                        st.image(chart_img, caption=f"Chart {i+1}", use_column_width=True)
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 AI-Indicator-Optimizer** - Powered by MiniCPM-4.1-8B & RTX 5090")


if __name__ == "__main__":
    main()