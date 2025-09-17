#!/usr/bin/env python3
"""
Demo Script für Dukascopy Data Connector
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from ai_indicator_optimizer.data.connector import DukascopyConnector, DukascopyConfig
from ai_indicator_optimizer.data.models import MarketData


def setup_logging():
    """Setup Logging für Demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_connector_features():
    """Demonstriert Data Connector Features"""
    print("=== Dukascopy Data Connector Demo ===")
    
    # Konfiguration
    config = DukascopyConfig(
        max_retries=2,
        timeout=10,
        cache_dir="./demo_cache"
    )
    
    # Connector mit reduzierter Worker-Anzahl für Demo
    connector = DukascopyConnector(parallel_workers=4, config=config)
    
    print(f"✓ Connector initialisiert mit {connector.parallel_workers} Workern")
    
    # Verfügbare Symbole
    symbols = connector.get_available_symbols()
    print(f"✓ Verfügbare Symbole: {', '.join(symbols)}")
    
    # Daten-Info
    start_time = datetime.now(timezone.utc) - timedelta(hours=2)
    end_time = datetime.now(timezone.utc) - timedelta(hours=1)
    
    info = connector.get_data_info("EURUSD", start_time, end_time)
    print(f"✓ Daten-Info für EURUSD:")
    print(f"  - Zeitraum: {info['start']} bis {info['end']}")
    print(f"  - Geschätzte Stunden: {info['estimated_hours']}")
    print(f"  - Unterstützt: {info['supported']}")
    
    # Timeframe-Tests
    timeframes = ["1m", "5m", "15m", "1h"]
    print(f"✓ Unterstützte Timeframes:")
    for tf in timeframes:
        try:
            seconds = connector._parse_timeframe(tf)
            print(f"  - {tf}: {seconds} Sekunden")
        except ValueError as e:
            print(f"  - {tf}: Fehler - {e}")
    
    # Mock-Daten für Demo (da echte Dukascopy-API Authentifizierung braucht)
    print("\n=== Mock-Daten Demo ===")
    
    # Simuliere Tick-Daten
    from ai_indicator_optimizer.data.models import TickData
    
    mock_ticks = []
    base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    
    for i in range(120):  # 2 Minuten Tick-Daten
        tick = TickData(
            timestamp=base_time + timedelta(seconds=i),
            bid=1.1000 + (i * 0.00001),
            ask=1.1002 + (i * 0.00001),
            volume=1000 + (i * 10)
        )
        mock_ticks.append(tick)
    
    print(f"✓ {len(mock_ticks)} Mock-Ticks generiert")
    
    # Konvertiere zu OHLCV
    ohlcv_data = connector._convert_ticks_to_ohlcv(mock_ticks, "1m")
    print(f"✓ {len(ohlcv_data)} OHLCV-Candles generiert")
    
    # Zeige erste Candle
    if ohlcv_data:
        candle = ohlcv_data[0]
        print(f"  Erste Candle:")
        print(f"    Zeit: {candle.timestamp}")
        print(f"    OHLC: {candle.open:.5f} / {candle.high:.5f} / {candle.low:.5f} / {candle.close:.5f}")
        print(f"    Volume: {candle.volume:.0f}")
    
    # MarketData erstellen
    market_data = MarketData(
        symbol="EURUSD",
        timeframe="1m",
        ohlcv_data=ohlcv_data,
        tick_data=mock_ticks
    )
    
    # Validierung
    validation_result = connector.validate_data_integrity(market_data)
    print(f"\n✓ Datenvalidierung:")
    print(f"  - Gültig: {validation_result['valid']}")
    print(f"  - Fehler: {len(validation_result['errors'])}")
    print(f"  - Warnungen: {len(validation_result['warnings'])}")
    
    if validation_result['ohlcv_validation']:
        stats = validation_result['ohlcv_validation']['stats']
        print(f"  - OHLCV Count: {stats['count']}")
    
    if validation_result['tick_validation']:
        stats = validation_result['tick_validation']['stats']
        print(f"  - Tick Count: {stats['count']}")
        print(f"  - Avg Spread: {stats['avg_spread']:.6f}")
    
    # NumPy Konvertierung
    numpy_data = market_data.to_numpy()
    print(f"\n✓ NumPy Array: {numpy_data.shape}")
    print(f"  - Spalten: Timestamp, Open, High, Low, Close, Volume")
    print(f"  - Erste Zeile: {numpy_data[0] if len(numpy_data) > 0 else 'Keine Daten'}")
    
    print("\n🎉 Demo abgeschlossen!")


def demo_parallel_processing():
    """Demonstriert parallele Verarbeitung"""
    print("\n=== Parallel Processing Demo ===")
    
    connector = DukascopyConnector(parallel_workers=8)
    
    # Zeitraum aufteilen
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    
    ranges = connector._split_time_range_by_hours(start, end)
    print(f"✓ Zeitraum aufgeteilt in {len(ranges)} Stunden-Chunks:")
    
    for i, (range_start, range_end) in enumerate(ranges[:5]):  # Zeige nur erste 5
        print(f"  {i+1}. {range_start} - {range_end}")
    
    if len(ranges) > 5:
        print(f"  ... und {len(ranges) - 5} weitere")
    
    print(f"✓ Würde {connector.parallel_workers} Worker für parallelen Download nutzen")


if __name__ == "__main__":
    setup_logging()
    demo_connector_features()
    demo_parallel_processing()