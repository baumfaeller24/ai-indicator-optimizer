"""
Unit Tests für Dukascopy Data Connector
"""

import pytest
import struct
import gzip
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List

from ai_indicator_optimizer.data.connector import (
    DukascopyConnector, 
    DukascopyConfig, 
    DukascopyDataValidator
)
from ai_indicator_optimizer.data.models import TickData, OHLCVData, MarketData


class TestDukascopyDataValidator:
    """Tests für DukascopyDataValidator"""
    
    def test_validate_tick_data_valid(self):
        """Test mit gültigen Tick-Daten"""
        tick_data = [
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                bid=1.1000,
                ask=1.1002,
                volume=1000.0
            ),
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 1, tzinfo=timezone.utc),
                bid=1.1001,
                ask=1.1003,
                volume=1500.0
            )
        ]
        
        validator = DukascopyDataValidator()
        result = validator.validate_tick_data(tick_data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["stats"]["count"] == 2
        assert abs(result["stats"]["avg_spread"] - 0.0002) < 0.0001  # Floating point precision
    
    def test_validate_tick_data_invalid_prices(self):
        """Test mit ungültigen Preisen"""
        tick_data = [
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                bid=1.1002,  # Bid > Ask
                ask=1.1000,
                volume=1000.0
            )
        ]
        
        validator = DukascopyDataValidator()
        result = validator.validate_tick_data(tick_data)
        
        assert result["valid"] is False
        assert any("Ask price <= Bid price" in error for error in result["errors"])
    
    def test_validate_ohlcv_data_valid(self):
        """Test mit gültigen OHLCV-Daten"""
        ohlcv_data = [
            OHLCVData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                open=1.1000,
                high=1.1010,
                low=1.0990,
                close=1.1005,
                volume=10000.0
            )
        ]
        
        validator = DukascopyDataValidator()
        result = validator.validate_ohlcv_data(ohlcv_data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["stats"]["count"] == 1
    
    def test_validate_ohlcv_data_invalid_ohlc(self):
        """Test mit ungültiger OHLC-Konsistenz"""
        ohlcv_data = [
            OHLCVData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                open=1.1000,
                high=1.0990,  # High < Open
                low=1.0995,
                close=1.1005,
                volume=10000.0
            )
        ]
        
        validator = DukascopyDataValidator()
        result = validator.validate_ohlcv_data(ohlcv_data)
        
        assert result["valid"] is False
        assert any("OHLC inconsistency" in error for error in result["errors"])


class TestDukascopyConnector:
    """Tests für DukascopyConnector"""
    
    @pytest.fixture
    def connector(self):
        """Test-Connector mit reduzierter Worker-Anzahl"""
        config = DukascopyConfig(
            max_retries=1,
            timeout=5,
            cache_dir="./test_cache"
        )
        return DukascopyConnector(parallel_workers=2, config=config)
    
    @pytest.fixture
    def mock_tick_data(self):
        """Mock Tick-Daten"""
        return [
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                bid=1.1000,
                ask=1.1002,
                volume=1000.0
            ),
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
                bid=1.1001,
                ask=1.1003,
                volume=1500.0
            ),
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc),
                bid=1.1002,
                ask=1.1004,
                volume=2000.0
            )
        ]
    
    def test_init(self, connector):
        """Test Initialisierung"""
        assert connector.parallel_workers == 2
        assert connector.config.max_retries == 1
        assert "EURUSD" in connector.symbol_mapping
    
    def test_get_available_symbols(self, connector):
        """Test verfügbare Symbole"""
        symbols = connector.get_available_symbols()
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert len(symbols) > 0
    
    def test_parse_timeframe(self, connector):
        """Test Timeframe-Parsing"""
        assert connector._parse_timeframe("1m") == 60
        assert connector._parse_timeframe("5m") == 300
        assert connector._parse_timeframe("1h") == 3600
        assert connector._parse_timeframe("1d") == 86400
        
        with pytest.raises(ValueError):
            connector._parse_timeframe("invalid")
    
    def test_split_time_range_by_hours(self, connector):
        """Test Zeitraum-Aufteilung"""
        start = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 15, 0, tzinfo=timezone.utc)
        
        ranges = connector._split_time_range_by_hours(start, end)
        
        assert len(ranges) == 3  # 10:00-11:00, 11:00-12:00, 12:00-12:15
        assert ranges[0][0].hour == 10
        assert ranges[0][0].minute == 0
        assert ranges[-1][1] == end
    
    def test_create_ohlcv_candle(self, connector, mock_tick_data):
        """Test OHLCV-Candle-Erstellung"""
        timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        candle = connector._create_ohlcv_candle(timestamp, mock_tick_data)
        
        assert candle.timestamp == timestamp
        assert candle.open == 1.1001  # Mid-Price des ersten Ticks
        assert candle.high == 1.1003  # Höchster Mid-Price
        assert candle.low == 1.1001   # Niedrigster Mid-Price
        assert candle.close == 1.1003 # Mid-Price des letzten Ticks
        assert candle.volume == 4500.0  # Summe aller Volumes
    
    def test_convert_ticks_to_ohlcv(self, connector, mock_tick_data):
        """Test Tick-zu-OHLCV-Konvertierung"""
        ohlcv_data = connector._convert_ticks_to_ohlcv(mock_tick_data, "1m")
        
        assert len(ohlcv_data) >= 2  # Mindestens 2 Candles (kann 3 sein wegen Sekunden-Timestamps)
        assert ohlcv_data[0].timestamp.minute == 0
        assert any(candle.timestamp.minute == 1 for candle in ohlcv_data)  # Mindestens eine Candle für Minute 1
    
    def test_parse_bi5_data(self, connector):
        """Test .bi5 Format-Parsing"""
        # Mock .bi5 Daten erstellen (20 Bytes pro Tick)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # Ein Tick: time_offset=0, ask=110020, bid=110000, ask_vol=1000, bid_vol=1000
        mock_data = struct.pack('>IIIII', 0, 110020, 110000, 1000000, 1000000)
        
        tick_data = connector._parse_bi5_data(mock_data, base_time)
        
        assert len(tick_data) == 1
        assert tick_data[0].timestamp == base_time
        assert abs(tick_data[0].ask - 1.1002) < 0.0001
        assert abs(tick_data[0].bid - 1.1000) < 0.0001
    
    @patch('requests.get')
    def test_download_with_retry_success(self, mock_get, connector):
        """Test erfolgreicher Download"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test_data"
        mock_get.return_value = mock_response
        
        result = connector._download_with_retry("http://test.com/data")
        
        assert result == b"test_data"
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_download_with_retry_404(self, mock_get, connector):
        """Test 404 Response"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = connector._download_with_retry("http://test.com/data")
        
        assert result is None
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_download_with_retry_failure(self, mock_get, connector):
        """Test fehlgeschlagener Download mit Retry"""
        mock_get.side_effect = Exception("Network error")
        
        result = connector._download_with_retry("http://test.com/data")
        
        assert result is None
        assert mock_get.call_count == connector.config.max_retries
    
    @patch.object(DukascopyConnector, '_fetch_tick_data_hour')
    def test_fetch_tick_data_parallel(self, mock_fetch_hour, connector):
        """Test paralleler Tick-Data-Download"""
        # Mock für _fetch_tick_data_hour
        mock_fetch_hour.return_value = [
            TickData(
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                bid=1.1000,
                ask=1.1002,
                volume=1000.0
            )
        ]
        
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        
        result = connector.fetch_tick_data("EURUSD", start, end)
        
        assert len(result) == 1  # Ein Tick pro Stunde
        assert mock_fetch_hour.call_count == 1  # Eine Stunde
    
    def test_fetch_tick_data_invalid_symbol(self, connector):
        """Test mit ungültigem Symbol"""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="Unsupported symbol"):
            connector.fetch_tick_data("INVALID", start, end)
    
    @patch.object(DukascopyConnector, 'fetch_tick_data')
    def test_fetch_ohlcv_data(self, mock_fetch_tick, connector, mock_tick_data):
        """Test OHLCV-Daten-Abruf"""
        mock_fetch_tick.return_value = mock_tick_data
        
        result = connector.fetch_ohlcv_data("EURUSD", "1m", 1)
        
        assert len(result) > 0
        assert all(isinstance(candle, OHLCVData) for candle in result)
        mock_fetch_tick.assert_called_once()
    
    def test_validate_data_integrity(self, connector):
        """Test Datenintegritäts-Validierung"""
        # Gültige Daten
        valid_data = MarketData(
            symbol="EURUSD",
            timeframe="1m",
            ohlcv_data=[
                OHLCVData(
                    timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    open=1.1000,
                    high=1.1010,
                    low=1.0990,
                    close=1.1005,
                    volume=10000.0
                )
            ]
        )
        
        result = connector.validate_data_integrity(valid_data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["ohlcv_validation"] is not None
    
    def test_get_data_info(self, connector):
        """Test Daten-Info-Abruf"""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        info = connector.get_data_info("EURUSD", start, end)
        
        assert info["symbol"] == "EURUSD"
        assert info["estimated_hours"] == 2
        assert info["supported"] is True
        assert info["parallel_workers"] == 2


class TestDukascopyIntegration:
    """Integration Tests für komplette Workflows"""
    
    @pytest.fixture
    def integration_connector(self):
        """Connector für Integration Tests"""
        config = DukascopyConfig(
            max_retries=1,
            timeout=10,
            cache_dir="./test_cache"
        )
        return DukascopyConnector(parallel_workers=4, config=config)
    
    @patch.object(DukascopyConnector, '_download_with_retry')
    def test_full_pipeline_mock(self, mock_download, integration_connector):
        """Test komplette Pipeline mit Mock-Daten"""
        # Mock .bi5 Daten
        mock_bi5_data = struct.pack('>IIIII', 0, 110020, 110000, 1000000, 1000000)
        mock_download.return_value = mock_bi5_data
        
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        
        # Tick-Daten abrufen
        tick_data = integration_connector.fetch_tick_data("EURUSD", start, end)
        
        # OHLCV-Daten generieren
        ohlcv_data = integration_connector._convert_ticks_to_ohlcv(tick_data, "1m")
        
        # MarketData erstellen
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="1m",
            ohlcv_data=ohlcv_data,
            tick_data=tick_data
        )
        
        # Validierung
        validation_result = integration_connector.validate_data_integrity(market_data)
        
        assert len(tick_data) > 0
        assert len(ohlcv_data) > 0
        assert validation_result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])