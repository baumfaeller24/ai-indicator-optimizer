"""
Unit Tests für Multimodal Data Processing Pipeline
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from ai_indicator_optimizer.data.processor import (
    IndicatorCalculator, 
    ChartRenderer, 
    MultimodalDatasetBuilder,
    DataProcessor,
    ProcessingConfig
)
from ai_indicator_optimizer.data.models import OHLCVData, IndicatorData, MarketData


class TestIndicatorCalculator:
    """Tests für IndicatorCalculator"""
    
    @pytest.fixture
    def calculator(self):
        return IndicatorCalculator(cpu_workers=4)  # Reduziert für Tests
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generiert Sample OHLCV-Daten"""
        data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        base_price = 1.1000
        
        for i in range(100):  # 100 Candles für Indikator-Berechnungen
            timestamp = base_time + timedelta(minutes=i)
            
            # Simuliere Preisbewegung
            price_change = np.random.normal(0, 0.0001)
            open_price = base_price + price_change
            close_price = open_price + np.random.normal(0, 0.0001)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0001))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0001))
            
            data.append(OHLCVData(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000 + np.random.randint(-100, 100)
            ))
            
            base_price = close_price
        
        return data
    
    def test_calculate_all_indicators(self, calculator, sample_ohlcv_data):
        """Test komplette Indikator-Berechnung"""
        indicators = calculator.calculate_all_indicators(sample_ohlcv_data)
        
        assert isinstance(indicators, IndicatorData)
        assert indicators.rsi is not None
        assert len(indicators.rsi) == len(sample_ohlcv_data)
        assert indicators.macd is not None
        assert 'macd' in indicators.macd
        assert 'signal' in indicators.macd
        assert 'histogram' in indicators.macd
        assert indicators.bollinger is not None
        assert 'upper' in indicators.bollinger
        assert 'middle' in indicators.bollinger
        assert 'lower' in indicators.bollinger
    
    def test_rsi_calculation(self, calculator, sample_ohlcv_data):
        """Test RSI-Berechnung"""
        df = calculator._ohlcv_to_dataframe(sample_ohlcv_data)
        rsi = calculator._calculate_rsi(df, 14)
        
        assert len(rsi) == len(sample_ohlcv_data)
        assert all(0 <= value <= 100 for value in rsi if not np.isnan(value))
    
    def test_macd_calculation(self, calculator, sample_ohlcv_data):
        """Test MACD-Berechnung"""
        df = calculator._ohlcv_to_dataframe(sample_ohlcv_data)
        macd = calculator._calculate_macd(df, 12, 26, 9)
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        assert len(macd['macd']) == len(sample_ohlcv_data)
    
    def test_bollinger_bands_calculation(self, calculator, sample_ohlcv_data):
        """Test Bollinger Bands-Berechnung"""
        df = calculator._ohlcv_to_dataframe(sample_ohlcv_data)
        bollinger = calculator._calculate_bollinger_bands(df, 20, 2)
        
        assert 'upper' in bollinger
        assert 'middle' in bollinger
        assert 'lower' in bollinger
        
        # Upper sollte immer >= Middle >= Lower sein
        for i in range(len(bollinger['upper'])):
            if not (np.isnan(bollinger['upper'][i]) or np.isnan(bollinger['middle'][i]) or np.isnan(bollinger['lower'][i])):
                assert bollinger['upper'][i] >= bollinger['middle'][i] >= bollinger['lower'][i]
    
    def test_empty_data(self, calculator):
        """Test mit leeren Daten"""
        indicators = calculator.calculate_all_indicators([])
        
        assert isinstance(indicators, IndicatorData)
        assert indicators.rsi is None
        assert indicators.macd is None


class TestChartRenderer:
    """Tests für ChartRenderer"""
    
    @pytest.fixture
    def config(self):
        return ProcessingConfig(
            chart_width=512,
            chart_height=384,
            gpu_acceleration=False  # Für Tests deaktiviert
        )
    
    @pytest.fixture
    def renderer(self, config):
        return ChartRenderer(config)
    
    @pytest.fixture
    def sample_data_with_indicators(self):
        """Sample-Daten mit Indikatoren"""
        # Einfache OHLCV-Daten
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        for i in range(50):
            ohlcv_data.append(OHLCVData(
                timestamp=base_time + timedelta(minutes=i),
                open=1.1000 + i * 0.0001,
                high=1.1005 + i * 0.0001,
                low=1.0995 + i * 0.0001,
                close=1.1002 + i * 0.0001,
                volume=1000
            ))
        
        # Mock-Indikatoren
        indicators = IndicatorData(
            rsi=[50 + i for i in range(50)],
            macd={
                'macd': [0.001 * i for i in range(50)],
                'signal': [0.0008 * i for i in range(50)],
                'histogram': [0.0002 * i for i in range(50)]
            },
            bollinger={
                'upper': [1.1010 + i * 0.0001 for i in range(50)],
                'middle': [1.1000 + i * 0.0001 for i in range(50)],
                'lower': [1.0990 + i * 0.0001 for i in range(50)]
            },
            sma={20: [1.1000 + i * 0.0001 for i in range(50)]},
            ema={12: [1.1001 + i * 0.0001 for i in range(50)]}
        )
        
        return ohlcv_data, indicators
    
    def test_generate_candlestick_chart(self, renderer, sample_data_with_indicators):
        """Test Candlestick-Chart-Generierung"""
        ohlcv_data, indicators = sample_data_with_indicators
        
        chart = renderer.generate_candlestick_chart(ohlcv_data, indicators)
        
        assert isinstance(chart, Image.Image)
        assert chart.size == (renderer.config.chart_width, renderer.config.chart_height)
        assert chart.mode == 'RGB'
    
    @patch('matplotlib.pyplot.show')  # Verhindert GUI-Anzeige in Tests
    def test_generate_multiple_timeframes(self, mock_show, renderer, sample_data_with_indicators):
        """Test Multi-Timeframe Chart-Generierung"""
        ohlcv_data, indicators = sample_data_with_indicators
        market_data = MarketData(symbol="EURUSD", timeframe="1m", ohlcv_data=ohlcv_data)
        
        charts = renderer.generate_multiple_timeframes(market_data, indicators)
        
        assert len(charts) == 4  # 4 Timeframes
        assert all(isinstance(chart, Image.Image) for chart in charts)
    
    def test_gpu_enhancement_fallback(self, renderer):
        """Test GPU-Enhancement Fallback"""
        # Mock-Image Array
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Sollte ohne Fehler funktionieren, auch ohne GPU
        enhanced = renderer._gpu_enhance_image(image_array)
        
        assert enhanced.shape == image_array.shape
        assert enhanced.dtype == np.uint8


class TestMultimodalDatasetBuilder:
    """Tests für MultimodalDatasetBuilder"""
    
    @pytest.fixture
    def config(self):
        return ProcessingConfig()
    
    @pytest.fixture
    def builder(self, config):
        return MultimodalDatasetBuilder(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample MarketData"""
        ohlcv_data = [
            OHLCVData(
                timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=timezone.utc),
                open=1.1000 + i * 0.0001,
                high=1.1005 + i * 0.0001,
                low=1.0995 + i * 0.0001,
                close=1.1002 + i * 0.0001,
                volume=1000
            ) for i in range(10)
        ]
        
        return MarketData(
            symbol="EURUSD",
            timeframe="1m",
            ohlcv_data=ohlcv_data
        )
    
    @pytest.fixture
    def sample_indicators(self):
        """Sample IndicatorData"""
        return IndicatorData(
            rsi=[50, 55, 60, 65, 70, 75, 80, 75, 70, 65],
            macd={
                'macd': [0.001 * i for i in range(10)],
                'signal': [0.0008 * i for i in range(10)],
                'histogram': [0.0002 * i for i in range(10)]
            }
        )
    
    @pytest.fixture
    def sample_chart_images(self):
        """Sample Chart Images"""
        return [
            Image.new('RGB', (224, 224), color='black'),
            Image.new('RGB', (224, 224), color='white')
        ]
    
    def test_create_training_sample(self, builder, sample_market_data, sample_indicators, sample_chart_images):
        """Test Training-Sample-Erstellung"""
        sample = builder.create_training_sample(
            sample_market_data, sample_indicators, sample_chart_images
        )
        
        assert 'numerical_features' in sample
        assert 'chart_images' in sample
        assert 'text_descriptions' in sample
        assert 'metadata' in sample
        
        assert isinstance(sample['numerical_features'], np.ndarray)
        assert len(sample['chart_images']) == 2
        assert isinstance(sample['text_descriptions'], list)
        assert sample['metadata']['symbol'] == 'EURUSD'
    
    def test_extract_numerical_features(self, builder, sample_market_data, sample_indicators):
        """Test numerische Feature-Extraktion"""
        features = builder._extract_numerical_features(sample_market_data, sample_indicators)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0
    
    def test_normalize_features(self, builder):
        """Test Feature-Normalisierung"""
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        normalized = builder._normalize_features(features)
        
        assert isinstance(normalized, np.ndarray)
        assert abs(np.mean(normalized)) < 1e-6  # Mean sollte ~0 sein
        assert abs(np.std(normalized) - 1.0) < 1e-6  # Std sollte ~1 sein
    
    def test_preprocess_images(self, builder, sample_chart_images):
        """Test Image-Preprocessing"""
        processed = builder._preprocess_images(sample_chart_images)
        
        assert len(processed) == 2
        assert all(isinstance(img, np.ndarray) for img in processed)
        assert all(img.shape == (224, 224, 3) for img in processed)
        assert all(img.dtype == np.float32 for img in processed)
        assert all(0 <= img.max() <= 1 for img in processed)
    
    def test_generate_text_descriptions(self, builder, sample_market_data, sample_indicators):
        """Test Text-Beschreibungs-Generierung"""
        descriptions = builder._generate_text_descriptions(sample_market_data, sample_indicators)
        
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0
        assert all(isinstance(desc, str) for desc in descriptions)
        assert any('EURUSD' in desc for desc in descriptions)


class TestDataProcessor:
    """Integration Tests für DataProcessor"""
    
    @pytest.fixture
    def config(self):
        return ProcessingConfig(
            cpu_workers=4,
            gpu_acceleration=False,
            chart_width=256,
            chart_height=192
        )
    
    @pytest.fixture
    def processor(self, config):
        return DataProcessor(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample MarketData für Integration Tests"""
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        for i in range(50):
            ohlcv_data.append(OHLCVData(
                timestamp=base_time + timedelta(minutes=i),
                open=1.1000 + i * 0.0001,
                high=1.1005 + i * 0.0001,
                low=1.0995 + i * 0.0001,
                close=1.1002 + i * 0.0001,
                volume=1000 + i * 10
            ))
        
        return MarketData(
            symbol="EURUSD",
            timeframe="1m",
            ohlcv_data=ohlcv_data
        )
    
    @patch('matplotlib.pyplot.show')
    def test_process_market_data_complete(self, mock_show, processor, sample_market_data):
        """Test komplette Marktdaten-Verarbeitung"""
        result = processor.process_market_data(sample_market_data)
        
        assert 'indicators' in result
        assert 'chart_images' in result
        assert 'training_sample' in result
        assert 'processing_stats' in result
        
        # Indikatoren
        indicators = result['indicators']
        assert isinstance(indicators, IndicatorData)
        assert indicators.rsi is not None
        
        # Charts
        charts = result['chart_images']
        assert len(charts) > 0
        assert all(isinstance(chart, Image.Image) for chart in charts)
        
        # Training Sample
        training_sample = result['training_sample']
        assert 'numerical_features' in training_sample
        assert 'chart_images' in training_sample
        assert 'text_descriptions' in training_sample
        
        # Stats
        stats = result['processing_stats']
        assert stats['data_points'] == 50
        assert stats['indicators_calculated'] == 8
    
    def test_backward_compatibility(self, processor, sample_market_data):
        """Test Backward Compatibility"""
        # Test alte Interface-Methoden
        indicators = processor.calculate_indicators(sample_market_data.ohlcv_data)
        assert isinstance(indicators, IndicatorData)
        
        # Multimodal Dataset
        dataset = processor.create_multimodal_dataset(sample_market_data)
        assert 'indicators' in dataset


class TestPerformanceOptimizations:
    """Tests für Performance-Optimierungen"""
    
    def test_parallel_indicator_calculation(self):
        """Test parallele Indikator-Berechnung"""
        calculator = IndicatorCalculator(cpu_workers=4)
        
        # Große Datenmenge
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        for i in range(1000):  # 1000 Candles
            ohlcv_data.append(OHLCVData(
                timestamp=base_time + timedelta(minutes=i),
                open=1.1000 + np.random.normal(0, 0.001),
                high=1.1005 + np.random.normal(0, 0.001),
                low=1.0995 + np.random.normal(0, 0.001),
                close=1.1002 + np.random.normal(0, 0.001),
                volume=1000
            ))
        
        import time
        start_time = time.time()
        indicators = calculator.calculate_all_indicators(ohlcv_data)
        end_time = time.time()
        
        # Sollte schnell sein (< 5 Sekunden für 1000 Candles)
        assert end_time - start_time < 5.0
        assert indicators.rsi is not None
        assert len(indicators.rsi) == 1000
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        """Test CPU-Fallback wenn GPU nicht verfügbar"""
        config = ProcessingConfig(gpu_acceleration=True)
        renderer = ChartRenderer(config)
        
        assert renderer.device.type == 'cpu'
        assert not renderer.use_gpu


if __name__ == "__main__":
    pytest.main([__file__, "-v"])