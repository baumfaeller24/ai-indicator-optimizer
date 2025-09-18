"""
Tests für MiniCPM-4.1-8B Integration
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from ai_indicator_optimizer.ai.multimodal_ai import (
    MultimodalAI, ModelConfig, InferenceConfig, GPUMemoryManager, MiniCPMModelWrapper
)
from ai_indicator_optimizer.ai.model_factory import ModelFactory
from ai_indicator_optimizer.ai.models import MultimodalInput, PatternAnalysis


class TestGPUMemoryManager:
    """Tests für GPU Memory Manager"""
    
    def test_memory_manager_initialization(self):
        """Test GPU Memory Manager Initialisierung"""
        manager = GPUMemoryManager(device_id=0)
        
        assert manager.device_id == 0
        assert manager.device.type in ["cuda", "cpu"]
    
    def test_memory_stats_cpu_fallback(self):
        """Test Memory Stats bei CPU-only System"""
        with patch('torch.cuda.is_available', return_value=False):
            manager = GPUMemoryManager()
            stats = manager.get_memory_stats()
            
            expected_keys = ["total", "allocated", "cached", "free"]
            assert all(key in stats for key in expected_keys)
            assert stats["total"] == 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB
    @patch('torch.cuda.memory_reserved', return_value=10 * 1024**3)  # 10GB
    @patch('torch.cuda.get_device_properties')
    def test_memory_stats_gpu(self, mock_props, mock_reserved, mock_allocated, mock_available):
        """Test Memory Stats mit GPU"""
        # Mock GPU Properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 32 * 1024**3  # 32GB
        mock_props.return_value = mock_device_props
        
        manager = GPUMemoryManager()
        stats = manager.get_memory_stats()
        
        assert stats["allocated_gb"] == 8
        assert stats["free_gb"] > 0
        assert "total" in stats
    
    def test_optimize_for_model_loading(self):
        """Test Model Loading Optimierung"""
        manager = GPUMemoryManager()
        
        # Mock verfügbares Memory
        with patch.object(manager, 'get_memory_stats', return_value={
            "free_gb": 20,
            "total": 32 * 1024**3,
            "allocated": 8 * 1024**3,
            "cached": 10 * 1024**3,
            "free": 22 * 1024**3
        }):
            config = manager.optimize_for_model_loading(model_size_gb=8)
            
            assert config["can_load_model"] is True
            assert config["recommended_batch_size"] >= 1
            assert "use_gradient_checkpointing" in config


class TestModelConfig:
    """Tests für Model-Konfiguration"""
    
    def test_default_config(self):
        """Test Standard-Konfiguration"""
        config = ModelConfig()
        
        assert config.model_name == "openbmb/MiniCPM-V-2_6"
        assert config.torch_dtype == torch.float16
        assert config.trust_remote_code is True
        assert config.max_batch_size == 8
    
    def test_rtx_5090_config(self):
        """Test RTX 5090 optimierte Konfiguration"""
        config = ModelConfig(
            use_tensor_cores=True,
            enable_mixed_precision=True,
            max_batch_size=8,
            image_resolution=448
        )
        
        assert config.use_tensor_cores is True
        assert config.enable_mixed_precision is True
        assert config.max_batch_size == 8
        assert config.image_resolution == 448


class TestMiniCPMModelWrapper:
    """Tests für MiniCPM Model Wrapper"""
    
    def test_wrapper_initialization(self):
        """Test Wrapper Initialisierung"""
        config = ModelConfig()
        wrapper = MiniCPMModelWrapper(config)
        
        assert wrapper.config == config
        assert wrapper.model is None
        assert wrapper.processor is None
        assert wrapper.tokenizer is None
    
    @patch('ai_indicator_optimizer.ai.multimodal_ai.AutoModel')
    @patch('ai_indicator_optimizer.ai.multimodal_ai.AutoProcessor')
    @patch('ai_indicator_optimizer.ai.multimodal_ai.AutoTokenizer')
    def test_model_loading_success(self, mock_tokenizer, mock_processor, mock_model):
        """Test erfolgreiches Model Loading"""
        # Mock Model Components
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        config = ModelConfig()
        wrapper = MiniCPMModelWrapper(config)
        
        # Mock GPU Memory Manager
        with patch.object(wrapper.gpu_manager, 'optimize_for_model_loading', return_value={
            "can_load_model": True,
            "use_4bit_loading": False,
            "use_8bit_loading": False
        }):
            success = wrapper.load_model()
            
            assert success is True
            assert wrapper.model is not None
            assert wrapper.processor is not None
            assert wrapper.tokenizer is not None
    
    def test_numerical_data_to_text(self):
        """Test numerische Daten zu Text Konvertierung"""
        config = ModelConfig()
        wrapper = MiniCPMModelWrapper(config)
        
        # Test mit validen Daten
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        text = wrapper._numerical_data_to_text(data)
        
        assert "mean=" in text
        assert "std=" in text
        assert "range=" in text
        
        # Test mit leeren Daten
        empty_data = np.array([])
        text_empty = wrapper._numerical_data_to_text(empty_data)
        assert "No numerical data available" in text_empty
    
    def test_prepare_multimodal_input(self):
        """Test multimodale Eingabe-Vorbereitung"""
        config = ModelConfig()
        wrapper = MiniCPMModelWrapper(config)
        
        # Mock Processor
        mock_processor = Mock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        wrapper.processor = mock_processor
        
        # Erstelle Test-Daten
        test_image = Image.new('RGB', (224, 224), color='red')
        numerical_data = np.array([1.0, 2.0, 3.0])
        text_prompt = "Analyze this chart"
        
        # Test Eingabe-Vorbereitung
        inputs = wrapper.prepare_multimodal_input(
            chart_images=[test_image],
            numerical_data=numerical_data,
            text_prompt=text_prompt
        )
        
        # Verify processor was called
        mock_processor.assert_called_once()


class TestMultimodalAI:
    """Tests für MultimodalAI Hauptklasse"""
    
    def test_ai_initialization(self):
        """Test MultimodalAI Initialisierung"""
        config = ModelConfig()
        ai = MultimodalAI(model_config=config)
        
        assert ai.model_config == config
        assert ai.analysis_count == 0
        assert ai.successful_analyses == 0
        assert "pattern_analysis" in ai.trading_prompts
    
    def test_trading_prompts_loading(self):
        """Test Trading-Prompts Loading"""
        ai = MultimodalAI()
        prompts = ai.trading_prompts
        
        required_prompts = ["pattern_analysis", "indicator_optimization", "strategy_generation"]
        assert all(prompt in prompts for prompt in required_prompts)
        
        # Verify prompt content
        assert "chart patterns" in prompts["pattern_analysis"].lower()
        assert "parameters" in prompts["indicator_optimization"].lower()
        assert "strategy" in prompts["strategy_generation"].lower()
    
    def test_extract_numerical_array(self):
        """Test numerisches Array Extraktion"""
        ai = MultimodalAI()
        
        # Test mit verschiedenen Indikator-Formaten
        indicators = {
            "RSI": 65.5,
            "MACD": {
                "macd": [0.1, 0.2, 0.3],
                "signal": [0.05, 0.15, 0.25]
            },
            "SMA": [100.1, 100.2, 100.3, 100.4, 100.5],
            "Volume": 1000000
        }
        
        numerical_array = ai._extract_numerical_array(indicators)
        
        assert isinstance(numerical_array, np.ndarray)
        assert len(numerical_array) > 0
        assert numerical_array.dtype == np.float32
    
    def test_parse_pattern_analysis_response(self):
        """Test Pattern Analysis Response Parsing"""
        ai = MultimodalAI()
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Test verschiedene Pattern-Responses
        test_cases = [
            ("I see a clear double top pattern with high confidence", "double_top", 0.8),
            ("This shows a triangle formation", "triangle", 0.5),
            ("Support level is visible", "support_resistance", 0.5),
            ("No clear pattern detected", "unknown", 0.5)
        ]
        
        for response_text, expected_pattern, min_confidence in test_cases:
            analysis = ai._parse_pattern_analysis_response(response_text, test_image)
            
            assert isinstance(analysis, PatternAnalysis)
            assert analysis.pattern_type == expected_pattern
            assert analysis.confidence_score >= min_confidence - 0.1  # Toleranz
    
    def test_parse_optimization_response(self):
        """Test Optimierungs-Response Parsing"""
        ai = MultimodalAI()
        
        response = "Based on analysis, I recommend RSI period=21, MACD fast=8, slow=21"
        optimized_params = ai._parse_optimization_response(response)
        
        assert isinstance(optimized_params, list)
        assert len(optimized_params) > 0
        
        # Verify RSI parameter
        rsi_param = next((p for p in optimized_params if p.indicator_name == "RSI"), None)
        assert rsi_param is not None
        assert "period" in rsi_param.parameters
    
    def test_extract_conditions(self):
        """Test Trading-Conditions Extraktion"""
        ai = MultimodalAI()
        
        # Test Entry Conditions
        entry_response = "Enter when RSI is below 30 and MACD crosses above signal line"
        entry_conditions = ai._extract_conditions(entry_response, "entry")
        
        assert isinstance(entry_conditions, list)
        assert len(entry_conditions) > 0
        
        # Test Exit Conditions
        exit_response = "Exit when RSI is above 70 or take profit at 2%"
        exit_conditions = ai._extract_conditions(exit_response, "exit")
        
        assert isinstance(exit_conditions, list)
        assert len(exit_conditions) > 0
    
    @patch.object(MiniCPMModelWrapper, 'load_model', return_value=True)
    def test_model_loading(self, mock_load):
        """Test Model Loading"""
        ai = MultimodalAI()
        success = ai.load_model()
        
        assert success is True
        mock_load.assert_called_once()
    
    def test_get_model_info(self):
        """Test Model Info Abruf"""
        ai = MultimodalAI()
        
        # Mock Model Wrapper
        ai.model_wrapper.device = torch.device("cpu")
        ai.model_wrapper.gpu_manager.get_memory_stats = Mock(return_value={"allocated_gb": 0})
        ai.model_wrapper.get_performance_stats = Mock(return_value={"avg_inference_time": 1.0})
        
        info = ai.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "loaded" in info
        assert "analysis_count" in info


class TestModelFactory:
    """Tests für Model Factory"""
    
    def test_factory_initialization(self):
        """Test Factory Initialisierung"""
        factory = ModelFactory()
        
        assert factory.resource_manager is None
        assert hasattr(factory, 'hardware_configs')
        assert "rtx_5090" in factory.hardware_configs
    
    def test_hardware_configs(self):
        """Test Hardware-Konfigurationen"""
        factory = ModelFactory()
        configs = factory.hardware_configs
        
        # Verify all expected configs exist
        expected_configs = ["rtx_5090", "rtx_4090", "rtx_3080", "cpu"]
        assert all(config in configs for config in expected_configs)
        
        # Verify RTX 5090 config
        rtx_5090_config = configs["rtx_5090"]
        assert rtx_5090_config.max_batch_size == 8
        assert rtx_5090_config.use_tensor_cores is True
        assert rtx_5090_config.load_in_8bit is False
    
    @patch('ai_indicator_optimizer.ai.model_factory.HardwareDetector')
    def test_detect_optimal_config(self, mock_detector):
        """Test automatische Konfigurationserkennung"""
        factory = ModelFactory()
        
        # Mock RTX 5090 Detection
        mock_detector.return_value.detect_hardware.return_value = {
            "gpu": {
                "available": True,
                "name": "NVIDIA GeForce RTX 5090",
                "memory_gb": 32
            }
        }
        
        config = factory.detect_optimal_config()
        
        assert config.max_batch_size == 8
        assert config.use_tensor_cores is True
    
    def test_create_multimodal_ai(self):
        """Test MultimodalAI Erstellung"""
        factory = ModelFactory()
        
        # Test mit vordefinierter Konfiguration
        ai = factory.create_multimodal_ai(config_name="rtx_5090")
        
        assert isinstance(ai, MultimodalAI)
        assert ai.model_config.max_batch_size == 8
    
    def test_create_inference_config(self):
        """Test Inference-Konfiguration Erstellung"""
        factory = ModelFactory()
        
        # Test verschiedene Task-Typen
        trading_config = factory.create_inference_config("trading_analysis", "balanced")
        pattern_config = factory.create_inference_config("pattern_recognition", "fast")
        strategy_config = factory.create_inference_config("strategy_generation", "quality")
        
        assert isinstance(trading_config, InferenceConfig)
        assert isinstance(pattern_config, InferenceConfig)
        assert isinstance(strategy_config, InferenceConfig)
        
        # Verify different parameters
        assert pattern_config.temperature < trading_config.temperature
        assert strategy_config.max_new_tokens > pattern_config.max_new_tokens
    
    def test_get_available_configs(self):
        """Test verfügbare Konfigurationen abrufen"""
        factory = ModelFactory()
        configs = factory.get_available_configs()
        
        assert isinstance(configs, dict)
        assert "rtx_5090" in configs
        assert "model_name" in configs["rtx_5090"]
        assert "max_batch_size" in configs["rtx_5090"]


class TestIntegrationScenarios:
    """Integration Tests für realistische Szenarien"""
    
    def test_complete_analysis_workflow(self):
        """Test kompletter Analyse-Workflow"""
        # Erstelle Test-Daten
        test_image = Image.new('RGB', (448, 448), color='green')
        numerical_indicators = {
            "RSI": 65.5,
            "MACD": {"macd": 0.1, "signal": 0.05},
            "SMA_20": 100.5
        }
        
        # Mock MultimodalAI
        ai = MultimodalAI()
        
        # Mock Model Loading
        with patch.object(ai, 'load_model', return_value=True):
            # Mock Model Wrapper
            ai.model_wrapper.model = Mock()
            ai.model_wrapper.prepare_multimodal_input = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
            ai.model_wrapper.generate_response = Mock(return_value="Double top pattern detected with high confidence")
            
            # Test Pattern Analysis
            analysis = ai.analyze_chart_pattern(test_image, numerical_indicators)
            
            assert isinstance(analysis, PatternAnalysis)
            assert analysis.pattern_type == "double_top"
            assert ai.analysis_count == 1
    
    def test_factory_to_ai_workflow(self):
        """Test Factory zu AI Workflow"""
        factory = ModelFactory()
        
        # Erstelle AI mit Factory
        ai = factory.create_multimodal_ai(config_name="rtx_5090")
        inference_config = factory.create_inference_config("trading_analysis", "balanced")
        
        assert isinstance(ai, MultimodalAI)
        assert isinstance(inference_config, InferenceConfig)
        assert ai.model_config.max_batch_size == 8
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_integration(self):
        """Test GPU Memory Integration (nur wenn GPU verfügbar)"""
        manager = GPUMemoryManager()
        stats = manager.get_memory_stats()
        
        # Verify GPU stats are realistic
        assert stats["total"] > 0
        assert stats["allocated_gb"] >= 0
        assert stats["free_gb"] >= 0
    
    def test_error_handling_scenarios(self):
        """Test Error Handling in verschiedenen Szenarien"""
        ai = MultimodalAI()
        
        # Test ohne geladenes Model
        test_image = Image.new('RGB', (224, 224), color='red')
        analysis = ai.analyze_chart_pattern(test_image)
        
        assert analysis.pattern_type == "analysis_failed"
        assert analysis.confidence_score == 0.0
        assert "Analysis failed" in analysis.description


if __name__ == "__main__":
    # Führe Tests aus
    pytest.main([__file__, "-v"])