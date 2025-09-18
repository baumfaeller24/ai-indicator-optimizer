#!/usr/bin/env python3
"""
MiniCPM-4.1-8B Integration Validation
Einfache Validierung ohne externe Dependencies
"""

import sys
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test alle wichtigen Imports"""
    print("🔍 Testing imports...")
    
    try:
        from ai_indicator_optimizer.ai.multimodal_ai import (
            MultimodalAI, ModelConfig, InferenceConfig, GPUMemoryManager, MiniCPMModelWrapper
        )
        print("✅ MultimodalAI imports successful")
        
        from ai_indicator_optimizer.ai.model_factory import ModelFactory
        print("✅ ModelFactory import successful")
        
        from ai_indicator_optimizer.ai.models import MultimodalInput, PatternAnalysis
        print("✅ Models import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_config():
    """Test ModelConfig"""
    print("\n🔧 Testing ModelConfig...")
    
    try:
        from ai_indicator_optimizer.ai.multimodal_ai import ModelConfig
        
        # Default config
        config = ModelConfig()
        assert config.model_name == "openbmb/MiniCPM-V-2_6"
        assert config.torch_dtype == torch.float16
        assert config.max_batch_size == 8
        print("✅ Default ModelConfig works")
        
        # Custom config
        custom_config = ModelConfig(
            max_batch_size=4,
            use_tensor_cores=True,
            enable_mixed_precision=True
        )
        assert custom_config.max_batch_size == 4
        assert custom_config.use_tensor_cores is True
        print("✅ Custom ModelConfig works")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelConfig test failed: {e}")
        return False

def test_gpu_memory_manager():
    """Test GPUMemoryManager"""
    print("\n💾 Testing GPUMemoryManager...")
    
    try:
        from ai_indicator_optimizer.ai.multimodal_ai import GPUMemoryManager
        
        manager = GPUMemoryManager()
        
        # Test memory stats
        stats = manager.get_memory_stats()
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "allocated" in stats
        print("✅ GPUMemoryManager.get_memory_stats() works")
        
        # Test optimization
        config = manager.optimize_for_model_loading(model_size_gb=8)
        assert isinstance(config, dict)
        assert "can_load_model" in config
        assert "recommended_batch_size" in config
        print("✅ GPUMemoryManager.optimize_for_model_loading() works")
        
        return True
        
    except Exception as e:
        print(f"❌ GPUMemoryManager test failed: {e}")
        return False

def test_model_wrapper():
    """Test MiniCPMModelWrapper"""
    print("\n🤖 Testing MiniCPMModelWrapper...")
    
    try:
        from ai_indicator_optimizer.ai.multimodal_ai import MiniCPMModelWrapper, ModelConfig
        
        config = ModelConfig()
        wrapper = MiniCPMModelWrapper(config)
        
        # Test initialization
        assert wrapper.config == config
        assert wrapper.model is None
        assert wrapper.processor is None
        assert wrapper.tokenizer is None
        print("✅ MiniCPMModelWrapper initialization works")
        
        # Test numerical data conversion
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        text = wrapper._numerical_data_to_text(test_data)
        assert "mean=" in text
        assert "std=" in text
        print("✅ MiniCPMModelWrapper._numerical_data_to_text() works")
        
        # Test empty data
        empty_text = wrapper._numerical_data_to_text(np.array([]))
        assert "No numerical data available" in empty_text
        print("✅ MiniCPMModelWrapper handles empty data")
        
        return True
        
    except Exception as e:
        print(f"❌ MiniCPMModelWrapper test failed: {e}")
        return False

def test_multimodal_ai():
    """Test MultimodalAI"""
    print("\n🧠 Testing MultimodalAI...")
    
    try:
        from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI, ModelConfig
        
        config = ModelConfig()
        ai = MultimodalAI(model_config=config)
        
        # Test initialization
        assert ai.model_config == config
        assert ai.analysis_count == 0
        assert ai.successful_analyses == 0
        print("✅ MultimodalAI initialization works")
        
        # Test trading prompts
        prompts = ai.trading_prompts
        required_prompts = ["pattern_analysis", "indicator_optimization", "strategy_generation"]
        assert all(prompt in prompts for prompt in required_prompts)
        print("✅ MultimodalAI trading prompts loaded")
        
        # Test numerical array extraction
        test_indicators = {
            "RSI": 65.5,
            "MACD": {"macd": [0.1, 0.2], "signal": [0.05, 0.15]},
            "SMA": [100.1, 100.2, 100.3]
        }
        
        numerical_array = ai._extract_numerical_array(test_indicators)
        assert isinstance(numerical_array, np.ndarray)
        assert len(numerical_array) > 0
        print("✅ MultimodalAI._extract_numerical_array() works")
        
        # Test pattern analysis parsing
        from ai_indicator_optimizer.ai.models import PatternAnalysis
        test_image = Image.new('RGB', (224, 224), color='red')
        
        response = "I see a clear double top pattern with high confidence"
        analysis = ai._parse_pattern_analysis_response(response, test_image)
        
        assert isinstance(analysis, PatternAnalysis)
        assert analysis.pattern_type == "double_top"
        assert analysis.confidence_score > 0.5
        print("✅ MultimodalAI._parse_pattern_analysis_response() works")
        
        return True
        
    except Exception as e:
        print(f"❌ MultimodalAI test failed: {e}")
        return False

def test_model_factory():
    """Test ModelFactory"""
    print("\n🏭 Testing ModelFactory...")
    
    try:
        from ai_indicator_optimizer.ai.model_factory import ModelFactory
        
        factory = ModelFactory()
        
        # Test initialization
        assert hasattr(factory, 'hardware_configs')
        assert "rtx_5090" in factory.hardware_configs
        assert "rtx_4090" in factory.hardware_configs
        assert "cpu" in factory.hardware_configs
        print("✅ ModelFactory initialization works")
        
        # Test config detection
        config = factory.detect_optimal_config()
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'max_batch_size')
        print("✅ ModelFactory.detect_optimal_config() works")
        
        # Test AI creation
        ai = factory.create_multimodal_ai(config_name="rtx_5090")
        assert ai is not None
        assert ai.model_config.max_batch_size == 8
        print("✅ ModelFactory.create_multimodal_ai() works")
        
        # Test inference config
        inference_config = factory.create_inference_config("trading_analysis", "balanced")
        assert hasattr(inference_config, 'temperature')
        assert hasattr(inference_config, 'max_new_tokens')
        print("✅ ModelFactory.create_inference_config() works")
        
        # Test available configs
        available_configs = factory.get_available_configs()
        assert isinstance(available_configs, dict)
        assert "rtx_5090" in available_configs
        print("✅ ModelFactory.get_available_configs() works")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelFactory test failed: {e}")
        return False

def test_hardware_detection():
    """Test Hardware Detection Integration"""
    print("\n🔧 Testing Hardware Detection...")
    
    try:
        from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
        
        detector = HardwareDetector()
        
        # Test Hardware Info Attributes
        assert hasattr(detector, 'cpu_info')
        assert hasattr(detector, 'gpu_info')
        assert hasattr(detector, 'memory_info')
        assert hasattr(detector, 'storage_info')
        print("✅ Hardware detector attributes exist")
        
        # Test Hardware Checks
        checks = detector.is_target_hardware()
        assert isinstance(checks, dict)
        assert "ryzen_9950x" in checks
        assert "rtx_5090" in checks
        assert "ram_192gb" in checks
        print("✅ Hardware checks work")
        
        # Test Worker Counts
        workers = detector.get_optimal_worker_counts()
        assert isinstance(workers, dict)
        assert "data_pipeline" in workers
        assert all(isinstance(count, int) and count > 0 for count in workers.values())
        print("✅ Worker count calculation works")
        
        # Test mit ModelFactory Integration
        from ai_indicator_optimizer.ai.model_factory import ModelFactory
        factory = ModelFactory()
        
        # Hardware-basierte Konfiguration
        optimal_config = factory.detect_optimal_config()
        print(f"✅ Optimal config detected: {optimal_config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection test failed: {e}")
        return False

def test_integration_workflow():
    """Test kompletter Integration-Workflow"""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        from ai_indicator_optimizer.ai.model_factory import ModelFactory
        from ai_indicator_optimizer.ai.models import PatternAnalysis
        
        # 1. Factory erstellen
        factory = ModelFactory()
        
        # 2. AI erstellen
        ai = factory.create_multimodal_ai(config_name="rtx_5090")
        
        # 3. Inference Config erstellen
        inference_config = factory.create_inference_config("trading_analysis", "balanced")
        
        # 4. Test-Daten vorbereiten
        test_image = Image.new('RGB', (448, 448), color='green')
        test_indicators = {
            "RSI": 65.5,
            "MACD": {"macd": 0.15, "signal": 0.10},
            "SMA_20": 1.1050
        }
        
        # 5. Model Info abrufen (ohne Model zu laden)
        model_info = ai.get_model_info()
        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        print("✅ Model info retrieval works")
        
        # 6. Numerical array extraction
        numerical_array = ai._extract_numerical_array(test_indicators)
        assert len(numerical_array) > 0
        print("✅ Numerical data processing works")
        
        # 7. Pattern analysis parsing (mock)
        mock_response = "Ascending triangle pattern detected with medium confidence"
        analysis = ai._parse_pattern_analysis_response(mock_response, test_image)
        assert isinstance(analysis, PatternAnalysis)
        print("✅ Pattern analysis parsing works")
        
        print("✅ Complete integration workflow successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False

def main():
    """Hauptvalidierung"""
    print("🚀 MiniCPM-4.1-8B Integration Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("ModelConfig", test_model_config),
        ("GPUMemoryManager", test_gpu_memory_manager),
        ("MiniCPMModelWrapper", test_model_wrapper),
        ("MultimodalAI", test_multimodal_ai),
        ("ModelFactory", test_model_factory),
        ("Hardware Detection", test_hardware_detection),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 MiniCPM-4.1-8B Integration is ready!")
        
        # Hardware Info
        try:
            from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
            detector = HardwareDetector()
            
            print(f"\n💻 System Info:")
            if detector.cpu_info:
                print(f"   CPU: {detector.cpu_info.model}")
                print(f"   Cores: {detector.cpu_info.cores_logical}")
            
            if detector.memory_info:
                print(f"   RAM: {detector.memory_info.total // (1024**3)} GB")
            
            if detector.gpu_info and len(detector.gpu_info) > 0:
                gpu = detector.gpu_info[0]
                print(f"   GPU: {gpu.name}")
                print(f"   VRAM: {gpu.memory_total // (1024**3)} GB")
            else:
                print(f"   GPU: Not available")
                
        except Exception as e:
            print(f"   Hardware info unavailable: {e}")
    else:
        print(f"\n⚠️ {failed} tests failed. Check implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)