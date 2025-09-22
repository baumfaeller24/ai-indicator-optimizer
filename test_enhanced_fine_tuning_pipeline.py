#!/usr/bin/env python3
"""
Test Enhanced Fine-Tuning Pipeline - Task 6
AI-Indicator-Optimizer

Tests:
- BarDatasetBuilder für Forward-Return-Label-Generierung
- Enhanced Feature Extraction mit technischen Indikatoren
- Polars-basierte Parquet-Export-Funktionalität
- GPU-optimierte Training-Loop mit Mixed-Precision
- Model-Checkpointing und Resume-Funktionalität
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import components
from ai_indicator_optimizer.ai.fine_tuning import (
    EnhancedFineTuningManager, 
    FineTuningConfig,
    create_fine_tuning_config,
    create_fine_tuning_manager
)
from ai_indicator_optimizer.dataset.bar_dataset_builder import BarDatasetBuilder
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor

# Mock Bar class for testing (no Nautilus dependency)
from dataclasses import dataclass
from typing import Any

@dataclass
class MockBar:
    """Mock Bar class for testing without Nautilus dependency"""
    bar_type: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_event: int
    ts_init: int

# Use MockBar instead of Nautilus Bar
Bar = MockBar


def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_enhanced_fine_tuning.log')
        ]
    )
    return logging.getLogger(__name__)


def generate_test_bars(num_bars: int = 1000) -> list:
    """Generate synthetic test bars"""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_bars} test bars...")
    
    bars = []
    base_price = 1.1000
    current_time = int(time.time() * 1e9)
    
    for i in range(num_bars):
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.0001)  # Small random walk
        trend = 0.00001 * np.sin(i / 100)  # Small trend component
        
        open_price = base_price + price_change
        high_price = open_price + abs(np.random.normal(0, 0.0002))
        low_price = open_price - abs(np.random.normal(0, 0.0002))
        close_price = open_price + price_change + trend
        volume = max(1000 + np.random.normal(0, 500), 100)
        
        # Update base price for next bar
        base_price = close_price
        
        # Create mock bar (no Nautilus dependency)
        bar = Bar(
            bar_type="EUR/USD.SIM-1-MINUTE-BID-EXTERNAL",
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            ts_event=current_time + i * 60 * 1e9,  # 1 minute intervals
            ts_init=current_time + i * 60 * 1e9
        )
        
        bars.append(bar)
    
    logger.info(f"Generated {len(bars)} test bars")
    return bars


def test_dataset_builder():
    """Test BarDatasetBuilder functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing BarDatasetBuilder...")
    
    try:
        # Create dataset builder
        builder = BarDatasetBuilder(
            horizon=5,
            min_bars=10,
            return_thresholds={
                "buy_threshold": 0.0003,
                "sell_threshold": -0.0003
            },
            include_technical_indicators=True
        )
        
        # Generate test bars
        test_bars = generate_test_bars(100)
        
        # Process bars
        for bar in test_bars:
            builder.on_bar(bar)
        
        # Get statistics
        stats = builder.get_stats()
        logger.info(f"Dataset Builder Stats: {stats}")
        
        # Export to Parquet
        output_path = "test_logs/test_dataset_builder.parquet"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        success = builder.to_parquet(output_path)
        
        if success:
            logger.info("✅ BarDatasetBuilder test passed")
            return output_path
        else:
            logger.error("❌ BarDatasetBuilder export failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ BarDatasetBuilder test failed: {e}")
        return None


def test_enhanced_feature_extractor():
    """Test Enhanced Feature Extractor"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Enhanced Feature Extractor...")
    
    try:
        # Create feature extractor
        extractor = EnhancedFeatureExtractor({
            "include_time_features": True,
            "include_technical_indicators": True,
            "include_pattern_features": True,
            "include_volatility_features": True
        })
        
        # Generate test bar
        test_bars = generate_test_bars(50)
        
        # Extract features from multiple bars
        all_features = []
        for bar in test_bars:
            features = extractor.extract_enhanced_features(bar)
            all_features.append(features)
        
        # Check feature completeness
        if all_features:
            sample_features = all_features[-1]
            logger.info(f"Sample features: {list(sample_features.keys())}")
            logger.info(f"Feature count: {len(sample_features)}")
            
            # Check for required features
            required_features = [
                'open', 'high', 'low', 'close', 'volume',
                'hour', 'minute', 'day_of_week',
                'sma_5', 'rsi_14', 'volatility_10'
            ]
            
            missing_features = [f for f in required_features if f not in sample_features]
            
            if not missing_features:
                logger.info("✅ Enhanced Feature Extractor test passed")
                return True
            else:
                logger.error(f"❌ Missing features: {missing_features}")
                return False
        else:
            logger.error("❌ No features extracted")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Feature Extractor test failed: {e}")
        return False


def test_fine_tuning_config():
    """Test Fine-Tuning Configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Fine-Tuning Configuration...")
    
    try:
        # Create config
        config = create_fine_tuning_config(
            model_name="test-model",
            output_dir="./test_models",
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=2,
            use_mixed_precision=True,
            forward_return_horizon=5,
            include_technical_indicators=True
        )
        
        # Validate config
        assert config.model_name == "test-model"
        assert config.learning_rate == 1e-4
        assert config.use_mixed_precision == True
        assert config.forward_return_horizon == 5
        
        logger.info("✅ Fine-Tuning Configuration test passed")
        return config
        
    except Exception as e:
        logger.error(f"❌ Fine-Tuning Configuration test failed: {e}")
        return None


def test_fine_tuning_manager_setup():
    """Test Fine-Tuning Manager Setup"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Fine-Tuning Manager Setup...")
    
    try:
        # Create config
        config = create_fine_tuning_config(
            output_dir="./test_models/fine_tuning",
            batch_size=2,
            num_epochs=1,
            use_mixed_precision=torch.cuda.is_available(),
            dataloader_num_workers=2
        )
        
        # Create manager
        manager = create_fine_tuning_manager(config)
        
        # Test setup
        setup_success = manager.setup_model()
        
        if setup_success:
            logger.info("✅ Fine-Tuning Manager Setup test passed")
            return manager
        else:
            logger.error("❌ Fine-Tuning Manager Setup failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Fine-Tuning Manager Setup test failed: {e}")
        return None


def test_training_data_preparation():
    """Test Training Data Preparation"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Training Data Preparation...")
    
    try:
        # Create manager
        config = create_fine_tuning_config(
            output_dir="./test_models/data_prep",
            forward_return_horizon=3
        )
        manager = create_fine_tuning_manager(config)
        
        # Generate test bars
        test_bars = generate_test_bars(200)
        
        # Prepare training data
        dataset_path = manager.prepare_training_data(test_bars)
        
        if dataset_path and Path(dataset_path).exists():
            # Check dataset size
            file_size = Path(dataset_path).stat().st_size
            logger.info(f"Dataset created: {dataset_path} ({file_size} bytes)")
            
            logger.info("✅ Training Data Preparation test passed")
            return dataset_path
        else:
            logger.error("❌ Training Data Preparation failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Training Data Preparation test failed: {e}")
        return None


def test_mini_training_loop():
    """Test Mini Training Loop"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Mini Training Loop...")
    
    try:
        # Create config for mini training
        config = create_fine_tuning_config(
            output_dir="./test_models/mini_training",
            batch_size=4,
            num_epochs=2,
            learning_rate=1e-3,
            use_mixed_precision=False,  # Disable for stability in test
            dataloader_num_workers=2,
            logging_steps=5,
            save_steps=50
        )
        
        # Create manager
        manager = create_fine_tuning_manager(config)
        
        # Generate larger dataset for training
        test_bars = generate_test_bars(500)
        
        # Prepare data
        dataset_path = manager.prepare_training_data(test_bars)
        
        if not dataset_path:
            logger.error("❌ Failed to prepare training data")
            return False
        
        # Run mini training
        logger.info("Starting mini training loop...")
        start_time = time.time()
        
        results = manager.fine_tune_model(dataset_path)
        
        training_time = time.time() - start_time
        
        # Check results
        if "error" not in results:
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Results: {results}")
            
            # Cleanup
            manager.cleanup()
            
            logger.info("✅ Mini Training Loop test passed")
            return True
        else:
            logger.error(f"❌ Training failed: {results.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Mini Training Loop test failed: {e}")
        return False


def test_model_checkpointing():
    """Test Model Checkpointing"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Model Checkpointing...")
    
    try:
        # Create config with checkpointing
        config = create_fine_tuning_config(
            output_dir="./test_models/checkpointing",
            batch_size=2,
            num_epochs=1,
            save_steps=10,
            save_total_limit=2
        )
        
        manager = create_fine_tuning_manager(config)
        
        # Setup model
        if not manager.setup_model():
            logger.error("❌ Model setup failed")
            return False
        
        # Save a checkpoint manually
        manager._save_checkpoint(epoch=0, val_loss=0.5)
        
        # Check if checkpoint exists
        checkpoint_dir = Path(config.output_dir) / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
        
        if checkpoint_files:
            logger.info(f"Checkpoint saved: {checkpoint_files[0]}")
            
            # Test loading checkpoint
            manager._load_checkpoint(str(checkpoint_files[0]))
            
            logger.info("✅ Model Checkpointing test passed")
            return True
        else:
            logger.error("❌ No checkpoint files found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model Checkpointing test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger = setup_logging()
    logger.info("🚀 Starting Enhanced Fine-Tuning Pipeline Test Suite")
    
    # Test results
    test_results = {}
    
    # Test 1: Dataset Builder
    logger.info("\n" + "="*50)
    dataset_path = test_dataset_builder()
    test_results["dataset_builder"] = dataset_path is not None
    
    # Test 2: Enhanced Feature Extractor
    logger.info("\n" + "="*50)
    test_results["feature_extractor"] = test_enhanced_feature_extractor()
    
    # Test 3: Fine-Tuning Configuration
    logger.info("\n" + "="*50)
    config = test_fine_tuning_config()
    test_results["fine_tuning_config"] = config is not None
    
    # Test 4: Fine-Tuning Manager Setup
    logger.info("\n" + "="*50)
    manager = test_fine_tuning_manager_setup()
    test_results["manager_setup"] = manager is not None
    
    # Test 5: Training Data Preparation
    logger.info("\n" + "="*50)
    training_dataset = test_training_data_preparation()
    test_results["data_preparation"] = training_dataset is not None
    
    # Test 6: Model Checkpointing
    logger.info("\n" + "="*50)
    test_results["checkpointing"] = test_model_checkpointing()
    
    # Test 7: Mini Training Loop (only if previous tests pass)
    logger.info("\n" + "="*50)
    if all([test_results["dataset_builder"], test_results["feature_extractor"], 
            test_results["manager_setup"]]):
        test_results["mini_training"] = test_mini_training_loop()
    else:
        logger.warning("⚠️ Skipping mini training loop due to previous test failures")
        test_results["mini_training"] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Enhanced Fine-Tuning Pipeline is ready!")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Please check the logs.")
    
    # Save test results
    results_file = "test_logs/enhanced_fine_tuning_test_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'test_results': test_results,
            'summary': {
                'passed': passed_tests,
                'total': total_tests,
                'success_rate': passed_tests / total_tests,
                'timestamp': datetime.now().isoformat()
            },
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'python_version': sys.version
            }
        }, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)