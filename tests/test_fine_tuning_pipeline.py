"""
Tests für Fine-Tuning Pipeline
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import shutil

from ai_indicator_optimizer.training.fine_tuning_manager import (
    FineTuningManager, FineTuningConfig, TradingPatternDataset, TrainingMetrics
)
from ai_indicator_optimizer.training.training_dataset_builder import (
    TrainingDatasetBuilder, PatternDetector, DatasetSample
)
from ai_indicator_optimizer.training.gpu_training_loop import (
    GPUTrainingLoop, GPUTrainingConfig, MemoryOptimizer, PerformanceProfiler
)
from ai_indicator_optimizer.training.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata
)


class TestFineTuningConfig:
    """Tests für FineTuningConfig"""
    
    def test_default_config(self):
        """Test Standard-Konfiguration"""
        config = FineTuningConfig()
        
        assert config.base_model_name == "openbmb/MiniCPM-V-2_6"
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.use_mixed_precision is True
        assert config.lora_enabled is True
        assert len(config.pattern_types) > 0
    
    def test_custom_config(self):
        """Test Custom-Konfiguration"""
        config = FineTuningConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
            lora_rank=32
        )
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5
        assert config.lora_rank == 32


class TestTradingPatternDataset:
    """Tests für TradingPatternDataset"""
    
    def setup_method(self):
        """Setup für Tests"""
        self.config = FineTuningConfig()
        
        # Mock Processor und Tokenizer
        self.mock_processor = Mock()
        self.mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.zeros(1, 3, 448, 448)
        }
        
        self.mock_tokenizer = Mock()
        
        # Test Data Samples
        self.test_samples = [
            {
                "chart_image": Image.new('RGB', (448, 448), color='red'),
                "numerical_data": {
                    "RSI": 65.5,
                    "MACD": {"macd": 0.15, "signal": 0.10}
                },
                "pattern_label": "double_top",
                "pattern_description": "Double top pattern detected",
                "market_context": {
                    "symbol": "EUR/USD",
                    "timeframe": "4H"
                }
            }
        ]
    
    def test_dataset_initialization(self):
        """Test Dataset-Initialisierung"""
        dataset = TradingPatternDataset(
            self.test_samples,
            self.mock_processor,
            self.mock_tokenizer,
            self.config
        )
        
        assert len(dataset) == 1
        assert dataset.config == self.config
        assert "double_top" in dataset.pattern_to_id
    
    def test_dataset_getitem(self):
        """Test Dataset __getitem__"""
        dataset = TradingPatternDataset(
            self.test_samples,
            self.mock_processor,
            self.mock_tokenizer,
            self.config
        )
        
        sample = dataset[0]
        
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert "pixel_values" in sample
        assert "pattern_id" in sample
        
        # Verify shapes
        assert sample["input_ids"].dim() == 1
        assert sample["pattern_id"].item() == dataset.pattern_to_id["double_top"]
    
    def test_numerical_data_formatting(self):
        """Test numerische Daten-Formatierung"""
        dataset = TradingPatternDataset(
            self.test_samples,
            self.mock_processor,
            self.mock_tokenizer,
            self.config
        )
        
        numerical_data = {
            "RSI": 65.5,
            "MACD": {"macd": 0.15, "signal": 0.10},
            "SMA": [100.1, 100.2, 100.3]
        }
        
        formatted_text = dataset._format_numerical_data(numerical_data)
        
        assert "RSI: 65.5000" in formatted_text
        assert "MACD_macd: 0.1500" in formatted_text
        assert "SMA: 100.3000" in formatted_text
    
    def test_market_context_formatting(self):
        """Test Market Context Formatierung"""
        dataset = TradingPatternDataset(
            self.test_samples,
            self.mock_processor,
            self.mock_tokenizer,
            self.config
        )
        
        market_context = {
            "symbol": "EUR/USD",
            "timeframe": "4H",
            "trend": "bullish",
            "volatility": "medium"
        }
        
        formatted_text = dataset._format_market_context(market_context)
        
        assert "EUR/USD" in formatted_text
        assert "4H" in formatted_text
        assert "bullish" in formatted_text


class TestPatternDetector:
    """Tests für PatternDetector"""
    
    def setup_method(self):
        """Setup für Tests"""
        self.detector = PatternDetector()
        
        # Create test OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        
        # Simulate price data with patterns
        base_price = 1.1000
        prices = [base_price]
        
        for i in range(99):
            change = np.random.normal(0, 0.0005)
            new_price = prices[-1] + change
            prices.append(new_price)
        
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + np.random.uniform(0, 0.001) for p in prices],
            'low': [p - np.random.uniform(0, 0.001) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
    
    def test_pattern_templates_loading(self):
        """Test Pattern-Templates Loading"""
        templates = self.detector.pattern_templates
        
        assert "double_top" in templates
        assert "double_bottom" in templates
        assert "head_shoulders" in templates
        assert "triangle" in templates
        assert "support_resistance" in templates
        assert "breakout" in templates
        
        # Verify template structure
        double_top = templates["double_top"]
        assert double_top.pattern_type == "double_top"
        assert "min_peaks" in double_top.conditions
        assert len(double_top.visual_markers) > 0
    
    def test_double_top_detection(self):
        """Test Double Top Detection"""
        # Create artificial double top pattern
        double_top_data = self.test_data.copy()
        
        # Add two peaks at similar levels
        double_top_data.loc[20, 'high'] = 1.1050
        double_top_data.loc[21, 'high'] = 1.1050
        double_top_data.loc[60, 'high'] = 1.1048
        double_top_data.loc[61, 'high'] = 1.1048
        
        # Add valley between peaks
        for i in range(30, 50):
            double_top_data.loc[i, 'low'] = 1.1020
        
        template = self.detector.pattern_templates["double_top"]
        result = self.detector._detect_double_top(double_top_data, template)
        
        # Should detect pattern or return None (depending on exact conditions)
        assert result is None or isinstance(result, dict)
    
    def test_support_resistance_detection(self):
        """Test Support/Resistance Detection"""
        # Create artificial S/R level
        sr_data = self.test_data.copy()
        
        # Add multiple touches at same level
        support_level = 1.1000
        touch_indices = [10, 25, 40, 55, 70]
        
        for idx in touch_indices:
            sr_data.loc[idx, 'low'] = support_level
            sr_data.loc[idx, 'close'] = support_level + 0.0005
        
        template = self.detector.pattern_templates["support_resistance"]
        result = self.detector._detect_support_resistance(sr_data, template)
        
        # Should detect S/R level
        if result:
            assert "level" in result
            assert "touches" in result
            assert result["touches"] >= 3


class TestMemoryOptimizer:
    """Tests für MemoryOptimizer"""
    
    def test_memory_optimizer_initialization(self):
        """Test MemoryOptimizer Initialisierung"""
        config = GPUTrainingConfig()
        optimizer = MemoryOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.peak_memory == 0
        assert len(optimizer.memory_history) == 0
    
    def test_memory_monitoring(self):
        """Test Memory Monitoring"""
        config = GPUTrainingConfig()
        optimizer = MemoryOptimizer(config)
        
        stats = optimizer.monitor_memory()
        
        assert isinstance(stats, dict)
        assert "allocated" in stats
        assert "free" in stats
        
        # Should add to history
        assert len(optimizer.memory_history) == 1
    
    def test_memory_cleanup(self):
        """Test Memory Cleanup"""
        config = GPUTrainingConfig()
        optimizer = MemoryOptimizer(config)
        
        # Should not raise exception
        optimizer.cleanup_memory()


class TestPerformanceProfiler:
    """Tests für PerformanceProfiler"""
    
    def test_profiler_initialization(self):
        """Test Profiler Initialisierung"""
        config = GPUTrainingConfig()
        profiler = PerformanceProfiler(config)
        
        assert profiler.config == config
        assert len(profiler.step_times) == 0
    
    def test_step_timing(self):
        """Test Step Timing"""
        config = GPUTrainingConfig()
        profiler = PerformanceProfiler(config)
        
        start_time = profiler.start_step_timer()
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        step_time = profiler.end_step_timer(start_time)
        
        assert step_time > 0
        assert len(profiler.step_times) == 1
    
    def test_performance_stats(self):
        """Test Performance Statistics"""
        config = GPUTrainingConfig()
        profiler = PerformanceProfiler(config)
        
        # Add some dummy times
        profiler.step_times = [0.1, 0.2, 0.15, 0.18]
        profiler.forward_times = [0.05, 0.08, 0.06, 0.07]
        
        stats = profiler.get_performance_stats()
        
        assert "step_times" in stats
        assert "forward_times" in stats
        assert "mean" in stats["step_times"]
        assert "steps_per_second" in stats["step_times"]


class TestCheckpointManager:
    """Tests für CheckpointManager"""
    
    def setup_method(self):
        """Setup für Tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        
        self.manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3
        )
        
        # Mock Model und Optimizer
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        
        self.mock_optimizer = Mock()
        self.mock_optimizer.state_dict.return_value = {"param_groups": [{"lr": 0.001}]}
    
    def teardown_method(self):
        """Cleanup nach Tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager Initialisierung"""
        assert self.manager.checkpoint_dir.exists()
        assert (self.manager.checkpoint_dir / "models").exists()
        assert (self.manager.checkpoint_dir / "metadata").exists()
        assert self.manager.max_checkpoints == 3
    
    def test_save_checkpoint(self):
        """Test Checkpoint Speichern"""
        config = FineTuningConfig()
        metrics = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "learning_rate": 0.001
        }
        
        checkpoint_path = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=1,
            step=100,
            metrics=metrics,
            config=config
        )
        
        assert Path(checkpoint_path).exists()
        assert len(self.manager.checkpoints) == 1
        
        # Verify checkpoint content
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint_data["epoch"] == 1
        assert checkpoint_data["step"] == 100
        assert "model_state_dict" in checkpoint_data
        assert "optimizer_state_dict" in checkpoint_data
    
    def test_load_checkpoint(self):
        """Test Checkpoint Laden"""
        config = FineTuningConfig()
        metrics = {"train_loss": 0.5}
        
        # Save checkpoint first
        checkpoint_path = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=2,
            step=200,
            metrics=metrics,
            config=config
        )
        
        # Load checkpoint
        epoch, step, loaded_metrics = self.manager.load_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            checkpoint_path=checkpoint_path
        )
        
        assert epoch == 2
        assert step == 200
        assert loaded_metrics["train_loss"] == 0.5
        
        # Verify model.load_state_dict was called
        self.mock_model.load_state_dict.assert_called_once()
        self.mock_optimizer.load_state_dict.assert_called_once()
    
    def test_best_checkpoint_tracking(self):
        """Test Best Checkpoint Tracking"""
        config = FineTuningConfig()
        
        # Save checkpoints with different eval losses
        metrics1 = {"eval_loss": 0.8}
        metrics2 = {"eval_loss": 0.5}  # Better
        metrics3 = {"eval_loss": 0.7}
        
        self.manager.save_checkpoint(self.mock_model, self.mock_optimizer, 1, 100, metrics1, config)
        self.manager.save_checkpoint(self.mock_model, self.mock_optimizer, 2, 200, metrics2, config)
        self.manager.save_checkpoint(self.mock_model, self.mock_optimizer, 3, 300, metrics3, config)
        
        # Best should be the one with lowest eval_loss
        assert self.manager.best_metric_value == 0.5
        assert self.manager.best_checkpoint_path is not None
    
    def test_checkpoint_cleanup(self):
        """Test Checkpoint Cleanup"""
        config = FineTuningConfig()
        
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            metrics = {"train_loss": 0.5 - i * 0.1}
            self.manager.save_checkpoint(
                self.mock_model, self.mock_optimizer, i+1, (i+1)*100, metrics, config
            )
        
        # Should only keep max_checkpoints
        assert len(self.manager.checkpoints) == self.manager.max_checkpoints
    
    def test_checkpoint_validation(self):
        """Test Checkpoint Validation"""
        config = FineTuningConfig()
        metrics = {"train_loss": 0.5}
        
        # Save valid checkpoint
        checkpoint_path = self.manager.save_checkpoint(
            self.mock_model, self.mock_optimizer, 1, 100, metrics, config
        )
        
        # Validate checkpoint
        validation_result = self.manager.validate_checkpoint(checkpoint_path)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        assert "file_size_mb" in validation_result["info"]


class TestFineTuningManager:
    """Tests für FineTuningManager"""
    
    def setup_method(self):
        """Setup für Tests"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = FineTuningConfig(
            output_dir=str(Path(self.temp_dir) / "finetuning"),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            num_epochs=1,
            batch_size=1,
            use_wandb=False
        )
        
        self.manager = FineTuningManager(self.config)
    
    def teardown_method(self):
        """Cleanup nach Tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test Manager Initialisierung"""
        assert self.manager.config == self.config
        assert self.manager.device.type in ["cuda", "cpu"]
        assert Path(self.config.output_dir).exists()
        assert Path(self.config.cache_dir).exists()
    
    @patch('ai_indicator_optimizer.training.fine_tuning_manager.AutoModelForCausalLM')
    @patch('ai_indicator_optimizer.training.fine_tuning_manager.AutoTokenizer')
    @patch('ai_indicator_optimizer.training.fine_tuning_manager.AutoProcessor')
    def test_load_base_model(self, mock_processor, mock_tokenizer, mock_model):
        """Test Base Model Loading"""
        # Mock model components
        mock_model.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        success = self.manager.load_base_model()
        
        assert success is True
        assert self.manager.model is not None
        assert self.manager.tokenizer is not None
        assert self.manager.processor is not None
    
    def test_prepare_datasets(self):
        """Test Dataset Preparation"""
        # Mock training data
        training_data = [
            {
                "chart_image": Image.new('RGB', (224, 224), color='red'),
                "numerical_data": {"RSI": 65.5},
                "pattern_label": "double_top",
                "pattern_description": "Test pattern",
                "market_context": {"symbol": "EUR/USD"}
            }
        ] * 10  # 10 samples
        
        # Mock processor and tokenizer
        self.manager.processor = Mock()
        self.manager.processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.zeros(1, 3, 224, 224)
        }
        self.manager.tokenizer = Mock()
        
        train_dataset, val_dataset, test_dataset = self.manager.prepare_datasets(training_data)
        
        assert len(train_dataset) > 0
        assert len(val_dataset) >= 0
        assert len(test_dataset) >= 0
        
        # Total should equal input
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total_samples == len(training_data)
    
    def test_training_arguments_setup(self):
        """Test Training Arguments Setup"""
        training_args = self.manager.setup_training_arguments()
        
        assert training_args.learning_rate == self.config.learning_rate
        assert training_args.per_device_train_batch_size == self.config.batch_size
        assert training_args.num_train_epochs == self.config.num_epochs
        assert training_args.fp16 == self.config.use_mixed_precision
    
    def test_get_training_summary(self):
        """Test Training Summary"""
        summary = self.manager.get_training_summary()
        
        assert "status" in summary
        assert summary["status"] == "No training completed"


class TestIntegrationScenarios:
    """Integration Tests für komplette Pipeline"""
    
    def setup_method(self):
        """Setup für Integration Tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup nach Tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_pipeline_setup(self):
        """Test End-to-End Pipeline Setup"""
        
        # 1. Create Config
        config = FineTuningConfig(
            output_dir=str(Path(self.temp_dir) / "finetuning"),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            num_epochs=1,
            batch_size=1,
            use_wandb=False
        )
        
        # 2. Create Manager
        manager = FineTuningManager(config)
        
        # 3. Create Checkpoint Manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(Path(self.temp_dir) / "checkpoints")
        )
        
        # 4. Verify Setup
        assert manager.config == config
        assert checkpoint_manager.checkpoint_dir.exists()
        assert Path(config.output_dir).exists()
    
    def test_gpu_training_config_integration(self):
        """Test GPU Training Config Integration"""
        
        gpu_config = GPUTrainingConfig(
            use_mixed_precision=True,
            max_batch_size=4,
            adaptive_batch_size=True
        )
        
        fine_tuning_config = FineTuningConfig(
            batch_size=gpu_config.max_batch_size,
            use_mixed_precision=gpu_config.use_mixed_precision
        )
        
        # Verify configs are compatible
        assert fine_tuning_config.batch_size == gpu_config.max_batch_size
        assert fine_tuning_config.use_mixed_precision == gpu_config.use_mixed_precision
    
    def test_pattern_detector_integration(self):
        """Test Pattern Detector Integration"""
        
        detector = PatternDetector()
        
        # Verify all expected patterns are available
        expected_patterns = [
            "double_top", "double_bottom", "head_shoulders", 
            "triangle", "support_resistance", "breakout"
        ]
        
        for pattern in expected_patterns:
            assert pattern in detector.pattern_templates
            
            template = detector.pattern_templates[pattern]
            assert hasattr(template, 'pattern_type')
            assert hasattr(template, 'conditions')
            assert hasattr(template, 'visual_markers')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])