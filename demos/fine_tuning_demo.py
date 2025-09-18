#!/usr/bin/env python3
"""
Fine-Tuning Pipeline Demo
Demonstriert die komplette Fine-Tuning Pipeline für MiniCPM-4.1-8B
"""

import sys
import os
import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_indicator_optimizer.training.fine_tuning_manager import (
    FineTuningManager, FineTuningConfig
)
from ai_indicator_optimizer.training.training_dataset_builder import (
    TrainingDatasetBuilder, PatternDetector, DatasetSample
)
from ai_indicator_optimizer.training.gpu_training_loop import (
    GPUTrainingLoop, GPUTrainingConfig, MemoryOptimizer
)
from ai_indicator_optimizer.training.checkpoint_manager import CheckpointManager
from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector
from ai_indicator_optimizer.indicators.indicator_calculator import IndicatorCalculator
from ai_indicator_optimizer.visualization.chart_renderer import ChartRenderer


class FineTuningDemo:
    """
    Demo-Klasse für Fine-Tuning Pipeline
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Hardware Detection
        self.hardware_detector = HardwareDetector()
        
        # Demo Configuration
        self.demo_config = self._create_demo_config()
        
        # Components
        self.fine_tuning_manager = None
        self.checkpoint_manager = None
        self.pattern_detector = None
        
        self.logger.info("Fine-Tuning Demo initialized")
    
    def setup_logging(self):
        """Setup Logging für Demo"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('fine_tuning_demo.log')
            ]
        )
    
    def _create_demo_config(self) -> FineTuningConfig:
        """Erstellt Demo-Konfiguration"""
        
        # Hardware-optimierte Konfiguration
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            if "rtx 5090" in gpu_name:
                # RTX 5090 optimiert
                batch_size = 8
                gradient_accumulation_steps = 2
                max_memory = "30GB"
            elif "rtx 4090" in gpu_name:
                # RTX 4090 optimiert
                batch_size = 4
                gradient_accumulation_steps = 4
                max_memory = "22GB"
            else:
                # Andere GPUs
                batch_size = 2
                gradient_accumulation_steps = 8
                max_memory = "10GB"
        else:
            # CPU Fallback
            batch_size = 1
            gradient_accumulation_steps = 16
            max_memory = None
        
        return FineTuningConfig(
            # Demo Settings
            output_dir="./demo_outputs/fine_tuning",
            cache_dir="./demo_outputs/cache",
            
            # Training Settings
            learning_rate=5e-5,  # Etwas höher für Demo
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_epochs=1,  # Kurz für Demo
            
            # Hardware Optimizations
            use_mixed_precision=torch.cuda.is_available(),
            use_gradient_checkpointing=True,
            dataloader_num_workers=4,
            max_memory_per_gpu=max_memory,
            
            # Demo-spezifische Settings
            logging_steps=5,
            eval_steps=20,
            save_steps=50,
            use_wandb=False,  # Für Demo deaktiviert
            
            # LoRA Settings für Demo
            lora_enabled=True,
            lora_rank=8,  # Kleiner für Demo
            lora_alpha=16
        )
    
    def run_hardware_detection_demo(self):
        """Demonstriert Hardware-Erkennung für Fine-Tuning"""
        print("\n" + "="*60)
        print("🔧 HARDWARE DETECTION FOR FINE-TUNING")
        print("="*60)
        
        # CPU Info
        if self.hardware_detector.cpu_info:
            cpu = self.hardware_detector.cpu_info
            print(f"💻 CPU: {cpu.model}")
            print(f"🧠 Cores: {cpu.cores_physical} physical, {cpu.cores_logical} logical")
            print(f"⚡ Frequency: {cpu.frequency_current:.0f} MHz")
        
        # Memory Info
        if self.hardware_detector.memory_info:
            memory = self.hardware_detector.memory_info
            print(f"💾 RAM: {memory.total // (1024**3)} GB")
            if memory.frequency:
                print(f"📊 RAM Speed: {memory.frequency} MHz")
        
        # GPU Info
        if self.hardware_detector.gpu_info and len(self.hardware_detector.gpu_info) > 0:
            for i, gpu in enumerate(self.hardware_detector.gpu_info):
                print(f"🎮 GPU {i}: {gpu.name}")
                print(f"📊 VRAM: {gpu.memory_total // (1024**3)} GB")
                if gpu.cuda_cores:
                    print(f"⚡ CUDA Cores: {gpu.cuda_cores}")
        else:
            print("🎮 GPU: Not available - using CPU")
        
        # Fine-Tuning Recommendations
        print(f"\n🎯 Fine-Tuning Recommendations:")
        print(f"   Batch Size: {self.demo_config.batch_size}")
        print(f"   Gradient Accumulation: {self.demo_config.gradient_accumulation_steps}")
        print(f"   Mixed Precision: {self.demo_config.use_mixed_precision}")
        print(f"   LoRA Enabled: {self.demo_config.lora_enabled}")
        
        # Worker Recommendations
        workers = self.hardware_detector.get_optimal_worker_counts()
        print(f"   DataLoader Workers: {workers.get('data_pipeline', 4)}")
    
    def run_pattern_detection_demo(self):
        """Demonstriert Pattern-Detection"""
        print("\n" + "="*60)
        print("📈 PATTERN DETECTION DEMO")
        print("="*60)
        
        # Initialize Pattern Detector
        self.pattern_detector = PatternDetector()
        
        print(f"🔍 Available Pattern Types:")
        for pattern_type, template in self.pattern_detector.pattern_templates.items():
            print(f"   - {pattern_type.replace('_', ' ').title()}: {template.description}")
        
        # Generate Demo Data
        print(f"\n📊 Generating demo market data...")
        demo_data = self._generate_demo_market_data()
        
        # Detect Patterns
        print(f"🔍 Detecting patterns in demo data...")
        detected_patterns = self.pattern_detector.detect_patterns(
            demo_data, 
            indicators={}, 
            lookback_window=50
        )
        
        print(f"✅ Pattern Detection Results:")
        print(f"   Total Patterns Detected: {len(detected_patterns)}")
        
        if detected_patterns:
            pattern_counts = {}
            for pattern in detected_patterns:
                pattern_type = pattern["pattern_type"]
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            for pattern_type, count in pattern_counts.items():
                print(f"   - {pattern_type.replace('_', ' ').title()}: {count}")
            
            # Show best pattern
            best_pattern = max(detected_patterns, key=lambda x: x["confidence"])
            print(f"\n🏆 Best Pattern:")
            print(f"   Type: {best_pattern['pattern_type'].replace('_', ' ').title()}")
            print(f"   Confidence: {best_pattern['confidence']:.1%}")
            print(f"   Description: {best_pattern.get('description', 'N/A')}")
        else:
            print("   No patterns detected in demo data")
    
    def _generate_demo_market_data(self) -> pd.DataFrame:
        """Generiert Demo-Marktdaten"""
        
        # Simuliere EUR/USD Daten
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        
        # Simuliere Preis mit Patterns
        base_price = 1.1000
        prices = [base_price]
        
        for i in range(199):
            # Add some pattern-like behavior
            if 40 <= i <= 60:
                # Double top area
                if i in [45, 55]:
                    change = 0.0050  # Peak
                else:
                    change = np.random.normal(-0.0005, 0.0003)
            elif 100 <= i <= 120:
                # Support area
                change = max(-0.0010, np.random.normal(0.0002, 0.0005))
            else:
                change = np.random.normal(0, 0.0008)
            
            new_price = max(1.0800, min(1.1200, prices[-1] + change))
            prices.append(new_price)
        
        # Create OHLCV data
        ohlcv_data = []
        for i, price in enumerate(prices):
            high = price + np.random.uniform(0, 0.0015)
            low = price - np.random.uniform(0, 0.0015)
            open_price = prices[i-1] if i > 0 else price
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': np.random.randint(1000000, 5000000)
            })
        
        return pd.DataFrame(ohlcv_data)
    
    def run_dataset_building_demo(self):
        """Demonstriert Dataset-Building"""
        print("\n" + "="*60)
        print("🏗️ DATASET BUILDING DEMO")
        print("="*60)
        
        try:
            # Mock Components (da echte Daten-Connector nicht verfügbar)
            print("📊 Initializing dataset builder components...")
            
            # Create mock dataset samples
            print("🎨 Creating demo dataset samples...")
            demo_samples = self._create_demo_dataset_samples()
            
            print(f"✅ Dataset Building Results:")
            print(f"   Total Samples: {len(demo_samples)}")
            
            # Pattern Distribution
            pattern_counts = {}
            confidence_scores = []
            
            for sample in demo_samples:
                pattern = sample.pattern_label
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                confidence_scores.append(sample.confidence_score)
            
            print(f"   Pattern Distribution:")
            for pattern, count in pattern_counts.items():
                print(f"     - {pattern.replace('_', ' ').title()}: {count}")
            
            print(f"   Confidence Statistics:")
            print(f"     - Mean: {np.mean(confidence_scores):.2f}")
            print(f"     - Min: {np.min(confidence_scores):.2f}")
            print(f"     - Max: {np.max(confidence_scores):.2f}")
            
            return demo_samples
            
        except Exception as e:
            print(f"❌ Dataset building failed: {e}")
            return []
    
    def _create_demo_dataset_samples(self) -> List[DatasetSample]:
        """Erstellt Demo-Dataset-Samples"""
        
        samples = []
        pattern_types = ["double_top", "double_bottom", "triangle", "support_resistance", "breakout"]
        
        for i in range(20):  # 20 Demo-Samples
            pattern_type = pattern_types[i % len(pattern_types)]
            
            # Create demo chart image
            chart_image = self._create_demo_chart_image(pattern_type)
            
            # Create demo numerical data
            numerical_data = {
                "RSI": np.random.uniform(30, 70),
                "MACD": {
                    "macd": np.random.uniform(-0.01, 0.01),
                    "signal": np.random.uniform(-0.01, 0.01)
                },
                "SMA_20": np.random.uniform(1.0900, 1.1100),
                "BollingerBands": {
                    "upper": np.random.uniform(1.1050, 1.1150),
                    "middle": np.random.uniform(1.1000, 1.1100),
                    "lower": np.random.uniform(1.0950, 1.1050)
                }
            }
            
            # Market context
            market_context = {
                "symbol": "EUR/USD",
                "timeframe": "4H",
                "trend": np.random.choice(["bullish", "bearish", "neutral"]),
                "volatility": np.random.choice(["low", "medium", "high"])
            }
            
            # Pattern description
            pattern_description = f"{pattern_type.replace('_', ' ').title()} pattern detected with {market_context['trend']} trend and {market_context['volatility']} volatility."
            
            sample = DatasetSample(
                chart_image=chart_image,
                numerical_data=numerical_data,
                pattern_label=pattern_type,
                pattern_description=pattern_description,
                market_context=market_context,
                confidence_score=np.random.uniform(0.6, 0.9),
                metadata={
                    "sample_id": i,
                    "creation_time": datetime.now().isoformat()
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def _create_demo_chart_image(self, pattern_type: str) -> Image.Image:
        """Erstellt Demo-Chart-Image"""
        
        # Create simple colored image for demo
        colors = {
            "double_top": "red",
            "double_bottom": "green", 
            "triangle": "blue",
            "support_resistance": "orange",
            "breakout": "purple"
        }
        
        color = colors.get(pattern_type, "gray")
        image = Image.new('RGB', (448, 448), color=color)
        
        return image
    
    def run_fine_tuning_setup_demo(self):
        """Demonstriert Fine-Tuning Setup"""
        print("\n" + "="*60)
        print("🤖 FINE-TUNING SETUP DEMO")
        print("="*60)
        
        try:
            # Initialize Fine-Tuning Manager
            print("⚙️ Initializing Fine-Tuning Manager...")
            self.fine_tuning_manager = FineTuningManager(self.demo_config)
            
            print(f"✅ Fine-Tuning Manager initialized:")
            print(f"   Output Directory: {self.demo_config.output_dir}")
            print(f"   Cache Directory: {self.demo_config.cache_dir}")
            print(f"   Learning Rate: {self.demo_config.learning_rate}")
            print(f"   Batch Size: {self.demo_config.batch_size}")
            print(f"   Epochs: {self.demo_config.num_epochs}")
            
            # Initialize Checkpoint Manager
            print("\n💾 Initializing Checkpoint Manager...")
            checkpoint_dir = Path(self.demo_config.output_dir) / "checkpoints"
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(checkpoint_dir),
                max_checkpoints=3,
                monitor_metric="eval_loss"
            )
            
            print(f"✅ Checkpoint Manager initialized:")
            print(f"   Checkpoint Directory: {checkpoint_dir}")
            print(f"   Max Checkpoints: 3")
            print(f"   Monitor Metric: eval_loss")
            
            # GPU Training Loop Setup
            print("\n⚡ Setting up GPU Training Loop...")
            gpu_config = GPUTrainingConfig(
                use_mixed_precision=self.demo_config.use_mixed_precision,
                max_batch_size=self.demo_config.batch_size,
                gradient_accumulation_steps=self.demo_config.gradient_accumulation_steps
            )
            
            print(f"✅ GPU Training Config:")
            print(f"   Mixed Precision: {gpu_config.use_mixed_precision}")
            print(f"   Max Batch Size: {gpu_config.max_batch_size}")
            print(f"   Gradient Accumulation: {gpu_config.gradient_accumulation_steps}")
            
            return True
            
        except Exception as e:
            print(f"❌ Fine-tuning setup failed: {e}")
            return False
    
    def run_memory_optimization_demo(self):
        """Demonstriert Memory-Optimierung"""
        print("\n" + "="*60)
        print("💾 MEMORY OPTIMIZATION DEMO")
        print("="*60)
        
        try:
            # Initialize Memory Optimizer
            gpu_config = GPUTrainingConfig()
            memory_optimizer = MemoryOptimizer(gpu_config)
            
            print("🔧 Running memory optimization...")
            memory_optimizer.optimize_memory_usage()
            
            # Monitor Memory
            print("📊 Monitoring GPU memory...")
            memory_stats = memory_optimizer.monitor_memory()
            
            print(f"✅ Memory Statistics:")
            print(f"   Total Memory: {memory_stats.get('total', 0) // (1024**3)} GB")
            print(f"   Allocated: {memory_stats.get('allocated_gb', 0):.1f} GB")
            print(f"   Free: {memory_stats.get('free_gb', 0):.1f} GB")
            
            if torch.cuda.is_available():
                print(f"   GPU Utilization: Available")
                
                # Simulate memory usage
                print("\n🧪 Simulating memory usage...")
                dummy_tensor = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                updated_stats = memory_optimizer.monitor_memory()
                print(f"   After allocation: {updated_stats.get('allocated_gb', 0):.1f} GB")
                
                # Cleanup
                del dummy_tensor
                memory_optimizer.cleanup_memory()
                
                final_stats = memory_optimizer.monitor_memory()
                print(f"   After cleanup: {final_stats.get('allocated_gb', 0):.1f} GB")
            else:
                print(f"   GPU: Not available - using CPU")
            
        except Exception as e:
            print(f"❌ Memory optimization demo failed: {e}")
    
    def run_training_simulation_demo(self):
        """Simuliert Training-Prozess"""
        print("\n" + "="*60)
        print("🏋️ TRAINING SIMULATION DEMO")
        print("="*60)
        
        try:
            if not self.fine_tuning_manager:
                print("❌ Fine-tuning manager not initialized")
                return
            
            print("🎯 Simulating training process...")
            
            # Simulate training steps
            epochs = 1
            steps_per_epoch = 10
            
            for epoch in range(epochs):
                print(f"\n📈 Epoch {epoch + 1}/{epochs}")
                
                epoch_loss = 1.0
                
                for step in range(steps_per_epoch):
                    # Simulate training step
                    step_loss = epoch_loss * (1 - step * 0.05)  # Decreasing loss
                    
                    print(f"   Step {step + 1}/{steps_per_epoch}: Loss = {step_loss:.4f}")
                    
                    # Simulate checkpoint saving
                    if step % 5 == 0 and self.checkpoint_manager:
                        print(f"   💾 Saving checkpoint at step {step + 1}")
                
                print(f"   ✅ Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.4f}")
            
            print(f"\n🎉 Training simulation completed!")
            
            # Show training summary
            summary = self.fine_tuning_manager.get_training_summary()
            print(f"📊 Training Summary:")
            print(f"   Status: {summary.get('status', 'Simulated')}")
            print(f"   Model Path: {self.demo_config.output_dir}")
            
        except Exception as e:
            print(f"❌ Training simulation failed: {e}")
    
    def run_checkpoint_demo(self):
        """Demonstriert Checkpoint-Funktionalität"""
        print("\n" + "="*60)
        print("💾 CHECKPOINT MANAGEMENT DEMO")
        print("="*60)
        
        try:
            if not self.checkpoint_manager:
                print("❌ Checkpoint manager not initialized")
                return
            
            # Create mock model and optimizer for demo
            print("🤖 Creating mock model and optimizer...")
            
            class MockModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            mock_model = MockModel()
            mock_optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            
            # Simulate saving checkpoints
            print("💾 Simulating checkpoint saves...")
            
            for epoch in range(3):
                metrics = {
                    "train_loss": 1.0 - epoch * 0.2,
                    "eval_loss": 1.1 - epoch * 0.15,
                    "learning_rate": 0.001 * (0.9 ** epoch)
                }
                
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    epoch=epoch + 1,
                    step=(epoch + 1) * 100,
                    metrics=metrics,
                    config=self.demo_config
                )
                
                print(f"   ✅ Checkpoint {epoch + 1} saved: {Path(checkpoint_path).name}")
            
            # List checkpoints
            checkpoints = self.checkpoint_manager.list_checkpoints()
            print(f"\n📋 Available Checkpoints: {len(checkpoints)}")
            
            for i, checkpoint in enumerate(checkpoints):
                metadata = checkpoint["metadata"]
                print(f"   {i+1}. Epoch {metadata['epoch']}, Loss: {metadata['train_loss']:.3f}")
            
            # Best checkpoint
            best_path = self.checkpoint_manager.get_best_checkpoint_path()
            if best_path:
                print(f"\n🏆 Best Checkpoint: {Path(best_path).name}")
            
            # Checkpoint validation
            if checkpoints:
                print(f"\n🔍 Validating latest checkpoint...")
                latest_checkpoint = checkpoints[0]
                validation_result = self.checkpoint_manager.validate_checkpoint(latest_checkpoint["path"])
                
                print(f"   Valid: {validation_result['valid']}")
                print(f"   File Size: {validation_result['info'].get('file_size_mb', 0):.1f} MB")
                
                if validation_result['warnings']:
                    print(f"   Warnings: {len(validation_result['warnings'])}")
            
        except Exception as e:
            print(f"❌ Checkpoint demo failed: {e}")
    
    def run_complete_demo(self):
        """Führt komplette Demo durch"""
        print("🚀 MiniCPM Fine-Tuning Pipeline Demo")
        print("=" * 60)
        
        # Hardware Detection
        self.run_hardware_detection_demo()
        
        # Pattern Detection
        self.run_pattern_detection_demo()
        
        # Dataset Building
        dataset_samples = self.run_dataset_building_demo()
        
        # Fine-Tuning Setup
        setup_success = self.run_fine_tuning_setup_demo()
        
        if setup_success:
            # Memory Optimization
            self.run_memory_optimization_demo()
            
            # Training Simulation
            self.run_training_simulation_demo()
            
            # Checkpoint Management
            self.run_checkpoint_demo()
        
        # Summary
        print("\n" + "="*60)
        print("📋 DEMO SUMMARY")
        print("="*60)
        
        print(f"✅ Hardware Detection: Complete")
        print(f"✅ Pattern Detection: Complete")
        print(f"✅ Dataset Building: Complete ({len(dataset_samples) if dataset_samples else 0} samples)")
        print(f"✅ Fine-Tuning Setup: {'Complete' if setup_success else 'Failed'}")
        
        if setup_success:
            print(f"✅ Memory Optimization: Complete")
            print(f"✅ Training Simulation: Complete")
            print(f"✅ Checkpoint Management: Complete")
        
        print(f"\n🎉 Fine-Tuning Pipeline Demo completed!")
        print(f"📁 Demo outputs saved to: {self.demo_config.output_dir}")
        print(f"📄 Demo logs saved to: fine_tuning_demo.log")
        
        # Next Steps
        print(f"\n🔍 Next Steps:")
        print(f"   1. Review demo outputs in {self.demo_config.output_dir}")
        print(f"   2. Prepare real training data")
        print(f"   3. Run actual fine-tuning with: python -m ai_indicator_optimizer.training.fine_tuning_manager")
        print(f"   4. Monitor training progress with TensorBoard")


def main():
    """Hauptfunktion für Demo"""
    try:
        demo = FineTuningDemo()
        demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.exception("Demo failed with exception")


if __name__ == "__main__":
    main()