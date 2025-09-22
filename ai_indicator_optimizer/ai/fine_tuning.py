#!/usr/bin/env python3
"""
Enhanced Fine-Tuning Pipeline mit Dataset Builder
Task 6 Implementation - AI-Indicator-Optimizer

Features:
- BarDatasetBuilder für automatische Forward-Return-Label-Generierung
- Enhanced Feature Extraction mit technischen Indikatoren
- Polars-basierte Parquet-Export-Funktionalität
- GPU-optimierte Training-Loop mit Mixed-Precision
- Model-Checkpointing und Resume-Funktionalität
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import os
from datetime import datetime
from collections import deque
import gc
from tqdm import tqdm

from .multimodal_ai import MultimodalAI
from .enhanced_feature_extractor import EnhancedFeatureExtractor
from ..dataset.bar_dataset_builder import BarDatasetBuilder
from ..logging.feature_prediction_logger import FeaturePredictionLogger
from ..multimodal.dynamic_fusion_agent import DynamicFusionAgent, MultimodalInput, FusionMode, ProcessingBackend
# Use flexible Bar type (can be Nautilus Bar or Mock Bar)
from typing import Protocol

class BarProtocol(Protocol):
    """Protocol for Bar objects (Nautilus or Mock)"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_init: int
    
    @property
    def bar_type(self) -> Any:
        ...

# Type alias for flexibility
Bar = BarProtocol


@dataclass
class FineTuningConfig:
    """Enhanced Fine-Tuning Konfiguration für Task 6"""
    
    # Model Settings
    model_name: str = "minicpm-4.1-8b"
    output_dir: str = "./models/finetuned"
    cache_dir: str = "./models/cache"
    
    # Training Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8  # RTX 5090 optimiert
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # GPU Optimizations (RTX 5090)
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    dataloader_num_workers: int = 8
    
    # Dataset Builder Settings
    forward_return_horizon: int = 5
    return_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "buy_threshold": 0.0003,
        "sell_threshold": -0.0003
    })
    include_technical_indicators: bool = True
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    log_predictions: bool = True


class TradingDataset(Dataset):
    """Enhanced Trading Dataset für Fine-Tuning"""
    
    def __init__(self, 
                 dataset_path: str,
                 feature_extractor: EnhancedFeatureExtractor,
                 config: FineTuningConfig):
        
        self.config = config
        self.feature_extractor = feature_extractor
        self.logger = logging.getLogger(__name__)
        
        # Load Dataset
        self.data = self._load_dataset(dataset_path)
        self.logger.info(f"Loaded {len(self.data)} samples from {dataset_path}")
    
    def _load_dataset(self, dataset_path: str) -> pl.DataFrame:
        """Load Parquet Dataset mit Polars"""
        try:
            if dataset_path.endswith('.parquet'):
                return pl.read_parquet(dataset_path)
            else:
                # Load all parquet files in directory
                parquet_files = list(Path(dataset_path).glob("*.parquet"))
                if parquet_files:
                    return pl.concat([pl.read_parquet(f) for f in parquet_files])
                else:
                    raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return pl.DataFrame()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample"""
        try:
            row = self.data.row(idx, named=True)
            
            # Extract features
            features = {k: v for k, v in row.items() 
                       if k not in ['label_fwd_ret_h5', 'label_class_h5', 'label_name_h5']}
            
            # Get labels with safe None handling
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=2):
                try:
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            forward_return = safe_float(row.get(f'label_fwd_ret_h{self.config.forward_return_horizon}'), 0.0)
            label_class = safe_int(row.get(f'label_class_h{self.config.forward_return_horizon}'), 2)  # Default HOLD
            
            # Convert to tensors with safe None handling
            
            feature_tensor = torch.tensor([
                safe_float(features.get('open'), 0.0),
                safe_float(features.get('high'), 0.0),
                safe_float(features.get('low'), 0.0),
                safe_float(features.get('close'), 0.0),
                safe_float(features.get('volume'), 0.0),
                safe_float(features.get('rsi_14'), 50.0),
                safe_float(features.get('sma_20', features.get('close', 0.0)), 0.0),
                safe_float(features.get('volatility_10'), 0.001),
                safe_float(features.get('hour'), 12.0),
                safe_float(features.get('is_london_session'), 0.0)
            ], dtype=torch.float32)
            
            return {
                'features': feature_tensor,
                'forward_return': torch.tensor(safe_float(forward_return), dtype=torch.float32),
                'label_class': torch.tensor(safe_int(label_class), dtype=torch.long),
                'timestamp': torch.tensor(safe_int(row.get('ts_ns', 0)), dtype=torch.long)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sample {idx}: {e}")
            # Return dummy sample
            return {
                'features': torch.zeros(10, dtype=torch.float32),
                'forward_return': torch.tensor(0.0, dtype=torch.float32),
                'label_class': torch.tensor(2, dtype=torch.long),
                'timestamp': torch.tensor(0, dtype=torch.long)
            }


class TradingModel(nn.Module):
    """Enhanced Trading Model für Forward-Return Prediction"""
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.1):
        
        super().__init__()
        
        # Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression Head (Forward Returns)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Classification Head (BUY/SELL/HOLD)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        encoded = self.feature_encoder(features)
        
        # Regression output
        forward_return = self.regression_head(encoded).squeeze(-1)
        
        # Classification output
        class_logits = self.classification_head(encoded)
        
        return forward_return, class_logits


class EnhancedFineTuningManager:
    """
    Enhanced Fine-Tuning Manager für Task 6
    
    Features:
    - BarDatasetBuilder Integration
    - GPU-optimierte Training-Loop
    - Mixed-Precision Training
    - Model Checkpointing
    - Enhanced Logging
    """
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Hardware Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Dataset Builder
        self.dataset_builder = BarDatasetBuilder(
            horizon=config.forward_return_horizon,
            return_thresholds=config.return_thresholds,
            include_technical_indicators=config.include_technical_indicators
        )
        
        # Feature Extractor
        self.feature_extractor = EnhancedFeatureExtractor({
            "include_time_features": True,
            "include_technical_indicators": True,
            "include_pattern_features": True,
            "include_volatility_features": True
        })
        
        # Dynamic Fusion Agent for Multimodal Processing
        self.fusion_agent = DynamicFusionAgent()
        self.fusion_initialized = False
        
        # Prediction Logger
        if config.log_predictions:
            self.prediction_logger = FeaturePredictionLogger(
                buffer_size=1000,
                output_path=f"{config.output_dir}/predictions/predictions.parquet"
            )
        else:
            self.prediction_logger = None
        
        # Training State
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup output directories"""
        directories = [
            self.config.output_dir,
            f"{self.config.output_dir}/checkpoints",
            f"{self.config.output_dir}/logs",
            f"{self.config.output_dir}/datasets",
            f"{self.config.output_dir}/predictions"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize_fusion_agent(self) -> bool:
        """Initialize Dynamic Fusion Agent for multimodal processing"""
        if not self.fusion_initialized:
            try:
                success = await self.fusion_agent.initialize_backends()
                self.fusion_initialized = success
                if success:
                    self.logger.info("✅ Dynamic Fusion Agent initialized")
                else:
                    self.logger.warning("⚠️ Dynamic Fusion Agent initialization failed")
                return success
            except Exception as e:
                self.logger.error(f"❌ Fusion agent initialization error: {e}")
                return False
        return True
    
    async def prepare_training_data_with_fusion(self, bars: List[Bar]) -> str:
        """
        Prepare Training Data mit BarDatasetBuilder und Dynamic Fusion Agent
        
        Args:
            bars: List of Nautilus Bar objects
            
        Returns:
            Path to generated dataset
        """
        self.logger.info(f"Preparing enhanced training data from {len(bars)} bars with multimodal fusion")
        
        # Initialize fusion agent
        await self.initialize_fusion_agent()
        
        # Process bars through dataset builder with enhanced features
        enhanced_samples = []
        
        for bar in tqdm(bars, desc="Processing bars with fusion"):
            # Standard dataset builder processing
            self.dataset_builder.on_bar(bar)
            
            # Enhanced multimodal processing if fusion agent available
            if self.fusion_initialized:
                try:
                    # Extract enhanced features
                    enhanced_features = self.feature_extractor.extract_enhanced_features(bar)
                    
                    # Create multimodal input
                    multimodal_input = MultimodalInput(
                        numerical_data={
                            "ohlcv": {
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": float(bar.volume)
                            },
                            "indicators": enhanced_features
                        },
                        text_prompt=f"Analyze {getattr(bar.bar_type, 'instrument_id', 'EUR/USD')} trading pattern",
                        metadata={
                            "timestamp": bar.ts_init,
                            "instrument": str(getattr(bar.bar_type, 'instrument_id', 'EUR/USD'))
                        }
                    )
                    
                    # Process with fusion agent
                    fusion_result = await self.fusion_agent.process_multimodal_input(
                        multimodal_input, 
                        fusion_mode=FusionMode.TEXT_DOMINANT  # Focus on numerical analysis
                    )
                    
                    # Store enhanced sample
                    enhanced_sample = {
                        "bar_features": enhanced_features,
                        "fusion_confidence": fusion_result.fusion_confidence,
                        "fusion_insights": fusion_result.combined_insights,
                        "processing_time": fusion_result.processing_time
                    }
                    enhanced_samples.append(enhanced_sample)
                    
                except Exception as e:
                    self.logger.debug(f"Fusion processing failed for bar: {e}")
        
        # Log fusion statistics
        if enhanced_samples:
            avg_confidence = np.mean([s["fusion_confidence"] for s in enhanced_samples])
            avg_processing_time = np.mean([s["processing_time"] for s in enhanced_samples])
            self.logger.info(f"Fusion processing: {len(enhanced_samples)} samples, avg confidence: {avg_confidence:.3f}, avg time: {avg_processing_time:.3f}s")
        
        # Continue with standard processing
        for bar in tqdm(bars, desc="Standard processing"):
            pass  # Already processed above
        
        # Export to Parquet
        dataset_path = f"{self.config.output_dir}/datasets/training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        success = self.dataset_builder.to_parquet(
            dataset_path,
            compression="zstd",
            include_metadata=True
        )
        
        if success:
            # Get statistics
            stats = self.dataset_builder.get_stats()
            self.logger.info(f"Dataset created: {stats}")
            
            # Save config (with JSON-safe stats conversion)
            config_path = dataset_path.replace('.parquet', '_config.json')
            
            # Convert Polars Series to JSON-safe format
            json_safe_stats = self._convert_stats_to_json_safe(stats)
            
            with open(config_path, 'w') as f:
                json.dump({
                    'config': self.config.__dict__,
                    'stats': json_safe_stats,
                    'created_at': datetime.now().isoformat()
                }, f, indent=2)
            
            # Save enhanced samples metadata
            if enhanced_samples:
                enhanced_metadata_path = dataset_path.replace('.parquet', '_enhanced_metadata.json')
                
                # Calculate fusion statistics safely
                confidences = [s.get("fusion_confidence", 0.0) for s in enhanced_samples]
                processing_times = [s.get("processing_time", 0.0) for s in enhanced_samples]
                
                with open(enhanced_metadata_path, 'w') as f:
                    json.dump({
                        'enhanced_samples_count': len(enhanced_samples),
                        'fusion_statistics': {
                            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                            'avg_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
                            'fusion_agent_used': self.fusion_initialized,
                            'confidence_range': {
                                'min': float(np.min(confidences)) if confidences else 0.0,
                                'max': float(np.max(confidences)) if confidences else 0.0
                            },
                            'processing_time_range': {
                                'min': float(np.min(processing_times)) if processing_times else 0.0,
                                'max': float(np.max(processing_times)) if processing_times else 0.0
                            }
                        }
                    }, f, indent=2)
            
            return dataset_path
        else:
            raise RuntimeError("Failed to create training dataset")
    
    def prepare_training_data(self, bars: List[Bar]) -> str:
        """
        Synchronous wrapper for prepare_training_data_with_fusion
        
        Args:
            bars: List of Nautilus Bar objects
            
        Returns:
            Path to generated dataset
        """
        import asyncio
        
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.prepare_training_data_with_fusion(bars))
    
    def _convert_stats_to_json_safe(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Polars Series and other non-JSON-serializable objects to JSON-safe format"""
        import polars as pl
        
        json_safe_stats = {}
        
        for key, value in stats.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                json_safe_stats[key] = self._convert_stats_to_json_safe(value)
            elif hasattr(value, 'to_dict') and hasattr(value, 'to_list'):
                # Polars DataFrame/Series
                try:
                    if hasattr(value, 'to_dict'):
                        json_safe_stats[key] = value.to_dict(as_series=False)
                    else:
                        json_safe_stats[key] = value.to_list()
                except Exception:
                    json_safe_stats[key] = str(value)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples recursively
                json_safe_stats[key] = [
                    self._convert_stats_to_json_safe({'item': item})['item'] if isinstance(item, dict) 
                    else str(item) if not isinstance(item, (str, int, float, bool, type(None)))
                    else item
                    for item in value
                ]
            elif isinstance(value, (str, int, float, bool, type(None))):
                # Already JSON-safe
                json_safe_stats[key] = value
            else:
                # Convert everything else to string
                json_safe_stats[key] = str(value)
        
        return json_safe_stats
    
    def setup_model(self) -> bool:
        """Setup model, optimizer, scheduler"""
        try:
            # Create model
            self.model = TradingModel(
                input_dim=10,  # Feature dimension
                hidden_dim=256,
                num_classes=3,  # BUY/SELL/HOLD
                dropout=0.1
            ).to(self.device)
            
            # Optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs * 1000,  # Approximate steps
                eta_min=self.config.learning_rate * 0.1
            )
            
            # Load checkpoint if specified
            if self.config.resume_from_checkpoint:
                self._load_checkpoint(self.config.resume_from_checkpoint)
            
            self.logger.info("Model setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            return False
    
    def fine_tune_model(self, dataset_path: str) -> Dict[str, Any]:
        """
        Enhanced Fine-Tuning mit GPU-Optimierung
        
        Args:
            dataset_path: Path to training dataset
            
        Returns:
            Training results
        """
        self.logger.info("Starting enhanced fine-tuning")
        
        # Setup model
        if not self.setup_model():
            return {"error": "Model setup failed"}
        
        # Create dataset and dataloader
        dataset = TradingDataset(dataset_path, self.feature_extractor, self.config)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        # Training loop
        training_start = time.time()
        training_metrics = []
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_reg_loss': train_metrics['reg_loss'],
                'train_cls_loss': train_metrics['cls_loss'],
                'val_loss': val_metrics['loss'],
                'val_reg_loss': val_metrics['reg_loss'],
                'val_cls_loss': val_metrics['cls_loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            }
            
            training_metrics.append(epoch_metrics)
            
            # Logging
            self.logger.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.3f}")
            
            # Save checkpoint (avoid division by zero)
            save_interval = max(1, self.config.save_steps // max(len(train_loader), 1))
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self._save_best_model()
        
        training_time = time.time() - training_start
        
        # Final results
        results = {
            'training_time': training_time,
            'best_loss': self.best_loss,
            'final_metrics': training_metrics[-1] if training_metrics else {},
            'total_epochs': self.config.num_epochs,
            'model_path': self.config.output_dir,
            'dataset_samples': len(dataset)
        }
        
        # Save results
        self._save_training_results(results, training_metrics)
        
        self.logger.info(f"Fine-tuning completed in {training_time:.2f}s")
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Training epoch mit Mixed-Precision"""
        self.model.train()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            features = batch['features'].to(self.device)
            forward_returns = batch['forward_return'].to(self.device)
            label_classes = batch['label_class'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    pred_returns, pred_classes = self.model(features)
                    
                    # Losses
                    reg_loss = nn.MSELoss()(pred_returns, forward_returns)
                    cls_loss = nn.CrossEntropyLoss()(pred_classes, label_classes)
                    total_batch_loss = reg_loss + 0.5 * cls_loss
                
                # Backward pass
                self.scaler.scale(total_batch_loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
            else:
                pred_returns, pred_classes = self.model(features)
                
                reg_loss = nn.MSELoss()(pred_returns, forward_returns)
                cls_loss = nn.CrossEntropyLoss()(pred_classes, label_classes)
                total_batch_loss = reg_loss + 0.5 * cls_loss
                
                total_batch_loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
            
            # Update progress
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'RegLoss': f'{reg_loss.item():.4f}',
                'ClsLoss': f'{cls_loss.item():.4f}'
            })
            
            # Log predictions
            if self.prediction_logger and batch_idx % self.config.logging_steps == 0:
                self._log_predictions(batch, pred_returns, pred_classes)
            
            self.global_step += 1
        
        return {
            'loss': total_loss / num_batches,
            'reg_loss': total_reg_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                forward_returns = batch['forward_return'].to(self.device)
                label_classes = batch['label_class'].to(self.device)
                
                pred_returns, pred_classes = self.model(features)
                
                # Losses
                reg_loss = nn.MSELoss()(pred_returns, forward_returns)
                cls_loss = nn.CrossEntropyLoss()(pred_classes, label_classes)
                total_batch_loss = reg_loss + 0.5 * cls_loss
                
                total_loss += total_batch_loss.item()
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                
                # Accuracy
                _, predicted = torch.max(pred_classes, 1)
                correct_predictions += (predicted == label_classes).sum().item()
                total_predictions += label_classes.size(0)
        
        num_batches = len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'reg_loss': total_reg_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'accuracy': accuracy
        }
    
    def _log_predictions(self, batch: Dict[str, torch.Tensor], pred_returns: torch.Tensor, pred_classes: torch.Tensor):
        """Log predictions for analysis"""
        if not self.prediction_logger:
            return
        
        try:
            batch_size = pred_returns.size(0)
            
            for i in range(min(batch_size, 10)):  # Log first 10 samples
                # Safely extract features with None handling
                features_dict = {}
                for j in range(batch['features'].size(1)):
                    try:
                        value = batch['features'][i][j].item()
                        features_dict[f'feature_{j}'] = float(value) if value is not None else 0.0
                    except (ValueError, TypeError, AttributeError):
                        features_dict[f'feature_{j}'] = 0.0
                
                # Safely extract prediction values
                try:
                    pred_return = float(pred_returns[i].item()) if pred_returns[i].item() is not None else 0.0
                    actual_return = float(batch['forward_return'][i].item()) if batch['forward_return'][i].item() is not None else 0.0
                    pred_class = int(torch.argmax(pred_classes[i]).item())
                    actual_class = int(batch['label_class'][i].item()) if batch['label_class'][i].item() is not None else 2
                    confidence = float(torch.softmax(pred_classes[i], dim=0).max().item())
                    timestamp = int(batch['timestamp'][i].item()) if batch['timestamp'][i].item() is not None else 0
                except (ValueError, TypeError, AttributeError):
                    continue  # Skip this sample if extraction fails
                
                prediction_dict = {
                    'action': 'BUY' if pred_class == 0 else 'SELL' if pred_class == 1 else 'HOLD',
                    'predicted_return': pred_return,
                    'actual_return': actual_return,
                    'predicted_class': pred_class,
                    'actual_class': actual_class,
                    'confidence': confidence
                }
                
                # Use correct method name for FeaturePredictionLogger
                self.prediction_logger.log(
                    ts_ns=timestamp,
                    instrument="EUR/USD",
                    features=features_dict,
                    prediction=prediction_dict,
                    confidence_score=confidence
                )
        except Exception as e:
            self.logger.warning(f"Prediction logging failed: {e}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = f"{self.config.output_dir}/checkpoints/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
    
    def _save_best_model(self):
        """Save best model"""
        best_model_path = f"{self.config.output_dir}/best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }, best_model_path)
        
        self.logger.info(f"Best model saved: {best_model_path}")
    
    def _save_training_results(self, results: Dict[str, Any], metrics: List[Dict[str, Any]]):
        """Save training results and metrics"""
        # Save results
        results_path = f"{self.config.output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics as Parquet for analysis
        if metrics:
            metrics_df = pl.DataFrame(metrics)
            metrics_path = f"{self.config.output_dir}/training_metrics.parquet"
            metrics_df.write_parquet(metrics_path, compression="zstd")
        
        self.logger.info(f"Training results saved: {results_path}")
    
    def validate_performance(self, test_dataset_path: str) -> Dict[str, float]:
        """
        Validate model performance on test dataset
        
        Args:
            test_dataset_path: Path to test dataset
            
        Returns:
            Performance metrics
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Load test dataset
        test_dataset = TradingDataset(test_dataset_path, self.feature_extractor, self.config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        )
        
        # Evaluate
        test_metrics = self._validate_epoch(test_loader)
        
        # Additional metrics
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                label_classes = batch['label_class'].to(self.device)
                
                _, pred_classes = self.model(features)
                
                all_predictions.extend(torch.argmax(pred_classes, dim=1).cpu().numpy())
                all_targets.extend(label_classes.cpu().numpy())
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        performance_metrics = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'reg_loss': test_metrics['reg_loss'],
            'cls_loss': test_metrics['cls_loss']
        }
        
        self.logger.info(f"Test Performance: {performance_metrics}")
        return performance_metrics
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
            self.model = None
        
        if self.optimizer:
            del self.optimizer
            self.optimizer = None
        
        if self.scheduler:
            del self.scheduler
            self.scheduler = None
        
        if self.scaler:
            del self.scaler
            self.scaler = None
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        # Flush prediction logger
        if self.prediction_logger:
            self.prediction_logger.flush()
        
        self.logger.info("Fine-tuning cleanup completed")


# Factory Functions
def create_fine_tuning_config(**kwargs) -> FineTuningConfig:
    """Create fine-tuning configuration"""
    return FineTuningConfig(**kwargs)


def create_fine_tuning_manager(config: FineTuningConfig) -> EnhancedFineTuningManager:
    """Create enhanced fine-tuning manager"""
    return EnhancedFineTuningManager(config)