"""
GPU-optimierte Training Loop für RTX 5090
Mixed-Precision Training mit Memory-Optimierung
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import json
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from ..ai.multimodal_ai import GPUMemoryManager
from .fine_tuning_manager import FineTuningConfig, TrainingMetrics


@dataclass
class GPUTrainingConfig:
    """GPU-spezifische Training-Konfiguration für RTX 5090"""
    
    # RTX 5090 Optimierungen
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile_model: bool = True  # PyTorch 2.0 Compilation
    
    # Memory Management für 32GB VRAM
    max_memory_usage: float = 0.9  # 90% von 32GB
    gradient_accumulation_steps: int = 4
    max_batch_size: int = 8
    adaptive_batch_size: bool = True
    
    # Performance Optimierungen
    pin_memory: bool = True
    non_blocking: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Monitoring
    memory_monitoring: bool = True
    performance_profiling: bool = True
    gradient_monitoring: bool = True


class MemoryOptimizer:
    """
    Memory-Optimierung für RTX 5090 Training
    """
    
    def __init__(self, config: GPUTrainingConfig):
        self.config = config
        self.gpu_memory_manager = GPUMemoryManager()
        self.logger = logging.getLogger(__name__)
        
        # Memory Tracking
        self.memory_history = []
        self.peak_memory = 0
        
    def optimize_memory_usage(self):
        """Optimiert GPU Memory Usage"""
        
        if not torch.cuda.is_available():
            return
        
        try:
            # Clear Cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set Memory Fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.config.max_memory_usage)
            
            # Enable Memory Pool
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory.set_per_process_memory_fraction(self.config.max_memory_usage)
            
            self.logger.info(f"GPU memory optimized for {self.config.max_memory_usage*100}% usage")
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitort GPU Memory Usage"""
        
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "free": 0}
        
        stats = self.gpu_memory_manager.get_memory_stats()
        
        # Track Peak Memory
        current_allocated = stats["allocated_gb"]
        if current_allocated > self.peak_memory:
            self.peak_memory = current_allocated
        
        # Add to History
        self.memory_history.append({
            "timestamp": time.time(),
            "allocated_gb": current_allocated,
            "free_gb": stats["free_gb"]
        })
        
        # Keep only last 1000 entries
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
        
        return stats
    
    def get_optimal_batch_size(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Bestimmt optimale Batch-Size für verfügbares Memory"""
        
        if not self.config.adaptive_batch_size:
            return self.config.max_batch_size
        
        try:
            # Start mit kleiner Batch-Size
            batch_size = 1
            max_batch_size = self.config.max_batch_size
            
            model.eval()
            
            while batch_size <= max_batch_size:
                try:
                    # Test Batch
                    test_batch = sample_input.repeat(batch_size, 1, 1, 1)
                    
                    with torch.no_grad():
                        _ = model(test_batch)
                    
                    # Check Memory Usage
                    memory_stats = self.monitor_memory()
                    memory_usage_percent = memory_stats["allocated_gb"] / 32.0  # RTX 5090 32GB
                    
                    if memory_usage_percent > 0.85:  # 85% Limit
                        break
                    
                    batch_size += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise e
            
            optimal_batch_size = max(1, batch_size - 1)
            self.logger.info(f"Optimal batch size determined: {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.error(f"Batch size optimization failed: {e}")
            return 1
        finally:
            model.train()
            torch.cuda.empty_cache()
    
    def cleanup_memory(self):
        """Bereinigt GPU Memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


class PerformanceProfiler:
    """
    Performance-Profiling für RTX 5090 Training
    """
    
    def __init__(self, config: GPUTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance Metrics
        self.step_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        
        # GPU Utilization
        self.gpu_utilization = []
        
    def start_step_timer(self) -> float:
        """Startet Step-Timer"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
    
    def end_step_timer(self, start_time: float) -> float:
        """Beendet Step-Timer"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        # Keep only last 1000 entries
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-1000:]
        
        return step_time
    
    def profile_forward_pass(self, forward_fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Profiled Forward Pass"""
        start_time = self.start_step_timer()
        
        result = forward_fn(*args, **kwargs)
        
        forward_time = self.end_step_timer(start_time)
        self.forward_times.append(forward_time)
        
        return result, forward_time
    
    def profile_backward_pass(self, loss: torch.Tensor) -> float:
        """Profiled Backward Pass"""
        start_time = self.start_step_timer()
        
        loss.backward()
        
        backward_time = self.end_step_timer(start_time)
        self.backward_times.append(backward_time)
        
        return backward_time
    
    def profile_optimizer_step(self, optimizer: optim.Optimizer, scaler: Optional[GradScaler] = None) -> float:
        """Profiled Optimizer Step"""
        start_time = self.start_step_timer()
        
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer_time = self.end_step_timer(start_time)
        self.optimizer_times.append(optimizer_time)
        
        return optimizer_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        
        stats = {}
        
        if self.step_times:
            stats["step_times"] = {
                "mean": np.mean(self.step_times),
                "std": np.std(self.step_times),
                "min": np.min(self.step_times),
                "max": np.max(self.step_times),
                "steps_per_second": 1.0 / np.mean(self.step_times)
            }
        
        if self.forward_times:
            stats["forward_times"] = {
                "mean": np.mean(self.forward_times),
                "percentage": np.mean(self.forward_times) / np.mean(self.step_times) * 100
            }
        
        if self.backward_times:
            stats["backward_times"] = {
                "mean": np.mean(self.backward_times),
                "percentage": np.mean(self.backward_times) / np.mean(self.step_times) * 100
            }
        
        if self.optimizer_times:
            stats["optimizer_times"] = {
                "mean": np.mean(self.optimizer_times),
                "percentage": np.mean(self.optimizer_times) / np.mean(self.step_times) * 100
            }
        
        return stats


class GradientMonitor:
    """
    Gradient-Monitoring für Training-Stabilität
    """
    
    def __init__(self, config: GPUTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Gradient Statistics
        self.gradient_norms = []
        self.gradient_stats = defaultdict(list)
        
    def monitor_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Monitort Gradienten"""
        
        total_norm = 0.0
        param_count = 0
        
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Per-Layer Statistics
                self.gradient_stats[name].append(param_norm.item())
                
                # Keep only last 100 entries per layer
                if len(self.gradient_stats[name]) > 100:
                    self.gradient_stats[name] = self.gradient_stats[name][-100:]
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Keep only last 1000 entries
        if len(self.gradient_norms) > 1000:
            self.gradient_norms = self.gradient_norms[-1000:]
        
        gradient_stats["total_norm"] = total_norm
        gradient_stats["param_count"] = param_count
        gradient_stats["avg_norm"] = total_norm / max(1, param_count)
        
        return gradient_stats
    
    def detect_gradient_issues(self) -> Dict[str, bool]:
        """Detektiert Gradient-Probleme"""
        
        issues = {
            "exploding_gradients": False,
            "vanishing_gradients": False,
            "nan_gradients": False
        }
        
        if len(self.gradient_norms) > 10:
            recent_norms = self.gradient_norms[-10:]
            
            # Exploding Gradients
            if np.mean(recent_norms) > 10.0:
                issues["exploding_gradients"] = True
            
            # Vanishing Gradients
            if np.mean(recent_norms) < 1e-6:
                issues["vanishing_gradients"] = True
            
            # NaN Gradients
            if any(np.isnan(norm) or np.isinf(norm) for norm in recent_norms):
                issues["nan_gradients"] = True
        
        return issues


class GPUTrainingLoop:
    """
    Haupt-Training-Loop optimiert für RTX 5090
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: GPUTrainingConfig,
                 fine_tuning_config: FineTuningConfig):
        
        self.model = model
        self.config = config
        self.fine_tuning_config = fine_tuning_config
        self.logger = logging.getLogger(__name__)
        
        # Device Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimizers
        self.memory_optimizer = MemoryOptimizer(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.gradient_monitor = GradientMonitor(config)
        
        # Mixed Precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Monitoring
        self.tensorboard_writer = None
        self.training_metrics = []
        
        # Setup
        self._setup_training()
    
    def _setup_training(self):
        """Setup Training Environment"""
        
        # Memory Optimization
        self.memory_optimizer.optimize_memory_usage()
        
        # Model Optimizations
        if self.config.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Model Compilation (PyTorch 2.0)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.logger.info("Model compiled with PyTorch 2.0")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Tensorboard Setup
        if self.fine_tuning_config.output_dir:
            tensorboard_dir = Path(self.fine_tuning_config.output_dir) / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Trainiert eine Epoch mit RTX 5090 Optimierungen"""
        
        self.model.train()
        epoch_metrics = {
            "train_loss": 0.0,
            "steps": 0,
            "learning_rate": 0.0,
            "memory_usage": 0.0,
            "step_time": 0.0
        }
        
        total_loss = 0.0
        num_steps = 0
        
        # Progress Bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            
            # Start Step Timer
            step_start_time = self.performance_profiler.start_step_timer()
            
            try:
                # Move Batch to GPU
                batch = {k: v.to(self.device, non_blocking=self.config.non_blocking) 
                        for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward Pass mit Mixed Precision
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs, forward_time = self.performance_profiler.profile_forward_pass(
                            self.model, **batch
                        )
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                else:
                    outputs, forward_time = self.performance_profiler.profile_forward_pass(
                        self.model, **batch
                    )
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Gradient Accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward Pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    backward_time = self.performance_profiler.profile_backward_pass(loss)
                
                # Gradient Monitoring
                if self.config.gradient_monitoring:
                    gradient_stats = self.gradient_monitor.monitor_gradients(self.model)
                    gradient_issues = self.gradient_monitor.detect_gradient_issues()
                    
                    # Log Gradient Issues
                    if any(gradient_issues.values()):
                        self.logger.warning(f"Gradient issues detected: {gradient_issues}")
                
                # Optimizer Step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    
                    # Gradient Clipping
                    if self.fine_tuning_config.max_grad_norm > 0:
                        if self.config.use_mixed_precision:
                            self.scaler.unscale_(optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.fine_tuning_config.max_grad_norm
                        )
                    
                    # Optimizer Step
                    optimizer_time = self.performance_profiler.profile_optimizer_step(
                        optimizer, self.scaler
                    )
                    
                    optimizer.zero_grad()
                
                # Memory Monitoring
                if self.config.memory_monitoring:
                    memory_stats = self.memory_optimizer.monitor_memory()
                    epoch_metrics["memory_usage"] = memory_stats["allocated_gb"]
                
                # Update Metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_steps += 1
                
                # Step Time
                step_time = self.performance_profiler.end_step_timer(step_start_time)
                epoch_metrics["step_time"] = step_time
                
                # Learning Rate
                epoch_metrics["learning_rate"] = optimizer.param_groups[0]['lr']
                
                # Update Progress Bar
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Memory": f"{epoch_metrics['memory_usage']:.1f}GB",
                    "Step/s": f"{1/step_time:.2f}"
                })
                
                # Tensorboard Logging
                if self.tensorboard_writer and step % self.fine_tuning_config.logging_steps == 0:
                    global_step = epoch * len(dataloader) + step
                    
                    self.tensorboard_writer.add_scalar("Train/Loss", loss.item(), global_step)
                    self.tensorboard_writer.add_scalar("Train/LearningRate", epoch_metrics["learning_rate"], global_step)
                    self.tensorboard_writer.add_scalar("Train/MemoryUsage", epoch_metrics["memory_usage"], global_step)
                    self.tensorboard_writer.add_scalar("Train/StepTime", step_time, global_step)
                    
                    if self.config.gradient_monitoring and 'gradient_stats' in locals():
                        self.tensorboard_writer.add_scalar("Train/GradientNorm", gradient_stats["total_norm"], global_step)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(f"GPU OOM at step {step}. Cleaning up memory...")
                    self.memory_optimizer.cleanup_memory()
                    
                    # Reduce Batch Size if adaptive
                    if self.config.adaptive_batch_size:
                        self.logger.info("Reducing batch size due to OOM")
                        # This would require dataloader recreation
                    
                    continue
                else:
                    raise e
        
        # Epoch Summary
        epoch_metrics["train_loss"] = total_loss / max(1, num_steps)
        epoch_metrics["steps"] = num_steps
        
        # Performance Stats
        performance_stats = self.performance_profiler.get_performance_stats()
        epoch_metrics.update(performance_stats)
        
        return epoch_metrics
    
    def evaluate_epoch(self, 
                      dataloader: DataLoader,
                      epoch: int) -> Dict[str, float]:
        """Evaluiert Model auf Validation Set"""
        
        self.model.eval()
        
        eval_metrics = {
            "eval_loss": 0.0,
            "eval_steps": 0,
            "eval_memory_usage": 0.0
        }
        
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Eval Epoch {epoch}")
            
            for step, batch in enumerate(pbar):
                
                try:
                    # Move Batch to GPU
                    batch = {k: v.to(self.device, non_blocking=self.config.non_blocking) 
                            for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Forward Pass
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model(**batch)
                    else:
                        outputs = self.model(**batch)
                    
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                    
                    total_loss += loss.item()
                    num_steps += 1
                    
                    # Memory Monitoring
                    if self.config.memory_monitoring and step % 10 == 0:
                        memory_stats = self.memory_optimizer.monitor_memory()
                        eval_metrics["eval_memory_usage"] = memory_stats["allocated_gb"]
                    
                    # Update Progress Bar
                    pbar.set_postfix({
                        "Eval Loss": f"{loss.item():.4f}",
                        "Memory": f"{eval_metrics['eval_memory_usage']:.1f}GB"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"GPU OOM during evaluation at step {step}")
                        self.memory_optimizer.cleanup_memory()
                        continue
                    else:
                        raise e
        
        eval_metrics["eval_loss"] = total_loss / max(1, num_steps)
        eval_metrics["eval_steps"] = num_steps
        
        # Tensorboard Logging
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Eval/Loss", eval_metrics["eval_loss"], epoch)
            self.tensorboard_writer.add_scalar("Eval/MemoryUsage", eval_metrics["eval_memory_usage"], epoch)
        
        return eval_metrics
    
    def save_checkpoint(self, 
                       epoch: int,
                       optimizer: optim.Optimizer,
                       metrics: Dict[str, float],
                       checkpoint_dir: str):
        """Speichert Training-Checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
            "fine_tuning_config": self.fine_tuning_config.__dict__
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       optimizer: optim.Optimizer) -> int:
        """Lädt Training-Checkpoint"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if self.scaler and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            epoch = checkpoint["epoch"]
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}, epoch {epoch}")
            return epoch
            
        except Exception as e:
            self.logger.error(f"Loading checkpoint failed: {e}")
            return 0
    
    def cleanup(self):
        """Bereinigt Training-Ressourcen"""
        
        # Memory Cleanup
        self.memory_optimizer.cleanup_memory()
        
        # Close Tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        self.logger.info("Training loop cleanup completed")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Gibt Training-Zusammenfassung zurück"""
        
        summary = {
            "device": str(self.device),
            "mixed_precision": self.config.use_mixed_precision,
            "gradient_checkpointing": self.config.use_gradient_checkpointing,
            "model_compiled": self.config.compile_model,
            "peak_memory_gb": self.memory_optimizer.peak_memory,
            "performance_stats": self.performance_profiler.get_performance_stats()
        }
        
        return summary