"""
Model Checkpointing und Resume-Funktionalität
Optimiert für lange Training-Sessions auf RTX 5090
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import shutil
import pickle
from datetime import datetime
import hashlib
import os

from .fine_tuning_manager import FineTuningConfig, TrainingMetrics
from .gpu_training_loop import GPUTrainingConfig


@dataclass
class CheckpointMetadata:
    """Metadaten für Checkpoint"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float]
    learning_rate: float
    timestamp: str
    model_hash: str
    config_hash: str
    gpu_memory_peak: float
    training_time: float
    best_metric: Optional[float] = None
    is_best: bool = False


class CheckpointManager:
    """
    Verwaltet Model-Checkpoints mit automatischem Resume
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_best_only: bool = False,
                 monitor_metric: str = "eval_loss",
                 mode: str = "min"):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode  # "min" or "max"
        
        self.logger = logging.getLogger(__name__)
        
        # Checkpoint Tracking
        self.checkpoints = []
        self.best_metric_value = float('inf') if mode == "min" else float('-inf')
        self.best_checkpoint_path = None
        
        # Setup
        self._setup_checkpoint_dir()
        self._load_checkpoint_registry()
    
    def _setup_checkpoint_dir(self):
        """Erstellt Checkpoint-Verzeichnis"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        (self.checkpoint_dir / "models").mkdir(exist_ok=True)
        (self.checkpoint_dir / "metadata").mkdir(exist_ok=True)
        (self.checkpoint_dir / "configs").mkdir(exist_ok=True)
        (self.checkpoint_dir / "logs").mkdir(exist_ok=True)
    
    def _load_checkpoint_registry(self):
        """Lädt Checkpoint-Registry"""
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                self.checkpoints = registry_data.get("checkpoints", [])
                self.best_metric_value = registry_data.get("best_metric_value", 
                    float('inf') if self.mode == "min" else float('-inf'))
                self.best_checkpoint_path = registry_data.get("best_checkpoint_path")
                
                self.logger.info(f"Loaded checkpoint registry with {len(self.checkpoints)} entries")
                
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint registry: {e}")
                self.checkpoints = []
    
    def _save_checkpoint_registry(self):
        """Speichert Checkpoint-Registry"""
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        
        registry_data = {
            "checkpoints": self.checkpoints,
            "best_metric_value": self.best_metric_value,
            "best_checkpoint_path": self.best_checkpoint_path,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Berechnet Hash des Model-States"""
        try:
            model_state = model.state_dict()
            
            # Serialize state dict
            model_bytes = pickle.dumps(model_state)
            
            # Calculate hash
            model_hash = hashlib.md5(model_bytes).hexdigest()
            
            return model_hash
            
        except Exception as e:
            self.logger.error(f"Model hash calculation failed: {e}")
            return "unknown"
    
    def _calculate_config_hash(self, config: Union[FineTuningConfig, Dict[str, Any]]) -> str:
        """Berechnet Hash der Konfiguration"""
        try:
            if isinstance(config, FineTuningConfig):
                config_dict = asdict(config)
            else:
                config_dict = config
            
            # Sort for consistent hashing
            config_str = json.dumps(config_dict, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            return config_hash
            
        except Exception as e:
            self.logger.error(f"Config hash calculation failed: {e}")
            return "unknown"
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       step: int,
                       metrics: Dict[str, float],
                       config: FineTuningConfig,
                       scaler: Optional[GradScaler] = None,
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Speichert Checkpoint"""
        
        try:
            # Checkpoint Filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}"
            checkpoint_path = self.checkpoint_dir / "models" / f"{checkpoint_name}.pt"
            
            # Calculate Hashes
            model_hash = self._calculate_model_hash(model)
            config_hash = self._calculate_config_hash(config)
            
            # Prepare Checkpoint Data
            checkpoint_data = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "model_hash": model_hash,
                "config_hash": config_hash,
                "timestamp": timestamp,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
            
            # Add Scaler State
            if scaler is not None:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()
            
            # Add Additional Data
            if additional_data:
                checkpoint_data["additional_data"] = additional_data
            
            # Save Checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Create Metadata
            metadata = CheckpointMetadata(
                epoch=epoch,
                step=step,
                train_loss=metrics.get("train_loss", 0.0),
                eval_loss=metrics.get("eval_loss"),
                learning_rate=metrics.get("learning_rate", 0.0),
                timestamp=timestamp,
                model_hash=model_hash,
                config_hash=config_hash,
                gpu_memory_peak=metrics.get("gpu_memory_peak", 0.0),
                training_time=metrics.get("training_time", 0.0)
            )
            
            # Check if Best Checkpoint
            current_metric = metrics.get(self.monitor_metric)
            if current_metric is not None:
                is_best = self._is_best_checkpoint(current_metric)
                metadata.is_best = is_best
                metadata.best_metric = current_metric
                
                if is_best:
                    self.best_metric_value = current_metric
                    self.best_checkpoint_path = str(checkpoint_path)
                    
                    # Create Best Model Symlink
                    best_link = self.checkpoint_dir / "best_model.pt"
                    if best_link.exists():
                        best_link.unlink()
                    best_link.symlink_to(checkpoint_path.name)
            
            # Save Metadata
            metadata_path = self.checkpoint_dir / "metadata" / f"{checkpoint_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Save Config
            config_path = self.checkpoint_dir / "configs" / f"{checkpoint_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            # Update Registry
            checkpoint_info = {
                "name": checkpoint_name,
                "path": str(checkpoint_path),
                "metadata_path": str(metadata_path),
                "config_path": str(config_path),
                "metadata": asdict(metadata)
            }
            
            self.checkpoints.append(checkpoint_info)
            
            # Cleanup Old Checkpoints
            if not self.save_best_only:
                self._cleanup_old_checkpoints()
            
            # Save Registry
            self._save_checkpoint_registry()
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            if metadata.is_best:
                self.logger.info(f"New best checkpoint! {self.monitor_metric}: {current_metric}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Checkpoint saving failed: {e}")
            raise
    
    def _is_best_checkpoint(self, current_metric: float) -> bool:
        """Prüft ob aktueller Checkpoint der beste ist"""
        if self.mode == "min":
            return current_metric < self.best_metric_value
        else:
            return current_metric > self.best_metric_value
    
    def _cleanup_old_checkpoints(self):
        """Bereinigt alte Checkpoints"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoints, 
            key=lambda x: x["metadata"]["timestamp"]
        )
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            try:
                # Remove files
                checkpoint_path = Path(checkpoint_info["path"])
                metadata_path = Path(checkpoint_info["metadata_path"])
                config_path = Path(checkpoint_info["config_path"])
                
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                if config_path.exists():
                    config_path.unlink()
                
                # Remove from registry
                self.checkpoints.remove(checkpoint_info)
                
                self.logger.info(f"Removed old checkpoint: {checkpoint_info['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {checkpoint_info['name']}: {e}")
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False,
                       scaler: Optional[GradScaler] = None) -> Tuple[int, int, Dict[str, float]]:
        """Lädt Checkpoint"""
        
        try:
            # Determine checkpoint path
            if load_best and self.best_checkpoint_path:
                checkpoint_path = self.best_checkpoint_path
            elif checkpoint_path is None:
                checkpoint_path = self._get_latest_checkpoint_path()
            
            if not checkpoint_path or not Path(checkpoint_path).exists():
                self.logger.warning("No checkpoint found to load")
                return 0, 0, {}
            
            # Load Checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            
            # Load Model State
            model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # Load Optimizer State
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            # Load Scaler State
            if scaler is not None and "scaler_state_dict" in checkpoint_data:
                scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
            
            # Extract Info
            epoch = checkpoint_data["epoch"]
            step = checkpoint_data["step"]
            metrics = checkpoint_data["metrics"]
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"Resumed from epoch {epoch}, step {step}")
            
            return epoch, step, metrics
            
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            raise
    
    def _get_latest_checkpoint_path(self) -> Optional[str]:
        """Gibt Pfad zum neuesten Checkpoint zurück"""
        if not self.checkpoints:
            return None
        
        # Sort by timestamp (newest first)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x["metadata"]["timestamp"],
            reverse=True
        )
        
        return sorted_checkpoints[0]["path"]
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Gibt Checkpoint-Informationen zurück"""
        
        for checkpoint_info in self.checkpoints:
            if checkpoint_info["path"] == checkpoint_path:
                return checkpoint_info
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Listet alle verfügbaren Checkpoints"""
        
        # Sort by timestamp (newest first)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x["metadata"]["timestamp"],
            reverse=True
        )
        
        return sorted_checkpoints
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Gibt Pfad zum besten Checkpoint zurück"""
        return self.best_checkpoint_path
    
    def export_checkpoint(self, 
                         checkpoint_path: str,
                         export_path: str,
                         include_optimizer: bool = False) -> str:
        """Exportiert Checkpoint für Deployment"""
        
        try:
            # Load Checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            
            # Prepare Export Data
            export_data = {
                "model_state_dict": checkpoint_data["model_state_dict"],
                "metrics": checkpoint_data["metrics"],
                "model_hash": checkpoint_data["model_hash"],
                "config_hash": checkpoint_data["config_hash"],
                "timestamp": checkpoint_data["timestamp"],
                "export_timestamp": datetime.now().isoformat(),
                "pytorch_version": checkpoint_data["pytorch_version"]
            }
            
            # Include Optimizer if requested
            if include_optimizer:
                export_data["optimizer_state_dict"] = checkpoint_data["optimizer_state_dict"]
            
            # Save Export
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(export_data, export_path)
            
            self.logger.info(f"Checkpoint exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Checkpoint export failed: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Validiert Checkpoint-Integrität"""
        
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Check file exists
            if not checkpoint_path.exists():
                validation_result["errors"].append("Checkpoint file does not exist")
                return validation_result
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            
            # Required keys
            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "step"]
            missing_keys = [key for key in required_keys if key not in checkpoint_data]
            
            if missing_keys:
                validation_result["errors"].extend([f"Missing key: {key}" for key in missing_keys])
            
            # Check model state dict
            model_state = checkpoint_data.get("model_state_dict", {})
            if not model_state:
                validation_result["errors"].append("Empty model state dict")
            
            # Check metrics
            metrics = checkpoint_data.get("metrics", {})
            validation_result["info"]["metrics"] = metrics
            
            # Check versions
            pytorch_version = checkpoint_data.get("pytorch_version")
            if pytorch_version != torch.__version__:
                validation_result["warnings"].append(
                    f"PyTorch version mismatch: checkpoint={pytorch_version}, current={torch.__version__}"
                )
            
            # File size check
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            validation_result["info"]["file_size_mb"] = file_size_mb
            
            if file_size_mb < 1:
                validation_result["warnings"].append("Checkpoint file is very small")
            
            # Success if no errors
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    def create_training_resume_info(self) -> Dict[str, Any]:
        """Erstellt Resume-Informationen für Training"""
        
        resume_info = {
            "has_checkpoints": len(self.checkpoints) > 0,
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "total_checkpoints": len(self.checkpoints),
            "checkpoint_dir": str(self.checkpoint_dir)
        }
        
        if self.checkpoints:
            # Latest checkpoint
            latest = max(self.checkpoints, key=lambda x: x["metadata"]["timestamp"])
            resume_info["latest_checkpoint"] = {
                "path": latest["path"],
                "epoch": latest["metadata"]["epoch"],
                "step": latest["metadata"]["step"],
                "train_loss": latest["metadata"]["train_loss"],
                "eval_loss": latest["metadata"]["eval_loss"],
                "timestamp": latest["metadata"]["timestamp"]
            }
            
            # Best checkpoint
            if self.best_checkpoint_path:
                best_info = self.get_checkpoint_info(self.best_checkpoint_path)
                if best_info:
                    resume_info["best_checkpoint"] = {
                        "path": best_info["path"],
                        "epoch": best_info["metadata"]["epoch"],
                        "step": best_info["metadata"]["step"],
                        "metric_value": best_info["metadata"]["best_metric"],
                        "timestamp": best_info["metadata"]["timestamp"]
                    }
        
        return resume_info
    
    def cleanup_corrupted_checkpoints(self) -> int:
        """Bereinigt korrupte Checkpoints"""
        
        corrupted_count = 0
        
        for checkpoint_info in self.checkpoints.copy():
            validation_result = self.validate_checkpoint(checkpoint_info["path"])
            
            if not validation_result["valid"]:
                try:
                    # Remove corrupted checkpoint
                    checkpoint_path = Path(checkpoint_info["path"])
                    metadata_path = Path(checkpoint_info["metadata_path"])
                    config_path = Path(checkpoint_info["config_path"])
                    
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    if config_path.exists():
                        config_path.unlink()
                    
                    self.checkpoints.remove(checkpoint_info)
                    corrupted_count += 1
                    
                    self.logger.warning(f"Removed corrupted checkpoint: {checkpoint_info['name']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to remove corrupted checkpoint: {e}")
        
        if corrupted_count > 0:
            self._save_checkpoint_registry()
        
        return corrupted_count
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Gibt Training-Statistiken zurück"""
        
        if not self.checkpoints:
            return {"message": "No checkpoints available"}
        
        # Extract metrics from all checkpoints
        train_losses = []
        eval_losses = []
        epochs = []
        
        for checkpoint_info in self.checkpoints:
            metadata = checkpoint_info["metadata"]
            
            epochs.append(metadata["epoch"])
            train_losses.append(metadata["train_loss"])
            
            if metadata["eval_loss"] is not None:
                eval_losses.append(metadata["eval_loss"])
        
        statistics = {
            "total_checkpoints": len(self.checkpoints),
            "epochs_trained": max(epochs) if epochs else 0,
            "train_loss": {
                "current": train_losses[-1] if train_losses else None,
                "best": min(train_losses) if train_losses else None,
                "worst": max(train_losses) if train_losses else None,
                "mean": np.mean(train_losses) if train_losses else None
            }
        }
        
        if eval_losses:
            statistics["eval_loss"] = {
                "current": eval_losses[-1],
                "best": min(eval_losses),
                "worst": max(eval_losses),
                "mean": np.mean(eval_losses)
            }
        
        return statistics