"""
MiniCPM Model Factory - Optimiert für RTX 5090 Hardware
"""

import torch
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from .multimodal_ai import MultimodalAI, ModelConfig, InferenceConfig
from ..core.resource_manager import ResourceManager
from ..core.hardware_detector import HardwareDetector


class ModelFactory:
    """
    Factory für MiniCPM Model-Instanzen mit Hardware-optimierter Konfiguration
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager
        self.hardware_detector = HardwareDetector()
        self.logger = logging.getLogger(__name__)
        
        # Hardware-spezifische Konfigurationen
        self.hardware_configs = self._create_hardware_configs()
    
    def _create_hardware_configs(self) -> Dict[str, ModelConfig]:
        """Erstellt Hardware-spezifische Model-Konfigurationen"""
        configs = {}
        
        # RTX 5090 Optimierte Konfiguration
        configs["rtx_5090"] = ModelConfig(
            model_name="openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float16,
            use_flash_attention=True,
            enable_mixed_precision=True,
            gradient_checkpointing=False,  # RTX 5090 hat genug VRAM
            max_batch_size=8,
            max_sequence_length=4096,
            image_resolution=448,
            use_tensor_cores=True,
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        # RTX 4090 Konfiguration (weniger VRAM)
        configs["rtx_4090"] = ModelConfig(
            model_name="openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float16,
            use_flash_attention=True,
            enable_mixed_precision=True,
            gradient_checkpointing=True,
            max_batch_size=4,
            max_sequence_length=2048,
            image_resolution=336,
            use_tensor_cores=True,
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        # RTX 3080/3090 Konfiguration
        configs["rtx_3080"] = ModelConfig(
            model_name="openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float16,
            use_flash_attention=False,  # Möglicherweise nicht unterstützt
            enable_mixed_precision=True,
            gradient_checkpointing=True,
            max_batch_size=2,
            max_sequence_length=1024,
            image_resolution=224,
            use_tensor_cores=True,
            load_in_8bit=True,
            load_in_4bit=False
        )
        
        # CPU Fallback Konfiguration
        configs["cpu"] = ModelConfig(
            model_name="openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float32,
            use_flash_attention=False,
            enable_mixed_precision=False,
            gradient_checkpointing=True,
            max_batch_size=1,
            max_sequence_length=512,
            image_resolution=224,
            use_tensor_cores=False,
            load_in_8bit=False,
            load_in_4bit=True  # CPU kann 4-bit nutzen
        )
        
        return configs
    
    def detect_optimal_config(self) -> ModelConfig:
        """
        Erkennt automatisch die optimale Konfiguration basierend auf Hardware
        """
        try:
            # Hardware Detection
            hardware_info = self.hardware_detector.detect_hardware()
            gpu_info = hardware_info.get("gpu", {})
            
            if not gpu_info.get("available", False):
                self.logger.info("No GPU detected, using CPU configuration")
                return self.hardware_configs["cpu"]
            
            gpu_name = gpu_info.get("name", "").lower()
            vram_gb = gpu_info.get("memory_gb", 0)
            
            # RTX 5090 Detection
            if "rtx 5090" in gpu_name or vram_gb >= 30:
                self.logger.info("RTX 5090 detected, using optimized configuration")
                return self.hardware_configs["rtx_5090"]
            
            # RTX 4090 Detection
            elif "rtx 4090" in gpu_name or (20 <= vram_gb < 30):
                self.logger.info("RTX 4090 detected, using optimized configuration")
                return self.hardware_configs["rtx_4090"]
            
            # RTX 3080/3090 Detection
            elif any(gpu in gpu_name for gpu in ["rtx 3080", "rtx 3090"]) or (10 <= vram_gb < 20):
                self.logger.info("RTX 3080/3090 detected, using optimized configuration")
                return self.hardware_configs["rtx_3080"]
            
            # Fallback für andere GPUs
            else:
                self.logger.info(f"Unknown GPU ({gpu_name}), using RTX 3080 configuration as fallback")
                return self.hardware_configs["rtx_3080"]
                
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            return self.hardware_configs["cpu"]
    
    def create_multimodal_ai(self, 
                           config_name: Optional[str] = None,
                           custom_config: Optional[ModelConfig] = None) -> MultimodalAI:
        """
        Erstellt MultimodalAI-Instanz mit optimaler Konfiguration
        """
        try:
            # Bestimme Konfiguration
            if custom_config:
                model_config = custom_config
                self.logger.info("Using custom model configuration")
            elif config_name and config_name in self.hardware_configs:
                model_config = self.hardware_configs[config_name]
                self.logger.info(f"Using predefined configuration: {config_name}")
            else:
                model_config = self.detect_optimal_config()
                self.logger.info("Using auto-detected optimal configuration")
            
            # Erstelle MultimodalAI-Instanz
            multimodal_ai = MultimodalAI(
                model_config=model_config,
                resource_manager=self.resource_manager
            )
            
            self.logger.info(f"MultimodalAI created with config: {model_config.model_name}")
            return multimodal_ai
            
        except Exception as e:
            self.logger.error(f"Failed to create MultimodalAI: {e}")
            raise
    
    def create_inference_config(self, 
                              task_type: str = "trading_analysis",
                              quality_level: str = "balanced") -> InferenceConfig:
        """
        Erstellt Inference-Konfiguration für verschiedene Trading-Tasks
        """
        configs = {
            "trading_analysis": {
                "fast": InferenceConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_new_tokens=256,
                    do_sample=True,
                    num_beams=1
                ),
                "balanced": InferenceConfig(
                    temperature=0.5,
                    top_p=0.9,
                    max_new_tokens=512,
                    do_sample=True,
                    num_beams=1
                ),
                "quality": InferenceConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_new_tokens=1024,
                    do_sample=True,
                    num_beams=2
                )
            },
            "pattern_recognition": {
                "fast": InferenceConfig(
                    temperature=0.1,
                    top_p=0.7,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                ),
                "balanced": InferenceConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_new_tokens=256,
                    do_sample=True,
                    num_beams=1
                ),
                "quality": InferenceConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_new_tokens=512,
                    do_sample=True,
                    num_beams=2
                )
            },
            "strategy_generation": {
                "fast": InferenceConfig(
                    temperature=0.6,
                    top_p=0.9,
                    max_new_tokens=512,
                    do_sample=True,
                    num_beams=1
                ),
                "balanced": InferenceConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=1024,
                    do_sample=True,
                    num_beams=1
                ),
                "quality": InferenceConfig(
                    temperature=0.8,
                    top_p=0.95,
                    max_new_tokens=2048,
                    do_sample=True,
                    num_beams=3
                )
            }
        }
        
        return configs.get(task_type, {}).get(quality_level, InferenceConfig())
    
    def benchmark_configuration(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Benchmarkt eine Model-Konfiguration
        """
        try:
            # Erstelle temporäre AI-Instanz
            ai = MultimodalAI(model_config=config)
            
            # Lade Model
            load_success = ai.load_model()
            
            if not load_success:
                return {"success": False, "error": "Model loading failed"}
            
            # Sammle Performance-Metriken
            model_info = ai.get_model_info()
            
            # Cleanup
            ai.cleanup()
            
            return {
                "success": True,
                "model_loaded": load_success,
                "memory_usage": model_info.get("memory_stats", {}),
                "config_summary": {
                    "model_name": config.model_name,
                    "torch_dtype": str(config.torch_dtype),
                    "max_batch_size": config.max_batch_size,
                    "use_flash_attention": config.use_flash_attention,
                    "enable_mixed_precision": config.enable_mixed_precision
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_available_configs(self) -> Dict[str, Dict[str, Any]]:
        """Gibt verfügbare Konfigurationen zurück"""
        return {
            name: {
                "model_name": config.model_name,
                "torch_dtype": str(config.torch_dtype),
                "max_batch_size": config.max_batch_size,
                "max_sequence_length": config.max_sequence_length,
                "image_resolution": config.image_resolution,
                "memory_efficient": config.load_in_8bit or config.load_in_4bit
            }
            for name, config in self.hardware_configs.items()
        }