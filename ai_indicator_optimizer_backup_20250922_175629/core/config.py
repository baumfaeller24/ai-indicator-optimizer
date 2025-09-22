"""
System-Konfiguration und Environment Setup
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """MiniCPM-4.1-8B Model Konfiguration"""
    model_name: str = "openbmb/MiniCPM-V-2_6"  # Placeholder für MiniCPM-4.1-8B
    cache_dir: str = "./models"
    device: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class DataConfig:
    """Daten-Pipeline Konfiguration"""
    symbol: str = "EURUSD"
    timeframe: str = "1m"
    lookback_days: int = 14
    data_dir: str = "./data"
    cache_size: int = 10000  # Anzahl Datenpunkte im Cache
    parallel_downloads: int = 32  # Nutze alle CPU Kerne
    validation_enabled: bool = True
    
    # Indikator Konfiguration
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'rsi': {'period': 14},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger': {'period': 20, 'std': 2},
        'sma': {'periods': [20, 50, 200]},
        'ema': {'periods': [12, 26]},
        'stochastic': {'k_period': 14, 'd_period': 3},
        'atr': {'period': 14},
        'adx': {'period': 14}
    })


@dataclass
class LibraryConfig:
    """Trading Library Konfiguration"""
    database_url: str = "postgresql://localhost/trading_library"
    cache_size_gb: int = 40  # 40GB In-Memory Cache (mehr Platz verfügbar)
    pattern_retention_days: int = 365
    strategy_retention_days: int = 180
    auto_population: bool = True
    
    # Pattern Mining Konfiguration
    historical_years: int = 10
    min_pattern_confidence: float = 0.7
    max_patterns_per_type: int = 1000


@dataclass
class HardwareConfig:
    """Hardware-spezifische Konfiguration"""
    target_cpu: str = "AMD Ryzen 9 9950X"
    target_gpu: str = "NVIDIA RTX 5090"
    target_ram_gb: int = 170  # Praktisch verfügbar, Reserve für System
    target_storage: str = "Samsung 9100 PRO"
    
    # Ressourcen-Limits
    max_cpu_workers: int = 32
    max_gpu_memory_fraction: float = 0.9
    max_ram_usage_fraction: float = 0.75  # 75% von 170GB = ~127GB für AI-Tasks
    
    # Performance Tuning
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False


@dataclass
class TrainingConfig:
    """Model Training Konfiguration"""
    batch_size: int = 8  # Wird dynamisch angepasst
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    
    # Fine-tuning spezifisch
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class PineScriptConfig:
    """Pine Script Generierung Konfiguration"""
    version: str = "5"
    max_indicators: int = 10
    enable_risk_management: bool = True
    default_stop_loss: float = 0.02  # 2%
    default_take_profit: float = 0.04  # 4%
    
    # Code-Optimierung
    enable_optimization: bool = True
    max_code_length: int = 5000
    validate_syntax: bool = True


class SystemConfig:
    """Zentrale System-Konfiguration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./config.json"
        
        # Default Konfigurationen
        self.model = ModelConfig()
        self.data = DataConfig()
        self.library = LibraryConfig()
        self.hardware = HardwareConfig()
        self.training = TrainingConfig()
        self.pine_script = PineScriptConfig()
        
        # Environment Setup
        self.setup_environment()
        
        # Lade Konfiguration falls vorhanden
        self.load_config()
    
    def setup_environment(self):
        """Initialisiert Python Environment"""
        
        # PyTorch Setup
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Optimierungen für RTX 5090
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory Management
            torch.cuda.empty_cache()
            
        else:
            print("CUDA not available, using CPU")
        
        # Multiprocessing Setup
        if hasattr(os, 'sched_setaffinity'):
            # Linux: Nutze alle verfügbaren CPU Kerne
            available_cpus = len(os.sched_getaffinity(0))
            print(f"Available CPU cores: {available_cpus}")
        
        # Environment Variables
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ['OMP_NUM_THREADS'] = str(self.hardware.max_cpu_workers)
        
        # Erstelle notwendige Verzeichnisse
        self._create_directories()
    
    def _create_directories(self):
        """Erstellt notwendige Verzeichnisse"""
        directories = [
            self.model.cache_dir,
            self.data.data_dir,
            "./logs",
            "./results",
            "./checkpoints"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Lädt Konfiguration aus JSON-Datei"""
        if not os.path.exists(self.config_path):
            self.save_config()  # Erstelle Default-Konfiguration
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update Konfigurationen
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            print(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Speichert aktuelle Konfiguration"""
        config_data = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'library': self.library.__dict__,
            'hardware': self.hardware.__dict__,
            'training': self.training.__dict__,
            'pine_script': self.pine_script.__dict__
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def update_config(self, section: str, **kwargs):
        """Aktualisiert Konfiguration"""
        if hasattr(self, section):
            config_obj = getattr(self, section)
            for key, value in kwargs.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    print(f"Warning: Unknown config key {section}.{key}")
        else:
            print(f"Warning: Unknown config section {section}")
    
    def get_torch_device(self) -> torch.device:
        """Gibt optimales PyTorch Device zurück"""
        if self.model.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.model.device)
    
    def get_torch_dtype(self) -> torch.dtype:
        """Gibt PyTorch Datentyp zurück"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.model.torch_dtype, torch.float16)
    
    def validate_hardware_requirements(self) -> Dict[str, bool]:
        """Validiert Hardware-Anforderungen"""
        from .hardware_detector import HardwareDetector
        
        detector = HardwareDetector()
        checks = detector.is_target_hardware()
        
        requirements_met = {
            'cpu_sufficient': detector.cpu_info.cores_logical >= 16 if detector.cpu_info else False,
            'gpu_available': torch.cuda.is_available(),
            'memory_sufficient': detector.memory_info.total >= 64 * 1024**3 if detector.memory_info else False,
            'target_hardware': all(checks.values())
        }
        
        return requirements_met
    
    def print_config_summary(self):
        """Gibt Konfigurations-Zusammenfassung aus"""
        print("=== System Configuration Summary ===")
        
        print(f"Model: {self.model.model_name}")
        print(f"Device: {self.get_torch_device()}")
        print(f"Data Symbol: {self.data.symbol}")
        print(f"Lookback Days: {self.data.lookback_days}")
        print(f"Max CPU Workers: {self.hardware.max_cpu_workers}")
        print(f"GPU Memory Fraction: {self.hardware.max_gpu_memory_fraction}")
        print(f"Library Cache: {self.library.cache_size_gb} GB")
        
        # Hardware Validation
        hw_status = self.validate_hardware_requirements()
        print(f"\nHardware Status:")
        for requirement, status in hw_status.items():
            print(f"  {requirement}: {'✓' if status else '✗'}")


# Global Config Instance
config = SystemConfig()