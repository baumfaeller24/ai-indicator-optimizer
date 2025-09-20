#!/usr/bin/env python3
"""
Environment-Variable-basierte Konfiguration
Phase 2 Implementation - Enhanced Multimodal Pattern Recognition Engine

Features:
- Multi-Environment-Support (Development, Staging, Production)
- Environment-Variable-basierte Konfiguration
- Configuration-Hot-Reload ohne System-Restart
- Secure Configuration Management
- Configuration Validation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time


@dataclass
class EnvironmentConfig:
    """Environment Configuration Data Structure"""
    
    # Environment Info
    environment: str = "development"
    debug_mode: bool = True
    
    # AI Configuration
    ai_endpoint: str = "http://localhost:8080/predictions/pattern_model"
    use_mock_ai: bool = True
    min_confidence: float = 0.7
    confidence_multiplier: float = 1.5
    
    # Trading Configuration
    base_position_size: int = 1000
    max_position_size: int = 5000
    min_position_size: int = 100
    max_risk_per_trade: float = 0.02
    
    # Feature Extraction Configuration
    include_time_features: bool = True
    include_technical_indicators: bool = True
    include_pattern_features: bool = True
    include_volatility_features: bool = True
    
    # Logging Configuration
    feature_log_base_path: str = "logs/ai/features"
    log_buffer_size: int = 1000
    log_rotation: str = "daily"
    log_memory_monitoring: bool = True
    
    # Live Control Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    kafka_servers: str = "localhost:9092"
    use_redis: bool = False
    use_kafka: bool = False
    
    # Performance Configuration
    max_history: int = 50
    chart_width: int = 800
    chart_height: int = 600
    
    # Security Configuration
    api_timeout: int = 30
    max_retries: int = 3
    
    # Additional Configuration
    additional_config: Dict[str, Any] = field(default_factory=dict)


class EnvironmentConfigManager:
    """
    Environment Configuration Manager für AI Trading System
    
    Phase 2 Features:
    - Multi-Environment-Support (dev/staging/prod)
    - Environment-Variable-basierte Konfiguration
    - Configuration-Hot-Reload
    - Secure Configuration Management
    - Configuration Validation und Fallbacks
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        environment: Optional[str] = None,
        auto_reload: bool = False
    ):
        """
        Initialize Environment Configuration Manager
        
        Args:
            config_file: Pfad zur Konfigurationsdatei
            environment: Environment-Name (dev/staging/prod)
            auto_reload: Ob automatisches Reload aktiviert werden soll
        """
        self.logger = logging.getLogger(__name__)
        
        # Environment Detection
        self.environment = environment or os.getenv("TRADING_ENV", "development")
        
        # Configuration File
        self.config_file = config_file or self._get_default_config_file()
        
        # Configuration State
        self.config = EnvironmentConfig()
        self.config_lock = threading.RLock()
        self.last_reload = datetime.now()
        
        # Auto-Reload
        self.auto_reload = auto_reload
        self.reload_thread = None
        self.reload_running = False
        
        # Load Initial Configuration
        self.reload_configuration()
        
        # Start Auto-Reload if enabled
        if self.auto_reload:
            self.start_auto_reload()
        
        self.logger.info(f"EnvironmentConfigManager initialized: env={self.environment}, auto_reload={auto_reload}")
    
    def _get_default_config_file(self) -> str:
        """Bestimme Standard-Konfigurationsdatei basierend auf Environment"""
        config_files = {
            "development": "config/dev/ai_config.json",
            "staging": "config/staging/ai_config.json",
            "production": "config/prod/ai_config.json"
        }
        
        return config_files.get(self.environment, "config/ai_config.json")
    
    def reload_configuration(self) -> bool:
        """
        Lade Konfiguration neu aus Environment-Variables und Config-File
        
        Returns:
            True wenn erfolgreich geladen
        """
        try:
            with self.config_lock:
                # 1. Load from Environment Variables
                env_config = self._load_from_environment()
                
                # 2. Load from Configuration File
                file_config = self._load_from_file()
                
                # 3. Merge Configurations (Environment overrides File)
                merged_config = self._merge_configurations(file_config, env_config)
                
                # 4. Validate Configuration
                validated_config = self._validate_configuration(merged_config)
                
                # 5. Update Current Configuration
                self.config = validated_config
                self.last_reload = datetime.now()
                
                self.logger.info(f"Configuration reloaded successfully: env={self.environment}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
            return False
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Lade Konfiguration aus Environment-Variables"""
        env_config = {}
        
        # Environment Info
        env_config["environment"] = os.getenv("TRADING_ENV", self.environment)
        env_config["debug_mode"] = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # AI Configuration
        if os.getenv("AI_ENDPOINT"):
            env_config["ai_endpoint"] = os.getenv("AI_ENDPOINT")
        if os.getenv("USE_MOCK_AI"):
            env_config["use_mock_ai"] = os.getenv("USE_MOCK_AI", "true").lower() == "true"
        if os.getenv("MIN_CONFIDENCE"):
            env_config["min_confidence"] = float(os.getenv("MIN_CONFIDENCE"))
        if os.getenv("CONFIDENCE_MULTIPLIER"):
            env_config["confidence_multiplier"] = float(os.getenv("CONFIDENCE_MULTIPLIER"))
        
        # Trading Configuration
        if os.getenv("BASE_POSITION_SIZE"):
            env_config["base_position_size"] = int(os.getenv("BASE_POSITION_SIZE"))
        if os.getenv("MAX_POSITION_SIZE"):
            env_config["max_position_size"] = int(os.getenv("MAX_POSITION_SIZE"))
        if os.getenv("MIN_POSITION_SIZE"):
            env_config["min_position_size"] = int(os.getenv("MIN_POSITION_SIZE"))
        if os.getenv("MAX_RISK_PER_TRADE"):
            env_config["max_risk_per_trade"] = float(os.getenv("MAX_RISK_PER_TRADE"))
        
        # Feature Configuration
        if os.getenv("INCLUDE_TIME_FEATURES"):
            env_config["include_time_features"] = os.getenv("INCLUDE_TIME_FEATURES", "true").lower() == "true"
        if os.getenv("INCLUDE_TECHNICAL_INDICATORS"):
            env_config["include_technical_indicators"] = os.getenv("INCLUDE_TECHNICAL_INDICATORS", "true").lower() == "true"
        if os.getenv("INCLUDE_PATTERN_FEATURES"):
            env_config["include_pattern_features"] = os.getenv("INCLUDE_PATTERN_FEATURES", "true").lower() == "true"
        if os.getenv("INCLUDE_VOLATILITY_FEATURES"):
            env_config["include_volatility_features"] = os.getenv("INCLUDE_VOLATILITY_FEATURES", "true").lower() == "true"
        
        # Logging Configuration
        if os.getenv("FEATURE_LOG_BASE_PATH"):
            env_config["feature_log_base_path"] = os.getenv("FEATURE_LOG_BASE_PATH")
        if os.getenv("LOG_BUFFER_SIZE"):
            env_config["log_buffer_size"] = int(os.getenv("LOG_BUFFER_SIZE"))
        if os.getenv("LOG_ROTATION"):
            env_config["log_rotation"] = os.getenv("LOG_ROTATION")
        if os.getenv("LOG_MEMORY_MONITORING"):
            env_config["log_memory_monitoring"] = os.getenv("LOG_MEMORY_MONITORING", "true").lower() == "true"
        
        # Live Control Configuration
        if os.getenv("REDIS_HOST"):
            env_config["redis_host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            env_config["redis_port"] = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_DB"):
            env_config["redis_db"] = int(os.getenv("REDIS_DB"))
        if os.getenv("KAFKA_SERVERS"):
            env_config["kafka_servers"] = os.getenv("KAFKA_SERVERS")
        if os.getenv("USE_REDIS"):
            env_config["use_redis"] = os.getenv("USE_REDIS", "false").lower() == "true"
        if os.getenv("USE_KAFKA"):
            env_config["use_kafka"] = os.getenv("USE_KAFKA", "false").lower() == "true"
        
        # Performance Configuration
        if os.getenv("MAX_HISTORY"):
            env_config["max_history"] = int(os.getenv("MAX_HISTORY"))
        if os.getenv("CHART_WIDTH"):
            env_config["chart_width"] = int(os.getenv("CHART_WIDTH"))
        if os.getenv("CHART_HEIGHT"):
            env_config["chart_height"] = int(os.getenv("CHART_HEIGHT"))
        
        # Security Configuration
        if os.getenv("API_TIMEOUT"):
            env_config["api_timeout"] = int(os.getenv("API_TIMEOUT"))
        if os.getenv("MAX_RETRIES"):
            env_config["max_retries"] = int(os.getenv("MAX_RETRIES"))
        
        return env_config
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Lade Konfiguration aus File"""
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_path}")
                return {}
            
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Environment-spezifische Konfiguration extrahieren
            if self.environment in file_config:
                return file_config[self.environment]
            else:
                return file_config
                
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")
            return {}
    
    def _merge_configurations(self, file_config: Dict, env_config: Dict) -> Dict[str, Any]:
        """Merge File- und Environment-Konfiguration (Environment hat Priorität)"""
        merged = file_config.copy()
        merged.update(env_config)
        return merged
    
    def _validate_configuration(self, config_dict: Dict[str, Any]) -> EnvironmentConfig:
        """Validiere und erstelle EnvironmentConfig"""
        try:
            # Erstelle EnvironmentConfig mit Defaults
            config = EnvironmentConfig()
            
            # Update mit geladenen Werten
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    # Unbekannte Konfiguration in additional_config speichern
                    config.additional_config[key] = value
            
            # Validierungen
            self._validate_config_values(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            # Return default configuration on validation failure
            return EnvironmentConfig()
    
    def _validate_config_values(self, config: EnvironmentConfig):
        """Validiere Konfigurationswerte"""
        # Confidence Validation
        if not (0.0 <= config.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {config.min_confidence}")
        
        if config.confidence_multiplier <= 0:
            raise ValueError(f"confidence_multiplier must be positive, got {config.confidence_multiplier}")
        
        # Position Size Validation
        if config.min_position_size <= 0:
            raise ValueError(f"min_position_size must be positive, got {config.min_position_size}")
        
        if config.max_position_size < config.min_position_size:
            raise ValueError(f"max_position_size must be >= min_position_size")
        
        if config.base_position_size < config.min_position_size or config.base_position_size > config.max_position_size:
            raise ValueError(f"base_position_size must be between min and max position size")
        
        # Risk Validation
        if not (0.0 < config.max_risk_per_trade <= 1.0):
            raise ValueError(f"max_risk_per_trade must be between 0.0 and 1.0, got {config.max_risk_per_trade}")
        
        # Buffer Size Validation
        if config.log_buffer_size <= 0:
            raise ValueError(f"log_buffer_size must be positive, got {config.log_buffer_size}")
        
        # Port Validation
        if not (1 <= config.redis_port <= 65535):
            raise ValueError(f"redis_port must be between 1 and 65535, got {config.redis_port}")
        
        # Chart Size Validation
        if config.chart_width <= 0 or config.chart_height <= 0:
            raise ValueError(f"Chart dimensions must be positive")
    
    def start_auto_reload(self):
        """Starte automatisches Configuration-Reload"""
        if self.reload_running:
            return
        
        self.reload_running = True
        self.reload_thread = threading.Thread(target=self._auto_reload_loop, daemon=True)
        self.reload_thread.start()
        
        self.logger.info("Auto-reload started")
    
    def stop_auto_reload(self):
        """Stoppe automatisches Configuration-Reload"""
        self.reload_running = False
        
        if self.reload_thread and self.reload_thread.is_alive():
            self.reload_thread.join(timeout=5)
        
        self.logger.info("Auto-reload stopped")
    
    def _auto_reload_loop(self):
        """Auto-Reload-Loop"""
        while self.reload_running:
            try:
                # Check if configuration file has changed
                if self._config_file_changed():
                    self.reload_configuration()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in auto-reload loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _config_file_changed(self) -> bool:
        """Prüfe ob Konfigurationsdatei geändert wurde"""
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                return False
            
            file_mtime = datetime.fromtimestamp(config_path.stat().st_mtime)
            return file_mtime > self.last_reload
            
        except Exception:
            return False
    
    def get_config(self) -> EnvironmentConfig:
        """Erhalte aktuelle Konfiguration (Thread-safe)"""
        with self.config_lock:
            return self.config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Erhalte Konfiguration als Dictionary"""
        with self.config_lock:
            config_dict = {}
            
            # Standard-Felder
            for field_name in self.config.__dataclass_fields__:
                if field_name != "additional_config":
                    config_dict[field_name] = getattr(self.config, field_name)
            
            # Additional Config
            config_dict.update(self.config.additional_config)
            
            return config_dict
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update Konfiguration zur Laufzeit
        
        Args:
            updates: Dictionary mit Updates
            
        Returns:
            True wenn erfolgreich
        """
        try:
            with self.config_lock:
                # Create updated config
                current_dict = self.get_config_dict()
                current_dict.update(updates)
                
                # Validate updated config
                updated_config = self._validate_configuration(current_dict)
                
                # Apply updates
                self.config = updated_config
                
                self.logger.info(f"Configuration updated: {list(updates.keys())}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Erhalte Environment-Informationen"""
        return {
            "environment": self.environment,
            "config_file": self.config_file,
            "auto_reload": self.auto_reload,
            "last_reload": self.last_reload.isoformat(),
            "reload_running": self.reload_running
        }
    
    def export_config_template(self, output_file: str):
        """Exportiere Konfiguration-Template"""
        try:
            template = {
                "development": {
                    "debug_mode": True,
                    "use_mock_ai": True,
                    "min_confidence": 0.6,
                    "base_position_size": 1000,
                    "use_redis": False,
                    "use_kafka": False
                },
                "staging": {
                    "debug_mode": False,
                    "use_mock_ai": False,
                    "min_confidence": 0.7,
                    "base_position_size": 2000,
                    "use_redis": True,
                    "use_kafka": False
                },
                "production": {
                    "debug_mode": False,
                    "use_mock_ai": False,
                    "min_confidence": 0.8,
                    "base_position_size": 5000,
                    "use_redis": True,
                    "use_kafka": True
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(template, f, indent=2)
            
            self.logger.info(f"Configuration template exported: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting config template: {e}")
    
    def __enter__(self):
        """Context Manager Support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support"""
        self.stop_auto_reload()


# Global Configuration Manager Instance
_config_manager: Optional[EnvironmentConfigManager] = None


def get_config_manager(
    config_file: Optional[str] = None,
    environment: Optional[str] = None,
    auto_reload: bool = False
) -> EnvironmentConfigManager:
    """
    Erhalte globale Configuration Manager Instance (Singleton)
    
    Args:
        config_file: Pfad zur Konfigurationsdatei
        environment: Environment-Name
        auto_reload: Ob automatisches Reload aktiviert werden soll
    
    Returns:
        EnvironmentConfigManager Instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = EnvironmentConfigManager(
            config_file=config_file,
            environment=environment,
            auto_reload=auto_reload
        )
    
    return _config_manager


def get_config() -> EnvironmentConfig:
    """
    Erhalte aktuelle Konfiguration (Convenience Function)
    
    Returns:
        EnvironmentConfig Instance
    """
    return get_config_manager().get_config()


if __name__ == "__main__":
    # Test des Environment Config Managers
    print("🧪 Testing EnvironmentConfigManager...")
    
    # Test mit verschiedenen Environments
    for env in ["development", "staging", "production"]:
        print(f"\\n📋 Testing environment: {env}")
        
        config_manager = EnvironmentConfigManager(environment=env)
        config = config_manager.get_config()
        
        print(f"   Environment: {config.environment}")
        print(f"   Debug Mode: {config.debug_mode}")
        print(f"   AI Endpoint: {config.ai_endpoint}")
        print(f"   Min Confidence: {config.min_confidence}")
        print(f"   Base Position Size: {config.base_position_size}")
        print(f"   Use Redis: {config.use_redis}")
    
    # Test Configuration Update
    print("\\n📋 Testing configuration update...")
    config_manager.update_config({"min_confidence": 0.9, "debug_mode": False})
    updated_config = config_manager.get_config()
    print(f"   Updated Min Confidence: {updated_config.min_confidence}")
    print(f"   Updated Debug Mode: {updated_config.debug_mode}")
    
    # Test Template Export
    print("\\n📋 Testing template export...")
    config_manager.export_config_template("test_config_template.json")
    
    print("✅ EnvironmentConfigManager Test abgeschlossen!")