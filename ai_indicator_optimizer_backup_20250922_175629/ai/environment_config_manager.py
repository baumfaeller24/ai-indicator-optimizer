#!/usr/bin/env python3
"""
Environment-Variable-basierte Konfiguration für produktive Deployments
Phase 2 Implementation - Core AI Enhancement

Features:
- Environment-Variable-basierte Konfiguration
- Multi-Environment-Support (Development, Staging, Production)
- Configuration-Hot-Reload
- Secure Configuration-Management
- Default-Value-Fallbacks
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from pathlib import Path


class EnvironmentConfigManager:
    """
    Environment-Variable-basierte Konfiguration für AI Trading System
    
    Phase 2 Core AI Enhancement:
    - Environment-Variable-basierte Konfiguration
    - Multi-Environment-Support
    - Configuration-Hot-Reload ohne Restart
    - Secure Configuration-Management
    - Type-Safe Configuration-Loading
    - Default-Value-Fallbacks
    """
    
    def __init__(self, config_prefix: str = "AI_TRADING", environment: Optional[str] = None):
        """
        Initialize Environment Config Manager
        
        Args:
            config_prefix: Prefix für Environment-Variables
            environment: Environment-Name (dev, staging, prod)
        """
        self.config_prefix = config_prefix
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.logger = logging.getLogger(__name__)
        
        # Configuration-Cache
        self._config_cache: Dict[str, Any] = {}
        self._last_reload = datetime.now()
        
        # Configuration-Files
        self.config_dir = Path("config")
        self.config_file = self.config_dir / f"{self.environment}.json"
        self.default_config_file = self.config_dir / "default.json"
        
        # Erstelle Config-Verzeichnis
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"EnvironmentConfigManager initialized for environment: {self.environment}")
    
    def get_config(self, key: str, default: Any = None, config_type: type = str) -> Any:
        """
        Erhalte Konfigurationswert mit Environment-Variable-Fallback
        
        Args:
            key: Konfigurationsschlüssel
            default: Default-Wert falls nicht gefunden
            config_type: Erwarteter Typ der Konfiguration
            
        Returns:
            Konfigurationswert
        """
        try:
            # 1. Environment-Variable (höchste Priorität)
            env_key = f"{self.config_prefix}_{key.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                return self._convert_type(env_value, config_type)
            
            # 2. Environment-spezifische Config-Datei
            if key in self._config_cache:
                return self._convert_type(self._config_cache[key], config_type)
            
            # 3. Lade aus Config-Datei
            config_value = self._load_from_config_file(key)
            if config_value is not None:
                return self._convert_type(config_value, config_type)
            
            # 4. Default-Wert
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting config for {key}: {e}")
            return default
    
    def get_ai_config(self) -> Dict[str, Any]:
        """
        Erhalte AI-spezifische Konfiguration
        
        Returns:
            Dictionary mit AI-Konfiguration
        """
        return {
            # AI-Model-Konfiguration
            "ai_endpoint": self.get_config("ai_endpoint", "http://localhost:8080/predictions/pattern_model"),
            "ai_timeout": self.get_config("ai_timeout", 30, int),
            "ai_retry_attempts": self.get_config("ai_retry_attempts", 3, int),
            "use_mock_ai": self.get_config("use_mock_ai", False, bool),
            
            # Confidence-Konfiguration
            "min_confidence": self.get_config("min_confidence", 0.7, float),
            "confidence_multiplier": self.get_config("confidence_multiplier", 1.5, float),
            "max_confidence_multiplier": self.get_config("max_confidence_multiplier", 3.0, float),
            
            # Feature-Extraction-Konfiguration
            "include_time_features": self.get_config("include_time_features", True, bool),
            "include_technical_indicators": self.get_config("include_technical_indicators", True, bool),
            "include_pattern_features": self.get_config("include_pattern_features", True, bool),
            "rsi_period": self.get_config("rsi_period", 14, int),
            "ma_short_period": self.get_config("ma_short_period", 5, int),
            "ma_long_period": self.get_config("ma_long_period", 20, int),
            
            # Pattern-Analyzer-Konfiguration
            "min_pattern_bars": self.get_config("min_pattern_bars", 3, int),
            "max_pattern_bars": self.get_config("max_pattern_bars", 10, int),
            "body_threshold": self.get_config("body_threshold", 0.1, float),
            "shadow_threshold": self.get_config("shadow_threshold", 2.0, float),
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Erhalte Trading-spezifische Konfiguration
        
        Returns:
            Dictionary mit Trading-Konfiguration
        """
        return {
            # Position-Sizing-Konfiguration
            "base_position_size": self.get_config("base_position_size", 1000, int),
            "max_position_size": self.get_config("max_position_size", 5000, int),
            "min_position_size": self.get_config("min_position_size", 100, int),
            
            # Risk-Management-Konfiguration
            "max_risk_per_trade": self.get_config("max_risk_per_trade", 0.02, float),
            "max_daily_loss": self.get_config("max_daily_loss", 1000, float),
            "max_drawdown": self.get_config("max_drawdown", 0.1, float),
            
            # Kelly-Criterion-Konfiguration
            "use_kelly_criterion": self.get_config("use_kelly_criterion", True, bool),
            "kelly_lookback": self.get_config("kelly_lookback", 50, int),
            "max_kelly_fraction": self.get_config("max_kelly_fraction", 0.25, float),
            
            # Trading-Session-Konfiguration
            "trading_sessions": self.get_config("trading_sessions", ["london", "ny"], list),
            "avoid_news_events": self.get_config("avoid_news_events", True, bool),
            "weekend_trading": self.get_config("weekend_trading", False, bool),
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Erhalte Logging-spezifische Konfiguration
        
        Returns:
            Dictionary mit Logging-Konfiguration
        """
        return {
            # Feature-Logging-Konfiguration
            "feature_log_base_path": self.get_config("feature_log_base_path", "logs/ai/features"),
            "log_buffer_size": self.get_config("log_buffer_size", 1000, int),
            "log_rotation": self.get_config("log_rotation", "daily"),
            "log_compression": self.get_config("log_compression", "zstd"),
            "log_include_pid": self.get_config("log_include_pid", True, bool),
            "log_memory_monitoring": self.get_config("log_memory_monitoring", True, bool),
            
            # Dataset-Builder-Konfiguration
            "dataset_horizon": self.get_config("dataset_horizon", 5, int),
            "min_dataset_bars": self.get_config("min_dataset_bars", 100, int),
            "dataset_export_path": self.get_config("dataset_export_path", "datasets"),
        }
    
    def get_live_control_config(self) -> Dict[str, Any]:
        """
        Erhalte Live-Control-spezifische Konfiguration
        
        Returns:
            Dictionary mit Live-Control-Konfiguration
        """
        return {
            # Redis-Konfiguration
            "redis_host": self.get_config("redis_host", "localhost"),
            "redis_port": self.get_config("redis_port", 6379, int),
            "redis_db": self.get_config("redis_db", 0, int),
            "redis_password": self.get_config("redis_password", None),
            
            # Kafka-Konfiguration
            "kafka_bootstrap_servers": self.get_config("kafka_bootstrap_servers", ["localhost:9092"], list),
            "kafka_control_topic": self.get_config("kafka_control_topic", "trading_control"),
            "kafka_status_topic": self.get_config("kafka_status_topic", "trading_status"),
            
            # Control-Konfiguration
            "enable_live_control": self.get_config("enable_live_control", True, bool),
            "control_check_interval": self.get_config("control_check_interval", 1.0, float),
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Erhalte Database-spezifische Konfiguration
        
        Returns:
            Dictionary mit Database-Konfiguration
        """
        return {
            # PostgreSQL-Konfiguration
            "db_host": self.get_config("db_host", "localhost"),
            "db_port": self.get_config("db_port", 5432, int),
            "db_name": self.get_config("db_name", "trading_library"),
            "db_user": self.get_config("db_user", "trading_user"),
            "db_password": self.get_config("db_password", ""),
            
            # Connection-Pool-Konfiguration
            "db_pool_size": self.get_config("db_pool_size", 10, int),
            "db_max_overflow": self.get_config("db_max_overflow", 20, int),
            "db_pool_timeout": self.get_config("db_pool_timeout", 30, int),
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """
        Erhalte Security-spezifische Konfiguration
        
        Returns:
            Dictionary mit Security-Konfiguration
        """
        return {
            # API-Security
            "api_key": self.get_config("api_key", ""),
            "api_secret": self.get_config("api_secret", ""),
            "jwt_secret": self.get_config("jwt_secret", ""),
            
            # Encryption
            "encryption_key": self.get_config("encryption_key", ""),
            "use_ssl": self.get_config("use_ssl", True, bool),
            
            # Rate-Limiting
            "rate_limit_requests": self.get_config("rate_limit_requests", 100, int),
            "rate_limit_window": self.get_config("rate_limit_window", 60, int),
        }
    
    def get_complete_config(self) -> Dict[str, Any]:
        """
        Erhalte komplette Konfiguration für alle Komponenten
        
        Returns:
            Dictionary mit kompletter Konfiguration
        """
        return {
            "environment": self.environment,
            "config_prefix": self.config_prefix,
            "ai": self.get_ai_config(),
            "trading": self.get_trading_config(),
            "logging": self.get_logging_config(),
            "live_control": self.get_live_control_config(),
            "database": self.get_database_config(),
            "security": self.get_security_config(),
            "last_reload": self._last_reload.isoformat()
        }
    
    def reload_config(self) -> None:
        """Reload Konfiguration aus Dateien"""
        try:
            self._config_cache.clear()
            
            # Lade Default-Konfiguration
            if self.default_config_file.exists():
                with open(self.default_config_file, 'r') as f:
                    default_config = json.load(f)
                    self._config_cache.update(default_config)
            
            # Lade Environment-spezifische Konfiguration
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    env_config = json.load(f)
                    self._config_cache.update(env_config)
            
            self._last_reload = datetime.now()
            self.logger.info(f"Configuration reloaded for environment: {self.environment}")
            
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
    
    def save_config_template(self) -> None:
        """Speichere Konfiguration-Template für aktuelles Environment"""
        try:
            template_config = {
                "# AI Configuration": {
                    "ai_endpoint": "http://localhost:8080/predictions/pattern_model",
                    "min_confidence": 0.7,
                    "use_mock_ai": False
                },
                "# Trading Configuration": {
                    "base_position_size": 1000,
                    "max_position_size": 5000,
                    "max_risk_per_trade": 0.02
                },
                "# Logging Configuration": {
                    "feature_log_base_path": "logs/ai/features",
                    "log_buffer_size": 1000,
                    "log_rotation": "daily"
                },
                "# Live Control Configuration": {
                    "redis_host": "localhost",
                    "redis_port": 6379,
                    "enable_live_control": True
                }
            }
            
            # Flache Struktur für JSON
            flat_config = {}
            for section, configs in template_config.items():
                if isinstance(configs, dict):
                    flat_config.update(configs)
                else:
                    flat_config[section] = configs
            
            with open(self.config_file, 'w') as f:
                json.dump(flat_config, f, indent=2)
            
            self.logger.info(f"Configuration template saved: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration template: {e}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validiere aktuelle Konfiguration
        
        Returns:
            Dictionary mit Validierungsfehlern
        """
        validation_errors = {
            "errors": [],
            "warnings": []
        }
        
        try:
            # AI-Konfiguration validieren
            ai_config = self.get_ai_config()
            
            if ai_config["min_confidence"] < 0 or ai_config["min_confidence"] > 1:
                validation_errors["errors"].append("min_confidence must be between 0 and 1")
            
            if ai_config["ai_timeout"] <= 0:
                validation_errors["errors"].append("ai_timeout must be positive")
            
            # Trading-Konfiguration validieren
            trading_config = self.get_trading_config()
            
            if trading_config["base_position_size"] <= 0:
                validation_errors["errors"].append("base_position_size must be positive")
            
            if trading_config["max_position_size"] < trading_config["base_position_size"]:
                validation_errors["errors"].append("max_position_size must be >= base_position_size")
            
            if trading_config["max_risk_per_trade"] <= 0 or trading_config["max_risk_per_trade"] > 1:
                validation_errors["errors"].append("max_risk_per_trade must be between 0 and 1")
            
            # Logging-Konfiguration validieren
            logging_config = self.get_logging_config()
            
            if logging_config["log_buffer_size"] <= 0:
                validation_errors["errors"].append("log_buffer_size must be positive")
            
            if logging_config["log_rotation"] not in ["daily", "hourly", "none"]:
                validation_errors["warnings"].append("log_rotation should be 'daily', 'hourly', or 'none'")
            
            # Live-Control-Konfiguration validieren
            control_config = self.get_live_control_config()
            
            if control_config["redis_port"] <= 0 or control_config["redis_port"] > 65535:
                validation_errors["errors"].append("redis_port must be between 1 and 65535")
            
        except Exception as e:
            validation_errors["errors"].append(f"Validation error: {e}")
        
        return validation_errors
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Erhalte Environment-Informationen
        
        Returns:
            Dictionary mit Environment-Informationen
        """
        return {
            "environment": self.environment,
            "config_prefix": self.config_prefix,
            "config_file": str(self.config_file),
            "config_file_exists": self.config_file.exists(),
            "default_config_file": str(self.default_config_file),
            "default_config_file_exists": self.default_config_file.exists(),
            "cached_config_keys": list(self._config_cache.keys()),
            "last_reload": self._last_reload.isoformat(),
            "environment_variables": {
                key: value for key, value in os.environ.items() 
                if key.startswith(self.config_prefix)
            }
        }
    
    # Private Methods
    def _load_from_config_file(self, key: str) -> Any:
        """Lade Wert aus Konfigurationsdatei"""
        try:
            if not self._config_cache:
                self.reload_config()
            
            return self._config_cache.get(key)
            
        except Exception as e:
            self.logger.error(f"Error loading from config file: {e}")
            return None
    
    def _convert_type(self, value: Any, target_type: type) -> Any:
        """Konvertiere Wert zu Ziel-Typ"""
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            
            elif target_type == list:
                if isinstance(value, str):
                    # JSON-String oder Komma-separierte Liste
                    try:
                        return json.loads(value)
                    except:
                        return [item.strip() for item in value.split(',')]
                return list(value) if not isinstance(value, list) else value
            
            elif target_type == dict:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value) if not isinstance(value, dict) else value
            
            else:
                return target_type(value)
                
        except Exception as e:
            self.logger.error(f"Error converting {value} to {target_type}: {e}")
            return value


# Factory Function
def create_environment_config_manager(
    config_prefix: str = "AI_TRADING", 
    environment: Optional[str] = None
) -> EnvironmentConfigManager:
    """
    Factory Function für Environment Config Manager
    
    Args:
        config_prefix: Prefix für Environment-Variables
        environment: Environment-Name
        
    Returns:
        EnvironmentConfigManager Instance
    """
    return EnvironmentConfigManager(config_prefix, environment)


# Global Config Manager Instance
_global_config_manager: Optional[EnvironmentConfigManager] = None

def get_global_config_manager() -> EnvironmentConfigManager:
    """
    Erhalte globale Config-Manager-Instanz (Singleton)
    
    Returns:
        Globale EnvironmentConfigManager Instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = create_environment_config_manager()
    return _global_config_manager


if __name__ == "__main__":
    # Test des Environment Config Managers
    print("🧪 Testing EnvironmentConfigManager...")
    
    # Test verschiedene Environments
    for env in ["development", "staging", "production"]:
        print(f"\\n📋 Testing environment: {env}")
        
        config_manager = create_environment_config_manager(environment=env)
        
        # Test Konfiguration-Loading
        ai_config = config_manager.get_ai_config()
        trading_config = config_manager.get_trading_config()
        
        print(f"   AI Endpoint: {ai_config['ai_endpoint']}")
        print(f"   Min Confidence: {ai_config['min_confidence']}")
        print(f"   Base Position Size: {trading_config['base_position_size']}")
        print(f"   Max Risk Per Trade: {trading_config['max_risk_per_trade']}")
        
        # Test Environment-Variable-Override
        os.environ["AI_TRADING_MIN_CONFIDENCE"] = "0.8"
        min_conf = config_manager.get_config("min_confidence", 0.7, float)
        print(f"   Min Confidence (with env override): {min_conf}")
        
        # Test Validation
        validation = config_manager.validate_config()
        print(f"   Validation Errors: {len(validation['errors'])}")
        print(f"   Validation Warnings: {len(validation['warnings'])}")
        
        # Cleanup
        if "AI_TRADING_MIN_CONFIDENCE" in os.environ:
            del os.environ["AI_TRADING_MIN_CONFIDENCE"]
    
    # Test Config-Template-Speicherung
    config_manager = create_environment_config_manager(environment="test")
    config_manager.save_config_template()
    
    # Test Environment-Info
    env_info = config_manager.get_environment_info()
    print(f"\\n📊 Environment Info:")
    print(f"   Environment: {env_info['environment']}")
    print(f"   Config File Exists: {env_info['config_file_exists']}")
    print(f"   Cached Keys: {len(env_info['cached_config_keys'])}")
    
    print("✅ EnvironmentConfigManager Test completed!")