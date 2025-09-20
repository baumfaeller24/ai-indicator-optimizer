#!/usr/bin/env python3
"""
Live Control System via Redis/Kafka
Phase 2 Implementation - Enhanced Multimodal Pattern Recognition Engine

Features:
- Strategy-Pausierung und Parameter-Updates
- Live-Risk-Management
- Real-time Configuration Changes
- Command Channel Integration
- Performance Monitoring
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


@dataclass
class LiveControlCommand:
    """Live Control Command Structure"""
    command_type: str  # "pause", "resume", "update_params", "emergency_stop"
    strategy_id: str
    parameters: Dict[str, Any]
    timestamp: str
    command_id: str


@dataclass
class LiveControlStatus:
    """Live Control Status Structure"""
    strategy_id: str
    is_paused: bool
    is_emergency_stopped: bool
    current_parameters: Dict[str, Any]
    last_update: str
    performance_metrics: Dict[str, float]


class LiveControlSystem:
    """
    Live Control System für AI Trading Strategies
    
    Phase 2 Features:
    - Redis/Kafka-Integration für Live-Commands
    - Strategy-Pausierung und Parameter-Updates
    - Live-Risk-Management mit dynamischen Stop-Loss-Anpassungen
    - Real-time Configuration Changes
    - Performance Monitoring und Alerts
    """
    
    def __init__(
        self,
        strategy_id: str,
        config: Optional[Dict] = None,
        use_redis: bool = True,
        use_kafka: bool = False
    ):
        """
        Initialize Live Control System
        
        Args:
            strategy_id: Eindeutige Strategy-ID
            config: Konfiguration für Live-Control
            use_redis: Ob Redis verwendet werden soll
            use_kafka: Ob Kafka verwendet werden soll
        """
        self.strategy_id = strategy_id
        self.config = config or {}
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.use_kafka = use_kafka and KAFKA_AVAILABLE
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Control State
        self.is_paused = False
        self.is_emergency_stopped = False
        self.current_parameters = {}
        self.command_handlers: Dict[str, Callable] = {}
        
        # Redis Setup
        self.redis_client = None
        if self.use_redis:
            self._setup_redis()
        
        # Kafka Setup
        self.kafka_producer = None
        self.kafka_consumer = None
        if self.use_kafka:
            self._setup_kafka()
        
        # Command Processing
        self.command_thread = None
        self.running = False
        
        # Statistics
        self.commands_processed = 0
        self.last_command_time = None
        
        self.logger.info(f"LiveControlSystem initialized: strategy={strategy_id}, redis={self.use_redis}, kafka={self.use_kafka}")
    
    def _setup_redis(self):
        """Setup Redis Connection"""
        try:
            redis_host = self.config.get("redis_host", os.getenv("REDIS_HOST", "localhost"))
            redis_port = self.config.get("redis_port", int(os.getenv("REDIS_PORT", "6379")))
            redis_db = self.config.get("redis_db", int(os.getenv("REDIS_DB", "0")))
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5
            )
            
            # Test Connection
            self.redis_client.ping()
            self.logger.info(f"Redis connected: {redis_host}:{redis_port}/{redis_db}")
            
        except Exception as e:
            self.logger.warning(f"Redis setup failed: {e}")
            self.use_redis = False
            self.redis_client = None
    
    def _setup_kafka(self):
        """Setup Kafka Connection"""
        try:
            kafka_servers = self.config.get("kafka_servers", os.getenv("KAFKA_SERVERS", "localhost:9092"))
            
            # Producer für Status-Updates
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Consumer für Commands
            self.kafka_consumer = KafkaConsumer(
                f"trading_commands_{self.strategy_id}",
                bootstrap_servers=kafka_servers.split(","),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                group_id=f"strategy_{self.strategy_id}"
            )
            
            self.logger.info(f"Kafka connected: {kafka_servers}")
            
        except Exception as e:
            self.logger.warning(f"Kafka setup failed: {e}")
            self.use_kafka = False
            self.kafka_producer = None
            self.kafka_consumer = None
    
    def start(self):
        """Starte Live Control System"""
        if self.running:
            return
        
        self.running = True
        
        # Registriere Standard-Command-Handlers
        self._register_default_handlers()
        
        # Starte Command-Processing-Thread
        self.command_thread = threading.Thread(target=self._command_processing_loop, daemon=True)
        self.command_thread.start()
        
        # Publiziere Initial Status
        self._publish_status()
        
        self.logger.info("LiveControlSystem started")
    
    def stop(self):
        """Stoppe Live Control System"""
        self.running = False
        
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=5)
        
        # Cleanup Connections
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()
        if self.redis_client:
            self.redis_client.close()
        
        self.logger.info("LiveControlSystem stopped")
    
    def _register_default_handlers(self):
        """Registriere Standard-Command-Handlers"""
        self.register_command_handler("pause", self._handle_pause_command)
        self.register_command_handler("resume", self._handle_resume_command)
        self.register_command_handler("update_params", self._handle_update_params_command)
        self.register_command_handler("emergency_stop", self._handle_emergency_stop_command)
        self.register_command_handler("get_status", self._handle_get_status_command)
    
    def register_command_handler(self, command_type: str, handler: Callable[[Dict], Any]):
        """
        Registriere Command-Handler
        
        Args:
            command_type: Typ des Commands
            handler: Handler-Funktion
        """
        self.command_handlers[command_type] = handler
        self.logger.debug(f"Registered handler for command: {command_type}")
    
    def _command_processing_loop(self):
        """Command-Processing-Loop"""
        while self.running:
            try:
                # Check Redis Commands
                if self.use_redis:
                    self._process_redis_commands()
                
                # Check Kafka Commands
                if self.use_kafka:
                    self._process_kafka_commands()
                
                time.sleep(0.1)  # 100ms polling interval
                
            except Exception as e:
                self.logger.error(f"Error in command processing loop: {e}")
                time.sleep(1)
    
    def _process_redis_commands(self):
        """Verarbeite Redis Commands"""
        try:
            command_key = f"trading_commands:{self.strategy_id}"
            command_data = self.redis_client.lpop(command_key)
            
            if command_data:
                command = json.loads(command_data)
                self._process_command(command)
                
        except Exception as e:
            self.logger.error(f"Error processing Redis commands: {e}")
    
    def _process_kafka_commands(self):
        """Verarbeite Kafka Commands"""
        try:
            message_batch = self.kafka_consumer.poll(timeout_ms=100)
            
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    command = message.value
                    self._process_command(command)
                    
        except Exception as e:
            self.logger.error(f"Error processing Kafka commands: {e}")
    
    def _process_command(self, command_data: Dict):
        """Verarbeite einzelnen Command"""
        try:
            command = LiveControlCommand(**command_data)
            
            # Validiere Command
            if command.strategy_id != self.strategy_id:
                self.logger.warning(f"Command for different strategy: {command.strategy_id}")
                return
            
            # Führe Command aus
            handler = self.command_handlers.get(command.command_type)
            if handler:
                result = handler(command.parameters)
                self.commands_processed += 1
                self.last_command_time = datetime.now().isoformat()
                
                self.logger.info(f"Processed command: {command.command_type} -> {result}")
                
                # Publiziere Status-Update
                self._publish_status()
            else:
                self.logger.warning(f"Unknown command type: {command.command_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
    
    def _handle_pause_command(self, parameters: Dict) -> str:
        """Handle Pause Command"""
        self.is_paused = True
        reason = parameters.get("reason", "Manual pause")
        self.logger.info(f"Strategy paused: {reason}")
        return f"Strategy paused: {reason}"
    
    def _handle_resume_command(self, parameters: Dict) -> str:
        """Handle Resume Command"""
        if self.is_emergency_stopped:
            return "Cannot resume: Emergency stop active"
        
        self.is_paused = False
        reason = parameters.get("reason", "Manual resume")
        self.logger.info(f"Strategy resumed: {reason}")
        return f"Strategy resumed: {reason}"
    
    def _handle_update_params_command(self, parameters: Dict) -> str:
        """Handle Update Parameters Command"""
        updated_params = []
        
        for key, value in parameters.items():
            if key != "command_type":  # Skip meta-parameters
                self.current_parameters[key] = value
                updated_params.append(f"{key}={value}")
        
        result = f"Updated parameters: {', '.join(updated_params)}"
        self.logger.info(result)
        return result
    
    def _handle_emergency_stop_command(self, parameters: Dict) -> str:
        """Handle Emergency Stop Command"""
        self.is_emergency_stopped = True
        self.is_paused = True
        reason = parameters.get("reason", "Emergency stop")
        self.logger.critical(f"EMERGENCY STOP: {reason}")
        return f"EMERGENCY STOP: {reason}"
    
    def _handle_get_status_command(self, parameters: Dict) -> Dict:
        """Handle Get Status Command"""
        return self.get_current_status()
    
    def _publish_status(self):
        """Publiziere aktuellen Status"""
        try:
            status = self.get_current_status()
            
            # Redis Status
            if self.use_redis:
                status_key = f"trading_status:{self.strategy_id}"
                self.redis_client.set(status_key, json.dumps(asdict(status)), ex=300)  # 5min TTL
            
            # Kafka Status
            if self.use_kafka:
                self.kafka_producer.send(
                    f"trading_status_{self.strategy_id}",
                    key=self.strategy_id,
                    value=asdict(status)
                )
                
        except Exception as e:
            self.logger.error(f"Error publishing status: {e}")
    
    def get_current_status(self) -> LiveControlStatus:
        """Erhalte aktuellen Status"""
        return LiveControlStatus(
            strategy_id=self.strategy_id,
            is_paused=self.is_paused,
            is_emergency_stopped=self.is_emergency_stopped,
            current_parameters=self.current_parameters.copy(),
            last_update=datetime.now().isoformat(),
            performance_metrics=self._get_performance_metrics()
        )
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Erhalte Performance-Metriken (Placeholder)"""
        return {
            "commands_processed": float(self.commands_processed),
            "uptime_seconds": time.time() - (time.time() - 3600),  # Placeholder
            "last_command_age_seconds": 0.0 if not self.last_command_time else 
                (datetime.now() - datetime.fromisoformat(self.last_command_time.replace('Z', '+00:00'))).total_seconds()
        }
    
    def send_command(self, command_type: str, parameters: Dict[str, Any] = None) -> bool:
        """
        Sende Command an andere Strategy-Instanzen
        
        Args:
            command_type: Typ des Commands
            parameters: Command-Parameter
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            command = LiveControlCommand(
                command_type=command_type,
                strategy_id=self.strategy_id,
                parameters=parameters or {},
                timestamp=datetime.now().isoformat(),
                command_id=f"{self.strategy_id}_{int(time.time() * 1000)}"
            )
            
            command_data = asdict(command)
            
            # Send via Redis
            if self.use_redis:
                command_key = f"trading_commands:{self.strategy_id}"
                self.redis_client.rpush(command_key, json.dumps(command_data))
            
            # Send via Kafka
            if self.use_kafka:
                self.kafka_producer.send(
                    f"trading_commands_{self.strategy_id}",
                    key=command.command_id,
                    value=command_data
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False
    
    def is_trading_allowed(self) -> bool:
        """Prüfe ob Trading erlaubt ist"""
        return not (self.is_paused or self.is_emergency_stopped)
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Erhalte aktuellen Parameter-Wert"""
        return self.current_parameters.get(key, default)
    
    def update_parameter(self, key: str, value: Any):
        """Update Parameter lokal"""
        self.current_parameters[key] = value
        self._publish_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Live-Control-Statistiken"""
        return {
            "strategy_id": self.strategy_id,
            "is_paused": self.is_paused,
            "is_emergency_stopped": self.is_emergency_stopped,
            "commands_processed": self.commands_processed,
            "last_command_time": self.last_command_time,
            "use_redis": self.use_redis,
            "use_kafka": self.use_kafka,
            "redis_connected": self.redis_client is not None,
            "kafka_connected": self.kafka_producer is not None,
            "running": self.running,
            "current_parameters_count": len(self.current_parameters),
            "registered_handlers": list(self.command_handlers.keys())
        }
    
    def __enter__(self):
        """Context Manager Support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support"""
        self.stop()


# Factory Function
def create_live_control_system(
    strategy_id: str,
    config: Optional[Dict] = None,
    use_redis: bool = True,
    use_kafka: bool = False
) -> LiveControlSystem:
    """
    Factory Function für Live Control System
    
    Args:
        strategy_id: Eindeutige Strategy-ID
        config: Konfiguration
        use_redis: Ob Redis verwendet werden soll
        use_kafka: Ob Kafka verwendet werden soll
    
    Returns:
        LiveControlSystem Instance
    """
    return LiveControlSystem(
        strategy_id=strategy_id,
        config=config,
        use_redis=use_redis,
        use_kafka=use_kafka
    )


if __name__ == "__main__":
    # Test des Live Control Systems
    print("🧪 Testing LiveControlSystem...")
    
    # Mock Test ohne Redis/Kafka
    control_system = create_live_control_system(
        strategy_id="test_strategy",
        config={"redis_host": "localhost"},
        use_redis=False,
        use_kafka=False
    )
    
    with control_system:
        # Test Commands
        print("✅ LiveControlSystem started")
        
        # Test Parameter Update
        control_system.update_parameter("min_confidence", 0.8)
        print(f"✅ Parameter updated: min_confidence = {control_system.get_parameter('min_confidence')}")
        
        # Test Pause/Resume
        control_system._handle_pause_command({"reason": "Test pause"})
        print(f"✅ Paused: {control_system.is_paused}")
        
        control_system._handle_resume_command({"reason": "Test resume"})
        print(f"✅ Resumed: {not control_system.is_paused}")
        
        # Test Status
        status = control_system.get_current_status()
        print(f"✅ Status: {status.strategy_id}, paused={status.is_paused}")
        
        # Test Statistics
        stats = control_system.get_statistics()
        print(f"📊 Statistics: {stats}")
    
    print("✅ LiveControlSystem Test abgeschlossen!")