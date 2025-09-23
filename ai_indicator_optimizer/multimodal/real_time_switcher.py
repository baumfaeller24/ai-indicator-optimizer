#!/usr/bin/env python3
"""
Real-time Switcher für Load-basiertes Model-Selection
U3 - Unified Multimodal Flow Integration - Day 4

Features:
- Real-time Load Monitoring für Ollama und TorchServe
- Intelligent Model Selection basierend auf Performance
- Seamless Failover zwischen Models
- Performance Tracking und Optimization
- Circuit Breaker Pattern für Reliability
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
import threading
from collections import deque, defaultdict

import numpy as np
import psutil
import aiohttp
import requests

# Import existing components
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler
from .multimodal_confidence_scorer import ConfidenceScore


class ModelType(Enum):
    """Verfügbare Model-Typen"""
    OLLAMA = "ollama"
    TORCHSERVE = "torchserve"


class RequestType(Enum):
    """Request-Typen für optimale Model-Selection"""
    VISION_HEAVY = "vision_heavy"      # Chart-Analyse dominant
    FEATURE_HEAVY = "feature_heavy"    # Numerische Features dominant
    BALANCED = "balanced"              # Ausgeglichene Anfrage
    BATCH = "batch"                    # Batch-Processing
    REAL_TIME = "real_time"           # Latenz-kritisch


class LoadLevel(Enum):
    """Load-Level-Kategorien"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%
    OVERLOADED = "overloaded"  # >100%


class CircuitState(Enum):
    """Circuit Breaker States"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Circuit breaker triggered
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class RequestContext:
    """Kontext für Inference-Request"""
    request_id: str
    request_type: RequestType
    priority: int  # 1-10, higher = more important
    max_latency_ms: int
    min_accuracy: float
    timeout_seconds: int
    retry_count: int
    metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Performance-Metriken für ein Model"""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    last_updated: float
    sample_count: int


@dataclass
class LoadMetrics:
    """Load-Metriken für ein Model"""
    current_requests: int
    queue_size: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    disk_io: float
    load_level: LoadLevel
    capacity_utilization: float
    last_updated: float


@dataclass
class ModelSelection:
    """Model-Selection-Ergebnis"""
    selected_model: ModelType
    confidence: float
    reason: str
    expected_latency: float
    expected_accuracy: float
    fallback_model: Optional[ModelType]
    selection_metadata: Dict[str, Any]


@dataclass
class InferenceRequest:
    """Inference-Request-Struktur"""
    request_id: str
    request_type: RequestType
    context: RequestContext
    payload: Dict[str, Any]
    timestamp: float


@dataclass
class InferenceResponse:
    """Inference-Response-Struktur"""
    request_id: str
    model_used: ModelType
    result: Dict[str, Any]
    latency_ms: float
    success: bool
    error_message: Optional[str]
    confidence_score: Optional[float]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class SwitcherConfig:
    """Konfiguration für Real-time Switcher"""
    # Load Monitoring
    monitoring_interval_seconds: float = 1.0
    load_history_size: int = 100
    performance_history_size: int = 1000
    
    # Model Selection
    latency_weight: float = 0.4
    accuracy_weight: float = 0.3
    load_weight: float = 0.2
    reliability_weight: float = 0.1
    
    # Thresholds
    high_load_threshold: float = 0.8
    overload_threshold: float = 0.95
    min_success_rate: float = 0.9
    max_latency_ms: int = 5000
    
    # Circuit Breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 30
    circuit_test_requests: int = 3
    
    # Failover
    enable_failover: bool = True
    failover_timeout_ms: int = 100
    max_retry_attempts: int = 3
    
    # Performance
    enable_performance_tracking: bool = True
    performance_window_minutes: int = 10
    adaptive_selection: bool = True


class LoadMonitor:
    """Überwacht Load-Metriken für Models"""
    
    def __init__(self, config: SwitcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load history
        self.load_history = {
            ModelType.OLLAMA: deque(maxlen=config.load_history_size),
            ModelType.TORCHSERVE: deque(maxlen=config.load_history_size)
        }
        
        # Current metrics
        self.current_metrics = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Request counters
        self.request_counters = defaultdict(int)
        self.request_timestamps = defaultdict(list)
    
    def start_monitoring(self):
        """Startet Load-Monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Load monitoring started")
    
    def stop_monitoring(self):
        """Stoppt Load-Monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Load monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring-Loop (läuft in separatem Thread)"""
        while self.monitoring_active:
            try:
                # Monitor Ollama
                ollama_metrics = self._get_ollama_metrics()
                self.load_history[ModelType.OLLAMA].append(ollama_metrics)
                self.current_metrics[ModelType.OLLAMA] = ollama_metrics
                
                # Monitor TorchServe
                torchserve_metrics = self._get_torchserve_metrics()
                self.load_history[ModelType.TORCHSERVE].append(torchserve_metrics)
                self.current_metrics[ModelType.TORCHSERVE] = torchserve_metrics
                
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval_seconds)
    
    def _get_ollama_metrics(self) -> LoadMetrics:
        """Ermittelt Ollama Load-Metriken"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics (if available)
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass
            
            # Request metrics
            current_requests = self.request_counters[ModelType.OLLAMA]
            
            # Estimate queue size (simplified)
            queue_size = max(0, current_requests - 4)  # Assume 4 concurrent capacity
            
            # Calculate load level
            load_factors = [cpu_usage / 100, memory_usage / 100, gpu_usage / 100]
            avg_load = np.mean(load_factors)
            
            if avg_load >= 0.95:
                load_level = LoadLevel.OVERLOADED
            elif avg_load >= 0.8:
                load_level = LoadLevel.VERY_HIGH
            elif avg_load >= 0.6:
                load_level = LoadLevel.HIGH
            elif avg_load >= 0.4:
                load_level = LoadLevel.MEDIUM
            elif avg_load >= 0.2:
                load_level = LoadLevel.LOW
            else:
                load_level = LoadLevel.VERY_LOW
            
            return LoadMetrics(
                current_requests=current_requests,
                queue_size=queue_size,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                network_io=0.0,  # Simplified
                disk_io=0.0,     # Simplified
                load_level=load_level,
                capacity_utilization=avg_load,
                last_updated=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Ollama metrics collection failed: {e}")
            return self._get_default_metrics()
    
    def _get_torchserve_metrics(self) -> LoadMetrics:
        """Ermittelt TorchServe Load-Metriken"""
        try:
            # Try to get TorchServe metrics via API
            try:
                response = requests.get("http://localhost:8081/metrics", timeout=2)
                if response.status_code == 200:
                    # Parse TorchServe metrics (simplified)
                    metrics_text = response.text
                    
                    # Extract key metrics (this would need proper parsing)
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    
                    # Estimate current requests from metrics
                    current_requests = self.request_counters[ModelType.TORCHSERVE]
                    
                else:
                    raise Exception(f"TorchServe metrics API returned {response.status_code}")
                    
            except Exception:
                # Fallback to system metrics
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                current_requests = self.request_counters[ModelType.TORCHSERVE]
            
            # GPU metrics
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass
            
            # Calculate load level
            load_factors = [cpu_usage / 100, memory_usage / 100, gpu_usage / 100]
            avg_load = np.mean(load_factors)
            
            if avg_load >= 0.95:
                load_level = LoadLevel.OVERLOADED
            elif avg_load >= 0.8:
                load_level = LoadLevel.VERY_HIGH
            elif avg_load >= 0.6:
                load_level = LoadLevel.HIGH
            elif avg_load >= 0.4:
                load_level = LoadLevel.MEDIUM
            elif avg_load >= 0.2:
                load_level = LoadLevel.LOW
            else:
                load_level = LoadLevel.VERY_LOW
            
            return LoadMetrics(
                current_requests=current_requests,
                queue_size=max(0, current_requests - 10),  # Assume 10 concurrent capacity
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                network_io=0.0,
                disk_io=0.0,
                load_level=load_level,
                capacity_utilization=avg_load,
                last_updated=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"TorchServe metrics collection failed: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> LoadMetrics:
        """Gibt Default-Metriken zurück bei Fehlern"""
        return LoadMetrics(
            current_requests=0,
            queue_size=0,
            cpu_usage=50.0,
            memory_usage=50.0,
            gpu_usage=50.0,
            network_io=0.0,
            disk_io=0.0,
            load_level=LoadLevel.MEDIUM,
            capacity_utilization=0.5,
            last_updated=time.time()
        )
    
    def increment_request_counter(self, model_type: ModelType):
        """Erhöht Request-Counter für Model"""
        self.request_counters[model_type] += 1
        self.request_timestamps[model_type].append(time.time())
        
        # Clean old timestamps (keep last 60 seconds)
        cutoff = time.time() - 60
        self.request_timestamps[model_type] = [
            ts for ts in self.request_timestamps[model_type] if ts > cutoff
        ]
    
    def decrement_request_counter(self, model_type: ModelType):
        """Verringert Request-Counter für Model"""
        self.request_counters[model_type] = max(0, self.request_counters[model_type] - 1)
    
    async def get_load_metrics(self, model_type: ModelType) -> LoadMetrics:
        """Gibt aktuelle Load-Metriken für Model zurück"""
        return self.current_metrics.get(model_type, self._get_default_metrics())
    
    def get_load_trend(self, model_type: ModelType, minutes: int = 5) -> float:
        """Berechnet Load-Trend für die letzten N Minuten"""
        try:
            history = list(self.load_history[model_type])
            if len(history) < 2:
                return 0.0
            
            # Get samples from last N minutes
            cutoff = time.time() - (minutes * 60)
            recent_samples = [
                sample for sample in history 
                if sample.last_updated > cutoff
            ]
            
            if len(recent_samples) < 2:
                return 0.0
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(recent_samples))
            y = [sample.capacity_utilization for sample in recent_samples]
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Load trend calculation failed: {e}")
            return 0.0


class PerformanceTracker:
    """Trackt Performance-Metriken für Models"""
    
    def __init__(self, config: SwitcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance history
        self.performance_history = {
            ModelType.OLLAMA: deque(maxlen=config.performance_history_size),
            ModelType.TORCHSERVE: deque(maxlen=config.performance_history_size)
        }
        
        # Current performance
        self.current_performance = {}
        
        # Request tracking
        self.active_requests = {}
        self.completed_requests = defaultdict(list)
    
    def start_request_tracking(self, request_id: str, model_type: ModelType):
        """Startet Request-Tracking"""
        self.active_requests[request_id] = {
            "model_type": model_type,
            "start_time": time.time()
        }
    
    def complete_request_tracking(self, 
                                request_id: str, 
                                success: bool, 
                                error_message: Optional[str] = None):
        """Beendet Request-Tracking und aktualisiert Metriken"""
        if request_id not in self.active_requests:
            return
        
        request_info = self.active_requests.pop(request_id)
        model_type = request_info["model_type"]
        latency = (time.time() - request_info["start_time"]) * 1000  # ms
        
        # Record completed request
        self.completed_requests[model_type].append({
            "latency_ms": latency,
            "success": success,
            "error_message": error_message,
            "timestamp": time.time()
        })
        
        # Keep only recent requests
        cutoff = time.time() - (self.config.performance_window_minutes * 60)
        self.completed_requests[model_type] = [
            req for req in self.completed_requests[model_type]
            if req["timestamp"] > cutoff
        ]
        
        # Update performance metrics
        self._update_performance_metrics(model_type)
    
    def _update_performance_metrics(self, model_type: ModelType):
        """Aktualisiert Performance-Metriken für Model"""
        try:
            requests = self.completed_requests[model_type]
            if not requests:
                return
            
            # Calculate metrics
            latencies = [req["latency_ms"] for req in requests]
            successes = [req["success"] for req in requests]
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95) if len(latencies) > 1 else avg_latency
            p99_latency = np.percentile(latencies, 99) if len(latencies) > 1 else avg_latency
            
            success_rate = np.mean(successes)
            error_rate = 1.0 - success_rate
            
            # Calculate throughput (requests per second)
            if len(requests) > 1:
                time_span = requests[-1]["timestamp"] - requests[0]["timestamp"]
                throughput = len(requests) / max(time_span, 1.0)
            else:
                throughput = 0.0
            
            # System metrics (simplified)
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = 0.0
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass
            
            # Create performance object
            performance = ModelPerformance(
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                throughput_rps=throughput,
                success_rate=success_rate,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                last_updated=time.time(),
                sample_count=len(requests)
            )
            
            # Update current performance
            self.current_performance[model_type] = performance
            
            # Add to history
            self.performance_history[model_type].append(performance)
            
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    async def get_recent_performance(self, model_type: ModelType) -> Optional[ModelPerformance]:
        """Gibt aktuelle Performance-Metriken zurück"""
        return self.current_performance.get(model_type)
    
    def get_performance_trend(self, model_type: ModelType, metric: str = "avg_latency_ms") -> float:
        """Berechnet Performance-Trend für Metrik"""
        try:
            history = list(self.performance_history[model_type])
            if len(history) < 2:
                return 0.0
            
            # Extract metric values
            values = [getattr(perf, metric, 0.0) for perf in history[-10:]]  # Last 10 samples
            
            if len(values) < 2:
                return 0.0
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            return slope
            
        except Exception as e:
            self.logger.error(f"Performance trend calculation failed: {e}")
            return 0.0


class CircuitBreaker:
    """Circuit Breaker Pattern für Model-Reliability"""
    
    def __init__(self, config: SwitcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Circuit states
        self.circuit_states = {
            ModelType.OLLAMA: CircuitState.CLOSED,
            ModelType.TORCHSERVE: CircuitState.CLOSED
        }
        
        # Failure tracking
        self.failure_counts = defaultdict(int)
        self.last_failure_times = {}
        self.recovery_start_times = {}
        self.test_request_counts = defaultdict(int)
    
    def record_success(self, model_type: ModelType):
        """Zeichnet erfolgreiche Request auf"""
        # Reset failure count on success
        self.failure_counts[model_type] = 0
        
        # If in half-open state, check if we can close circuit
        if self.circuit_states[model_type] == CircuitState.HALF_OPEN:
            self.test_request_counts[model_type] += 1
            
            if self.test_request_counts[model_type] >= self.config.circuit_test_requests:
                # Enough successful test requests - close circuit
                self.circuit_states[model_type] = CircuitState.CLOSED
                self.test_request_counts[model_type] = 0
                self.logger.info(f"Circuit breaker closed for {model_type.value}")
    
    def record_failure(self, model_type: ModelType):
        """Zeichnet fehlgeschlagene Request auf"""
        self.failure_counts[model_type] += 1
        self.last_failure_times[model_type] = time.time()
        
        # Check if we should open circuit
        if (self.failure_counts[model_type] >= self.config.circuit_failure_threshold and
            self.circuit_states[model_type] == CircuitState.CLOSED):
            
            self.circuit_states[model_type] = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker opened for {model_type.value}")
        
        # If in half-open state, go back to open
        elif self.circuit_states[model_type] == CircuitState.HALF_OPEN:
            self.circuit_states[model_type] = CircuitState.OPEN
            self.test_request_counts[model_type] = 0
            self.logger.warning(f"Circuit breaker reopened for {model_type.value}")
    
    def can_execute_request(self, model_type: ModelType) -> bool:
        """Prüft ob Request ausgeführt werden kann"""
        state = self.circuit_states[model_type]
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            last_failure = self.last_failure_times.get(model_type, 0)
            if time.time() - last_failure >= self.config.circuit_recovery_timeout:
                # Move to half-open state
                self.circuit_states[model_type] = CircuitState.HALF_OPEN
                self.recovery_start_times[model_type] = time.time()
                self.test_request_counts[model_type] = 0
                self.logger.info(f"Circuit breaker half-opened for {model_type.value}")
                return True
            return False
        elif state == CircuitState.HALF_OPEN:
            # Allow limited test requests
            return self.test_request_counts[model_type] < self.config.circuit_test_requests
        
        return False
    
    def get_circuit_state(self, model_type: ModelType) -> CircuitState:
        """Gibt aktuellen Circuit-State zurück"""
        return self.circuit_states[model_type]


class FallbackStrategies:
    """Fallback-Strategien bei Model-Failures"""
    
    def __init__(self, config: SwitcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def execute_fallback(self, 
                             request: InferenceRequest,
                             primary_error: Exception,
                             secondary_error: Exception) -> InferenceResponse:
        """Führt Fallback-Strategie aus wenn beide Models fehlschlagen"""
        try:
            # Strategy 1: Return cached result if available
            cached_result = await self._try_cached_result(request)
            if cached_result:
                return cached_result
            
            # Strategy 2: Return simplified/mock result
            mock_result = await self._generate_mock_result(request)
            if mock_result:
                return mock_result
            
            # Strategy 3: Return error response
            return InferenceResponse(
                request_id=request.request_id,
                model_used=ModelType.OLLAMA,  # Default
                result={"error": "All models failed", "fallback": True},
                latency_ms=0.0,
                success=False,
                error_message=f"Primary: {primary_error}, Secondary: {secondary_error}",
                confidence_score=0.0,
                timestamp=time.time(),
                metadata={"fallback_strategy": "error_response"}
            )
            
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                model_used=ModelType.OLLAMA,
                result={"error": "Fallback failed"},
                latency_ms=0.0,
                success=False,
                error_message=str(e),
                confidence_score=0.0,
                timestamp=time.time(),
                metadata={"fallback_strategy": "fallback_failed"}
            )
    
    async def _try_cached_result(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Versucht gecachtes Ergebnis zu verwenden"""
        # Simplified cache lookup (would need proper implementation)
        return None
    
    async def _generate_mock_result(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Generiert Mock-Ergebnis als Fallback"""
        try:
            # Generate basic mock result based on request type
            if request.request_type == RequestType.VISION_HEAVY:
                mock_result = {
                    "patterns": [{"name": "neutral", "confidence": 0.5}],
                    "trend": "sideways",
                    "confidence": 0.3,
                    "mock": True
                }
            else:
                mock_result = {
                    "signal": "hold",
                    "confidence": 0.3,
                    "features": {"mock": True},
                    "mock": True
                }
            
            return InferenceResponse(
                request_id=request.request_id,
                model_used=ModelType.OLLAMA,
                result=mock_result,
                latency_ms=10.0,
                success=True,
                error_message=None,
                confidence_score=0.3,
                timestamp=time.time(),
                metadata={"fallback_strategy": "mock_result"}
            )
            
        except Exception as e:
            self.logger.error(f"Mock result generation failed: {e}")
            return None


class RealTimeSwitcher:
    """
    Hauptklasse für Load-basiertes Switching zwischen Ollama und TorchServe
    """
    
    def __init__(self, 
                 ollama_client: Optional[OllamaVisionClient] = None,
                 torchserve_handler: Optional[TorchServeHandler] = None,
                 config: Optional[SwitcherConfig] = None):
        
        self.config = config or SwitcherConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model clients
        self.ollama_client = ollama_client or OllamaVisionClient()
        self.torchserve_handler = torchserve_handler or TorchServeHandler()
        
        # Core components
        self.load_monitor = LoadMonitor(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.circuit_breaker = CircuitBreaker(self.config)
        self.fallback_strategies = FallbackStrategies(self.config)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "ollama_requests": 0,
            "torchserve_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "failover_count": 0,
            "average_selection_time": 0.0
        }
        
        # Start monitoring
        self.load_monitor.start_monitoring()
        
        self.logger.info("Real-time Switcher initialized")
    
    async def select_optimal_model(self, 
                                 request: InferenceRequest) -> ModelSelection:
        """
        Wählt optimales Model basierend auf Load und Performance
        
        Args:
            request: Inference-Anfrage mit Typ und Kontext
            
        Returns:
            ModelSelection mit gewähltem Model und Begründung
        """
        start_time = time.time()
        
        try:
            context = request.context
            
            # Get current load and performance metrics
            ollama_load = await self.load_monitor.get_load_metrics(ModelType.OLLAMA)
            torchserve_load = await self.load_monitor.get_load_metrics(ModelType.TORCHSERVE)
            
            ollama_perf = await self.performance_tracker.get_recent_performance(ModelType.OLLAMA)
            torchserve_perf = await self.performance_tracker.get_recent_performance(ModelType.TORCHSERVE)
            
            # Check circuit breaker states
            ollama_available = self.circuit_breaker.can_execute_request(ModelType.OLLAMA)
            torchserve_available = self.circuit_breaker.can_execute_request(ModelType.TORCHSERVE)
            
            # Calculate scores for each model
            ollama_score = self._calculate_model_score(
                ModelType.OLLAMA, ollama_load, ollama_perf, ollama_available, context
            )
            
            torchserve_score = self._calculate_model_score(
                ModelType.TORCHSERVE, torchserve_load, torchserve_perf, torchserve_available, context
            )
            
            # Select best model
            if ollama_score > torchserve_score:
                selected_model = ModelType.OLLAMA
                fallback_model = ModelType.TORCHSERVE if torchserve_available else None
                confidence = ollama_score
                reason = self._generate_selection_reason(ModelType.OLLAMA, ollama_score, torchserve_score)
            else:
                selected_model = ModelType.TORCHSERVE
                fallback_model = ModelType.OLLAMA if ollama_available else None
                confidence = torchserve_score
                reason = self._generate_selection_reason(ModelType.TORCHSERVE, torchserve_score, ollama_score)
            
            # Estimate expected performance
            if selected_model == ModelType.OLLAMA and ollama_perf:
                expected_latency = ollama_perf.avg_latency_ms
                expected_accuracy = ollama_perf.success_rate
            elif selected_model == ModelType.TORCHSERVE and torchserve_perf:
                expected_latency = torchserve_perf.avg_latency_ms
                expected_accuracy = torchserve_perf.success_rate
            else:
                # Default estimates
                expected_latency = 1000.0 if selected_model == ModelType.OLLAMA else 500.0
                expected_accuracy = 0.9
            
            selection_time = time.time() - start_time
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["average_selection_time"] = (
                (self.stats["average_selection_time"] * (self.stats["total_requests"] - 1) + selection_time) /
                self.stats["total_requests"]
            )
            
            return ModelSelection(
                selected_model=selected_model,
                confidence=confidence,
                reason=reason,
                expected_latency=expected_latency,
                expected_accuracy=expected_accuracy,
                fallback_model=fallback_model,
                selection_metadata={
                    "ollama_score": ollama_score,
                    "torchserve_score": torchserve_score,
                    "ollama_load": ollama_load.load_level.value,
                    "torchserve_load": torchserve_load.load_level.value,
                    "selection_time_ms": selection_time * 1000,
                    "circuit_states": {
                        "ollama": self.circuit_breaker.get_circuit_state(ModelType.OLLAMA).value,
                        "torchserve": self.circuit_breaker.get_circuit_state(ModelType.TORCHSERVE).value
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            
            # Fallback to default selection
            return ModelSelection(
                selected_model=ModelType.OLLAMA,
                confidence=0.5,
                reason=f"Selection failed: {e}",
                expected_latency=1000.0,
                expected_accuracy=0.8,
                fallback_model=ModelType.TORCHSERVE,
                selection_metadata={"error": str(e)}
            )
    
    def _calculate_model_score(self,
                             model_type: ModelType,
                             load_metrics: LoadMetrics,
                             performance: Optional[ModelPerformance],
                             available: bool,
                             context: RequestContext) -> float:
        """Berechnet Score für Model basierend auf verschiedenen Faktoren"""
        if not available:
            return 0.0
        
        score_components = []
        
        # Load score (lower load = higher score)
        load_score = 1.0 - load_metrics.capacity_utilization
        score_components.append(("load", load_score, self.config.load_weight))
        
        # Performance score
        if performance:
            # Latency score (lower latency = higher score)
            if context.max_latency_ms > 0:
                latency_score = max(0, 1.0 - (performance.avg_latency_ms / context.max_latency_ms))
            else:
                latency_score = max(0, 1.0 - (performance.avg_latency_ms / 5000))  # Default 5s max
            
            # Accuracy score
            accuracy_score = performance.success_rate
            
            # Reliability score (based on error rate)
            reliability_score = 1.0 - performance.error_rate
            
            score_components.extend([
                ("latency", latency_score, self.config.latency_weight),
                ("accuracy", accuracy_score, self.config.accuracy_weight),
                ("reliability", reliability_score, self.config.reliability_weight)
            ])
        else:
            # Default scores if no performance data
            score_components.extend([
                ("latency", 0.7, self.config.latency_weight),
                ("accuracy", 0.8, self.config.accuracy_weight),
                ("reliability", 0.8, self.config.reliability_weight)
            ])
        
        # Request type preference
        type_preference = self._get_type_preference(model_type, context.request_type)
        score_components.append(("type_preference", type_preference, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        total_weight = sum(weight for _, _, weight in score_components)
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5
        
        return max(0.0, min(1.0, final_score))
    
    def _get_type_preference(self, model_type: ModelType, request_type: RequestType) -> float:
        """Gibt Type-Präferenz für Model zurück"""
        preferences = {
            (ModelType.OLLAMA, RequestType.VISION_HEAVY): 0.9,
            (ModelType.OLLAMA, RequestType.BALANCED): 0.7,
            (ModelType.OLLAMA, RequestType.REAL_TIME): 0.6,
            (ModelType.TORCHSERVE, RequestType.FEATURE_HEAVY): 0.9,
            (ModelType.TORCHSERVE, RequestType.BATCH): 0.8,
            (ModelType.TORCHSERVE, RequestType.REAL_TIME): 0.8,
        }
        
        return preferences.get((model_type, request_type), 0.7)
    
    def _generate_selection_reason(self, 
                                 selected_model: ModelType,
                                 selected_score: float,
                                 other_score: float) -> str:
        """Generiert Begründung für Model-Selection"""
        score_diff = selected_score - other_score
        
        if score_diff > 0.3:
            confidence_level = "high"
        elif score_diff > 0.1:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return f"{selected_model.value} selected with {confidence_level} confidence (score: {selected_score:.3f} vs {other_score:.3f})"
    
    async def execute_with_failover(self, 
                                  request: InferenceRequest) -> InferenceResponse:
        """Führt Inference mit automatischem Failover aus"""
        # Select primary model
        selection = await self.select_optimal_model(request)
        primary_model = selection.selected_model
        
        # Try primary model
        try:
            response = await self._execute_inference(request, primary_model)
            
            if response.success:
                # Success - update tracking
                self.circuit_breaker.record_success(primary_model)
                self.performance_tracker.complete_request_tracking(
                    request.request_id, True
                )
                self.stats["successful_requests"] += 1
                return response
            else:
                raise Exception(response.error_message or "Primary model failed")
                
        except Exception as primary_error:
            self.logger.warning(f"Primary model {primary_model.value} failed: {primary_error}")
            
            # Record failure
            self.circuit_breaker.record_failure(primary_model)
            self.performance_tracker.complete_request_tracking(
                request.request_id, False, str(primary_error)
            )
            
            # Try failover if enabled and fallback available
            if self.config.enable_failover and selection.fallback_model:
                try:
                    self.stats["failover_count"] += 1
                    
                    response = await self._execute_inference(request, selection.fallback_model)
                    
                    if response.success:
                        self.circuit_breaker.record_success(selection.fallback_model)
                        self.stats["successful_requests"] += 1
                        response.metadata["failover"] = True
                        return response
                    else:
                        raise Exception(response.error_message or "Fallback model failed")
                        
                except Exception as secondary_error:
                    self.logger.error(f"Fallback model {selection.fallback_model.value} failed: {secondary_error}")
                    self.circuit_breaker.record_failure(selection.fallback_model)
                    
                    # Both models failed - use fallback strategy
                    self.stats["failed_requests"] += 1
                    return await self.fallback_strategies.execute_fallback(
                        request, primary_error, secondary_error
                    )
            else:
                # No failover - return error
                self.stats["failed_requests"] += 1
                return InferenceResponse(
                    request_id=request.request_id,
                    model_used=primary_model,
                    result={"error": "Primary model failed, no failover available"},
                    latency_ms=0.0,
                    success=False,
                    error_message=str(primary_error),
                    confidence_score=0.0,
                    timestamp=time.time(),
                    metadata={"no_failover": True}
                )
    
    async def _execute_inference(self, 
                               request: InferenceRequest,
                               model_type: ModelType) -> InferenceResponse:
        """Führt Inference auf spezifischem Model aus"""
        start_time = time.time()
        
        # Update counters
        self.load_monitor.increment_request_counter(model_type)
        self.performance_tracker.start_request_tracking(request.request_id, model_type)
        
        if model_type == ModelType.OLLAMA:
            self.stats["ollama_requests"] += 1
        else:
            self.stats["torchserve_requests"] += 1
        
        try:
            # Execute based on model type
            if model_type == ModelType.OLLAMA:
                result = await self._execute_ollama_inference(request)
            else:
                result = await self._execute_torchserve_inference(request)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                model_used=model_type,
                result=result,
                latency_ms=latency_ms,
                success=True,
                error_message=None,
                confidence_score=result.get("confidence", 0.8),
                timestamp=time.time(),
                metadata={"execution_time": latency_ms}
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                model_used=model_type,
                result={"error": str(e)},
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
                confidence_score=0.0,
                timestamp=time.time(),
                metadata={"execution_time": latency_ms}
            )
            
        finally:
            # Update counters
            self.load_monitor.decrement_request_counter(model_type)
    
    async def _execute_ollama_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Führt Ollama-Inference aus"""
        try:
            # Extract payload for Ollama
            chart_path = request.payload.get("chart_path")
            context = request.payload.get("context", {})
            
            if chart_path:
                # Vision analysis
                result = await self.ollama_client.analyze_chart_pattern(chart_path, context)
            else:
                # Text analysis
                prompt = request.payload.get("prompt", "Analyze the market data")
                result = await self.ollama_client.generate_response(prompt)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ollama inference failed: {e}")
            raise
    
    async def _execute_torchserve_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Führt TorchServe-Inference aus"""
        try:
            # Extract payload for TorchServe
            features = request.payload.get("features", {})
            model_type = request.payload.get("model_type", "pattern_recognition")
            
            # Execute TorchServe inference
            result = await asyncio.to_thread(
                self.torchserve_handler.process_features,
                features,
                model_type
            )
            
            # Convert result to dict if needed
            if hasattr(result, '__dict__'):
                return result.__dict__
            elif isinstance(result, dict):
                return result
            else:
                return {"result": str(result), "confidence": 0.8}
            
        except Exception as e:
            self.logger.error(f"TorchServe inference failed: {e}")
            raise
    
    def get_switcher_stats(self) -> Dict[str, Any]:
        """Gibt Switcher-Statistiken zurück"""
        # Get load metrics
        ollama_load = self.load_monitor.current_metrics.get(ModelType.OLLAMA)
        torchserve_load = self.load_monitor.current_metrics.get(ModelType.TORCHSERVE)
        
        # Get performance metrics
        ollama_perf = self.performance_tracker.current_performance.get(ModelType.OLLAMA)
        torchserve_perf = self.performance_tracker.current_performance.get(ModelType.TORCHSERVE)
        
        # Get circuit states
        circuit_states = {
            "ollama": self.circuit_breaker.get_circuit_state(ModelType.OLLAMA).value,
            "torchserve": self.circuit_breaker.get_circuit_state(ModelType.TORCHSERVE).value
        }
        
        return {
            **self.stats,
            "load_metrics": {
                "ollama": asdict(ollama_load) if ollama_load else None,
                "torchserve": asdict(torchserve_load) if torchserve_load else None
            },
            "performance_metrics": {
                "ollama": asdict(ollama_perf) if ollama_perf else None,
                "torchserve": asdict(torchserve_perf) if torchserve_perf else None
            },
            "circuit_states": circuit_states,
            "monitoring_active": self.load_monitor.monitoring_active
        }
    
    def shutdown(self):
        """Beendet Switcher und alle Monitoring-Threads"""
        try:
            self.load_monitor.stop_monitoring()
            self.logger.info("Real-time Switcher shutdown complete")
        except Exception as e:
            self.logger.error(f"Switcher shutdown error: {e}")


# Factory Function
def create_real_time_switcher(
    ollama_client: Optional[OllamaVisionClient] = None,
    torchserve_handler: Optional[TorchServeHandler] = None,
    config: Optional[SwitcherConfig] = None
) -> RealTimeSwitcher:
    """Factory function für Real-time Switcher"""
    return RealTimeSwitcher(ollama_client, torchserve_handler, config)


# Testing Function
async def test_real_time_switcher():
    """Test function für Real-time Switcher"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test request
    request = InferenceRequest(
        request_id="test_001",
        request_type=RequestType.VISION_HEAVY,
        context=RequestContext(
            request_id="test_001",
            request_type=RequestType.VISION_HEAVY,
            priority=5,
            max_latency_ms=2000,
            min_accuracy=0.8,
            timeout_seconds=10,
            retry_count=0,
            metadata={}
        ),
        payload={
            "chart_path": "test_chart.png",
            "context": {"timeframe": "1h", "symbol": "EUR/USD"}
        },
        timestamp=time.time()
    )
    
    # Test switcher
    switcher = create_real_time_switcher()
    
    # Test model selection
    selection = await switcher.select_optimal_model(request)
    print(f"Selected Model: {selection.selected_model.value}")
    print(f"Confidence: {selection.confidence:.3f}")
    print(f"Reason: {selection.reason}")
    
    # Test stats
    stats = switcher.get_switcher_stats()
    print(f"Switcher Stats: {stats}")
    
    # Cleanup
    switcher.shutdown()
    
    print("Real-time Switcher test completed")


if __name__ == "__main__":
    asyncio.run(test_real_time_switcher())