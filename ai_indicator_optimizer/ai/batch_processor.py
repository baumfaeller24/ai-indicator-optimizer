#!/usr/bin/env python3
"""
Batch-Processing-Support für Enhanced Pine Script Code Generator
Phase 3 Implementation - Task 9

Features:
- Parallel-Processing für mehrere Feature-Dictionaries
- GPU-Batch-Optimierung für AI-Models
- Queue-Management für große Datenmengen
- Performance-Monitoring und Load-Balancing
- Integration mit TorchServe Handler
- Enhanced Feature Extractor Integration
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import threading
from threading import Lock, Event
import queue
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json

# GPU/CUDA Support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Nautilus Imports
from nautilus_trader.model.data import Bar

# Local Imports
from .enhanced_feature_extractor import EnhancedFeatureExtractor
from .torchserve_handler import TorchServeHandler


@dataclass
class BatchJob:
    """Einzelner Batch-Job für Processing"""
    job_id: str
    job_type: str  # 'feature_extraction', 'pattern_analysis', 'pine_generation'
    data: Any
    priority: int = 1  # 1=niedrig, 5=hoch
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Ergebnis eines Batch-Jobs"""
    job_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    processed_at: datetime = field(default_factory=datetime.now)
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Haupt-Batch-Processor für Enhanced Pine Script Code Generator
    
    Features:
    - Multi-Threading/Multi-Processing
    - GPU-Batch-Optimierung
    - Priority-Queue-Management
    - Load-Balancing
    - Performance-Monitoring
    - Integration mit Enhanced Feature Extractor
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Processing-Konfiguration
        self.max_workers = self.config.get("max_workers", mp.cpu_count())
        self.max_batch_size = self.config.get("max_batch_size", 32)
        self.batch_timeout = self.config.get("batch_timeout", 5.0)  # Sekunden
        self.use_gpu = self.config.get("use_gpu", TORCH_AVAILABLE and torch.cuda.is_available())
        self.processing_mode = self.config.get("processing_mode", "thread")  # 'thread', 'process', 'async'
        
        # Queues und State
        self.job_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchResult] = {}
        
        # Threading
        self.workers: List[threading.Thread] = []
        self.executor: Optional[concurrent.futures.Executor] = None
        self.running = False
        self.shutdown_event = Event()
        self.queue_lock = Lock()
        
        # Components
        self.feature_extractor = EnhancedFeatureExtractor()
        self.torchserve_handler = None  # Wird bei Bedarf initialisiert
        
        # Performance-Tracking
        self.stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "gpu_utilization": 0.0,
            "queue_size_history": [],
            "throughput_history": []
        }
        
        # GPU-Setup
        if self.use_gpu:
            self._setup_gpu()
        
        self.logger.info(f"BatchProcessor initialized: workers={self.max_workers}, gpu={self.use_gpu}, mode={self.processing_mode}")
    
    def _setup_gpu(self):
        """Setup GPU für Batch-Processing"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, disabling GPU")
            self.use_gpu = False
            return
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, disabling GPU")
            self.use_gpu = False
            return
        
        # GPU-Info
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
        
        self.logger.info(f"GPU Setup: {gpu_name} ({gpu_memory:.1f}GB), {gpu_count} GPUs available")
        
        # GPU-Memory-Management
        torch.cuda.empty_cache()
    
    def start(self):
        """Starte Batch-Processor"""
        if self.running:
            self.logger.warning("BatchProcessor already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Executor basierend auf Processing-Mode
        if self.processing_mode == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.processing_mode == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Worker-Threads starten
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Monitoring-Thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="BatchMonitor",
            daemon=True
        )
        monitor_thread.start()
        
        self.logger.info(f"BatchProcessor started with {len(self.workers)} workers")
    
    def stop(self, timeout: float = 10.0):
        """Stoppe Batch-Processor"""
        if not self.running:
            return
        
        self.logger.info("Stopping BatchProcessor...")
        
        # Shutdown-Signal
        self.running = False
        self.shutdown_event.set()
        
        # Warte auf Worker-Threads
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        # Executor shutdown
        if self.executor:
            self.executor.shutdown(wait=True, timeout=timeout)
        
        self.workers.clear()
        self.logger.info("BatchProcessor stopped")
    
    def submit_job(self, job: BatchJob) -> str:
        """
        Füge Job zur Processing-Queue hinzu
        
        Args:
            job: BatchJob zum Processing
            
        Returns:
            Job-ID für Tracking
        """
        try:
            with self.queue_lock:
                # Priority-Queue: (priority, timestamp, job)
                # Niedrigere Zahlen = höhere Priorität
                priority_value = -job.priority  # Invertiere für korrekte Sortierung
                timestamp = time.time()
                self.job_queue.put((priority_value, timestamp, job))
                self.active_jobs[job.job_id] = job
            
            self.logger.debug(f"Job submitted: {job.job_id} (type: {job.job_type}, priority: {job.priority})")
            return job.job_id
            
        except Exception as e:
            self.logger.error(f"Error submitting job {job.job_id}: {e}")
            raise
    
    def submit_batch_jobs(self, jobs: List[BatchJob]) -> List[str]:
        """Füge mehrere Jobs gleichzeitig hinzu"""
        job_ids = []
        for job in jobs:
            job_id = self.submit_job(job)
            job_ids.append(job_id)
        
        self.logger.info(f"Submitted batch of {len(jobs)} jobs")
        return job_ids
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[BatchResult]:
        """
        Erhalte Ergebnis für Job-ID
        
        Args:
            job_id: Job-ID
            timeout: Timeout in Sekunden
            
        Returns:
            BatchResult oder None
        """
        start_time = time.time()
        
        while True:
            # Prüfe completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            
            # Timeout-Check
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout waiting for job {job_id}")
                return None
            
            # Kurz warten
            time.sleep(0.1)
    
    def get_batch_results(self, job_ids: List[str], timeout: Optional[float] = None) -> List[Optional[BatchResult]]:
        """Erhalte Ergebnisse für mehrere Job-IDs"""
        results = []
        for job_id in job_ids:
            result = self.get_result(job_id, timeout)
            results.append(result)
        return results
    
    def _worker_loop(self):
        """Haupt-Worker-Loop"""
        worker_id = threading.current_thread().name
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Hole Job aus Queue (mit Timeout)
                try:
                    priority, timestamp, job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Verarbeite Job
                result = self._process_job(job, worker_id)
                
                # Speichere Ergebnis
                with self.queue_lock:
                    self.completed_jobs[job.job_id] = result
                    if job.job_id in self.active_jobs:
                        del self.active_jobs[job.job_id]
                
                # Callback ausführen
                if job.callback:
                    try:
                        job.callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback error for job {job.job_id}: {e}")
                
                # Queue-Task als erledigt markieren
                self.job_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def _process_job(self, job: BatchJob, worker_id: str) -> BatchResult:
        """Verarbeite einzelnen Job"""
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing job {job.job_id} (type: {job.job_type}) on worker {worker_id}")
            
            # Job-Type-spezifische Verarbeitung
            if job.job_type == "feature_extraction":
                result_data = self._process_feature_extraction(job.data)
            elif job.job_type == "pattern_analysis":
                result_data = self._process_pattern_analysis(job.data)
            elif job.job_type == "pine_generation":
                result_data = self._process_pine_generation(job.data)
            elif job.job_type == "batch_inference":
                result_data = self._process_batch_inference(job.data)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            processing_time = time.time() - start_time
            
            # Update Stats
            self.stats["jobs_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            return BatchResult(
                job_id=job.job_id,
                success=True,
                result=result_data,
                processing_time=processing_time,
                worker_id=worker_id,
                metadata=job.metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            # Update Stats
            self.stats["jobs_failed"] += 1
            
            return BatchResult(
                job_id=job.job_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                worker_id=worker_id,
                metadata=job.metadata
            )
    
    def _process_feature_extraction(self, data: Any) -> Dict[str, Any]:
        """Verarbeite Feature-Extraktion mit Enhanced Feature Extractor"""
        
        try:
            if isinstance(data, dict) and "bars" in data:
                bars = data["bars"]
                
                # Enhanced Feature Extraction
                if isinstance(bars, list) and len(bars) > 0:
                    # Konvertiere zu Bar-Objekten falls nötig
                    if isinstance(bars[0], dict):
                        # Mock Bar-Objekte für Demo
                        processed_bars = bars
                    else:
                        processed_bars = bars
                    
                    # Feature-Extraktion für jeden Bar
                    features_list = []
                    for bar_data in processed_bars:
                        if isinstance(bar_data, dict):
                            # Verwende Enhanced Feature Extractor
                            features = self.feature_extractor.extract_features_from_dict(bar_data)
                            features_list.append(features)
                    
                    return {
                        "features_extracted": len(features_list),
                        "features_list": features_list,
                        "feature_dimensions": len(features_list[0]) if features_list else 0,
                        "extraction_timestamp": datetime.now().isoformat(),
                        "bars_processed": len(bars)
                    }
                
                return {"error": "No valid bars provided"}
            
            elif isinstance(data, dict) and "market_data" in data:
                # Einzelne Market-Data-Verarbeitung
                market_data = data["market_data"]
                features = self.feature_extractor.extract_features_from_dict(market_data)
                
                return {
                    "features": features.tolist() if hasattr(features, 'tolist') else features,
                    "feature_count": len(features) if hasattr(features, '__len__') else 0,
                    "extraction_timestamp": datetime.now().isoformat()
                }
            
            return {"error": "Invalid data format for feature extraction"}
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return {"error": str(e)}
    
    def _process_pattern_analysis(self, data: Any) -> Dict[str, Any]:
        """Verarbeite Pattern-Analyse mit AI-Models"""
        
        try:
            if isinstance(data, dict) and "features" in data:
                features = data["features"]
                
                # Mock Pattern Analysis mit AI-Model
                if isinstance(features, list):
                    patterns_detected = []
                    
                    for i, feature_set in enumerate(features):
                        # Simuliere Pattern-Erkennung
                        confidence = np.random.uniform(0.6, 0.95)
                        pattern_type = np.random.choice([
                            "doji", "hammer", "shooting_star", "engulfing_bull", 
                            "engulfing_bear", "morning_star", "evening_star"
                        ])
                        
                        patterns_detected.append({
                            "position": i,
                            "pattern_type": pattern_type,
                            "confidence": float(confidence),
                            "strength": float(confidence * np.random.uniform(0.8, 1.2))
                        })
                    
                    return {
                        "patterns_detected": len(patterns_detected),
                        "patterns": patterns_detected,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "features_analyzed": len(features)
                    }
                
                return {"error": "Features must be a list"}
            
            return {"error": "Invalid data format for pattern analysis"}
            
        except Exception as e:
            self.logger.error(f"Pattern analysis error: {e}")
            return {"error": str(e)}
    
    def _process_pine_generation(self, data: Any) -> Dict[str, Any]:
        """Verarbeite Pine Script Generation"""
        
        try:
            if isinstance(data, dict) and "strategy_config" in data:
                strategy_config = data["strategy_config"]
                
                # Mock Pine Script Generation
                indicator_type = strategy_config.get("indicator_type", "rsi")
                parameters = strategy_config.get("parameters", {})
                
                # Generiere Pine Script basierend auf Konfiguration
                pine_script = self._generate_pine_script(indicator_type, parameters)
                
                return {
                    "pine_script": pine_script,
                    "indicator_type": indicator_type,
                    "parameters_used": parameters,
                    "generation_timestamp": datetime.now().isoformat(),
                    "script_length": len(pine_script)
                }
            
            return {"error": "Invalid data format for pine generation"}
            
        except Exception as e:
            self.logger.error(f"Pine generation error: {e}")
            return {"error": str(e)}
    
    def _generate_pine_script(self, indicator_type: str, parameters: Dict[str, Any]) -> str:
        """Generiere Pine Script Code"""
        
        if indicator_type == "rsi":
            rsi_period = parameters.get("rsi_period", 14)
            overbought = parameters.get("overbought", 70)
            oversold = parameters.get("oversold", 30)
            
            return f"""
//@version=5
indicator("AI RSI Strategy", shorttitle="AI-RSI", overlay=false)

// Parameters
rsi_period = input.int({rsi_period}, title="RSI Period", minval=1, maxval=200)
overbought = input.float({overbought}, title="Overbought Level", minval=50, maxval=100)
oversold = input.float({oversold}, title="Oversold Level", minval=0, maxval=50)

// Calculations
rsi_value = ta.rsi(close, rsi_period)

// Signals
buy_signal = ta.crossover(rsi_value, oversold)
sell_signal = ta.crossunder(rsi_value, overbought)

// Plots
plot(rsi_value, title="RSI", color=color.blue, linewidth=2)
hline(overbought, title="Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, title="Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, title="Midline", color=color.gray, linestyle=hline.style_dotted)

// Signal Shapes
plotshape(buy_signal, style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)
plotshape(sell_signal, style=shape.triangledown, location=location.top, color=color.red, size=size.small)

// Alerts
alertcondition(buy_signal, title="RSI Buy Signal", message="RSI crossed above oversold level")
alertcondition(sell_signal, title="RSI Sell Signal", message="RSI crossed below overbought level")
"""
        
        elif indicator_type == "macd":
            fast_length = parameters.get("fast_length", 12)
            slow_length = parameters.get("slow_length", 26)
            signal_length = parameters.get("signal_length", 9)
            
            return f"""
//@version=5
indicator("AI MACD Strategy", shorttitle="AI-MACD", overlay=false)

// Parameters
fast_length = input.int({fast_length}, title="Fast Length", minval=1)
slow_length = input.int({slow_length}, title="Slow Length", minval=1)
signal_length = input.int({signal_length}, title="Signal Length", minval=1)

// Calculations
[macd_line, signal_line, histogram] = ta.macd(close, fast_length, slow_length, signal_length)

// Signals
buy_signal = ta.crossover(macd_line, signal_line)
sell_signal = ta.crossunder(macd_line, signal_line)

// Plots
plot(macd_line, title="MACD", color=color.blue, linewidth=2)
plot(signal_line, title="Signal", color=color.red, linewidth=1)
plot(histogram, title="Histogram", color=color.gray, style=plot.style_histogram)
hline(0, title="Zero Line", color=color.black, linestyle=hline.style_solid)

// Signal Shapes
plotshape(buy_signal, style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)
plotshape(sell_signal, style=shape.triangledown, location=location.top, color=color.red, size=size.small)

// Alerts
alertcondition(buy_signal, title="MACD Buy Signal", message="MACD crossed above signal line")
alertcondition(sell_signal, title="MACD Sell Signal", message="MACD crossed below signal line")
"""
        
        else:
            # Default Simple Moving Average
            ma_period = parameters.get("ma_period", 20)
            
            return f"""
//@version=5
indicator("AI MA Strategy", shorttitle="AI-MA", overlay=true)

// Parameters
ma_period = input.int({ma_period}, title="MA Period", minval=1, maxval=500)

// Calculations
ma_value = ta.sma(close, ma_period)

// Signals
buy_signal = ta.crossover(close, ma_value)
sell_signal = ta.crossunder(close, ma_value)

// Plots
plot(ma_value, title="Moving Average", color=color.blue, linewidth=2)

// Signal Shapes
plotshape(buy_signal, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small)
plotshape(sell_signal, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small)

// Alerts
alertcondition(buy_signal, title="MA Buy Signal", message="Price crossed above moving average")
alertcondition(sell_signal, title="MA Sell Signal", message="Price crossed below moving average")
"""
    
    def _process_batch_inference(self, data: Any) -> Dict[str, Any]:
        """Verarbeite Batch-Inference mit GPU-Optimierung"""
        
        try:
            if not isinstance(data, dict) or "features_batch" not in data:
                return {"error": "Invalid data format for batch inference"}
            
            features_batch = data["features_batch"]
            
            # GPU-Batch-Processing falls verfügbar
            if self.use_gpu and TORCH_AVAILABLE:
                return self._gpu_batch_inference(features_batch)
            else:
                return self._cpu_batch_inference(features_batch)
                
        except Exception as e:
            self.logger.error(f"Batch inference error: {e}")
            return {"error": str(e)}
    
    def _gpu_batch_inference(self, features_batch: List[Dict]) -> Dict[str, Any]:
        """GPU-optimierte Batch-Inference"""
        
        try:
            # Konvertiere zu Tensor
            feature_arrays = []
            for features in features_batch:
                if isinstance(features, dict):
                    # Verwende Enhanced Feature Extractor Format
                    feature_array = self.feature_extractor.extract_features_from_dict(features)
                    feature_arrays.append(feature_array)
                elif isinstance(features, (list, np.ndarray)):
                    feature_arrays.append(np.array(features, dtype=np.float32))
            
            batch_tensor = torch.tensor(
                np.stack(feature_arrays),
                dtype=torch.float32
            ).cuda()
            
            # Mock AI-Model Inference
            with torch.no_grad():
                # Simuliere Neural Network
                predictions = torch.sigmoid(torch.sum(batch_tensor, dim=1) / batch_tensor.shape[1])
                confidences = torch.rand(len(features_batch)).cuda() * 0.4 + 0.6  # 0.6-1.0
                
                # Pattern-Klassifikation
                pattern_logits = torch.randn(len(features_batch), 7).cuda()  # 7 Pattern-Typen
                pattern_probs = torch.softmax(pattern_logits, dim=1)
            
            # Konvertiere zurück zu CPU
            predictions_cpu = predictions.cpu().numpy()
            confidences_cpu = confidences.cpu().numpy()
            pattern_probs_cpu = pattern_probs.cpu().numpy()
            
            results = []
            pattern_names = ["doji", "hammer", "shooting_star", "engulfing_bull", 
                           "engulfing_bear", "morning_star", "evening_star"]
            
            for i, (pred, conf, pattern_prob) in enumerate(zip(predictions_cpu, confidences_cpu, pattern_probs_cpu)):
                top_pattern_idx = np.argmax(pattern_prob)
                
                results.append({
                    "prediction": float(pred),
                    "confidence": float(conf),
                    "direction": "UP" if pred > 0.5 else "DOWN",
                    "top_pattern": pattern_names[top_pattern_idx],
                    "pattern_confidence": float(pattern_prob[top_pattern_idx]),
                    "all_patterns": {
                        pattern_names[j]: float(pattern_prob[j]) 
                        for j in range(len(pattern_names))
                    }
                })
            
            return {
                "batch_size": len(features_batch),
                "predictions": results,
                "processing_device": "GPU",
                "inference_timestamp": datetime.now().isoformat(),
                "gpu_memory_used": torch.cuda.memory_allocated() / 1e9  # GB
            }
            
        except Exception as e:
            self.logger.error(f"GPU batch inference error: {e}")
            # Fallback zu CPU
            return self._cpu_batch_inference(features_batch)
    
    def _cpu_batch_inference(self, features_batch: List[Dict]) -> Dict[str, Any]:
        """CPU-basierte Batch-Inference"""
        
        results = []
        
        for features in features_batch:
            try:
                # Feature-Extraktion
                if isinstance(features, dict):
                    feature_array = self.feature_extractor.extract_features_from_dict(features)
                    feature_sum = np.sum(feature_array)
                else:
                    feature_sum = sum(features.values()) if isinstance(features, dict) else 0
                
                # Mock CPU-basierte Inference
                prediction = 1.0 / (1.0 + np.exp(-feature_sum / 1000))  # Sigmoid
                confidence = np.random.uniform(0.6, 0.95)
                
                # Mock Pattern-Erkennung
                pattern_names = ["doji", "hammer", "shooting_star", "engulfing_bull", 
                               "engulfing_bear", "morning_star", "evening_star"]
                top_pattern = np.random.choice(pattern_names)
                pattern_confidence = np.random.uniform(0.5, 0.9)
                
                results.append({
                    "prediction": float(prediction),
                    "confidence": float(confidence),
                    "direction": "UP" if prediction > 0.5 else "DOWN",
                    "top_pattern": top_pattern,
                    "pattern_confidence": float(pattern_confidence)
                })
                
            except Exception as e:
                self.logger.error(f"CPU inference error for single item: {e}")
                results.append({
                    "prediction": 0.5,
                    "confidence": 0.0,
                    "direction": "UNKNOWN",
                    "error": str(e)
                })
        
        return {
            "batch_size": len(features_batch),
            "predictions": results,
            "processing_device": "CPU",
            "inference_timestamp": datetime.now().isoformat()
        }
    
    def _monitoring_loop(self):
        """Monitoring-Loop für Performance-Tracking"""
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Queue-Size tracking
                queue_size = self.job_queue.qsize()
                self.stats["queue_size_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "queue_size": queue_size,
                    "active_jobs": len(self.active_jobs)
                })
                
                # Throughput berechnen
                if self.stats["jobs_processed"] > 0:
                    avg_time = self.stats["total_processing_time"] / self.stats["jobs_processed"]
                    throughput = 1.0 / avg_time if avg_time > 0 else 0
                    
                    self.stats["throughput_history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "throughput_jobs_per_sec": throughput,
                        "avg_processing_time": avg_time
                    })
                
                # GPU-Utilization (falls verfügbar)
                if self.use_gpu and TORCH_AVAILABLE:
                    try:
                        gpu_util = torch.cuda.utilization()
                        self.stats["gpu_utilization"] = gpu_util
                    except:
                        pass
                
                # Cleanup alte History-Einträge (behalte nur letzte 100)
                for key in ["queue_size_history", "throughput_history"]:
                    if len(self.stats[key]) > 100:
                        self.stats[key] = self.stats[key][-100:]
                
                # Warte 5 Sekunden
                self.shutdown_event.wait(5.0)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte aktuelle Statistiken"""
        
        current_stats = self.stats.copy()
        
        # Zusätzliche Metriken
        current_stats.update({
            "running": self.running,
            "worker_count": len(self.workers),
            "queue_size": self.job_queue.qsize(),
            "active_jobs_count": len(self.active_jobs),
            "completed_jobs_count": len(self.completed_jobs),
            "use_gpu": self.use_gpu,
            "processing_mode": self.processing_mode,
            "max_workers": self.max_workers,
            "max_batch_size": self.max_batch_size
        })
        
        return current_stats
    
    def clear_completed_jobs(self, older_than_hours: float = 24.0):
        """Räume alte completed jobs auf"""
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        jobs_to_remove = [
            job_id for job_id, result in self.completed_jobs.items()
            if result.processed_at < cutoff_time
        ]
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
        
        self.logger.info(f"Cleaned up {len(jobs_to_remove)} old completed jobs")


# Factory Function
def create_batch_processor(config: Optional[Dict] = None) -> BatchProcessor:
    """
    Factory Function für Batch Processor
    
    Args:
        config: Processor-Konfiguration
        
    Returns:
        BatchProcessor Instance
    """
    return BatchProcessor(config=config)


# Demo/Test Function
def demo_batch_processor():
    """Demo für Batch Processor"""
    
    print("🧪 Testing Enhanced Batch Processor...")
    
    # Processor erstellen und starten
    processor = create_batch_processor({
        "max_workers": 4,
        "max_batch_size": 16,
        "use_gpu": True,
        "processing_mode": "thread"
    })
    
    processor.start()
    
    try:
        # Test-Jobs erstellen
        jobs = []
        
        # Feature Extraction Jobs
        for i in range(3):
            job = BatchJob(
                job_id=f"feature_extraction_{i}",
                job_type="feature_extraction",
                data={
                    "bars": [
                        {
                            "open": 1.1000 + i * 0.0001,
                            "high": 1.1005 + i * 0.0001,
                            "low": 1.0995 + i * 0.0001,
                            "close": 1.1002 + i * 0.0001,
                            "volume": 1500 + i * 100,
                            "timestamp": datetime.now().isoformat()
                        }
                        for j in range(20)
                    ]
                },
                priority=2
            )
            jobs.append(job)
        
        # Pattern Analysis Job
        pattern_job = BatchJob(
            job_id="pattern_analysis_1",
            job_type="pattern_analysis",
            data={
                "features": [
                    [0.5, 0.3, 0.8, 0.2, 0.9] for _ in range(10)
                ]
            },
            priority=3
        )
        jobs.append(pattern_job)
        
        # Pine Generation Job
        pine_job = BatchJob(
            job_id="pine_generation_1",
            job_type="pine_generation",
            data={
                "strategy_config": {
                    "indicator_type": "rsi",
                    "parameters": {
                        "rsi_period": 14,
                        "overbought": 70,
                        "oversold": 30
                    }
                }
            },
            priority=5  # Höchste Priorität
        )
        jobs.append(pine_job)
        
        # Batch Inference Job
        features_batch = [
            {
                "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1002,
                "volume": 1500, "rsi_14": 65.5, "macd": 0.001, "hour": 14
            },
            {
                "open": 1.1002, "high": 1.1007, "low": 1.0997, "close": 1.1004,
                "volume": 1600, "rsi_14": 45.2, "macd": -0.002, "hour": 15
            }
        ]
        
        batch_job = BatchJob(
            job_id="batch_inference_1",
            job_type="batch_inference",
            data={"features_batch": features_batch},
            priority=4
        )
        jobs.append(batch_job)
        
        # Jobs submitten
        job_ids = processor.submit_batch_jobs(jobs)
        print(f"✅ Submitted {len(job_ids)} jobs")
        
        # Warte auf Ergebnisse
        print("⏳ Waiting for results...")
        results = processor.get_batch_results(job_ids, timeout=30.0)
        
        # Ergebnisse anzeigen
        print(f"\n📊 Results:")
        for i, result in enumerate(results):
            if result:
                print(f"   Job {i+1}: {'✅ Success' if result.success else '❌ Failed'} ({result.processing_time:.3f}s)")
                if result.success:
                    if "features_extracted" in str(result.result):
                        feat_count = result.result.get("features_extracted", 0)
                        print(f"     Features Extracted: {feat_count}")
                    elif "patterns_detected" in str(result.result):
                        pattern_count = result.result.get("patterns_detected", 0)
                        print(f"     Patterns Detected: {pattern_count}")
                    elif "pine_script" in str(result.result):
                        script_length = result.result.get("script_length", 0)
                        print(f"     Pine Script Length: {script_length} chars")
                    elif "predictions" in str(result.result):
                        pred_count = len(result.result.get("predictions", []))
                        print(f"     Predictions: {pred_count}")
            else:
                print(f"   Job {i+1}: ⏰ Timeout")
        
        # Statistiken
        stats = processor.get_statistics()
        print(f"\n📈 Statistics:")
        print(f"   Jobs Processed: {stats['jobs_processed']}")
        print(f"   Jobs Failed: {stats['jobs_failed']}")
        print(f"   GPU Enabled: {stats['use_gpu']}")
        print(f"   Workers: {stats['worker_count']}")
        print(f"   Queue Size: {stats['queue_size']}")
        
    finally:
        # Processor stoppen
        processor.stop()
        print("\n🛑 Batch Processor stopped")


if __name__ == "__main__":
    demo_batch_processor()