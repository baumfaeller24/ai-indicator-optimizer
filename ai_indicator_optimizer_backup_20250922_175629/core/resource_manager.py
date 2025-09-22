"""
Ressourcen-Manager für optimale Hardware-Auslastung
"""

import torch
import multiprocessing as mp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import threading
import time
from .hardware_detector import HardwareDetector


@dataclass
class ResourceAllocation:
    cpu_workers: Dict[str, int]
    gpu_devices: List[int]
    memory_limit: int
    storage_paths: List[str]


@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Dict[int, int]
    gpu_utilization: Dict[int, float]
    timestamp: float


class ResourceManager:
    """
    Verwaltet Hardware-Ressourcen für optimale Auslastung
    """
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware = hardware_detector
        self.allocation = self._create_allocation()
        self.monitoring = False
        self.usage_history = []
        self._lock = threading.Lock()
        
        # Executor Pools
        self.cpu_executors = {}
        self.gpu_context = None
        
        self._setup_executors()
        self._setup_gpu_context()
    
    def _create_allocation(self) -> ResourceAllocation:
        """Erstellt optimale Ressourcen-Allokation"""
        
        # CPU Worker Verteilung
        cpu_workers = self.hardware.get_optimal_worker_counts()
        
        # GPU Devices
        gpu_devices = list(range(len(self.hardware.gpu_info))) if self.hardware.gpu_info else []
        
        # Memory Limit (75% der verfügbaren RAM für AI-Tasks, Rest für System)
        memory_limit = int(self.hardware.memory_info.total * 0.75) if self.hardware.memory_info else 8 * 1024**3
        
        # Storage Paths (priorisiert nach Geschwindigkeit)
        storage_paths = []
        if self.hardware.storage_info:
            # Sortiere nach Geschwindigkeit (NVMe > SSD > HDD)
            sorted_storage = sorted(
                self.hardware.storage_info,
                key=lambda x: (
                    x.type == 'NVMe' and 2 or 
                    x.type == 'SSD' and 1 or 0,
                    x.read_speed or 0
                ),
                reverse=True
            )
            storage_paths = [s.device for s in sorted_storage]
        
        return ResourceAllocation(
            cpu_workers=cpu_workers,
            gpu_devices=gpu_devices,
            memory_limit=memory_limit,
            storage_paths=storage_paths
        )
    
    def _setup_executors(self):
        """Initialisiert CPU Executor Pools"""
        for task_name, worker_count in self.allocation.cpu_workers.items():
            self.cpu_executors[task_name] = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context('spawn')
            )
    
    def _setup_gpu_context(self):
        """Initialisiert GPU Context"""
        if torch.cuda.is_available() and self.allocation.gpu_devices:
            # Setze primäre GPU (RTX 5090 falls vorhanden)
            primary_gpu = 0
            for i, gpu in enumerate(self.hardware.gpu_info):
                if 'rtx 5090' in gpu.name.lower():
                    primary_gpu = i
                    break
            
            torch.cuda.set_device(primary_gpu)
            self.gpu_context = {
                'primary_device': primary_gpu,
                'available_devices': self.allocation.gpu_devices,
                'memory_fraction': 0.9  # 90% GPU Memory
            }
            
            # GPU Memory Management
            for device_id in self.allocation.gpu_devices:
                torch.cuda.set_per_process_memory_fraction(
                    self.gpu_context['memory_fraction'], 
                    device_id
                )
    
    def get_cpu_executor(self, task_type: str) -> ProcessPoolExecutor:
        """Gibt CPU Executor für spezifischen Task zurück"""
        return self.cpu_executors.get(task_type, self.cpu_executors['data_pipeline'])
    
    def get_gpu_device(self, preferred_device: Optional[int] = None) -> int:
        """Gibt optimale GPU Device ID zurück"""
        if not self.gpu_context:
            raise RuntimeError("No GPU available")
        
        if preferred_device is not None and preferred_device in self.allocation.gpu_devices:
            return preferred_device
        
        return self.gpu_context['primary_device']
    
    def allocate_memory(self, size_bytes: int) -> bool:
        """Prüft und reserviert Memory"""
        available = psutil.virtual_memory().available
        
        if size_bytes > available:
            return False
        
        if size_bytes > self.allocation.memory_limit:
            print(f"Warning: Requested memory ({size_bytes // 1024**3} GB) exceeds limit ({self.allocation.memory_limit // 1024**3} GB)")
            return False
        
        return True
    
    def get_optimal_batch_size(self, model_memory_mb: int, data_size_mb: int) -> int:
        """Berechnet optimale Batch Size basierend auf verfügbarer GPU Memory"""
        if not self.gpu_context:
            return 1
        
        gpu_id = self.gpu_context['primary_device']
        gpu_info = self.hardware.gpu_info[gpu_id]
        
        # Verfügbare GPU Memory (90% der Gesamt-Memory)
        available_memory_mb = (gpu_info.memory_total * 0.9) // (1024 * 1024)
        
        # Memory für Model
        model_memory_required = model_memory_mb
        
        # Verbleibende Memory für Batches
        remaining_memory = available_memory_mb - model_memory_required
        
        if remaining_memory <= 0:
            return 1
        
        # Batch Size basierend auf Data Size
        max_batch_size = int(remaining_memory // data_size_mb)
        
        # Optimale Batch Size (Power of 2, max 64)
        optimal_batch_size = 1
        while optimal_batch_size * 2 <= max_batch_size and optimal_batch_size < 64:
            optimal_batch_size *= 2
        
        return max(1, optimal_batch_size)
    
    def start_monitoring(self, interval: float = 1.0):
        """Startet Ressourcen-Monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                usage = self._collect_usage_stats()
                
                with self._lock:
                    self.usage_history.append(usage)
                    # Behalte nur letzte 1000 Einträge
                    if len(self.usage_history) > 1000:
                        self.usage_history.pop(0)
                
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stoppt Ressourcen-Monitoring"""
        self.monitoring = False
    
    def _collect_usage_stats(self) -> ResourceUsage:
        """Sammelt aktuelle Ressourcen-Nutzung"""
        # CPU Usage
        cpu_percent = psutil.cpu_percent()
        
        # Memory Usage
        memory_percent = psutil.virtual_memory().percent
        
        # GPU Usage
        gpu_memory_used = {}
        gpu_utilization = {}
        
        if torch.cuda.is_available():
            for device_id in self.allocation.gpu_devices:
                try:
                    # GPU Memory
                    memory_used = torch.cuda.memory_allocated(device_id)
                    gpu_memory_used[device_id] = memory_used
                    
                    # GPU Utilization (approximation)
                    gpu_utilization[device_id] = 0.0  # Würde nvidia-ml-py erfordern
                    
                except Exception:
                    gpu_memory_used[device_id] = 0
                    gpu_utilization[device_id] = 0.0
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_used=gpu_memory_used,
            gpu_utilization=gpu_utilization,
            timestamp=time.time()
        )
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Gibt aktuelle Ressourcen-Nutzung zurück"""
        with self._lock:
            return self.usage_history[-1] if self.usage_history else None
    
    def get_usage_history(self, last_n: int = 100) -> List[ResourceUsage]:
        """Gibt Ressourcen-Nutzungs-Historie zurück"""
        with self._lock:
            return self.usage_history[-last_n:]
    
    def optimize_for_task(self, task_type: str) -> Dict[str, Any]:
        """Optimiert Ressourcen für spezifischen Task"""
        optimizations = {
            'data_processing': {
                'cpu_workers': self.allocation.cpu_workers['data_pipeline'],
                'memory_limit': self.allocation.memory_limit // 6,  # ~21GB für Datenverarbeitung
                'io_threads': 4
            },
            'model_training': {
                'gpu_device': self.get_gpu_device() if self.gpu_context else None,
                'batch_size': self.get_optimal_batch_size(8000, 100) if self.gpu_context else 1,
                'memory_limit': int(self.allocation.memory_limit * 0.6),  # ~76GB für Training
                'mixed_precision': True if self.gpu_context else False
            },
            'inference': {
                'gpu_device': self.get_gpu_device() if self.gpu_context else None,
                'batch_size': self.get_optimal_batch_size(8000, 50) if self.gpu_context else 1,
                'memory_limit': self.allocation.memory_limit // 5,  # ~25GB für Inference
                'cpu_workers': 2
            },
            'chart_rendering': {
                'gpu_device': self.get_gpu_device() if self.gpu_context else None,
                'cpu_workers': self.allocation.cpu_workers['chart_render'],
                'memory_limit': self.allocation.memory_limit // 10  # ~13GB für Charts
            }
        }
        
        return optimizations.get(task_type, {})
    
    def cleanup(self):
        """Bereinigt Ressourcen"""
        self.stop_monitoring()
        
        # Schließe CPU Executors
        for executor in self.cpu_executors.values():
            executor.shutdown(wait=True)
        
        # GPU Memory Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def print_allocation_summary(self):
        """Gibt Ressourcen-Allokations-Zusammenfassung aus"""
        print("=== Resource Allocation Summary ===")
        
        print("CPU Workers:")
        for task, count in self.allocation.cpu_workers.items():
            print(f"  {task}: {count} workers")
        
        print(f"GPU Devices: {self.allocation.gpu_devices}")
        print(f"Memory Limit: {self.allocation.memory_limit // (1024**3)} GB")
        print(f"Storage Paths: {self.allocation.storage_paths}")
        
        if self.gpu_context:
            print(f"Primary GPU: {self.gpu_context['primary_device']}")
            print(f"GPU Memory Fraction: {self.gpu_context['memory_fraction']}")
        
        # Current Usage
        current_usage = self.get_current_usage()
        if current_usage:
            print(f"\nCurrent Usage:")
            print(f"  CPU: {current_usage.cpu_percent:.1f}%")
            print(f"  Memory: {current_usage.memory_percent:.1f}%")
            for gpu_id, memory in current_usage.gpu_memory_used.items():
                print(f"  GPU {gpu_id}: {memory // (1024**2)} MB")


class ResourceContext:
    """Context Manager für Ressourcen-Management"""
    
    def __init__(self, resource_manager: ResourceManager, task_type: str):
        self.resource_manager = resource_manager
        self.task_type = task_type
        self.optimizations = None
    
    def __enter__(self):
        self.optimizations = self.resource_manager.optimize_for_task(self.task_type)
        return self.optimizations
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup falls nötig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()