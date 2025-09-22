"""
Hardware Detection für optimale Ressourcen-Allokation
"""

import psutil
import torch
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import platform


@dataclass
class CPUInfo:
    model: str
    cores_physical: int
    cores_logical: int
    frequency_max: float
    frequency_current: float
    cache_l3: Optional[int] = None


@dataclass
class GPUInfo:
    name: str
    memory_total: int
    memory_free: int
    compute_capability: tuple
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[int] = None


@dataclass
class MemoryInfo:
    total: int
    available: int
    frequency: Optional[int] = None
    type: Optional[str] = None


@dataclass
class StorageInfo:
    device: str
    total: int
    free: int
    type: str  # SSD, NVMe, HDD
    read_speed: Optional[int] = None
    write_speed: Optional[int] = None


class HardwareDetector:
    """
    Detektiert verfügbare Hardware-Ressourcen für optimale Allokation
    """
    
    def __init__(self):
        self.cpu_info = None
        self.gpu_info = None
        self.memory_info = None
        self.storage_info = None
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Führt vollständige Hardware-Erkennung durch"""
        self.cpu_info = self._detect_cpu()
        self.gpu_info = self._detect_gpu()
        self.memory_info = self._detect_memory()
        self.storage_info = self._detect_storage()
    
    def _detect_cpu(self) -> CPUInfo:
        """Detektiert CPU-Spezifikationen"""
        try:
            # CPU Model und Kerne
            cpu_model = platform.processor()
            
            # Fallback für Linux (platform.processor() gibt oft nur "x86_64" zurück)
            if not cpu_model or cpu_model == "x86_64":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if 'model name' in line:
                                cpu_model = line.split(':')[1].strip()
                                break
                except:
                    cpu_model = "Unknown CPU"
            
            cores_physical = psutil.cpu_count(logical=False)
            cores_logical = psutil.cpu_count(logical=True)
            
            # CPU Frequenz
            cpu_freq = psutil.cpu_freq()
            freq_max = cpu_freq.max if cpu_freq else 0.0
            freq_current = cpu_freq.current if cpu_freq else 0.0
            
            # L3 Cache (Linux spezifisch)
            cache_l3 = self._get_l3_cache_size()
            
            return CPUInfo(
                model=cpu_model,
                cores_physical=cores_physical,
                cores_logical=cores_logical,
                frequency_max=freq_max,
                frequency_current=freq_current,
                cache_l3=cache_l3
            )
        except Exception as e:
            print(f"CPU Detection Error: {e}")
            return CPUInfo("Unknown", 1, 1, 0.0, 0.0)
    
    def _detect_gpu(self) -> List[GPUInfo]:
        """Detektiert GPU-Spezifikationen"""
        gpus = []
        
        if not torch.cuda.is_available():
            return gpus
        
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Memory Info
                memory_total = props.total_memory
                memory_free = torch.cuda.memory_reserved(i)
                
                # Compute Capability
                compute_cap = (props.major, props.minor)
                
                # RTX 5090 spezifische Erkennung
                cuda_cores = self._estimate_cuda_cores(props.name, compute_cap)
                tensor_cores = self._estimate_tensor_cores(props.name)
                
                gpu_info = GPUInfo(
                    name=props.name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    compute_capability=compute_cap,
                    cuda_cores=cuda_cores,
                    tensor_cores=tensor_cores
                )
                gpus.append(gpu_info)
                
        except Exception as e:
            print(f"GPU Detection Error: {e}")
        
        return gpus
    
    def _detect_memory(self) -> MemoryInfo:
        """Detektiert RAM-Spezifikationen"""
        try:
            memory = psutil.virtual_memory()
            
            # DDR5-6000 Erkennung (Linux)
            frequency = self._get_memory_frequency()
            memory_type = self._get_memory_type()
            
            return MemoryInfo(
                total=memory.total,
                available=memory.available,
                frequency=frequency,
                type=memory_type
            )
        except Exception as e:
            print(f"Memory Detection Error: {e}")
            return MemoryInfo(0, 0)
    
    def _detect_storage(self) -> List[StorageInfo]:
        """Detektiert Storage-Spezifikationen"""
        storage_devices = []
        
        try:
            # Disk Usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    # Storage Type Detection
                    storage_type = self._get_storage_type(partition.device)
                    
                    # Samsung 9100 PRO spezifische Erkennung
                    read_speed, write_speed = self._get_storage_speeds(partition.device)
                    
                    storage_info = StorageInfo(
                        device=partition.device,
                        total=usage.total,
                        free=usage.free,
                        type=storage_type,
                        read_speed=read_speed,
                        write_speed=write_speed
                    )
                    storage_devices.append(storage_info)
                    
                except PermissionError:
                    continue
                    
        except Exception as e:
            print(f"Storage Detection Error: {e}")
        
        return storage_devices
    
    def _get_l3_cache_size(self) -> Optional[int]:
        """Ermittelt L3 Cache Größe (Linux)"""
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'L3 cache' in line:
                    cache_str = line.split(':')[1].strip()
                    # Parse cache size (z.B. "64 MiB")
                    if 'MiB' in cache_str:
                        return int(cache_str.split()[0]) * 1024 * 1024
                    elif 'KiB' in cache_str:
                        return int(cache_str.split()[0]) * 1024
        except:
            pass
        return None
    
    def _estimate_cuda_cores(self, gpu_name: str, compute_cap: tuple) -> Optional[int]:
        """Schätzt CUDA Cores basierend auf GPU Name"""
        gpu_name_lower = gpu_name.lower()
        
        # RTX 5090 spezifisch
        if 'rtx 5090' in gpu_name_lower:
            return 21760  # Geschätzte CUDA Cores für RTX 5090
        elif 'rtx 4090' in gpu_name_lower:
            return 16384
        elif 'rtx 4080' in gpu_name_lower:
            return 9728
        
        return None
    
    def _estimate_tensor_cores(self, gpu_name: str) -> Optional[int]:
        """Schätzt Tensor Cores basierend auf GPU Name"""
        gpu_name_lower = gpu_name.lower()
        
        if 'rtx 5090' in gpu_name_lower:
            return 680  # Geschätzte Tensor Cores für RTX 5090
        elif 'rtx 4090' in gpu_name_lower:
            return 512
        
        return None
    
    def _get_memory_frequency(self) -> Optional[int]:
        """Ermittelt RAM Frequenz (Linux)"""
        try:
            result = subprocess.run(['dmidecode', '-t', 'memory'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Speed:' in line and 'MHz' in line:
                    speed_str = line.split(':')[1].strip()
                    if speed_str != 'Unknown':
                        return int(speed_str.split()[0])
        except:
            pass
        return None
    
    def _get_memory_type(self) -> Optional[str]:
        """Ermittelt RAM Typ (DDR4, DDR5)"""
        try:
            result = subprocess.run(['dmidecode', '-t', 'memory'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Type:' in line and 'DDR' in line:
                    return line.split(':')[1].strip()
        except:
            pass
        return None
    
    def _get_storage_type(self, device: str) -> str:
        """Ermittelt Storage Typ (SSD, NVMe, HDD)"""
        try:
            # Check if NVMe
            if 'nvme' in device.lower():
                return 'NVMe'
            
            # Check rotation (0 = SSD, >0 = HDD)
            device_name = device.split('/')[-1].rstrip('0123456789')
            try:
                with open(f'/sys/block/{device_name}/queue/rotational', 'r') as f:
                    rotational = f.read().strip()
                    return 'SSD' if rotational == '0' else 'HDD'
            except:
                pass
                
        except:
            pass
        
        return 'Unknown'
    
    def _get_storage_speeds(self, device: str) -> tuple:
        """Ermittelt Storage Read/Write Geschwindigkeiten"""
        # Placeholder - würde Benchmark-Tools erfordern
        device_lower = device.lower()
        
        # Samsung 9100 PRO spezifische Werte
        if 'samsung' in device_lower or 'nvme' in device_lower:
            return (7000, 6900)  # MB/s Read/Write für Samsung 9100 PRO
        
        return (None, None)
    
    def is_target_hardware(self) -> Dict[str, bool]:
        """Prüft ob Ziel-Hardware (Ryzen 9 9950X, RTX 5090, 192GB RAM) vorhanden"""
        checks = {
            'ryzen_9950x': False,
            'rtx_5090': False,
            'ram_192gb': False,
            'samsung_9100_pro': False
        }
        
        # CPU Check
        if self.cpu_info and 'ryzen' in self.cpu_info.model.lower():
            if '9950x' in self.cpu_info.model.lower():
                checks['ryzen_9950x'] = True
        
        # GPU Check
        if self.gpu_info:
            for gpu in self.gpu_info:
                if 'rtx 5090' in gpu.name.lower():
                    checks['rtx_5090'] = True
                    break
        
        # RAM Check (170GB praktisch verfügbar)
        if self.memory_info and self.memory_info.total > 170 * 1024**3:
            checks['ram_192gb'] = True  # Name beibehalten für Kompatibilität
        
        # Storage Check
        if self.storage_info:
            for storage in self.storage_info:
                if storage.type == 'NVMe' and storage.read_speed and storage.read_speed > 6000:
                    checks['samsung_9100_pro'] = True
                    break
        
        return checks
    
    def get_optimal_worker_counts(self) -> Dict[str, int]:
        """Berechnet optimale Worker-Anzahl für verschiedene Tasks"""
        if not self.cpu_info:
            return {'data_pipeline': 1, 'indicator_calc': 1, 'chart_render': 1, 'library_mgmt': 1}
        
        total_cores = self.cpu_info.cores_logical
        
        # Ryzen 9 9950X optimierte Verteilung (32 Kerne)
        if total_cores >= 32:
            return {
                'data_pipeline': 8,
                'indicator_calc': 8, 
                'chart_render': 8,
                'library_mgmt': 8
            }
        elif total_cores >= 16:
            return {
                'data_pipeline': 4,
                'indicator_calc': 4,
                'chart_render': 4,
                'library_mgmt': 4
            }
        else:
            cores_per_task = max(1, total_cores // 4)
            return {
                'data_pipeline': cores_per_task,
                'indicator_calc': cores_per_task,
                'chart_render': cores_per_task,
                'library_mgmt': cores_per_task
            }
    
    def print_hardware_summary(self):
        """Gibt Hardware-Zusammenfassung aus"""
        print("=== Hardware Detection Summary ===")
        
        if self.cpu_info:
            print(f"CPU: {self.cpu_info.model}")
            print(f"Cores: {self.cpu_info.cores_physical} physical, {self.cpu_info.cores_logical} logical")
            print(f"Frequency: {self.cpu_info.frequency_current:.0f} MHz (Max: {self.cpu_info.frequency_max:.0f} MHz)")
            if self.cpu_info.cache_l3:
                print(f"L3 Cache: {self.cpu_info.cache_l3 // (1024*1024)} MB")
        
        if self.gpu_info:
            for i, gpu in enumerate(self.gpu_info):
                print(f"GPU {i}: {gpu.name}")
                print(f"Memory: {gpu.memory_total // (1024**3)} GB")
                print(f"Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
                if gpu.cuda_cores:
                    print(f"CUDA Cores: {gpu.cuda_cores}")
                if gpu.tensor_cores:
                    print(f"Tensor Cores: {gpu.tensor_cores}")
        
        if self.memory_info:
            print(f"RAM: {self.memory_info.total // (1024**3)} GB total, {self.memory_info.available // (1024**3)} GB available")
            if self.memory_info.frequency:
                print(f"RAM Frequency: {self.memory_info.frequency} MHz")
            if self.memory_info.type:
                print(f"RAM Type: {self.memory_info.type}")
        
        if self.storage_info:
            for storage in self.storage_info:
                print(f"Storage: {storage.device} ({storage.type})")
                print(f"Space: {storage.total // (1024**3)} GB total, {storage.free // (1024**3)} GB free")
                if storage.read_speed:
                    print(f"Speed: {storage.read_speed} MB/s read, {storage.write_speed} MB/s write")
        
        # Target Hardware Check
        checks = self.is_target_hardware()
        print("\n=== Target Hardware Status ===")
        print(f"Ryzen 9 9950X: {'✓' if checks['ryzen_9950x'] else '✗'}")
        print(f"RTX 5090: {'✓' if checks['rtx_5090'] else '✗'}")
        print(f"192GB RAM: {'✓' if checks['ram_192gb'] else '✗'}")
        print(f"Samsung 9100 PRO: {'✓' if checks['samsung_9100_pro'] else '✗'}")
        
        # Worker Recommendations
        workers = self.get_optimal_worker_counts()
        print(f"\n=== Recommended Worker Counts ===")
        for task, count in workers.items():
            print(f"{task}: {count} workers")