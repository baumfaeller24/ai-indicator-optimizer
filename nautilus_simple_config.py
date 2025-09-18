#!/usr/bin/env python3
"""
🚀 Vereinfachte Nautilus-Konfiguration für RTX 5090 + Ryzen 9950X
Fokus auf funktionierende Basis-Konfiguration mit ChatGPT-Optimierungen
"""

import os
import psutil
import torch
from pathlib import Path

class SimpleNautilusConfig:
    """
    Vereinfachte, funktionierende Nautilus-Konfiguration
    Basiert auf ChatGPT-Hardware-Optimierungen
    """
    
    def __init__(self):
        # Hardware-Erkennung (ChatGPT-optimiert)
        self.cpu_cores_logical = psutil.cpu_count(logical=True)    # 32 Threads
        self.cpu_cores_physical = psutil.cpu_count(logical=False)  # 16 Kerne
        self.memory_gb = psutil.virtual_memory().total // (1024**3)
        
        # GPU-Erkennung (robuster)
        self.gpu_available = torch.cuda.is_available()
        try:
            if self.gpu_available:
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                self.gpu_name = torch.cuda.get_device_name(0)
            else:
                self.gpu_memory_gb = 0
                self.gpu_name = "None"
        except Exception:
            self.gpu_memory_gb = 0
            self.gpu_name = "Error"
        
        print(f"🖥️ Hardware Detection:")
        print(f"   CPU: {self.cpu_cores_physical} physical / {self.cpu_cores_logical} logical cores")
        print(f"   RAM: {self.memory_gb} GB")
        print(f"   GPU: {self.gpu_name} ({self.gpu_memory_gb} GB VRAM)")
    
    def setup_environment_variables(self):
        """ChatGPT-optimierte Umgebungsvariablen"""
        
        # CPU-Optimierungen
        threads = min(self.cpu_cores_logical, 32)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMBA_NUM_THREADS"] = str(threads)
        os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"
        os.environ["OMP_DYNAMIC"] = "true"
        
        # GPU-Optimierungen
        if self.gpu_available:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
                os.environ["TORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
                
                # High-Performance Matmul (ChatGPT-Empfehlung)
                torch.set_float32_matmul_precision("high")
                
                # KI-Optimierungen
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                
            except Exception as e:
                print(f"⚠️ GPU optimization warning: {e}")
        
        # Memory-Optimierungen
        os.environ["PYTHONHASHSEED"] = "0"
        
        # Nautilus-Optimierungen (skaliert mit RAM)
        cache_capacity = int(self.memory_gb * 1_000_000)  # 182M bei 182GB
        os.environ["NAUTILUS_CACHE_CAPACITY"] = str(cache_capacity)
        os.environ["NAUTILUS_THREADING"] = "true"
        
        print("🔧 Environment Variables konfiguriert:")
        print(f"   Threads: {threads}")
        print(f"   Cache Capacity: {cache_capacity:,}")
        if self.gpu_available:
            print(f"   GPU Optimization: ✅")
    
    def create_directories(self):
        """Erstellt Nautilus-Verzeichnisse"""
        directories = [
            "nautilus_data",
            "nautilus_logs",
            "nautilus_cache", 
            "nautilus_config",
            "catalog",
            "strategies",
            "adapters",
            "backtest_results",
            "live_data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True, parents=True)
        
        print(f"📁 {len(directories)} Nautilus-Verzeichnisse erstellt")
    
    def create_basic_config_file(self):
        """Erstellt Basis-Konfigurationsdatei"""
        config_content = f"""# Nautilus Trader Configuration
# Hardware: {self.cpu_cores_physical}C/{self.cpu_cores_logical}T, {self.memory_gb}GB RAM, {self.gpu_name}

# Environment Settings
export OMP_NUM_THREADS={min(self.cpu_cores_logical, 32)}
export MKL_NUM_THREADS={min(self.cpu_cores_logical, 32)}
export NUMBA_NUM_THREADS={min(self.cpu_cores_logical, 32)}
export NAUTILUS_CACHE_CAPACITY={int(self.memory_gb * 1_000_000)}

# GPU Settings (RTX 5090)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# Performance Optimizations
# - Tick Capacity: {int(self.memory_gb * 100_000):,} (ChatGPT-optimiert)
# - Bar Capacity: {int(self.memory_gb * 5_000):,}
# - Cache Entries: {int(self.memory_gb * 1_000_000):,}

# Hardware Specs:
# CPU: AMD Ryzen 9 9950X ({self.cpu_cores_physical} cores / {self.cpu_cores_logical} threads)
# RAM: {self.memory_gb} GB DDR5-6000
# GPU: {self.gpu_name} ({self.gpu_memory_gb} GB VRAM)
"""
        
        config_path = Path("nautilus_config") / "hardware_config.sh"
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"💾 Konfiguration gespeichert: {config_path}")
        return config_path
    
    def test_nautilus_import(self):
        """Testet Nautilus-Import und grundlegende Funktionalität"""
        try:
            import nautilus_trader
            print(f"✅ NautilusTrader {nautilus_trader.__version__} erfolgreich importiert")
            
            # Test basic imports
            from nautilus_trader.model.identifiers import TraderId
            from nautilus_trader.model.currencies import USD
            
            trader_id = TraderId("AI-OPTIMIZER-001")
            print(f"✅ Basis-Komponenten funktionieren: {trader_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Nautilus-Import fehlgeschlagen: {e}")
            return False
    
    def validate_performance_expectations(self):
        """Validiert Performance-Erwartungen basierend auf Hardware"""
        print("\n🎯 Performance-Erwartungen (ChatGPT-Benchmark):")
        
        # Tick-Processing-Schätzung
        estimated_tps = self.cpu_cores_logical * 100_000  # Konservative Schätzung
        print(f"   Geschätzte Tick-Rate: {estimated_tps:,} Ticks/Sekunde")
        
        # Memory-Kapazität
        tick_capacity = int(self.memory_gb * 100_000)
        print(f"   Tick-Kapazität (RAM): {tick_capacity:,} Ticks")
        
        # GPU-Inferenz
        if self.gpu_available:
            estimated_ips = 1_000_000  # 1M Inferenzen/Sek auf RTX 5090
            print(f"   Geschätzte GPU-Inferenz: {estimated_ips:,} Inferenzen/Sekunde")
        
        # ChatGPT-Ziel
        print(f"\n   🎯 ChatGPT-Ziel: 1-3 Millionen Ticks/Sekunde")
        if estimated_tps >= 1_000_000:
            print(f"   ✅ Hardware sollte Ziel erreichen!")
        else:
            print(f"   ⚠️ Möglicherweise unter Ziel")

def main():
    """Hauptfunktion für vereinfachtes Nautilus-Setup"""
    print("🚀 Vereinfachtes Nautilus Hardware Setup")
    print("=" * 50)
    
    # Konfiguration erstellen
    config = SimpleNautilusConfig()
    
    # Environment Variables setzen
    config.setup_environment_variables()
    
    # Verzeichnisse erstellen
    config.create_directories()
    
    # Konfigurationsdatei erstellen
    config_path = config.create_basic_config_file()
    
    # Nautilus-Import testen
    nautilus_works = config.test_nautilus_import()
    
    # Performance-Erwartungen
    config.validate_performance_expectations()
    
    print("\n" + "=" * 50)
    if nautilus_works:
        print("✅ Nautilus Hardware Setup erfolgreich!")
        print(f"📁 Konfiguration: {config_path}")
        print("🚀 Bereit für Nautilus-Entwicklung!")
    else:
        print("❌ Setup teilweise fehlgeschlagen")
        print("🔧 Prüfe Nautilus-Installation")
    
    return nautilus_works

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)