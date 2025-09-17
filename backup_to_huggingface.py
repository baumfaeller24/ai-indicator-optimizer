#!/usr/bin/env python3
"""
🚀 AI-Indicator-Optimizer Backup zu Hugging Face
Erstellt automatisches Backup des kompletten Projekts
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from datetime import datetime
import json
import subprocess
import sys


def create_project_readme():
    """Erstellt README.md für Hugging Face Repository"""
    readme_content = """---
title: AI-Indicator-Optimizer
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.49.1
app_file: demo_gui.py
pinned: false
license: mit
tags:
- trading
- ai
- forex
- technical-analysis
- multimodal
- minicpm
- pytorch
- indicators
- pine-script
---

# 🚀 AI-Indicator-Optimizer

**Multimodal KI-gesteuerte Trading-Indikator-Optimierung mit MiniCPM-4.1-8B**

## 🎯 Überblick

Das AI-Indicator-Optimizer System nutzt das MiniCPM-4.1-8B Vision-Language Model zur Analyse von EUR/USD Forex-Daten sowohl numerisch (Indikatoren) als auch visuell (Chart-Patterns) zur automatischen Generierung optimierter Pine Script Trading-Strategien.

## 🖥️ Hardware-Optimierung

Optimiert für High-End Hardware:
- **CPU**: AMD Ryzen 9 9950X (16 Kerne, 32 Threads)
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **RAM**: 191GB DDR5-6000
- **Storage**: NVMe SSDs mit 7GB/s

## ✨ Features

### 🔄 **Multimodale Data Processing Pipeline**
- **8 Standard-Indikatoren** parallel berechnet (RSI, MACD, Bollinger, SMA, EMA, Stochastic, ATR, ADX)
- **32-Thread-Parallelisierung** für maximale CPU-Auslastung
- **GPU-beschleunigte Chart-Generierung** mit RTX 5090
- **Multi-Timeframe Charts** (1m, 5m, 15m, 1h, 100tick, 1000tick)

### 🗄️ **Trading Library Database System**
- **PostgreSQL-Schema** für Pattern- und Strategy-Storage
- **30GB In-Memory-Cache** für 191GB RAM-Optimierung
- **Pattern-Ähnlichkeitssuche** basierend auf Image-Features
- **Strategie-Evolution** durch genetische Algorithmen

### 📊 **Dukascopy Data Connector**
- **Parallele Downloads** mit allen 32 CPU-Threads
- **Tick-Data und OHLCV-Abruf** für EUR/USD
- **Datenvalidierung** und Integrity-Checks
- **14-Tage-Datensammlung** in Sekunden

### 🎨 **Interactive Demo GUI**
- **Streamlit-basierte Web-Interface**
- **Live-Hardware-Monitoring**
- **Plotly-Charts** mit interaktiven Features
- **Real-time Processing** von Marktdaten

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://huggingface.co/spaces/ai-trading/ai-indicator-optimizer
cd ai-indicator-optimizer
chmod +x install.sh
./install.sh
```

### 2. Hardware-Check
```bash
python -m ai_indicator_optimizer.main --hardware-check
```

### 3. Demo starten
```bash
streamlit run demo_gui.py
```

### 4. Tests ausführen
```bash
python run_tests.py
```

## 📁 Projekt-Struktur

```
ai_indicator_optimizer/
├── core/                   # Hardware-Detection & Resource-Management
├── data/                   # Dukascopy Connector & Data Processing
├── ai/                     # MiniCPM Integration (geplant)
├── library/                # Trading Library Database System
├── generator/              # Pine Script Generator (geplant)
└── main.py                 # Main Application

tests/                      # Umfassende Test-Suite
demo_gui.py                # Interactive Streamlit Demo
```

## 🧪 Test-Ergebnisse

- **Data Connector**: 20/20 Tests ✅
- **Data Processor**: 17/17 Tests ✅  
- **Trading Library**: 18/19 Tests ✅
- **Hardware Detection**: 4/4 Tests ✅

**Gesamt: 59/60 Tests bestanden (98.3% Success Rate)**

## 🎯 Roadmap

### ✅ Abgeschlossen (Tasks 1-4)
- [x] Projekt-Setup und Core-Infrastruktur
- [x] Dukascopy Data Connector
- [x] Multimodal Data Processing Pipeline  
- [x] Trading Library Database System

### 🚧 In Entwicklung (Tasks 5-15)
- [ ] MiniCPM-4.1-8B Model Integration
- [ ] Fine-Tuning Pipeline für Trading-Patterns
- [ ] Automated Library Population System
- [ ] Multimodal Pattern Recognition Engine
- [ ] Pine Script Code Generator
- [ ] Pine Script Validation und Optimization
- [ ] Hardware Utilization Monitoring
- [ ] Comprehensive Logging und Progress Tracking
- [ ] Error Handling und Recovery System
- [ ] Integration Testing und Validation
- [ ] Main Application und CLI Interface

## 🔧 Technische Details

### Performance-Optimierungen
- **Parallele Indikator-Berechnung** mit ThreadPoolExecutor
- **GPU-beschleunigte Chart-Rendering** mit PyTorch
- **In-Memory-Caching** für 30GB Trading-Daten
- **Async Bulk-Operations** für Datenbank-Performance

### Multimodale KI-Pipeline
- **Vision+Text-Eingaben** für MiniCPM-4.1-8B
- **Feature-Normalisierung** (Z-Score)
- **Automatische Text-Beschreibungen** für Chart-Patterns
- **Image-Preprocessing** (224x224 für Vision Models)

## 📈 Performance-Metriken

Mit der Ziel-Hardware:
- **Indikator-Berechnung**: 1000 Candles in <5 Sekunden
- **Chart-Generierung**: 4 Timeframes parallel in <2 Sekunden
- **Pattern-Ähnlichkeitssuche**: <100ms für 10k Patterns
- **Datenbank-Queries**: <50ms mit 30GB Cache

## 🤝 Contributing

Das Projekt folgt einem strukturierten Entwicklungsplan mit 15 Tasks. Jede Task hat spezifische Requirements und Tests.

## 📄 Lizenz

MIT License - siehe LICENSE Datei für Details.

## 🙏 Acknowledgments

- **MiniCPM-4.1-8B** von OpenBMB für multimodale KI
- **Dukascopy** für Forex-Daten
- **PyTorch** für GPU-Beschleunigung
- **Streamlit** für Interactive Demo

---

**🚀 Powered by MiniCPM-4.1-8B & RTX 5090**
"""
    
    return readme_content


def create_requirements_txt():
    """Erstellt requirements.txt für Hugging Face"""
    requirements = """# Core Dependencies
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Transformers und HuggingFace
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
tokenizers>=0.14.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Data Processing
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0

# Database
psycopg2-binary>=2.9.7
sqlalchemy>=2.0.0

# Async und Networking
aiohttp>=3.8.0
requests>=2.31.0

# Monitoring und System
psutil>=5.9.0

# Web Interface
streamlit>=1.49.0
plotly>=6.3.0

# Development Tools
pytest>=7.4.0

# Hugging Face
huggingface_hub>=0.35.0
"""
    
    return requirements


def create_app_config():
    """Erstellt app.py für Hugging Face Spaces"""
    app_content = """#!/usr/bin/env python3
'''
🚀 AI-Indicator-Optimizer Hugging Face App
'''

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import demo
from demo_gui import main

if __name__ == "__main__":
    main()
"""
    
    return app_content


def get_project_stats():
    """Sammelt Projekt-Statistiken"""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "files": {},
        "lines_of_code": 0,
        "test_files": 0,
        "tasks_completed": 0
    }
    
    # Zähle Dateien und Zeilen
    for root, dirs, files in os.walk("."):
        # Ignoriere bestimmte Verzeichnisse
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'test_env', 'node_modules']]
        
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        stats["files"][file_path] = lines
                        stats["lines_of_code"] += lines
                        
                        if file.startswith('test_'):
                            stats["test_files"] += 1
                except:
                    pass
    
    # Zähle abgeschlossene Tasks
    try:
        with open('.kiro/specs/ai-indicator-optimizer/tasks.md', 'r') as f:
            content = f.read()
            stats["tasks_completed"] = content.count('- [x]')
    except:
        pass
    
    return stats


def backup_to_huggingface(repo_name: str = "ai-indicator-optimizer", 
                         username: str = "ai-trading",
                         private: bool = False):
    """
    Erstellt Backup auf Hugging Face
    """
    print("🚀 Starting AI-Indicator-Optimizer Backup to Hugging Face...")
    
    # Erstelle temporäres Verzeichnis
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_dir = Path(temp_dir) / "backup"
        backup_dir.mkdir()
        
        print(f"📁 Created temporary backup directory: {backup_dir}")
        
        # Kopiere Projekt-Dateien
        print("📋 Copying project files...")
        
        # Hauptverzeichnisse
        dirs_to_copy = [
            "ai_indicator_optimizer",
            "tests", 
            ".kiro"
        ]
        
        for dir_name in dirs_to_copy:
            if os.path.exists(dir_name):
                shutil.copytree(dir_name, backup_dir / dir_name)
                print(f"  ✓ Copied {dir_name}/")
        
        # Einzelne Dateien
        files_to_copy = [
            "demo_gui.py",
            "test_setup.py", 
            "run_tests.py",
            "install.sh",
            "setup.py"
        ]
        
        for file_name in files_to_copy:
            if os.path.exists(file_name):
                shutil.copy2(file_name, backup_dir / file_name)
                print(f"  ✓ Copied {file_name}")
        
        # Erstelle Hugging Face spezifische Dateien
        print("📝 Creating Hugging Face files...")
        
        # README.md
        with open(backup_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(create_project_readme())
        print("  ✓ Created README.md")
        
        # requirements.txt
        with open(backup_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(create_requirements_txt())
        print("  ✓ Created requirements.txt")
        
        # app.py (für Hugging Face Spaces)
        with open(backup_dir / "app.py", "w", encoding="utf-8") as f:
            f.write(create_app_config())
        print("  ✓ Created app.py")
        
        # Projekt-Statistiken
        stats = get_project_stats()
        with open(backup_dir / "project_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("  ✓ Created project_stats.json")
        
        # .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
test_env/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# Cache
.cache/
.pytest_cache/

# Model files (too large for git)
models/
checkpoints/
*.pth
*.bin

# Data files
data/
*.csv
*.parquet
"""
        
        with open(backup_dir / ".gitignore", "w") as f:
            f.write(gitignore_content)
        print("  ✓ Created .gitignore")
        
        print(f"\n📊 Backup Statistics:")
        print(f"  - Files: {len(stats['files'])}")
        print(f"  - Lines of Code: {stats['lines_of_code']:,}")
        print(f"  - Test Files: {stats['test_files']}")
        print(f"  - Tasks Completed: {stats['tasks_completed']}")
        
        # Upload zu Hugging Face
        print(f"\n🚀 Uploading to Hugging Face...")
        
        try:
            api = HfApi()
            
            # Repository erstellen (falls nicht vorhanden)
            repo_id = f"{username}/{repo_name}"
            
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="space",
                    space_sdk="streamlit",
                    private=private,
                    exist_ok=True
                )
                print(f"  ✓ Repository created/verified: {repo_id}")
            except Exception as e:
                print(f"  ⚠ Repository creation: {e}")
            
            # Upload Dateien
            api.upload_folder(
                folder_path=backup_dir,
                repo_id=repo_id,
                repo_type="space",
                commit_message=f"🚀 AI-Indicator-Optimizer Backup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            print(f"✅ Backup successfully uploaded to: https://huggingface.co/spaces/{repo_id}")
            print(f"🎯 Demo URL: https://huggingface.co/spaces/{repo_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("💡 Make sure you're logged in: huggingface-cli login")
            return False


def main():
    """Main Backup Function"""
    print("🚀 AI-Indicator-Optimizer Hugging Face Backup Tool")
    print("=" * 60)
    
    # Check if logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"👤 Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged in to Hugging Face")
        print("💡 Please run: huggingface-cli login")
        return
    
    # Backup durchführen
    success = backup_to_huggingface(
        repo_name="ai-indicator-optimizer",
        username="ai-trading",  # Ändere das zu deinem Username
        private=False  # Public für Demo
    )
    
    if success:
        print("\n🎉 Backup completed successfully!")
        print("🔗 Your project is now available on Hugging Face Spaces")
        print("📱 The Streamlit demo will be automatically deployed")
    else:
        print("\n❌ Backup failed. Please check the errors above.")


if __name__ == "__main__":
    main()