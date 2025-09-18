Perfekt! Dann zeige ich dir zwei Dinge:

---

## 🟢 **Variante 1: PyTorch Nightly Build mit CUDA 12.8**

Das ist der schnellste Weg, um sm_120-Support zu testen – vorausgesetzt, du hast bereits einen passenden NVIDIA-Treiber (mindestens Version R570 oder höher) installiert.

### 🔧 Vorbereitung

1. **Optional**: Erstelle ein frisches Python-Environment:
   
   ```bash
   python3 -m venv ~/venvs/torch-nightly
   source ~/venvs/torch-nightly/bin/activate
   ```

2. **Installiere Nightly PyTorch mit CUDA 12.8:**
   
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Teste, ob sm_120 vorhanden ist:**
   
   ```python
   python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
   ```
   
   → Achte auf Einträge wie `sm_120` in der Arch-Liste.

4. **Testlauf auf deiner GPU:**
   
   ```python
   python -c "import torch; x = torch.randn(2).cuda(); print(x * 2)"
   ```

---

## 🛠️ **Variante 2: Build from Source mit TORCH_CUDA_ARCH_LIST="sm_120"**

Wenn Nightly nicht funktioniert oder du volle Kontrolle willst, dann so:

### 📦 Schritt-für-Schritt

```bash
# Installiere Abhängigkeiten (für Ubuntu 24.04+)
sudo apt update
sudo apt install -y git cmake ninja-build libopenblas-dev \
    libomp-dev python3-dev python3-pip

# CUDA 12.8 vorausgesetzt installiert und in PATH
# (Falls nicht, NVIDIA Toolkit 12.8 manuell installieren)

# Virtuelle Umgebung (optional)
python3 -m venv ~/venvs/torch-build
source ~/venvs/torch-build/bin/activate

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Setze CUDA Arch-List
export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
export USE_NINJA=1

# Installiere Python-Abhängigkeiten
pip install -r requirements.txt

# Build starten
python setup.py install
```

### ✅ Danach prüfen:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.cuda.is_available())
```

---

Wenn du willst, kann ich dir ein vollständiges Skript zum Copy-Paste oder zur Automatisierung basteln. Sag einfach Bescheid!
