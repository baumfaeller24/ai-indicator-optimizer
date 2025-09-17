#!/bin/bash

# AI-Indicator-Optimizer Installation Script
# Optimiert für Ryzen 9 9950X + RTX 5090 + 192GB RAM

set -e

echo "=== AI-Indicator-Optimizer Installation ==="

# Check Python Version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.9+ required, found $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected - CPU-only mode"
fi

# Create Virtual Environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install package in development mode
echo "Installing AI-Indicator-Optimizer..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p logs data models results checkpoints

# Hardware Check
echo "Running hardware detection..."
python -m ai_indicator_optimizer.main --hardware-check

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run hardware check:"
echo "  python -m ai_indicator_optimizer.main --hardware-check"
echo ""
echo "To start the application:"
echo "  python -m ai_indicator_optimizer.main"