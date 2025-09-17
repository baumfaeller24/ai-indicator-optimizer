"""
Setup Script für AI-Indicator-Optimizer
"""

from setuptools import setup, find_packages
import sys
import subprocess

# Check Python Version
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9 oder höher erforderlich")

# Hardware Detection für optimale Installation
def detect_cuda():
    """Detektiert CUDA Verfügbarkeit"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_torch_index_url():
    """Gibt PyTorch Index URL basierend auf CUDA zurück"""
    if detect_cuda():
        return "https://download.pytorch.org/whl/cu121"  # CUDA 12.1
    else:
        return "https://download.pytorch.org/whl/cpu"

# Requirements basierend auf Hardware
def get_requirements():
    """Lädt Requirements basierend auf verfügbarer Hardware"""
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    
    # Filtere Kommentare und leere Zeilen
    requirements = [req for req in requirements if req and not req.startswith('#')]
    
    return requirements

setup(
    name="ai-indicator-optimizer",
    version="1.0.0",
    description="Multimodal KI-gesteuerte Trading-Indikator-Optimierung mit MiniCPM-4.1-8B",
    long_description=open('README.md', 'r', encoding='utf-8').read() if __import__('os').path.exists('README.md') else "",
    long_description_content_type="text/markdown",
    author="AI Trading Research",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={
        'cuda': [
            'nvidia-ml-py>=12.535.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'black>=23.9.0',
            'flake8>=6.1.0',
            'mypy>=1.6.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'ai-optimizer=ai_indicator_optimizer.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="trading, ai, machine-learning, forex, technical-analysis, pine-script",
    project_urls={
        "Documentation": "https://github.com/ai-trading/ai-indicator-optimizer",
        "Source": "https://github.com/ai-trading/ai-indicator-optimizer",
        "Tracker": "https://github.com/ai-trading/ai-indicator-optimizer/issues",
    },
)

# Post-Installation Hardware Check
if __name__ == "__main__":
    import os
    
    print("=== AI-Indicator-Optimizer Installation ===")
    print(f"Python Version: {sys.version}")
    print(f"CUDA Available: {detect_cuda()}")
    print(f"PyTorch Index: {get_torch_index_url()}")
    
    # Hardware Detection nach Installation
    try:
        from ai_indicator_optimizer.core import HardwareDetector
        detector = HardwareDetector()
        detector.print_hardware_summary()
    except ImportError:
        print("Hardware detection will be available after installation")