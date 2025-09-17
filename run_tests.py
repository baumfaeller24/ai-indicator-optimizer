#!/usr/bin/env python3
"""
Test Runner für AI-Indicator-Optimizer
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Führt alle Tests aus"""
    print("=== AI-Indicator-Optimizer Test Suite ===")
    
    # Stelle sicher, dass wir im richtigen Verzeichnis sind
    os.chdir(Path(__file__).parent)
    
    # Installiere pytest falls nicht vorhanden
    try:
        import pytest
    except ImportError:
        print("Installing pytest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)
        import pytest
    
    # Installiere requests für Tests
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    
    # Führe Tests aus
    test_args = [
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    print(f"Running: pytest {' '.join(test_args)}")
    result = pytest.main(test_args)
    
    if result == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result}")
    
    return result


if __name__ == "__main__":
    sys.exit(run_tests())