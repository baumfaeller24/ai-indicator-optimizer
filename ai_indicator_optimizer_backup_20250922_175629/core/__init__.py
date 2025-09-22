"""
Core Module: Hardware-Detection, Ressourcen-Management und System-Konfiguration
"""

from .hardware_detector import HardwareDetector
from .resource_manager import ResourceManager
from .config import SystemConfig

__all__ = ['HardwareDetector', 'ResourceManager', 'SystemConfig']