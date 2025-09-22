"""
Main Application Entry Point
"""

import argparse
import sys
import logging
from pathlib import Path

from .core import HardwareDetector, ResourceManager, SystemConfig


def setup_logging(log_level: str = "INFO"):
    """Konfiguriert Logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai_optimizer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main Application Entry Point"""
    parser = argparse.ArgumentParser(
        description="AI-Indicator-Optimizer: Multimodal Trading Strategy Optimization"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='./config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--hardware-check',
        action='store_true',
        help='Run hardware detection and exit'
    )
    
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Setup environment and exit'
    )
    
    args = parser.parse_args()
    
    # Setup Logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AI-Indicator-Optimizer")
    
    try:
        # Initialize System Configuration
        config = SystemConfig(args.config)
        config.print_config_summary()
        
        # Hardware Detection
        hardware_detector = HardwareDetector()
        hardware_detector.print_hardware_summary()
        
        if args.hardware_check:
            logger.info("Hardware check completed")
            return 0
        
        # Resource Manager
        resource_manager = ResourceManager(hardware_detector)
        resource_manager.print_allocation_summary()
        
        # Start Resource Monitoring
        resource_manager.start_monitoring()
        
        if args.setup_only:
            logger.info("Setup completed")
            resource_manager.cleanup()
            return 0
        
        logger.info("System initialization completed")
        logger.info("Ready for trading strategy optimization")
        
        # Hier würde die Hauptlogik implementiert werden
        # Für jetzt nur Setup und Hardware-Check
        
        # Cleanup
        resource_manager.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())