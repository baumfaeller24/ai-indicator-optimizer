#!/usr/bin/env python3
"""
Test Script für AI-Indicator-Optimizer Setup
"""

import sys
import importlib
import traceback


def test_imports():
    """Testet alle wichtigen Imports"""
    print("=== Testing Imports ===")
    
    modules_to_test = [
        'ai_indicator_optimizer',
        'ai_indicator_optimizer.core',
        'ai_indicator_optimizer.core.hardware_detector',
        'ai_indicator_optimizer.core.resource_manager',
        'ai_indicator_optimizer.core.config',
        'ai_indicator_optimizer.data',
        'ai_indicator_optimizer.ai',
        'ai_indicator_optimizer.library',
        'ai_indicator_optimizer.generator',
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
        except Exception as e:
            print(f"✗ {module_name}: Unexpected error - {e}")
    
    print(f"\nImport Success Rate: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)


def test_hardware_detection():
    """Testet Hardware-Detection"""
    print("\n=== Testing Hardware Detection ===")
    
    try:
        from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
        
        detector = HardwareDetector()
        detector.print_hardware_summary()
        
        print("✓ Hardware detection successful")
        return True
        
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
        traceback.print_exc()
        return False


def test_resource_manager():
    """Testet Resource Manager"""
    print("\n=== Testing Resource Manager ===")
    
    try:
        from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
        from ai_indicator_optimizer.core.resource_manager import ResourceManager
        
        detector = HardwareDetector()
        resource_manager = ResourceManager(detector)
        resource_manager.print_allocation_summary()
        
        # Test Resource Optimization
        opts = resource_manager.optimize_for_task('data_processing')
        print(f"✓ Data processing optimization: {opts}")
        
        resource_manager.cleanup()
        print("✓ Resource manager test successful")
        return True
        
    except Exception as e:
        print(f"✗ Resource manager test failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Testet System Configuration"""
    print("\n=== Testing System Configuration ===")
    
    try:
        from ai_indicator_optimizer.core.config import SystemConfig
        
        config = SystemConfig()
        config.print_config_summary()
        
        print("✓ System configuration test successful")
        return True
        
    except Exception as e:
        print(f"✗ System configuration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main Test Function"""
    print("AI-Indicator-Optimizer Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Resource Manager", test_resource_manager),
        ("System Configuration", test_config),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())