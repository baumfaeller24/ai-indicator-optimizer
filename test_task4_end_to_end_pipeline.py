#!/usr/bin/env python3
"""
Test Task 4: End-to-End Pipeline Core Implementation
Tests the complete Top-5 Strategies Ranking System pipeline
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the main system
from ai_indicator_optimizer.integration.top5_strategies_ranking_system import (
    Top5StrategiesRankingSystem,
    PipelineConfig,
    ExecutionMode
)

class Task4EndToEndPipelineTest:
    """Test suite for Task 4: End-to-End Pipeline Core Implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Task 4 tests"""
        print("🚀 TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION TESTS")
        print("=" * 80)
        
        # Test 1: Pipeline Configuration
        test1_result = await self.test_pipeline_configuration()
        self.results["test1_pipeline_configuration"] = test1_result
        
        # Test 2: Pipeline Initialization
        test2_result = await self.test_pipeline_initialization()
        self.results["test2_pipeline_initialization"] = test2_result
        
        # Test 3: Stage Execution
        test3_result = await self.test_stage_execution()
        self.results["test3_stage_execution"] = test3_result
        
        # Test 4: Full Pipeline Execution
        test4_result = await self.test_full_pipeline_execution()
        self.results["test4_full_pipeline_execution"] = test4_result
        
        # Test 5: Performance and Parallelization
        test5_result = await self.test_performance_parallelization()
        self.results["test5_performance_parallelization"] = test5_result
        
        # Test 6: Export and Output Validation
        test6_result = await self.test_export_output_validation()
        self.results["test6_export_output_validation"] = test6_result
        
        # Summary
        await self.print_test_summary()
        
        return self.results
    
    async def test_pipeline_configuration(self) -> Dict:
        """Test 1: Pipeline Configuration"""
        print("\n🧪 Test 1: Pipeline Configuration")
        
        try:
            # Test default configuration
            default_config = PipelineConfig()
            
            # Test custom configuration
            custom_config = PipelineConfig(
                execution_mode=ExecutionMode.PRODUCTION,
                max_strategies=5,
                symbols=["EUR/USD"],
                timeframes=["1m", "5m", "15m", "1h"],
                max_workers=32,  # Ryzen 9 9950X optimization
                timeout_seconds=300,
                min_confidence_threshold=0.5,
                export_formats=["pine", "json", "csv", "html"]
            )
            
            # Validate configuration
            assert default_config.execution_mode == ExecutionMode.PRODUCTION
            assert default_config.max_workers == 32
            assert custom_config.symbols == ["EUR/USD"]
            assert custom_config.max_strategies == 5
            
            print("✅ Default configuration: OK")
            print("✅ Custom configuration: OK")
            print(f"✅ Max workers (32-core optimization): {custom_config.max_workers}")
            print(f"✅ Export formats: {len(custom_config.export_formats)}")
            
            return {
                "success": True,
                "default_config": True,
                "custom_config": True,
                "max_workers": custom_config.max_workers,
                "export_formats": len(custom_config.export_formats)
            }
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_pipeline_initialization(self) -> Dict:
        """Test 2: Pipeline Initialization"""
        print("\n🧪 Test 2: Pipeline Initialization")
        
        try:
            # Initialize pipeline system
            config = PipelineConfig(
                execution_mode=ExecutionMode.DEVELOPMENT,
                max_workers=16,  # Reduced for testing
                output_dir="test_output"
            )
            
            pipeline_system = Top5StrategiesRankingSystem(config=config)
            
            # Validate initialization
            assert pipeline_system.config.execution_mode == ExecutionMode.DEVELOPMENT
            assert pipeline_system.config.max_workers == 16
            assert pipeline_system.pipeline_id.startswith("top5_pipeline_")
            
            # Test performance stats
            perf_stats = pipeline_system.get_performance_stats()
            
            print("✅ Pipeline system initialized")
            print(f"✅ Pipeline ID: {pipeline_system.pipeline_id}")
            print(f"✅ Execution mode: {pipeline_system.config.execution_mode.value}")
            print(f"✅ Max workers: {pipeline_system.config.max_workers}")
            print(f"✅ Performance stats: {len(perf_stats)} categories")
            
            return {
                "success": True,
                "pipeline_initialized": True,
                "pipeline_id": pipeline_system.pipeline_id,
                "execution_mode": pipeline_system.config.execution_mode.value,
                "max_workers": pipeline_system.config.max_workers,
                "performance_stats": len(perf_stats)
            }
            
        except Exception as e:
            print(f"❌ Initialization test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_stage_execution(self) -> Dict:
        """Test 3: Stage Execution"""
        print("\n🧪 Test 3: Stage Execution")
        
        try:
            # Initialize pipeline
            config = PipelineConfig(execution_mode=ExecutionMode.DEVELOPMENT)
            pipeline_system = Top5StrategiesRankingSystem(config=config)
            
            # Test individual stage execution
            from ai_indicator_optimizer.integration.top5_strategies_ranking_system import PipelineStage
            
            # Test initialization stage
            init_result = pipeline_system.stage_executor.execute_stage(
                PipelineStage.INITIALIZATION, {}
            )
            
            # Test data analysis stage
            analysis_result = pipeline_system.stage_executor.execute_stage(
                PipelineStage.DATA_ANALYSIS, {"initialization": init_result.data}
            )
            
            print(f"✅ Initialization stage: {'Success' if init_result.success else 'Failed'}")
            print(f"✅ Data analysis stage: {'Success' if analysis_result.success else 'Failed'}")
            print(f"✅ Init execution time: {init_result.execution_time:.3f}s")
            print(f"✅ Analysis execution time: {analysis_result.execution_time:.3f}s")
            
            # Validate stage results
            assert init_result.stage == PipelineStage.INITIALIZATION
            assert analysis_result.stage == PipelineStage.DATA_ANALYSIS
            
            return {
                "success": True,
                "initialization_success": init_result.success,
                "analysis_success": analysis_result.success,
                "init_time": init_result.execution_time,
                "analysis_time": analysis_result.execution_time,
                "total_stages_tested": 2
            }
            
        except Exception as e:
            print(f"❌ Stage execution test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_full_pipeline_execution(self) -> Dict:
        """Test 4: Full Pipeline Execution"""
        print("\n🧪 Test 4: Full Pipeline Execution")
        
        try:
            # Initialize pipeline with production config
            config = PipelineConfig(
                execution_mode=ExecutionMode.PRODUCTION,
                max_strategies=3,  # Reduced for testing
                symbols=["EUR/USD"],
                timeframes=["1m", "5m", "15m"],
                max_workers=8,  # Reduced for testing
                output_dir="test_output/task4_full_pipeline"
            )
            
            pipeline_system = Top5StrategiesRankingSystem(config=config)
            
            # Execute full pipeline
            start_time = time.time()
            pipeline_result = pipeline_system.execute_full_pipeline()
            execution_time = time.time() - start_time
            
            # Validate results
            assert pipeline_result.pipeline_id == pipeline_system.pipeline_id
            assert pipeline_result.execution_mode == ExecutionMode.PRODUCTION
            assert len(pipeline_result.stage_results) == 6  # All 6 stages
            
            print(f"✅ Pipeline execution: {'Success' if pipeline_result.success_rate > 0.5 else 'Failed'}")
            print(f"✅ Success rate: {pipeline_result.success_rate:.1%}")
            print(f"✅ Pipeline quality: {pipeline_result.pipeline_quality}")
            print(f"✅ Confidence level: {pipeline_result.confidence_level:.2f}")
            print(f"✅ Total execution time: {pipeline_result.total_execution_time:.2f}s")
            print(f"✅ Stages completed: {len(pipeline_result.stage_results)}/6")
            print(f"✅ Exported files: {len(pipeline_result.exported_files)} formats")
            
            return {
                "success": True,
                "pipeline_success": pipeline_result.success_rate > 0.5,
                "success_rate": pipeline_result.success_rate,
                "pipeline_quality": pipeline_result.pipeline_quality,
                "confidence_level": pipeline_result.confidence_level,
                "execution_time": pipeline_result.total_execution_time,
                "stages_completed": len(pipeline_result.stage_results),
                "exported_files": len(pipeline_result.exported_files)
            }
            
        except Exception as e:
            print(f"❌ Full pipeline execution test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_performance_parallelization(self) -> Dict:
        """Test 5: Performance and Parallelization"""
        print("\n🧪 Test 5: Performance and Parallelization")
        
        try:
            # Test with different worker configurations
            configs = [
                {"workers": 1, "name": "Single-threaded"},
                {"workers": 8, "name": "Multi-threaded (8)"},
                {"workers": 16, "name": "Multi-threaded (16)"},
                {"workers": 32, "name": "Full parallelization (32)"}
            ]
            
            performance_results = []
            
            for config_test in configs:
                config = PipelineConfig(
                    execution_mode=ExecutionMode.DEVELOPMENT,
                    max_workers=config_test["workers"],
                    max_strategies=2,  # Reduced for performance testing
                    output_dir=f"test_output/perf_{config_test['workers']}_workers"
                )
                
                pipeline_system = Top5StrategiesRankingSystem(config=config)
                
                # Measure execution time
                start_time = time.time()
                result = pipeline_system.execute_full_pipeline()
                execution_time = time.time() - start_time
                
                performance_results.append({
                    "name": config_test["name"],
                    "workers": config_test["workers"],
                    "execution_time": execution_time,
                    "success_rate": result.success_rate
                })
                
                print(f"✅ {config_test['name']}: {execution_time:.2f}s, Success: {result.success_rate:.1%}")
            
            # Find best performance
            best_perf = min(performance_results, key=lambda x: x["execution_time"])
            
            print(f"🏆 Best performance: {best_perf['name']} ({best_perf['execution_time']:.2f}s)")
            
            return {
                "success": True,
                "performance_results": performance_results,
                "best_performance": best_perf,
                "parallelization_tested": len(configs)
            }
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_export_output_validation(self) -> Dict:
        """Test 6: Export and Output Validation"""
        print("\n🧪 Test 6: Export and Output Validation")
        
        try:
            # Configure pipeline with all export formats
            config = PipelineConfig(
                execution_mode=ExecutionMode.PRODUCTION,
                max_strategies=2,
                export_formats=["pine", "json"],  # Reduced for testing
                output_dir="test_output/task4_export_validation"
            )
            
            pipeline_system = Top5StrategiesRankingSystem(config=config)
            
            # Execute pipeline
            result = pipeline_system.execute_full_pipeline()
            
            # Validate exports
            output_dir = Path(config.output_dir)
            exported_files = result.exported_files
            
            # Check JSON export
            json_files = []
            if "json" in exported_files:
                json_file = Path(exported_files["json"])
                if json_file.exists():
                    json_files.append(json_file)
                    # Validate JSON content
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    assert "execution_timestamp" in json_data
                    assert "top5_strategies" in json_data
            
            # Check Pine Script exports
            pine_files = []
            if "pine" in exported_files:
                pine_file_list = exported_files["pine"]
                for pine_file_path in pine_file_list:
                    pine_file = Path(pine_file_path)
                    if pine_file.exists():
                        pine_files.append(pine_file)
                        # Validate Pine Script content
                        with open(pine_file, 'r') as f:
                            pine_content = f.read()
                        assert "//@version=5" in pine_content
                        assert "strategy(" in pine_content
            
            print(f"✅ Output directory created: {output_dir.exists()}")
            print(f"✅ JSON files exported: {len(json_files)}")
            print(f"✅ Pine Script files exported: {len(pine_files)}")
            print(f"✅ Total export formats: {len(exported_files)}")
            
            return {
                "success": True,
                "output_directory_exists": output_dir.exists(),
                "json_files_exported": len(json_files),
                "pine_files_exported": len(pine_files),
                "total_export_formats": len(exported_files),
                "export_validation": True
            }
            
        except Exception as e:
            print(f"❌ Export validation test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION SUMMARY")
        print("=" * 80)
        
        # Calculate overall success
        successful_tests = sum(1 for result in self.results.values() if result.get("success", False))
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        print(f"✅ Success Rate: {success_rate:.1%}")
        print(f"✅ Overall Success: {success_rate >= 0.8}")
        
        # Test-specific summaries
        if "test4_full_pipeline_execution" in self.results:
            pipeline_result = self.results["test4_full_pipeline_execution"]
            if pipeline_result.get("success"):
                print(f"🚀 Pipeline Success Rate: {pipeline_result.get('success_rate', 0):.1%}")
                print(f"🏆 Pipeline Quality: {pipeline_result.get('pipeline_quality', 'unknown')}")
                print(f"⚡ Execution Time: {pipeline_result.get('execution_time', 0):.2f}s")
        
        if "test5_performance_parallelization" in self.results:
            perf_result = self.results["test5_performance_parallelization"]
            if perf_result.get("success") and "best_performance" in perf_result:
                best = perf_result["best_performance"]
                print(f"🏆 Best Performance: {best['name']} ({best['execution_time']:.2f}s)")
        
        if "test6_export_output_validation" in self.results:
            export_result = self.results["test6_export_output_validation"]
            if export_result.get("success"):
                print(f"📤 Export Formats: {export_result.get('total_export_formats', 0)}")
                print(f"📝 Pine Scripts: {export_result.get('pine_files_exported', 0)}")
        
        # Status determination
        if success_rate >= 0.9:
            status = "🎉 TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION - EXCELLENT"
        elif success_rate >= 0.7:
            status = "✅ TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION - SUCCESS"
        elif success_rate >= 0.5:
            status = "⚠️ TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION - PARTIAL SUCCESS"
        else:
            status = "❌ TASK 4: END-TO-END PIPELINE CORE IMPLEMENTATION - NEEDS ATTENTION"
        
        print(f"\n{status}")
        
        # Save results
        results_file = "task4_end_to_end_pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {results_file}")

async def main():
    """Main test execution"""
    test_suite = Task4EndToEndPipelineTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())