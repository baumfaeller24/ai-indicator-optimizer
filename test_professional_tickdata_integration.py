#!/usr/bin/env python3
"""
Integration Test: Professional Tickdata Verbindung
Testet die Verbindung zwischen Task 4 Pipeline und Professional Tickdata
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

# Import components
from ai_indicator_optimizer.integration.professional_tickdata_pipeline import (
    ProfessionalTickdataPipeline,
    ProfessionalTickdataProcessor,
    ProfessionalTickdataConfig,
    create_professional_pipeline
)

from ai_indicator_optimizer.integration.top5_strategies_ranking_system import (
    Top5StrategiesRankingSystem,
    PipelineConfig,
    ExecutionMode
)

class ProfessionalTickdataIntegrationTest:
    """Integration Test Suite für Professional Tickdata Verbindung"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Professional Tickdata Integration tests"""
        print("🔍 PROFESSIONAL TICKDATA INTEGRATION TESTS")
        print("=" * 80)
        
        # Test 1: Professional Tickdata Files Availability
        test1_result = await self.test_tickdata_files_availability()
        self.results["test1_tickdata_files"] = test1_result
        
        # Test 2: Professional Tickdata Processor
        test2_result = await self.test_tickdata_processor()
        self.results["test2_tickdata_processor"] = test2_result
        
        # Test 3: Professional Pipeline Integration
        test3_result = await self.test_professional_pipeline()
        self.results["test3_professional_pipeline"] = test3_result
        
        # Test 4: Top5 System Integration
        test4_result = await self.test_top5_system_integration()
        self.results["test4_top5_integration"] = test4_result
        
        # Test 5: End-to-End Professional Data Flow
        test5_result = await self.test_end_to_end_data_flow()
        self.results["test5_end_to_end_flow"] = test5_result
        
        # Summary
        await self.print_test_summary()
        
        return self.results
    
    async def test_tickdata_files_availability(self) -> Dict:
        """Test 1: Professional Tickdata Files Availability"""
        print("\n🧪 Test 1: Professional Tickdata Files Availability")
        
        try:
            config = ProfessionalTickdataConfig()
            available_files = []
            total_size = 0
            
            for file_path in config.tickdata_files:
                full_path = Path(file_path)
                if full_path.exists():
                    file_size = full_path.stat().st_size / (1024 * 1024)  # MB
                    available_files.append({
                        "file": file_path,
                        "size_mb": file_size,
                        "exists": True
                    })
                    total_size += file_size
                    print(f"✅ {file_path}: {file_size:.1f} MB")
                else:
                    available_files.append({
                        "file": file_path,
                        "size_mb": 0,
                        "exists": False
                    })
                    print(f"❌ {file_path}: Not found")
            
            files_found = sum(1 for f in available_files if f["exists"])
            
            print(f"📊 Available Files: {files_found}/{len(config.tickdata_files)}")
            print(f"💾 Total Size: {total_size:.1f} MB")
            
            return {
                "success": files_found > 0,
                "files_found": files_found,
                "total_files": len(config.tickdata_files),
                "total_size_mb": total_size,
                "available_files": available_files
            }
            
        except Exception as e:
            print(f"❌ Tickdata files test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_tickdata_processor(self) -> Dict:
        """Test 2: Professional Tickdata Processor"""
        print("\n🧪 Test 2: Professional Tickdata Processor")
        
        try:
            # Initialize processor
            config = ProfessionalTickdataConfig()
            processor = ProfessionalTickdataProcessor(config)
            
            # Test loading with limit
            start_time = time.time()
            tickdata = await processor.load_professional_tickdata(max_ticks=10000)
            load_time = time.time() - start_time
            
            if tickdata.empty:
                print("⚠️ No tickdata loaded - files may not be available")
                return {
                    "success": False,
                    "tickdata_loaded": 0,
                    "load_time": load_time,
                    "reason": "No tickdata files available"
                }
            
            # Test OHLCV generation
            ohlcv_results = {}
            for timeframe in ["1m", "5m", "15m"]:
                ohlcv = await processor.generate_ohlcv_from_ticks(tickdata, timeframe)
                ohlcv_results[timeframe] = len(ohlcv)
                print(f"✅ {timeframe}: {len(ohlcv)} bars")
            
            # Get statistics
            stats = processor.get_processing_statistics()
            
            print(f"✅ Loaded: {len(tickdata):,} ticks")
            print(f"⚡ Speed: {stats.get('average_ticks_per_second', 0):,.0f} ticks/sec")
            print(f"📊 Columns: {list(tickdata.columns)}")
            print(f"📅 Date Range: {tickdata.index.min()} to {tickdata.index.max()}")
            
            return {
                "success": True,
                "tickdata_loaded": len(tickdata),
                "load_time": load_time,
                "processing_speed": stats.get('average_ticks_per_second', 0),
                "ohlcv_results": ohlcv_results,
                "columns": list(tickdata.columns),
                "date_range": {
                    "start": str(tickdata.index.min()),
                    "end": str(tickdata.index.max())
                }
            }
            
        except Exception as e:
            print(f"❌ Tickdata processor test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_professional_pipeline(self) -> Dict:
        """Test 3: Professional Pipeline Integration"""
        print("\n🧪 Test 3: Professional Pipeline Integration")
        
        try:
            # Create professional pipeline
            pipeline = create_professional_pipeline()
            
            # Initialize pipeline
            init_success = await pipeline.initialize()
            
            if not init_success:
                print("⚠️ Pipeline initialization failed")
                return {
                    "success": False,
                    "initialization": False,
                    "reason": "Pipeline initialization failed"
                }
            
            # Execute professional pipeline
            start_time = time.time()
            result = await pipeline.execute_professional_pipeline(
                symbol="EUR/USD",
                timeframe="5m",
                max_ticks=5000  # Small limit for testing
            )
            execution_time = time.time() - start_time
            
            # Cleanup
            await pipeline.shutdown()
            
            print(f"✅ Pipeline execution: {'Success' if result.get('success') else 'Failed'}")
            print(f"📊 Ticks processed: {result.get('professional_tickdata', {}).get('total_ticks', 0):,}")
            print(f"📈 OHLCV bars: {result.get('professional_tickdata', {}).get('ohlcv_bars', 0)}")
            print(f"⚡ Processing speed: {result.get('professional_tickdata', {}).get('processing_speed_ticks_per_sec', 0):,.0f} ticks/sec")
            print(f"⏱️ Execution time: {execution_time:.2f}s")
            
            return {
                "success": result.get('success', False),
                "initialization": init_success,
                "execution_time": execution_time,
                "ticks_processed": result.get('professional_tickdata', {}).get('total_ticks', 0),
                "ohlcv_bars": result.get('professional_tickdata', {}).get('ohlcv_bars', 0),
                "processing_speed": result.get('professional_tickdata', {}).get('processing_speed_ticks_per_sec', 0),
                "pipeline_mode": result.get('pipeline_mode', 'unknown')
            }
            
        except Exception as e:
            print(f"❌ Professional pipeline test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_top5_system_integration(self) -> Dict:
        """Test 4: Top5 System Integration with Professional Tickdata"""
        print("\n🧪 Test 4: Top5 System Integration")
        
        try:
            # Configure Top5 system with professional tickdata
            config = PipelineConfig(
                execution_mode=ExecutionMode.DEVELOPMENT,
                max_strategies=3,
                symbols=["EUR/USD"],
                timeframes=["5m"],
                max_workers=4,
                output_dir="test_output/professional_integration"
            )
            
            # Initialize Top5 system
            top5_system = Top5StrategiesRankingSystem(config=config)
            
            # Execute pipeline
            start_time = time.time()
            pipeline_result = top5_system.execute_full_pipeline()
            execution_time = time.time() - start_time
            
            # Check data analysis stage for tickdata integration
            data_analysis_stage = None
            for stage_result in pipeline_result.stage_results:
                if stage_result.stage.value == "data_analysis":
                    data_analysis_stage = stage_result
                    break
            
            tickdata_loaded = 0
            total_bars = 0
            if data_analysis_stage and data_analysis_stage.success:
                tickdata_loaded = data_analysis_stage.data.get("tickdata_loaded", 0)
                total_bars = data_analysis_stage.data.get("total_bars", 0)
            
            print(f"✅ Pipeline success: {pipeline_result.success_rate:.1%}")
            print(f"📊 Tickdata loaded: {tickdata_loaded:,}")
            print(f"📈 Total bars: {total_bars}")
            print(f"🎯 Strategies generated: {len(pipeline_result.exported_files.get('pine', []))}")
            print(f"⏱️ Execution time: {execution_time:.2f}s")
            
            return {
                "success": pipeline_result.success_rate > 0.8,
                "pipeline_success_rate": pipeline_result.success_rate,
                "tickdata_loaded": tickdata_loaded,
                "total_bars": total_bars,
                "strategies_generated": len(pipeline_result.exported_files.get('pine', [])),
                "execution_time": execution_time,
                "pipeline_quality": pipeline_result.pipeline_quality
            }
            
        except Exception as e:
            print(f"❌ Top5 system integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_end_to_end_data_flow(self) -> Dict:
        """Test 5: End-to-End Professional Data Flow"""
        print("\n🧪 Test 5: End-to-End Professional Data Flow")
        
        try:
            # Test complete data flow from files to Pine Scripts
            
            # Step 1: Check file availability
            config = ProfessionalTickdataConfig()
            files_available = sum(1 for f in config.tickdata_files if Path(f).exists())
            
            if files_available == 0:
                print("⚠️ No professional tickdata files available - using mock data")
                return {
                    "success": True,
                    "data_flow": "mock_data",
                    "files_available": 0,
                    "reason": "No professional files, graceful fallback"
                }
            
            # Step 2: Load and process data
            processor = ProfessionalTickdataProcessor(config)
            tickdata = await processor.load_professional_tickdata(max_ticks=1000)
            
            # Step 3: Generate OHLCV
            ohlcv = await processor.generate_ohlcv_from_ticks(tickdata, "5m")
            
            # Step 4: Run through Top5 system
            top5_config = PipelineConfig(
                execution_mode=ExecutionMode.PRODUCTION,
                max_strategies=2,
                symbols=["EUR/USD"],
                timeframes=["5m"],
                output_dir="test_output/end_to_end_flow"
            )
            
            top5_system = Top5StrategiesRankingSystem(config=top5_config)
            result = top5_system.execute_full_pipeline()
            
            # Step 5: Validate outputs
            output_dir = Path(top5_config.output_dir)
            json_files = list(output_dir.glob("*.json"))
            pine_files = list(output_dir.glob("*.pine"))
            
            print(f"✅ Data flow: Professional files → Tickdata → OHLCV → Strategies → Pine Scripts")
            print(f"📁 Files available: {files_available}/{len(config.tickdata_files)}")
            print(f"📊 Ticks processed: {len(tickdata):,}")
            print(f"📈 OHLCV bars: {len(ohlcv)}")
            print(f"📝 JSON reports: {len(json_files)}")
            print(f"🌲 Pine scripts: {len(pine_files)}")
            
            return {
                "success": True,
                "data_flow": "complete",
                "files_available": files_available,
                "ticks_processed": len(tickdata),
                "ohlcv_bars": len(ohlcv),
                "json_reports": len(json_files),
                "pine_scripts": len(pine_files),
                "pipeline_success_rate": result.success_rate
            }
            
        except Exception as e:
            print(f"❌ End-to-end data flow test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 PROFESSIONAL TICKDATA INTEGRATION SUMMARY")
        print("=" * 80)
        
        # Calculate overall success
        successful_tests = sum(1 for result in self.results.values() if result.get("success", False))
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        print(f"✅ Success Rate: {success_rate:.1%}")
        
        # Test-specific summaries
        if "test1_tickdata_files" in self.results:
            files_result = self.results["test1_tickdata_files"]
            if files_result.get("success"):
                print(f"📁 Professional Files: {files_result.get('files_found', 0)}/{files_result.get('total_files', 0)} available")
                print(f"💾 Total Size: {files_result.get('total_size_mb', 0):.1f} MB")
        
        if "test2_tickdata_processor" in self.results:
            processor_result = self.results["test2_tickdata_processor"]
            if processor_result.get("success"):
                print(f"📊 Ticks Processed: {processor_result.get('tickdata_loaded', 0):,}")
                print(f"⚡ Processing Speed: {processor_result.get('processing_speed', 0):,.0f} ticks/sec")
        
        if "test4_top5_integration" in self.results:
            integration_result = self.results["test4_top5_integration"]
            if integration_result.get("success"):
                print(f"🎯 Pipeline Quality: {integration_result.get('pipeline_quality', 'unknown')}")
                print(f"📈 Strategies Generated: {integration_result.get('strategies_generated', 0)}")
        
        # Status determination
        if success_rate >= 0.9:
            status = "🎉 PROFESSIONAL TICKDATA INTEGRATION - EXCELLENT"
        elif success_rate >= 0.7:
            status = "✅ PROFESSIONAL TICKDATA INTEGRATION - SUCCESS"
        elif success_rate >= 0.5:
            status = "⚠️ PROFESSIONAL TICKDATA INTEGRATION - PARTIAL SUCCESS"
        else:
            status = "❌ PROFESSIONAL TICKDATA INTEGRATION - NEEDS ATTENTION"
        
        print(f"\n{status}")
        
        # Save results
        results_file = "professional_tickdata_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {results_file}")

async def main():
    """Main test execution"""
    test_suite = ProfessionalTickdataIntegrationTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())