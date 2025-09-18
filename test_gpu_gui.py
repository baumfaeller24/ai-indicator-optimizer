#!/usr/bin/env python3
"""
Simple GPU Test GUI to verify hardware detection
"""

import streamlit as st
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_indicator_optimizer.core.hardware_detector import HardwareDetector
from ai_indicator_optimizer.core.resource_manager import ResourceManager

st.set_page_config(
    page_title="GPU Test - AI Indicator Optimizer",
    page_icon="🎮",
    layout="wide"
)

@st.cache_resource
def get_hardware_info():
    """Hardware-Informationen laden"""
    detector = HardwareDetector()
    resource_manager = ResourceManager(detector)
    return detector, resource_manager

def main():
    st.title("🎮 GPU Test - AI Indicator Optimizer")
    st.markdown("---")
    
    # Hardware Status
    st.header("🖥️ Hardware Status")
    
    try:
        detector, resource_manager = get_hardware_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💻 CPU Information")
            if detector.cpu_info:
                st.success(f"**Model:** {detector.cpu_info.model}")
                st.info(f"**Cores:** {detector.cpu_info.cores_physical} physical, {detector.cpu_info.cores_logical} logical")
                st.info(f"**Frequency:** {detector.cpu_info.frequency_current:.0f} MHz (Max: {detector.cpu_info.frequency_max:.0f} MHz)")
                if detector.cpu_info.cache_l3:
                    st.info(f"**L3 Cache:** {detector.cpu_info.cache_l3 // (1024*1024)} MB")
            else:
                st.error("CPU information not available")
        
        with col2:
            st.subheader("🎮 GPU Information")
            if detector.gpu_info and len(detector.gpu_info) > 0:
                for i, gpu in enumerate(detector.gpu_info):
                    st.success(f"**GPU {i}:** {gpu.name}")
                    st.info(f"**VRAM:** {gpu.memory_total // (1024**3)} GB")
                    st.info(f"**Compute Capability:** {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
                    if gpu.cuda_cores:
                        st.info(f"**CUDA Cores:** {gpu.cuda_cores:,}")
                    if gpu.tensor_cores:
                        st.info(f"**Tensor Cores:** {gpu.tensor_cores}")
            else:
                st.error("❌ No GPU detected")
        
        # Memory Information
        st.subheader("💾 Memory Information")
        if detector.memory_info:
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Total RAM", f"{detector.memory_info.total // (1024**3)} GB")
            with col4:
                st.metric("Available RAM", f"{detector.memory_info.available // (1024**3)} GB")
            
            if detector.memory_info.frequency:
                st.info(f"**RAM Frequency:** {detector.memory_info.frequency} MHz")
            if detector.memory_info.type:
                st.info(f"**RAM Type:** {detector.memory_info.type}")
        
        # Target Hardware Check
        st.subheader("🎯 Target Hardware Status")
        checks = detector.is_target_hardware()
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if checks['ryzen_9950x']:
                st.success("✅ Ryzen 9 9950X")
            else:
                st.error("❌ Ryzen 9 9950X")
        
        with col6:
            if checks['rtx_5090']:
                st.success("✅ RTX 5090")
            else:
                st.error("❌ RTX 5090")
        
        with col7:
            if checks['ram_192gb']:
                st.success("✅ 192GB RAM")
            else:
                st.error("❌ 192GB RAM")
        
        with col8:
            if checks['samsung_9100_pro']:
                st.success("✅ Samsung 9100 PRO")
            else:
                st.error("❌ Samsung 9100 PRO")
        
        # PyTorch CUDA Test
        st.subheader("🔥 PyTorch CUDA Test")
        try:
            import torch
            
            col9, col10, col11 = st.columns(3)
            
            with col9:
                st.metric("PyTorch Version", torch.__version__)
            
            with col10:
                if torch.cuda.is_available():
                    st.success("✅ CUDA Available")
                else:
                    st.error("❌ CUDA Not Available")
            
            with col11:
                if torch.cuda.is_available():
                    st.metric("CUDA Version", torch.version.cuda)
                else:
                    st.error("No CUDA")
            
            if torch.cuda.is_available():
                st.info(f"**GPU Count:** {torch.cuda.device_count()}")
                st.info(f"**GPU Name:** {torch.cuda.get_device_name(0)}")
                st.info(f"**Supported Architectures:** {', '.join(torch.cuda.get_arch_list())}")
                
                # Test GPU computation
                try:
                    x = torch.randn(1000, 1000).cuda()
                    y = torch.mm(x, x.t())
                    st.success("✅ GPU computation test passed!")
                except Exception as e:
                    st.error(f"❌ GPU computation test failed: {e}")
            
        except ImportError:
            st.error("PyTorch not installed")
        except Exception as e:
            st.error(f"PyTorch test failed: {e}")
        
        # Resource Manager Info
        st.subheader("⚡ Resource Manager")
        workers = detector.get_optimal_worker_counts()
        
        worker_cols = st.columns(len(workers))
        for i, (task, count) in enumerate(workers.items()):
            with worker_cols[i]:
                st.metric(task.replace('_', ' ').title(), f"{count} workers")
        
        if hasattr(resource_manager, 'gpu_devices') and resource_manager.gpu_devices:
            st.success(f"**Primary GPU:** {resource_manager.gpu_devices[0]}")
            st.info(f"**GPU Memory Fraction:** {resource_manager.gpu_memory_fraction}")
        else:
            st.warning("GPU devices not configured in resource manager")
        
    except Exception as e:
        st.error(f"Hardware detection failed: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()