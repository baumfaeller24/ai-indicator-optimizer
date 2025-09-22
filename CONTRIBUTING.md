# 🤝 Contributing to AI-Indicator-Optimizer

Thank you for your interest in contributing to the AI-Indicator-Optimizer project! This document provides guidelines and information for contributors.

## 📋 **Table of Contents**

- [Project Overview](#project-overview)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Performance Standards](#performance-standards)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Pull Request Process](#pull-request-process)

---

## 🎯 **Project Overview**

The AI-Indicator-Optimizer is an enterprise-grade AI trading system that combines:
- **Multimodal AI Analysis** (Vision + Text) using MiniCPM-4.1-8B
- **Investment Bank Level Performance** (27,273 ticks/second)
- **Production-Ready Components** (TorchServe, Live Control, Enhanced Logging)
- **Complete End-to-End Pipeline** from tickdata to Pine Script strategies

### **Current Status**
- **Progress:** 21/30 tasks completed (70%)
- **Phase:** Baustein C2 - End-to-End Pipeline Integration
- **Performance:** World-class benchmarks achieved and maintained

---

## 💻 **Development Setup**

### **Hardware Requirements**
```
Minimum Requirements:
├── CPU: 16+ cores (AMD Ryzen 7 or Intel i7 equivalent)
├── GPU: NVIDIA RTX 3080+ (16GB+ VRAM)
├── RAM: 64GB+ DDR4/DDR5
└── Storage: 1TB+ NVMe SSD

Recommended (Development Hardware):
├── CPU: AMD Ryzen 9 9950X (32 cores)
├── GPU: NVIDIA RTX 5090 (33.7GB VRAM)
├── RAM: 182GB+ DDR5
└── Storage: Samsung 9100 PRO NVMe SSD
```

### **Software Requirements**
```bash
# Operating System
- Linux (Ubuntu 22.04+ recommended)
- Windows 11 (with WSL2)
- macOS 13+ (limited GPU support)

# Python Environment
- Python 3.11+
- CUDA 12.8+
- PyTorch 2.0+

# AI Infrastructure
- Ollama with MiniCPM-4.1-8B model
- TorchServe (optional for production features)
```

### **Installation Steps**

#### **1. Clone Repository**
```bash
git clone https://github.com/your-repo/ai-indicator-optimizer.git
cd ai-indicator-optimizer
```

#### **2. Setup Python Environment**
```bash
# Create virtual environment
python -m venv test_env

# Activate environment
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### **3. Install AI Infrastructure**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull MiniCPM-4.1-8B model
ollama pull minicpm-v:latest

# Verify installation
ollama list
```

#### **4. Verify Setup**
```bash
# Test hardware detection
python -c "from ai_indicator_optimizer.main_application import MainApplication; app = MainApplication(); app.detect_hardware()"

# Test AI integration
python demo_baustein_b3_working.py

# Run enhanced fine-tuning tests
python test_enhanced_fine_tuning_pipeline.py
```

---

## 📝 **Contributing Guidelines**

### **🎯 Contribution Areas**

#### **High Priority (Baustein C2)**
- **End-to-End Pipeline Implementation** (Tasks 4-12)
- **Multimodal Flow Integration** (Dynamic Fusion Agent)
- **Performance Optimization** (Hardware utilization)
- **Production Dashboard** (HTML/JSON/CSV export)

#### **Medium Priority**
- **Testing & Validation** (Integration tests, benchmarks)
- **Documentation** (API docs, tutorials, examples)
- **Code Quality** (Refactoring, optimization)
- **Error Handling** (Robustness, recovery systems)

#### **Low Priority**
- **Additional Features** (Multi-asset support, cloud deployment)
- **UI/UX Improvements** (Enhanced dashboard, visualizations)
- **Research & Development** (New AI models, algorithms)

### **🔄 Development Workflow**

#### **1. Issue Assignment**
- Browse [GitHub Issues](https://github.com/your-repo/ai-indicator-optimizer/issues)
- Comment on issues you'd like to work on
- Wait for assignment before starting work
- Create new issues for bugs or feature requests

#### **2. Branch Strategy**
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b bugfix/issue-description

# For documentation
git checkout -b docs/documentation-update
```

#### **3. Development Process**
- Follow [code standards](#code-standards)
- Write comprehensive tests
- Maintain performance benchmarks
- Update documentation
- Commit frequently with clear messages

#### **4. Testing Requirements**
```bash
# Run all tests before submitting
python -m pytest tests/ -v

# Run performance benchmarks
python test_enhanced_fine_tuning_pipeline.py

# Validate specific components
python demo_baustein_b3_working.py
```

---

## 🔧 **Code Standards**

### **📋 Python Style Guide**
- **PEP 8 Compliance:** Use `black` and `flake8` for formatting
- **Type Hints:** All functions must include type annotations
- **Docstrings:** Google-style docstrings for all public methods
- **Import Organization:** Use `isort` for import sorting

#### **Example Code Style**
```python
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import polars as pl

class ExampleComponent:
    """Example component following project standards.
    
    This class demonstrates the coding standards used throughout
    the AI-Indicator-Optimizer project.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def process_data(
        self, 
        data: pl.DataFrame, 
        parameters: Dict[str, float]
    ) -> Dict[str, Union[float, List[float]]]:
        """Process input data with given parameters.
        
        Args:
            data: Input DataFrame with OHLCV data
            parameters: Processing parameters
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if data.is_empty():
            raise ValueError("Input data cannot be empty")
            
        # Processing logic here
        results = {"processed_count": len(data)}
        
        self.logger.info(f"Processed {len(data)} records")
        return results
```

### **🏗️ Architecture Patterns**

#### **Dependency Injection**
```python
# Use dependency injection for testability
class DataProcessor:
    def __init__(self, connector: DataConnector, logger: Logger):
        self.connector = connector
        self.logger = logger
```

#### **Configuration Management**
```python
# Use structured configuration
@dataclass
class ProcessingConfig:
    batch_size: int = 1000
    timeout_seconds: int = 30
    enable_gpu: bool = True
```

#### **Error Handling**
```python
# Implement comprehensive error handling
try:
    result = self.process_data(data)
except DataProcessingError as e:
    self.logger.error(f"Processing failed: {e}")
    return self._get_fallback_result()
except Exception as e:
    self.logger.critical(f"Unexpected error: {e}")
    raise
```

---

## 🧪 **Testing Requirements**

### **📊 Test Coverage Standards**
- **Minimum Coverage:** 85% for new code
- **Critical Components:** 95% coverage required
- **Performance Tests:** All benchmarks must pass
- **Integration Tests:** End-to-end pipeline validation

### **🔬 Test Categories**

#### **Unit Tests**
```python
import pytest
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor

class TestEnhancedFeatureExtractor:
    """Test suite for EnhancedFeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create test extractor instance."""
        return EnhancedFeatureExtractor(
            enable_time_features=True,
            enable_technical_indicators=True
        )
    
    def test_feature_extraction(self, extractor, sample_bars):
        """Test basic feature extraction functionality."""
        features = extractor.extract_features(sample_bars)
        
        assert len(features) > 0
        assert 'rsi_14' in features.columns
        assert 'hour_sin' in features.columns
        
    def test_performance_benchmark(self, extractor, large_dataset):
        """Test performance meets benchmarks."""
        start_time = time.time()
        features = extractor.extract_features(large_dataset)
        processing_time = time.time() - start_time
        
        # Must process at least 50 bars/second
        bars_per_second = len(large_dataset) / processing_time
        assert bars_per_second >= 50
```

#### **Integration Tests**
```python
def test_end_to_end_pipeline():
    """Test complete pipeline from data to Pine Script."""
    # Setup test data
    test_data = generate_test_tickdata(1000)
    
    # Run pipeline
    pipeline = Top5StrategiesRankingSystem()
    results = pipeline.execute(test_data)
    
    # Validate results
    assert len(results['top_strategies']) == 5
    assert all(s['confidence'] > 0.5 for s in results['top_strategies'])
    assert results['pine_scripts'] is not None
```

#### **Performance Tests**
```python
@pytest.mark.performance
def test_tick_processing_benchmark():
    """Validate tick processing performance."""
    processor = TickDataProcessor()
    test_ticks = generate_test_ticks(100000)
    
    start_time = time.time()
    results = processor.process_ticks(test_ticks)
    processing_time = time.time() - start_time
    
    ticks_per_second = len(test_ticks) / processing_time
    
    # Must maintain investment bank level performance
    assert ticks_per_second >= 25000
```

---

## ⚡ **Performance Standards**

### **📈 Benchmark Requirements**

All contributions must maintain or improve existing performance benchmarks:

| Component | Minimum Performance | Target Performance |
|-----------|-------------------|-------------------|
| **Tick Processing** | 25,000 ticks/sec | 27,273+ ticks/sec |
| **Strategy Evaluation** | 100,000 evals/min | 130,123+ evals/min |
| **TorchServe Throughput** | 25,000 req/s | 30,933+ req/s |
| **Live Control Rate** | 500,000 ops/s | 551,882+ ops/s |
| **Hardware Utilization** | 90% efficiency | 95%+ efficiency |

### **🔧 Performance Testing**
```bash
# Run performance benchmarks
python test_enhanced_fine_tuning_pipeline.py

# Validate specific components
python -m ai_indicator_optimizer.benchmarks.tick_processing
python -m ai_indicator_optimizer.benchmarks.strategy_evaluation
python -m ai_indicator_optimizer.benchmarks.hardware_utilization
```

### **📊 Performance Monitoring**
- **Continuous Benchmarking:** All PRs must pass performance tests
- **Regression Detection:** Automatic alerts for performance degradation
- **Hardware Optimization:** Maintain 95%+ hardware utilization
- **Memory Efficiency:** Smart buffer management for large datasets

---

## 📚 **Documentation**

### **📝 Documentation Standards**

#### **Code Documentation**
- **Docstrings:** All public methods require comprehensive docstrings
- **Type Hints:** Complete type annotations for all functions
- **Comments:** Complex algorithms require inline comments
- **Examples:** Include usage examples in docstrings

#### **API Documentation**
```python
def process_multimodal_input(
    self,
    numerical_data: Dict[str, float],
    text_prompt: str,
    chart_image: Optional[np.ndarray] = None
) -> MultimodalResult:
    """Process multimodal input for trading analysis.
    
    Combines numerical indicators, text prompts, and optional chart images
    to generate comprehensive trading insights using MiniCPM-4.1-8B.
    
    Args:
        numerical_data: Dictionary of technical indicators and market data.
            Expected keys: 'rsi', 'macd', 'bb_position', 'volume_ratio'
        text_prompt: Natural language description of market conditions
            or specific analysis request
        chart_image: Optional candlestick chart as numpy array (H, W, 3)
    
    Returns:
        MultimodalResult containing:
            - confidence: Overall confidence score (0.0-1.0)
            - predictions: List of trading predictions
            - reasoning: AI-generated explanation
            - processing_time: Execution time in seconds
    
    Raises:
        ValueError: If numerical_data is empty or contains invalid values
        ModelError: If AI model inference fails
    
    Example:
        >>> agent = DynamicFusionAgent()
        >>> result = agent.process_multimodal_input(
        ...     numerical_data={'rsi': 65.2, 'macd': 0.15},
        ...     text_prompt="Analyze bullish momentum"
        ... )
        >>> print(f"Confidence: {result.confidence:.2f}")
        Confidence: 0.87
    """
```

#### **README Updates**
- Update performance metrics for new features
- Add usage examples for new components
- Update installation instructions if needed
- Maintain accurate project status

---

## 🐛 **Issue Reporting**

### **🔍 Bug Reports**

Use the following template for bug reports:

```markdown
## Bug Report

**Description:**
Brief description of the bug

**Environment:**
- OS: [Ubuntu 22.04, Windows 11, etc.]
- Python: [3.11.5]
- GPU: [RTX 5090, RTX 4090, etc.]
- CUDA: [12.8]

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Error Messages:**
```
Paste error messages here
```

**Performance Impact:**
- Tick processing: [X ticks/sec vs expected Y ticks/sec]
- Memory usage: [X GB vs expected Y GB]

**Additional Context:**
Any other relevant information
```

### **💡 Feature Requests**

Use the following template for feature requests:

```markdown
## Feature Request

**Feature Description:**
Clear description of the proposed feature

**Use Case:**
Why is this feature needed?

**Proposed Implementation:**
High-level implementation approach

**Performance Considerations:**
Expected impact on system performance

**Priority:**
- [ ] Critical (blocks current development)
- [ ] High (important for next release)
- [ ] Medium (nice to have)
- [ ] Low (future consideration)

**Additional Context:**
Any other relevant information
```

---

## 🔄 **Pull Request Process**

### **📋 PR Checklist**

Before submitting a pull request, ensure:

- [ ] **Code Quality**
  - [ ] Follows project coding standards
  - [ ] Includes comprehensive type hints
  - [ ] Has proper error handling
  - [ ] Passes all linting checks (`black`, `flake8`, `isort`)

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New tests added for new functionality
  - [ ] Performance benchmarks maintained
  - [ ] Integration tests updated if needed

- [ ] **Documentation**
  - [ ] Code is properly documented
  - [ ] README updated if needed
  - [ ] API documentation updated
  - [ ] Examples provided for new features

- [ ] **Performance**
  - [ ] Performance benchmarks pass
  - [ ] No regression in processing speed
  - [ ] Memory usage optimized
  - [ ] Hardware utilization maintained

### **📝 PR Template**

```markdown
## Pull Request

**Description:**
Brief description of changes

**Type of Change:**
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update

**Related Issues:**
Fixes #[issue_number]

**Testing:**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks pass
- [ ] Manual testing completed

**Performance Impact:**
- Tick processing: [X ticks/sec] (baseline: 27,273 ticks/sec)
- Memory usage: [X GB] (baseline: 15.3% of 182GB)
- Hardware utilization: [X%] (baseline: 95%+)

**Checklist:**
- [ ] Code follows project standards
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Performance validated

**Screenshots/Logs:**
If applicable, add screenshots or log outputs
```

### **🔍 Review Process**

1. **Automated Checks:** All CI/CD checks must pass
2. **Code Review:** At least one maintainer review required
3. **Performance Validation:** Benchmarks must be maintained
4. **Documentation Review:** Ensure documentation is complete
5. **Final Approval:** Maintainer approval required for merge

---

## 🏆 **Recognition**

### **🌟 Contributor Levels**

#### **Bronze Contributors**
- First successful PR merged
- Bug fixes and documentation improvements
- Recognition in CONTRIBUTORS.md

#### **Silver Contributors**
- 5+ PRs merged
- Feature implementations
- Performance improvements
- Mentoring new contributors

#### **Gold Contributors**
- 15+ PRs merged
- Major feature development
- Architecture improvements
- Code review responsibilities

#### **Platinum Contributors**
- 30+ PRs merged
- Project leadership
- Strategic direction input
- Maintainer privileges

### **🎉 Recognition Program**
- **Monthly Recognition:** Top contributors featured in project updates
- **Annual Awards:** Outstanding contributor recognition
- **Conference Opportunities:** Speaking opportunities at relevant conferences
- **Professional Network:** Connection with industry professionals

---

## 📞 **Getting Help**

### **💬 Communication Channels**
- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and discussions
- **Project Wiki:** Detailed documentation and guides
- **Code Reviews:** Learning through PR feedback

### **📚 Learning Resources**
- **Project Documentation:** Complete technical documentation
- **Code Examples:** Comprehensive examples in `/examples`
- **Performance Reports:** Detailed performance analysis
- **Architecture Guides:** System design documentation

### **🤝 Mentorship**
- **New Contributor Guide:** Step-by-step onboarding
- **Pair Programming:** Available for complex features
- **Code Review Learning:** Detailed feedback on submissions
- **Career Development:** Industry connections and opportunities

---

## 📄 **License**

By contributing to AI-Indicator-Optimizer, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the AI-Indicator-Optimizer project! Together, we're building the world's most advanced AI trading system with investment bank level performance.** 🚀

---

**📅 Last Updated:** September 22, 2025  
**🎯 Current Focus:** Baustein C2 - End-to-End Pipeline Integration  
**📊 Project Status:** 70% Complete (21/30 tasks)  
**🚀 Next Milestone:** Task 4 - End-to-End Pipeline Core Implementation