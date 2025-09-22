---
name: 🐛 Bug Report
about: Create a report to help us improve the AI-Indicator-Optimizer
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Report

**📋 Description**
A clear and concise description of what the bug is.

**🖥️ Environment**
- **OS:** [Ubuntu 22.04, Windows 11, macOS 13, etc.]
- **Python:** [3.11.5, 3.12.0, etc.]
- **GPU:** [RTX 5090, RTX 4090, RTX 3080, CPU-only, etc.]
- **CUDA:** [12.8, 12.1, N/A, etc.]
- **RAM:** [182GB, 64GB, 32GB, etc.]
- **Project Version:** [v2.0.0, main branch, commit hash]

**🔄 Steps to Reproduce**
1. Step one
2. Step two
3. Step three
4. See error

**✅ Expected Behavior**
A clear and concise description of what you expected to happen.

**❌ Actual Behavior**
A clear and concise description of what actually happened.

**📊 Performance Impact**
- **Tick Processing:** [X ticks/sec vs expected 27,273+ ticks/sec]
- **Memory Usage:** [X GB vs expected <15% of available RAM]
- **GPU Utilization:** [X% vs expected 95%+]
- **Processing Time:** [X seconds vs expected Y seconds]

**🔍 Error Messages**
```
Paste complete error messages and stack traces here
```

**📝 Logs**
```
Paste relevant log entries here (remove sensitive information)
```

**🧪 Minimal Reproducible Example**
```python
# Provide minimal code that reproduces the issue
from ai_indicator_optimizer.main_application import MainApplication

app = MainApplication()
# Steps that cause the bug...
```

**🔧 Attempted Solutions**
- [ ] Restarted the application
- [ ] Cleared cache/temporary files
- [ ] Reinstalled dependencies
- [ ] Checked hardware requirements
- [ ] Reviewed documentation

**📸 Screenshots/Charts**
If applicable, add screenshots or performance charts to help explain the problem.

**🎯 Component Affected**
- [ ] Data Processing (DukascopyConnector, IndicatorCalculator)
- [ ] AI Integration (MiniCPM-4.1-8B, Ollama, TorchServe)
- [ ] Pattern Recognition (VisualPatternAnalyzer, PatternValidator)
- [ ] Strategy Evaluation (AIStrategyEvaluator, Top-5-Ranking)
- [ ] Production Components (Live Control, Enhanced Logging)
- [ ] Main Application (CLI, Configuration)
- [ ] Documentation/Installation
- [ ] Other: ___________

**📋 Additional Context**
Add any other context about the problem here.

**🚨 Severity**
- [ ] **Critical** - System crashes, data loss, security issue
- [ ] **High** - Major functionality broken, significant performance degradation
- [ ] **Medium** - Feature not working as expected, minor performance impact
- [ ] **Low** - Cosmetic issue, documentation error, enhancement request

---

**📞 For urgent issues affecting production systems, please also contact the maintainers directly.**