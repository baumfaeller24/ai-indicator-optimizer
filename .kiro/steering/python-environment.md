---
inclusion: always
---

# Python Environment Configuration

## WICHTIG: Virtuelle Umgebung

**IMMER die virtuelle Umgebung aktivieren vor Python-Befehlen:**

```bash
source test_env/bin/activate
```

## Standard Python-Befehle

Alle Python-Befehle müssen in der virtuellen Umgebung ausgeführt werden:

```bash
# ✅ RICHTIG:
source test_env/bin/activate && python script.py

# ❌ FALSCH:
python script.py
```

## Verfügbare Umgebungen

- **Hauptumgebung:** `test_env/` (Standard für AI-Indicator-Optimizer)
- **Alle Dependencies installiert:** PyTorch, Transformers, Nautilus, etc.

## Debugging

Falls Module fehlen:
1. Prüfe virtuelle Umgebung: `which python`
2. Aktiviere venv: `source test_env/bin/activate`
3. Prüfe Installation: `pip list | grep [module]`

**Diese Regel gilt für ALLE Python-Ausführungen in diesem Projekt!**