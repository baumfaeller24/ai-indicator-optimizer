#!/usr/bin/env python3
"""
Session Context Manager für AI-Indicator-Optimizer
Stellt sicher, dass der "rote Faden" nie verloren geht
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class ProjectContextManager:
    """Verwaltet Projekt-Kontext zwischen Chat-Sessions"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / "project_state.json"
        self.tracker_file = self.project_root / "PROJECT_TRACKER.md"
        self.spec_file = self.project_root / "PROJECT_SPECIFICATION.md"
        
    def load_current_state(self) -> Dict[str, Any]:
        """Lädt aktuellen Projekt-Status"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_state()
    
    def save_state(self, state: Dict[str, Any]):
        """Speichert aktuellen Projekt-Status"""
        state["project_info"]["last_updated"] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_session_summary(self) -> str:
        """Erstellt Session-Zusammenfassung für neue Chat-Sessions"""
        state = self.load_current_state()
        
        summary = f"""
# 🎯 AI-Indicator-Optimizer - Session Context

**Project:** {state['project_info']['name']} v{state['project_info']['version']}
**Architecture:** {state['project_info']['architecture']}
**Last Updated:** {state['project_info']['last_updated']}

## 📊 Current Status
- **Phase:** {state['project_info']['current_phase']}
- **Task:** {state['project_info']['current_task']}
- **Progress:** {state['project_info']['overall_progress']}% ({state['project_info']['completed_weeks']}/{state['project_info']['total_weeks']} weeks)

## 🖥️ Hardware Configuration
- **CPU:** {state['hardware_config']['cpu']}
- **GPU:** {state['hardware_config']['gpu']} 
- **RAM:** {state['hardware_config']['ram']}
- **CUDA:** {state['hardware_config']['cuda_version']}
- **PyTorch:** {state['hardware_config']['pytorch_version']}

## 📊 Data Sources Status
### Implemented:
"""
        
        for source in state['data_sources']['implemented']:
            summary += f"- **{source['name']}:** {source['status']} ({', '.join(source['markets'])})\n"
        
        summary += "\n### Planned:\n"
        for source in state['data_sources']['planned']:
            summary += f"- **{source['name']}:** {source['priority']} Priority ({', '.join(source['markets'])})\n"
        
        summary += f"""
## 🤖 AI Components
- **Model:** {state['ai_components']['model']}
- **Framework:** {state['ai_components']['framework']}
- **GPU Acceleration:** {'✅' if state['ai_components']['gpu_acceleration'] else '❌'}
- **Multimodal:** {'✅' if state['ai_components']['multimodal'] else '❌'}

## 📋 Next Actions
"""
        
        for action in state['next_actions']:
            summary += f"- [ ] {action}\n"
        
        summary += f"""
## 🎯 Key Decisions Made
"""
        
        for decision in state['key_decisions']:
            summary += f"- **{decision['date']}:** {decision['decision']} - {decision['rationale']}\n"
        
        return summary
    
    def update_task_progress(self, phase: str, task_id: str, progress: int, status: str = None):
        """Aktualisiert Task-Progress"""
        state = self.load_current_state()
        
        if phase in state['phase_status']:
            for task in state['phase_status'][phase]['tasks']:
                if task['id'] == task_id:
                    task['progress'] = progress
                    if status:
                        task['status'] = status
                    break
        
        self.save_state(state)
    
    def add_decision(self, decision: str, rationale: str, impact: str):
        """Fügt neue Entscheidung hinzu"""
        state = self.load_current_state()
        
        new_decision = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "decision": decision,
            "rationale": rationale, 
            "impact": impact
        }
        
        state['key_decisions'].append(new_decision)
        self.save_state(state)
    
    def get_current_focus(self) -> Dict[str, str]:
        """Gibt aktuellen Fokus zurück"""
        state = self.load_current_state()
        
        return {
            "phase": state['project_info']['current_phase'],
            "task": state['project_info']['current_task'],
            "progress": f"{state['project_info']['overall_progress']}%",
            "next_milestone": self._get_next_milestone(state)
        }
    
    def _get_next_milestone(self, state: Dict[str, Any]) -> str:
        """Ermittelt nächsten Meilenstein"""
        current_phase = state['project_info']['current_phase']
        
        if "Phase 1" in current_phase:
            return "Nautilus Foundation Complete (Week 4)"
        elif "Phase 2" in current_phase:
            return "AI Engine Integration (Week 12)"
        elif "Phase 3" in current_phase:
            return "Production System Ready (Week 18)"
        else:
            return "Enterprise Features Complete (Week 22)"
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Erstellt Default-State falls Datei nicht existiert"""
        return {
            "project_info": {
                "name": "AI-Indicator-Optimizer",
                "version": "3.0",
                "architecture": "Nautilus-First",
                "current_phase": "Phase 1 - Nautilus Foundation",
                "current_task": "1.1 - Nautilus Core Setup",
                "overall_progress": 0
            }
        }

def print_session_context():
    """Druckt Session-Kontext für neue Chat-Sessions"""
    manager = ProjectContextManager()
    print(manager.get_session_summary())

def update_progress(phase: str, task_id: str, progress: int, status: str = None):
    """CLI für Progress-Updates"""
    manager = ProjectContextManager()
    manager.update_task_progress(phase, task_id, progress, status)
    print(f"✅ Updated {phase} Task {task_id}: {progress}% ({status})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "context":
            print_session_context()
        elif sys.argv[1] == "update" and len(sys.argv) >= 5:
            phase = sys.argv[2]
            task_id = sys.argv[3] 
            progress = int(sys.argv[4])
            status = sys.argv[5] if len(sys.argv) > 5 else None
            update_progress(phase, task_id, progress, status)
    else:
        print_session_context()