#!/usr/bin/env python3
"""
Smart Backup Manager - Intelligente Backup-Koordination
Verhindert Konflikte zwischen verschiedenen Backup-Methoden
"""

import os
import json
import fcntl
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

class SmartBackupManager:
    """Intelligenter Backup-Manager mit Lock-System"""
    
    def __init__(self):
        self.lock_file = Path("backup.lock")
        self.state_file = Path("project_state.json")
        self.logger = logging.getLogger(__name__)
        
    def acquire_lock(self, timeout: int = 30) -> bool:
        """Erwirbt exklusiven Backup-Lock"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            
            # Non-blocking lock attempt
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Schreibe Lock-Info
            lock_info = {
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat(),
                "process": "smart_backup_manager"
            }
            
            json.dump(lock_info, self.lock_fd, indent=2)
            self.lock_fd.flush()
            
            return True
            
        except (IOError, OSError):
            # Lock bereits vergeben
            return False
    
    def release_lock(self):
        """Gibt Backup-Lock frei"""
        try:
            if hasattr(self, 'lock_fd'):
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                
            if self.lock_file.exists():
                self.lock_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Lock release failed: {e}")
    
    def is_backup_needed(self) -> bool:
        """Prüft ob Backup notwendig ist"""
        try:
            if not self.state_file.exists():
                return True
            
            # Prüfe letzte Änderung
            last_modified = datetime.fromtimestamp(self.state_file.stat().st_mtime)
            
            # Backup nur wenn Änderungen in letzten 10 Minuten
            if datetime.now() - last_modified < timedelta(minutes=10):
                return True
            
            # Prüfe Git-Status
            import subprocess
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True)
            
            return bool(result.stdout.strip())
            
        except Exception:
            return True  # Im Zweifel: Backup machen
    
    def smart_backup(self, backup_type: str = "auto") -> bool:
        """Intelligentes Backup mit Lock-System"""
        
        # Prüfe ob Backup notwendig
        if not self.is_backup_needed():
            self.logger.info("ℹ️ No backup needed - no recent changes")
            return True
        
        # Versuche Lock zu erwerben
        if not self.acquire_lock():
            self.logger.info("⏳ Another backup process running - skipping")
            return False
        
        try:
            self.logger.info(f"🔄 Starting {backup_type} backup...")
            
            # Führe Backup durch
            import subprocess
            result = subprocess.run(["python", "backup_to_github.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Smart backup completed")
                return True
            else:
                self.logger.error(f"❌ Backup failed: {result.stderr}")
                return False
                
        finally:
            self.release_lock()

def safe_backup():
    """Sichere Backup-Funktion für Cron/Daemon"""
    manager = SmartBackupManager()
    return manager.smart_backup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    safe_backup()