#!/usr/bin/env python3
"""
Automatischer Backup-Scheduler für AI-Indicator-Optimizer
Führt regelmäßige Backups und Updates durch
"""

import schedule
import time
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup_scheduler.log'),
        logging.StreamHandler()
    ]
)

class AutoBackupScheduler:
    """Automatischer Backup-Scheduler"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(".")
        self.last_backup = None
        
    def quick_backup(self):
        """Schnelles Backup (nur State-Files)"""
        try:
            self.logger.info("🔄 Starting quick backup...")
            
            # Backup nur wichtige State-Files
            files_to_backup = [
                "project_state.json",
                "PROJECT_TRACKER.md"
            ]
            
            for file in files_to_backup:
                if os.path.exists(file):
                    subprocess.run(["git", "add", file], check=False)
            
            # Commit mit Timestamp
            timestamp = datetime.now().strftime("%H:%M")
            commit_msg = f"⏰ Auto-backup: {timestamp}"
            
            result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Push zu GitHub
                subprocess.run(["git", "push"], check=False)
                self.logger.info("✅ Quick backup completed")
                self.last_backup = datetime.now()
            else:
                self.logger.info("ℹ️ No changes to backup")
                
        except Exception as e:
            self.logger.error(f"❌ Quick backup failed: {e}")
    
    def full_backup(self):
        """Vollständiges Backup (alle Projekt-Files)"""
        try:
            self.logger.info("🚀 Starting full backup...")
            
            # Führe vollständiges Backup-Script aus
            result = subprocess.run(["python", "backup_to_github.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Full backup completed")
                self.last_backup = datetime.now()
            else:
                self.logger.error(f"❌ Full backup failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Full backup failed: {e}")
    
    def update_project_state(self):
        """Aktualisiert Projekt-State mit aktueller Zeit"""
        try:
            with open("project_state.json", 'r') as f:
                state = json.load(f)
            
            # Update last_updated
            state["project_info"]["last_updated"] = datetime.now().isoformat()
            
            with open("project_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info("📊 Project state updated")
            
        except Exception as e:
            self.logger.error(f"❌ State update failed: {e}")
    
    def health_check(self):
        """System-Health-Check"""
        try:
            self.logger.info("🔍 Running health check...")
            
            # Prüfe wichtige Dateien
            critical_files = [
                "PROJECT_SPECIFICATION.md",
                "PROJECT_TRACKER.md", 
                "project_state.json",
                "session_context.py"
            ]
            
            missing_files = []
            for file in critical_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                self.logger.warning(f"⚠️ Missing files: {missing_files}")
            else:
                self.logger.info("✅ All critical files present")
            
            # Prüfe Git-Status
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True)
            
            if result.stdout.strip():
                self.logger.info(f"📝 Uncommitted changes detected")
            else:
                self.logger.info("✅ Git repository clean")
                
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
    
    def setup_schedules(self):
        """Setup aller automatischen Schedules"""
        
        # Alle 15 Minuten: Quick Backup
        schedule.every(15).minutes.do(self.quick_backup)
        
        # Alle 2 Stunden: Project State Update
        schedule.every(2).hours.do(self.update_project_state)
        
        # Täglich um 9:00: Full Backup
        schedule.every().day.at("09:00").do(self.full_backup)
        
        # Täglich um 18:00: Health Check
        schedule.every().day.at("18:00").do(self.health_check)
        
        # Wöchentlich Sonntags: Full Backup + Health Check
        schedule.every().sunday.at("10:00").do(self.full_backup)
        schedule.every().sunday.at("10:05").do(self.health_check)
        
        self.logger.info("⏰ Backup schedules configured:")
        self.logger.info("   • Quick Backup: Every 15 minutes")
        self.logger.info("   • State Update: Every 2 hours") 
        self.logger.info("   • Full Backup: Daily at 9:00 AM")
        self.logger.info("   • Health Check: Daily at 6:00 PM")
        self.logger.info("   • Weekly Full: Sundays at 10:00 AM")
    
    def run_scheduler(self):
        """Startet den Scheduler (läuft kontinuierlich)"""
        self.logger.info("🚀 Starting Auto-Backup Scheduler...")
        self.setup_schedules()
        
        # Initial Health Check
        self.health_check()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scheduler error: {e}")

def run_daemon():
    """Läuft als Daemon im Hintergrund"""
    scheduler = AutoBackupScheduler()
    scheduler.run_scheduler()

def manual_backup():
    """Manuelles Backup für Testing"""
    scheduler = AutoBackupScheduler()
    scheduler.full_backup()

def status_check():
    """Status-Check für Testing"""
    scheduler = AutoBackupScheduler()
    scheduler.health_check()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "daemon":
            run_daemon()
        elif sys.argv[1] == "backup":
            manual_backup()
        elif sys.argv[1] == "status":
            status_check()
        else:
            print("Usage: python auto_backup_scheduler.py [daemon|backup|status]")
    else:
        print("🔄 AI-Indicator-Optimizer Auto-Backup Scheduler")
        print("Usage:")
        print("  python auto_backup_scheduler.py daemon   # Run continuous scheduler")
        print("  python auto_backup_scheduler.py backup   # Manual backup")
        print("  python auto_backup_scheduler.py status   # Health check")