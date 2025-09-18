#!/bin/bash
# Setup Cron-Jobs für automatische Backups

PROJECT_DIR="$(pwd)"
PYTHON_PATH="/usr/bin/python3"

echo "🔄 Setting up Cron-Jobs for AI-Indicator-Optimizer Auto-Backup"
echo "Project Directory: $PROJECT_DIR"

# Erstelle Cron-Jobs
(crontab -l 2>/dev/null; cat << EOF

# AI-Indicator-Optimizer Auto-Backup Jobs
# Smart backup every 15 minutes (only if changes detected)
*/15 * * * * cd "$PROJECT_DIR" && $PYTHON_PATH smart_backup_manager.py >> backup_cron.log 2>&1

# Health check daily at 18:00
0 18 * * * cd "$PROJECT_DIR" && $PYTHON_PATH auto_backup_scheduler.py status >> backup_cron.log 2>&1

# Session context update every hour (for chat continuity)
0 * * * * cd "$PROJECT_DIR" && $PYTHON_PATH session_context.py context > last_session_context.md 2>&1

EOF
) | crontab -

echo "✅ Cron-Jobs installed successfully!"
echo ""
echo "📋 Installed schedules:"
echo "   • Quick Backup: Every 15 minutes"
echo "   • Health Check: Daily at 6:00 PM"
echo "   • Full Backup: Daily at 9:00 AM"
echo "   • Weekly Backup: Sundays at 10:00 AM"
echo ""
echo "📝 Logs will be written to: backup_cron.log"
echo ""
echo "🔍 To view current cron jobs: crontab -l"
echo "🗑️ To remove cron jobs: crontab -r"