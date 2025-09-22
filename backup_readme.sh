#!/bin/bash
# README Backup Script
DATE=$(date +%Y%m%d_%H%M%S)
cp README.md "README_backup_$DATE.md"
echo "✅ README backed up to README_backup_$DATE.md"