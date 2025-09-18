#!/bin/bash

# 🚀 GitHub Repository Setup Commands
# Führe diese Befehle aus, nachdem du das GitHub Repository erstellt hast

echo "🚀 Setting up GitHub Repository..."

# 1. Setze main als Standard-Branch
git branch -M main

# 2. Füge GitHub Remote hinzu (ERSETZE USERNAME mit deinem GitHub Username!)
echo "📡 Adding GitHub remote..."
echo "⚠️  WICHTIG: Ersetze 'USERNAME' mit deinem echten GitHub Username!"
read -p "Dein GitHub Username: " username

if [ -z "$username" ]; then
    echo "❌ Username ist erforderlich!"
    exit 1
fi

git remote add origin https://github.com/$username/ai-indicator-optimizer.git

# 3. Push zum GitHub Repository
echo "📤 Pushing to GitHub..."
git push -u origin main

# 4. Überprüfe Status
echo "✅ Repository Setup Complete!"
echo "🔗 Repository URL: https://github.com/$username/ai-indicator-optimizer"
echo "🎯 GitHub Actions werden automatisch gestartet"

# 5. Zeige Repository-Statistiken
echo ""
echo "📊 Repository Statistics:"
git log --oneline | wc -l | xargs echo "Commits:"
git ls-files | wc -l | xargs echo "Files:"
git ls-files | xargs wc -l | tail -1 | awk '{print "Lines of Code: " $1}'

echo ""
echo "🎉 GitHub Repository ist bereit!"
echo "💡 Nächste Schritte:"
echo "   1. Gehe zu https://github.com/$username/ai-indicator-optimizer"
echo "   2. Überprüfe dass alle Dateien hochgeladen wurden"
echo "   3. Schaue dir die GitHub Actions an (Tests laufen automatisch)"
echo "   4. Teile den Repository-Link!"