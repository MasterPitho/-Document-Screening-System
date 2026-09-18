@echo off
cd /d "%~dp0"
title Document Screening Dashboard
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { Start-Process python -ArgumentList 'server.py' -WorkingDirectory '%~dp0' -WindowStyle Minimized; exit 1 }"
timeout /t 3 /nobreak >nul
start "" "http://localhost:3000/index.html"