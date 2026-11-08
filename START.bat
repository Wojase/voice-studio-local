@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -X utf8 app.py --open
if errorlevel 1 pause
