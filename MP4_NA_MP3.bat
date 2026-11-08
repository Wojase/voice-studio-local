@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag an MP4 file or folder onto this BAT file.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m studio.media "%~1" --output "%~dp0converted"
pause
