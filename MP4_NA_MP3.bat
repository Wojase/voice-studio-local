@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Przeciagnij plik MP4 lub folder na ten plik BAT.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m studio.media "%~1" --output "%~dp0converted"
pause
