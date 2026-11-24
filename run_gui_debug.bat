@echo off
cd /d "%~dp0"
python -m scrtsf.debug_app
if errorlevel 1 pause
