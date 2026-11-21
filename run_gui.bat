@echo off
cd /d "%~dp0"
python -m scrtsf
if errorlevel 1 pause
