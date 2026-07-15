@echo off
REM Double-click wrapper; the PowerShell launcher owns secure token persistence.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-server.ps1"
