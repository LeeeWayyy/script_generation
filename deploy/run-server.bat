@echo off
REM Simple launcher for the transcript API server on Windows (double-click or run from cmd).
REM For token generation, firewall, and LAN-IP hints, prefer run-server.ps1 instead.

if "%TRANSCRIPT_HOST%"=="" set TRANSCRIPT_HOST=0.0.0.0
if "%TRANSCRIPT_PORT%"=="" set TRANSCRIPT_PORT=8000
if "%TRANSCRIPT_MODEL%"=="" set TRANSCRIPT_MODEL=large-v3

if "%TRANSCRIPT_TOKEN%"=="" (
    echo WARNING: TRANSCRIPT_TOKEN not set - server will run WITHOUT authentication.
)
if "%HF_TOKEN%"=="" (
    echo WARNING: HF_TOKEN not set - speaker diarization will fail until you set it.
)

transcript-server --host %TRANSCRIPT_HOST% --port %TRANSCRIPT_PORT% --model %TRANSCRIPT_MODEL%
