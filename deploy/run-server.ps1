# Start the transcript API server on a Windows GPU host (e.g. the 5090 PC).
#
# Usage (PowerShell, from the project root):
#   .\deploy\run-server.ps1
#
# First run prints a TRANSCRIPT_TOKEN — copy it to your Mac. It also opens the
# firewall port for your local network so the Mac can reach the server.

$ErrorActionPreference = "Stop"

# --- config (override via environment variables) ---------------------------
$HostAddr = if ($env:TRANSCRIPT_HOST) { $env:TRANSCRIPT_HOST } else { "0.0.0.0" }
$Port     = if ($env:TRANSCRIPT_PORT) { $env:TRANSCRIPT_PORT } else { "8000" }
$Model    = if ($env:TRANSCRIPT_MODEL) { $env:TRANSCRIPT_MODEL } else { "large-v3" }

# --- auth token ------------------------------------------------------------
if (-not $env:TRANSCRIPT_TOKEN) {
    $TokenFile = if ($env:TRANSCRIPT_TOKEN_FILE) {
        $env:TRANSCRIPT_TOKEN_FILE
    } else {
        Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "transcript\server.token"
    }
    $TokenFile = [IO.Path]::GetFullPath($TokenFile)
    $TokenDir = Split-Path -Parent $TokenFile
    New-Item -ItemType Directory -Force -Path $TokenDir | Out-Null
    if (Test-Path $TokenFile) {
        $env:TRANSCRIPT_TOKEN = (Get-Content -Raw $TokenFile).Trim()
    } else {
        $env:TRANSCRIPT_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(24))"
        Set-Content -Path $TokenFile -Value $env:TRANSCRIPT_TOKEN -Encoding ASCII
        Write-Host "Saved TRANSCRIPT_TOKEN to $TokenFile" -ForegroundColor Yellow
        Write-Host "On your Mac, run:  export TRANSCRIPT_TOKEN=$($env:TRANSCRIPT_TOKEN)" -ForegroundColor Yellow
    }
    if (-not $env:TRANSCRIPT_TOKEN) {
        throw "Token file is empty: $TokenFile"
    }
    $CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe $TokenFile /inheritance:r /grant:r "${CurrentUser}:(F)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not restrict token file permissions: $TokenFile"
    }
}

# --- HF token reminder -----------------------------------------------------
if (-not $env:HF_TOKEN) {
    Write-Host "WARNING: HF_TOKEN not set - speaker diarization will fail until you set it." -ForegroundColor Red
}

# --- open the firewall for this port on Private networks (best effort) ------
$ruleName = "transcript-server-$Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    try {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $Port -Profile Private | Out-Null
        Write-Host "Opened firewall TCP $Port for Private networks." -ForegroundColor Green
    } catch {
        Write-Host "Could not add firewall rule (run as Administrator once if the Mac can't connect)." -ForegroundColor Yellow
    }
}

# --- show the LAN IP the Mac should target ---------------------------------
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -First 1).IPAddress
if ($ip) {
    Write-Host "Server will be reachable at:  http://${ip}:${Port}" -ForegroundColor Cyan
    Write-Host "On your Mac:  export TRANSCRIPT_SERVER=http://${ip}:${Port}" -ForegroundColor Cyan
}

Write-Host "Starting transcript-server (model=$Model) ..." -ForegroundColor Green
transcript-server --host $HostAddr --port $Port --model $Model
