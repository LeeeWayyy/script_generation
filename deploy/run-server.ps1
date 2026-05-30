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
    $env:TRANSCRIPT_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(24))"
    Write-Host "Generated TRANSCRIPT_TOKEN = $($env:TRANSCRIPT_TOKEN)" -ForegroundColor Yellow
    Write-Host "On your Mac, run:  export TRANSCRIPT_TOKEN=$($env:TRANSCRIPT_TOKEN)" -ForegroundColor Yellow
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
