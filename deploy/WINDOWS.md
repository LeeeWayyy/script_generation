# Running the server on a Windows GPU host (RTX 5090)

This is the setup for **5090 PC = server, Mac = client, same LAN**.

## 1. One-time setup on the Windows PC

Install the prerequisites:

```powershell
# ffmpeg (via winget or choco), then verify
winget install Gyan.FFmpeg
ffmpeg -version

# Python 3.10+ from python.org if you don't have it
```

Create a venv and install with the GPU PyTorch wheel + server extra:

```powershell
cd C:\path\to\transcript
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Blackwell (5090, sm_120) needs a recent CUDA wheel:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[server]"
```

Set your Hugging Face token (needed for speaker labels). To persist it across
reboots:

```powershell
setx HF_TOKEN "hf_xxxxx"
# (reopen the terminal so setx takes effect)
```

For extraction OCR, explicitly allow PaddleOCR's first download on a trusted
connected host, or point at a pinned cache containing `det`, `rec`, and `cls`
subdirectories:

```powershell
setx TRANSCRIPT_OCR_ALLOW_DOWNLOAD "1"
# offline/pinned alternative:
# setx TRANSCRIPT_OCR_MODEL_DIR "D:\transcript-data\ocr"

# optional durable bundle location (default is the user's cache directory)
setx TRANSCRIPT_DATA_DIR "D:\transcript-data\extractions"
```

See [`docs/DOCUMENTATION.md`](../docs/DOCUMENTATION.md) for OCR metadata, bundle
TTL/cleanup, and every server setting.

## 2. Start the server

```powershell
.\deploy\run-server.ps1
```

On first run it will:
- generate a `TRANSCRIPT_TOKEN`, save it owner-only under Local AppData, and
  print it once (copy it to the Mac),
- open the Windows Firewall for the port on **Private** networks,
- print the exact `http://<lan-ip>:8000` URL to use from the Mac.

> If the Mac still can't connect: make sure your LAN is set to a **Private**
> network in Windows (not Public), and that you ran the script in an elevated
> (Administrator) PowerShell at least once so the firewall rule could be added.

## 3. Keep it running (optional)

**Easiest:** just leave the PowerShell window open.

**As a background Windows service** (auto-start on boot) using
[NSSM](https://nssm.cc/):

```powershell
# After downloading nssm.exe:
nssm install transcript-server "C:\path\to\transcript\.venv\Scripts\transcript-server.exe" "--host 0.0.0.0 --port 8000 --model large-v3"
nssm set transcript-server AppEnvironmentExtra HF_TOKEN=hf_xxx TRANSCRIPT_TOKEN=your_token
nssm start transcript-server
# logs: nssm set ... AppStdout / AppStderr to files, or use Event Viewer
```

**Or Task Scheduler:** create a task "At startup" that runs
`powershell -File C:\path\to\transcript\deploy\run-server.ps1`, set to run
whether logged in or not, with HF_TOKEN/TRANSCRIPT_TOKEN configured as system
environment variables.

## 4. Use it from the Mac

On the Mac install this project's client extra — no torch or WhisperX:

```bash
git clone <this-repository-url>
cd transcript
pip install -e ".[client]"

export TRANSCRIPT_SERVER=http://<windows-lan-ip>:8000
export TRANSCRIPT_TOKEN=<token-from-step-2>

transcript-remote meeting.mp4 -f srt -o meeting.srt
transcript-remote "https://youtube.com/watch?v=..." -f txt
extract-remote slides.zip --kind image_note --out-dir ./extractions
```

The console scripts come from this package; installing `requests` alone does not
install `transcript-remote` or `extract-remote`. Override the persisted token
location with `TRANSCRIPT_TOKEN_FILE` if needed.

Test connectivity first:

```bash
curl http://<windows-lan-ip>:8000/health
```
