# ======================================================================
#  GraphRAG++  Quick-Start
#  Works from any working directory — can be pinned as a shortcut or
#  run via: powershell -File "C:\...\GraphRAG\start.ps1"
# ======================================================================

# ── Locate the project root no matter how the script was invoked ──────
# $PSScriptRoot is reliable for direct .ps1 invocations (PS 3+).
# $MyInvocation.MyCommand.Path covers edge cases (dot-source, shortcuts).
if ($PSScriptRoot) {
    $Root = $PSScriptRoot
} elseif ($MyInvocation.MyCommand.Path) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    # Last resort: ask the OS where this script lives
    $Root = Split-Path -Parent (
        (Get-Item $MyInvocation.MyCommand.Definition -ErrorAction SilentlyContinue).FullName
    )
}

if (-not $Root -or -not (Test-Path $Root)) {
    Write-Host "[ERROR] Cannot determine project root. Run the script directly." -ForegroundColor Red
    exit 1
}

# ── Config ─────────────────────────────────────────────────────────────
$ModelFile    = "graphrag-plus-plus-qwen35-4b-q3_k_m-fixed.gguf"
$LlamaExe     = "llama-b9066\llama-server.exe"
$LlamaPort    = 8080
$BackendPort  = 8000
$FrontendPort = 3000

# ── Helpers ────────────────────────────────────────────────────────────
function Write-Step([int]$n, [string]$msg) {
    Write-Host ""
    Write-Host "  [$n/3] $msg" -ForegroundColor Cyan
}
function Write-Ok([string]$msg)   { Write-Host "        OK  : $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "        WARN: $msg" -ForegroundColor Yellow }

function Wait-Port([int]$port, [int]$timeoutSec = 30) {
    $deadline = [DateTime]::Now.AddSeconds($timeoutSec)
    while ([DateTime]::Now -lt $deadline) {
        try {
            $tcp = [System.Net.Sockets.TcpClient]::new()
            $tcp.Connect("127.0.0.1", $port)
            $tcp.Close()
            return $true
        } catch {}
        Start-Sleep -Milliseconds 400
    }
    return $false
}

function Clear-Port([int]$port) {
    $pids = (netstat -ano 2>$null |
        Select-String ":$port\s" |
        ForEach-Object { ($_ -split '\s+')[-1] } |
        Where-Object { $_ -match '^\d+$' } |
        Sort-Object -Unique)
    foreach ($p in $pids) {
        try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {}
    }
}

# ── Banner ─────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ================================================" -ForegroundColor DarkCyan
Write-Host "          GraphRAG++  Quick-Start               " -ForegroundColor Cyan
Write-Host "          Local AMD/Vulkan GPU Inference        " -ForegroundColor DarkCyan
Write-Host "  ================================================" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Project: $Root" -ForegroundColor DarkGray
Write-Host ""

# ── Pre-flight checks ──────────────────────────────────────────────────
$LlamaPath = Join-Path $Root $LlamaExe
$ModelPath  = Join-Path $Root $ModelFile
$hasLlama   = Test-Path $LlamaPath
$hasModel   = Test-Path $ModelPath

if (-not $hasLlama) {
    Write-Warn "llama-server not found — LLM step will be skipped."
    Write-Host "          Expected: $LlamaPath" -ForegroundColor DarkGray
}
if (-not $hasModel) {
    Write-Warn "Model file not found — LLM step will be skipped."
    Write-Host "          Expected: $ModelPath" -ForegroundColor DarkGray
}

# ── Clear stale port occupants ──────────────────────────────────────────
Write-Host "  Clearing stale processes on ports $LlamaPort, $BackendPort, $FrontendPort..." -ForegroundColor DarkGray
Clear-Port $LlamaPort
Clear-Port $BackendPort
Clear-Port $FrontendPort
Start-Sleep -Milliseconds 600

# ── Step 1: llama-server ───────────────────────────────────────────────
Write-Step 1 "Starting llama-server (Vulkan GPU)..."

if ($hasLlama -and $hasModel) {
    $LlamaArgs = @(
        "-m", $ModelPath,
        "--port", $LlamaPort,
        "--host", "127.0.0.1",
        "--n-gpu-layers", "99",
        "--ctx-size", "4096",
        "--threads", "6",
        "--log-disable"
    )
    $llamaProc = Start-Process -FilePath $LlamaPath `
        -ArgumentList $LlamaArgs `
        -WorkingDirectory $Root `
        -PassThru -WindowStyle Minimized

    Write-Host "        Waiting for llama-server on :$LlamaPort..." -ForegroundColor DarkGray
    if (Wait-Port $LlamaPort 30) {
        Write-Ok "llama-server ready  http://127.0.0.1:$LlamaPort  (PID $($llamaProc.Id))"
        $env:LOCAL_LLAMA_CPP_URL = "http://127.0.0.1:$LlamaPort"
    } else {
        Write-Warn "llama-server did not respond in 30 s — check the minimised window."
    }
} else {
    Write-Warn "Skipping llama-server (missing binary or model). Backend will use HF API fallback."
}

# ── Step 2: Python backend ─────────────────────────────────────────────
Write-Step 2 "Starting FastAPI backend..."

$BackendDir = Join-Path $Root "backend"
$BackendCmd = "Set-Location '$BackendDir'; python main.py"
Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $BackendCmd `
    -WindowStyle Normal

Write-Host "        Waiting for backend on :$BackendPort..." -ForegroundColor DarkGray
if (Wait-Port $BackendPort 30) {
    Write-Ok "Backend ready  http://localhost:$BackendPort"
} else {
    Write-Warn "Backend did not respond in 30 s — check the backend window."
}

# ── Step 3: Next.js frontend ───────────────────────────────────────────
Write-Step 3 "Starting Next.js frontend..."

$FrontendDir = Join-Path $Root "frontend"
$FrontendCmd = "Set-Location '$FrontendDir'; npm run dev"
Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $FrontendCmd `
    -WindowStyle Normal

Write-Host "        Waiting for frontend on :$FrontendPort..." -ForegroundColor DarkGray
if (Wait-Port $FrontendPort 45) {
    Write-Ok "Frontend ready  http://localhost:$FrontendPort"
} else {
    Write-Warn "Frontend did not respond in 45 s — check the frontend window."
}

# ── Done ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "     GraphRAG++ is running                       " -ForegroundColor Green
Write-Host "                                                  " -ForegroundColor Green
Write-Host "     UI      ->  http://localhost:$FrontendPort          " -ForegroundColor Green
Write-Host "     API     ->  http://localhost:$BackendPort          " -ForegroundColor Green
Write-Host "     LLM     ->  http://localhost:$LlamaPort         " -ForegroundColor Green
Write-Host "                                                  " -ForegroundColor Green
Write-Host "     Press Ctrl+C in any window to stop a service" -ForegroundColor DarkGray
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""

Start-Sleep -Seconds 2
Start-Process "http://localhost:$FrontendPort"
