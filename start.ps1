# ======================================================================
#             GraphRAG++ Quick-Start Script                             
#             Starts: llama-server -> backend -> frontend                
# ======================================================================

$Root       = $PSScriptRoot
$ModelFile  = "graphrag-plus-plus-qwen35-4b-q3_k_m-fixed.gguf"
$LlamaExe   = "llama-b9066\llama-server.exe"
$LlamaPort  = 8080
$BackendPort = 8000
$FrontendPort = 3000

# ---- Helpers ---------------------------------------------------------
function Write-Step([int]$n, [string]$msg) {
    Write-Host ""
    Write-Host "  [$n/3] $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "        OK: $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "        WARN: $msg" -ForegroundColor Yellow
}

function Wait-Port([int]$port, [int]$timeoutSec = 20) {
    $deadline = [DateTime]::Now.AddSeconds($timeoutSec)
    while ([DateTime]::Now -lt $deadline) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", $port)
            $tcp.Close()
            return $true
        } catch {}
        Start-Sleep -Milliseconds 400
    }
    return $false
}

# ---- Banner ----------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  ================================================" -ForegroundColor DarkCyan
Write-Host "          GraphRAG++  Quick-Start  v2             " -ForegroundColor Cyan
Write-Host "          Local AMD/Vulkan GPU Inference          " -ForegroundColor DarkCyan
Write-Host "  ================================================" -ForegroundColor DarkCyan
Write-Host ""

# ---- Pre-flight checks -----------------------------------------------
if (-not (Test-Path (Join-Path $Root $LlamaExe))) {
    Write-Host "  [ERROR] $LlamaExe not found in $Root" -ForegroundColor Red
    Write-Host "          Download a Vulkan build from:" -ForegroundColor Red
    Write-Host "          https://github.com/ggerganov/llama.cpp/releases" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $Root $ModelFile))) {
    Write-Host "  [ERROR] Model file not found: $ModelFile" -ForegroundColor Red
    Write-Host "          Run: python download_model.py" -ForegroundColor Yellow
    exit 1
}

# ---- Step 1: llama-server (GPU inference) ----------------------------
Write-Step 1 "Starting llama-server (Vulkan GPU)..."
$LlamaArgs = @(
    "-m", (Join-Path $Root $ModelFile),
    "--port", $LlamaPort,
    "--host", "127.0.0.1",
    "--n-gpu-layers", "99",
    "--ctx-size", "4096",
    "--threads", "6",
    "--log-disable"
)
$llamaProc = Start-Process -FilePath (Join-Path $Root $LlamaExe) `
    -ArgumentList $LlamaArgs `
    -WorkingDirectory $Root `
    -PassThru -WindowStyle Minimized

Write-Host "        Waiting for llama-server on port $LlamaPort..." -ForegroundColor DarkGray

if (Wait-Port $LlamaPort 30) {
    Write-Ok "llama-server ready on http://127.0.0.1:$LlamaPort (PID $($llamaProc.Id))"
} else {
    Write-Warn "llama-server didn't respond in 30s - check the minimized window for errors."
}

# ---- Step 2: Python backend ------------------------------------------
Write-Step 2 "Starting FastAPI backend..."

# Point the backend at the local llama-server
$env:LOCAL_LLAMA_CPP_URL = "http://127.0.0.1:$LlamaPort"

$BackendCmd = "Set-Location '$Root\backend'; python main.py"
Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $BackendCmd `
    -WindowStyle Normal

Write-Host "        Waiting for backend on port $BackendPort..." -ForegroundColor DarkGray
if (Wait-Port $BackendPort 25) {
    Write-Ok "Backend ready on http://localhost:$BackendPort"
} else {
    Write-Warn "Backend didn't respond in 25s - check the backend window."
}

# ---- Step 3: Next.js frontend ----------------------------------------
Write-Step 3 "Starting Next.js frontend..."

$FrontendCmd = "Set-Location '$Root\frontend'; npm run dev"
Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $FrontendCmd `
    -WindowStyle Normal

Write-Host "        Waiting for frontend on port $FrontendPort..." -ForegroundColor DarkGray
if (Wait-Port $FrontendPort 40) {
    Write-Ok "Frontend ready on http://localhost:$FrontendPort"
} else {
    Write-Warn "Frontend didn't respond in 40s - check the frontend window."
}

# ---- Done ------------------------------------------------------------
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "     [OK] GraphRAG++ is running!                  " -ForegroundColor Green
Write-Host "                                                  " -ForegroundColor Green
Write-Host "     UI        ->  http://localhost:3000          " -ForegroundColor Green
Write-Host "     Backend   ->  http://localhost:8000          " -ForegroundColor Green
Write-Host "     LLM       ->  http://localhost:8080          " -ForegroundColor Green
Write-Host "                                                  " -ForegroundColor Green
Write-Host "     Mode: Local AMD Radeon (Vulkan)              " -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""

# Open browser after a short delay
Start-Sleep -Seconds 2
Start-Process "http://localhost:$FrontendPort"

Write-Host "  Press Ctrl+C in any window to stop that service." -ForegroundColor DarkGray
Write-Host ""
