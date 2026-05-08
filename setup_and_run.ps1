# GraphRAG++ Unified Startup Script (v3 — Zero-Compilers Edition)
# ==============================================================

$LLAMA_VERSION = "b5476" # Supports qwen35 architecture; min b4900+ required
$LLAMA_ZIP = "llama-$LLAMA_VERSION-bin-win-vulkan-x64.zip"
$LLAMA_URL = "https://github.com/ggerganov/llama.cpp/releases/download/$LLAMA_VERSION/$LLAMA_ZIP"

Write-Host "`n[1/4] Ensuring llama-server.exe is available..." -ForegroundColor Cyan
if (-not (Test-Path "llama-server.exe")) {
    Write-Host "Llama-server not found locally. Downloading pre-built Vulkan binary (no compiler needed)..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $LLAMA_URL -OutFile $LLAMA_ZIP
    Write-Host "Extracting..." -ForegroundColor Yellow
    Expand-Archive -Path $LLAMA_ZIP -DestinationPath "." -Force
    Remove-Item $LLAMA_ZIP
}

Write-Host "`n[2/4] Downloading/Verifying Fine-Tuned Model (GGUF)..." -ForegroundColor Cyan
python download_model.py

Write-Host "`n[3/4] Starting Services (Backend + Frontend)..." -ForegroundColor Cyan
# Start backend in its own window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python main.py"
# Start frontend in its own window
if (Test-Path frontend) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
}

Write-Host "`n[4/4] Starting Local GPU Inference Server (AMD/Vulkan)..." -ForegroundColor Cyan
# Start the downloaded binary
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\llama-server.exe -m graphrag-plus-plus-qwen35-4b-q3_k_m.gguf --port 8080 --n-gpu-layers 99"

Write-Host "`n====================================================" -ForegroundColor Green
Write-Host "Success! GraphRAG++ is now running." -ForegroundColor Green
Write-Host "Mode: Local AMD/Vulkan GPU"
Write-Host "UI: http://localhost:3000"
Write-Host "====================================================`n"
