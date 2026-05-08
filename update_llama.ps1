# ======================================================================
#     GraphRAG++ - One-Time llama.cpp Updater                          
#     Downloads a Vulkan build that supports the 'qwen35' architecture  
#     Run this ONCE, then use start.ps1 to launch everything.           
# ======================================================================

$Root        = $PSScriptRoot
$Version     = "b5476"
$Zip         = "llama-$Version-bin-win-vulkan-x64.zip"
$DownloadUrl = "https://github.com/ggerganov/llama.cpp/releases/download/$Version/$Zip"
$TempDir     = Join-Path $env:TEMP "llama_update_$Version"

Write-Host ""
Write-Host "  GraphRAG++ llama.cpp Updater" -ForegroundColor Cyan
Write-Host "  Target build : $Version  (qwen35-capable)" -ForegroundColor DarkCyan
Write-Host "  Destination  : $Root" -ForegroundColor DarkCyan
Write-Host ""

# ---- Download --------------------------------------------------------
Write-Host ""
Write-Host "  [1/3] Downloading $Zip..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($DownloadUrl, (Join-Path $TempDir $Zip))
    Write-Host "        OK: Download complete" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Download failed: $_" -ForegroundColor Red
    Write-Host "          Try manually from: https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
    exit 1
}

# ---- Extract ---------------------------------------------------------
Write-Host "  [2/3] Extracting..." -ForegroundColor Cyan
$ExtractDir = Join-Path $TempDir "extracted"
Expand-Archive -Path (Join-Path $TempDir $Zip) -DestinationPath $ExtractDir -Force
Write-Host "        OK: Extracted" -ForegroundColor Green

# ---- Copy binaries ---------------------------------------------------
Write-Host "  [3/3] Copying binaries to $Root..." -ForegroundColor Cyan

$ExeSearch = Get-ChildItem -Path $ExtractDir -Filter "llama-server.exe" -Recurse | Select-Object -First 1
if (-not $ExeSearch) {
    Write-Host "  [ERROR] llama-server.exe not found in the archive." -ForegroundColor Red
    exit 1
}
$BinDir = $ExeSearch.DirectoryName

$copied = 0
Get-ChildItem -Path $BinDir -Include "*.exe","*.dll" | ForEach-Object {
    $dest = Join-Path $Root $_.Name
    Copy-Item $_.FullName -Destination $dest -Force
    $copied++
}
Write-Host "        OK: Copied $copied files" -ForegroundColor Green

# ---- Cleanup ---------------------------------------------------------
Remove-Item -Path $TempDir -Recurse -Force

# ---- Verify ----------------------------------------------------------
Write-Host ""
$serverPath = Join-Path $Root "llama-server.exe"
if (Test-Path $serverPath) {
    Write-Host "  =================================================" -ForegroundColor Green
    Write-Host "    [OK]  llama.cpp updated to $Version!           " -ForegroundColor Green
    Write-Host "                                                   " -ForegroundColor Green
    Write-Host "    Now run: .\start.ps1                           " -ForegroundColor Green
    Write-Host "  =================================================" -ForegroundColor Green
} else {
    Write-Host "  [WARN] llama-server.exe not found after copy - check manually." -ForegroundColor Yellow
}
Write-Host ""
