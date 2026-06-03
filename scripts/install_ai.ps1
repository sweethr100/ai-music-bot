param(
    [string]$ApplioRepo = "https://github.com/IAHispano/Applio.git",
    [string]$TorchCuda = "cu128"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VendorDir = Join-Path $Root "vendor"
$ApplioDir = Join-Path $VendorDir "Applio"
$TorchIndex = "https://download.pytorch.org/whl/$TorchCuda"

function Install-CudaTorch {
    param([string]$PythonExe)
    & $PythonExe -m pip install -U pip
    & $PythonExe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url $TorchIndex
}

function Test-CudaTorch {
    param([string]$PythonExe, [string]$Label)
    & $PythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('$Label CUDA OK:', torch.__version__, torch.cuda.get_device_name(0))"
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi를 찾지 못했습니다. NVIDIA 드라이버를 먼저 설치한 뒤 다시 실행해 주세요."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git을 찾지 못했습니다. Git을 설치한 뒤 다시 실행해 주세요."
}

nvidia-smi

Write-Host "Installing CUDA PyTorch for bot environment ($TorchCuda)..."
Install-CudaTorch -PythonExe "python"

Write-Host "Installing bot + AI music dependencies..."
python -m pip install -r (Join-Path $Root "requirements.txt")
Install-CudaTorch -PythonExe "python"
Test-CudaTorch -PythonExe "python" -Label "Bot"

if (-not (Test-Path $VendorDir)) {
    New-Item -ItemType Directory -Path $VendorDir | Out-Null
}

if (-not (Test-Path $ApplioDir)) {
    Write-Host "Downloading Applio engine..."
    git clone $ApplioRepo $ApplioDir
} else {
    Write-Host "Applio already exists. Pulling latest changes..."
    git -C $ApplioDir pull --ff-only
}

$ApplioVenv = Join-Path $ApplioDir "env"
if (-not (Test-Path $ApplioVenv)) {
    Write-Host "Creating Applio virtual environment..."
    python -m venv $ApplioVenv
}

$ApplioPython = Join-Path $ApplioVenv "Scripts\python.exe"
Write-Host "Installing CUDA PyTorch for Applio environment ($TorchCuda)..."
Install-CudaTorch -PythonExe $ApplioPython

$ApplioRequirements = Join-Path $ApplioDir "requirements.txt"
if (Test-Path $ApplioRequirements) {
    Write-Host "Installing Applio requirements..."
    & $ApplioPython -m pip install -r $ApplioRequirements
    Install-CudaTorch -PythonExe $ApplioPython
}

Test-CudaTorch -PythonExe $ApplioPython -Label "Applio"

$VoiceModels = Join-Path $Root "voice_models"
if (-not (Test-Path $VoiceModels)) {
    New-Item -ItemType Directory -Path $VoiceModels | Out-Null
}

Write-Host ""
Write-Host "Done."
Write-Host "Put RVC model files in voice_models\<voice-name>\"
Write-Host "Example: voice_models\myvoice\myvoice.pth and voice_models\myvoice\myvoice.index"
