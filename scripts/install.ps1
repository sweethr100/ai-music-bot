param(
    [string]$ApplioRepo = "https://github.com/IAHispano/Applio.git",
    [string]$TorchCuda = "cu128",
    [string]$TorchVersion = "2.8.0",
    [string]$TorchVisionVersion = "0.23.0",
    [string]$TorchAudioVersion = "2.8.0"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VendorDir = Join-Path $Root "vendor"
$ApplioDir = Join-Path $VendorDir "Applio"
$TorchIndex = "https://download.pytorch.org/whl/$TorchCuda"

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command $($Arguments -join ' ')"
    }
}

function Install-CudaTorch {
    param([string]$PythonExe)
    Invoke-Checked $PythonExe @("-m", "pip", "install", "-U", "pip")
    Invoke-Checked $PythonExe @(
        "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "torch==$TorchVersion",
        "torchvision==$TorchVisionVersion",
        "torchaudio==$TorchAudioVersion",
        "pillow<12.0,>=8.0",
        "--index-url", $TorchIndex
    )
}

function Install-RequirementsWithoutTorch {
    param([string]$PythonExe, [string]$RequirementsPath)

    $FilteredRequirements = Join-Path ([System.IO.Path]::GetTempPath()) "applio-requirements-no-torch.txt"
    Get-Content $RequirementsPath | Where-Object {
        $_ -notmatch '^(torch|torchaudio|torchvision)([=<>!~; ].*)?$'
    } | Set-Content -Path $FilteredRequirements -Encoding UTF8
    Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", $FilteredRequirements)
}

function Install-ApplioCompatibilityFixes {
    param([string]$PythonExe)

    Invoke-Checked $PythonExe @("-m", "pip", "install", "pillow<12.0,>=8.0")
}

function Install-ApplioPrerequisiteModels {
    param([string]$PythonExe)

    Push-Location $ApplioDir
    try {
        Invoke-Checked $PythonExe @(
            "core.py",
            "prerequisites",
            "--pretraineds_hifigan", "False",
            "--models", "True",
            "--exe", "False"
        )
    } finally {
        Pop-Location
    }
}

function Install-MultiSingerSeparator {
    param([string]$PythonExe)

    Write-Host "Installing AI duet multi-singer separator (Asteroid / MedleyVox support)..."
    Invoke-Checked $PythonExe @(
        "-m", "pip", "install",
        "asteroid-filterbanks>=0.4.0",
        "huggingface-hub",
        "pandas",
        "pytorch-lightning",
        "torchmetrics==0.11.4",
        "torch-optimizer==0.1.0",
        "torch-stoi>=0.1.2",
        "pytorch-ranger>=0.1.1",
        "typer>=0.20.0,<0.26.0",
        "httpx",
        "mir-eval",
        "pystoi"
    )
    Invoke-Checked $PythonExe @("-m", "pip", "install", "asteroid==0.7.0", "--no-deps")
    Invoke-Checked $PythonExe @(
        "-c",
        "import asteroid; print('AI duet multi-singer separator OK:', asteroid.__version__)"
    )
    Write-Host "Downloading default AI duet multi-singer model (Cyru5/MedleyVox)..."
    Invoke-Checked $PythonExe @(
        "-c",
        "from huggingface_hub import hf_hub_download; path = hf_hub_download(repo_id='Cyru5/MedleyVox', filename='vocals 100.pth'); print('Default AI duet model cached:', path)"
    )
}

function Test-CudaTorch {
    param([string]$PythonExe, [string]$Label)
    Invoke-Checked $PythonExe @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('$Label CUDA OK:', torch.__version__, torch.cuda.get_device_name(0))")
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
Invoke-Checked "python" @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt"))
Install-CudaTorch -PythonExe "python"
Install-MultiSingerSeparator -PythonExe "python"
Test-CudaTorch -PythonExe "python" -Label "Bot"

if (-not (Test-Path $VendorDir)) {
    New-Item -ItemType Directory -Path $VendorDir | Out-Null
}

if (-not (Test-Path $ApplioDir)) {
    Write-Host "Downloading Applio engine..."
    Invoke-Checked "git" @("clone", $ApplioRepo, $ApplioDir)
} else {
    Write-Host "Applio already exists. Pulling latest changes..."
    Invoke-Checked "git" @("-C", $ApplioDir, "pull", "--ff-only")
}

$ApplioVenv = Join-Path $ApplioDir "env"
if (-not (Test-Path $ApplioVenv)) {
    Write-Host "Creating Applio virtual environment..."
    Invoke-Checked "python" @("-m", "venv", $ApplioVenv)
}

$ApplioPython = Join-Path $ApplioVenv "Scripts\python.exe"
Write-Host "Installing CUDA PyTorch for Applio environment ($TorchCuda)..."
Install-CudaTorch -PythonExe $ApplioPython

$ApplioRequirements = Join-Path $ApplioDir "requirements.txt"
if (Test-Path $ApplioRequirements) {
    Write-Host "Installing Applio requirements..."
    Install-RequirementsWithoutTorch -PythonExe $ApplioPython -RequirementsPath $ApplioRequirements
    Install-CudaTorch -PythonExe $ApplioPython
    Install-ApplioCompatibilityFixes -PythonExe $ApplioPython
    Install-ApplioPrerequisiteModels -PythonExe $ApplioPython
}

Test-CudaTorch -PythonExe $ApplioPython -Label "Applio"

$VoiceModels = Join-Path $Root "voice_models"
if (-not (Test-Path $VoiceModels)) {
    New-Item -ItemType Directory -Path $VoiceModels | Out-Null
}

$Recordings = Join-Path $Root "data\recordings"
if (-not (Test-Path $Recordings)) {
    New-Item -ItemType Directory -Path $Recordings | Out-Null
}

Write-Host ""
Write-Host "Done."
Write-Host "Put RVC model files in voice_models\<voice-name>\"
Write-Host "Example: voice_models\myvoice\myvoice.pth and voice_models\myvoice\myvoice.index"
