#!/usr/bin/env bash
set -euo pipefail

APPLIO_REPO="${APPLIO_REPO:-https://github.com/IAHispano/Applio.git}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT/vendor"
APPLIO_DIR="$VENDOR_DIR/Applio"

install_cuda_torch() {
  local python_exe="$1"
  "$python_exe" -m pip install -U pip
  "$python_exe" -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"
  install_audio_save_dependencies "$python_exe"
}

install_audio_save_dependencies() {
  local python_exe="$1"
  local torchcodec_requirement
  torchcodec_requirement="$("$python_exe" -c '
import sys
import torch

version = torch.__version__.split("+", 1)[0].split(".")
major, minor = int(version[0]), int(version[1])
if sys.version_info >= (3, 13) and (major, minor) < (2, 8):
    raise SystemExit(
        f"Python 3.13에서 TorchCodec을 쓰려면 torch 2.8 이상이 필요합니다. "
        f"현재 torch는 {torch.__version__}입니다. `TORCH_CUDA=cu128`로 다시 설치하거나 Python 3.12 venv를 사용해 주세요."
    )
if (major, minor) >= (2, 11):
    print("torchcodec")
elif (major, minor) == (2, 10):
    print("torchcodec==0.10.*")
elif (major, minor) == (2, 9):
    print("torchcodec==0.9.*")
elif (major, minor) == (2, 8):
    print("torchcodec==0.7.*")
elif (major, minor) == (2, 7):
    print("torchcodec==0.5.*")
else:
    print("torchcodec==0.2.*")
')"

  "$python_exe" -m pip install "soundfile>=0.12.1"
  "$python_exe" -m pip install "$torchcodec_requirement"
}

test_cuda_torch() {
  local python_exe="$1"
  local label="$2"
  "$python_exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('${label} CUDA OK:', torch.__version__, torch.cuda.get_device_name(0))"
}

test_audio_save() {
  local python_exe="$1"
  local label="$2"
  "$python_exe" -c "import tempfile, torch, torchaudio; path = tempfile.mktemp(suffix='.wav'); torchaudio.save(path, torch.zeros(1, 16000), 16000); print('${label} audio save OK:', path)"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi를 찾지 못했습니다. NVIDIA 드라이버를 먼저 설치한 뒤 다시 실행해 주세요." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git을 찾지 못했습니다. Git을 설치한 뒤 다시 실행해 주세요." >&2
  exit 1
fi

nvidia-smi

echo "Installing CUDA PyTorch for bot environment (${TORCH_CUDA})..."
install_cuda_torch python

echo "Installing bot + AI music dependencies..."
python -m pip install -r "$ROOT/requirements.txt"
install_cuda_torch python
test_cuda_torch python "Bot"
test_audio_save python "Bot"

mkdir -p "$VENDOR_DIR"

if [ ! -d "$APPLIO_DIR/.git" ]; then
  echo "Downloading Applio engine..."
  git clone "$APPLIO_REPO" "$APPLIO_DIR"
else
  echo "Applio already exists. Pulling latest changes..."
  git -C "$APPLIO_DIR" pull --ff-only
fi

if [ ! -d "$APPLIO_DIR/env" ]; then
  echo "Creating Applio virtual environment..."
  python -m venv "$APPLIO_DIR/env"
fi

echo "Installing CUDA PyTorch for Applio environment (${TORCH_CUDA})..."
install_cuda_torch "$APPLIO_DIR/env/bin/python"

if [ -f "$APPLIO_DIR/requirements.txt" ]; then
  echo "Installing Applio requirements..."
  "$APPLIO_DIR/env/bin/python" -m pip install -r "$APPLIO_DIR/requirements.txt"
  install_cuda_torch "$APPLIO_DIR/env/bin/python"
fi

test_cuda_torch "$APPLIO_DIR/env/bin/python" "Applio"
test_audio_save "$APPLIO_DIR/env/bin/python" "Applio"

mkdir -p "$ROOT/voice_models"
mkdir -p "$ROOT/data/recordings"

echo
echo "Done."
echo "Put RVC model files in voice_models/<voice-name>/"
echo "Example: voice_models/myvoice/myvoice.pth and voice_models/myvoice/myvoice.index"
