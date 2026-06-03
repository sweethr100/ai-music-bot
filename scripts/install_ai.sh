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
}

test_cuda_torch() {
  local python_exe="$1"
  local label="$2"
  "$python_exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('${label} CUDA OK:', torch.__version__, torch.cuda.get_device_name(0))"
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

mkdir -p "$ROOT/voice_models"

echo
echo "Done."
echo "Put RVC model files in voice_models/<voice-name>/"
echo "Example: voice_models/myvoice/myvoice.pth and voice_models/myvoice/myvoice.index"
