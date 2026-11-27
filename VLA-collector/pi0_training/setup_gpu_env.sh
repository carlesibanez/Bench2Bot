#!/usr/bin/env bash
set -euo pipefail

# Setup script for fine-tuning Hugging Face LeRobot π₀ (lerobot/pi0_base)
# on a Linux + NVIDIA GPU computer.
#
# Usage:
#   bash setup_gpu_env.sh
#
# Optional environment variables:
#   VENV_DIR=.venv-pi0          Target virtualenv directory
#   PYTHON_VERSION=3.12         Preferred Python version (when using uv)
#   LEROBOT_FROM_GIT=0          Set to 1 to install lerobot directly from GitHub main
#   TORCH_INDEX_URL=""          Optional custom PyTorch wheel index URL (e.g. cu124/cu126/cu128)

VENV_DIR="${VENV_DIR:-.venv-pi0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_FROM_GIT="${LEROBOT_FROM_GIT:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "=== 1/5 Checking GPU & system tools ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed on this machine." >&2
fi

if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg: $(ffmpeg -version | head -n 1)"
else
    echo "Note: system ffmpeg binary not found (PyAV bundles libav wheels, though ffmpeg CLI is recommended)."
fi

echo ""
echo "=== 2/5 Creating Python ${PYTHON_VERSION} virtual environment in '${VENV_DIR}' ==="
if command -v uv >/dev/null 2>&1; then
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    INSTALLER="uv pip install"
else
    PY_BIN=""
    for candidate in "python${PYTHON_VERSION}" "python3.12" "python3.11" "python3.10" "python3"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PY_BIN="${candidate}"
            break
        fi
    done
    if [[ -z "${PY_BIN}" ]]; then
        echo "ERROR: No suitable Python 3.10-3.12 interpreter found." >&2
        exit 1
    fi
    echo "Using ${PY_BIN}: $(${PY_BIN} --version)"
    "${PY_BIN}" -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip wheel setuptools
    INSTALLER="python -m pip install"
fi

echo ""
echo "=== 3/5 Installing PyTorch (CUDA) ==="
if [[ -n "${TORCH_INDEX_URL}" ]]; then
    ${INSTALLER} torch torchvision --index-url "${TORCH_INDEX_URL}"
else
    ${INSTALLER} torch torchvision
fi

echo ""
echo "=== 4/5 Installing LeRobot [pi, peft, training], PyAV, and pytest ==="
if [[ "${LEROBOT_FROM_GIT}" == "1" ]]; then
    ${INSTALLER} "lerobot[pi,peft,training]@git+https://github.com/huggingface/lerobot.git" av pytest
else
    ${INSTALLER} "lerobot[pi,peft,training]" av pytest
fi

echo ""
echo "=== 5/5 Verifying installation ==="
python - <<'PY'
import sys
import torch
import lerobot
import peft
import transformers
import av
import trackio

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()} (devices: {torch.cuda.device_count()})")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"bf16 support : {torch.cuda.is_bf16_supported()}")
print(f"LeRobot      : {lerobot.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"PEFT         : {peft.__version__}")
print(f"PyAV         : {av.__version__}")
print(f"Trackio      : {trackio.__version__}")
PY

lerobot-train --help >/dev/null
echo "lerobot-train CLI: OK"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "Next steps:"
echo "  1. Activate env : source ${VENV_DIR}/bin/activate"
echo "  2. Accept license on https://huggingface.co/google/paligemma-3b-pt-224"
echo "  3. Log in to HF : hf auth login"
echo "  4. Dry run      : python train_pi0.py --dataset-dir /path/to/lerobot_pi0_train --dry-run"
echo "============================================================"
