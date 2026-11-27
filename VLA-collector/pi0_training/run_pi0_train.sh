#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Editable π₀ (lerobot/pi0_base) Training Configuration
# ------------------------------------------------------------------------------
# Edit the defaults below directly, OR override any of them per-run via env vars:
#   MODE=full BATCH_SIZE=8 STEPS=20000 bash run_pi0_train.sh --tmux
#
# Extra CLI flags (such as --dry-run or --skip-hf-check) are forwarded to
# train_pi0.py. Pass --tmux to launch inside a persistent tmux session so SSH
# disconnects will NOT kill your training run.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Environment & Dataset Paths ---
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv-pi0}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/../lerobot_pi0_train}"
REPO_ID="${REPO_ID:-local/ur5e_titration}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi0_base}"

# --- Fine-Tuning Strategy & Hyperparameters ---
MODE="${MODE:-lora}"                  # Options: lora | expert-only | freeze-vision | full
LORA_R="${LORA_R:-16}"                # LoRA rank (used when MODE=lora)
BATCH_SIZE="${BATCH_SIZE:-16}"        # RTX 5090 (32 GB): 16-32 for lora/expert-only, 4-8 for full
STEPS="${STEPS:-10000}"               # Total optimizer steps
SAVE_FREQ="${SAVE_FREQ:-2000}"        # Save checkpoint every N steps
LOG_FREQ="${LOG_FREQ:-100}"           # Log metrics every N steps
NUM_WORKERS="${NUM_WORKERS:-4}"       # DataLoader workers
CHUNK_SIZE="${CHUNK_SIZE:-50}"        # Action chunk horizon
N_ACTION_STEPS="${N_ACTION_STEPS:-50}" # Executed steps per chunk
LR="${LR:-}"                          # Optional peak LR override (empty = pi0_base default 2.5e-5)

# --- Output & Resume ---
OUTPUT_DIR="${OUTPUT_DIR:-}"          # Empty = auto-timestamped ./outputs/pi0_<mode>_<timestamp>
JOB_NAME="${JOB_NAME:-}"              # Empty = pi0_<mode>
RESUME_DIR="${RESUME_DIR:-}"          # Set to existing run dir (or train_config.json) to resume

# --- Experiment Tracking (Trackio — LeRobot 0.7+ built-in tracker) ---
USE_TRACKIO="${USE_TRACKIO:-1}"       # 1 = enable Trackio logging, 0 = disable
TRACKIO_PROJECT="${TRACKIO_PROJECT:-ur5e-pi0}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-}"  # Optional HF Space (e.g. "your_user/pi0-logs") to view remotely

# --- tmux & Console Logs ---
TMUX_SESSION="${TMUX_SESSION:-pi0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

# Optional extra flags passed straight to train_pi0.py, e.g.:
#   EXTRA_ARGS=("--extra-arg=--dataset.image_transforms.enable=true")
EXTRA_ARGS=()

# ==============================================================================
# Parse Launcher Flags (--tmux vs train_pi0.py passthrough)
# ==============================================================================
LAUNCH_TMUX=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "${arg}" in
        --tmux)
            LAUNCH_TMUX=1
            ;;
        *)
            PASSTHROUGH_ARGS+=("${arg}")
            ;;
    esac
done

# ==============================================================================
# 1. Activate Virtual Environment
# ==============================================================================
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
    echo "ERROR: Virtual environment not found at '${VENV_DIR}'." >&2
    echo "Run 'bash ${SCRIPT_DIR}/setup_gpu_env.sh' first." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"
echo "[venv] Activated ${VENV_DIR} ($(python --version))"

# ==============================================================================
# 2. Verify Trackio Availability (when enabled)
# ==============================================================================
if [[ "${USE_TRACKIO}" == "1" ]]; then
    if ! python -c "import trackio" >/dev/null 2>&1; then
        echo "ERROR: 'trackio' is not installed in '${VENV_DIR}'." >&2
        echo "Install it with: pip install 'lerobot[pi,peft,training]'  (or re-run setup_gpu_env.sh)" >&2
        exit 1
    fi
    if [[ -n "${TRACKIO_SPACE_ID}" ]]; then
        echo "[trackio] Enabled (project='${TRACKIO_PROJECT}', space='${TRACKIO_SPACE_ID}')."
    else
        echo "[trackio] Enabled (local SQLite DB, project='${TRACKIO_PROJECT}')."
    fi
fi

# ==============================================================================
# 3. Optional tmux Detached Launch (Safe for SSH Disconnects)
# ==============================================================================
if [[ "${LAUNCH_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: 'tmux' is not installed. Install it (e.g. 'sudo apt install tmux') or run inside 'screen'." >&2
        exit 1
    fi

    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "ERROR: tmux session '${TMUX_SESSION}' already exists." >&2
        echo "Attach to it with:  tmux attach -t ${TMUX_SESSION}" >&2
        echo "Or pick a new name: TMUX_SESSION=pi0_run2 bash $0 --tmux" >&2
        exit 1
    fi

    ENV_PREFIX=""
    for var_name in \
        VENV_DIR DATASET_DIR REPO_ID PRETRAINED_PATH \
        MODE LORA_R BATCH_SIZE STEPS SAVE_FREQ LOG_FREQ NUM_WORKERS \
        CHUNK_SIZE N_ACTION_STEPS LR OUTPUT_DIR JOB_NAME RESUME_DIR \
        USE_TRACKIO TRACKIO_PROJECT TRACKIO_SPACE_ID LOG_DIR; do
        ENV_PREFIX+="${var_name}=$(printf '%q' "${!var_name}") "
    done
    for secret_var in HF_TOKEN HF_HOME WANDB_API_KEY; do
        if [[ -n "${!secret_var:-}" ]]; then
            ENV_PREFIX+="${secret_var}=$(printf '%q' "${!secret_var}") "
        fi
    done

    QUOTED_ARGS=""
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        QUOTED_ARGS="$(printf '%q ' "${PASSTHROUGH_ARGS[@]}")"
    fi

    TMUX_CMD="cd $(printf '%q' "${SCRIPT_DIR}") && ${ENV_PREFIX}bash $(printf '%q' "${SCRIPT_DIR}/run_pi0_train.sh") ${QUOTED_ARGS}; rc=\$?; echo ''; echo \"[Training exited with code \${rc}. Press Ctrl+D or type exit to close.]\"; exec bash"
    tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_CMD}"

    echo ""
    echo "============================================================"
    echo "Training launched in background tmux session: '${TMUX_SESSION}'"
    echo "  Attach & watch live : tmux attach -t ${TMUX_SESSION}"
    echo "  Detach safely (SSH) : press Ctrl+B, release, then press D"
    echo "  Check GPU usage     : watch -n 2 nvidia-smi"
    if [[ "${USE_TRACKIO}" == "1" ]]; then
        if [[ -n "${TRACKIO_SPACE_ID}" ]]; then
            echo "  Trackio Space       : https://huggingface.co/spaces/${TRACKIO_SPACE_ID}"
        else
            echo "  Trackio dashboard   : trackio show --project ${TRACKIO_PROJECT}"
        fi
    fi
    echo "============================================================"
    exit 0
fi

# ==============================================================================
# 4. Build train_pi0.py Command and Run with Persistent Logging
# ==============================================================================
CMD=(python "${SCRIPT_DIR}/train_pi0.py")

if [[ -n "${RESUME_DIR}" ]]; then
    CMD+=(--resume "${RESUME_DIR}")
    if [[ -n "${STEPS}" ]]; then
        CMD+=(--steps "${STEPS}")
    fi
else
    CMD+=(
        --dataset-dir "${DATASET_DIR}"
        --repo-id "${REPO_ID}"
        --pretrained-path "${PRETRAINED_PATH}"
        --mode "${MODE}"
        --lora-r "${LORA_R}"
        --batch-size "${BATCH_SIZE}"
        --steps "${STEPS}"
        --save-freq "${SAVE_FREQ}"
        --log-freq "${LOG_FREQ}"
        --num-workers "${NUM_WORKERS}"
        --chunk-size "${CHUNK_SIZE}"
        --n-action-steps "${N_ACTION_STEPS}"
    )

    if [[ -n "${LR}" ]]; then
        CMD+=(--lr "${LR}")
    fi
    if [[ -n "${OUTPUT_DIR}" ]]; then
        CMD+=(--output-dir "${OUTPUT_DIR}")
    fi
    if [[ -n "${JOB_NAME}" ]]; then
        CMD+=(--job-name "${JOB_NAME}")
    fi
    if [[ "${USE_TRACKIO}" == "1" ]]; then
        CMD+=(--trackio --trackio-project "${TRACKIO_PROJECT}")
        if [[ -n "${TRACKIO_SPACE_ID}" ]]; then
            CMD+=(--trackio-space-id "${TRACKIO_SPACE_ID}")
        fi
    fi
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[log] Saving console output to ${LOG_FILE}"
cd "${SCRIPT_DIR}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
