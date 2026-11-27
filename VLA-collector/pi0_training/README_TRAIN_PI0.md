# Fine-Tuning $\pi_0$ (`lerobot/pi0_base`) on Your UR5e LeRobot Dataset

This folder (`pi0_training/`) is completely self-contained. Copy it together with your converted LeRobot v3.0 dataset (`lerobot_pi0_train/`, and optionally `lerobot_pi0_test/`) onto your Linux NVIDIA GPU computer.

---

## 1. What to Copy to the GPU Computer

On your data-collection/conversion PC, copy these directories to the GPU machine (e.g. via `scp`, `rsync`, or USB drive):

```text
gpu_workspace/
├── pi0_training/              # This entire folder
│   ├── train_pi0.py
│   ├── run_pi0_train.sh       # Editable launcher (venv + W&B + tmux)
│   ├── eval_pi0_offline.py
│   ├── setup_gpu_env.sh
│   ├── README_TRAIN_PI0.md
│   └── tests/
├── lerobot_pi0_train/         # Converted training dataset (LeRobot v3.0)
└── lerobot_pi0_test/          # Optional held-out test dataset (for eval_pi0_offline.py)
```

> **Tip:** Confirm `lerobot_pi0_train/meta/info.json` has `total_episodes > 0` and `total_frames > 0` before transferring (so you know the conversion completed and finalized cleanly).

---

## 2. Environment Setup & Hugging Face Auth (One-Time)

1. **Create the `.venv-pi0` virtual environment and install `lerobot[pi,peft,training]` + CUDA PyTorch + Trackio:**
   ```bash
   cd gpu_workspace/pi0_training
   bash setup_gpu_env.sh
   source .venv-pi0/bin/activate
   ```
   > **RTX 5090 (32 GB / Blackwell `sm_120`) note:** `setup_gpu_env.sh` creates an isolated `.venv-pi0` folder without modifying your system Python. If PyTorch prints a warning about `sm_120` compatibility on the 5090, rerun setup with CUDA 12.8 wheels:
   > ```bash
   > TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash setup_gpu_env.sh
   > ```
   > *(If `lerobot/pi0_base` ever requires an unreleased upstream change, re-run with `LEROBOT_FROM_GIT=1 bash setup_gpu_env.sh`.)*

2. **Accept the PaliGemma tokenizer license & authenticate:**
   $\pi_0$ uses the gated `google/paligemma-3b-pt-224` tokenizer.
   - Visit [https://huggingface.co/google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) and accept the license terms on your Hugging Face account.
   - On the GPU machine, log in:
     ```bash
     hf auth login
     ```
     *(Or export `HF_TOKEN=hf_...` in your shell.)*

---

## 3. Running Over SSH with `tmux` + Trackio (Recommended)

> **Note:** Recent LeRobot (`0.7+`) replaced Weights & Biases (`--wandb.*`) with Hugging Face **Trackio** (`--trackio.*`). `run_pi0_train.sh` enables Trackio automatically.

Running training directly in a plain SSH shell will **kill the job if your SSH connection drops**. Use **`run_pi0_train.sh --tmux`** so training keeps running after you disconnect:

1. **Adjust parameters (optional):** Open `run_pi0_train.sh` and edit the top block (`MODE`, `BATCH_SIZE`, `STEPS`, `TRACKIO_PROJECT`, `TRACKIO_SPACE_ID`, etc.), or override them on the command line:
   - On an **RTX 5090 (32 GB)**: `MODE=lora BATCH_SIZE=16` (or `32`) runs comfortably; `MODE=full BATCH_SIZE=4` or `8` also fits in 32 GB.
2. **Dry-run first:**
   ```bash
   bash run_pi0_train.sh --dry-run
   ```
3. **Launch in a detached `tmux` session (activates `.venv-pi0`, logs to `logs/` and Trackio):**
   ```bash
   bash run_pi0_train.sh --tmux
   ```
   *(Or override settings inline: `MODE=full BATCH_SIZE=8 STEPS=20000 bash run_pi0_train.sh --tmux`.)*
   - To view live training curves from any browser without SSH port-forwarding, pass a Hugging Face Space ID:
     ```bash
     TRACKIO_SPACE_ID=your_hf_user/pi0-training bash run_pi0_train.sh --tmux
     ```
4. **Monitor & detach safely:**
   - Attach to live terminal output: `tmux attach -t pi0`
   - **Detach without stopping training:** press **`Ctrl+B`**, release, then press **`D`** (you can now close SSH).
   - View the local Trackio dashboard via SSH port-forwarding (`ssh -L 7860:localhost:7860 user@gpu-host`, then run `source .venv-pi0/bin/activate && trackio show --project ur5e-pi0`).
   - Check GPU utilization or saved checkpoints over SSH anytime:
     ```bash
     watch -n 2 nvidia-smi
     tail -f logs/train_*.log
     ls -lh outputs/*/checkpoints/
     ```

---

## 4. Dataset Compatibility Notes (Automatic)

Your converted dataset already matches $\pi_0$'s requirements without rewriting or `--rename_map`:
- **Cameras:** `observation.images.world` and `observation.images.wrist` (`224x224`, AV1 `.mp4`). Passing `--policy.type=pi0 --policy.pretrained_path=lerobot/pi0_base` automatically binds your dataset's camera keys and 4D state/action shapes while loading pretrained weights.
- **State & Action:** 4D `observation.state` (`[x, y, z, gripper]`) and 4D `action` (`[dx, dy, dz, gripper]`) are automatically zero-padded to 32D inside `PI0Policy` and sliced back to 4D for loss & inference.
- **Already-delta actions:** Because your converter exports `dx, dy, dz, gripper` in delta mode, `--policy.use_relative_actions` is kept `false` by default (never double-difference delta actions).
- **Video backend:** Uses `--dataset.video_backend=pyav` by default for reliable AV1 decoding on Linux servers.

---

## 5. Training Commands

### Step A: Dry Run (Verify Dataset, GPU, and Auth)
```bash
python train_pi0.py \
  --dataset-dir ../lerobot_pi0_train \
  --dry-run
```

### Step B: Quick 20-Step Smoke Test
Before running for hours, verify forward/backward pass, loss logging, and checkpoint saving:
```bash
python train_pi0.py \
  --dataset-dir ../lerobot_pi0_train \
  --steps 20 \
  --save-freq 20 \
  --log-freq 5 \
  --batch-size 4 \
  --output-dir ./outputs/pi0_smoke_test
```

### Step C: Full Training Runs by GPU Size

- **Default: LoRA Fine-Tuning (Recommended for 16–24 GB GPUs & small demo counts)**
  Targets the action expert attention layers (`q_proj`, `v_proj`) plus state/action/time projections (~12–16 GB VRAM):
  ```bash
  python train_pi0.py \
    --dataset-dir ../lerobot_pi0_train \
    --mode lora \
    --lora-r 16 \
    --batch-size 8 \
    --steps 10000 \
    --save-freq 2000
  ```

- **Action Expert Only (`--mode expert-only`, ~12–16 GB VRAM)**
  Freezes the entire PaliGemma VLM and trains only the Gemma-300M action expert & projections:
  ```bash
  python train_pi0.py \
    --dataset-dir ../lerobot_pi0_train \
    --mode expert-only \
    --batch-size 8 \
    --steps 10000
  ```

- **Full Fine-Tuning (`--mode full`, recommended on 40–80 GB A100 / H100 / L40S)**
  ```bash
  python train_pi0.py \
    --dataset-dir ../lerobot_pi0_train \
    --mode full \
    --batch-size 16 \
    --steps 20000 \
    --save-freq 2500
  ```

- **Multi-GPU Training (e.g., 2× A100 / 2× RTX 4090)**
  ```bash
  python train_pi0.py \
    --dataset-dir ../lerobot_pi0_train \
    --mode full \
    --num-gpus 2 \
    --batch-size 16 \
    --steps 20000
  ```

### Step D: Resume an Interrupted Run
Point `--resume` at your existing output folder (or directly at `train_config.json`):
```bash
python train_pi0.py --resume ./outputs/pi0_lora_20261126_193000 --steps 20000
```

### Step E: Optional Weights & Biases or Image Augmentation
```bash
python train_pi0.py \
  --dataset-dir ../lerobot_pi0_train \
  --wandb --wandb-project ur5e-pi0 \
  --extra-arg=--dataset.image_transforms.enable=true
```

---

## 6. Offline Evaluation on `lerobot_pi0_test`

`lerobot-train` only evaluates simulated `gym` environments during training. To compare your saved checkpoints on your held-out real-robot test split (`lerobot_pi0_test`), run:

```bash
python eval_pi0_offline.py \
  --checkpoint ./outputs/pi0_lora_20261126_193000 \
  --dataset-dir ../lerobot_pi0_test \
  --stride 10 \
  --output-json ./outputs/pi0_lora_20261126_193000/eval_test_metrics.json
```

This scores every checkpoint (`002000`, `004000`, ..., `last`) in physical units (`dx, dy, dz` MAE/RMSE in meters and `gripper` MAE/RMSE in raw counts), helping you pick the best checkpoint without overfitting.

---

## 7. Copying the Trained Model Back

Copy the winning checkpoint directory (which contains `model.safetensors` / `adapter_model.safetensors`, `config.json`, and the `policy_preprocessor.json` / `policy_postprocessor.json` normalization files) back to your robot PC:

```text
outputs/pi0_lora_.../checkpoints/006000/pretrained_model/
```
