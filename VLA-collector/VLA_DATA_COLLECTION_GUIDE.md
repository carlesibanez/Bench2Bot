# UR5e Titration VLA Data Collection & LeRobot Guide

This guide explains how to transfer the collector to the robot-control computer, verify `ur_rtde` Remote-Control freedrive, collect raw demonstrations fast, and export them to a `LeRobotDataset` (storing **both** 7D Joint+Gripper and 7D TCP+Gripper) for SmolVLA fine-tuning.

---

## 1. Environment Setup (Transferring to Another Computer)

> **Python version note:** Use **Python 3.10 or 3.11** on the robot PC (especially on Windows) so `ur-rtde`, `pyrealsense2`, and `lerobot` install cleanly from prebuilt wheels.

### Option A: Using `uv` (recommended, fast)

On the **robot-control PC** (lightweight install, no PyTorch needed for live collection):
```bash
uv venv --python 3.11
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate
uv pip install -r requirements-robot.txt
```

On the **conversion / GPU training PC** (or same PC if converting locally):
```bash
uv pip install -r requirements-convert.txt
```
Or install everything in one shot with `uv sync --extra convert`.

### Option B: Standard `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-robot.txt
# On the conversion / training machine:
pip install -r requirements-convert.txt
```

---

## 2. Hardware & Freedrive Verification Checklist

1. **UR5e Teach Pendant:** Switch the robot to **Remote Control** mode (make sure no PolyScope program is actively running or locking port `30004`).
2. **Robotiq Hand-E Gripper:** Connected through the UR tool/controller URCap on port `63352` (default IP `192.168.0.2`).
3. **Cameras:**
   - **World camera:** USB webcam plugged into USB 3.0.
   - **Wrist camera:** Intel RealSense D405 plugged into USB 3.0 (different USB host controller / root hub from the webcam if possible to avoid bandwidth saturation).
4. **Verify Remote-Control Freedrive First:**
   ```bash
   python test_freedrive_rtde.py --ip 192.168.0.2
   ```
   Confirm both `freedriveMode` (XYZ translation with wrist orientation locked) and `teachMode` (full 6-DoF) engage and release cleanly.

---

## 3. Running the GUI App

Launch the 4-tab GUI:
```bash
python run_vla_collector.py
```

### Offline / Simulation Testing
Leave **Simulation Mode** checked in **Tab 1** (and `mock` cameras in **Tab 3**) to test anchor capture, delta preview, dry runs, raw recording, and LeRobot conversion without touching hardware.

### Real Hardware Workflow

#### A. One-Time Setup (**Tab 3 & Tab 4**)
1. **Tab 3 — Cameras Setup:** Click **Scan Connected Cameras**.
   - World camera: select `opencv`, pick the webcam index (e.g. `0`), `640x480`, `30 FPS`, click **Apply & Start**.
   - Wrist camera: select `realsense`, pick your RealSense D405 serial number, `640x480`, `30 FPS`, click **Apply & Start**. (Uncheck *D405 Auto-Exposure* and set a fixed exposure if lighting flickers.)
2. **Tab 4 — Settings & Export:**
   - Set **Recording FPS** (`20` recommended, range `10–30`).
   - Set **Raw Data Directory** (default `./vla_data/raw`).
   - Set **Default Task Prompt** (e.g. `"Pick up the pipette from the holder, aspirate liquid from the reagent vial, and dispense it into the target vial."`).
   - Review `z_min_safety_m` (meters in UR base frame — prevents any virtual delta waypoint from hitting the table) and `moveL` speed (`0.15 m/s` default).
   - Click **Save Settings JSON** if you want to reuse them across sessions.

#### B. Inspect / Customize the Sequence (**Tab 2 — Sequence Builder**)
- The default titration preset defines 4 anchors (`home`, `pipette_grasp`, `vial1_in`, `vial2_in`) and 14 steps covering:
  1. Hover (`+8 cm` base Z) → Open (`180`) → Descend to `pipette_grasp` → Grasp (`200`) → Lift (`+15 cm`)
  2. Hover Vial 1 (`+10 cm`) → Squeeze (`240`) → Descend to `vial1_in` → Unsqueeze (`200`, `0.6s` ramp) → Lift (`+10 cm`)
  3. Hover Vial 2 (`+10 cm`) → Descend to `vial2_in` → Squeeze (`240`, `0.6s` ramp) → Lift (`+10 cm`)
- **Base-frame Z deltas:** Positive `dz` (`+0.10`) moves **straight up** in the UR base frame.
- Toggle **Include in recording** per step or add/reorder steps, then click **Save Preset...** if customized.

#### C. Fast Episode Collection Loop (**Tab 1 — Collect & Monitor**)
1. Uncheck **Simulation Mode**, verify IP (`192.168.0.2`), and click **Connect Robot**.
2. **Capture `home` once:** Move arm to your clear starting pose and click **Capture** next to `home`.
3. **Randomize the scene** (move pipette holder / reagent vial / stirrer vial):
   - Press **`Space`** (or click **Freedrive: Locked XYZ**) → move empty gripper onto the pipette bulb → click **Capture** for `pipette_grasp`.
   - Click **Hold Bulb (`200`)**, lift pipette out of holder, guide tip into Vial 1 → **Capture** `vial1_in`.
   - Guide tip into Vial 2 → **Capture** `vial2_in`.
   - Press **`Space`** again to exit freedrive.
4. Click **Auto-Return Pipette & Home** (unrecorded) to place the pipette back into the holder and move to `home` cleanly.
5. *(First run only)* Click **Dry Run (0.5x, No Rec)** to verify clearances before recording.
6. Click **RUN & RECORD EPISODE**! Keep hands out of view; press **`Esc`** or the red **STOP / ABORT** button anytime to halt immediately.
7. If an episode went poorly, click **Discard Last** right away.

---

## 4. Raw Episode Format

Each episode is saved independently under `<raw_data_dir>/episodes/episode_XXXX/`:
- `world_cam.mp4` — world webcam RGB video (or `.avi` if `MJPG` fallback used)
- `wrist_cam.mp4` — wrist RealSense D405 RGB video
- `wrist_depth/000000.png` and subsequent numbered files — lossless 16-bit depth aligned to the corresponding zero-based wrist RGB frame and telemetry row, saved automatically with the RealSense wrist backend. Multiply pixel values by `metadata.json` → `wrist_depth.meters_per_unit` to get metres. Zero marks invalid depth.
- `telemetry.npz` — synchronized arrays (`timestamps`, `wall_timestamps`, `q_actual`, `qd_actual`, `tcp_actual`, `gripper_raw`, `gripper_norm`, `step_index`, `phase_tags`)
- `metadata.json` — prompt, FPS, frame count, robot settings, and sequence/anchor snapshot

---

## 5. Converting Raw → LeRobot Dataset & SmolVLA Fine-Tuning

### Convert via GUI or CLI
In **Tab 4 (Settings & Export)** click **Convert Raw -> LeRobot Now**, or run the CLI on any machine:
```bash
python convert_raw_to_lerobot.py \
  --raw-dir ./vla_data/raw \
  --out-dir ./vla_data/lerobot_dataset \
  --repo-id local/ur5e_titration \
  --action-space joints
```

- **Both representations saved:** With `--action-space joints`, `observation.state` & `action` hold `[j0..j5, gripper_norm]` (7D), while `observation.state_tcp` & `action_tcp` hold `[x, y, z, rx, ry, rz, gripper_norm]` (7D). Pass `--action-space tcp` to swap which is primary.
- **Optional phase split:** Add `--split-by-phase` if you ever want to re-export the same raw episodes sliced into `grasp_pipette`, `aspirate`, and `dispense` sub-episodes.

### Starter SmolVLA Fine-Tuning Command (LeRobot)
Once converted, point `lerobot-train` at the local dataset directory (adjust batch size / steps for your GPU):
```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=local/ur5e_titration \
  --dataset.root=./vla_data/lerobot_dataset \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=./outputs/smolvla_titration
```
