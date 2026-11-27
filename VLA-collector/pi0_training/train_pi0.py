#!/usr/bin/env python3
"""Portable CLI wrapper to fine-tune Hugging Face LeRobot π₀ (lerobot/pi0_base).

Designed to be copied onto a GPU machine alongside a LeRobot v3.0 dataset
(e.g., ``lerobot_pi0_train``). It performs preflight checks on the dataset,
PyTorch/CUDA environment, and Hugging Face PaliGemma access before launching
``lerobot-train`` (or ``accelerate launch`` for multi-GPU runs).

This script has zero dependencies on ``vla_collector`` and uses only the Python
standard library at import time.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PRETRAINED_PATH = "lerobot/pi0_base"
DEFAULT_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
DEFAULT_REPO_ID = "local/ur5e_titration"
VALID_MODES = ("lora", "expert-only", "freeze-vision", "full")
VALID_VIDEO_BACKENDS = ("pyav", "torchcodec", "auto")


@dataclass
class DatasetInspection:
    """Summary of a validated local LeRobot v3.0 dataset."""

    dataset_dir: Path
    codebase_version: str
    robot_type: str
    fps: int
    total_episodes: int
    total_frames: int
    total_tasks: int
    camera_features: dict[str, dict[str, Any]]
    state_shape: list[int]
    state_names: list[str]
    action_shape: list[int]
    action_names: list[str]
    is_delta_action: bool


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version_str.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def inspect_lerobot_dataset(dataset_dir: str | Path) -> DatasetInspection:
    """Validate a local LeRobot v3.0 dataset directory and summarize its schema."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    missing_items: list[str] = []
    required_files = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
    ]
    required_dirs = [
        root / "data",
        root / "videos",
    ]
    for fpath in required_files:
        if not fpath.is_file():
            missing_items.append(str(fpath.relative_to(root)))
    for dpath in required_dirs:
        if not dpath.is_dir():
            missing_items.append(str(dpath.relative_to(root)) + "/")

    if missing_items:
        raise FileNotFoundError(
            f"Dataset at {root} is missing required LeRobot v3.0 files/folders: "
            + ", ".join(missing_items)
        )

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))

    codebase_version = str(info.get("codebase_version", ""))
    if codebase_version != "v3.0":
        raise ValueError(
            f"Expected LeRobot dataset codebase_version 'v3.0', found '{codebase_version}' in {root / 'meta' / 'info.json'}"
        )

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    total_tasks = int(info.get("total_tasks", 0))
    fps = int(info.get("fps", 0))
    robot_type = str(info.get("robot_type", "unknown"))

    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError(
            f"Dataset at {root} is empty or unfinalized "
            f"(total_episodes={total_episodes}, total_frames={total_frames})."
        )
    if fps <= 0:
        raise ValueError(f"Dataset at {root} has invalid fps={fps}.")

    features = info.get("features", {})
    if not isinstance(features, dict) or not features:
        raise ValueError(f"Dataset at {root} has no features defined in meta/info.json.")

    camera_features: dict[str, dict[str, Any]] = {}
    for key, spec in features.items():
        if not isinstance(spec, dict):
            continue
        dtype = str(spec.get("dtype", ""))
        if dtype in ("video", "image") or key.startswith("observation.images."):
            vid_info = spec.get("info") if isinstance(spec.get("info"), dict) else {}
            camera_features[key] = {
                "dtype": dtype,
                "shape": list(spec.get("shape", [])),
                "codec": vid_info.get("video.codec", "unknown"),
                "fps": vid_info.get("video.fps", fps),
            }

    if not camera_features:
        raise ValueError(
            f"No visual camera features (observation.images.*) found in {root / 'meta' / 'info.json'}."
        )

    state_spec = features.get("observation.state")
    action_spec = features.get("action")
    if not isinstance(state_spec, dict) or not isinstance(action_spec, dict):
        raise ValueError(
            f"Dataset at {root} must contain both 'observation.state' and 'action' features."
        )

    state_shape = [int(x) for x in state_spec.get("shape", [])]
    state_names = [str(x) for x in (state_spec.get("names") or [])]
    action_shape = [int(x) for x in action_spec.get("shape", [])]
    action_names = [str(x) for x in (action_spec.get("names") or [])]

    non_gripper_actions = [name for name in action_names if "gripper" not in name.lower()]
    is_delta_action = bool(non_gripper_actions) and all(
        len(name) > 1 and name.startswith("d") for name in non_gripper_actions
    )

    return DatasetInspection(
        dataset_dir=root,
        codebase_version=codebase_version,
        robot_type=robot_type,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        camera_features=camera_features,
        state_shape=state_shape,
        state_names=state_names,
        action_shape=action_shape,
        action_names=action_names,
        is_delta_action=is_delta_action,
    )


def resolve_lerobot_train_cmd(num_gpus: int = 1) -> list[str]:
    """Resolve the base launcher command for lerobot-train (single or multi-GPU)."""
    if num_gpus > 1:
        accelerate_exe = shutil.which("accelerate")
        base = [accelerate_exe] if accelerate_exe else [sys.executable, "-m", "accelerate.commands.accelerate_cli"]
        return [
            *base,
            "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "-m",
            "lerobot.scripts.lerobot_train",
        ]

    lerobot_train_exe = shutil.which("lerobot-train")
    if lerobot_train_exe:
        return [lerobot_train_exe]

    if importlib.util.find_spec("lerobot.scripts.lerobot_train") is not None:
        return [sys.executable, "-m", "lerobot.scripts.lerobot_train"]

    # Fallback name for dry-run inspection when lerobot is not installed locally.
    return ["lerobot-train"]


def _probe_first_video_pyav(dataset_dir: Path) -> str | None:
    """Decode one frame from the first video file using PyAV to catch AV1 codec issues early."""
    videos_dir = dataset_dir / "videos"
    mp4_files = sorted(videos_dir.rglob("*.mp4"))
    if not mp4_files:
        return f"No .mp4 files found under {videos_dir}"

    import av  # type: ignore

    first_mp4 = mp4_files[0]
    with av.open(str(first_mp4)) as container:
        for frame in container.decode(video=0):
            _ = frame.to_ndarray(format="rgb24")
            return None
    return f"Video file {first_mp4} yielded 0 decodable frames."


def check_environment(
    mode: str,
    video_backend: str,
    dataset_dir: Path,
    num_gpus: int = 1,
) -> tuple[list[str], list[str]]:
    """Verify installed LeRobot, pi0/peft dependencies, and video decoding support."""
    warnings: list[str] = []
    errors: list[str] = []

    lerobot_train_exe = shutil.which("lerobot-train")
    has_module = importlib.util.find_spec("lerobot.scripts.lerobot_train") is not None
    if not lerobot_train_exe and not has_module:
        errors.append(
            "LeRobot is not installed (neither 'lerobot-train' on PATH nor 'lerobot.scripts.lerobot_train' importable). "
            "Run setup_gpu_env.sh or: pip install 'lerobot[pi,peft]'"
        )
    else:
        try:
            lr_version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError:
            lr_version = "0.0.0"
        ver_tuple = _parse_version_tuple(lr_version)
        if ver_tuple and ver_tuple < (0, 5):
            warnings.append(
                f"Installed lerobot=={lr_version} is older than 0.5.0. "
                "The upstream 'lerobot/pi0_base' preprocessor uses 'relative_actions_processor', "
                "which requires a recent lerobot[pi] release on the GPU training machine."
            )

    if importlib.util.find_spec("transformers") is None:
        errors.append("Missing 'transformers'. Install with: pip install 'lerobot[pi,peft]'")
    if importlib.util.find_spec("sentencepiece") is None:
        warnings.append("Package 'sentencepiece' not found; install 'lerobot[pi]' to ensure tokenizer support.")
    if mode == "lora" and importlib.util.find_spec("peft") is None:
        errors.append("LoRA mode requires 'peft'. Install with: pip install 'lerobot[pi,peft]'")
    if num_gpus > 1 and importlib.util.find_spec("accelerate") is None:
        errors.append("Multi-GPU training requires 'accelerate'.")

    if video_backend == "pyav":
        if importlib.util.find_spec("av") is None:
            errors.append("video_backend='pyav' requires PyAV ('av'). Install with: pip install av")
        else:
            try:
                err_msg = _probe_first_video_pyav(dataset_dir)
                if err_msg:
                    warnings.append(f"PyAV video probe warning: {err_msg}")
            except Exception as exc:
                errors.append(f"PyAV failed to decode sample dataset video (check AV1/ffmpeg support): {exc}")
    elif video_backend == "torchcodec":
        if importlib.util.find_spec("torchcodec") is None:
            errors.append(
                "video_backend='torchcodec' was requested, but 'torchcodec' is not installed. "
                "Use --video-backend pyav or install torchcodec."
            )
    elif video_backend == "auto" and importlib.util.find_spec("av") is not None:
        try:
            err_msg = _probe_first_video_pyav(dataset_dir)
            if err_msg:
                warnings.append(f"PyAV video probe warning: {err_msg}")
        except Exception as exc:
            warnings.append(f"Sample video decode test with PyAV raised: {exc}")

    if errors:
        raise RuntimeError("Environment preflight failed:\n  - " + "\n  - ".join(errors))

    return resolve_lerobot_train_cmd(num_gpus=num_gpus), warnings


def check_gpu_and_dtype(
    requested_dtype: str,
    mode: str,
    batch_size: int,
    num_gpus: int,
) -> tuple[str, list[str]]:
    """Verify CUDA GPU availability, VRAM budget, and bfloat16 support."""
    warnings: list[str] = []
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available (torch.cuda.is_available() == False). "
            "Training π₀ requires an NVIDIA CUDA GPU (pass --skip-gpu-check only for CPU dry-runs)."
        )

    device_count = torch.cuda.device_count()
    if device_count < num_gpus:
        raise RuntimeError(
            f"Requested --num-gpus={num_gpus}, but only {device_count} CUDA GPU(s) were detected."
        )

    gpu_vrams_gb: list[float] = []
    gpu_names: list[str] = []
    for idx in range(num_gpus):
        props = torch.cuda.get_device_properties(idx)
        gpu_names.append(props.name)
        gpu_vrams_gb.append(props.total_memory / (1024**3))

    min_vram_gb = min(gpu_vrams_gb)
    print(
        f"[GPU] Detected {device_count} GPU(s); using {num_gpus}: "
        + ", ".join(f"{name} ({vram:.1f} GB)" for name, vram in zip(gpu_names, gpu_vrams_gb))
    )

    resolved_dtype = requested_dtype
    if requested_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        warnings.append(
            "GPU does not natively support bfloat16 (Compute Capability < 8.0). "
            "Falling back to --policy.dtype=float32."
        )
        resolved_dtype = "float32"

    if mode == "full" and min_vram_gb < 23.0:
        warnings.append(
            f"Full fine-tuning on {min_vram_gb:.1f} GB VRAM is likely to OOM. "
            "Consider '--mode lora' or '--mode expert-only' with '--batch-size 4' or '8'."
        )
    elif mode == "full" and min_vram_gb < 32.0 and batch_size > 2:
        warnings.append(
            f"Full fine-tuning on {min_vram_gb:.1f} GB VRAM with batch_size={batch_size} is tight. "
            "If you hit CUDA OOM, lower --batch-size to 1 or 2, or use '--mode lora'."
        )
    elif mode in ("lora", "expert-only") and min_vram_gb < 16.0 and batch_size > 4:
        warnings.append(
            f"{mode} on {min_vram_gb:.1f} GB VRAM with batch_size={batch_size} may be tight; "
            "try --batch-size 2 or 4 if you hit CUDA OOM."
        )

    return resolved_dtype, warnings


def check_huggingface_access(
    pretrained_path: str,
    tokenizer_repo: str = DEFAULT_TOKENIZER_REPO,
) -> list[str]:
    """Check Hugging Face Hub access to the base model and gated PaliGemma tokenizer."""
    warnings: list[str] = []
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run setup_gpu_env.sh or pip install 'lerobot[pi,peft]'."
        ) from exc

    api = HfApi()
    repos_to_check = [tokenizer_repo]
    if not Path(pretrained_path).expanduser().exists():
        repos_to_check.append(pretrained_path)

    for repo_id in repos_to_check:
        try:
            api.model_info(repo_id)
        except Exception as exc:
            msg = str(exc)
            exc_name = type(exc).__name__
            if any(tok in msg or tok in exc_name for tok in ("401", "403", "GatedRepo", "Unauthorized", "Access")):
                raise RuntimeError(
                    f"Cannot access Hugging Face repo '{repo_id}' ({exc_name}).\n"
                    f"  1. Accept the model license at: https://huggingface.co/{repo_id}\n"
                    f"  2. Log in on this machine:      hf auth login   (or export HF_TOKEN=hf_...)"
                ) from exc
            raise RuntimeError(
                f"Failed to verify Hugging Face repo '{repo_id}': {exc}. "
                "If running fully offline with cached weights, pass --offline or --skip-hf-check."
            ) from exc

    return warnings


def resolve_resume_config_path(resume_target: str | Path) -> Path:
    """Resolve a run directory or train_config.json path for resuming training."""
    target = Path(resume_target).expanduser().resolve()
    if target.is_file():
        if target.name != "train_config.json":
            raise ValueError(f"Expected train_config.json when passing a file to --resume, got: {target}")
        return target

    if not target.is_dir():
        raise FileNotFoundError(f"Resume path does not exist: {target}")

    candidates = [
        target / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
        target / "pretrained_model" / "train_config.json",
        target / "train_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checkpoints_dir = target / "checkpoints"
    if checkpoints_dir.is_dir():
        step_dirs = sorted(
            [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
            reverse=True,
        )
        for step_dir in step_dirs:
            cfg = step_dir / "pretrained_model" / "train_config.json"
            if cfg.is_file():
                return cfg

    raise FileNotFoundError(
        f"Could not locate checkpoints/last/pretrained_model/train_config.json under {target}"
    )


def default_output_dir(mode: str, base_dir: str | Path = "outputs") -> Path:
    """Generate a collision-free timestamped output directory path (without creating it)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir).expanduser().resolve() / f"pi0_{mode}_{stamp}"


def detect_tracker_backend() -> str:
    """Detect whether the installed LeRobot version uses Trackio (0.7+) or legacy W&B."""
    try:
        from lerobot.configs.train import TrainPipelineConfig  # type: ignore

        fields = getattr(TrainPipelineConfig, "__dataclass_fields__", {})
        if "trackio" in fields:
            return "trackio"
        if "wandb" in fields:
            return "wandb"
    except Exception:
        pass
    return "trackio"


def build_train_command(
    args: argparse.Namespace,
    resolved_dtype: str,
    output_dir: Path | None,
    lerobot_train_cmd: Sequence[str],
    tracker_backend: str | None = None,
) -> list[str]:
    """Construct the exact lerobot-train (or accelerate launch) command line."""
    cmd = list(lerobot_train_cmd)

    if getattr(args, "resume", None):
        config_path = resolve_resume_config_path(args.resume)
        cmd.extend([
            f"--config_path={config_path}",
            "--resume=true",
        ])
        if getattr(args, "steps", None) is not None and getattr(args, "steps_explicit", False):
            cmd.append(f"--steps={args.steps}")
        for extra in getattr(args, "extra_arg", []) or []:
            cmd.append(extra)
        return cmd

    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--policy-repo-id is required when --push-to-hub is enabled.")
    if args.n_action_steps > args.chunk_size:
        raise ValueError(
            f"--n-action-steps ({args.n_action_steps}) cannot exceed --chunk-size ({args.chunk_size})."
        )
    if output_dir is None:
        raise ValueError("output_dir must be provided for a non-resume training run.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "LeRobot refuses to overwrite existing output_dir; choose a new --output-dir or pass --resume."
        )

    job_name = args.job_name or f"pi0_{args.mode}"

    cmd.extend([
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
    ])
    if args.video_backend != "auto":
        cmd.append(f"--dataset.video_backend={args.video_backend}")

    cmd.extend([
        "--policy.type=pi0",
        f"--policy.pretrained_path={args.pretrained_path}",
        f"--policy.device={args.device}",
        f"--policy.dtype={resolved_dtype}",
        f"--policy.gradient_checkpointing={'true' if args.gradient_checkpointing else 'false'}",
        f"--policy.compile_model={'true' if args.compile_model else 'false'}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}",
    ])

    if args.push_to_hub and args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    if args.empty_cameras > 0:
        cmd.append(f"--policy.empty_cameras={args.empty_cameras}")
    if args.lr is not None:
        cmd.append(f"--policy.optimizer_lr={args.lr}")
    if args.use_relative_actions:
        cmd.append("--policy.use_relative_actions=true")
        cmd.append('--policy.relative_exclude_joints=["gripper"]')

    if args.mode == "lora":
        cmd.extend([
            "--peft.method_type=LORA",
            f"--peft.r={args.lora_r}",
        ])
        if args.lora_target_modules:
            cmd.append(f"--peft.target_modules={args.lora_target_modules}")
    elif args.mode == "expert-only":
        cmd.extend([
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
        ])
    elif args.mode == "freeze-vision":
        cmd.extend([
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=false",
        ])
    elif args.mode == "full":
        cmd.extend([
            "--policy.freeze_vision_encoder=false",
            "--policy.train_expert_only=false",
        ])
    else:
        raise ValueError(f"Unsupported mode '{args.mode}'. Expected one of {VALID_MODES}.")

    cmd.extend([
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ])

    backend = tracker_backend or detect_tracker_backend()
    if getattr(args, "trackio", False):
        if backend == "trackio":
            cmd.append("--trackio.enable=true")
            cmd.append(f"--trackio.project={args.trackio_project}")
            if getattr(args, "trackio_space_id", None):
                cmd.append(f"--trackio.space_id={args.trackio_space_id}")
        else:
            cmd.append("--wandb.enable=true")
            cmd.append(f"--wandb.project={args.trackio_project}")
    elif getattr(args, "wandb", False):
        if backend == "wandb":
            cmd.append("--wandb.enable=true")
            cmd.append(f"--wandb.project={args.wandb_project}")
            if getattr(args, "wandb_entity", None):
                cmd.append(f"--wandb.entity={args.wandb_entity}")
        else:
            cmd.append("--trackio.enable=true")
            cmd.append(f"--trackio.project={args.wandb_project}")
            if getattr(args, "trackio_space_id", None):
                cmd.append(f"--trackio.space_id={args.trackio_space_id}")

    for extra in args.extra_arg or []:
        cmd.append(extra)

    return cmd


def _format_command_pretty(cmd: Sequence[str]) -> str:
    return " \\\n  ".join(shlex.quote(str(part)) for part in cmd)


def _print_dataset_summary(inspection: DatasetInspection) -> None:
    print(f"[Dataset] {inspection.dataset_dir}")
    print(
        f"  format={inspection.codebase_version}, robot_type={inspection.robot_type}, "
        f"episodes={inspection.total_episodes}, frames={inspection.total_frames}, "
        f"tasks={inspection.total_tasks}, fps={inspection.fps}"
    )
    for cam_key, cam_info in inspection.camera_features.items():
        print(
            f"  camera '{cam_key}': shape={cam_info['shape']}, "
            f"dtype={cam_info['dtype']}, codec={cam_info['codec']}"
        )
    print(f"  state : shape={inspection.state_shape}, names={inspection.state_names}")
    print(
        f"  action: shape={inspection.action_shape}, names={inspection.action_names} "
        f"({'delta' if inspection.is_delta_action else 'absolute'})"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portable Hugging Face LeRobot π₀ (lerobot/pi0_base) fine-tuning launcher."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to local LeRobot v3.0 dataset directory (required unless --resume is used).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Dataset repo_id metadata string (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=DEFAULT_PRETRAINED_PATH,
        help=f"Base π₀ checkpoint or local path (default: {DEFAULT_PRETRAINED_PATH}).",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="lora",
        help="Fine-tuning strategy: lora (default, ~12-16 GB VRAM), expert-only, freeze-vision, or full.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when --mode=lora (default: 16).")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Optional override for --peft.target_modules.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size (default: 8).")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps (default: 10000).")
    parser.add_argument("--save-freq", type=int, default=2000, help="Checkpoint interval in steps (default: 2000).")
    parser.add_argument("--log-freq", type=int, default=100, help="Console/W&B log interval in steps (default: 100).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count (default: 4).")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed (default: 1000).")
    parser.add_argument("--lr", type=float, default=None, help="Optional override for --policy.optimizer_lr.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk horizon (default: 50).")
    parser.add_argument("--n-action-steps", type=int, default=50, help="Executed action steps per chunk (default: 50).")
    parser.add_argument(
        "--empty-cameras",
        type=int,
        default=0,
        help="Optional number of dummy empty cameras to pad (default: 0).",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Model precision (default: bfloat16; auto-falls back to float32 on pre-Ampere GPUs).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target training device (default: cuda).")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (>1 uses accelerate launch).")
    parser.add_argument(
        "--video-backend",
        choices=VALID_VIDEO_BACKENDS,
        default="pyav",
        help="LeRobot video decoding backend (default: pyav for reliable AV1 decoding).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation gradient checkpointing (default: enabled).",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=False,
        help="Enable torch.compile on the π₀ model (off by default).",
    )
    parser.add_argument(
        "--use-relative-actions",
        action="store_true",
        default=False,
        help="Enable LeRobot relative_actions transform (ONLY use if dataset actions are absolute!).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (must not exist yet; defaults to ./outputs/pi0_<mode>_<timestamp>).",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Job name for logs/tracker (default: pi0_<mode>).")
    parser.add_argument("--resume", type=str, default=None, help="Resume an existing run dir or train_config.json.")
    parser.add_argument("--trackio", action="store_true", default=False, help="Enable Trackio experiment tracking (LeRobot 0.7+).")
    parser.add_argument("--trackio-project", type=str, default="ur5e-pi0", help="Trackio project name (default: ur5e-pi0).")
    parser.add_argument(
        "--trackio-space-id",
        type=str,
        default=None,
        help="Optional Hugging Face Space ID (e.g. 'user/pi0-logs') to sync Trackio live.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Legacy W&B flag (mapped to Trackio on LeRobot 0.7+).",
    )
    parser.add_argument("--wandb-project", type=str, default="ur5e-pi0", help="Legacy W&B project name (default: ur5e-pi0).")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional legacy W&B entity/team.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push final trained policy to Hugging Face Hub.",
    )
    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default=None,
        help="Target HF Hub repo ID (required when --push-to-hub is set).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw flag forwarded to lerobot-train (repeatable, e.g. --extra-arg=--dataset.image_transforms.enable=true).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run preflight checks and print command without launching.")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 and skip HF Hub checks.")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip lerobot/video backend environment checks.")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip CUDA GPU availability/VRAM checks.")
    parser.add_argument("--skip-hf-check", action="store_true", help="Skip Hugging Face Hub access verification.")

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args.steps_explicit = any(tok == "--steps" or tok.startswith("--steps=") for tok in raw_argv)
    return args


def _write_launch_manifest(
    output_dir: Path | None,
    cmd: Sequence[str],
    args: argparse.Namespace,
    inspection: DatasetInspection | None,
    resolved_dtype: str,
    exit_code: int,
) -> None:
    if output_dir is None or not output_dir.is_dir():
        return
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "mode": args.mode,
        "resolved_dtype": resolved_dtype,
        "pretrained_path": args.pretrained_path,
        "command": list(cmd),
        "dataset": {
            "dir": str(inspection.dataset_dir),
            "codebase_version": inspection.codebase_version,
            "robot_type": inspection.robot_type,
            "fps": inspection.fps,
            "total_episodes": inspection.total_episodes,
            "total_frames": inspection.total_frames,
            "camera_features": inspection.camera_features,
            "state_names": inspection.state_names,
            "action_names": inspection.action_names,
            "is_delta_action": inspection.is_delta_action,
        }
        if inspection
        else None,
    }
    manifest_path = output_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Manifest] Saved {manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    all_warnings: list[str] = []

    if args.num_gpus < 1:
        print("Error: --num-gpus must be >= 1.", file=sys.stderr)
        return 2

    if not args.resume and not args.dataset_dir:
        print("Error: --dataset-dir is required unless --resume is specified.", file=sys.stderr)
        return 2

    inspection: DatasetInspection | None = None
    if args.dataset_dir:
        try:
            inspection = inspect_lerobot_dataset(args.dataset_dir)
        except Exception as exc:
            print(f"[Dataset Error] {exc}", file=sys.stderr)
            return 1
        _print_dataset_summary(inspection)

        if inspection.is_delta_action and args.use_relative_actions:
            all_warnings.append(
                f"Dataset action names {inspection.action_names} already look like delta actions, "
                "but --use-relative-actions was set! Enabling relative actions on already-delta data "
                "will double-difference your actions."
            )

    dataset_dir_for_probe = inspection.dataset_dir if inspection else Path(".")
    if args.skip_env_check:
        lerobot_train_cmd = resolve_lerobot_train_cmd(num_gpus=args.num_gpus)
    else:
        try:
            lerobot_train_cmd, env_warnings = check_environment(
                mode=args.mode,
                video_backend=args.video_backend,
                dataset_dir=dataset_dir_for_probe,
                num_gpus=args.num_gpus,
            )
            all_warnings.extend(env_warnings)
        except Exception as exc:
            print(f"[Environment Error] {exc}", file=sys.stderr)
            return 1

    resolved_dtype = args.dtype
    if not args.skip_gpu_check:
        try:
            resolved_dtype, gpu_warnings = check_gpu_and_dtype(
                requested_dtype=args.dtype,
                mode=args.mode,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus,
            )
            all_warnings.extend(gpu_warnings)
        except Exception as exc:
            print(f"[GPU Error] {exc}", file=sys.stderr)
            return 1

    if not args.resume and not args.skip_hf_check and not args.offline:
        try:
            hf_warnings = check_huggingface_access(args.pretrained_path)
            all_warnings.extend(hf_warnings)
        except Exception as exc:
            print(f"[Hugging Face Auth Error] {exc}", file=sys.stderr)
            return 1

    output_dir: Path | None = None
    if not args.resume:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else default_output_dir(args.mode)
        )

    tracker_backend = detect_tracker_backend()
    if args.wandb and tracker_backend == "trackio":
        all_warnings.append(
            "LeRobot 0.7+ replaced Weights & Biases with Trackio; mapping --wandb to --trackio.enable=true."
        )
    elif args.trackio and tracker_backend == "wandb":
        all_warnings.append(
            "Installed LeRobot uses legacy W&B instead of Trackio; mapping --trackio to --wandb.enable=true."
        )

    try:
        cmd = build_train_command(
            args=args,
            resolved_dtype=resolved_dtype,
            output_dir=output_dir,
            lerobot_train_cmd=lerobot_train_cmd,
            tracker_backend=tracker_backend,
        )
    except Exception as exc:
        print(f"[Config Error] {exc}", file=sys.stderr)
        return 1

    for warn in all_warnings:
        print(f"[WARNING] {warn}", file=sys.stderr)

    print("\n[Command]")
    print(_format_command_pretty(cmd))

    if args.dry_run:
        print("\n[Dry Run] Preflight passed. Exiting without launching training.")
        return 0

    env = os.environ.copy()
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    print("\n[Launching lerobot-train]\n")
    proc = subprocess.run(cmd, env=env)
    if not args.resume:
        _write_launch_manifest(
            output_dir=output_dir,
            cmd=cmd,
            args=args,
            inspection=inspection,
            resolved_dtype=resolved_dtype,
            exit_code=proc.returncode,
        )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
