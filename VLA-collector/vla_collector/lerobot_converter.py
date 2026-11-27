"""Convert raw collected episodes into a LeRobotDataset (Joints, TCP, or XYZ+Gripper)."""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .recorder import list_raw_episodes

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False


DEFAULT_PHASE_PROMPTS: Dict[str, str] = {
    "grasp_pipette": "Pick up the pipette from the holder",
    "aspirate": "Aspirate liquid from the reagent vial",
    "dispense": "Dispense liquid into the target vial",
}


@dataclass
class EpisodeSegment:
    task_prompt: str
    frame_indices: np.ndarray
    joints_state: np.ndarray  # (T, 7) float32
    joints_action: np.ndarray  # (T, 7) float32
    tcp_state: np.ndarray  # (T, 7) float32
    tcp_action: np.ndarray  # (T, 7) float32
    xyz_state: np.ndarray  # (T, 4) float32
    xyz_action: np.ndarray  # (T, 4) float32


@dataclass
class ConversionSummary:
    episodes_converted: int
    segments_written: int
    total_frames: int
    fps: int
    action_space: str
    output_dir: Path
    action_mode: str = "absolute"
    gripper_units: str = "norm"


def _import_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        pass
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not installed in this environment. "
            "Install 'lerobot' (see requirements-convert.txt) before converting."
        ) from exc


def _compute_actions(states: np.ndarray, action_mode: str = "absolute") -> np.ndarray:
    """Compute next-step absolute or delta+absolute-gripper actions within a segment."""
    if action_mode not in ("absolute", "delta"):
        raise ValueError("action_mode must be 'absolute' or 'delta'.")
    if len(states) == 0:
        return states.copy()

    actions = np.empty_like(states)
    if action_mode == "absolute":
        if len(states) > 1:
            actions[:-1] = states[1:]
        actions[-1] = states[-1]
        return actions

    # Delta mode: pose/joint dims are step deltas (0 at final frame); gripper is absolute target.
    if len(states) > 1:
        actions[:-1, :-1] = states[1:, :-1] - states[:-1, :-1]
        actions[:-1, -1] = states[1:, -1]
    actions[-1, :-1] = 0.0
    actions[-1, -1] = states[-1, -1]
    return actions


def build_episode_segments(
    episode_dir: str | Path,
    split_by_phase: bool = False,
    phase_prompts: Optional[Dict[str, str]] = None,
    action_mode: str = "absolute",
    gripper_units: str = "norm",
) -> Tuple[dict, List[EpisodeSegment]]:
    """Load telemetry + metadata from one raw episode and prepare state/action segments."""
    if action_mode not in ("absolute", "delta"):
        raise ValueError("action_mode must be 'absolute' or 'delta'.")
    if gripper_units not in ("norm", "raw"):
        raise ValueError("gripper_units must be 'norm' or 'raw'.")

    ep_path = Path(episode_dir)
    meta = json.loads((ep_path / "metadata.json").read_text(encoding="utf-8"))
    gripper_key = "gripper_raw" if gripper_units == "raw" else "gripper_norm"
    with np.load(ep_path / "telemetry.npz", allow_pickle=False) as telem:
        q_actual = np.asarray(telem["q_actual"], dtype=np.float32)
        tcp_actual = np.asarray(telem["tcp_actual"], dtype=np.float32)
        gripper_col = np.asarray(telem[gripper_key], dtype=np.float32).reshape(-1, 1)
        phase_tags = np.asarray(telem["phase_tags"], dtype=str)

    joints_full = np.concatenate([q_actual, gripper_col], axis=1).astype(np.float32)
    tcp_full = np.concatenate([tcp_actual, gripper_col], axis=1).astype(np.float32)
    xyz_full = np.concatenate([tcp_actual[:, :3], gripper_col], axis=1).astype(np.float32)
    total_n = len(joints_full)

    prompts_map = phase_prompts or DEFAULT_PHASE_PROMPTS
    default_prompt = str(meta.get("task_prompt", "Perform robotic titration"))

    if not split_by_phase or total_n == 0:
        indices = np.arange(total_n, dtype=np.int64)
        seg = EpisodeSegment(
            task_prompt=default_prompt,
            frame_indices=indices,
            joints_state=joints_full,
            joints_action=_compute_actions(joints_full, action_mode=action_mode),
            tcp_state=tcp_full,
            tcp_action=_compute_actions(tcp_full, action_mode=action_mode),
            xyz_state=xyz_full,
            xyz_action=_compute_actions(xyz_full, action_mode=action_mode),
        )
        return meta, [seg]

    # Contiguous phase runs
    segments: List[EpisodeSegment] = []
    start = 0
    while start < total_n:
        tag = str(phase_tags[start])
        end = start + 1
        while end < total_n and str(phase_tags[end]) == tag:
            end += 1
        indices = np.arange(start, end, dtype=np.int64)
        j_slice = joints_full[start:end]
        t_slice = tcp_full[start:end]
        x_slice = xyz_full[start:end]
        prompt = prompts_map.get(tag, tag or default_prompt)
        segments.append(
            EpisodeSegment(
                task_prompt=prompt,
                frame_indices=indices,
                joints_state=j_slice,
                joints_action=_compute_actions(j_slice, action_mode=action_mode),
                tcp_state=t_slice,
                tcp_action=_compute_actions(t_slice, action_mode=action_mode),
                xyz_state=x_slice,
                xyz_action=_compute_actions(x_slice, action_mode=action_mode),
            )
        )
        start = end

    return meta, segments


def _crop_image(rgb: np.ndarray, crop_xyxy: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop_xyxy is None:
        return rgb
    x0, y0, x1, y1 = crop_xyxy
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"Crop box {crop_xyxy} is out of bounds for frame dimensions {(w, h)}."
        )
    return rgb[y0:y1, x0:x1]


def _resize_with_pad(rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize centered on a black canvas of target_hw=(H, W)."""
    target_h, target_w = target_hw
    h, w = rgb.shape[:2]
    if (h, w) == (target_h, target_w):
        return rgb
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def _resize_image(
    rgb: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    resize_mode: str = "stretch",
) -> np.ndarray:
    if resize_hw is None:
        return rgb
    if resize_mode == "pad":
        return _resize_with_pad(rgb, resize_hw)
    if resize_mode == "stretch":
        h, w = resize_hw
        if rgb.shape[0] != h or rgb.shape[1] != w:
            interp = cv2.INTER_AREA if (h < rgb.shape[0] or w < rgb.shape[1]) else cv2.INTER_LINEAR
            return cv2.resize(rgb, (w, h), interpolation=interp)
        return rgb
    raise ValueError("resize_mode must be 'stretch' or 'pad'.")


def _read_all_video_frames_rgb(
    video_path: Path,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_xyxy: Optional[Tuple[int, int, int, int]] = None,
    resize_mode: str = "stretch",
) -> List[np.ndarray]:
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required to read raw episode videos.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _crop_image(rgb, crop_xyxy)
            rgb = _resize_image(rgb, resize_hw, resize_mode=resize_mode)
            frames.append(np.ascontiguousarray(rgb, dtype=np.uint8))
    finally:
        cap.release()
    return frames


def _create_lerobot_dataset(
    dataset_cls: Any,
    repo_id: str,
    out_dir: Path,
    fps: int,
    features: Dict[str, Any],
):
    sig = inspect.signature(dataset_cls.create)
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "fps": int(fps),
        "features": features,
    }
    if "root" in sig.parameters:
        kwargs["root"] = out_dir
    if "robot_type" in sig.parameters:
        kwargs["robot_type"] = "ur5e_robotiq"
    if "use_videos" in sig.parameters:
        kwargs["use_videos"] = True
    return dataset_cls.create(**kwargs)


def _add_frame_compat(dataset: Any, frame_dict: Dict[str, Any], task_prompt: str) -> None:
    sig = inspect.signature(dataset.add_frame)
    if "task" in sig.parameters:
        dataset.add_frame(frame_dict, task=task_prompt)
    else:
        frame_with_task = dict(frame_dict)
        frame_with_task["task"] = task_prompt
        dataset.add_frame(frame_with_task)


def _save_episode_compat(dataset: Any, task_prompt: str) -> None:
    sig = inspect.signature(dataset.save_episode)
    if "task" in sig.parameters:
        dataset.save_episode(task=task_prompt)
    else:
        dataset.save_episode()


def _filter_episodes(
    episodes,
    include_episodes: Optional[Sequence[int]],
    exclude_episodes: Optional[Sequence[int]],
):
    selected = list(episodes)
    if include_episodes is not None:
        wanted = set(int(idx) for idx in include_episodes)
        found_ids = {ep.episode_index for ep in selected}
        missing = sorted(wanted - found_ids)
        if missing:
            raise ValueError(f"Requested episode indices not found: {missing}")
        selected = [ep for ep in selected if ep.episode_index in wanted]
    if exclude_episodes is not None:
        excluded = set(int(idx) for idx in exclude_episodes)
        selected = [ep for ep in selected if ep.episode_index not in excluded]
    return selected


def convert_raw_to_lerobot(
    raw_dir: str | Path,
    out_dir: str | Path,
    repo_id: str = "local/ur5e_titration",
    action_space: str = "joints",
    split_by_phase: bool = False,
    resize_hw: Optional[Tuple[int, int]] = None,
    overwrite: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    dataset_cls: Any = None,
    action_mode: str = "absolute",
    gripper_units: str = "norm",
    include_secondary: bool = False,
    include_episodes: Optional[Sequence[int]] = None,
    exclude_episodes: Optional[Sequence[int]] = None,
    world_crop_xyxy: Optional[Tuple[int, int, int, int]] = None,
    resize_mode: str = "stretch",
) -> ConversionSummary:
    """Convert raw episode folders into a unified LeRobotDataset."""
    if action_space not in ("joints", "tcp", "xyz_gripper"):
        raise ValueError("action_space must be 'joints', 'tcp', or 'xyz_gripper'.")
    if action_mode not in ("absolute", "delta"):
        raise ValueError("action_mode must be 'absolute' or 'delta'.")
    if gripper_units not in ("norm", "raw"):
        raise ValueError("gripper_units must be 'norm' or 'raw'.")
    if resize_mode not in ("stretch", "pad"):
        raise ValueError("resize_mode must be 'stretch' or 'pad'.")

    episodes = _filter_episodes(
        list_raw_episodes(raw_dir),
        include_episodes=include_episodes,
        exclude_episodes=exclude_episodes,
    )
    if not episodes:
        raise FileNotFoundError(f"No valid raw episodes selected under {Path(raw_dir) / 'episodes'}")

    ds_cls = dataset_cls or _import_lerobot_dataset_cls()
    target_root = Path(out_dir)
    if target_root.exists():
        if overwrite:
            shutil.rmtree(target_root)
        else:
            raise FileExistsError(f"Output directory already exists: {target_root}")

    first_meta = json.loads((episodes[0].episode_dir / "metadata.json").read_text(encoding="utf-8"))
    fps = int(first_meta.get("fps", 20))
    if resize_hw is not None:
        world_shape = (resize_hw[0], resize_hw[1], 3)
        wrist_shape = (resize_hw[0], resize_hw[1], 3)
    else:
        w_res = first_meta.get("world_resolution", [640, 480])
        r_res = first_meta.get("wrist_resolution", [640, 480])
        if world_crop_xyxy is not None:
            x0, y0, x1, y1 = world_crop_xyxy
            world_shape = (int(y1 - y0), int(x1 - x0), 3)
        else:
            world_shape = (int(w_res[1]), int(w_res[0]), 3)
        wrist_shape = (int(r_res[1]), int(r_res[0]), 3)

    joint_state_names = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
    joint_action_names = (
        ["dj0", "dj1", "dj2", "dj3", "dj4", "dj5", "gripper"]
        if action_mode == "delta"
        else joint_state_names
    )
    tcp_state_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    tcp_action_names = (
        ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        if action_mode == "delta"
        else tcp_state_names
    )
    xyz_state_names = ["x", "y", "z", "gripper"]
    xyz_action_names = (
        ["dx", "dy", "dz", "gripper"] if action_mode == "delta" else xyz_state_names
    )

    if action_space == "xyz_gripper":
        primary_state_names = xyz_state_names
        primary_action_names = xyz_action_names
        primary_dim = 4
    elif action_space == "joints":
        primary_state_names = joint_state_names
        primary_action_names = joint_action_names
        primary_dim = 7
    else:
        primary_state_names = tcp_state_names
        primary_action_names = tcp_action_names
        primary_dim = 7

    features: Dict[str, Any] = {
        "observation.images.world": {
            "dtype": "video",
            "shape": world_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": wrist_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (primary_dim,),
            "names": primary_state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (primary_dim,),
            "names": primary_action_names,
        },
    }

    secondary_suffix: Optional[str] = None
    if include_secondary and action_space in ("joints", "tcp"):
        secondary_suffix = "tcp" if action_space == "joints" else "joints"
        sec_state_names = tcp_state_names if action_space == "joints" else joint_state_names
        sec_action_names = tcp_action_names if action_space == "joints" else joint_action_names
        features[f"observation.state_{secondary_suffix}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": sec_state_names,
        }
        features[f"action_{secondary_suffix}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": sec_action_names,
        }

    dataset = _create_lerobot_dataset(
        dataset_cls=ds_cls,
        repo_id=repo_id,
        out_dir=target_root,
        fps=fps,
        features=features,
    )

    total_frames = 0
    segments_written = 0
    for idx, ep in enumerate(episodes):
        if progress_cb is not None:
            progress_cb(
                f"Converting episode {idx + 1}/{len(episodes)}: {ep.episode_dir.name} ({ep.num_frames} frames)..."
            )
        meta, segments = build_episode_segments(
            ep.episode_dir,
            split_by_phase=split_by_phase,
            action_mode=action_mode,
            gripper_units=gripper_units,
        )
        world_video = ep.episode_dir / str(meta.get("world_video", "world_cam.mp4"))
        wrist_video = ep.episode_dir / str(meta.get("wrist_video", "wrist_cam.mp4"))

        world_frames = _read_all_video_frames_rgb(
            world_video,
            resize_hw=resize_hw,
            crop_xyxy=world_crop_xyxy,
            resize_mode=resize_mode,
        )
        wrist_frames = _read_all_video_frames_rgb(
            wrist_video,
            resize_hw=resize_hw,
            crop_xyxy=None,
            resize_mode=resize_mode,
        )
        n_telem = sum(len(seg.frame_indices) for seg in segments)
        if len(world_frames) != n_telem or len(wrist_frames) != n_telem:
            raise RuntimeError(
                f"Frame mismatch in {ep.episode_dir.name}: "
                f"telem={n_telem}, world={len(world_frames)}, wrist={len(wrist_frames)}"
            )

        for seg in segments:
            for local_i, global_i in enumerate(seg.frame_indices):
                if action_space == "xyz_gripper":
                    primary_s = seg.xyz_state[local_i]
                    primary_a = seg.xyz_action[local_i]
                elif action_space == "joints":
                    primary_s = seg.joints_state[local_i]
                    primary_a = seg.joints_action[local_i]
                else:
                    primary_s = seg.tcp_state[local_i]
                    primary_a = seg.tcp_action[local_i]

                frame_dict: Dict[str, Any] = {
                    "observation.images.world": world_frames[int(global_i)],
                    "observation.images.wrist": wrist_frames[int(global_i)],
                    "observation.state": primary_s,
                    "action": primary_a,
                }
                if secondary_suffix is not None:
                    secondary_s = (
                        seg.tcp_state[local_i]
                        if action_space == "joints"
                        else seg.joints_state[local_i]
                    )
                    secondary_a = (
                        seg.tcp_action[local_i]
                        if action_space == "joints"
                        else seg.joints_action[local_i]
                    )
                    frame_dict[f"observation.state_{secondary_suffix}"] = secondary_s
                    frame_dict[f"action_{secondary_suffix}"] = secondary_a

                _add_frame_compat(dataset, frame_dict, seg.task_prompt)
                total_frames += 1

            _save_episode_compat(dataset, seg.task_prompt)
            segments_written += 1

    finalize_fn = getattr(dataset, "finalize", None)
    if callable(finalize_fn):
        finalize_fn()

    if progress_cb is not None:
        progress_cb(
            f"Done! Converted {len(episodes)} episodes ({segments_written} LeRobot episodes, {total_frames} frames)."
        )

    return ConversionSummary(
        episodes_converted=len(episodes),
        segments_written=segments_written,
        total_frames=total_frames,
        fps=fps,
        action_space=action_space,
        output_dir=target_root,
        action_mode=action_mode,
        gripper_units=gripper_units,
    )
