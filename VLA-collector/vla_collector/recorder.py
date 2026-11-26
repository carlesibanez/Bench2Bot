"""Fixed-FPS raw episode recorder storing videos, wrist depth, and robot telemetry."""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cameras import BaseCamera, CameraFrame
from .config import RecordingSettings, RobotSettings, SequencePreset
from .robot import RobotBackend

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False


@dataclass
class EpisodeInfo:
    episode_dir: Path
    episode_index: int
    num_frames: int
    duration_s: float
    fps: int
    task_prompt: str
    created_at: str


_EP_PATTERN = re.compile(r"^episode_(\d+)$")


def _episodes_root(raw_data_dir: str | Path) -> Path:
    return Path(raw_data_dir) / "episodes"


def list_raw_episodes(raw_data_dir: str | Path) -> List[EpisodeInfo]:
    """Scan raw_data_dir/episodes and return sorted EpisodeInfo metadata."""
    root = _episodes_root(raw_data_dir)
    if not root.exists():
        return []

    found: List[EpisodeInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        match = _EP_PATTERN.match(child.name)
        if not match:
            continue
        meta_path = child / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            found.append(
                EpisodeInfo(
                    episode_dir=child,
                    episode_index=int(meta.get("episode_index", int(match.group(1)))),
                    num_frames=int(meta.get("num_frames", 0)),
                    duration_s=float(meta.get("duration_s", 0.0)),
                    fps=int(meta.get("fps", 20)),
                    task_prompt=str(meta.get("task_prompt", "")),
                    created_at=str(meta.get("created_at", "")),
                )
            )
        except Exception:
            continue
    return sorted(found, key=lambda ep: ep.episode_index)


def delete_episode(episode_dir: str | Path) -> bool:
    path = Path(episode_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def _next_episode_index(raw_data_dir: str | Path) -> int:
    root = _episodes_root(raw_data_dir)
    if not root.exists():
        return 0
    max_idx = -1
    for child in root.iterdir():
        if child.is_dir():
            m = _EP_PATTERN.match(child.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _open_video_writer(
    base_path_no_ext: Path,
    width: int,
    height: int,
    fps: int,
    preferred_codec: str,
) -> Tuple[object, Path, str]:
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for recording video.")

    candidates: List[Tuple[str, str]] = []
    codec = (preferred_codec or "mp4v").strip()
    if codec.upper() == "MJPG":
        candidates.append(("MJPG", ".avi"))
        candidates.append(("mp4v", ".mp4"))
    else:
        candidates.append((codec, ".mp4"))
        candidates.append(("MJPG", ".avi"))

    for fourcc_str, ext in candidates:
        out_path = base_path_no_ext.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer, out_path, fourcc_str
        writer.release()

    raise RuntimeError(f"Failed to open VideoWriter for {base_path_no_ext}")


class EpisodeRecorder:
    """Fixed-rate synchronized recorder for robot telemetry + world/wrist video."""

    def __init__(
        self,
        robot: RobotBackend,
        world_cam: BaseCamera,
        wrist_cam: BaseCamera,
        recording_settings: RecordingSettings,
    ):
        self.robot = robot
        self.world_cam = world_cam
        self.wrist_cam = wrist_cam
        self.settings = recording_settings

        self._active_lock = threading.Lock()
        self._recording_active = False
        self._step_index = -1
        self._step_name = ""
        self._phase_tag = ""

        self._stop_event = threading.Event()
        self._sample_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._frame_queue: Optional[queue.Queue] = None

        self._episode_dir: Optional[Path] = None
        self._episode_index: int = -1
        self._task_prompt: str = ""
        self._preset_snapshot: Dict = {}
        self._robot_settings_snapshot: Dict = {}
        self._world_video_path: Optional[Path] = None
        self._wrist_video_path: Optional[Path] = None
        self._wrist_depth_dir: Optional[Path] = None
        self._wrist_depth_scale_m: Optional[float] = None
        self._world_writer = None
        self._wrist_writer = None
        self._video_codec_used: str = ""
        self._world_wh: Tuple[int, int] = (640, 480)
        self._wrist_wh: Tuple[int, int] = (640, 480)

        self._timestamps: List[float] = []
        self._wall_timestamps: List[float] = []
        self._q_actual: List[np.ndarray] = []
        self._qd_actual: List[np.ndarray] = []
        self._tcp_actual: List[np.ndarray] = []
        self._gripper_raw: List[float] = []
        self._gripper_norm: List[float] = []
        self._step_indices: List[int] = []
        self._phase_tags: List[str] = []

        self._written_world = 0
        self._written_wrist = 0
        self._written_depth = 0
        self._writer_error: Optional[str] = None
        self._t0: float = 0.0

    def is_running(self) -> bool:
        return self._sample_thread is not None and self._sample_thread.is_alive()

    def set_recording_active(
        self,
        active: bool,
        step_index: int,
        step_name: str,
        phase_tag: str,
    ) -> None:
        with self._active_lock:
            self._recording_active = bool(active)
            self._step_index = int(step_index)
            self._step_name = str(step_name)
            self._phase_tag = str(phase_tag)

    def _wait_initial_frame(self, cam: BaseCamera, name: str, timeout_s: float = 3.0) -> CameraFrame:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            frame = cam.get_latest()
            if frame is not None:
                return frame
            time.sleep(0.02)
        raise RuntimeError(f"Camera '{name}' did not deliver a frame within {timeout_s:.1f}s.")

    def start(
        self,
        task_prompt: str,
        preset: SequencePreset,
        robot_settings: RobotSettings,
        start_active: bool = False,
    ) -> Path:
        if self.is_running():
            raise RuntimeError("Recorder is already running.")
        if not self.robot.is_connected():
            raise RuntimeError("Robot must be connected before recording.")
        if not self.world_cam.is_running() or not self.wrist_cam.is_running():
            raise RuntimeError("Both world and wrist cameras must be running before recording.")

        world_f0 = self._wait_initial_frame(self.world_cam, "world")
        wrist_f0 = self._wait_initial_frame(self.wrist_cam, "wrist")
        self._world_wh = (world_f0.image.shape[1], world_f0.image.shape[0])
        self._wrist_wh = (wrist_f0.image.shape[1], wrist_f0.image.shape[0])

        ep_idx = _next_episode_index(self.settings.raw_data_dir)
        ep_dir = _episodes_root(self.settings.raw_data_dir) / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        fps = max(1, int(self.settings.fps))
        depth_dir = ep_dir / "wrist_depth" if wrist_f0.depth is not None else None
        w_writer = None
        try:
            if depth_dir is not None:
                depth_dir.mkdir()
            w_writer, w_path, w_codec = _open_video_writer(
                ep_dir / "world_cam",
                self._world_wh[0],
                self._world_wh[1],
                fps,
                self.settings.video_codec,
            )
            r_writer, r_path, _ = _open_video_writer(
                ep_dir / "wrist_cam",
                self._wrist_wh[0],
                self._wrist_wh[1],
                fps,
                w_codec,
            )
        except Exception:
            if w_writer is not None:
                w_writer.release()
            shutil.rmtree(ep_dir, ignore_errors=True)
            raise

        self._episode_dir = ep_dir
        self._episode_index = ep_idx
        self._task_prompt = task_prompt
        self._preset_snapshot = preset.to_dict()
        self._robot_settings_snapshot = robot_settings.to_dict()
        self._world_writer = w_writer
        self._wrist_writer = r_writer
        self._world_video_path = w_path
        self._wrist_video_path = r_path
        self._wrist_depth_dir = depth_dir
        self._wrist_depth_scale_m = wrist_f0.depth_scale_m
        self._video_codec_used = w_codec

        self._timestamps = []
        self._wall_timestamps = []
        self._q_actual = []
        self._qd_actual = []
        self._tcp_actual = []
        self._gripper_raw = []
        self._gripper_norm = []
        self._step_indices = []
        self._phase_tags = []
        self._written_world = 0
        self._written_wrist = 0
        self._written_depth = 0
        self._writer_error = None

        with self._active_lock:
            self._recording_active = bool(start_active)
            self._step_index = 0 if start_active else -1
            self._step_name = ""
            self._phase_tag = ""

        self._stop_event.clear()
        self._frame_queue = queue.Queue(maxsize=128)
        self._t0 = time.perf_counter()

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"ep-writer-{ep_idx:04d}",
            daemon=True,
        )
        self._writer_thread.start()

        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            name=f"ep-sampler-{ep_idx:04d}",
            daemon=True,
        )
        self._sample_thread.start()
        return ep_dir

    def _writer_loop(self) -> None:
        assert self._frame_queue is not None
        while True:
            item = self._frame_queue.get()
            if item is None:
                self._frame_queue.task_done()
                break
            world_img, wrist_img, depth, depth_scale_m = item
            try:
                if world_img.shape[1] != self._world_wh[0] or world_img.shape[0] != self._world_wh[1]:
                    world_img = cv2.resize(world_img, self._world_wh)
                if wrist_img.shape[1] != self._wrist_wh[0] or wrist_img.shape[0] != self._wrist_wh[1]:
                    wrist_img = cv2.resize(wrist_img, self._wrist_wh)
                if self._wrist_depth_dir is not None:
                    if depth is None or depth.dtype != np.uint16 or depth.shape != wrist_img.shape[:2]:
                        raise RuntimeError("Missing or invalid depth for the wrist RGB frame.")
                    if depth_scale_m is None or depth_scale_m != self._wrist_depth_scale_m:
                        raise RuntimeError("Wrist depth scale is missing or changed during recording.")
                    depth_path = self._wrist_depth_dir / f"{self._written_wrist:06d}.png"
                    if not cv2.imwrite(str(depth_path), depth):
                        raise RuntimeError(f"Could not write wrist depth frame {depth_path}.")
                    self._written_depth += 1
                self._world_writer.write(world_img)
                self._written_world += 1
                self._wrist_writer.write(wrist_img)
                self._written_wrist += 1
            except Exception as exc:
                self._writer_error = str(exc)
            finally:
                self._frame_queue.task_done()

    def _sample_loop(self) -> None:
        fps = max(1, int(self.settings.fps))
        period = 1.0 / fps
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            sleep_for = next_tick - now
            if sleep_for > 0:
                if self._stop_event.wait(sleep_for):
                    break
            else:
                # If we fell behind by > 1 frame, re-anchor to avoid catch-up bursts
                if -sleep_for > period:
                    next_tick = time.perf_counter()

            next_tick += period

            with self._active_lock:
                active = self._recording_active
                step_idx = self._step_index
                phase_tag = self._phase_tag

            if not active:
                continue

            world_frame = self.world_cam.get_latest()
            wrist_frame = self.wrist_cam.get_latest()
            if world_frame is None or wrist_frame is None:
                continue

            state = self.robot.get_state()
            assert self._frame_queue is not None
            try:
                self._frame_queue.put_nowait(
                    (
                        world_frame.image.copy(),
                        wrist_frame.image.copy(),
                        wrist_frame.depth,
                        wrist_frame.depth_scale_m,
                    )
                )
            except queue.Full:
                self._writer_error = "Video writer queue overflowed; consider lowering FPS."
                continue

            sample_t = time.perf_counter() - self._t0
            self._timestamps.append(sample_t)
            self._wall_timestamps.append(time.time())
            self._q_actual.append(state.q)
            self._qd_actual.append(state.qd)
            self._tcp_actual.append(state.tcp)
            self._gripper_raw.append(state.gripper_raw)
            self._gripper_norm.append(state.gripper_norm)
            self._step_indices.append(step_idx)
            self._phase_tags.append(phase_tag)

    def stop(self, save: bool = True) -> Optional[EpisodeInfo]:
        self._stop_event.set()
        if self._sample_thread is not None and self._sample_thread.is_alive():
            self._sample_thread.join(timeout=3.0)
        self._sample_thread = None

        if self._frame_queue is not None:
            self._frame_queue.put(None)
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join()
        self._writer_thread = None
        self._frame_queue = None

        if self._world_writer is not None:
            self._world_writer.release()
            self._world_writer = None
        if self._wrist_writer is not None:
            self._wrist_writer.release()
            self._wrist_writer = None

        ep_dir = self._episode_dir
        self._episode_dir = None
        if ep_dir is None:
            return None

        num_frames = len(self._timestamps)
        if (not save) or num_frames == 0:
            delete_episode(ep_dir)
            return None

        depth_mismatch = self._wrist_depth_dir is not None and self._written_depth != num_frames
        if self._written_world != num_frames or self._written_wrist != num_frames or depth_mismatch:
            delete_episode(ep_dir)
            depth_count = f", wrist_depth={self._written_depth}" if self._wrist_depth_dir else ""
            raise RuntimeError(
                f"Frame count mismatch: telemetry={num_frames}, "
                f"world_video={self._written_world}, wrist_video={self._written_wrist}"
                f"{depth_count}. {self._writer_error or ''}".rstrip()
            )

        if self._writer_error:
            # Surface any serious queue overflow error only if counts diverged (already checked),
            # otherwise keep episode valid.
            pass

        t_arr = np.asarray(self._timestamps, dtype=np.float64)
        # Re-zero timestamps to first recorded frame for clean episode-relative timeline
        if len(t_arr) > 0:
            t_arr = t_arr - t_arr[0]
        duration_s = float(t_arr[-1]) if len(t_arr) > 1 else 0.0

        phase_arr = np.asarray(self._phase_tags, dtype=str)

        np.savez_compressed(
            ep_dir / "telemetry.npz",
            timestamps=t_arr,
            wall_timestamps=np.asarray(self._wall_timestamps, dtype=np.float64),
            q_actual=np.asarray(self._q_actual, dtype=np.float32),
            qd_actual=np.asarray(self._qd_actual, dtype=np.float32),
            tcp_actual=np.asarray(self._tcp_actual, dtype=np.float32),
            gripper_raw=np.asarray(self._gripper_raw, dtype=np.float32),
            gripper_norm=np.asarray(self._gripper_norm, dtype=np.float32),
            step_index=np.asarray(self._step_indices, dtype=np.int32),
            phase_tags=phase_arr,
        )

        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "episode_index": self._episode_index,
            "task_prompt": self._task_prompt,
            "fps": int(self.settings.fps),
            "num_frames": num_frames,
            "duration_s": round(duration_s, 4),
            "created_at": created_at,
            "video_codec": self._video_codec_used,
            "world_video": self._world_video_path.name if self._world_video_path else "world_cam.mp4",
            "wrist_video": self._wrist_video_path.name if self._wrist_video_path else "wrist_cam.mp4",
            "world_resolution": [self._world_wh[0], self._world_wh[1]],
            "wrist_resolution": [self._wrist_wh[0], self._wrist_wh[1]],
            "robot_settings": self._robot_settings_snapshot,
            "sequence_preset": self._preset_snapshot,
        }
        if self._wrist_depth_dir is not None:
            metadata["wrist_depth"] = {
                "directory": self._wrist_depth_dir.name,
                "filename_pattern": "{frame_index:06d}.png",
                "format": "uint16_png",
                "aligned_to": "wrist_video",
                "meters_per_unit": self._wrist_depth_scale_m,
                "invalid_value": 0,
            }
        (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return EpisodeInfo(
            episode_dir=ep_dir,
            episode_index=self._episode_index,
            num_frames=num_frames,
            duration_s=duration_s,
            fps=int(self.settings.fps),
            task_prompt=self._task_prompt,
            created_at=created_at,
        )
