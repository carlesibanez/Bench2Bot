"""Dual-camera abstraction supporting OpenCV webcams, Intel RealSense D405, and Mock cameras."""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from .config import CameraConfig

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    rs = None  # type: ignore
    HAS_REALSENSE = False


@dataclass
class CameraFrame:
    image: np.ndarray  # BGR uint8 HxWx3
    timestamp: float
    frame_id: int
    depth: Optional[np.ndarray] = None  # RGB-aligned uint16 HxW
    depth_scale_m: Optional[float] = None


def _apply_transforms(image: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    out = image
    if cfg.rotate_180:
        out = np.rot90(out, 2)
    if cfg.flip_h:
        out = np.fliplr(out)
    if cfg.flip_v:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


class BaseCamera:
    """Background-threaded camera reader exposing non-blocking latest frame snapshots."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[CameraFrame] = None
        self._frame_id = 0
        self._timestamps: Deque[float] = deque(maxlen=30)
        self._running = False
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def measured_fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._last_error = None
        self._open_device()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running = False
        try:
            self._close_device()
        except Exception:
            pass

    def get_latest(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._latest

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue
                depth = None
                depth_scale_m = None
                if isinstance(frame, tuple):
                    frame, raw_depth, depth_scale_m = frame
                    depth = _apply_transforms(raw_depth, self.config)
                transformed = _apply_transforms(frame, self.config)
                now = time.perf_counter()
                with self._lock:
                    self._frame_id += 1
                    self._latest = CameraFrame(
                        image=transformed,
                        timestamp=now,
                        frame_id=self._frame_id,
                        depth=depth,
                        depth_scale_m=depth_scale_m,
                    )
                    self._timestamps.append(now)
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            self._running = False

    def _open_device(self) -> None:
        raise NotImplementedError

    def _read_frame(self) -> Optional[np.ndarray | Tuple[np.ndarray, np.ndarray, float]]:
        raise NotImplementedError

    def _close_device(self) -> None:
        raise NotImplementedError


class OpenCVCamera(BaseCamera):
    """USB webcam or UVC camera via OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap = None

    def _open_device(self) -> None:
        if not HAS_CV2:
            raise RuntimeError("opencv-python is not installed.")
        try:
            index = int(self.config.device_id) if str(self.config.device_id).strip() else 0
        except ValueError:
            index = self.config.device_id

        if sys.platform.startswith("win") and isinstance(index, int):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open OpenCV camera '{self.config.device_id}'.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
        cap.set(cv2.CAP_PROP_FPS, int(self.config.fps))
        self._cap = cap

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RealSenseCamera(BaseCamera):
    """Intel RealSense (e.g., D405 wrist camera) via pyrealsense2."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._pipeline = None
        self._align = None

    def _open_device(self) -> None:
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 is not installed.")

        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        serial = str(self.config.device_id).strip()
        if serial:
            rs_cfg.enable_device(serial)

        rs_cfg.enable_stream(
            rs.stream.color,
            int(self.config.width),
            int(self.config.height),
            rs.format.bgr8,
            int(self.config.fps),
        )
        rs_cfg.enable_stream(
            rs.stream.depth,
            int(self.config.width),
            int(self.config.height),
            rs.format.z16,
            int(self.config.fps),
        )
        align = rs.align(rs.stream.color)
        profile = pipeline.start(rs_cfg)
        self._apply_exposure_options(profile)
        self._align = align
        self._pipeline = pipeline

    def _apply_exposure_options(self, profile) -> None:
        try:
            device = profile.get_device()
            for sensor in device.query_sensors():
                if not sensor.supports(rs.option.enable_auto_exposure):
                    continue
                # On D405, RGB comes from the Stereo Module; on D435/D415 it's RGB Camera.
                sensor.set_option(
                    rs.option.enable_auto_exposure,
                    1.0 if self.config.rs_auto_exposure else 0.0,
                )
                if (not self.config.rs_auto_exposure) and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(self.config.rs_exposure))
        except Exception:
            # Non-fatal if firmware locks exposure option during start
            pass

    def _read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if self._pipeline is None or self._align is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:
            return None
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        return (
            np.asanyarray(color_frame.get_data()).copy(),
            np.asanyarray(depth_frame.get_data()).copy(),
            float(depth_frame.get_units()),
        )

    def _close_device(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None


class MockCamera(BaseCamera):
    """Synthetic camera producing animated BGR frames for offline/simulation testing."""

    def _open_device(self) -> None:
        self._start_time = time.perf_counter()
        self._counter = 0

    def _read_frame(self) -> Optional[np.ndarray]:
        period = 1.0 / max(1, int(self.config.fps))
        time.sleep(period)
        w = max(64, int(self.config.width))
        h = max(48, int(self.config.height))

        img = np.zeros((h, w, 3), dtype=np.uint8)
        is_wrist = "wrist" in self.config.name.lower()
        img[:, :] = (36, 42, 48) if is_wrist else (45, 38, 32)

        t = time.perf_counter() - self._start_time
        self._counter += 1
        cx = int(w * (0.5 + 0.32 * math.sin(t * 2.0)))
        cy = int(h * (0.5 + 0.25 * math.cos(t * 1.6)))

        if HAS_CV2:
            accent = (80, 210, 140) if is_wrist else (220, 165, 75)
            cv2.grid = None
            cv2.circle(img, (cx, cy), max(12, min(w, h) // 14), accent, -1)
            cv2.circle(img, (cx, cy), max(16, min(w, h) // 10), (235, 235, 235), 2)
            cv2.putText(
                img,
                f"MOCK [{self.config.name}] {w}x{h}",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
            )
            cv2.putText(
                img,
                f"frame={self._counter} t={t:.2f}s",
                (14, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                1,
            )
        else:
            r = max(6, min(w, h) // 16)
            y0, y1 = max(0, cy - r), min(h, cy + r)
            x0, x1 = max(0, cx - r), min(w, cx + r)
            img[y0:y1, x0:x1] = (80, 210, 140) if is_wrist else (220, 165, 75)

        return img

    def _close_device(self) -> None:
        return None


def create_camera(config: CameraConfig) -> BaseCamera:
    backend = (config.backend or "mock").strip().lower()
    if backend == "opencv":
        return OpenCVCamera(config)
    if backend == "realsense":
        return RealSenseCamera(config)
    if backend == "mock":
        return MockCamera(config)
    raise ValueError(f"Unknown camera backend: {config.backend!r}")


def list_opencv_cameras(max_index: Optional[int] = None) -> List[str]:
    """Return indices that deliver a frame, using device nodes for Linux discovery."""
    if not HAS_CV2:
        return []
    if max_index is not None:
        indices = range(max_index)
    elif sys.platform.startswith("linux"):
        indices = sorted(
            int(path.name[5:])
            for path in Path("/dev").glob("video*")
            if path.name[5:].isdecimal()
        )
    else:
        indices = range(6)
    found: List[str] = []
    for idx in indices:
        cap = (
            cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if sys.platform.startswith("win")
            else cv2.VideoCapture(idx)
        )
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(str(idx))
        finally:
            cap.release()
    return found


def list_realsense_devices() -> List[Tuple[str, str]]:
    """Return list of (serial_number, product_name) for connected RealSense devices."""
    if not HAS_REALSENSE:
        return []
    devices: List[Tuple[str, str]] = []
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            devices.append((serial, name))
    except Exception:
        pass
    return devices
