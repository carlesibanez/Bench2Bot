"""UR5e (ur_rtde) + Robotiq Hand-E robot backend and real-time simulation backend."""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import RobotSettings

try:
    import rtde_control
    import rtde_receive

    HAS_UR_RTDE = True
except ImportError:
    rtde_control = None  # type: ignore
    rtde_receive = None  # type: ignore
    HAS_UR_RTDE = False

try:
    from robotiq.robotiq_gripper import RobotiqGripper

    HAS_ROBOTIQ = True
except ImportError:
    RobotiqGripper = None  # type: ignore
    HAS_ROBOTIQ = False


@dataclass
class RobotState:
    q: np.ndarray  # shape (6,) rad
    qd: np.ndarray  # shape (6,) rad/s
    tcp: np.ndarray  # shape (6,) [x, y, z, rx, ry, rz]
    gripper_raw: float  # 0..255
    gripper_norm: float  # 0.0..1.0
    timestamp: float


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _sleep_abortable(duration_s: float, stop_event: Optional[threading.Event]) -> bool:
    """Sleep up to duration_s, returning False early if stop_event is set."""
    if duration_s <= 0:
        return not (stop_event and stop_event.is_set())
    if stop_event is None:
        time.sleep(duration_s)
        return True
    return not stop_event.wait(duration_s)


class RobotBackend(ABC):
    """Unified interface for real UR5e+Robotiq hardware and simulated robot."""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def get_state(self) -> RobotState:
        ...

    @abstractmethod
    def start_freedrive(self, lock_orientation: bool = True) -> str:
        """Enable freedrive and return active mode label ('freedrive' or 'teach')."""
        ...

    @abstractmethod
    def stop_freedrive(self) -> None:
        ...

    @abstractmethod
    def is_freedrive_active(self) -> bool:
        ...

    @abstractmethod
    def move_l(
        self,
        tcp: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        ...

    @abstractmethod
    def move_j(
        self,
        joints: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        ...

    @abstractmethod
    def move_gripper(
        self,
        pos: int,
        speed: int = 125,
        force: int = 125,
        ramp_duration_s: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        ...

    @abstractmethod
    def stop_motion(self) -> None:
        ...


class UR5eRTDEBackend(RobotBackend):
    """Hardware backend using persistent ur_rtde interfaces + Robotiq URCap socket."""

    def __init__(self, settings: RobotSettings):
        self.settings = settings
        self._rtde_r = None
        self._rtde_c = None
        self._gripper = None
        self.gripper_error: Optional[str] = None
        self._connected = False
        self._freedrive_mode: Optional[str] = None

        self._gripper_lock = threading.Lock()
        self._gripper_raw = float(settings.gripper_open)
        self._gripper_stop = threading.Event()
        self._gripper_thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        if self._connected:
            return
        if not HAS_UR_RTDE:
            raise RuntimeError(
                "ur_rtde is not installed in this Python environment. "
                "Install 'ur-rtde' or enable Simulation mode."
            )

        ip = self.settings.ip
        rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        try:
            rtde_c = rtde_control.RTDEControlInterface(ip)
        except Exception as exc:
            try:
                rtde_r.disconnect()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to connect RTDEControlInterface to {ip}. "
                "Ensure the UR5e pendant is in Remote Control mode."
            ) from exc

        gripper = None
        self.gripper_error = None
        if HAS_ROBOTIQ:
            try:
                gripper = RobotiqGripper()
                gripper.connect(ip, int(self.settings.gripper_port), socket_timeout=2.0)
                if not gripper.is_active():
                    gripper.activate(auto_calibrate=False)
                self._gripper_raw = float(gripper.get_current_position())
            except Exception as exc:
                self.gripper_error = f"Gripper at {ip}:{self.settings.gripper_port}: {exc}"
                if gripper is not None:
                    try:
                        gripper.disconnect()
                    except Exception:
                        pass
                gripper = None
        else:
            self.gripper_error = "The Outreach Robotiq driver could not be imported."

        self._rtde_r = rtde_r
        self._rtde_c = rtde_c
        self._gripper = gripper
        self._connected = True

        if self._gripper is not None:
            self._gripper_stop.clear()
            self._gripper_thread = threading.Thread(
                target=self._poll_gripper_loop,
                name="robotiq-poll",
                daemon=True,
            )
            self._gripper_thread.start()

    def _poll_gripper_loop(self) -> None:
        while not self._gripper_stop.is_set() and self._gripper is not None:
            try:
                pos = float(self._gripper.get_current_position())
                with self._gripper_lock:
                    self._gripper_raw = pos
            except Exception:
                pass
            self._gripper_stop.wait(0.04)  # ~25 Hz

    def disconnect(self) -> None:
        self._gripper_stop.set()
        if self._gripper_thread is not None and self._gripper_thread.is_alive():
            self._gripper_thread.join(timeout=1.0)
        self._gripper_thread = None

        try:
            self.stop_freedrive()
        except Exception:
            pass

        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
            self._gripper = None

        if self._rtde_c is not None:
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass
            self._rtde_c = None

        if self._rtde_r is not None:
            try:
                self._rtde_r.disconnect()
            except Exception:
                pass
            self._rtde_r = None

        self._connected = False
        self._freedrive_mode = None

    def is_connected(self) -> bool:
        return (
            self._connected
            and self._rtde_r is not None
            and self._rtde_c is not None
            and bool(self._rtde_c.isConnected())
        )

    def get_state(self) -> RobotState:
        if not self._connected or self._rtde_r is None:
            raise RuntimeError("Robot is not connected.")
        q = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        qd = np.asarray(self._rtde_r.getActualQd(), dtype=np.float64)
        tcp = np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float64)
        with self._gripper_lock:
            g_raw = self._gripper_raw
        return RobotState(
            q=q,
            qd=qd,
            tcp=tcp,
            gripper_raw=g_raw,
            gripper_norm=_clamp(g_raw / 255.0, 0.0, 1.0),
            timestamp=time.perf_counter(),
        )

    def start_freedrive(self, lock_orientation: bool = True) -> str:
        if not self.is_connected() or self._rtde_c is None:
            raise RuntimeError("Robot is not connected.")
        if self._freedrive_mode is not None:
            self.stop_freedrive()

        if not self._rtde_c.isProgramRunning():
            raise RuntimeError(
                "The RTDE control program is stopped. Check that the pendant is in Remote Control "
                "and the robot is powered on with brakes released, then reconnect."
            )
        if lock_orientation:
            mode = "freedrive"
            ok = self._rtde_c.freedriveMode([1, 1, 1, 0, 0, 0], [0.0] * 6)
        else:
            mode = "teach"
            ok = self._rtde_c.teachMode()
        if not ok:
            raise RuntimeError(f"UR5e rejected {mode}. Check the pendant for a safety or mode message.")
        self._freedrive_mode = mode
        return mode

    def stop_freedrive(self) -> None:
        if self._rtde_c is None:
            self._freedrive_mode = None
            return
        if self._freedrive_mode is None:
            return
        end_mode = (
            self._rtde_c.endFreedriveMode
            if self._freedrive_mode == "freedrive"
            else self._rtde_c.endTeachMode
        )
        if not end_mode():
            raise RuntimeError(f"UR5e did not confirm exit from {self._freedrive_mode} mode.")
        self._freedrive_mode = None

    def is_freedrive_active(self) -> bool:
        return self._freedrive_mode is not None

    def move_l(
        self,
        tcp: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if not self.is_connected() or self._rtde_c is None:
            raise RuntimeError("Robot is not connected.")
        if self.is_freedrive_active():
            self.stop_freedrive()

        target = [float(v) for v in tcp]
        ok = self._rtde_c.moveL(target, float(speed), float(acc), True)
        if not ok:
            return False

        target_arr = np.asarray(target, dtype=np.float64)
        while True:
            if stop_event is not None and stop_event.is_set():
                try:
                    self._rtde_c.stopL(1.0)
                except Exception:
                    pass
                return False

            cur_tcp = np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float64)
            pos_err = float(np.linalg.norm(cur_tcp[:3] - target_arr[:3]))
            steady = bool(self._rtde_c.isSteady())
            if pos_err < 0.0015 and steady:
                return True
            if steady and pos_err < 0.003:
                return True
            time.sleep(0.02)

    def move_j(
        self,
        joints: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if not self.is_connected() or self._rtde_c is None:
            raise RuntimeError("Robot is not connected.")
        if self.is_freedrive_active():
            self.stop_freedrive()

        target = [float(v) for v in joints]
        ok = self._rtde_c.moveJ(target, float(speed), float(acc), True)
        if not ok:
            return False

        target_arr = np.asarray(target, dtype=np.float64)
        while True:
            if stop_event is not None and stop_event.is_set():
                try:
                    self._rtde_c.stopJ(2.0)
                except Exception:
                    pass
                return False

            cur_q = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
            joint_err = float(np.max(np.abs(cur_q - target_arr)))
            steady = bool(self._rtde_c.isSteady())
            if joint_err < 0.005 and steady:
                return True
            time.sleep(0.02)

    def move_gripper(
        self,
        pos: int,
        speed: int = 125,
        force: int = 125,
        ramp_duration_s: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if self._gripper is None:
            raise RuntimeError(self.gripper_error or "Gripper is not connected.")
        target_pos = int(_clamp(pos, 0, 255))
        spd = int(_clamp(speed, 0, 255))
        frc = int(_clamp(force, 0, 255))

        with self._gripper_lock:
            start_pos = self._gripper_raw

        if ramp_duration_s and ramp_duration_s > 0.05:
            steps = max(2, int(round(ramp_duration_s * 18)))
            dt = ramp_duration_s / steps
            for i in range(1, steps + 1):
                if stop_event is not None and stop_event.is_set():
                    return False
                interp = int(round(start_pos + (target_pos - start_pos) * (i / steps)))
                accepted, _ = self._gripper.move(interp, spd, frc)
                if not accepted:
                    raise RuntimeError(f"Gripper did not acknowledge position {interp}.")
                if not _sleep_abortable(dt, stop_event):
                    return False
            return True

        accepted, _ = self._gripper.move(target_pos, spd, frc)
        if not accepted:
            raise RuntimeError(f"Gripper did not acknowledge position {target_pos}.")
        return _sleep_abortable(0.25, stop_event)

    def stop_motion(self) -> None:
        if self._rtde_c is None:
            return
        try:
            self._rtde_c.stopL(1.5)
        except Exception:
            pass
        try:
            self.stop_freedrive()
        except Exception:
            pass


class SimulatedRobotBackend(RobotBackend):
    """Real-time simulated UR5e + Robotiq backend for offline testing and GUI demos."""

    DEFAULT_HOME_Q = np.array(
        [0.89725, -1.71096, 2.04204, -0.37420, 0.96959, 3.17661],
        dtype=np.float64,
    )
    DEFAULT_HOME_TCP = np.array(
        [0.30, -0.25, 0.35, 3.14159, 0.0, 0.0],
        dtype=np.float64,
    )

    def __init__(self, settings: RobotSettings, speed_multiplier: float = 1.0):
        self.settings = settings
        self.speed_multiplier = max(0.1, float(speed_multiplier))
        self._lock = threading.Lock()
        self._connected = False
        self._freedrive_mode: Optional[str] = None
        self._q = self.DEFAULT_HOME_Q.copy()
        self._qd = np.zeros(6, dtype=np.float64)
        self._tcp = self.DEFAULT_HOME_TCP.copy()
        self._gripper_raw = float(settings.gripper_open)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._freedrive_mode = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                q=self._q.copy(),
                qd=self._qd.copy(),
                tcp=self._tcp.copy(),
                gripper_raw=self._gripper_raw,
                gripper_norm=_clamp(self._gripper_raw / 255.0, 0.0, 1.0),
                timestamp=time.perf_counter(),
            )

    def start_freedrive(self, lock_orientation: bool = True) -> str:
        if not self._connected:
            raise RuntimeError("Simulated robot is not connected.")
        self._freedrive_mode = "freedrive" if lock_orientation else "teach"
        return self._freedrive_mode

    def stop_freedrive(self) -> None:
        self._freedrive_mode = None

    def is_freedrive_active(self) -> bool:
        return self._freedrive_mode is not None

    def sim_nudge(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Shift simulated TCP and slightly adjust joints to mimic manual freedrive."""
        with self._lock:
            self._tcp[0] += dx
            self._tcp[1] += dy
            self._tcp[2] = max(0.01, self._tcp[2] + dz)
            self._q[0] += dx * 1.5
            self._q[1] -= dz * 1.8
            self._q[2] += dy * 1.4

    def move_l(
        self,
        tcp: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if not self._connected:
            raise RuntimeError("Simulated robot is not connected.")
        self.stop_freedrive()

        with self._lock:
            start_tcp = self._tcp.copy()
            start_q = self._q.copy()

        target_tcp = np.asarray(tcp, dtype=np.float64)
        dist = float(np.linalg.norm(target_tcp[:3] - start_tcp[:3]))
        duration = max(0.12, dist / max(0.02, float(speed))) / self.speed_multiplier
        steps = max(4, int(math.ceil(duration / 0.02)))
        dt = duration / steps

        # Deterministic synthetic joint delta coupled to TCP delta for realistic telemetry
        delta_xyz = target_tcp[:3] - start_tcp[:3]
        target_q = start_q + np.array(
            [delta_xyz[0] * 1.5, -delta_xyz[2] * 1.8, delta_xyz[1] * 1.4, 0.0, 0.0, 0.0]
        )

        for i in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                with self._lock:
                    self._qd.fill(0.0)
                return False
            alpha = i / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            with self._lock:
                self._tcp = start_tcp + (target_tcp - start_tcp) * smooth
                prev_q = self._q.copy()
                self._q = start_q + (target_q - start_q) * smooth
                self._qd = (self._q - prev_q) / max(1e-4, dt)
            if not _sleep_abortable(dt, stop_event):
                with self._lock:
                    self._qd.fill(0.0)
                return False

        with self._lock:
            self._qd.fill(0.0)
        return True

    def move_j(
        self,
        joints: Sequence[float],
        speed: float,
        acc: float,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if not self._connected:
            raise RuntimeError("Simulated robot is not connected.")
        self.stop_freedrive()

        with self._lock:
            start_q = self._q.copy()

        target_q = np.asarray(joints, dtype=np.float64)
        max_delta = float(np.max(np.abs(target_q - start_q)))
        duration = max(0.15, max_delta / max(0.05, float(speed))) / self.speed_multiplier
        steps = max(4, int(math.ceil(duration / 0.02)))
        dt = duration / steps

        for i in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                with self._lock:
                    self._qd.fill(0.0)
                return False
            alpha = i / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            with self._lock:
                prev_q = self._q.copy()
                self._q = start_q + (target_q - start_q) * smooth
                self._qd = (self._q - prev_q) / max(1e-4, dt)
            if not _sleep_abortable(dt, stop_event):
                with self._lock:
                    self._qd.fill(0.0)
                return False

        with self._lock:
            self._qd.fill(0.0)
        return True

    def move_gripper(
        self,
        pos: int,
        speed: int = 125,
        force: int = 125,
        ramp_duration_s: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        if not self._connected:
            raise RuntimeError("Simulated robot is not connected.")
        target_pos = float(_clamp(pos, 0, 255))
        with self._lock:
            start_pos = self._gripper_raw

        duration = (
            max(0.05, float(ramp_duration_s)) if ramp_duration_s > 0 else 0.12
        ) / self.speed_multiplier
        steps = max(2, int(math.ceil(duration / 0.03)))
        dt = duration / steps

        for i in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                return False
            with self._lock:
                self._gripper_raw = start_pos + (target_pos - start_pos) * (i / steps)
            if not _sleep_abortable(dt, stop_event):
                return False
        return True

    def stop_motion(self) -> None:
        with self._lock:
            self._qd.fill(0.0)
            self._freedrive_mode = None
