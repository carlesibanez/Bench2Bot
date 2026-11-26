"""Virtual delta waypoint resolution, safety validation, and sequence execution."""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import AnchorPose, RobotSettings, SequencePreset, SequenceStep
from .robot import RobotBackend


@dataclass
class ResolvedTarget:
    tcp: Optional[List[float]] = None
    joints: Optional[List[float]] = None
    ok: bool = True
    error: str = ""


def rv_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert UR axis-angle rotation vector (rx, ry, rz) to a 3x3 rotation matrix."""
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)

    kx = rx / theta
    ky = ry / theta
    kz = rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c

    return np.array(
        [
            [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
        ],
        dtype=np.float64,
    )


def resolve_step_target(
    step: SequenceStep,
    anchors: Dict[str, AnchorPose],
    settings: RobotSettings,
) -> ResolvedTarget:
    """Resolve the target pose/joints for a step and enforce safety limits."""
    if step.step_type not in ("anchor_move", "delta_move"):
        return ResolvedTarget(ok=True)

    ref_name = (step.ref_anchor or "").strip()
    if not ref_name or ref_name not in anchors:
        return ResolvedTarget(ok=False, error=f"Missing reference anchor '{ref_name}'.")

    anchor = anchors[ref_name]
    if not anchor.is_captured or anchor.tcp is None:
        return ResolvedTarget(ok=False, error=f"Anchor '{ref_name}' is not captured yet.")

    base_tcp = list(anchor.tcp)
    if step.step_type == "anchor_move":
        target_tcp = list(base_tcp)
        target_joints = list(anchor.joints) if anchor.joints is not None else None
    else:
        delta_vec = np.array([float(step.dx), float(step.dy), float(step.dz)], dtype=np.float64)
        if step.frame == "tool":
            rot = rv_to_rotation_matrix(base_tcp[3], base_tcp[4], base_tcp[5])
            world_delta = rot @ delta_vec
        else:
            world_delta = delta_vec

        target_tcp = [
            base_tcp[0] + float(world_delta[0]),
            base_tcp[1] + float(world_delta[1]),
            base_tcp[2] + float(world_delta[2]),
            base_tcp[3],
            base_tcp[4],
            base_tcp[5],
        ]
        target_joints = None

    lock_name = (settings.lock_orientation_to_anchor or "").strip()
    if lock_name and lock_name in anchors and anchors[lock_name].is_captured:
        lock_tcp = anchors[lock_name].tcp
        if lock_tcp is not None:
            target_tcp[3] = lock_tcp[3]
            target_tcp[4] = lock_tcp[4]
            target_tcp[5] = lock_tcp[5]

    if target_tcp[2] < float(settings.z_min_safety_m):
        return ResolvedTarget(
            tcp=target_tcp,
            joints=target_joints,
            ok=False,
            error=(
                f"Target Z={target_tcp[2]:.4f}m is below z_min_safety_m="
                f"{settings.z_min_safety_m:.4f}m."
            ),
        )

    if step.motion_type == "moveJ" and step.step_type == "anchor_move" and target_joints is None:
        return ResolvedTarget(
            tcp=target_tcp,
            ok=False,
            error=f"Anchor '{ref_name}' has no captured joint angles for moveJ.",
        )

    return ResolvedTarget(tcp=target_tcp, joints=target_joints, ok=True)


def validate_sequence(
    preset: SequencePreset,
    settings: RobotSettings,
) -> List[Tuple[int, ResolvedTarget]]:
    """Resolve and validate every step in a preset before execution."""
    return [
        (idx, resolve_step_target(step, preset.anchors, settings))
        for idx, step in enumerate(preset.steps)
    ]


class SequenceRunner:
    """Executes an ordered SequencePreset on a RobotBackend with optional EpisodeRecorder."""

    def __init__(self, robot: RobotBackend, settings: RobotSettings):
        self.robot = robot
        self.settings = settings
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def abort(self) -> None:
        self._stop_event.set()
        try:
            self.robot.stop_motion()
        except Exception:
            pass

    def run(
        self,
        preset: SequencePreset,
        recorder=None,
        speed_scale: float = 1.0,
        on_step: Optional[Callable[[int, SequenceStep], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Run all steps sequentially. Returns True on completion, False if aborted."""
        self._stop_event.clear()
        if not self.robot.is_connected():
            raise RuntimeError("Robot is not connected.")

        if not preset.steps:
            raise RuntimeError("Sequence has no steps to execute.")

        resolved_list = validate_sequence(preset, self.settings)
        errors = [
            f"Step {idx + 1} ({preset.steps[idx].name}): {res.error}"
            for idx, res in resolved_list
            if not res.ok
        ]
        if errors:
            raise ValueError("Pre-flight validation failed:\n" + "\n".join(errors))

        scale = max(0.05, min(2.0, float(speed_scale)))
        if self.robot.is_freedrive_active():
            self.robot.stop_freedrive()

        for idx, (step, (_, resolved)) in enumerate(zip(preset.steps, resolved_list)):
            if self._stop_event.is_set():
                return False

            if on_step is not None:
                on_step(idx, step)
            if on_log is not None:
                on_log(
                    f"[{idx + 1}/{len(preset.steps)}] {step.name}"
                    + (" [REC]" if (recorder is not None and step.record) else "")
                )

            if recorder is not None:
                recorder.set_recording_active(
                    bool(step.record),
                    idx,
                    step.name,
                    step.phase_tag or step.name,
                )

            ok = self._execute_single_step(step, resolved, scale)
            if not ok or self._stop_event.is_set():
                return False

        if recorder is not None:
            recorder.set_recording_active(False, -1, "", "")
        return True

    def _execute_single_step(
        self,
        step: SequenceStep,
        resolved: ResolvedTarget,
        speed_scale: float,
    ) -> bool:
        if step.step_type in ("anchor_move", "delta_move"):
            use_movej = (
                step.motion_type == "moveJ"
                and step.step_type == "anchor_move"
                and resolved.joints is not None
            )
            if use_movej:
                spd = (
                    float(step.speed) if step.speed is not None else float(self.settings.movej_speed)
                ) * speed_scale
                acc = (
                    float(step.acc) if step.acc is not None else float(self.settings.movej_acc)
                ) * speed_scale
                return self.robot.move_j(resolved.joints, spd, acc, self._stop_event)

            assert resolved.tcp is not None
            spd = (
                float(step.speed) if step.speed is not None else float(self.settings.movel_speed)
            ) * speed_scale
            acc = (
                float(step.acc) if step.acc is not None else float(self.settings.movel_acc)
            ) * speed_scale
            return self.robot.move_l(resolved.tcp, spd, acc, self._stop_event)

        if step.step_type == "gripper":
            ramp = float(step.gripper_ramp_duration_s) / speed_scale
            return self.robot.move_gripper(
                pos=int(step.gripper_pos),
                speed=int(self.settings.gripper_speed),
                force=int(self.settings.gripper_force),
                ramp_duration_s=ramp,
                stop_event=self._stop_event,
            )

        if step.step_type == "dwell":
            wait_s = max(0.0, float(step.dwell_s) / speed_scale)
            return not self._stop_event.wait(wait_s)

        raise ValueError(f"Unsupported step_type: {step.step_type!r}")


def auto_return_pipette_and_home(
    robot: RobotBackend,
    preset: SequencePreset,
    settings: RobotSettings,
    stop_event: Optional[threading.Event] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> bool:
    """Unrecorded helper: lift safely, return pipette to holder, open gripper, and go Home."""
    if not robot.is_connected():
        raise RuntimeError("Robot is not connected.")

    if robot.is_freedrive_active():
        robot.stop_freedrive()

    anchors = preset.anchors
    pipette_anchor = anchors.get("pipette_grasp")
    home_anchor = anchors.get("home")

    if pipette_anchor is None or not pipette_anchor.is_captured or pipette_anchor.tcp is None:
        raise ValueError("Anchor 'pipette_grasp' must be captured before auto-returning pipette.")

    spd = float(settings.movel_speed)
    acc = float(settings.movel_acc)

    def _log(msg: str) -> None:
        if on_log is not None:
            on_log(msg)

    # 1. Lift vertically from current pose for clearance
    cur_tcp = robot.get_state().tcp.tolist()
    lift_cur = list(cur_tcp)
    lift_cur[2] = max(lift_cur[2] + 0.12, float(pipette_anchor.tcp[2]) + 0.15)
    _log("Reset: lifting vertically from current pose...")
    if not robot.move_l(lift_cur, spd, acc, stop_event):
        return False

    # 2. Transit above pipette holder
    above_holder = list(pipette_anchor.tcp)
    above_holder[2] = above_holder[2] + 0.15
    _log("Reset: moving above pipette holder...")
    if not robot.move_l(above_holder, spd, acc, stop_event):
        return False

    # 3. Lower slowly into pipette holder
    _log("Reset: placing pipette back into holder...")
    if not robot.move_l(list(pipette_anchor.tcp), min(0.08, spd), acc, stop_event):
        return False

    # 4. Open gripper to release bulb
    _log(f"Reset: opening gripper ({settings.gripper_open})...")
    if not robot.move_gripper(
        settings.gripper_open,
        settings.gripper_speed,
        settings.gripper_force,
        ramp_duration_s=0.0,
        stop_event=stop_event,
    ):
        return False

    # 5. Lift clear of pipette
    _log("Reset: lifting clear of pipette...")
    if not robot.move_l(above_holder, spd, acc, stop_event):
        return False

    # 6. Move to home anchor (with optional XY/Z jitter for VLA spatial diversity)
    if home_anchor is not None and home_anchor.is_captured and home_anchor.tcp is not None:
        home_tcp = list(home_anchor.tcp)
        j_xy = max(0.0, float(settings.home_jitter_xy_m))
        j_z = max(0.0, float(settings.home_jitter_z_m))
        if j_xy > 0.0:
            home_tcp[0] += random.uniform(-j_xy, j_xy)
            home_tcp[1] += random.uniform(-j_xy, j_xy)
        if j_z > 0.0:
            home_tcp[2] = max(
                float(settings.z_min_safety_m),
                home_tcp[2] + random.uniform(-j_z, j_z),
            )
        _log("Reset: moving to Home position...")
        if not robot.move_l(home_tcp, spd, acc, stop_event):
            return False

    _log("Reset complete.")
    return True
