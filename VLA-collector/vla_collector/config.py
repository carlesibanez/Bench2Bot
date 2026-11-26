"""Configuration dataclasses, JSON persistence, and default titration preset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RobotSettings:
    ip: str = "192.168.0.2"
    gripper_port: int = 63352
    simulation_mode: bool = True
    movel_speed: float = 0.15
    movel_acc: float = 0.20
    movej_speed: float = 0.50
    movej_acc: float = 0.50
    z_min_safety_m: float = 0.02
    home_jitter_xy_m: float = 0.0
    home_jitter_z_m: float = 0.0
    gripper_open: int = 180
    gripper_grasp: int = 200
    gripper_squeeze: int = 240
    gripper_speed: int = 125
    gripper_force: int = 125
    lock_orientation_to_anchor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotSettings":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)


@dataclass
class CameraConfig:
    name: str = "world"
    backend: str = "opencv"  # "opencv" | "realsense" | "mock"
    device_id: str = "0"
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_h: bool = False
    flip_v: bool = False
    rotate_180: bool = False
    rs_auto_exposure: bool = True
    rs_exposure: int = 8500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)


@dataclass
class RecordingSettings:
    fps: int = 20
    raw_data_dir: str = "./vla_data/raw"
    task_prompt: str = (
        "Pick up the pipette from the holder, aspirate liquid from the reagent vial, "
        "and dispense it into the target vial."
    )
    video_codec: str = "mp4v"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingSettings":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)


@dataclass
class AppSettings:
    robot: RobotSettings = field(default_factory=RobotSettings)
    world_camera: CameraConfig = field(
        default_factory=lambda: CameraConfig(name="world", backend="opencv", device_id="0")
    )
    wrist_camera: CameraConfig = field(
        default_factory=lambda: CameraConfig(name="wrist", backend="realsense", device_id="")
    )
    recording: RecordingSettings = field(default_factory=RecordingSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot": self.robot.to_dict(),
            "world_camera": self.world_camera.to_dict(),
            "wrist_camera": self.wrist_camera.to_dict(),
            "recording": self.recording.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        data = data or {}
        return cls(
            robot=RobotSettings.from_dict(data.get("robot", {})),
            world_camera=CameraConfig.from_dict(
                {"name": "world", **data.get("world_camera", {})}
            ),
            wrist_camera=CameraConfig.from_dict(
                {"name": "wrist", **data.get("wrist_camera", {})}
            ),
            recording=RecordingSettings.from_dict(data.get("recording", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "AppSettings":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class AnchorPose:
    name: str
    tcp: Optional[List[float]] = None
    joints: Optional[List[float]] = None
    captured_at: Optional[str] = None

    @property
    def is_captured(self) -> bool:
        return self.tcp is not None and len(self.tcp) == 6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnchorPose":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)


@dataclass
class SequenceStep:
    name: str
    step_type: str = "delta_move"  # "anchor_move" | "delta_move" | "gripper" | "dwell"
    ref_anchor: str = ""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    frame: str = "base"  # "base" | "tool"
    motion_type: str = "moveL"  # "moveL" | "moveJ"
    speed: Optional[float] = None
    acc: Optional[float] = None
    gripper_pos: int = 180
    gripper_ramp_duration_s: float = 0.0
    dwell_s: float = 0.1
    record: bool = True
    phase_tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequenceStep":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)


@dataclass
class SequencePreset:
    name: str = "Default Titration Workflow"
    anchors: Dict[str, AnchorPose] = field(default_factory=dict)
    steps: List[SequenceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "anchors": {k: v.to_dict() for k, v in self.anchors.items()},
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequencePreset":
        data = data or {}
        raw_anchors = data.get("anchors", {})
        anchors: Dict[str, AnchorPose] = {}
        for key, val in raw_anchors.items():
            anchor = AnchorPose.from_dict({"name": key, **val})
            anchors[anchor.name] = anchor
        steps = [SequenceStep.from_dict(item) for item in data.get("steps", [])]
        return cls(name=data.get("name", "Custom Sequence"), anchors=anchors, steps=steps)

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "SequencePreset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def default_app_settings(simulation_mode: bool = True) -> AppSettings:
    settings = AppSettings()
    settings.robot.simulation_mode = simulation_mode
    if simulation_mode:
        settings.world_camera.backend = "mock"
        settings.wrist_camera.backend = "mock"
    return settings


def default_titration_preset() -> SequencePreset:
    anchors = {
        "home": AnchorPose(name="home"),
        "pipette_grasp": AnchorPose(name="pipette_grasp"),
        "vial1_in": AnchorPose(name="vial1_in"),
        "vial2_in": AnchorPose(name="vial2_in"),
    }
    steps = [
        SequenceStep(
            name="Hover over pipette",
            step_type="delta_move",
            ref_anchor="pipette_grasp",
            dz=0.08,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="grasp_pipette",
        ),
        SequenceStep(
            name="Open gripper for approach",
            step_type="gripper",
            gripper_pos=180,
            gripper_ramp_duration_s=0.0,
            record=True,
            phase_tag="grasp_pipette",
        ),
        SequenceStep(
            name="Descend to pipette bulb",
            step_type="anchor_move",
            ref_anchor="pipette_grasp",
            motion_type="moveL",
            record=True,
            phase_tag="grasp_pipette",
        ),
        SequenceStep(
            name="Grasp pipette bulb",
            step_type="gripper",
            gripper_pos=200,
            gripper_ramp_duration_s=0.2,
            record=True,
            phase_tag="grasp_pipette",
        ),
        SequenceStep(
            name="Lift pipette from holder",
            step_type="delta_move",
            ref_anchor="pipette_grasp",
            dz=0.15,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="grasp_pipette",
        ),
        SequenceStep(
            name="Hover over reagent vial",
            step_type="delta_move",
            ref_anchor="vial1_in",
            dz=0.10,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="aspirate",
        ),
        SequenceStep(
            name="Squeeze bulb above reagent vial",
            step_type="gripper",
            gripper_pos=240,
            gripper_ramp_duration_s=0.4,
            record=True,
            phase_tag="aspirate",
        ),
        SequenceStep(
            name="Descend into reagent vial",
            step_type="anchor_move",
            ref_anchor="vial1_in",
            motion_type="moveL",
            record=True,
            phase_tag="aspirate",
        ),
        SequenceStep(
            name="Unsqueeze bulb to aspirate",
            step_type="gripper",
            gripper_pos=200,
            gripper_ramp_duration_s=0.6,
            record=True,
            phase_tag="aspirate",
        ),
        SequenceStep(
            name="Lift from reagent vial",
            step_type="delta_move",
            ref_anchor="vial1_in",
            dz=0.10,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="aspirate",
        ),
        SequenceStep(
            name="Hover over stirrer vial",
            step_type="delta_move",
            ref_anchor="vial2_in",
            dz=0.10,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="dispense",
        ),
        SequenceStep(
            name="Descend into stirrer vial",
            step_type="anchor_move",
            ref_anchor="vial2_in",
            motion_type="moveL",
            record=True,
            phase_tag="dispense",
        ),
        SequenceStep(
            name="Squeeze bulb to dispense",
            step_type="gripper",
            gripper_pos=240,
            gripper_ramp_duration_s=0.6,
            record=True,
            phase_tag="dispense",
        ),
        SequenceStep(
            name="Lift from stirrer vial",
            step_type="delta_move",
            ref_anchor="vial2_in",
            dz=0.10,
            frame="base",
            motion_type="moveL",
            record=True,
            phase_tag="dispense",
        ),
    ]
    return SequencePreset(
        name="Default Titration Workflow",
        anchors=anchors,
        steps=steps,
    )
