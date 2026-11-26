"""4-tab Tkinter GUI for UR5e + Robotiq + Dual-Camera VLA Data Collection."""

from __future__ import annotations

import copy
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, Optional, Tuple

from .cameras import (
    BaseCamera,
    create_camera,
    list_opencv_cameras,
    list_realsense_devices,
)
from .config import (
    AnchorPose,
    AppSettings,
    CameraConfig,
    SequencePreset,
    SequenceStep,
    default_app_settings,
    default_titration_preset,
)
from .lerobot_converter import convert_raw_to_lerobot
from .recorder import EpisodeInfo, EpisodeRecorder, delete_episode, list_raw_episodes
from .robot import RobotBackend, SimulatedRobotBackend, UR5eRTDEBackend
from .sequence_runner import (
    SequenceRunner,
    auto_return_pipette_and_home,
    validate_sequence,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


DEFAULT_SETTINGS_PATH = Path("configs/app_settings.json")
DEFAULT_PRESET_PATH = Path("configs/default_titration_sequence.json")


class VLACollectorApp(tk.Tk):
    """Desktop GUI for teaching waypoints, dual-camera monitoring, raw recording, and LeRobot export."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        preset: Optional[SequencePreset] = None,
    ):
        super().__init__()
        self.title("UR5e Titration VLA Data Collector")
        self.geometry("1380x860")
        self.minsize(1160, 740)

        if settings is not None:
            self.settings = settings
        elif DEFAULT_SETTINGS_PATH.exists():
            self.settings = AppSettings.load_json(DEFAULT_SETTINGS_PATH)
        else:
            self.settings = default_app_settings(simulation_mode=True)

        if preset is not None:
            self.preset = preset
        elif DEFAULT_PRESET_PATH.exists():
            self.preset = SequencePreset.load_json(DEFAULT_PRESET_PATH)
        else:
            self.preset = default_titration_preset()

        self.robot: Optional[RobotBackend] = None
        self.world_cam: Optional[BaseCamera] = None
        self.wrist_cam: Optional[BaseCamera] = None
        self.runner: Optional[SequenceRunner] = None
        self.recorder: Optional[EpisodeRecorder] = None

        self._busy = False
        self._is_recording = False
        self._active_step_idx: Optional[int] = None
        self._stop_event = threading.Event()
        self._last_episode: Optional[EpisodeInfo] = None

        self._anchor_labels: Dict[str, ttk.Label] = {}
        self._anchor_frame_inner: Optional[ttk.Frame] = None
        self._cam_widgets: Dict[str, Dict] = {}

        self._build_styles()
        self._build_ui()
        self._refresh_anchor_capture_panel()
        self._refresh_sequence_tree()
        self._sync_settings_to_ui()
        self._refresh_episode_counter()

        self.bind_all("<Escape>", self._on_escape_key)
        self.bind_all("<space>", self._on_space_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(100, self._ui_poll_loop)

    # ------------------------------------------------------------------
    # Styles & Layout
    # ------------------------------------------------------------------
    def _build_styles(self) -> None:
        style = ttk.Style(self)
        try:
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass
        style.configure("Abort.TButton", foreground="white", background="#c62828", font=("Segoe UI", 10, "bold"))
        style.map("Abort.TButton", background=[("active", "#b71c1c")])
        style.configure("Record.TButton", foreground="white", background="#1b5e20", font=("Segoe UI", 10, "bold"))
        style.map("Record.TButton", background=[("active", "#144818")])
        style.configure("ActiveFD.TButton", foreground="white", background="#ef6c00", font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_collect = ttk.Frame(self.notebook, padding=8)
        self.tab_sequence = ttk.Frame(self.notebook, padding=8)
        self.tab_cameras = ttk.Frame(self.notebook, padding=8)
        self.tab_settings = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab_collect, text="1. Collect & Monitor")
        self.notebook.add(self.tab_sequence, text="2. Sequence Builder")
        self.notebook.add(self.tab_cameras, text="3. Cameras Setup")
        self.notebook.add(self.tab_settings, text="4. Settings & Export")

        self._build_collect_tab()
        self._build_sequence_tab()
        self._build_cameras_tab()
        self._build_settings_tab()

    # ------------------------------------------------------------------
    # Tab 1: Collect & Monitor
    # ------------------------------------------------------------------
    def _build_collect_tab(self) -> None:
        paned = ttk.PanedWindow(self.tab_collect, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        right = ttk.Frame(paned)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        # Connection card
        conn_lf = ttk.LabelFrame(left, text="Robot Connection", padding=8)
        conn_lf.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(conn_lf)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="UR5e IP:").pack(side=tk.LEFT)
        self.ip_var = tk.StringVar(value=self.settings.robot.ip)
        ttk.Entry(row1, textvariable=self.ip_var, width=15).pack(side=tk.LEFT, padx=(4, 10))
        self.sim_var = tk.BooleanVar(value=self.settings.robot.simulation_mode)
        ttk.Checkbutton(
            row1,
            text="Simulation Mode",
            variable=self.sim_var,
            command=self._on_sim_toggle,
        ).pack(side=tk.LEFT)

        row2 = ttk.Frame(conn_lf)
        row2.pack(fill=tk.X, pady=(6, 0))
        self.connect_btn = ttk.Button(row2, text="Connect Robot", command=self._toggle_robot_connect)
        self.connect_btn.pack(side=tk.LEFT)
        self.conn_status_lbl = ttk.Label(row2, text="Disconnected", foreground="#8b0000")
        self.conn_status_lbl.pack(side=tk.LEFT, padx=(10, 10))
        self.ep_count_lbl = ttk.Label(row2, text="Episodes: 0", font=("Segoe UI", 9, "bold"))
        self.ep_count_lbl.pack(side=tk.RIGHT)

        # Freedrive & Quick Gripper
        fd_lf = ttk.LabelFrame(left, text="Freedrive & Gripper (Remote Mode)", padding=8)
        fd_lf.pack(fill=tk.X, pady=(0, 8))

        fd_row = ttk.Frame(fd_lf)
        fd_row.pack(fill=tk.X)
        self.fd_locked_btn = ttk.Button(
            fd_row,
            text="Freedrive: Locked XYZ (Space)",
            command=lambda: self._toggle_freedrive(lock_orientation=True),
            state=tk.DISABLED,
        )
        self.fd_locked_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.fd_6dof_btn = ttk.Button(
            fd_row,
            text="Freedrive: 6-DoF",
            command=lambda: self._toggle_freedrive(lock_orientation=False),
            state=tk.DISABLED,
        )
        self.fd_6dof_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        grip_row = ttk.Frame(fd_lf)
        grip_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(grip_row, text="Gripper:").pack(side=tk.LEFT, padx=(0, 4))
        self.grip_open_btn = ttk.Button(
            grip_row,
            text="Open",
            command=lambda: self._quick_gripper(self.settings.robot.gripper_open),
            state=tk.DISABLED,
        )
        self.grip_open_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.grip_hold_btn = ttk.Button(
            grip_row,
            text="Hold Bulb",
            command=lambda: self._quick_gripper(self.settings.robot.gripper_grasp),
            state=tk.DISABLED,
        )
        self.grip_hold_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.grip_sq_btn = ttk.Button(
            grip_row,
            text="Squeeze",
            command=lambda: self._quick_gripper(self.settings.robot.gripper_squeeze),
            state=tk.DISABLED,
        )
        self.grip_sq_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Simulation nudge controls
        self.sim_nudge_lf = ttk.LabelFrame(left, text="Simulation Helpers (Offline Testing)", padding=6)
        self.sim_nudge_lf.pack(fill=tk.X, pady=(0, 8))
        nudge_row = ttk.Frame(self.sim_nudge_lf)
        nudge_row.pack(fill=tk.X)
        for label, delta in [
            ("-X", (-0.03, 0, 0)),
            ("+X", (0.03, 0, 0)),
            ("-Y", (0, -0.03, 0)),
            ("+Y", (0, 0.03, 0)),
            ("-Z", (0, 0, -0.03)),
            ("+Z", (0, 0, 0.03)),
        ]:
            ttk.Button(
                nudge_row,
                text=label,
                width=4,
                command=lambda d=delta: self._sim_nudge(*d),
            ).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            nudge_row,
            text="Seed Sim Anchors",
            command=self._seed_sim_anchors,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        # Anchor Capture panel
        anch_lf = ttk.LabelFrame(left, text="Waypoint Anchors (Capture Current Pose)", padding=8)
        anch_lf.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self._anchor_frame_inner = ttk.Frame(anch_lf)
        self._anchor_frame_inner.pack(fill=tk.X)

        # Execution & Recording panel
        run_lf = ttk.LabelFrame(left, text="Episode Execution & Recording", padding=8)
        run_lf.pack(fill=tk.X, pady=(0, 8))

        prompt_row = ttk.Frame(run_lf)
        prompt_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(prompt_row, text="Prompt:").pack(side=tk.LEFT)
        self.collect_prompt_var = tk.StringVar(value=self.settings.recording.task_prompt)
        ttk.Entry(prompt_row, textvariable=self.collect_prompt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        btn_row1 = ttk.Frame(run_lf)
        btn_row1.pack(fill=tk.X, pady=(0, 4))
        self.dry_run_btn = ttk.Button(
            btn_row1,
            text="Dry Run (0.5x, No Rec)",
            command=lambda: self._start_sequence(record=False, speed_scale=0.5),
            state=tk.DISABLED,
        )
        self.dry_run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.reset_btn = ttk.Button(
            btn_row1,
            text="Auto-Return Pipette & Home",
            command=self._start_auto_return,
            state=tk.DISABLED,
        )
        self.reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_row2 = ttk.Frame(run_lf)
        btn_row2.pack(fill=tk.X, pady=(0, 6))
        self.rec_run_btn = ttk.Button(
            btn_row2,
            text="RUN & RECORD EPISODE",
            style="Record.TButton",
            command=lambda: self._start_sequence(record=True, speed_scale=1.0),
            state=tk.DISABLED,
        )
        self.rec_run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.discard_btn = ttk.Button(
            btn_row2,
            text="Discard Last",
            command=self._discard_last_episode,
            state=tk.DISABLED,
        )
        self.discard_btn.pack(side=tk.LEFT)

        self.abort_btn = ttk.Button(
            run_lf,
            text="STOP / ABORT MOTION (Esc)",
            style="Abort.TButton",
            command=self._abort_action,
        )
        self.abort_btn.pack(fill=tk.X)

        # Action Log
        log_lf = ttk.LabelFrame(left, text="Activity Log", padding=6)
        log_lf.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_lf, height=8, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Right side: Dual Camera Feeds + Telemetry
        cam_bar = ttk.Frame(right)
        cam_bar.pack(fill=tk.X, pady=(0, 6))
        self.start_cams_btn = ttk.Button(
            cam_bar, text="Start Both Cameras", command=self._start_all_cameras
        )
        self.start_cams_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_cams_btn = ttk.Button(
            cam_bar, text="Stop Cameras", command=self._stop_all_cameras
        )
        self.stop_cams_btn.pack(side=tk.LEFT)
        self.rec_badge_lbl = ttk.Label(
            cam_bar, text="● IDLE", foreground="#555555", font=("Segoe UI", 10, "bold")
        )
        self.rec_badge_lbl.pack(side=tk.RIGHT)

        feeds_frame = ttk.Frame(right)
        feeds_frame.pack(fill=tk.BOTH, expand=True)
        feeds_frame.columnconfigure(0, weight=1)
        feeds_frame.columnconfigure(1, weight=1)
        feeds_frame.rowconfigure(0, weight=1)

        world_lf = ttk.LabelFrame(feeds_frame, text="World Camera (Webcam)", padding=6)
        world_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.world_feed_lbl = ttk.Label(world_lf, text="World camera stopped", anchor=tk.CENTER)
        self.world_feed_lbl.pack(fill=tk.BOTH, expand=True)
        self.world_fps_lbl = ttk.Label(world_lf, text="FPS: --")
        self.world_fps_lbl.pack(anchor=tk.W)

        wrist_lf = ttk.LabelFrame(feeds_frame, text="Wrist Camera (RealSense D405)", padding=6)
        wrist_lf.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.wrist_feed_lbl = ttk.Label(wrist_lf, text="Wrist camera stopped", anchor=tk.CENTER)
        self.wrist_feed_lbl.pack(fill=tk.BOTH, expand=True)
        self.wrist_fps_lbl = ttk.Label(wrist_lf, text="FPS: --")
        self.wrist_fps_lbl.pack(anchor=tk.W)

        telem_lf = ttk.LabelFrame(right, text="Live Robot Telemetry", padding=8)
        telem_lf.pack(fill=tk.X, pady=(8, 0))
        self.telem_tcp_lbl = ttk.Label(telem_lf, text="TCP (m, rad): --", font=("Consolas", 9))
        self.telem_tcp_lbl.pack(anchor=tk.W)
        self.telem_q_lbl = ttk.Label(telem_lf, text="Joints (rad): --", font=("Consolas", 9))
        self.telem_q_lbl.pack(anchor=tk.W)
        self.telem_grip_lbl = ttk.Label(
            telem_lf, text="Gripper: --   |   Active Step: Idle", font=("Consolas", 9)
        )
        self.telem_grip_lbl.pack(anchor=tk.W)

    def _refresh_anchor_capture_panel(self) -> None:
        if self._anchor_frame_inner is None:
            return
        for child in self._anchor_frame_inner.winfo_children():
            child.destroy()
        self._anchor_labels.clear()

        for row_idx, (name, anchor) in enumerate(self.preset.anchors.items()):
            ttk.Label(self._anchor_frame_inner, text=name, width=14, font=("Segoe UI", 9, "bold")).grid(
                row=row_idx, column=0, sticky="w", pady=2
            )
            btn = ttk.Button(
                self._anchor_frame_inner,
                text="Capture",
                width=9,
                command=lambda n=name: self._capture_anchor(n),
            )
            btn.grid(row=row_idx, column=1, padx=4, pady=2)
            status_lbl = ttk.Label(self._anchor_frame_inner, text=self._format_anchor_status(anchor))
            status_lbl.grid(row=row_idx, column=2, sticky="w", padx=4)
            self._anchor_labels[name] = status_lbl

        if hasattr(self, "seq_ref_combo"):
            names = list(self.preset.anchors.keys())
            self.seq_ref_combo["values"] = names
            self.lock_orient_combo["values"] = [""] + names

    @staticmethod
    def _format_anchor_status(anchor: AnchorPose) -> str:
        if not anchor.is_captured or anchor.tcp is None:
            return "Unset"
        return f"({anchor.tcp[0]:+.3f}, {anchor.tcp[1]:+.3f}, {anchor.tcp[2]:+.3f}) m"

    # ------------------------------------------------------------------
    # Tab 2: Sequence Builder
    # ------------------------------------------------------------------
    def _build_sequence_tab(self) -> None:
        top_bar = ttk.Frame(self.tab_sequence)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top_bar, text="Preset:").pack(side=tk.LEFT)
        self.preset_name_var = tk.StringVar(value=self.preset.name)
        ttk.Entry(top_bar, textvariable=self.preset_name_var, width=28).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(top_bar, text="Save Preset...", command=self._save_preset_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Load Preset...", command=self._load_preset_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Reset Default Titration", command=self._reset_default_preset).pack(
            side=tk.LEFT, padx=8
        )

        # Anchor bar
        anch_bar = ttk.LabelFrame(self.tab_sequence, text="Anchors & Orientation Lock", padding=6)
        anch_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(anch_bar, text="+ Add Anchor", command=self._add_anchor_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(anch_bar, text="Delete Anchor...", command=self._delete_anchor_dialog).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(anch_bar, text="Lock delta orientation to anchor:").pack(side=tk.LEFT, padx=(16, 4))
        self.lock_orient_var = tk.StringVar(value=self.settings.robot.lock_orientation_to_anchor)
        self.lock_orient_combo = ttk.Combobox(
            anch_bar,
            textvariable=self.lock_orient_var,
            values=[""] + list(self.preset.anchors.keys()),
            width=16,
            state="readonly",
        )
        self.lock_orient_combo.pack(side=tk.LEFT)
        self.lock_orient_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_lock_orient_changed())

        # Main split: tree + editor
        split = ttk.PanedWindow(self.tab_sequence, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)

        tree_frame = ttk.Frame(split)
        editor_lf = ttk.LabelFrame(split, text="Selected Step Editor", padding=10)
        split.add(tree_frame, weight=3)
        split.add(editor_lf, weight=2)

        cols = ("idx", "name", "type", "anchor", "delta", "frame", "motion", "rec", "phase", "status")
        self.seq_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        headers = {
            "idx": ("#", 32),
            "name": ("Step Name", 190),
            "type": ("Type", 85),
            "anchor": ("Anchor / Grip", 105),
            "delta": ("Delta (m)", 120),
            "frame": ("Frame", 55),
            "motion": ("Motion", 60),
            "rec": ("Rec", 42),
            "phase": ("Phase", 95),
            "status": ("Resolved XYZ / Status", 180),
        }
        for c, (label, width) in headers.items():
            self.seq_tree.heading(c, text=label)
            self.seq_tree.column(c, width=width, anchor=tk.W)
        self.seq_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.seq_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.seq_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.seq_tree.configure(yscrollcommand=sb.set)

        # Editor fields
        self.step_vars = {
            "name": tk.StringVar(),
            "step_type": tk.StringVar(value="delta_move"),
            "ref_anchor": tk.StringVar(),
            "dx": tk.StringVar(value="0.0"),
            "dy": tk.StringVar(value="0.0"),
            "dz": tk.StringVar(value="0.0"),
            "frame": tk.StringVar(value="base"),
            "motion_type": tk.StringVar(value="moveL"),
            "speed": tk.StringVar(value=""),
            "acc": tk.StringVar(value=""),
            "gripper_pos": tk.StringVar(value="180"),
            "gripper_ramp": tk.StringVar(value="0.0"),
            "dwell_s": tk.StringVar(value="0.1"),
            "record": tk.BooleanVar(value=True),
            "phase_tag": tk.StringVar(value=""),
        }

        grid = ttk.Frame(editor_lf)
        grid.pack(fill=tk.X)

        def _row(r: int, label: str, widget):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8))
            widget.grid(row=r, column=1, sticky="ew", pady=2)

        grid.columnconfigure(1, weight=1)
        _row(0, "Name", ttk.Entry(grid, textvariable=self.step_vars["name"]))
        _row(
            1,
            "Step Type",
            ttk.Combobox(
                grid,
                textvariable=self.step_vars["step_type"],
                values=["anchor_move", "delta_move", "gripper", "dwell"],
                state="readonly",
            ),
        )
        self.seq_ref_combo = ttk.Combobox(
            grid,
            textvariable=self.step_vars["ref_anchor"],
            values=list(self.preset.anchors.keys()),
            state="readonly",
        )
        _row(2, "Ref Anchor", self.seq_ref_combo)
        _row(3, "dx (m)", ttk.Entry(grid, textvariable=self.step_vars["dx"]))
        _row(4, "dy (m)", ttk.Entry(grid, textvariable=self.step_vars["dy"]))
        _row(5, "dz (m)", ttk.Entry(grid, textvariable=self.step_vars["dz"]))
        _row(
            6,
            "Delta Frame",
            ttk.Combobox(grid, textvariable=self.step_vars["frame"], values=["base", "tool"], state="readonly"),
        )
        _row(
            7,
            "Motion Type",
            ttk.Combobox(
                grid, textvariable=self.step_vars["motion_type"], values=["moveL", "moveJ"], state="readonly"
            ),
        )
        _row(8, "Speed override", ttk.Entry(grid, textvariable=self.step_vars["speed"]))
        _row(9, "Acc override", ttk.Entry(grid, textvariable=self.step_vars["acc"]))
        _row(10, "Gripper (0-255)", ttk.Entry(grid, textvariable=self.step_vars["gripper_pos"]))
        _row(11, "Gripper ramp (s)", ttk.Entry(grid, textvariable=self.step_vars["gripper_ramp"]))
        _row(12, "Dwell (s)", ttk.Entry(grid, textvariable=self.step_vars["dwell_s"]))
        _row(13, "Phase Tag", ttk.Entry(grid, textvariable=self.step_vars["phase_tag"]))
        _row(14, "Record Step", ttk.Checkbutton(grid, text="Include in recording", variable=self.step_vars["record"]))

        actions1 = ttk.Frame(editor_lf)
        actions1.pack(fill=tk.X, pady=(10, 4))
        ttk.Button(actions1, text="Apply Edit", command=self._apply_step_edit).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions1, text="Add Delta", command=lambda: self._add_step("delta_move")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions1, text="Add Anchor", command=lambda: self._add_step("anchor_move")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )

        actions2 = ttk.Frame(editor_lf)
        actions2.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(actions2, text="Add Gripper", command=lambda: self._add_step("gripper")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions2, text="Add Dwell", command=lambda: self._add_step("dwell")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions2, text="Duplicate", command=self._duplicate_step).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )

        actions3 = ttk.Frame(editor_lf)
        actions3.pack(fill=tk.X)
        ttk.Button(actions3, text="Move Up", command=lambda: self._move_step(-1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions3, text="Move Down", command=lambda: self._move_step(1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(actions3, text="Delete", command=self._delete_step).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )

    def _refresh_sequence_tree(self, select_idx: Optional[int] = None) -> None:
        self.seq_tree.delete(*self.seq_tree.get_children())
        resolved_all = validate_sequence(self.preset, self.settings.robot)
        for idx, (step, (_, res)) in enumerate(zip(self.preset.steps, resolved_all)):
            if step.step_type == "gripper":
                anchor_col = f"pos={step.gripper_pos}"
                delta_col = f"ramp={step.gripper_ramp_duration_s:.2f}s"
            elif step.step_type == "dwell":
                anchor_col = "--"
                delta_col = f"{step.dwell_s:.2f}s"
            else:
                anchor_col = step.ref_anchor
                delta_col = (
                    f"({step.dx:+.3f}, {step.dy:+.3f}, {step.dz:+.3f})"
                    if step.step_type == "delta_move"
                    else "(0, 0, 0)"
                )

            if not res.ok:
                status_col = f"ERR: {res.error}"
            elif res.tcp is not None:
                status_col = f"OK ({res.tcp[0]:+.3f}, {res.tcp[1]:+.3f}, {res.tcp[2]:+.3f})"
            else:
                status_col = "OK"

            self.seq_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    idx + 1,
                    step.name,
                    step.step_type,
                    anchor_col,
                    delta_col,
                    step.frame if step.step_type == "delta_move" else "--",
                    step.motion_type if step.step_type in ("anchor_move", "delta_move") else "--",
                    "Yes" if step.record else "No",
                    step.phase_tag,
                    status_col,
                ),
            )

        if select_idx is not None and 0 <= select_idx < len(self.preset.steps):
            self.seq_tree.selection_set(str(select_idx))
            self.seq_tree.see(str(select_idx))

    def _on_tree_select(self, _event=None) -> None:
        sel = self.seq_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        step = self.preset.steps[idx]
        self.step_vars["name"].set(step.name)
        self.step_vars["step_type"].set(step.step_type)
        self.step_vars["ref_anchor"].set(step.ref_anchor)
        self.step_vars["dx"].set(str(step.dx))
        self.step_vars["dy"].set(str(step.dy))
        self.step_vars["dz"].set(str(step.dz))
        self.step_vars["frame"].set(step.frame)
        self.step_vars["motion_type"].set(step.motion_type)
        self.step_vars["speed"].set("" if step.speed is None else str(step.speed))
        self.step_vars["acc"].set("" if step.acc is None else str(step.acc))
        self.step_vars["gripper_pos"].set(str(step.gripper_pos))
        self.step_vars["gripper_ramp"].set(str(step.gripper_ramp_duration_s))
        self.step_vars["dwell_s"].set(str(step.dwell_s))
        self.step_vars["record"].set(bool(step.record))
        self.step_vars["phase_tag"].set(step.phase_tag)

    def _step_from_editor(self) -> SequenceStep:
        spd_txt = self.step_vars["speed"].get().strip()
        acc_txt = self.step_vars["acc"].get().strip()
        return SequenceStep(
            name=self.step_vars["name"].get().strip() or "Unnamed Step",
            step_type=self.step_vars["step_type"].get().strip(),
            ref_anchor=self.step_vars["ref_anchor"].get().strip(),
            dx=float(self.step_vars["dx"].get().strip() or 0.0),
            dy=float(self.step_vars["dy"].get().strip() or 0.0),
            dz=float(self.step_vars["dz"].get().strip() or 0.0),
            frame=self.step_vars["frame"].get().strip() or "base",
            motion_type=self.step_vars["motion_type"].get().strip() or "moveL",
            speed=float(spd_txt) if spd_txt else None,
            acc=float(acc_txt) if acc_txt else None,
            gripper_pos=int(float(self.step_vars["gripper_pos"].get().strip() or 180)),
            gripper_ramp_duration_s=float(self.step_vars["gripper_ramp"].get().strip() or 0.0),
            dwell_s=float(self.step_vars["dwell_s"].get().strip() or 0.1),
            record=bool(self.step_vars["record"].get()),
            phase_tag=self.step_vars["phase_tag"].get().strip(),
        )

    def _apply_step_edit(self) -> None:
        sel = self.seq_tree.selection()
        if not sel:
            messagebox.showinfo("Sequence Builder", "Select a step first.")
            return
        idx = int(sel[0])
        try:
            self.preset.steps[idx] = self._step_from_editor()
        except ValueError as exc:
            messagebox.showerror("Invalid Value", str(exc))
            return
        self._refresh_sequence_tree(select_idx=idx)

    def _add_step(self, step_type: str) -> None:
        first_anchor = next(iter(self.preset.anchors.keys()), "pipette_grasp")
        new_step = SequenceStep(
            name=f"New {step_type}",
            step_type=step_type,
            ref_anchor=first_anchor if step_type in ("anchor_move", "delta_move") else "",
            dz=0.05 if step_type == "delta_move" else 0.0,
            gripper_pos=self.settings.robot.gripper_open,
            record=True,
        )
        sel = self.seq_tree.selection()
        insert_idx = int(sel[0]) + 1 if sel else len(self.preset.steps)
        self.preset.steps.insert(insert_idx, new_step)
        self._refresh_sequence_tree(select_idx=insert_idx)
        self._on_tree_select()

    def _duplicate_step(self) -> None:
        sel = self.seq_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        dup = copy.deepcopy(self.preset.steps[idx])
        dup.name = f"{dup.name} (copy)"
        self.preset.steps.insert(idx + 1, dup)
        self._refresh_sequence_tree(select_idx=idx + 1)

    def _move_step(self, direction: int) -> None:
        sel = self.seq_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        target = idx + direction
        if not (0 <= target < len(self.preset.steps)):
            return
        self.preset.steps[idx], self.preset.steps[target] = (
            self.preset.steps[target],
            self.preset.steps[idx],
        )
        self._refresh_sequence_tree(select_idx=target)

    def _delete_step(self) -> None:
        sel = self.seq_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        del self.preset.steps[idx]
        self._refresh_sequence_tree(select_idx=min(idx, len(self.preset.steps) - 1))

    # ------------------------------------------------------------------
    # Tab 3: Cameras Setup
    # ------------------------------------------------------------------
    def _build_cameras_tab(self) -> None:
        top = ttk.Frame(self.tab_cameras)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(top, text="Scan Connected Cameras", command=self._scan_cameras_async).pack(side=tk.LEFT)
        self.cam_scan_lbl = ttk.Label(top, text="Click Scan to detect USB webcams and RealSense D405.")
        self.cam_scan_lbl.pack(side=tk.LEFT, padx=10)

        cards = ttk.Frame(self.tab_cameras)
        cards.pack(fill=tk.BOTH, expand=True)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        cards.rowconfigure(0, weight=1)

        self._build_single_camera_card(
            parent=cards,
            col=0,
            key="world",
            title="World Camera (USB Webcam)",
            cfg=self.settings.world_camera,
        )
        self._build_single_camera_card(
            parent=cards,
            col=1,
            key="wrist",
            title="Wrist Camera (Intel RealSense D405)",
            cfg=self.settings.wrist_camera,
        )

    def _build_single_camera_card(self, parent, col: int, key: str, title: str, cfg: CameraConfig) -> None:
        lf = ttk.LabelFrame(parent, text=title, padding=10)
        lf.grid(row=0, column=col, sticky="nsew", padx=4)

        vars_map = {
            "backend": tk.StringVar(value=cfg.backend),
            "device_id": tk.StringVar(value=cfg.device_id),
            "width": tk.StringVar(value=str(cfg.width)),
            "height": tk.StringVar(value=str(cfg.height)),
            "fps": tk.StringVar(value=str(cfg.fps)),
            "flip_h": tk.BooleanVar(value=cfg.flip_h),
            "flip_v": tk.BooleanVar(value=cfg.flip_v),
            "rotate_180": tk.BooleanVar(value=cfg.rotate_180),
            "rs_auto_exp": tk.BooleanVar(value=cfg.rs_auto_exposure),
            "rs_exp": tk.StringVar(value=str(cfg.rs_exposure)),
        }

        form = ttk.Frame(lf)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)

        def _f(r: int, lbl: str, w):
            ttk.Label(form, text=lbl).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8))
            w.grid(row=r, column=1, sticky="ew", pady=2)

        _f(
            0,
            "Backend",
            ttk.Combobox(
                form, textvariable=vars_map["backend"], values=["opencv", "realsense", "mock"], state="readonly"
            ),
        )
        dev_combo = ttk.Combobox(form, textvariable=vars_map["device_id"], values=[cfg.device_id])
        _f(1, "Index / Serial", dev_combo)
        _f(2, "Width", ttk.Entry(form, textvariable=vars_map["width"]))
        _f(3, "Height", ttk.Entry(form, textvariable=vars_map["height"]))
        _f(4, "Target FPS", ttk.Entry(form, textvariable=vars_map["fps"]))

        orient_row = ttk.Frame(lf)
        orient_row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(orient_row, text="Flip H", variable=vars_map["flip_h"]).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(orient_row, text="Flip V", variable=vars_map["flip_v"]).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(orient_row, text="Rotate 180", variable=vars_map["rotate_180"]).pack(side=tk.LEFT, padx=4)

        rs_row = ttk.Frame(lf)
        rs_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(rs_row, text="D405 Auto-Exposure", variable=vars_map["rs_auto_exp"]).pack(side=tk.LEFT)
        ttk.Label(rs_row, text="Manual us:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(rs_row, textvariable=vars_map["rs_exp"], width=8).pack(side=tk.LEFT)

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill=tk.X, pady=6)
        ttk.Button(
            btn_row,
            text="Apply & Start",
            command=lambda k=key: self._start_single_camera(k),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(
            btn_row,
            text="Stop",
            command=lambda k=key: self._stop_single_camera(k),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        status_lbl = ttk.Label(lf, text="Stopped", foreground="#555555")
        status_lbl.pack(anchor=tk.W)
        preview_lbl = ttk.Label(lf, text="Camera preview", anchor=tk.CENTER)
        preview_lbl.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self._cam_widgets[key] = {
            "vars": vars_map,
            "dev_combo": dev_combo,
            "status_lbl": status_lbl,
            "preview_lbl": preview_lbl,
        }

    # ------------------------------------------------------------------
    # Tab 4: Settings & Export
    # ------------------------------------------------------------------
    def _build_settings_tab(self) -> None:
        outer = ttk.Frame(self.tab_settings)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        rec_lf = ttk.LabelFrame(outer, text="Recording Settings", padding=10)
        rec_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        rec_lf.columnconfigure(1, weight=1)

        self.set_vars = {
            "rec_fps": tk.StringVar(value=str(self.settings.recording.fps)),
            "raw_dir": tk.StringVar(value=self.settings.recording.raw_data_dir),
            "task_prompt": tk.StringVar(value=self.settings.recording.task_prompt),
            "codec": tk.StringVar(value=self.settings.recording.video_codec),
            "ip": tk.StringVar(value=self.settings.robot.ip),
            "gripper_port": tk.StringVar(value=str(self.settings.robot.gripper_port)),
            "movel_speed": tk.StringVar(value=str(self.settings.robot.movel_speed)),
            "movel_acc": tk.StringVar(value=str(self.settings.robot.movel_acc)),
            "movej_speed": tk.StringVar(value=str(self.settings.robot.movej_speed)),
            "movej_acc": tk.StringVar(value=str(self.settings.robot.movej_acc)),
            "z_min": tk.StringVar(value=str(self.settings.robot.z_min_safety_m)),
            "jitter_xy": tk.StringVar(value=str(self.settings.robot.home_jitter_xy_m)),
            "jitter_z": tk.StringVar(value=str(self.settings.robot.home_jitter_z_m)),
            "grip_open": tk.StringVar(value=str(self.settings.robot.gripper_open)),
            "grip_grasp": tk.StringVar(value=str(self.settings.robot.gripper_grasp)),
            "grip_squeeze": tk.StringVar(value=str(self.settings.robot.gripper_squeeze)),
            "grip_speed": tk.StringVar(value=str(self.settings.robot.gripper_speed)),
            "grip_force": tk.StringVar(value=str(self.settings.robot.gripper_force)),
        }

        ttk.Label(rec_lf, text="Recording FPS (10-30)").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Spinbox(rec_lf, from_=10, to=30, textvariable=self.set_vars["rec_fps"], width=8).grid(
            row=0, column=1, sticky="w", pady=3
        )

        ttk.Label(rec_lf, text="Raw Data Directory").grid(row=1, column=0, sticky="w", pady=3)
        dir_row = ttk.Frame(rec_lf)
        dir_row.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Entry(dir_row, textvariable=self.set_vars["raw_dir"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            dir_row,
            text="Browse...",
            command=lambda: self._browse_dir(self.set_vars["raw_dir"]),
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(rec_lf, text="Default Task Prompt").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(rec_lf, textvariable=self.set_vars["task_prompt"]).grid(row=2, column=1, sticky="ew", pady=3)

        ttk.Label(rec_lf, text="Video Codec").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            rec_lf, textvariable=self.set_vars["codec"], values=["mp4v", "MJPG"], state="readonly", width=10
        ).grid(row=3, column=1, sticky="w", pady=3)

        # Robot & Safety Settings
        rob_lf = ttk.LabelFrame(outer, text="Robot & Safety Settings", padding=10)
        rob_lf.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        rob_lf.columnconfigure(1, weight=1)
        rob_lf.columnconfigure(3, weight=1)

        fields = [
            ("UR5e IP", "ip", "Gripper Port", "gripper_port"),
            ("moveL speed (m/s)", "movel_speed", "moveL acc", "movel_acc"),
            ("moveJ speed (rad/s)", "movej_speed", "moveJ acc", "movej_acc"),
            ("z_min safety (m)", "z_min", "Home Jitter XY (m)", "jitter_xy"),
            ("Home Jitter Z (m)", "jitter_z", "Gripper Open", "grip_open"),
            ("Gripper Grasp", "grip_grasp", "Gripper Squeeze", "grip_squeeze"),
            ("Gripper Speed", "grip_speed", "Gripper Force", "grip_force"),
        ]
        for r, (l1, k1, l2, k2) in enumerate(fields):
            ttk.Label(rob_lf, text=l1).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 4))
            ttk.Entry(rob_lf, textvariable=self.set_vars[k1], width=14).grid(
                row=r, column=1, sticky="ew", pady=2, padx=(0, 10)
            )
            ttk.Label(rob_lf, text=l2).grid(row=r, column=2, sticky="w", pady=2, padx=(0, 4))
            ttk.Entry(rob_lf, textvariable=self.set_vars[k2], width=14).grid(row=r, column=3, sticky="ew", pady=2)

        set_btns = ttk.Frame(outer)
        set_btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Button(set_btns, text="Apply Settings", command=self._apply_settings_from_ui).pack(side=tk.LEFT, padx=2)
        ttk.Button(set_btns, text="Save Settings JSON...", command=self._save_settings_dialog).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(set_btns, text="Load Settings JSON...", command=self._load_settings_dialog).pack(
            side=tk.LEFT, padx=2
        )

        # LeRobot Converter Panel
        exp_lf = ttk.LabelFrame(outer, text="Convert Raw Episodes -> LeRobot Dataset", padding=10)
        exp_lf.grid(row=2, column=0, columnspan=2, sticky="nsew")
        outer.rowconfigure(2, weight=1)
        exp_lf.columnconfigure(1, weight=1)

        self.exp_out_var = tk.StringVar(value="./vla_data/lerobot_dataset")
        self.exp_repo_var = tk.StringVar(value="local/ur5e_titration")
        self.exp_space_var = tk.StringVar(value="joints")
        self.exp_split_var = tk.BooleanVar(value=False)
        self.exp_resize_var = tk.StringVar(value="")

        ttk.Label(exp_lf, text="Output Dataset Dir").grid(row=0, column=0, sticky="w", pady=3)
        out_row = ttk.Frame(exp_lf)
        out_row.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Entry(out_row, textvariable=self.exp_out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse...", command=lambda: self._browse_dir(self.exp_out_var)).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        opts_row = ttk.Frame(exp_lf)
        opts_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(opts_row, text="Repo ID:").pack(side=tk.LEFT)
        ttk.Entry(opts_row, textvariable=self.exp_repo_var, width=22).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opts_row, text="Primary State/Action:").pack(side=tk.LEFT)
        ttk.Combobox(
            opts_row, textvariable=self.exp_space_var, values=["joints", "tcp"], state="readonly", width=8
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opts_row, text="Resize (WxH):").pack(side=tk.LEFT)
        ttk.Combobox(
            opts_row, textvariable=self.exp_resize_var, values=["", "640x480", "320x240"], width=10
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(opts_row, text="Split by phase tags", variable=self.exp_split_var).pack(side=tk.LEFT)

        self.convert_btn = ttk.Button(
            exp_lf, text="Convert Raw -> LeRobot Now", command=self._run_lerobot_export
        )
        self.convert_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 6))

        self.exp_log_text = tk.Text(exp_lf, height=7, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
        self.exp_log_text.grid(row=3, column=0, columnspan=2, sticky="nsew")
        exp_lf.rowconfigure(3, weight=1)

    # ------------------------------------------------------------------
    # Logging & Settings Sync
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _exp_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.exp_log_text.configure(state=tk.NORMAL)
        self.exp_log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.exp_log_text.see(tk.END)
        self.exp_log_text.configure(state=tk.DISABLED)

    def _browse_dir(self, target_var: tk.StringVar) -> None:
        chosen = filedialog.askdirectory(initialdir=target_var.get() or ".")
        if chosen:
            target_var.set(chosen)

    def _sync_settings_to_ui(self) -> None:
        self.ip_var.set(self.settings.robot.ip)
        self.sim_var.set(self.settings.robot.simulation_mode)
        self.collect_prompt_var.set(self.settings.recording.task_prompt)
        self.lock_orient_var.set(self.settings.robot.lock_orientation_to_anchor)

        self.set_vars["rec_fps"].set(str(self.settings.recording.fps))
        self.set_vars["raw_dir"].set(self.settings.recording.raw_data_dir)
        self.set_vars["task_prompt"].set(self.settings.recording.task_prompt)
        self.set_vars["codec"].set(self.settings.recording.video_codec)
        self.set_vars["ip"].set(self.settings.robot.ip)
        self.set_vars["gripper_port"].set(str(self.settings.robot.gripper_port))
        self.set_vars["movel_speed"].set(str(self.settings.robot.movel_speed))
        self.set_vars["movel_acc"].set(str(self.settings.robot.movel_acc))
        self.set_vars["movej_speed"].set(str(self.settings.robot.movej_speed))
        self.set_vars["movej_acc"].set(str(self.settings.robot.movej_acc))
        self.set_vars["z_min"].set(str(self.settings.robot.z_min_safety_m))
        self.set_vars["jitter_xy"].set(str(self.settings.robot.home_jitter_xy_m))
        self.set_vars["jitter_z"].set(str(self.settings.robot.home_jitter_z_m))
        self.set_vars["grip_open"].set(str(self.settings.robot.gripper_open))
        self.set_vars["grip_grasp"].set(str(self.settings.robot.gripper_grasp))
        self.set_vars["grip_squeeze"].set(str(self.settings.robot.gripper_squeeze))
        self.set_vars["grip_speed"].set(str(self.settings.robot.gripper_speed))
        self.set_vars["grip_force"].set(str(self.settings.robot.gripper_force))

        for key, cfg in [("world", self.settings.world_camera), ("wrist", self.settings.wrist_camera)]:
            v = self._cam_widgets[key]["vars"]
            v["backend"].set(cfg.backend)
            v["device_id"].set(cfg.device_id)
            v["width"].set(str(cfg.width))
            v["height"].set(str(cfg.height))
            v["fps"].set(str(cfg.fps))
            v["flip_h"].set(cfg.flip_h)
            v["flip_v"].set(cfg.flip_v)
            v["rotate_180"].set(cfg.rotate_180)
            v["rs_auto_exp"].set(cfg.rs_auto_exposure)
            v["rs_exp"].set(str(cfg.rs_exposure))

    def _apply_settings_from_ui(self) -> bool:
        try:
            fps = int(self.set_vars["rec_fps"].get())
            if not (1 <= fps <= 60):
                raise ValueError("Recording FPS must be between 1 and 60.")
            self.settings.recording.fps = fps
            self.settings.recording.raw_data_dir = self.set_vars["raw_dir"].get().strip() or "./vla_data/raw"
            prompt = self.set_vars["task_prompt"].get().strip()
            if prompt:
                self.settings.recording.task_prompt = prompt
                self.collect_prompt_var.set(prompt)
            self.settings.recording.video_codec = self.set_vars["codec"].get().strip() or "mp4v"

            r = self.settings.robot
            r.ip = self.set_vars["ip"].get().strip() or "192.168.0.2"
            self.ip_var.set(r.ip)
            r.gripper_port = int(self.set_vars["gripper_port"].get())
            r.movel_speed = float(self.set_vars["movel_speed"].get())
            r.movel_acc = float(self.set_vars["movel_acc"].get())
            r.movej_speed = float(self.set_vars["movej_speed"].get())
            r.movej_acc = float(self.set_vars["movej_acc"].get())
            r.z_min_safety_m = float(self.set_vars["z_min"].get())
            r.home_jitter_xy_m = float(self.set_vars["jitter_xy"].get())
            r.home_jitter_z_m = float(self.set_vars["jitter_z"].get())
            r.gripper_open = int(self.set_vars["grip_open"].get())
            r.gripper_grasp = int(self.set_vars["grip_grasp"].get())
            r.gripper_squeeze = int(self.set_vars["grip_squeeze"].get())
            r.gripper_speed = int(self.set_vars["grip_speed"].get())
            r.gripper_force = int(self.set_vars["grip_force"].get())
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return False

        self._refresh_sequence_tree()
        self._refresh_episode_counter()
        self._log("Settings applied.")
        return True

    def _save_settings_dialog(self) -> None:
        if not self._apply_settings_from_ui():
            return
        DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(DEFAULT_SETTINGS_PATH.parent),
            initialfile=DEFAULT_SETTINGS_PATH.name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.settings.save_json(path)
            self._log(f"Saved settings to {path}")

    def _load_settings_dialog(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(DEFAULT_SETTINGS_PATH.parent) if DEFAULT_SETTINGS_PATH.parent.exists() else ".",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.settings = AppSettings.load_json(path)
            self._sync_settings_to_ui()
            self._refresh_sequence_tree()
            self._refresh_episode_counter()
            self._log(f"Loaded settings from {path}")

    # ------------------------------------------------------------------
    # Robot Connection, Freedrive, Anchors
    # ------------------------------------------------------------------
    def _on_sim_toggle(self) -> None:
        is_sim = bool(self.sim_var.get())
        self.settings.robot.simulation_mode = is_sim
        if is_sim:
            self._cam_widgets["world"]["vars"]["backend"].set("mock")
            self._cam_widgets["wrist"]["vars"]["backend"].set("mock")
            self._log("Switched to Simulation Mode (cameras defaulted to Mock).")
        else:
            self._cam_widgets["world"]["vars"]["backend"].set("opencv")
            self._cam_widgets["wrist"]["vars"]["backend"].set("realsense")
            self._log("Switched to Hardware Mode (world=OpenCV, wrist=RealSense).")

    def _toggle_robot_connect(self) -> None:
        if self.robot is not None and self.robot.is_connected():
            self._disconnect_robot()
        else:
            self._connect_robot()

    def _connect_robot(self) -> None:
        self.settings.robot.ip = self.ip_var.get().strip() or "192.168.0.2"
        self.settings.robot.simulation_mode = bool(self.sim_var.get())
        self.connect_btn.configure(state=tk.DISABLED)
        self.conn_status_lbl.configure(text="Connecting...", foreground="#b26a00")

        backend: RobotBackend = (
            SimulatedRobotBackend(self.settings.robot)
            if self.settings.robot.simulation_mode
            else UR5eRTDEBackend(self.settings.robot)
        )

        def worker():
            try:
                backend.connect()
                self.after(0, lambda: self._on_robot_connected(backend))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_robot_connect_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_robot_connected(self, backend: RobotBackend) -> None:
        self.robot = backend
        mode_tag = "SIM" if self.settings.robot.simulation_mode else "UR5e"
        self.conn_status_lbl.configure(text=f"Connected ({mode_tag})", foreground="#1b5e20")
        self.connect_btn.configure(text="Disconnect", state=tk.NORMAL)
        self._set_robot_controls_state(True)
        self._log(f"Connected to {mode_tag} backend.")
        if isinstance(backend, UR5eRTDEBackend) and backend.gripper_error:
            self._log(backend.gripper_error)

    def _on_robot_connect_failed(self, exc: Exception) -> None:
        self.robot = None
        self.conn_status_lbl.configure(text="Disconnected", foreground="#8b0000")
        self.connect_btn.configure(text="Connect Robot", state=tk.NORMAL)
        self._set_robot_controls_state(False)
        self._log(f"Connection failed: {exc}")
        messagebox.showerror("Robot Connection Failed", str(exc))

    def _disconnect_robot(self) -> None:
        if self._busy:
            self._abort_action()
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None
        self.conn_status_lbl.configure(text="Disconnected", foreground="#8b0000")
        self.connect_btn.configure(text="Connect Robot", state=tk.NORMAL)
        self._set_robot_controls_state(False)
        self._refresh_freedrive_buttons()
        self._log("Robot disconnected.")

    def _set_robot_controls_state(self, connected: bool) -> None:
        state = tk.NORMAL if (connected and not self._busy) else tk.DISABLED
        for btn in (
            self.fd_locked_btn,
            self.fd_6dof_btn,
            self.grip_open_btn,
            self.grip_hold_btn,
            self.grip_sq_btn,
            self.dry_run_btn,
            self.reset_btn,
            self.rec_run_btn,
        ):
            btn.configure(state=state)
        self.discard_btn.configure(
            state=tk.NORMAL if (self._last_episode is not None and not self._busy) else tk.DISABLED
        )

    def _toggle_freedrive(self, lock_orientation: bool) -> None:
        if self.robot is None or not self.robot.is_connected() or self._busy:
            return
        try:
            if self.robot.is_freedrive_active():
                self.robot.stop_freedrive()
                self._log("Freedrive disabled.")
            else:
                mode = self.robot.start_freedrive(lock_orientation=lock_orientation)
                self._log(f"Freedrive ENABLED ({mode}). Move arm manually, then toggle off.")
        except Exception as exc:
            self._log(f"Freedrive error: {exc}")
            messagebox.showerror("Freedrive Error", str(exc))
        self._refresh_freedrive_buttons()

    def _refresh_freedrive_buttons(self) -> None:
        active = self.robot is not None and self.robot.is_freedrive_active()
        self.fd_locked_btn.configure(style="ActiveFD.TButton" if active else "TButton")
        self.fd_6dof_btn.configure(style="ActiveFD.TButton" if active else "TButton")

    def _quick_gripper(self, pos: int) -> None:
        if self.robot is None or not self.robot.is_connected() or self._busy:
            return
        self._busy = True
        self._set_robot_controls_state(True)

        def worker():
            try:
                accepted = self.robot.move_gripper(
                    pos=pos,
                    speed=self.settings.robot.gripper_speed,
                    force=self.settings.robot.gripper_force,
                )
                if accepted:
                    self.after(0, lambda: self._log(f"Gripper command accepted for position {pos}."))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_quick_gripper_error(e))
            finally:
                self.after(0, self._on_worker_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_quick_gripper_error(self, exc: Exception) -> None:
        self._log(f"Gripper error: {exc}")
        messagebox.showerror("Gripper Error", str(exc))

    def _sim_nudge(self, dx: float, dy: float, dz: float) -> None:
        if isinstance(self.robot, SimulatedRobotBackend) and self.robot.is_connected():
            self.robot.sim_nudge(dx, dy, dz)

    def _seed_sim_anchors(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        samples = {
            "home": ([0.30, -0.25, 0.35, 3.14159, 0.0, 0.0], [0.90, -1.71, 2.04, -0.37, 0.97, 3.18]),
            "pipette_grasp": ([0.42, -0.18, 0.14, 3.14159, 0.0, 0.0], [1.08, -1.20, 2.06, -0.89, 1.06, 3.19]),
            "vial1_in": ([0.35, -0.32, 0.10, 3.14159, 0.0, 0.0], [0.98, -1.74, 2.42, -0.71, 1.96, 3.16]),
            "vial2_in": ([0.20, -0.35, 0.10, 3.14159, 0.0, 0.0], [1.57, -1.55, 1.99, -0.48, 1.49, 3.16]),
        }
        for name, anchor in self.preset.anchors.items():
            tcp, q = samples.get(
                name,
                ([0.30, -0.25, 0.25, 3.14159, 0.0, 0.0], [0.90, -1.71, 2.04, -0.37, 0.97, 3.18]),
            )
            anchor.tcp = list(tcp)
            anchor.joints = list(q)
            anchor.captured_at = now
        self._refresh_anchor_capture_panel()
        self._refresh_sequence_tree()
        self._log("Seeded simulated anchors for offline testing.")

    def _capture_anchor(self, name: str) -> None:
        if self.robot is None or not self.robot.is_connected():
            messagebox.showwarning("Not Connected", "Connect the robot before capturing anchors.")
            return
        state = self.robot.get_state()
        anchor = self.preset.anchors[name]
        anchor.tcp = [float(v) for v in state.tcp]
        anchor.joints = [float(v) for v in state.q]
        anchor.captured_at = datetime.now().isoformat(timespec="seconds")
        self._refresh_anchor_capture_panel()
        self._refresh_sequence_tree()
        self._log(f"Captured '{name}' at {self._format_anchor_status(anchor)}")

    # ------------------------------------------------------------------
    # Sequence & Recording Execution
    # ------------------------------------------------------------------
    def _start_sequence(self, record: bool, speed_scale: float) -> None:
        if self.robot is None or not self.robot.is_connected():
            messagebox.showwarning("Not Connected", "Connect the robot first.")
            return
        if self._busy:
            return

        if record and (
            self.world_cam is None
            or not self.world_cam.is_running()
            or self.wrist_cam is None
            or not self.wrist_cam.is_running()
        ):
            messagebox.showwarning(
                "Cameras Not Running",
                "Start both World and Wrist cameras before recording an episode.",
            )
            return

        prompt = self.collect_prompt_var.get().strip() or self.settings.recording.task_prompt
        self.settings.recording.task_prompt = prompt

        self._busy = True
        self._is_recording = record
        self._active_step_idx = None
        self._stop_event.clear()
        self._set_robot_controls_state(True)

        runner = SequenceRunner(self.robot, self.settings.robot)
        self.runner = runner
        recorder = (
            EpisodeRecorder(self.robot, self.world_cam, self.wrist_cam, self.settings.recording)
            if record
            else None
        )
        self.recorder = recorder

        def worker():
            completed = False
            ep_info: Optional[EpisodeInfo] = None
            try:
                if recorder is not None:
                    recorder.start(
                        task_prompt=prompt,
                        preset=self.preset,
                        robot_settings=self.settings.robot,
                        start_active=False,
                    )
                completed = runner.run(
                    preset=self.preset,
                    recorder=recorder,
                    speed_scale=speed_scale,
                    on_step=lambda idx, step: self.after(
                        0, lambda i=idx: setattr(self, "_active_step_idx", i)
                    ),
                    on_log=lambda msg: self.after(0, lambda m=msg: self._log(m)),
                )
                if recorder is not None:
                    ep_info = recorder.stop(save=completed)
            except Exception as exc:
                if recorder is not None:
                    try:
                        recorder.stop(save=False)
                    except Exception:
                        pass
                self.after(0, lambda e=exc: self._on_run_error(e))
            finally:
                self.after(0, lambda: self._on_sequence_finished(completed, record, ep_info))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_error(self, exc: Exception) -> None:
        self._log(f"Execution error: {exc}")
        messagebox.showerror("Execution Error", str(exc))

    def _on_sequence_finished(
        self,
        completed: bool,
        recorded: bool,
        ep_info: Optional[EpisodeInfo],
    ) -> None:
        self.runner = None
        self.recorder = None
        self._is_recording = False
        self._active_step_idx = None
        self._on_worker_done()

        if recorded and completed and ep_info is not None:
            self._last_episode = ep_info
            self._refresh_episode_counter()
            self._set_robot_controls_state(self.robot is not None and self.robot.is_connected())
            self._log(
                f"Saved {ep_info.episode_dir.name}: {ep_info.num_frames} frames "
                f"({ep_info.duration_s:.2f}s @ {ep_info.fps} FPS)."
            )
        elif recorded and not completed:
            self._log("Recording aborted — partial episode discarded.")
        elif completed:
            self._log("Dry run completed.")
        else:
            self._log("Dry run aborted.")

    def _start_auto_return(self) -> None:
        if self.robot is None or not self.robot.is_connected() or self._busy:
            return
        self._busy = True
        self._stop_event.clear()
        self._set_robot_controls_state(True)

        def worker():
            ok = False
            try:
                ok = auto_return_pipette_and_home(
                    robot=self.robot,
                    preset=self.preset,
                    settings=self.settings.robot,
                    stop_event=self._stop_event,
                    on_log=lambda msg: self.after(0, lambda m=msg: self._log(m)),
                )
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_run_error(e))
            finally:
                self.after(
                    0,
                    lambda: (
                        self._log("Auto-return finished." if ok else "Auto-return stopped."),
                        self._on_worker_done(),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_worker_done(self) -> None:
        self._busy = False
        self._set_robot_controls_state(self.robot is not None and self.robot.is_connected())
        self._refresh_freedrive_buttons()

    def _abort_action(self) -> None:
        self._stop_event.set()
        if self.runner is not None:
            self.runner.abort()
        if self.robot is not None:
            try:
                self.robot.stop_motion()
            except Exception:
                pass
        self._refresh_freedrive_buttons()
        self._log("STOP / ABORT triggered.")

    def _discard_last_episode(self) -> None:
        if self._last_episode is None:
            return
        ep = self._last_episode
        if delete_episode(ep.episode_dir):
            self._log(f"Discarded {ep.episode_dir.name}.")
        self._last_episode = None
        self._refresh_episode_counter()
        self._set_robot_controls_state(self.robot is not None and self.robot.is_connected())

    def _refresh_episode_counter(self) -> None:
        eps = list_raw_episodes(self.settings.recording.raw_data_dir)
        self.ep_count_lbl.configure(text=f"Episodes: {len(eps)}")

    # ------------------------------------------------------------------
    # Sequence Preset Helpers
    # ------------------------------------------------------------------
    def _add_anchor_dialog(self) -> None:
        name = simpledialog.askstring("New Anchor", "Enter anchor name (e.g., vial3_in):", parent=self)
        if not name:
            return
        clean = name.strip()
        if clean in self.preset.anchors:
            messagebox.showwarning("Duplicate", f"Anchor '{clean}' already exists.")
            return
        self.preset.anchors[clean] = AnchorPose(name=clean)
        self._refresh_anchor_capture_panel()
        self._refresh_sequence_tree()

    def _delete_anchor_dialog(self) -> None:
        name = simpledialog.askstring("Delete Anchor", "Enter anchor name to delete:", parent=self)
        if not name:
            return
        clean = name.strip()
        if clean in self.preset.anchors:
            del self.preset.anchors[clean]
            self._refresh_anchor_capture_panel()
            self._refresh_sequence_tree()

    def _on_lock_orient_changed(self) -> None:
        self.settings.robot.lock_orientation_to_anchor = self.lock_orient_var.get().strip()
        self._refresh_sequence_tree()

    def _save_preset_dialog(self) -> None:
        self.preset.name = self.preset_name_var.get().strip() or "Custom Sequence"
        DEFAULT_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(DEFAULT_PRESET_PATH.parent),
            initialfile=DEFAULT_PRESET_PATH.name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.preset.save_json(path)
            self._log(f"Saved preset to {path}")

    def _load_preset_dialog(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(DEFAULT_PRESET_PATH.parent) if DEFAULT_PRESET_PATH.parent.exists() else ".",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.preset = SequencePreset.load_json(path)
            self.preset_name_var.set(self.preset.name)
            self._refresh_anchor_capture_panel()
            self._refresh_sequence_tree()
            self._log(f"Loaded preset from {path}")

    def _reset_default_preset(self) -> None:
        self.preset = default_titration_preset()
        self.preset_name_var.set(self.preset.name)
        self._refresh_anchor_capture_panel()
        self._refresh_sequence_tree()
        self._log("Reset sequence to default titration workflow.")

    # ------------------------------------------------------------------
    # Cameras Control & Scanning
    # ------------------------------------------------------------------
    def _camera_config_from_ui(self, key: str) -> CameraConfig:
        v = self._cam_widgets[key]["vars"]
        return CameraConfig(
            name=key,
            backend=v["backend"].get().strip(),
            device_id=v["device_id"].get().strip(),
            width=int(v["width"].get()),
            height=int(v["height"].get()),
            fps=int(v["fps"].get()),
            flip_h=bool(v["flip_h"].get()),
            flip_v=bool(v["flip_v"].get()),
            rotate_180=bool(v["rotate_180"].get()),
            rs_auto_exposure=bool(v["rs_auto_exp"].get()),
            rs_exposure=int(float(v["rs_exp"].get() or 8500)),
        )

    def _start_single_camera(self, key: str) -> None:
        self._stop_single_camera(key)
        try:
            cfg = self._camera_config_from_ui(key)
            cam = create_camera(cfg)
            cam.start()
        except Exception as exc:
            self._cam_widgets[key]["status_lbl"].configure(text=f"Error: {exc}", foreground="#8b0000")
            self._log(f"Camera '{key}' failed to start: {exc}")
            messagebox.showerror("Camera Error", f"{key}: {exc}")
            return

        if key == "world":
            self.settings.world_camera = cfg
            self.world_cam = cam
        else:
            self.settings.wrist_camera = cfg
            self.wrist_cam = cam

        self._cam_widgets[key]["status_lbl"].configure(
            text=f"Running ({cfg.backend}, id={cfg.device_id or 'auto'})",
            foreground="#1b5e20",
        )
        self._log(f"Started {key} camera ({cfg.backend}).")

    def _stop_single_camera(self, key: str) -> None:
        cam = self.world_cam if key == "world" else self.wrist_cam
        if cam is not None:
            cam.stop()
        if key == "world":
            self.world_cam = None
            self.world_feed_lbl.configure(image="", text="World camera stopped")
            self.world_fps_lbl.configure(text="FPS: --")
        else:
            self.wrist_cam = None
            self.wrist_feed_lbl.configure(image="", text="Wrist camera stopped")
            self.wrist_fps_lbl.configure(text="FPS: --")
        self._cam_widgets[key]["status_lbl"].configure(text="Stopped", foreground="#555555")
        self._cam_widgets[key]["preview_lbl"].configure(image="", text="Camera preview")

    def _start_all_cameras(self) -> None:
        self._start_single_camera("world")
        self._start_single_camera("wrist")

    def _stop_all_cameras(self) -> None:
        self._stop_single_camera("world")
        self._stop_single_camera("wrist")

    def _scan_cameras_async(self) -> None:
        self.cam_scan_lbl.configure(text="Scanning cameras...")

        def worker():
            ocv = list_opencv_cameras()
            rs_devs = list_realsense_devices()
            self.after(0, lambda: self._on_cameras_scanned(ocv, rs_devs))

        threading.Thread(target=worker, daemon=True).start()

    def _on_cameras_scanned(self, ocv, rs_devs) -> None:
        rs_serials = [s for s, _ in rs_devs]
        all_Choices = ocv + rs_serials
        if ocv:
            self._cam_widgets["world"]["dev_combo"]["values"] = ocv
        if all_Choices:
            self._cam_widgets["wrist"]["dev_combo"]["values"] = rs_serials or ocv
        rs_desc = ", ".join(f"{name} ({ser})" for ser, name in rs_devs) or "none"
        ocv_desc = ", ".join(ocv) or "none"
        msg = f"OpenCV indices: [{ocv_desc}] | RealSense: [{rs_desc}]"
        self.cam_scan_lbl.configure(text=msg)
        self._log(f"Camera scan -> {msg}")

    # ------------------------------------------------------------------
    # LeRobot Export
    # ------------------------------------------------------------------
    def _run_lerobot_export(self) -> None:
        if not self._apply_settings_from_ui():
            return
        raw_dir = self.settings.recording.raw_data_dir
        out_dir = self.exp_out_var.get().strip()
        repo_id = self.exp_repo_var.get().strip() or "local/ur5e_titration"
        action_space = self.exp_space_var.get().strip() or "joints"
        split_phase = bool(self.exp_split_var.get())
        resize_txt = self.exp_resize_var.get().strip().lower()
        resize_hw: Optional[Tuple[int, int]] = None
        if "x" in resize_txt:
            w_s, h_s = resize_txt.split("x", 1)
            resize_hw = (int(h_s), int(w_s))

        self.convert_btn.configure(state=tk.DISABLED)
        self._exp_log(f"Starting conversion from {raw_dir} -> {out_dir}...")

        def worker():
            try:
                summary = convert_raw_to_lerobot(
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                    repo_id=repo_id,
                    action_space=action_space,
                    split_by_phase=split_phase,
                    resize_hw=resize_hw,
                    overwrite=True,
                    progress_cb=lambda m: self.after(0, lambda msg=m: self._exp_log(msg)),
                )
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Conversion Complete",
                        f"Converted {summary.episodes_converted} episodes ({summary.total_frames} frames)\n"
                        f"Saved to: {summary.output_dir}",
                    ),
                )
            except Exception as exc:
                self.after(0, lambda e=exc: self._exp_log(f"Conversion error: {e}"))
                self.after(0, lambda e=exc: messagebox.showerror("Conversion Failed", str(e)))
            finally:
                self.after(0, lambda: self.convert_btn.configure(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Periodic UI Refresh & Keyboard Shortcuts
    # ------------------------------------------------------------------
    def _render_cam_to_label(
        self,
        cam: Optional[BaseCamera],
        lbl_main: ttk.Label,
        lbl_tab: ttk.Label,
        fps_lbl: ttk.Label,
        size: Tuple[int, int] = (380, 285),
    ) -> None:
        if cam is None or not cam.is_running():
            return
        frame = cam.get_latest()
        if frame is None:
            return
        fps_lbl.configure(text=f"FPS: {cam.measured_fps:.1f}")

        if Image is None or ImageTk is None:
            return
        img_bgr = frame.image
        if cv2 is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb).resize(size)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        lbl_main.configure(image=tk_img, text="")
        lbl_main.image = tk_img
        lbl_tab.configure(image=tk_img, text="")
        lbl_tab.image = tk_img

    def _ui_poll_loop(self) -> None:
        try:
            self._render_cam_to_label(
                self.world_cam,
                self.world_feed_lbl,
                self._cam_widgets["world"]["preview_lbl"],
                self.world_fps_lbl,
            )
            self._render_cam_to_label(
                self.wrist_cam,
                self.wrist_feed_lbl,
                self._cam_widgets["wrist"]["preview_lbl"],
                self.wrist_fps_lbl,
            )

            if self._is_recording:
                self.rec_badge_lbl.configure(text="● RECORDING", foreground="#c62828")
            elif self._busy:
                self.rec_badge_lbl.configure(text="● RUNNING", foreground="#ef6c00")
            else:
                self.rec_badge_lbl.configure(text="● IDLE", foreground="#555555")

            if self.robot is not None and self.robot.is_connected():
                st = self.robot.get_state()
                self.telem_tcp_lbl.configure(
                    text="TCP (m, rad): [" + ", ".join(f"{v:+.3f}" for v in st.tcp) + "]"
                )
                self.telem_q_lbl.configure(
                    text="Joints (rad): [" + ", ".join(f"{v:+.3f}" for v in st.q) + "]"
                )
                step_txt = (
                    f"{self._active_step_idx + 1}/{len(self.preset.steps)} "
                    f"({self.preset.steps[self._active_step_idx].name})"
                    if self._active_step_idx is not None
                    and 0 <= self._active_step_idx < len(self.preset.steps)
                    else "Idle"
                )
                self.telem_grip_lbl.configure(
                    text=f"Gripper: {st.gripper_raw:.0f}/255 ({st.gripper_norm:.2f})   |   Step: {step_txt}"
                )
        except Exception:
            pass
        self.after(66, self._ui_poll_loop)  # ~15 Hz UI preview refresh

    def _on_escape_key(self, _event=None) -> None:
        self._abort_action()

    def _on_space_key(self, event=None) -> None:
        widget = self.focus_get()
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox, ttk.Spinbox)):
            return
        self._toggle_freedrive(lock_orientation=True)

    def _on_close(self) -> None:
        self._abort_action()
        self._stop_all_cameras()
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    app = VLACollectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
