"""CLI entry point to convert raw collected UR5e episodes into a LeRobotDataset."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from vla_collector.lerobot_converter import convert_raw_to_lerobot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw UR5e VLA episodes into a LeRobotDataset."
    )
    parser.add_argument(
        "--raw-dir",
        default="./vla_data/raw",
        help="Path to raw collection root containing episodes/episode_XXXX (default: ./vla_data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        default="./vla_data/lerobot_dataset",
        help="Output directory for the generated LeRobotDataset (default: ./vla_data/lerobot_dataset).",
    )
    parser.add_argument(
        "--repo-id",
        default="local/ur5e_titration",
        help="LeRobot dataset repo_id metadata string (default: local/ur5e_titration).",
    )
    parser.add_argument(
        "--action-space",
        choices=["joints", "tcp", "xyz_gripper"],
        default="xyz_gripper",
        help="Primary representation for observation.state/action (default: xyz_gripper).",
    )
    parser.add_argument(
        "--action-mode",
        choices=["absolute", "delta"],
        default="delta",
        help="Action formulation: 'delta' (delta arm + absolute next gripper) or 'absolute' (default: delta).",
    )
    parser.add_argument(
        "--gripper-units",
        choices=["norm", "raw"],
        default="raw",
        help="Gripper representation: 'raw' (0..255) or 'norm' (0.0..1.0) (default: raw).",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also store secondary tcp/joints state and action keys (off by default to keep 4 standard keys).",
    )
    parser.add_argument(
        "--episodes",
        default="",
        help="Optional comma-separated episode indices to include (e.g. '2,3,4').",
    )
    parser.add_argument(
        "--exclude-episodes",
        default="",
        help="Optional comma-separated episode indices to exclude (e.g. '0').",
    )
    parser.add_argument(
        "--world-crop",
        default="",
        help="Optional crop box 'x0,y0,x1,y1' applied to world camera frames before resize (e.g. '0,0,530,480').",
    )
    parser.add_argument(
        "--split-by-phase",
        action="store_true",
        help="Optionally slice each episode into separate per-phase LeRobot episodes (grasp_pipette, aspirate, dispense).",
    )
    parser.add_argument(
        "--resize",
        default="",
        help="Optional resize WxH (e.g. 224x224 or 320x240) applied during conversion.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "pad"],
        default="pad",
        help="How --resize scales frames: aspect-preserving 'pad' (letterbox) or 'stretch' (default: pad).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if --out-dir already exists instead of replacing it.",
    )
    return parser.parse_args()


def _parse_resize(spec: str) -> Optional[Tuple[int, int]]:
    text = spec.strip().lower()
    if not text:
        return None
    if "x" not in text:
        raise ValueError(f"Invalid --resize format '{spec}'. Expected WxH, e.g. 640x480.")
    w_str, h_str = text.split("x", 1)
    return (int(h_str), int(w_str))


def _parse_int_list(spec: str) -> Optional[List[int]]:
    text = spec.strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_crop_box(spec: str) -> Optional[Tuple[int, int, int, int]]:
    text = spec.strip()
    if not text:
        return None
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid --world-crop format '{spec}'. Expected 'x0,y0,x1,y1'.")
    return (parts[0], parts[1], parts[2], parts[3])


def main() -> int:
    args = parse_args()
    resize_hw = _parse_resize(args.resize)
    include_episodes = _parse_int_list(args.episodes)
    exclude_episodes = _parse_int_list(args.exclude_episodes)
    world_crop_xyxy = _parse_crop_box(args.world_crop)

    summary = convert_raw_to_lerobot(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        action_space=args.action_space,
        split_by_phase=bool(args.split_by_phase),
        resize_hw=resize_hw,
        overwrite=not args.no_overwrite,
        progress_cb=print,
        action_mode=args.action_mode,
        gripper_units=args.gripper_units,
        include_secondary=bool(args.include_secondary),
        include_episodes=include_episodes,
        exclude_episodes=exclude_episodes,
        world_crop_xyxy=world_crop_xyxy,
        resize_mode=args.resize_mode,
    )
    print(
        f"\nConversion Summary:\n"
        f"  Raw episodes : {summary.episodes_converted}\n"
        f"  LR episodes  : {summary.segments_written}\n"
        f"  Total frames : {summary.total_frames}\n"
        f"  FPS          : {summary.fps}\n"
        f"  Space / Mode : {summary.action_space} ({summary.action_mode}, gripper={summary.gripper_units})\n"
        f"  Output dir   : {summary.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
