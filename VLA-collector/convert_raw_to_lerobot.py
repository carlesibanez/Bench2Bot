"""CLI entry point to convert raw collected UR5e episodes into a LeRobotDataset."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

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
        choices=["joints", "tcp"],
        default="joints",
        help="Primary representation for observation.state/action (default: joints; complementary representation is always stored too).",
    )
    parser.add_argument(
        "--split-by-phase",
        action="store_true",
        help="Optionally slice each episode into separate per-phase LeRobot episodes (grasp_pipette, aspirate, dispense).",
    )
    parser.add_argument(
        "--resize",
        default="",
        help="Optional resize WxH (e.g. 320x240 or 640x480) applied during conversion.",
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


def main() -> int:
    args = parse_args()
    resize_hw = _parse_resize(args.resize)

    summary = convert_raw_to_lerobot(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        action_space=args.action_space,
        split_by_phase=bool(args.split_by_phase),
        resize_hw=resize_hw,
        overwrite=not args.no_overwrite,
        progress_cb=print,
    )
    print(
        f"\nConversion Summary:\n"
        f"  Raw episodes : {summary.episodes_converted}\n"
        f"  LR episodes  : {summary.segments_written}\n"
        f"  Total frames : {summary.total_frames}\n"
        f"  FPS          : {summary.fps}\n"
        f"  Primary space: {summary.action_space} (+ complementary saved)\n"
        f"  Output dir   : {summary.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
