"""Unit tests for the portable pi0_training launcher and offline evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PI0_TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(PI0_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PI0_TRAINING_DIR))

import eval_pi0_offline  # noqa: E402
import train_pi0  # noqa: E402


REAL_TRAIN_DATASET = Path(__file__).resolve().parents[3] / "vla_data" / "lerobot_pi0_train"


def _create_mock_lerobot_v3_dataset(
    root: Path,
    *,
    codebase_version: str = "v3.0",
    total_episodes: int = 3,
    total_frames: int = 120,
    action_names: list[str] | None = None,
    include_stats: bool = True,
) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "observation.images.world" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": codebase_version,
        "robot_type": "ur5e_robotiq",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": 20,
        "features": {
            "observation.images.world": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.codec": "av1", "video.fps": 20},
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.codec": "av1", "video.fps": 20},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": [4],
                "names": action_names or ["dx", "dy", "dz", "gripper"],
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    if include_stats:
        (meta / "stats.json").write_text("{}", encoding="utf-8")
    (meta / "tasks.parquet").write_bytes(b"PAR1")
    return root


class TestDatasetInspection(unittest.TestCase):
    def test_valid_mock_v3_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            insp = train_pi0.inspect_lerobot_dataset(ds_dir)
            self.assertEqual(insp.codebase_version, "v3.0")
            self.assertEqual(insp.total_episodes, 3)
            self.assertEqual(insp.total_frames, 120)
            self.assertEqual(insp.fps, 20)
            self.assertIn("observation.images.world", insp.camera_features)
            self.assertIn("observation.images.wrist", insp.camera_features)
            self.assertEqual(insp.state_names, ["x", "y", "z", "gripper"])
            self.assertEqual(insp.action_names, ["dx", "dy", "dz", "gripper"])
            self.assertTrue(insp.is_delta_action)

    def test_absolute_action_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(
                Path(tmp) / "ds", action_names=["x", "y", "z", "gripper"]
            )
            insp = train_pi0.inspect_lerobot_dataset(ds_dir)
            self.assertFalse(insp.is_delta_action)

    def test_missing_stats_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds", include_stats=False)
            with self.assertRaises(FileNotFoundError):
                train_pi0.inspect_lerobot_dataset(ds_dir)

    def test_empty_episodes_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(
                Path(tmp) / "ds", total_episodes=0, total_frames=0
            )
            with self.assertRaises(ValueError):
                train_pi0.inspect_lerobot_dataset(ds_dir)

    def test_wrong_codebase_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds", codebase_version="v2.1")
            with self.assertRaises(ValueError):
                train_pi0.inspect_lerobot_dataset(ds_dir)

    @unittest.skipUnless(REAL_TRAIN_DATASET.is_dir(), "Local lerobot_pi0_train dataset not present")
    def test_real_local_lerobot_pi0_train_dataset(self):
        insp = train_pi0.inspect_lerobot_dataset(REAL_TRAIN_DATASET)
        self.assertEqual(insp.codebase_version, "v3.0")
        self.assertGreater(insp.total_episodes, 0)
        self.assertGreater(insp.total_frames, 0)
        self.assertTrue(insp.is_delta_action)


class TestCommandBuilder(unittest.TestCase):
    def test_default_lora_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            out_dir = Path(tmp) / "outputs" / "run1"
            args = train_pi0.parse_args(["--dataset-dir", str(ds_dir), "--output-dir", str(out_dir)])
            cmd = train_pi0.build_train_command(args, "bfloat16", out_dir, ["lerobot-train"])

            self.assertEqual(cmd[0], "lerobot-train")
            self.assertIn("--policy.type=pi0", cmd)
            self.assertIn("--policy.pretrained_path=lerobot/pi0_base", cmd)
            self.assertIn("--policy.push_to_hub=false", cmd)
            self.assertFalse(any(c.startswith("--wandb.") for c in cmd))
            self.assertFalse(any(c.startswith("--trackio.") for c in cmd))
            self.assertIn("--peft.method_type=LORA", cmd)
            self.assertIn("--peft.r=16", cmd)
            self.assertIn("--policy.dtype=bfloat16", cmd)
            self.assertIn("--dataset.video_backend=pyav", cmd)
            self.assertNotIn("--policy.use_relative_actions=true", cmd)
            # Ensure output_dir was NOT created by build_train_command
            self.assertFalse(out_dir.exists())

    def test_tracker_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            out_dir = Path(tmp) / "outputs" / "run_track"

            args_trackio = train_pi0.parse_args([
                "--dataset-dir", str(ds_dir),
                "--trackio", "--trackio-project", "myproj", "--trackio-space-id", "user/space",
            ])
            cmd_t = train_pi0.build_train_command(
                args_trackio, "bfloat16", out_dir, ["lerobot-train"], tracker_backend="trackio"
            )
            self.assertIn("--trackio.enable=true", cmd_t)
            self.assertIn("--trackio.project=myproj", cmd_t)
            self.assertIn("--trackio.space_id=user/space", cmd_t)
            self.assertFalse(any(c.startswith("--wandb.") for c in cmd_t))

            args_legacy_on_07 = train_pi0.parse_args([
                "--dataset-dir", str(ds_dir), "--wandb", "--wandb-project", "myproj"
            ])
            cmd_mapped = train_pi0.build_train_command(
                args_legacy_on_07, "bfloat16", out_dir, ["lerobot-train"], tracker_backend="trackio"
            )
            self.assertIn("--trackio.enable=true", cmd_mapped)
            self.assertIn("--trackio.project=myproj", cmd_mapped)
            self.assertFalse(any(c.startswith("--wandb.") for c in cmd_mapped))

            args_legacy_on_04 = train_pi0.parse_args([
                "--dataset-dir", str(ds_dir),
                "--wandb", "--wandb-project", "myproj", "--wandb-entity", "myteam",
            ])
            cmd_w = train_pi0.build_train_command(
                args_legacy_on_04, "bfloat16", out_dir, ["lerobot-train"], tracker_backend="wandb"
            )
            self.assertIn("--wandb.enable=true", cmd_w)
            self.assertIn("--wandb.project=myproj", cmd_w)
            self.assertIn("--wandb.entity=myteam", cmd_w)

    def test_other_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            out_dir = Path(tmp) / "outputs" / "run"

            for mode, expected_flags in [
                ("full", ["--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false"]),
                ("freeze-vision", ["--policy.freeze_vision_encoder=true", "--policy.train_expert_only=false"]),
                ("expert-only", ["--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true"]),
            ]:
                args = train_pi0.parse_args(["--dataset-dir", str(ds_dir), "--mode", mode])
                cmd = train_pi0.build_train_command(args, "bfloat16", out_dir, ["lerobot-train"])
                for flag in expected_flags:
                    self.assertIn(flag, cmd)
                self.assertFalse(any(c.startswith("--peft.") for c in cmd))

    def test_resume_command_resolves_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_01"
            cfg = run_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text("{}", encoding="utf-8")

            args = train_pi0.parse_args(["--resume", str(run_dir), "--steps", "20000"])
            cmd = train_pi0.build_train_command(args, "bfloat16", None, ["lerobot-train"])
            self.assertEqual(cmd, ["lerobot-train", f"--config_path={cfg.resolve()}", "--resume=true", "--steps=20000"])

    def test_multi_gpu_command_uses_accelerate(self):
        base_cmd = train_pi0.resolve_lerobot_train_cmd(num_gpus=2)
        self.assertIn("launch", base_cmd)
        self.assertIn("--multi_gpu", base_cmd)
        self.assertIn("--num_processes=2", base_cmd)
        self.assertIn("lerobot.scripts.lerobot_train", base_cmd)

    def test_push_to_hub_requires_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            out_dir = Path(tmp) / "out"
            args = train_pi0.parse_args(["--dataset-dir", str(ds_dir), "--push-to-hub"])
            with self.assertRaises(ValueError):
                train_pi0.build_train_command(args, "bfloat16", out_dir, ["lerobot-train"])

            args_ok = train_pi0.parse_args([
                "--dataset-dir", str(ds_dir), "--push-to-hub", "--policy-repo-id", "user/pi0_ur5e"
            ])
            cmd = train_pi0.build_train_command(args_ok, "bfloat16", out_dir, ["lerobot-train"])
            self.assertIn("--policy.push_to_hub=true", cmd)
            self.assertIn("--policy.repo_id=user/pi0_ur5e", cmd)

    def test_dry_run_warns_on_double_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = _create_mock_lerobot_v3_dataset(Path(tmp) / "ds")
            out_dir = Path(tmp) / "out"
            # Should exit 0 on dry-run while emitting warning on stderr
            rc = train_pi0.main([
                "--dataset-dir", str(ds_dir),
                "--output-dir", str(out_dir),
                "--use-relative-actions",
                "--dry-run",
                "--skip-env-check",
                "--skip-gpu-check",
                "--skip-hf-check",
            ])
            self.assertEqual(rc, 0)


class TestOfflineEvaluatorHelpers(unittest.TestCase):
    def test_discover_checkpoints_single_and_multi(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            for step in ("004000", "002000", "last"):
                p = run_dir / "checkpoints" / step / "pretrained_model"
                p.mkdir(parents=True, exist_ok=True)
                (p / "config.json").write_text("{}", encoding="utf-8")

            discovered = eval_pi0_offline.discover_checkpoints(run_dir)
            labels = [eval_pi0_offline._checkpoint_label(p) for p in discovered]
            self.assertEqual(labels, ["002000", "004000", "last"])

            single = eval_pi0_offline.discover_checkpoints(
                run_dir / "checkpoints" / "002000" / "pretrained_model"
            )
            self.assertEqual(len(single), 1)
            self.assertEqual(eval_pi0_offline._checkpoint_label(single[0]), "002000")

    def test_compute_chunk_error_metrics_excludes_padding(self):
        # Shape: (N=1, T=3, D=2). Last timestep (t=2) is padded with huge error and must be ignored.
        preds = np.array([[[1.0, 10.0], [2.0, 20.0], [999.0, 999.0]]])
        tgts = np.array([[[1.0, 12.0], [4.0, 20.0], [0.0, 0.0]]])
        pads = np.array([[False, False, True]])

        metrics = eval_pi0_offline.compute_chunk_error_metrics(
            preds, tgts, action_is_pad=pads, action_names=["dx", "gripper"]
        )
        # Valid diffs: t=0 -> [0, -2], t=1 -> [-2, 0]. Mean abs error = (0+2+2+0)/4 = 1.0
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["mse"], 2.0)
        self.assertAlmostEqual(metrics["step0_mae"], 1.0)
        self.assertAlmostEqual(metrics["per_dim"]["dx"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["per_dim"]["gripper"]["mae"], 1.0)
        self.assertEqual(metrics["valid_steps"], 2)

    def test_eval_parse_args_n_action_steps(self):
        args = eval_pi0_offline.parse_args([
            "--checkpoint", "c",
            "--dataset-dir", "d",
            "--n-action-steps", "16",
            "--stride", "16",
            "--only-checkpoints", "002000,010000",
            "--save-predictions", "preds.json",
        ])
        self.assertEqual(args.n_action_steps, 16)
        self.assertEqual(args.stride, 16)
        self.assertEqual(args.only_checkpoints, "002000,010000")
        self.assertEqual(args.save_predictions, "preds.json")

    def test_filter_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            for step in ("002000", "010000", "016000", "020000"):
                p = run_dir / "checkpoints" / step / "pretrained_model"
                p.mkdir(parents=True, exist_ok=True)
                (p / "config.json").write_text("{}", encoding="utf-8")

            all_ckpts = eval_pi0_offline.discover_checkpoints(run_dir)
            filtered = eval_pi0_offline.filter_checkpoints(all_ckpts, "2000, 016000")
            labels = [eval_pi0_offline._checkpoint_label(p) for p in filtered]
            self.assertEqual(labels, ["002000", "016000"])

            with self.assertRaises(ValueError):
                eval_pi0_offline.filter_checkpoints(all_ckpts, "002000,999999")


if __name__ == "__main__":
    unittest.main()
