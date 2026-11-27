#!/usr/bin/env python3
"""Offline checkpoint evaluator for fine-tuned LeRobot π₀ policies.

Evaluates one or more saved checkpoints (``pretrained_model/``) against a
held-out local LeRobot v3.0 dataset (e.g. ``lerobot_pi0_test``), reporting
action-chunk MSE/MAE and per-dimension error in physical units (meters for
``dx, dy, dz``; raw units for ``gripper``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _ckpt_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort numbered step checkpoints chronologically, with 'last' at the end."""
    # Typical layout: <run>/checkpoints/002000/pretrained_model
    parent_name = path.parent.name if path.name == "pretrained_model" else path.name
    if parent_name.isdigit():
        return (0, int(parent_name), str(path))
    if parent_name == "last":
        return (1, 0, str(path))
    return (2, 0, str(path))


def _checkpoint_label(path: Path) -> str:
    if path.name == "pretrained_model" and path.parent.name:
        return path.parent.name
    return path.name


def discover_checkpoints(target: str | Path) -> list[Path]:
    """Discover one or more ``pretrained_model`` checkpoint directories."""
    root = Path(target).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")

    if (root / "config.json").is_file():
        return [root]

    if (root / "pretrained_model" / "config.json").is_file():
        return [root / "pretrained_model"]

    search_base = root / "checkpoints" if (root / "checkpoints").is_dir() else root
    found: list[Path] = []
    seen_resolved: set[Path] = set()

    for child in sorted(search_base.iterdir(), key=_ckpt_sort_key):
        if not child.is_dir():
            continue
        candidate = child / "pretrained_model" if (child / "pretrained_model").is_dir() else child
        if not (candidate / "config.json").is_file():
            continue
        resolved = candidate.resolve()
        # Avoid double-counting 'last' if it is a symlink/copy pointing to the same step dir.
        if child.name == "last" and resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        found.append(candidate)

    if not found:
        raise FileNotFoundError(
            f"No LeRobot checkpoints (directories containing config.json) found under {root}"
        )
    return found


def filter_checkpoints(checkpoints: Sequence[Path], only_spec: str | None) -> list[Path]:
    """Filter discovered checkpoints by comma-separated labels (e.g. '002000,010000')."""
    if not only_spec or not only_spec.strip():
        return list(checkpoints)

    tokens = [part.strip() for part in only_spec.split(",") if part.strip()]
    available = [(_checkpoint_label(p), p) for p in checkpoints]

    selected: list[Path] = []
    selected_set: set[Path] = set()
    missing: list[str] = []

    for tok in tokens:
        matched: Path | None = None
        for label, path in available:
            if label == tok or (tok.isdigit() and label.isdigit() and int(label) == int(tok)):
                matched = path
                break
        if matched is None:
            missing.append(tok)
        elif matched not in selected_set:
            selected_set.add(matched)
            selected.append(matched)

    if missing:
        avail_labels = [label for label, _ in available]
        raise ValueError(
            f"Requested checkpoint(s) not found: {missing}. Available checkpoints: {avail_labels}"
        )
    return selected


def _scalar_from_item(item: dict[str, Any], key: str, default: Any) -> Any:
    val = item.get(key, default)
    if val is None:
        return default
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 1:
        return val[0]
    return val


def compute_chunk_error_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    action_is_pad: np.ndarray | None = None,
    action_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute masked chunk MSE/MAE, step-0 error, and per-dimension metrics.

    Args:
        predictions: Array of shape ``(N, chunk_size, action_dim)`` or ``(chunk_size, action_dim)``.
        targets: Array matching ``predictions`` shape.
        action_is_pad: Optional boolean array of shape ``(N, chunk_size)`` or ``(chunk_size,)``
            where ``True`` marks padded frames beyond episode end.
        action_names: Optional list of dimension names (e.g. ``["dx", "dy", "dz", "gripper"]``).
    """
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred[None, :, :]
        tgt = tgt[None, :, :]
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch between predictions {pred.shape} and targets {tgt.shape}.")
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N, T, D), got shape {pred.shape}.")

    n_samples, horizon, action_dim = pred.shape
    if action_is_pad is not None:
        pad_mask = np.asarray(action_is_pad, dtype=bool)
        if pad_mask.ndim == 1:
            pad_mask = pad_mask[None, :]
        if pad_mask.shape != (n_samples, horizon):
            raise ValueError(
                f"action_is_pad shape {pad_mask.shape} does not match (N={n_samples}, T={horizon})."
            )
        valid_mask = ~pad_mask
    else:
        valid_mask = np.ones((n_samples, horizon), dtype=bool)

    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        raise ValueError("All action steps are masked as padding; cannot compute metrics.")

    diff = pred - tgt
    abs_err = np.abs(diff)
    sq_err = diff**2

    # Mask padded timesteps out
    valid_3d = valid_mask[:, :, None]
    overall_mae = float((abs_err * valid_3d).sum() / (valid_count * action_dim))
    overall_mse = float((sq_err * valid_3d).sum() / (valid_count * action_dim))
    overall_rmse = float(np.sqrt(overall_mse))

    # Immediate step-0 metrics (always evaluate where step 0 is not padded)
    step0_valid = valid_mask[:, 0]
    if step0_valid.any():
        step0_abs = abs_err[step0_valid, 0, :]
        step0_sq = sq_err[step0_valid, 0, :]
        step0_mae = float(step0_abs.mean())
        step0_rmse = float(np.sqrt(step0_sq.mean()))
    else:
        step0_mae = float("nan")
        step0_rmse = float("nan")

    names = list(action_names) if action_names and len(action_names) == action_dim else [
        f"dim_{d}" for d in range(action_dim)
    ]
    per_dim: dict[str, dict[str, float]] = {}
    for d, name in enumerate(names):
        dim_mae = float((abs_err[:, :, d] * valid_mask).sum() / valid_count)
        dim_mse = float((sq_err[:, :, d] * valid_mask).sum() / valid_count)
        dim_step0_mae = float(abs_err[step0_valid, 0, d].mean()) if step0_valid.any() else float("nan")
        per_dim[name] = {
            "mae": dim_mae,
            "rmse": float(np.sqrt(dim_mse)),
            "step0_mae": dim_step0_mae,
        }

    return {
        "samples": n_samples,
        "horizon": horizon,
        "valid_steps": valid_count,
        "mae": overall_mae,
        "mse": overall_mse,
        "rmse": overall_rmse,
        "step0_mae": step0_mae,
        "step0_rmse": step0_rmse,
        "per_dim": per_dim,
    }


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch  # type: ignore

    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_checkpoint(
    ckpt_dir: Path,
    dataset_dir: Path,
    repo_id: str,
    video_backend: str,
    device: str,
    stride: int,
    max_samples: int | None,
    num_inference_steps: int | None,
    n_action_steps: int | None = None,
    return_predictions: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Load one checkpoint and evaluate it on the held-out LeRobotDataset."""
    import torch  # type: ignore
    from lerobot.configs.policies import PreTrainedConfig  # type: ignore
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # type: ignore
    from lerobot.policies.factory import make_policy, make_pre_post_processors  # type: ignore

    policy_cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    policy_cfg.pretrained_path = ckpt_dir
    policy_cfg.device = device
    if num_inference_steps is not None and hasattr(policy_cfg, "num_inference_steps"):
        policy_cfg.num_inference_steps = int(num_inference_steps)

    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_dir))
    chunk_size = int(getattr(policy_cfg, "chunk_size", 50))
    if n_action_steps is not None:
        if n_action_steps < 1 or n_action_steps > chunk_size:
            raise ValueError(
                f"--n-action-steps ({n_action_steps}) must be between 1 and chunk_size ({chunk_size})."
            )
        policy_cfg.n_action_steps = int(n_action_steps)
    eval_horizon = min(int(getattr(policy_cfg, "n_action_steps", chunk_size)), chunk_size)
    delta_timestamps = {"action": [i / ds_meta.fps for i in range(chunk_size)]}

    ds_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": str(dataset_dir),
        "delta_timestamps": delta_timestamps,
    }
    if video_backend != "auto":
        ds_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(**ds_kwargs)

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    indices = list(range(0, len(dataset), max(1, stride)))
    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]
    if not indices:
        raise ValueError(f"Dataset at {dataset_dir} yielded 0 samples to evaluate.")

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pads: list[np.ndarray] = []
    start_states: list[np.ndarray] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []

    with torch.inference_mode():
        for idx in indices:
            raw_item = dict(dataset[idx])
            gt_action = np.asarray(raw_item.pop("action"), dtype=np.float32)
            gt_pad = raw_item.pop("action_is_pad", None)
            if gt_pad is not None:
                gt_pad_arr = np.asarray(gt_pad, dtype=bool)
            else:
                gt_pad_arr = np.zeros((gt_action.shape[0],), dtype=bool)

            if return_predictions:
                state_val = raw_item.get("observation.state")
                if state_val is not None:
                    if hasattr(state_val, "detach"):
                        state_np = state_val.detach().cpu().numpy().astype(np.float32)
                    else:
                        state_np = np.asarray(state_val, dtype=np.float32)
                else:
                    state_np = np.zeros((0,), dtype=np.float32)
                start_states.append(state_np)
                episode_indices.append(int(_scalar_from_item(raw_item, "episode_index", 0)))
                frame_indices.append(int(_scalar_from_item(raw_item, "frame_index", idx)))
                timestamps.append(float(_scalar_from_item(raw_item, "timestamp", idx / float(ds_meta.fps))))

            processed = preprocessor(raw_item)
            pred_norm = policy.predict_action_chunk(processed)
            pred_unnorm = postprocessor(pred_norm)
            pred_np = pred_unnorm.squeeze(0).detach().cpu().numpy()

            # Slice to the evaluated execution horizon (e.g. n_action_steps=16) and action dim
            horizon = min(eval_horizon, gt_action.shape[0])
            pred_np = pred_np[:horizon, : gt_action.shape[1]]
            gt_action = gt_action[:horizon]
            gt_pad_arr = gt_pad_arr[:horizon]
            preds.append(pred_np)
            targets.append(gt_action)
            pads.append(gt_pad_arr)

    action_names = ds_meta.features.get("action", {}).get("names")
    pred_stack = np.stack(preds, axis=0)
    tgt_stack = np.stack(targets, axis=0)
    pad_stack = np.stack(pads, axis=0)
    metrics = compute_chunk_error_metrics(
        predictions=pred_stack,
        targets=tgt_stack,
        action_is_pad=pad_stack,
        action_names=action_names,
    )
    metrics["checkpoint"] = str(ckpt_dir)
    metrics["label"] = _checkpoint_label(ckpt_dir)
    if not return_predictions:
        return metrics

    state_names = ds_meta.features.get("observation.state", {}).get("names")
    pred_payload = {
        "fps": int(ds_meta.fps),
        "state_names": state_names,
        "action_names": action_names,
        "windows": {
            "sample_indices": [int(i) for i in indices],
            "episode_indices": episode_indices,
            "frame_indices": frame_indices,
            "timestamps": [round(t, 6) for t in timestamps],
            "start_states": np.round(np.stack(start_states, axis=0), 6).tolist(),
            "gt_actions": np.round(tgt_stack, 6).tolist(),
            "pred_actions": np.round(pred_stack, 6).tolist(),
            "action_is_pad": pad_stack.tolist(),
        },
    }
    return metrics, pred_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved π₀ checkpoints offline on a held-out LeRobot v3.0 dataset."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a pretrained_model dir, a checkpoints/ dir, or a training output dir.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to local held-out LeRobot v3.0 dataset (e.g., ../lerobot_pi0_test).",
    )
    parser.add_argument("--repo-id", default="local/ur5e_titration", help="Dataset repo_id metadata string.")
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec", "auto"),
        default="pyav",
        help="Video decoding backend (default: pyav).",
    )
    parser.add_argument("--device", default="auto", help="Device ('auto', 'cuda', or 'cpu').")
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Evaluate every Nth frame across the dataset (default: 10).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of sampled frames to evaluate.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional override for flow-matching denoising steps.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override evaluated execution horizon (e.g. 16) without modifying checkpoint config.json (defaults to checkpoint's n_action_steps).",
    )
    parser.add_argument(
        "--only-checkpoints",
        type=str,
        default=None,
        help="Optional comma-separated checkpoint labels to evaluate (e.g. '002000,010000,016000,020000').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary JSON.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Optional JSON file path to save per-window ground-truth states/actions and predicted chunks for plotting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checkpoints = filter_checkpoints(discover_checkpoints(args.checkpoint), args.only_checkpoints)
    except Exception as exc:
        print(f"[Checkpoint Error] {exc}", file=sys.stderr)
        return 1

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not (dataset_dir / "meta" / "info.json").is_file():
        print(f"[Dataset Error] Missing meta/info.json under {dataset_dir}", file=sys.stderr)
        return 1

    device = _resolve_device(args.device)
    print(f"[Eval] Device={device}, Checkpoints={len(checkpoints)}, Dataset={dataset_dir}")

    results: list[dict[str, Any]] = []
    prediction_entries: list[dict[str, Any]] = []
    shared_meta: dict[str, Any] = {}
    want_preds = bool(args.save_predictions)

    for ckpt in checkpoints:
        print(f"\n--- Evaluating {_checkpoint_label(ckpt)} ({ckpt}) ---")
        eval_out = evaluate_checkpoint(
            ckpt_dir=ckpt,
            dataset_dir=dataset_dir,
            repo_id=args.repo_id,
            video_backend=args.video_backend,
            device=device,
            stride=args.stride,
            max_samples=args.max_samples,
            num_inference_steps=args.num_inference_steps,
            n_action_steps=args.n_action_steps,
            return_predictions=want_preds,
        )
        if want_preds:
            metrics, pred_payload = eval_out  # type: ignore[misc]
            shared_meta = {
                "fps": pred_payload["fps"],
                "state_names": pred_payload["state_names"],
                "action_names": pred_payload["action_names"],
            }
            prediction_entries.append({
                "label": metrics["label"],
                "checkpoint": metrics["checkpoint"],
                "metrics": metrics,
                "windows": pred_payload["windows"],
            })
        else:
            metrics = eval_out  # type: ignore[assignment]
        results.append(metrics)
        dim_str = ", ".join(
            f"{name}_mae={vals['mae']:.5f}" for name, vals in metrics["per_dim"].items()
        )
        print(
            f"  samples={metrics['samples']}, horizon={metrics['horizon']}, "
            f"chunk_mae={metrics['mae']:.5f}, chunk_rmse={metrics['rmse']:.5f}, "
            f"step0_mae={metrics['step0_mae']:.5f}"
        )
        print(f"  per-dim: {dim_str}")

    best = min(results, key=lambda r: r["mae"])
    print("\n=== Summary ===")
    for r in results:
        marker = "  <-- BEST" if r["checkpoint"] == best["checkpoint"] else ""
        print(
            f"  {r['label']:>10s} | mae={r['mae']:.5f} | rmse={r['rmse']:.5f} | "
            f"step0_mae={r['step0_mae']:.5f}{marker}"
        )

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results, "best": best["checkpoint"]}, indent=2), encoding="utf-8")
        print(f"\n[Saved] {out_path}")

    if args.save_predictions:
        preds_path = Path(args.save_predictions).expanduser().resolve()
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_doc = {
            "dataset_dir": str(dataset_dir),
            "stride": int(args.stride),
            "horizon": int(results[0]["horizon"]) if results else None,
            "fps": shared_meta.get("fps"),
            "state_names": shared_meta.get("state_names"),
            "action_names": shared_meta.get("action_names"),
            "checkpoints": prediction_entries,
        }
        preds_path.write_text(json.dumps(preds_doc, indent=2), encoding="utf-8")
        print(f"[Saved predictions] {preds_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
