#!/usr/bin/env python3
"""Formal fixed-budget full-head fusion evaluation with target calibration.

The target cohort is split once by patient into a 60% labeled pool and a 40%
fixed query set.  Within every repeated draw, adaptation and calibration
patients are disjoint.  The complete nonlinear mortality head is fitted on an
exact-positive adaptation support; intercept-only and Platt calibration are
then fitted on a separate exact-positive calibration support.  Query labels
are never used for fitting.

Uncertainty is summarized with a paired two-source bootstrap: each bootstrap
draw selects one of the repeated support fits and independently resamples
query patients as clusters.  Its percentile interval therefore includes both
support-draw variability and query-patient sampling variability; it is wider
than a patient-only interval conditional on one fitted support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

METHOD_FROZEN = "frozen"
METHOD_TARGET = "full_head_target"
METHOD_FUSION = "full_head_fusion"
METHOD_FUSION_INTERCEPT = "fusion_intercept_only"
METHOD_FUSION_PLATT = "fusion_platt"
FORMAL_METHODS = (
    METHOD_FROZEN,
    METHOD_TARGET,
    METHOD_FUSION,
    METHOD_FUSION_INTERCEPT,
    METHOD_FUSION_PLATT,
)
METRICS = ("roc_auc", "average_precision", "brier")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _run("git", "status", "--short")
    return {
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "git_branch": _run("git", "branch", "--show-current"),
        "git_worktree_dirty": bool(status) if status is not None else None,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        if temporary.stat().st_size <= 0:
            raise OSError(f"temporary output is empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Dict[str, Any], path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            _json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False
        )
        handle.write("\n")
    try:
        if temporary.stat().st_size <= 0:
            raise OSError(f"temporary output is empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        if temporary.stat().st_size <= 0:
            raise OSError(f"temporary output is empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _filter_partition(
    frame: pd.DataFrame, value: str | None, *, label: str
) -> pd.DataFrame:
    if value is None or value == "":
        return frame
    if "site_partition" not in frame.columns:
        raise SystemExit(
            f"{label} cache has no site_partition required for {value!r}"
        )
    selected = frame.loc[frame["site_partition"].astype(str) == value].copy()
    if selected.empty:
        available = sorted(frame["site_partition"].astype(str).unique())
        raise SystemExit(
            f"{label} cache has no site_partition={value!r}; available={available}"
        )
    return selected


def _filter_group(
    frame: pd.DataFrame,
    column: str | None,
    value: str | None,
    *,
    label: str,
) -> pd.DataFrame:
    """Select one target group while tolerating numeric/string CSV dtypes."""

    if column is None and value is None:
        return frame
    if not column or value is None:
        raise SystemExit(
            "--target-group-column and --target-group-value must be supplied together"
        )
    if column not in frame.columns:
        raise SystemExit(f"{label} cache has no group column {column!r}")
    mask = frame[column].astype(str) == str(value)
    if not mask.any():
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        numeric_column = pd.to_numeric(frame[column], errors="coerce")
        if pd.notna(numeric_value):
            mask = numeric_column == float(numeric_value)
    selected = frame.loc[mask].copy()
    if selected.empty:
        available = sorted(frame[column].dropna().astype(str).unique())
        raise SystemExit(
            f"{label} cache has no {column}={value!r}; available={available}"
        )
    return selected


def disjoint_exact_positive_samples(
    patient_positive: Mapping[str, int],
    pool: Sequence[str],
    *,
    adaptation_positive: int,
    calibration_positive: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw disjoint adaptation/calibration patients at natural prevalence."""

    if adaptation_positive < 1 or calibration_positive < 1:
        raise ValueError("adaptation and calibration positive budgets must be >=1")
    positives = np.asarray(
        [str(patient) for patient in pool if patient_positive[str(patient)] == 1],
        dtype=str,
    )
    negatives = np.asarray(
        [str(patient) for patient in pool if patient_positive[str(patient)] == 0],
        dtype=str,
    )
    total_positive = adaptation_positive + calibration_positive
    if total_positive > len(positives):
        raise ValueError(
            f"combined positive budget {total_positive} exceeds pool {len(positives)}"
        )
    negative_per_positive = len(negatives) / len(positives)
    adaptation_negative = max(1, int(round(adaptation_positive * negative_per_positive)))
    calibration_negative = max(
        1, int(round(calibration_positive * negative_per_positive))
    )
    if adaptation_negative + calibration_negative > len(negatives):
        raise ValueError("combined negative requirement exceeds adaptation pool")

    positive_order = rng.permutation(positives)
    negative_order = rng.permutation(negatives)
    adaptation = np.concatenate(
        [
            positive_order[:adaptation_positive],
            negative_order[:adaptation_negative],
        ]
    )
    calibration = np.concatenate(
        [
            positive_order[adaptation_positive:total_positive],
            negative_order[
                adaptation_negative : adaptation_negative + calibration_negative
            ],
        ]
    )
    if set(adaptation) & set(calibration):
        raise AssertionError("adaptation/calibration patient leakage")
    return adaptation, calibration


def fixed_calibration_nested_samples(
    patient_positive: Mapping[str, int],
    pool: Sequence[str],
    *,
    adaptation_positive: int,
    calibration_positive: int,
    calibration_anchor_adaptation_positive: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw nested K supports while preserving the K=anchor calibration draw.

    The random patient order is identical for every K.  Patients for the
    anchor adaptation support come first, followed by the fixed calibration
    support.  If K exceeds the anchor (MIMIC K=500), adaptation continues
    after the calibration block.  At K=anchor this is exactly the historical
    ``disjoint_exact_positive_samples`` draw, so existing K=200 results remain
    reusable.
    """

    anchor = int(calibration_anchor_adaptation_positive)
    requested = int(adaptation_positive)
    if requested < 1 or calibration_positive < 1 or anchor < 1:
        raise ValueError("adaptation, calibration, and anchor budgets must be >=1")
    positives = np.asarray(
        [str(patient) for patient in pool if patient_positive[str(patient)] == 1],
        dtype=str,
    )
    negatives = np.asarray(
        [str(patient) for patient in pool if patient_positive[str(patient)] == 0],
        dtype=str,
    )
    positive_required = max(anchor, requested) + calibration_positive
    if positive_required > len(positives):
        raise ValueError(
            f"nested positive requirement {positive_required} exceeds pool "
            f"{len(positives)}"
        )
    negative_per_positive = len(negatives) / len(positives)

    def negative_count(positive_count: int) -> int:
        return max(1, int(round(positive_count * negative_per_positive)))

    anchor_negative = negative_count(anchor)
    requested_negative = negative_count(requested)
    calibration_negative = negative_count(calibration_positive)
    negative_required = max(anchor_negative, requested_negative) + calibration_negative
    if negative_required > len(negatives):
        raise ValueError(
            f"nested negative requirement {negative_required} exceeds pool "
            f"{len(negatives)}"
        )

    positive_order = rng.permutation(positives)
    negative_order = rng.permutation(negatives)
    anchor_positive_block = positive_order[:anchor]
    calibration_positive_block = positive_order[
        anchor : anchor + calibration_positive
    ]
    anchor_negative_block = negative_order[:anchor_negative]
    calibration_negative_block = negative_order[
        anchor_negative : anchor_negative + calibration_negative
    ]
    if requested <= anchor:
        adaptation_positive_block = anchor_positive_block[:requested]
        adaptation_negative_block = anchor_negative_block[:requested_negative]
    else:
        extra_positive = positive_order[
            anchor + calibration_positive :
            anchor + calibration_positive + (requested - anchor)
        ]
        adaptation_positive_block = np.concatenate(
            [anchor_positive_block, extra_positive]
        )
        extra_negative = negative_order[
            anchor_negative + calibration_negative :
            anchor_negative + calibration_negative
            + (requested_negative - anchor_negative)
        ]
        adaptation_negative_block = np.concatenate(
            [anchor_negative_block, extra_negative]
        )
    adaptation = np.concatenate(
        [adaptation_positive_block, adaptation_negative_block]
    )
    calibration = np.concatenate(
        [calibration_positive_block, calibration_negative_block]
    )
    if set(adaptation) & set(calibration):
        raise AssertionError("nested adaptation/calibration patient leakage")
    return adaptation, calibration


@dataclass(frozen=True)
class OrderedBinaryMetrics:
    """Pre-sorted binary predictions for fast weighted cluster bootstrap."""

    y: np.ndarray
    p: np.ndarray
    order: np.ndarray
    group_starts: np.ndarray

    @classmethod
    def build(cls, y: np.ndarray, p: np.ndarray) -> "OrderedBinaryMetrics":
        labels = np.asarray(y, dtype=np.int8)
        probability = np.asarray(p, dtype=float)
        if len(labels) != len(probability) or not len(labels):
            raise ValueError("labels and probabilities must align and be non-empty")
        order = np.argsort(-probability, kind="stable")
        ordered_probability = probability[order]
        starts = np.r_[
            0,
            np.flatnonzero(ordered_probability[1:] != ordered_probability[:-1]) + 1,
        ]
        return cls(labels, probability, order, starts.astype(np.int64))

    def evaluate(self, sample_weight: np.ndarray) -> Dict[str, float]:
        weight = np.asarray(sample_weight, dtype=float)
        if len(weight) != len(self.y):
            raise ValueError("bootstrap weights and prediction rows must align")
        ordered_y = self.y[self.order]
        ordered_weight = weight[self.order]
        positive = ordered_weight * ordered_y
        negative = ordered_weight * (1 - ordered_y)
        group_positive = np.add.reduceat(positive, self.group_starts)
        group_negative = np.add.reduceat(negative, self.group_starts)
        total_positive = float(group_positive.sum())
        total_negative = float(group_negative.sum())
        if total_positive <= 0.0 or total_negative <= 0.0:
            return {metric: np.nan for metric in METRICS}

        cumulative_positive = np.cumsum(group_positive)
        cumulative_negative = np.cumsum(group_negative)
        negative_below = total_negative - cumulative_negative
        auc_numerator = np.sum(
            group_positive * (negative_below + 0.5 * group_negative)
        )
        roc_auc = float(auc_numerator / (total_positive * total_negative))
        precision_denominator = cumulative_positive + cumulative_negative
        precision = np.divide(
            cumulative_positive,
            precision_denominator,
            out=np.zeros_like(cumulative_positive),
            where=precision_denominator > 0.0,
        )
        average_precision = float(
            np.sum((group_positive / total_positive) * precision)
        )
        brier = float(
            np.sum(weight * np.square(self.p - self.y)) / np.sum(weight)
        )
        return {
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "brier": brier,
        }


def paired_joint_bootstrap(
    y: np.ndarray,
    patient_ids: np.ndarray,
    predictions: Mapping[int, Mapping[str, np.ndarray]],
    *,
    n_bootstrap: int,
    random_seed: int,
    methods: Sequence[str] = FORMAL_METHODS,
    reference_method: str = METHOD_FROZEN,
) -> pd.DataFrame:
    """Jointly sample one support fit and query patients for paired metrics."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    replicate_ids = sorted(predictions)
    if not replicate_ids:
        raise ValueError("no repeated predictions supplied")
    method_order = tuple(methods)
    if not method_order or reference_method not in method_order:
        raise ValueError("methods must include the reference method")
    for replicate in replicate_ids:
        if set(predictions[replicate]) != set(method_order):
            raise ValueError(f"replicate {replicate} has incomplete methods")

    patients, inverse = np.unique(
        np.asarray(patient_ids, dtype=str), return_inverse=True
    )
    evaluators = {
        replicate: {
            method: OrderedBinaryMetrics.build(y, probability)
            for method, probability in methods.items()
        }
        for replicate, methods in predictions.items()
    }
    rng = np.random.default_rng(random_seed)
    rows = []
    for bootstrap in range(n_bootstrap):
        selected_replicate = int(rng.choice(replicate_ids))
        sampled_patient_indices = rng.integers(
            0, len(patients), size=len(patients)
        )
        patient_counts = np.bincount(
            sampled_patient_indices, minlength=len(patients)
        ).astype(float)
        row_weight = patient_counts[inverse]
        method_metrics = {
            method: evaluator.evaluate(row_weight)
            for method, evaluator in evaluators[selected_replicate].items()
        }
        reference = method_metrics[reference_method]
        for method in method_order:
            metrics = method_metrics[method]
            rows.append(
                {
                    "bootstrap": int(bootstrap),
                    "support_replicate": selected_replicate,
                    "method": method,
                    **metrics,
                    **{
                        f"delta_{metric}": float(
                            metrics[metric] - reference[metric]
                        )
                        for metric in METRICS
                    },
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_bootstrap(
    replicate_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    alpha: float,
    methods: Sequence[str] = FORMAL_METHODS,
    reference_method: str = METHOD_FROZEN,
) -> pd.DataFrame:
    """Combine mean repeated-support point estimates with bootstrap intervals."""

    rows = []
    method_order = tuple(methods)
    if not method_order or reference_method not in method_order:
        raise ValueError("methods must include the reference method")
    reference_points = replicate_metrics.loc[
        replicate_metrics["method"] == reference_method
    ]
    for method in method_order:
        points = replicate_metrics.loc[replicate_metrics["method"] == method]
        boot = bootstrap.loc[bootstrap["method"] == method]
        for metric in METRICS:
            point = float(points[metric].mean())
            reference_point = float(reference_points[metric].mean())
            values = boot[metric].dropna().to_numpy(dtype=float)
            deltas = boot[f"delta_{metric}"].dropna().to_numpy(dtype=float)
            lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
            delta_lo, delta_hi = np.quantile(
                deltas, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "point_estimate_mean_over_support_repeats": point,
                    "support_repeat_std": float(points[metric].std(ddof=1)),
                    "joint_bootstrap_percentile_lo": float(lo),
                    "joint_bootstrap_percentile_hi": float(hi),
                    "delta_vs_frozen": float(point - reference_point),
                    "delta_joint_bootstrap_percentile_lo": float(delta_lo),
                    "delta_joint_bootstrap_percentile_hi": float(delta_hi),
                    "n_support_repeats": int(points["replicate"].nunique()),
                    "n_bootstrap": int(boot["bootstrap"].nunique()),
                }
            )
    return pd.DataFrame.from_records(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-head-features", type=Path, required=True)
    parser.add_argument("--source-head-features", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--source-site-partition", default="development")
    parser.add_argument(
        "--target-site-partition",
        default=None,
        help="Optional site_partition value required in the target cache.",
    )
    parser.add_argument(
        "--target-group-column",
        default=None,
        help="Optional target group column, for example hospital_number.",
    )
    parser.add_argument(
        "--target-group-value",
        default=None,
        help="One value from --target-group-column to evaluate.",
    )
    parser.add_argument("--adaptation-positive", type=int, default=200)
    parser.add_argument("--calibration-positive", type=int, default=50)
    parser.add_argument(
        "--calibration-anchor-adaptation-positive",
        type=int,
        default=None,
        help=(
            "Keep calibration draws fixed across K by reserving this many "
            "adaptation-positive patients before the calibration block. "
            "Use 200 to remain identical to the existing formal K=200 run."
        ),
    )
    parser.add_argument("--target-weight", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--evaluation-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-source", type=float, default=0.1)
    parser.add_argument("--calibration-bins", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", type=int, default=None, metavar="IDX")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    for path, label in (
        (args.target_head_features, "target head-feature parquet"),
        (args.source_head_features, "source head-feature parquet"),
        (args.bundle, "source bundle"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"{label} is missing or empty: {path}")
    if args.repeats < 1 or args.bootstrap < 1:
        raise SystemExit("--repeats and --bootstrap must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be in (0, 1)")

    output_names = (
        "replicate_metrics.csv",
        "calibration_parameters.csv",
        "bootstrap_replicates.csv",
        "bootstrap_summary.csv",
        "query_predictions.parquet",
        "query_predictions.parquet.metadata.json",
        "formal_report.json",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path for name in output_names if (path := args.output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to overwrite outputs; use a new directory or --overwrite:\n  "
            + "\n  ".join(str(path) for path in existing)
        )

    import torch

    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from patient_data.mortality.full_head_fusion import (
        _full_head_probabilities,
        train_tethered_full_head,
    )
    from patient_data.mortality.small_sample_adaptation import (
        METHOD_INTERCEPT,
        METHOD_PLATT,
        _logit,
        _sigmoid,
        adaptation_metrics,
        patient_partition,
        prepare_head_feature_frame,
        source_head_probabilities,
    )
    from patient_data.mortality.transfer_calibration import (
        fit_intercept_only,
        fit_platt,
    )
    from patient_data.mortality.strats_model import STraTS, STraTSConfig

    if torch.cuda.is_available() and args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    bundle_hash = _sha256(args.bundle)
    input_metadata: Dict[str, Any] = {}
    for path, label in (
        (args.target_head_features, "target"),
        (args.source_head_features, "source"),
    ):
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        metadata = None
        if sidecar.is_file():
            with sidecar.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            recorded_hash = metadata.get("bundle_sha256")
            if recorded_hash and recorded_hash != bundle_hash:
                raise SystemExit(
                    f"{label} cache was produced by a different bundle: "
                    f"{recorded_hash} != {bundle_hash}"
                )
        input_metadata[label] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "sidecar": metadata,
        }

    target, target_columns = prepare_head_feature_frame(
        pd.read_parquet(args.target_head_features), horizon_days=args.horizon_days
    )
    source, source_columns = prepare_head_feature_frame(
        pd.read_parquet(args.source_head_features), horizon_days=args.horizon_days
    )
    source = _filter_partition(
        source, args.source_site_partition, label="source"
    )
    target = _filter_partition(
        target, args.target_site_partition, label="target"
    )
    target = _filter_group(
        target,
        args.target_group_column,
        args.target_group_value,
        label="target",
    )
    if target_columns != source_columns:
        raise SystemExit("source and target feature columns differ")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    model = STraTS(STraTSConfig(**bundle["cfg"]))
    model.load_state_dict(bundle["model_state"])
    if model.head is None:
        raise SystemExit("source bundle has no mortality head")
    expected_features = int(model.cfg.d_model + model.cfg.n_static)
    if len(target_columns) != expected_features:
        raise SystemExit(
            f"feature dimension {len(target_columns)} != expected {expected_features}"
        )

    target_ids = target["patient_id_hash"].to_numpy(dtype=str)
    target_y = target["y_true"].to_numpy(dtype=int)
    target_p = target["y_prob"].to_numpy(dtype=float)
    target_x = target[target_columns].to_numpy(dtype=np.float32)
    source_ids = source["patient_id_hash"].to_numpy(dtype=str)
    source_y = source["y_true"].to_numpy(dtype=int)
    source_p = source["y_prob"].to_numpy(dtype=float)
    source_x = source[source_columns].to_numpy(dtype=np.float32)
    partition = patient_partition(
        target,
        evaluation_fraction=args.evaluation_fraction,
        random_seed=args.seed,
    )
    evaluation_mask = np.isin(target_ids, partition.evaluation)
    y_eval = target_y[evaluation_mask]
    p_eval = target_p[evaluation_mask]
    x_eval = target_x[evaluation_mask]
    ids_eval = target_ids[evaluation_mask]

    print(
        f"[formal_fusion] target_rows={len(target):,} "
        f"target_patients={target['patient_id_hash'].nunique():,} "
        f"pool_positive={sum(partition.patient_positive[p] for p in partition.pool):,} "
        f"query_patients={len(partition.evaluation):,} "
        f"query_rows={len(y_eval):,} source_patients="
        f"{source['patient_id_hash'].nunique():,} device={device}",
        flush=True,
    )

    target_check = source_head_probabilities(
        model.head,
        target_x,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    source_check = source_head_probabilities(
        model.head,
        source_x,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    target_probability_error = float(np.max(np.abs(target_check - target_p)))
    source_probability_error = float(np.max(np.abs(source_check - source_p)))
    if max(target_probability_error, source_probability_error) > 2e-5:
        raise SystemExit(
            "feature cache and source head mismatch: "
            f"target={target_probability_error:.6g}, "
            f"source={source_probability_error:.6g}"
        )

    rng = np.random.default_rng(args.seed + 1)
    replicate_rows = []
    calibration_rows = []
    predictions: Dict[int, Dict[str, np.ndarray]] = {}
    prediction_frame = target.loc[
        evaluation_mask,
        ["cohort", "horizon_days", "patient_id_hash", "t_pred", "y_true"],
    ].reset_index(drop=True)
    prediction_frame["frozen_probability"] = p_eval.astype(np.float32)
    base_metrics = adaptation_metrics(
        y_eval, p_eval, n_bins=args.calibration_bins
    )

    for replicate in range(args.repeats):
        if args.calibration_anchor_adaptation_positive is None:
            adaptation_patients, calibration_patients = disjoint_exact_positive_samples(
                partition.patient_positive,
                partition.pool,
                adaptation_positive=args.adaptation_positive,
                calibration_positive=args.calibration_positive,
                rng=rng,
            )
        else:
            adaptation_patients, calibration_patients = fixed_calibration_nested_samples(
                partition.patient_positive,
                partition.pool,
                adaptation_positive=args.adaptation_positive,
                calibration_positive=args.calibration_positive,
                calibration_anchor_adaptation_positive=(
                    args.calibration_anchor_adaptation_positive
                ),
                rng=rng,
            )
        adaptation_mask = np.isin(target_ids, adaptation_patients)
        calibration_mask = np.isin(target_ids, calibration_patients)
        if np.any(adaptation_mask & calibration_mask):
            raise AssertionError("adaptation/calibration row leakage")
        if np.any((adaptation_mask | calibration_mask) & evaluation_mask):
            raise AssertionError("labeled support/query patient leakage")
        y_adaptation = target_y[adaptation_mask]
        ids_adaptation = target_ids[adaptation_mask]
        y_calibration = target_y[calibration_mask]
        common = {
            "replicate": int(replicate),
            "adaptation_n_patients": int(len(adaptation_patients)),
            "adaptation_n_positive_patients": int(args.adaptation_positive),
            "adaptation_n_samples": int(adaptation_mask.sum()),
            "calibration_n_patients": int(len(calibration_patients)),
            "calibration_n_positive_patients": int(args.calibration_positive),
            "calibration_n_samples": int(calibration_mask.sum()),
            "evaluation_n_patients": int(len(partition.evaluation)),
            "evaluation_n_samples": int(len(y_eval)),
        }
        print(
            f"[formal_fusion] replicate={replicate + 1}/{args.repeats} "
            f"adaptation_patients={len(adaptation_patients):,} "
            f"calibration_patients={len(calibration_patients):,} starting",
            flush=True,
        )

        target_head, target_diagnostics = train_tethered_full_head(
            target_x[adaptation_mask],
            y_adaptation,
            ids_adaptation,
            source_head=model.head,
            target_weight=1.0,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_source=args.l2_source,
            random_seed=(
                args.seed
                + 100_000 * replicate
                + 10 * args.adaptation_positive
            ),
        )
        target_probability = _full_head_probabilities(
            target_head,
            x_eval,
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        fusion_head, fusion_diagnostics = train_tethered_full_head(
            target_x[adaptation_mask],
            y_adaptation,
            ids_adaptation,
            source_head=model.head,
            source_features=source_x,
            source_y=source_y,
            source_patient_ids=source_ids,
            target_weight=args.target_weight,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_source=args.l2_source,
            random_seed=(
                args.seed
                + 100_000 * replicate
                + 10 * args.adaptation_positive
                + 1
            ),
        )
        fusion_probability = _full_head_probabilities(
            fusion_head,
            x_eval,
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        fusion_calibration_probability = _full_head_probabilities(
            fusion_head,
            target_x[calibration_mask],
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        calibrated_predictions: Dict[str, np.ndarray] = {}
        fusion_logits_eval = _logit(fusion_probability)
        for calibration_method, output_method in (
            (METHOD_INTERCEPT, METHOD_FUSION_INTERCEPT),
            (METHOD_PLATT, METHOD_FUSION_PLATT),
        ):
            fitted_calibration = (
                fit_intercept_only(
                    y_calibration, fusion_calibration_probability
                )
                if calibration_method == METHOD_INTERCEPT
                else fit_platt(
                    y_calibration, fusion_calibration_probability
                )
            )
            intercept = float(fitted_calibration.intercept)
            slope = float(fitted_calibration.slope)
            calibrated_predictions[output_method] = _sigmoid(
                intercept + slope * fusion_logits_eval
            )
            calibration_rows.append(
                {
                    **common,
                    "method": output_method,
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "fit_strategy": fitted_calibration.fit_strategy,
                    "l2_penalty": float(fitted_calibration.l2_penalty),
                    "optimizer_iterations": int(
                        fitted_calibration.optimizer_iterations
                    ),
                }
            )

        repeat_predictions = {
            METHOD_FROZEN: p_eval.copy(),
            METHOD_TARGET: target_probability,
            METHOD_FUSION: fusion_probability,
            **calibrated_predictions,
        }
        predictions[replicate] = repeat_predictions
        for method, probability in repeat_predictions.items():
            diagnostics: Mapping[str, Any]
            if method == METHOD_TARGET:
                diagnostics = target_diagnostics
            elif method in {
                METHOD_FUSION,
                METHOD_FUSION_INTERCEPT,
                METHOD_FUSION_PLATT,
            }:
                diagnostics = fusion_diagnostics
            else:
                diagnostics = {
                    "fit_target_loss": np.nan,
                    "fit_source_loss": np.nan,
                    "fit_objective": np.nan,
                    "parameter_delta_l2": 0.0,
                    "parameter_delta_max_abs": 0.0,
                    "trainable_parameters": 0,
                }
            replicate_rows.append(
                {
                    **common,
                    "method": method,
                    **diagnostics,
                    **adaptation_metrics(
                        y_eval, probability, n_bins=args.calibration_bins
                    ),
                }
            )
        for method, probability in repeat_predictions.items():
            if method == METHOD_FROZEN:
                continue
            prediction_frame[
                f"{method}_r{replicate:02d}_probability"
            ] = probability.astype(np.float32)
        print(
            f"[formal_fusion] replicate={replicate + 1}/{args.repeats} complete "
            f"fusion_auc={replicate_rows[-3]['roc_auc']:.4f}",
            flush=True,
        )

    replicate_metrics = pd.DataFrame.from_records(replicate_rows)
    calibration_parameters = pd.DataFrame.from_records(calibration_rows)
    print(
        f"[formal_fusion] starting {args.bootstrap:,} paired joint bootstrap draws",
        flush=True,
    )
    bootstrap = paired_joint_bootstrap(
        y_eval,
        ids_eval,
        predictions,
        n_bootstrap=args.bootstrap,
        random_seed=args.bootstrap_seed,
    )
    bootstrap_summary = summarize_bootstrap(
        replicate_metrics, bootstrap, alpha=args.alpha
    )

    predictions_path = args.output_dir / "query_predictions.parquet"
    _atomic_parquet(prediction_frame, predictions_path)
    prediction_metadata = {
        "schema_version": "formal_full_head_fusion_predictions_v1",
        "n_rows": int(len(prediction_frame)),
        "n_patients": int(prediction_frame["patient_id_hash"].nunique()),
        "n_positive_rows": int(prediction_frame["y_true"].sum()),
        "columns": list(prediction_frame.columns),
        "identifier_policy": "pseudonymized hash copied from input cache",
        "raw_identifiers_persisted": False,
        "file_sha256": _sha256(predictions_path),
        "file_size_bytes": int(predictions_path.stat().st_size),
    }
    _atomic_json(
        prediction_metadata,
        args.output_dir / "query_predictions.parquet.metadata.json",
    )
    _atomic_csv(replicate_metrics, args.output_dir / "replicate_metrics.csv")
    _atomic_csv(
        calibration_parameters,
        args.output_dir / "calibration_parameters.csv",
    )
    _atomic_csv(bootstrap, args.output_dir / "bootstrap_replicates.csv")
    _atomic_csv(bootstrap_summary, args.output_dir / "bootstrap_summary.csv")

    report = {
        "schema_version": "formal_full_head_fusion_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "horizon_days": int(args.horizon_days),
        "input": {
            **input_metadata,
            "bundle_path": str(args.bundle),
            "bundle_sha256": bundle_hash,
        },
        **_git_metadata(),
        "design": (
            "fixed patient query; repeated disjoint exact-positive adaptation "
            "and calibration supports"
        ),
        "split_unit": "patient_id_hash",
        "query_estimand": "prediction_day",
        "encoder_policy": "frozen",
        "adapted_parameters": int(
            sum(parameter.numel() for parameter in model.head.parameters())
        ),
        "source_site_partition": args.source_site_partition or None,
        "target_site_partition": args.target_site_partition or None,
        "target_group_column": args.target_group_column,
        "target_group_value": args.target_group_value,
        "source_n_patients": int(source["patient_id_hash"].nunique()),
        "source_n_samples": int(len(source)),
        "target_n_patients": int(target["patient_id_hash"].nunique()),
        "target_n_samples": int(len(target)),
        "pool_n_patients": int(len(partition.pool)),
        "pool_n_positive_patients": int(
            sum(partition.patient_positive[p] for p in partition.pool)
        ),
        "query_n_patients": int(len(partition.evaluation)),
        "query_n_positive_patients": int(
            sum(partition.patient_positive[p] for p in partition.evaluation)
        ),
        "query_n_samples": int(len(y_eval)),
        "adaptation_positive_patients": int(args.adaptation_positive),
        "calibration_positive_patients": int(args.calibration_positive),
        "calibration_anchor_adaptation_positive_patients": (
            int(args.calibration_anchor_adaptation_positive)
            if args.calibration_anchor_adaptation_positive is not None
            else int(args.adaptation_positive)
        ),
        "calibration_support_policy": (
            "fixed_across_k_and_identical_to_anchor_formal_draw"
            if args.calibration_anchor_adaptation_positive is not None
            else "specific_to_this_adaptation_budget"
        ),
        "platt_fit_policy": (
            "historical unpenalized Newton; standardized unpenalized L-BFGS-B "
            "retry; weak slope-only L2=0.01 fallback for nonconvergence or "
            "near separation"
        ),
        "platt_weak_l2_fallback_count": int(
            (
                calibration_parameters["fit_strategy"]
                == "weak_l2_lbfgsb_fallback"
            ).sum()
        ),
        "total_labeled_positive_patients_per_repeat": int(
            args.adaptation_positive + args.calibration_positive
        ),
        "target_weight": float(args.target_weight),
        "repeats": int(args.repeats),
        "evaluation_fraction": float(args.evaluation_fraction),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "l2_source": float(args.l2_source),
        "bootstrap": {
            "n": int(args.bootstrap),
            "seed": int(args.bootstrap_seed),
            "alpha": float(args.alpha),
            "unit": "patient cluster",
            "support_uncertainty": "one repeated support fit sampled per draw",
            "interval_interpretation": (
                "joint support-draw and query-patient percentile interval; "
                "not conditional on one fixed support"
            ),
        },
        "target_probability_max_abs_error": target_probability_error,
        "source_probability_max_abs_error": source_probability_error,
        "methods": list(FORMAL_METHODS),
        "base_metrics": base_metrics,
        "elapsed_seconds": float(time.time() - started),
        "outputs": {name: str(args.output_dir / name) for name in output_names[:-1]},
    }
    _atomic_json(report, args.output_dir / "formal_report.json")
    print("[formal_fusion] summary:", flush=True)
    selected = bootstrap_summary.loc[
        bootstrap_summary["metric"].isin(["roc_auc", "brier"])
    ]
    for row in selected.itertuples(index=False):
        print(
            f"  {row.method} {row.metric}: "
            f"{row.point_estimate_mean_over_support_repeats:.4f} "
            f"joint95%=[{row.joint_bootstrap_percentile_lo:.4f}, "
            f"{row.joint_bootstrap_percentile_hi:.4f}] "
            f"delta={row.delta_vs_frozen:+.4f}",
            flush=True,
        )
    print(f"[formal_fusion] outputs: {args.output_dir}", flush=True)
    print(
        f"[formal_fusion] done in {time.time() - started:.1f}s", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
