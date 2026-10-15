#!/usr/bin/env python3
"""Formal frozen-model Platt recalibration on an independent target support.

This is the missing ``0 adaptation deaths + 50 calibration deaths`` arm in
the manuscript's staged-update comparison.  The script reconstructs the same
patient query and the same disjoint adaptation/calibration support draws used
by the fixed K=200 full-head experiment.  Adaptation patients are reserved so
that the calibration patients match that experiment exactly, but their labels
are never used to update the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.formal_full_head_fusion import (
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _filter_group,
    _filter_partition,
    _git_metadata,
    _sha256,
    disjoint_exact_positive_samples,
    paired_joint_bootstrap,
    summarize_bootstrap,
)

METHOD_FROZEN = "frozen"
METHOD_FROZEN_PLATT = "frozen_platt"
RECALIBRATION_METHODS = (METHOD_FROZEN, METHOD_FROZEN_PLATT)


def prepare_recalibration_frame(
    frame: pd.DataFrame, *, horizon_days: int
) -> pd.DataFrame:
    """Validate the anonymous columns needed by recalibration-only."""

    required = {
        "cohort",
        "horizon_days",
        "patient_id_hash",
        "t_pred",
        "y_true",
        "y_prob",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"target table is missing columns: {missing}")
    forbidden = {"patient_id", "person_id", "PERSON_ID_NEW", "icu_index", "stay_id"}
    if exposed := sorted(forbidden & set(frame.columns)):
        raise ValueError(f"raw identifier columns are forbidden: {exposed}")
    out = frame.loc[
        pd.to_numeric(frame["horizon_days"], errors="raise").astype(int)
        == int(horizon_days)
    ].copy()
    if out.empty:
        raise ValueError(f"no rows for horizon_days={horizon_days}")
    if out["cohort"].astype(str).nunique() != 1:
        raise ValueError("target table must contain exactly one cohort")
    if out["patient_id_hash"].isna().any() or out["t_pred"].isna().any():
        raise ValueError("hashed patient IDs and prediction times cannot be missing")
    if out.duplicated(
        ["cohort", "horizon_days", "patient_id_hash", "t_pred"]
    ).any():
        raise ValueError("target table contains duplicate prediction rows")
    out["patient_id_hash"] = out["patient_id_hash"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(np.int8)
    out["y_prob"] = pd.to_numeric(out["y_prob"], errors="raise").astype(float)
    if not set(out["y_true"].unique()).issubset({0, 1}):
        raise ValueError("y_true must contain only 0/1")
    probabilities = out["y_prob"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("y_prob must be finite and in [0,1]")
    return out.reset_index(drop=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-head-features", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, choices=(1, 2, 3, 7), required=True)
    parser.add_argument("--target-site-partition")
    parser.add_argument("--target-group-column")
    parser.add_argument("--target-group-value")
    parser.add_argument("--reserved-adaptation-positive", type=int, default=200)
    parser.add_argument("--calibration-positive", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--evaluation-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    for path, label in (
        (args.target_head_features, "target head-feature parquet"),
        (args.bundle, "source bundle"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"{label} is missing or empty: {path}")
    if args.reserved_adaptation_positive < 1 or args.calibration_positive < 1:
        raise SystemExit("positive-event budgets must be positive")
    if args.repeats < 1 or args.bootstrap < 1:
        raise SystemExit("--repeats and --bootstrap must be positive")
    if not 0.0 < args.evaluation_fraction < 1.0:
        raise SystemExit("--evaluation-fraction must be in (0,1)")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be in (0,1)")

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
        args.output_dir / name
        for name in output_names
        if (args.output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to overwrite outputs; use --overwrite:\n  "
            + "\n  ".join(map(str, existing))
        )

    from patient_data.mortality.transfer_calibration import (
        _base_metrics,
        _bin_rows,
        _equal_frequency_bins,
        fit_platt,
        patient_partition,
    )

    def evaluation_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        metrics = _base_metrics(y, p)
        ici, _rows = _bin_rows(
            y=y,
            probabilities=p,
            bins=_equal_frequency_bins(p, args.calibration_bins),
        )
        metrics["ici"] = ici
        return metrics

    bundle_hash = _sha256(args.bundle)
    sidecar_path = args.target_head_features.with_suffix(
        args.target_head_features.suffix + ".metadata.json"
    )
    sidecar = None
    if sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        recorded_hash = sidecar.get("bundle_sha256")
        if recorded_hash and recorded_hash != bundle_hash:
            raise SystemExit(
                "target cache was produced by a different bundle: "
                f"{recorded_hash} != {bundle_hash}"
            )
        recorded_horizon = sidecar.get("death_horizon_days")
        if recorded_horizon is not None and int(recorded_horizon) != args.horizon_days:
            raise SystemExit(
                "target cache horizon mismatch: "
                f"{recorded_horizon} != {args.horizon_days}"
            )

    target = prepare_recalibration_frame(
        pd.read_parquet(args.target_head_features),
        horizon_days=args.horizon_days,
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
    target_ids = target["patient_id_hash"].to_numpy(dtype=str)
    target_y = target["y_true"].to_numpy(dtype=int)
    target_p = target["y_prob"].to_numpy(dtype=float)
    partition = patient_partition(
        target,
        evaluation_fraction=args.evaluation_fraction,
        random_seed=args.seed,
    )
    evaluation_mask = np.isin(target_ids, partition.evaluation)
    y_eval = target_y[evaluation_mask]
    p_eval = target_p[evaluation_mask]
    ids_eval = target_ids[evaluation_mask]
    print(
        f"[formal_recalibration] target_rows={len(target):,} "
        f"target_patients={target['patient_id_hash'].nunique():,} "
        f"pool_positive={sum(partition.patient_positive[p] for p in partition.calibration_pool):,} "
        f"query_patients={len(partition.evaluation):,} "
        f"query_rows={len(y_eval):,}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 1)
    replicate_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    predictions: dict[int, dict[str, np.ndarray]] = {}
    prediction_frame = target.loc[
        evaluation_mask,
        ["cohort", "horizon_days", "patient_id_hash", "t_pred", "y_true"],
    ].reset_index(drop=True)
    prediction_frame["frozen_probability"] = p_eval.astype(np.float32)
    frozen_metrics = evaluation_metrics(y_eval, p_eval)

    for replicate in range(args.repeats):
        reserved_adaptation, calibration_patients = disjoint_exact_positive_samples(
            partition.patient_positive,
            partition.calibration_pool,
            adaptation_positive=args.reserved_adaptation_positive,
            calibration_positive=args.calibration_positive,
            rng=rng,
        )
        reserved_mask = np.isin(target_ids, reserved_adaptation)
        calibration_mask = np.isin(target_ids, calibration_patients)
        if np.any(reserved_mask & calibration_mask):
            raise AssertionError("reserved adaptation/calibration leakage")
        if np.any((reserved_mask | calibration_mask) & evaluation_mask):
            raise AssertionError("support/query patient leakage")
        y_calibration = target_y[calibration_mask]
        p_calibration = target_p[calibration_mask]
        fitted = fit_platt(y_calibration, p_calibration)
        intercept, slope = fitted.intercept, fitted.slope
        platt_probability = fitted.transform(p_eval)
        common = {
            "replicate": int(replicate),
            "reserved_adaptation_n_patients": int(len(reserved_adaptation)),
            "reserved_adaptation_n_positive_patients": int(
                args.reserved_adaptation_positive
            ),
            "calibration_n_patients": int(len(calibration_patients)),
            "calibration_n_positive_patients": int(args.calibration_positive),
            "calibration_n_samples": int(calibration_mask.sum()),
            "evaluation_n_patients": int(len(partition.evaluation)),
            "evaluation_n_samples": int(len(y_eval)),
        }
        calibration_rows.append(
            {
                **common,
                "method": METHOD_FROZEN_PLATT,
                "intercept": float(intercept),
                "slope": float(slope),
                "fit_strategy": fitted.fit_strategy,
                "l2_penalty": float(fitted.l2_penalty),
                "optimizer_iterations": int(fitted.optimizer_iterations),
            }
        )
        repeat_predictions = {
            METHOD_FROZEN: p_eval.copy(),
            METHOD_FROZEN_PLATT: platt_probability,
        }
        predictions[replicate] = repeat_predictions
        for method, probability in repeat_predictions.items():
            metrics = (
                frozen_metrics
                if method == METHOD_FROZEN
                else evaluation_metrics(y_eval, probability)
            )
            replicate_rows.append({**common, "method": method, **metrics})
        prediction_frame[
            f"{METHOD_FROZEN_PLATT}_r{replicate:02d}_probability"
        ] = platt_probability.astype(np.float32)
        print(
            f"[formal_recalibration] replicate={replicate + 1}/{args.repeats} "
            f"calibration_patients={len(calibration_patients):,} complete",
            flush=True,
        )

    replicate_metrics = pd.DataFrame.from_records(replicate_rows)
    calibration_parameters = pd.DataFrame.from_records(calibration_rows)
    bootstrap = paired_joint_bootstrap(
        y_eval,
        ids_eval,
        predictions,
        n_bootstrap=args.bootstrap,
        random_seed=args.bootstrap_seed,
        methods=RECALIBRATION_METHODS,
        reference_method=METHOD_FROZEN,
    )
    bootstrap_summary = summarize_bootstrap(
        replicate_metrics,
        bootstrap,
        alpha=args.alpha,
        methods=RECALIBRATION_METHODS,
        reference_method=METHOD_FROZEN,
    )

    predictions_path = args.output_dir / "query_predictions.parquet"
    _atomic_parquet(prediction_frame, predictions_path)
    _atomic_json(
        {
            "schema_version": "formal_recalibration_only_predictions_v1",
            "n_rows": int(len(prediction_frame)),
            "n_patients": int(prediction_frame["patient_id_hash"].nunique()),
            "n_positive_rows": int(prediction_frame["y_true"].sum()),
            "columns": list(prediction_frame.columns),
            "identifier_policy": "pseudonymized hash copied from input cache",
            "raw_identifiers_persisted": False,
            "file_sha256": _sha256(predictions_path),
            "file_size_bytes": int(predictions_path.stat().st_size),
        },
        args.output_dir / "query_predictions.parquet.metadata.json",
    )
    _atomic_csv(replicate_metrics, args.output_dir / "replicate_metrics.csv")
    _atomic_csv(
        calibration_parameters, args.output_dir / "calibration_parameters.csv"
    )
    _atomic_csv(bootstrap, args.output_dir / "bootstrap_replicates.csv")
    _atomic_csv(bootstrap_summary, args.output_dir / "bootstrap_summary.csv")
    report = {
        "schema_version": "formal_recalibration_only_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "horizon_days": int(args.horizon_days),
        "input": {
            "target_path": str(args.target_head_features),
            "target_sha256": _sha256(args.target_head_features),
            "target_sidecar": sidecar,
            "bundle_path": str(args.bundle),
            "bundle_sha256": bundle_hash,
        },
        **_git_metadata(),
        "design": (
            "frozen source probabilities; repeated independent Platt "
            "calibration supports; fixed untouched patient query"
        ),
        "split_unit": "patient_id_hash",
        "query_estimand": "prediction_day",
        "encoder_policy": "frozen",
        "adapted_parameters": 0,
        "target_site_partition": args.target_site_partition or None,
        "target_group_column": args.target_group_column,
        "target_group_value": args.target_group_value,
        "target_n_patients": int(target["patient_id_hash"].nunique()),
        "target_n_samples": int(len(target)),
        "pool_n_patients": int(len(partition.calibration_pool)),
        "pool_n_positive_patients": int(
            sum(
                partition.patient_positive[p]
                for p in partition.calibration_pool
            )
        ),
        "query_n_patients": int(len(partition.evaluation)),
        "query_n_positive_patients": int(
            sum(partition.patient_positive[p] for p in partition.evaluation)
        ),
        "query_n_samples": int(len(y_eval)),
        "adaptation_positive_patients_used": 0,
        "reserved_adaptation_positive_patients": int(
            args.reserved_adaptation_positive
        ),
        "calibration_positive_patients": int(args.calibration_positive),
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
        "repeats": int(args.repeats),
        "evaluation_fraction": float(args.evaluation_fraction),
        "seed": int(args.seed),
        "bootstrap": {
            "n": int(args.bootstrap),
            "seed": int(args.bootstrap_seed),
            "alpha": float(args.alpha),
            "unit": "patient cluster",
            "support_uncertainty": "one repeated calibration support per draw",
        },
        "methods": list(RECALIBRATION_METHODS),
        "elapsed_seconds": float(time.time() - started),
        "outputs": {name: str(args.output_dir / name) for name in output_names[:-1]},
    }
    _atomic_json(report, args.output_dir / "formal_report.json")
    print("[formal_recalibration] summary:", flush=True)
    for row in bootstrap_summary.itertuples(index=False):
        if row.metric not in {"roc_auc", "brier"}:
            continue
        print(
            f"  {row.method} {row.metric}: "
            f"{row.point_estimate_mean_over_support_repeats:.4f} "
            f"joint95%=[{row.joint_bootstrap_percentile_lo:.4f}, "
            f"{row.joint_bootstrap_percentile_hi:.4f}] "
            f"delta={row.delta_vs_frozen:+.4f}",
            flush=True,
        )
    print(f"[formal_recalibration] outputs: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
