"""Anchor-split robustness analysis with split-specific PhysicsSim HPO.

Each valid split contains one eligible site from each geographic class:
``southern_route``, ``northern_route``, and ``north_of_mountains``.  The three
sites are used only as calibration anchors for split-specific three-anchor LOO
HPO.  The remaining eight archaeological sites are held out until final RMSE
evaluation.  Splits are never ranked to choose a replacement calibration set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    FILE_PATHS,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import get_anchor_labels, load_site_points
from library.geometry import get_lcc_bounds, get_lcc_parameters


REGION_ORDER = ("southern_route", "northern_route", "north_of_mountains")
REQUIRED_COLUMNS = {
    "model_name",
    "lon",
    "lat",
    "current_use_role",
    "region_class",
    "anchor_eligible",
}
RESUME_CONFIG_KEYS = (
    "candidate_workbook",
    "region_order",
    "split_rule",
    "hpo_validation",
    "final_evaluation",
    "selection_policy",
    "boundary_policy",
    "n_all_valid_splits",
    "selected_split_ids",
    "seeds",
    "hpo_seeds",
    "final_evaluation_seeds",
    "alpha_range",
    "beta_range",
    "w_dis",
    "base_spring_stiffness",
    "base_directional_force",
    "base_repulsion_strength",
    "lcc_bounds",
    "lcc_parameters",
    "input_sha256",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric_grid(minimum: float, maximum: float, step: float, *, name: str) -> list[float]:
    values = np.asarray([minimum, maximum, step], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} grid values must be finite.")
    if step <= 0.0:
        raise ValueError(f"{name} step must be positive.")
    if minimum > maximum:
        raise ValueError(f"{name} minimum cannot exceed maximum.")
    grid = np.arange(minimum, maximum + 1e-12, step, dtype=float)
    if len(grid) == 0 or not np.isclose(grid[-1], maximum, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} range must end exactly on maximum with the requested step.")
    return [float(value) for value in grid]


def _selected_splits(
    all_splits: Sequence[AnchorSplit],
    split_ids: Sequence[str] | None,
    max_splits: int,
) -> list[AnchorSplit]:
    selected = list(all_splits)
    if split_ids:
        requested = set(split_ids)
        selected = [split for split in selected if split.split_id in requested]
        missing = sorted(requested - {split.split_id for split in selected})
        if missing:
            raise ValueError(f"Unknown --split-ids: {missing}")
    if max_splits < 0:
        raise ValueError("--max-splits cannot be negative.")
    if max_splits > 0:
        selected = selected[:max_splits]
    if not selected:
        raise ValueError("No anchor splits selected.")
    return selected


def _bootstrap_ci_mean(values: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorSplit:
    split_id: str
    southern_route_anchor: str
    northern_route_anchor: str
    north_of_mountains_anchor: str
    test_labels: tuple[str, ...]
    is_original_split: bool

    @property
    def anchor_labels(self) -> tuple[str, str, str]:
        return (
            self.southern_route_anchor,
            self.northern_route_anchor,
            self.north_of_mountains_anchor,
        )

    @property
    def final_frame_anchor(self) -> str:
        return self.southern_route_anchor


def _coerce_eligible(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"anchor_eligible must be TRUE or FALSE, got {value!r}")


def load_anchor_candidate_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anchor classification workbook not found: {path}")
    df = pd.read_excel(path, sheet_name="Candidates", dtype=object)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Candidate workbook is missing columns: {missing_columns}")

    df = df.copy()
    df["model_name"] = df["model_name"].fillna("").astype(str).str.strip()
    if (df["model_name"] == "").any():
        raise ValueError("Candidate workbook contains a blank model_name.")
    duplicates = sorted(df.loc[df["model_name"].duplicated(keep=False), "model_name"].unique())
    if duplicates:
        raise ValueError(f"Candidate workbook contains duplicate model_name values: {duplicates}")

    source_rows = load_site_points()
    site_names = [row["name"] for row in source_rows]
    if set(df["model_name"]) != set(site_names) or len(df) != len(site_names):
        missing = sorted(set(site_names) - set(df["model_name"]))
        extra = sorted(set(df["model_name"]) - set(site_names))
        raise ValueError(f"Candidate workbook must contain exactly the 11 site points; missing={missing}, extra={extra}")

    source_by_name = {row["name"]: row for row in source_rows}
    for row in df.to_dict(orient="records"):
        source = source_by_name[row["model_name"]]
        try:
            lon_matches = np.isclose(float(row["lon"]), float(source["lon"]), rtol=0.0, atol=1e-10)
            lat_matches = np.isclose(float(row["lat"]), float(source["lat"]), rtol=0.0, atol=1e-10)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid lon/lat in workbook for {row['model_name']}") from exc
        if not lon_matches or not lat_matches:
            raise ValueError(f"Workbook lon/lat differs from site_rmse_points.csv for {row['model_name']}")
        if str(row["current_use_role"]).strip() != str(source["use_role"]).strip():
            raise ValueError(f"Workbook current_use_role differs from site_rmse_points.csv for {row['model_name']}")

    df["anchor_eligible"] = df["anchor_eligible"].map(_coerce_eligible)
    df["region_class"] = df["region_class"].fillna("").astype(str).str.strip()
    eligible = df["anchor_eligible"]
    blank_region = df.loc[eligible & (df["region_class"] == ""), "model_name"].tolist()
    if blank_region:
        raise ValueError(f"Eligible sites still need region_class ({len(blank_region)} rows).")
    invalid_region = df.loc[eligible & ~df["region_class"].isin(REGION_ORDER), ["model_name", "region_class"]]
    if not invalid_region.empty:
        raise ValueError(f"Unknown region_class values: {invalid_region.to_dict(orient='records')}")
    for region in REGION_ORDER:
        if not bool((eligible & (df["region_class"] == region)).any()):
            raise ValueError(f"No eligible site is assigned to required region: {region}")
    return df


def build_anchor_splits(candidate_df: pd.DataFrame) -> list[AnchorSplit]:
    eligible = candidate_df[candidate_df["anchor_eligible"]].copy()
    groups = {
        region: sorted(eligible.loc[eligible["region_class"] == region, "model_name"].tolist())
        for region in REGION_ORDER
    }
    all_sites = sorted(candidate_df["model_name"].tolist())
    original = frozenset(get_anchor_labels())
    splits: list[AnchorSplit] = []
    for split_index, anchors in enumerate(product(*(groups[region] for region in REGION_ORDER)), start=1):
        anchor_set = frozenset(anchors)
        tests = tuple(label for label in all_sites if label not in anchor_set)
        splits.append(
            AnchorSplit(
                split_id=f"split_{split_index:03d}",
                southern_route_anchor=anchors[0],
                northern_route_anchor=anchors[1],
                north_of_mountains_anchor=anchors[2],
                test_labels=tests,
                is_original_split=anchor_set == original,
            )
        )
    return splits


def _split_definition_rows(splits: Iterable[AnchorSplit]) -> list[dict]:
    return [
        {
            "split_id": split.split_id,
            "southern_route_anchor": split.southern_route_anchor,
            "northern_route_anchor": split.northern_route_anchor,
            "north_of_mountains_anchor": split.north_of_mountains_anchor,
            "final_frame_anchor": split.final_frame_anchor,
            "anchor_labels": "|".join(split.anchor_labels),
            "test_labels": "|".join(split.test_labels),
            "is_original_split": split.is_original_split,
        }
        for split in splits
    ]


def _completed_split(split_dir: Path) -> bool:
    required = (
        "gridsearch_config.json",
        "selected_final_summary.json",
        "selected_final_runs_by_seed.csv",
        "selected_final_site_errors.csv",
    )
    if not all((split_dir / name).is_file() for name in required):
        return False
    try:
        summary = json.loads((split_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(split_dir / "selected_final_runs_by_seed.csv")
        errors = pd.read_csv(split_dir / "selected_final_site_errors.csv")
    except (json.JSONDecodeError, OSError, pd.errors.ParserError):
        return False
    return (
        {"alpha", "beta"}.issubset(summary)
        and {"RMSE_final_test_km", "E_distance_stress", "E_direction_vr"}.issubset(runs.columns)
        and {"site_label", "error_km", "squared_error_km2"}.issubset(errors.columns)
        and not runs.empty
        and not errors.empty
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": _utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _archive_incomplete_split(split_dir: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = archive_root / f"{split_dir.name}_{timestamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{split_dir.name}_{timestamp}_{suffix:02d}"
        suffix += 1
    split_dir.rename(destination)
    return destination


def _assert_resume_config_compatible(existing: dict, requested: dict) -> None:
    mismatches = [key for key in RESUME_CONFIG_KEYS if existing.get(key) != requested.get(key)]
    if mismatches:
        details = {key: {"existing": existing.get(key), "requested": requested.get(key)} for key in mismatches}
        raise ValueError(
            "Resume configuration differs from the existing experiment; use a new --outdir. "
            f"Mismatched fields: {json.dumps(details, ensure_ascii=False)}"
        )


def preflight_anchor_split_robustness(
    *,
    candidates_path: str | Path,
    seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    outdir: str | Path,
    final_seeds: Sequence[int] | None = None,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    split_ids: Sequence[str] | None = None,
    max_splits: int = 0,
    resume: bool = False,
) -> dict:
    candidates_path = Path(candidates_path)
    candidates = load_anchor_candidate_table(candidates_path)
    all_splits = build_anchor_splits(candidates)
    selected = _selected_splits(all_splits, split_ids, max_splits)
    alpha_values = _numeric_grid(alpha_min, alpha_max, alpha_step, name="alpha")
    beta_values = _numeric_grid(beta_min, beta_max, beta_step, name="beta")
    weight_values = {
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in weight_values.values()):
        raise ValueError(f"All base weights must be finite and positive: {weight_values}")

    seed_values = [int(seed) for seed in seeds]
    final_seed_values = seed_values if final_seeds is None else [int(seed) for seed in final_seeds]
    for label, values in (("HPO seeds", seed_values), ("final-evaluation seeds", final_seed_values)):
        if not values:
            raise ValueError(f"At least one {label} value is required.")
        if len(values) != len(set(values)):
            raise ValueError(f"{label.capitalize()} must be unique.")
        if any(seed < 0 for seed in values):
            raise ValueError(f"{label.capitalize()} must be non-negative integers.")

    site_names = set(candidates["model_name"].astype(str))
    original_matches = [split.split_id for split in all_splits if split.is_original_split]
    if len(original_matches) != 1:
        raise ValueError(f"Expected exactly one original anchor split, got {original_matches}.")
    for split in all_splits:
        anchors = set(split.anchor_labels)
        tests = set(split.test_labels)
        if len(anchors) != 3 or len(tests) != 8 or anchors & tests or anchors | tests != site_names:
            raise ValueError(f"Invalid three-anchor/eight-test partition in {split.split_id}.")
        if split.final_frame_anchor not in anchors:
            raise ValueError(f"Final frame anchor is not a calibration anchor in {split.split_id}.")

    input_paths = {
        "candidate_workbook": candidates_path,
        "site_points": Path(FILE_PATHS["ground_truth_path"]),
        "distance_edges": Path(FILE_PATHS["chen_data"]),
        "direction_edges": Path(FILE_PATHS["directional_data"]),
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Required experiment inputs are missing: {missing_inputs}")

    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()) and not resume:
        raise FileExistsError(f"Output folder is non-empty: {outdir}. Use a new path or --resume.")
    existing_completed = 0
    existing_incomplete = 0
    if outdir.exists():
        for split in selected:
            split_dir = outdir / "splits" / split.split_id
            if _completed_split(split_dir):
                existing_completed += 1
            elif split_dir.exists() and any(split_dir.iterdir()):
                existing_incomplete += 1

    disk_parent = outdir.parent
    while not disk_parent.exists() and disk_parent != disk_parent.parent:
        disk_parent = disk_parent.parent
    free_disk_bytes = int(shutil.disk_usage(disk_parent).free)
    if free_disk_bytes < 1_000_000_000:
        raise OSError(f"Less than 1 GB free on the output drive: {free_disk_bytes} bytes.")

    hpo_runs_per_split = len(alpha_values) * len(beta_values) * 3 * len(seed_values)
    return {
        "checked_at_utc": _utc_now(),
        "candidate_workbook": str(candidates_path),
        "n_site_points": int(len(candidates)),
        "n_all_valid_splits": int(len(all_splits)),
        "n_selected_splits": int(len(selected)),
        "selected_split_ids": [split.split_id for split in selected],
        "original_split_id": original_matches[0],
        "seeds": seed_values,
        "hpo_seeds": seed_values,
        "final_evaluation_seeds": final_seed_values,
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "base_weights": weight_values,
        "hpo_runs_per_split": int(hpo_runs_per_split),
        "final_runs_per_split": int(len(final_seed_values)),
        "expected_total_model_runs": int(len(selected) * (hpo_runs_per_split + len(final_seed_values))),
        "existing_completed_splits": int(existing_completed),
        "existing_incomplete_splits": int(existing_incomplete),
        "resume": bool(resume),
        "free_disk_bytes": free_disk_bytes,
        "weight_mapping": {
            "directional_force": "base_directional_force * 10**alpha",
            "repulsion_strength": "base_repulsion_strength * 10**beta",
        },
        "boundary_policy": "fixed common grid; do not expand individual split grids; report boundary-selection frequency",
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
    }


def _load_split_results(split: AnchorSplit, split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    runs = pd.read_csv(split_dir / "selected_final_runs_by_seed.csv")
    errors = pd.read_csv(split_dir / "selected_final_site_errors.csv")
    summary = json.loads((split_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
    for frame in (runs, errors):
        frame.insert(0, "split_id", split.split_id)
        frame.insert(1, "is_original_split", split.is_original_split)
        frame.insert(2, "anchor_labels", "|".join(split.anchor_labels))
        frame.insert(3, "final_frame_anchor", split.final_frame_anchor)
    return runs, errors, summary


def _summarize_split(split: AnchorSplit, runs: pd.DataFrame, selected_summary: dict) -> dict:
    rmse = runs["RMSE_final_test_km"].to_numpy(float)
    stress = runs["E_distance_stress"].to_numpy(float)
    vr = runs["E_direction_vr"].to_numpy(float)
    ci_lo, ci_hi = _bootstrap_ci_mean(rmse, n_boot=10_000, seed=0)
    return {
        "split_id": split.split_id,
        "is_original_split": split.is_original_split,
        "southern_route_anchor": split.southern_route_anchor,
        "northern_route_anchor": split.northern_route_anchor,
        "north_of_mountains_anchor": split.north_of_mountains_anchor,
        "final_frame_anchor": split.final_frame_anchor,
        "test_labels": "|".join(split.test_labels),
        "selected_alpha": float(selected_summary["alpha"]),
        "selected_beta": float(selected_summary["beta"]),
        "selected_on_alpha_boundary": bool(selected_summary.get("selected_on_alpha_boundary", False)),
        "selected_on_beta_boundary": bool(selected_summary.get("selected_on_beta_boundary", False)),
        "selected_on_grid_boundary": bool(selected_summary.get("selected_on_grid_boundary", False)),
        "n_seeds": int(len(rmse)),
        "RMSE_final_test_mean_km": float(np.mean(rmse)),
        "RMSE_final_test_std_km": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else float("nan"),
        "RMSE_final_test_ci95_low_km": ci_lo,
        "RMSE_final_test_ci95_high_km": ci_hi,
        "E_distance_stress_mean": float(np.mean(stress)),
        "E_distance_stress_std": float(np.std(stress, ddof=1)) if len(stress) > 1 else float("nan"),
        "E_direction_vr_mean": float(np.mean(vr)),
        "E_direction_vr_std": float(np.std(vr, ddof=1)) if len(vr) > 1 else float("nan"),
    }


def _write_aggregate_outputs(
    outdir: Path,
    definitions: pd.DataFrame,
    all_runs: list[pd.DataFrame],
    all_errors: list[pd.DataFrame],
    summaries: list[dict],
) -> None:
    definitions.to_csv(outdir / "anchor_split_definitions.csv", index=False, encoding="utf-8-sig")
    if not summaries:
        return
    runs = pd.concat(all_runs, ignore_index=True)
    errors = pd.concat(all_errors, ignore_index=True)
    split_summary = pd.DataFrame(summaries).sort_values("split_id").reset_index(drop=True)
    runs.to_csv(outdir / "anchor_split_final_runs.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(outdir / "anchor_split_site_errors.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(outdir / "anchor_split_summary.csv", index=False, encoding="utf-8-sig")

    site_summary = (
        errors.groupby("site_label", as_index=False)
        .agg(
            n_heldout_observations=("error_km", "size"),
            n_splits=("split_id", "nunique"),
            error_mean_km=("error_km", "mean"),
            error_std_km=("error_km", lambda x: x.std(ddof=1)),
            mean_squared_error_km2=("squared_error_km2", "mean"),
        )
        .sort_values("site_label")
    )
    site_summary["site_rmse_km"] = np.sqrt(site_summary["mean_squared_error_km2"])
    site_summary.to_csv(outdir / "anchor_site_heldout_summary.csv", index=False, encoding="utf-8-sig")

    split_means = split_summary["RMSE_final_test_mean_km"].to_numpy(float)
    within_variances = split_summary["RMSE_final_test_std_km"].to_numpy(float) ** 2
    original_rows = split_summary[split_summary["is_original_split"].astype(bool)]
    original_mean = float(original_rows.iloc[0]["RMSE_final_test_mean_km"]) if len(original_rows) == 1 else None
    original_percentile = (
        float(100.0 * np.mean(split_means <= original_mean)) if original_mean is not None else None
    )
    between_sd = float(np.std(split_means, ddof=1)) if len(split_means) > 1 else None
    finite_within = within_variances[np.isfinite(within_variances)]
    pooled_within_sd = float(math.sqrt(np.mean(finite_within))) if len(finite_within) else None
    global_summary = {
        "estimand": "pipeline RMSE sensitivity to stratified anchor/test split with split-specific HPO",
        "n_completed_splits": int(len(split_summary)),
        "split_mean_rmse_mean_km": float(np.mean(split_means)),
        "split_mean_rmse_std_km": between_sd,
        "split_mean_rmse_median_km": float(np.median(split_means)),
        "split_mean_rmse_min_km": float(np.min(split_means)),
        "split_mean_rmse_max_km": float(np.max(split_means)),
        "pooled_within_split_seed_sd_km": pooled_within_sd,
        "between_to_within_sd_ratio": (
            between_sd / pooled_within_sd
            if between_sd is not None and pooled_within_sd is not None and pooled_within_sd > 0
            else None
        ),
        "site_balanced_rmse_km": float(math.sqrt(site_summary["mean_squared_error_km2"].mean())),
        "original_split_mean_rmse_km": original_mean,
        "original_split_percentile_lower_is_better": original_percentile,
        "n_selected_on_grid_boundary": int(split_summary["selected_on_grid_boundary"].sum()),
        "selected_on_grid_boundary_rate": float(split_summary["selected_on_grid_boundary"].mean()),
        "boundary_policy": "fixed common grid; no split-specific expansion; boundary frequency is reported",
        "interpretation_note": "No split is selected as a replacement calibration set; all valid splits are reported.",
    }
    (outdir / "anchor_split_global_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (
        split_summary.groupby(["selected_alpha", "selected_beta"], as_index=False)
        .size()
        .rename(columns={"size": "n_splits"})
        .sort_values("n_splits", ascending=False)
        .to_csv(outdir / "selected_hyperparameter_frequency.csv", index=False, encoding="utf-8-sig")
    )
    _plot_split_rmse_summary(split_summary, outdir)


def _plot_split_rmse_summary(split_summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = split_summary.sort_values("RMSE_final_test_mean_km").reset_index(drop=True)
    x = np.arange(len(ordered))
    means = ordered["RMSE_final_test_mean_km"].to_numpy(float)
    lower = means - ordered["RMSE_final_test_ci95_low_km"].to_numpy(float)
    upper = ordered["RMSE_final_test_ci95_high_km"].to_numpy(float) - means
    original = ordered["is_original_split"].astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(max(8.0, len(ordered) * 0.22), 5.2))
    ax.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="o", ms=4, lw=0.8, capsize=2, color="#4472C4")
    if original.any():
        ax.scatter(x[original], means[original], marker="*", s=180, color="#C00000", edgecolor="black", label="Original split", zorder=4)
        ax.legend()
    ax.set_xlabel("Anchor split, sorted by mean held-out RMSE")
    ax.set_ylabel("Held-out test RMSE (km), mean and 95% bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "anchor_split_rmse_stability.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(means, bins="auto", color="#5B9BD5", edgecolor="white")
    if original.any():
        original_mean = float(means[original][0])
        ax.axvline(original_mean, color="#C00000", lw=2, label="Original split")
        ax.legend()
    ax.set_xlabel("Split-level mean held-out test RMSE (km)")
    ax.set_ylabel("Number of anchor splits")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "anchor_split_rmse_distribution.png", dpi=220)
    plt.close(fig)


def run_anchor_split_robustness(
    *,
    candidates_path: str | Path,
    seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    w_dis: float,
    base_spring_stiffness: float,
    base_directional_force: float,
    base_repulsion_strength: float,
    outdir: str | Path,
    final_seeds: Sequence[int] | None = None,
    split_ids: Sequence[str] | None = None,
    max_splits: int = 0,
    resume: bool = False,
    generate_split_plots: bool = False,
) -> dict:
    from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import run_anchor_loo_gridsearch_pareto

    candidates = load_anchor_candidate_table(candidates_path)
    all_splits = build_anchor_splits(candidates)
    selected_splits = _selected_splits(all_splits, split_ids, max_splits)

    preflight = preflight_anchor_split_robustness(
        candidates_path=candidates_path,
        seeds=seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        w_dis=w_dis,
        base_spring_stiffness=base_spring_stiffness,
        base_directional_force=base_directional_force,
        base_repulsion_strength=base_repulsion_strength,
        outdir=outdir,
        split_ids=split_ids,
        max_splits=max_splits,
        resume=resume,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits_root = outdir / "splits"
    splits_root.mkdir(exist_ok=True)
    status_root = outdir / "split_status"
    archive_root = outdir / "interrupted_attempts"
    event_log = outdir / "experiment_events.jsonl"
    definitions = pd.DataFrame(_split_definition_rows(all_splits))
    definitions.to_csv(outdir / "anchor_split_definitions.csv", index=False, encoding="utf-8-sig")

    config = {
        "experiment": "anchor_split_robustness_with_split_specific_hpo",
        "candidate_workbook": str(Path(candidates_path)),
        "region_order": list(REGION_ORDER),
        "split_rule": "one eligible site from each region; southern route site is final frame anchor",
        "hpo_validation": "three-anchor leave-one-anchor-out; held-out anchor never defines its fold frame",
        "final_evaluation": "remaining eight sites only",
        "selection_policy": "report every selected split; never select anchors by final RMSE",
        "boundary_policy": "fixed common grid; do not expand individual split grids; report boundary-selection frequency",
        "n_all_valid_splits": len(all_splits),
        "selected_split_ids": [split.split_id for split in selected_splits],
        "seeds": list(map(int, seeds)),
        "hpo_seeds": list(map(int, seeds)),
        "final_evaluation_seeds": list(map(int, final_seeds if final_seeds is not None else seeds)),
        "alpha_range": [alpha_min, alpha_max, alpha_step],
        "beta_range": [beta_min, beta_max, beta_step],
        "w_dis": w_dis,
        "base_spring_stiffness": base_spring_stiffness,
        "base_directional_force": base_directional_force,
        "base_repulsion_strength": base_repulsion_strength,
        "lcc_bounds": dict(zip(["lon_min", "lon_max", "lat_min", "lat_max"], map(float, get_lcc_bounds()))),
        "lcc_parameters": dict(zip(["lat_1", "lat_2", "lon_0"], map(float, get_lcc_parameters()))),
        "input_sha256": {
            "candidate_workbook": _sha256(candidates_path),
            "site_points": _sha256(FILE_PATHS["ground_truth_path"]),
            "distance_edges": _sha256(FILE_PATHS["chen_data"]),
            "direction_edges": _sha256(FILE_PATHS["directional_data"]),
        },
    }
    config_path = outdir / "anchor_split_config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"Configuration already exists: {config_path}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        _assert_resume_config_compatible(existing_config, config)
    else:
        _write_json(config_path, config)

    preflight_name = "preflight_report.json" if not resume else f"preflight_report_resume_{datetime.now():%Y%m%d_%H%M%S}.json"
    _write_json(outdir / preflight_name, preflight)
    _append_event(
        event_log,
        {
            "event": "preflight_passed",
            "resume": bool(resume),
            "n_selected_splits": len(selected_splits),
            "expected_total_model_runs": preflight["expected_total_model_runs"],
        },
    )

    all_runs: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    summaries: list[dict] = []
    fresh_split_durations: list[float] = []
    for index, split in enumerate(selected_splits, start=1):
        split_dir = splits_root / split.split_id
        status_path = status_root / f"{split.split_id}.json"
        print(f"[{index}/{len(selected_splits)}] {split.split_id}: anchors={split.anchor_labels}")
        if _completed_split(split_dir):
            if not resume:
                raise FileExistsError(f"Completed split output already exists: {split_dir}")
            print("  [Resume] using completed split output")
            _append_event(event_log, {"event": "resume_completed_split", "split_id": split.split_id})
            _write_json(
                status_path,
                {
                    "split_id": split.split_id,
                    "status": "completed",
                    "resumed_at_utc": _utc_now(),
                    "output_dir": str(split_dir),
                },
            )
        else:
            if split_dir.exists() and any(split_dir.iterdir()):
                if not resume:
                    raise RuntimeError(
                        f"Incomplete split folder exists: {split_dir}. Use --resume to preserve and retry it."
                    )
                archived = _archive_incomplete_split(split_dir, archive_root)
                print(f"  [Resume] archived incomplete attempt to {archived}")
                _append_event(
                    event_log,
                    {
                        "event": "archive_incomplete_split",
                        "split_id": split.split_id,
                        "archived_to": str(archived),
                    },
                )
            split_started = time.perf_counter()
            _append_event(
                event_log,
                {
                    "event": "split_started",
                    "split_id": split.split_id,
                    "split_index": index,
                    "n_selected_splits": len(selected_splits),
                    "anchors": list(split.anchor_labels),
                },
            )
            _write_json(
                status_path,
                {
                    "split_id": split.split_id,
                    "status": "running",
                    "started_at_utc": _utc_now(),
                    "process_id": os.getpid(),
                    "output_dir": str(split_dir),
                },
            )
            try:
                run_anchor_loo_gridsearch_pareto(
                    seeds=seeds,
                    final_seeds=final_seeds,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    alpha_step=alpha_step,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    beta_step=beta_step,
                    w_dis=w_dis,
                    base_spring_stiffness=base_spring_stiffness,
                    base_directional_force=base_directional_force,
                    base_repulsion_strength=base_repulsion_strength,
                    refer_pos_sim=DEFAULT_REFER_POS_SIM,
                    outdir=split_dir,
                    anchor_labels_override=split.anchor_labels,
                    test_labels_override=split.test_labels,
                    final_frame_anchor_label=split.final_frame_anchor,
                    generate_plots=generate_split_plots,
                    save_final_positions=True,
                )
                if not _completed_split(split_dir):
                    raise RuntimeError(f"Split finished without all required outputs: {split_dir}")
            except BaseException as exc:
                elapsed_seconds = time.perf_counter() - split_started
                _append_event(
                    event_log,
                    {
                        "event": "split_failed",
                        "split_id": split.split_id,
                        "elapsed_seconds": elapsed_seconds,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                _write_json(
                    status_path,
                    {
                        "split_id": split.split_id,
                        "status": "failed",
                        "failed_at_utc": _utc_now(),
                        "process_id": os.getpid(),
                        "output_dir": str(split_dir),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            elapsed_seconds = time.perf_counter() - split_started
            fresh_split_durations.append(elapsed_seconds)
            remaining = len(selected_splits) - index
            eta_seconds = float(np.mean(fresh_split_durations) * remaining)
            print(
                f"  [Completed] {split.split_id} in {elapsed_seconds / 60.0:.1f} min; "
                f"estimated remaining {eta_seconds / 3600.0:.1f} h"
            )
            _append_event(
                event_log,
                {
                    "event": "split_completed",
                    "split_id": split.split_id,
                    "elapsed_seconds": elapsed_seconds,
                    "estimated_remaining_seconds": eta_seconds,
                },
            )
            _write_json(
                status_path,
                {
                    "split_id": split.split_id,
                    "status": "completed",
                    "completed_at_utc": _utc_now(),
                    "process_id": os.getpid(),
                    "output_dir": str(split_dir),
                },
            )
        runs, errors, selected_summary = _load_split_results(split, split_dir)
        all_runs.append(runs)
        all_errors.append(errors)
        summaries.append(_summarize_split(split, runs, selected_summary))
        _write_aggregate_outputs(outdir, definitions, all_runs, all_errors, summaries)

    return {
        "outdir": outdir,
        "n_all_valid_splits": len(all_splits),
        "n_completed_splits": len(summaries),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-specific HPO robustness analysis across stratified anchors")
    parser.add_argument("--candidates", default="data/anchor_robustness_candidates.xlsx")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--final-seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Seeds used after HPO for final held-out RMSE evaluation.",
    )
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--beta-min", type=float, default=-2.0)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.5)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    parser.add_argument("--outdir", default=str(Path(OUTPUT_DIR) / "ch5_anchor_split_robustness"))
    parser.add_argument("--split-ids", default="", help="Optional comma-separated deterministic split IDs")
    parser.add_argument("--max-splits", type=int, default=0, help="Smoke-test limit; 0 runs every valid split")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-split-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_ids = [value.strip() for value in args.split_ids.split(",") if value.strip()]
    run_anchor_split_robustness(
        candidates_path=args.candidates,
        seeds=_parse_seed_list(args.seeds),
        final_seeds=_parse_seed_list(args.final_seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        outdir=args.outdir,
        split_ids=split_ids,
        max_splits=args.max_splits,
        resume=args.resume,
        generate_split_plots=args.generate_split_plots,
    )


if __name__ == "__main__":
    main()
