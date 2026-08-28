"""Refresh paper_results/current from the selected formal experiment outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_paper_figures import export_figures
from scripts.export_paper_tables import (
    export_dc_smacof_vs_distdir_comparison,
    export_distdir_vs_distdiranch_comparison,
    export_distdiranch_vs_full_comparison,
    export_distonly_vs_distdir_comparison,
    export_overall_model_comparison,
    export_progressive_chain_summary,
    export_random_layout_summary,
    export_smacof_dc_smacof_baseline_full_statistics,
    export_smacof_vs_distonly_comparison,
)
from scripts.export_result_chapter_tables import export_result_chapter_tables
from scripts.export_physics_bfgs_comparison import export_physics_bfgs_comparison


DEFAULT_AS_OUTDIR = (
    PROJECT_ROOT
    / "outputs"
    / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
)
DEFAULT_DC_HPO_OUTDIR = (
    PROJECT_ROOT / "outputs" / "ch5_dc_smacof_hparam_wang_current_alpha_-4_0_seed0_9_20260721"
)
DEFAULT_PAPER_RESULTS = PROJECT_ROOT / "paper_results" / "current"
DEFAULT_BFGS_OUTDIR = (
    PROJECT_ROOT
    / "outputs"
    / "ch5_scipy_bfgs_hpo_selected_alpha_0p5_beta_-0p5_100seeds_20260823"
)

AS_MAIN_FILES = [
    "progressive_config.json",
    "progressive_paired_comparisons.csv",
    "progressive_run_status.csv",
    "progressive_summary.csv",
    "random_align_percentiles.csv",
    "random_align_summary.csv",
]
AS_SUPPLEMENTARY_FILES = [
    "progressive_final_positions_y_up_sim.csv",
    "progressive_runs_by_seed.csv",
    "random_align_runs.csv",
]
BOOTSTRAP_FORMAL_FILES = [
    "bootstrap_config.json",
    "bootstrap_run_parameters.csv",
    "bootstrap_samples_y_up_sim.csv",
    "bootstrap_samples_lonlat.csv",
    "bootstrap_ellipse_summary.csv",
    "bootstrap_kde_status.csv",
    "bootstrap_anchor_drift.csv",
    "confidence_ellipses.png",
    "confidence_ellipses.svg",
    "combined_kde_density.png",
    "combined_kde_density.svg",
]
BFGS_FORMAL_FILES = [
    "bfgs_experiment_config.json",
    "bfgs_runs_by_seed.csv",
    "bfgs_summary.csv",
    "bfgs_final_positions_y_up_sim.csv",
]
CORE_METRICS = ["E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_test_km"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _copy_file(source: Path, destination: Path, source_map: dict[str, str], paper_results: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing paper-results source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_map[destination.relative_to(paper_results).as_posix()] = str(source.resolve())


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clear_flat_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if not child.is_file():
            raise RuntimeError(f"Refusing to clear non-file entry from formal table directory: {child}")
        child.unlink()


def _remove_flat_directory(path: Path) -> None:
    if not path.exists():
        return
    _clear_flat_directory(path)
    path.rmdir()


def _validate_formal_sources(as_outdir: Path, dc_hpo_outdir: Path) -> dict:
    config = json.loads((as_outdir / "progressive_config.json").read_text(encoding="utf-8"))
    dc_selected = pd.read_csv(dc_hpo_outdir / "dc_smacof_selected_candidate.csv").iloc[0]
    runs = pd.read_csv(as_outdir / "progressive_runs_by_seed.csv")
    positions = pd.read_csv(as_outdir / "progressive_final_positions_y_up_sim.csv")

    if int(config.get("failure_count", -1)) != 0:
        raise ValueError(f"Formal AS contains failed runs: {config.get('failure_count')}")
    dc_params = config.get("dc_smacof_hpo", {})
    expected = {"alpha": -2.0, "w_weight": 1.0, "v_weight": 0.01}
    for key, value in expected.items():
        if not np.isclose(float(dc_params.get(key, np.nan)), value, rtol=0.0, atol=1e-12):
            raise ValueError(f"Formal AS {key} is not the selected DC-SMACOF value: {dc_params.get(key)}")
        if not np.isclose(float(dc_selected[key]), value, rtol=0.0, atol=1e-12):
            raise ValueError(f"DC-SMACOF HPO candidate {key} mismatch: {dc_selected[key]}")

    expected_counts = {
        "PhysicsSim-DistOnly": 100,
        "PhysicsSim-DistDir": 100,
        "PhysicsSim-DistDirAnch": 100,
        "PhysicsSim-Full": 100,
        "SMACOF": 100,
        "DC-SMACOF": 100,
        "Random+Align": 1000,
    }
    actual_counts = runs[runs["status"] == "ok"].groupby("variant").size().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected formal AS run counts: {actual_counts}")
    if (runs["status"] != "ok").any():
        raise ValueError("Formal AS includes non-ok run status rows.")
    if not np.isfinite(runs[CORE_METRICS].to_numpy(float)).all():
        raise ValueError("Formal AS contains non-finite core metrics.")
    if not np.isfinite(positions[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)).all():
        raise ValueError("Formal AS contains non-finite final positions.")
    return config


def _validate_bootstrap_source(bootstrap_outdir: Path, as_config: dict) -> dict:
    config_path = bootstrap_outdir / "bootstrap_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing bootstrap config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("method_classification") != (
        "parameter_perturbation_repeated_simulation_not_observation_resampling"
    ):
        raise ValueError("Bootstrap method classification is missing or unexpected.")
    if config.get("perturbation_space") != "HPO log10 alpha/beta with w_dis fixed at 1":
        raise ValueError("Bootstrap output does not use the formal HPO alpha/beta perturbation design.")
    if int(config.get("failure_count", -1)) != 0:
        raise ValueError(f"Bootstrap output contains failed runs: {config.get('failure_count')}")
    n_bootstrap = int(config.get("n_bootstrap", 0))
    if n_bootstrap < 300:
        raise ValueError(
            f"Formal bootstrap output requires at least 300 runs; found {n_bootstrap}. "
            "Smoke-test output must not be copied into paper_results."
        )

    selection = config.get("hpo_selection", {})
    for key in ("alpha", "beta"):
        if not np.isclose(
            float(selection.get(key, np.nan)),
            float(as_config.get(key, np.nan)),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Bootstrap HPO {key} does not match the formal AS: "
                f"{selection.get(key)} != {as_config.get(key)}"
            )
    if config.get("input_validation", {}).get("lcc_matches_current_data") is not True:
        raise ValueError("Bootstrap output did not verify the current LCC parameters.")
    if config.get("input_validation", {}).get("site_roles_match_current_data") is not True:
        raise ValueError("Bootstrap output did not verify current anchor/test roles.")
    recorded_sources = [
        (
            selection.get("selected_result_file"),
            selection.get("selected_result_sha256"),
            "selected HPO candidate",
        ),
        (
            config.get("input_validation", {}).get("physics_hpo_config"),
            config.get("input_validation", {}).get("physics_hpo_config_sha256"),
            "PhysicsSim HPO grid config",
        ),
    ]
    for recorded_path, recorded_hash, label in recorded_sources:
        if not recorded_path or not recorded_hash:
            raise ValueError(f"Bootstrap config does not record the {label} path and SHA-256.")
        source_path = _resolve_recorded_path(str(recorded_path))
        if not source_path.is_file():
            raise FileNotFoundError(f"Recorded {label} source is missing: {source_path}")
        if _sha256(source_path).lower() != str(recorded_hash).lower():
            raise ValueError(f"Recorded {label} SHA-256 no longer matches its source file.")

    for name in BOOTSTRAP_FORMAL_FILES:
        if not (bootstrap_outdir / name).is_file():
            raise FileNotFoundError(f"Missing formal bootstrap artifact: {bootstrap_outdir / name}")
    artifact_hashes = config.get("artifact_sha256", {})
    for name in BOOTSTRAP_FORMAL_FILES:
        if name == "bootstrap_config.json":
            continue
        expected_hash = str(artifact_hashes.get(name, "")).lower()
        actual_hash = _sha256(bootstrap_outdir / name).lower()
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"Bootstrap artifact SHA-256 mismatch: {name}")

    runs = pd.read_csv(bootstrap_outdir / "bootstrap_run_parameters.csv")
    positions = pd.read_csv(bootstrap_outdir / "bootstrap_samples_y_up_sim.csv")
    lonlat = pd.read_csv(bootstrap_outdir / "bootstrap_samples_lonlat.csv")
    ellipses = pd.read_csv(bootstrap_outdir / "bootstrap_ellipse_summary.csv")
    kde_status = pd.read_csv(bootstrap_outdir / "bootstrap_kde_status.csv")
    anchor_drift = pd.read_csv(bootstrap_outdir / "bootstrap_anchor_drift.csv")

    if len(runs) != n_bootstrap:
        raise ValueError(f"Bootstrap run count mismatch: {len(runs)} != {n_bootstrap}")
    if sorted(runs["bootstrap_index"].astype(int).tolist()) != list(range(n_bootstrap)):
        raise ValueError("Bootstrap run indices are incomplete or duplicated.")
    if runs["simulation_seed"].duplicated().any():
        raise ValueError("Bootstrap simulation seeds must be unique.")
    required_run_columns = {
        "alpha", "beta", "alpha_noise", "beta_noise", "w_dis", "w_dir", "w_reg",
        "spring_stiffness", "directional_force", "repulsion_strength",
    }
    missing_run_columns = required_run_columns.difference(runs.columns)
    if missing_run_columns:
        raise ValueError(f"Bootstrap run parameters are missing HPO fields: {sorted(missing_run_columns)}")

    selected_alpha = float(selection["alpha"])
    selected_beta = float(selection["beta"])
    selected_w_dis = float(selection.get("w_dis", 1.0))
    selected_spring = float(selection["spring_stiffness"])
    selected_directional = float(selection["directional_force"])
    selected_repulsion = float(selection["repulsion_strength"])
    if not np.allclose(runs["w_dis"], selected_w_dis, rtol=0.0, atol=1e-12):
        raise ValueError("Bootstrap w_dis must remain fixed at the selected HPO value.")
    if not np.allclose(runs["spring_stiffness"], selected_spring, rtol=0.0, atol=1e-9):
        raise ValueError("Bootstrap spring stiffness must remain fixed.")
    expected_w_dir = selected_w_dis * np.power(10.0, runs["alpha"].to_numpy(float))
    expected_w_reg = selected_w_dis * np.power(10.0, runs["beta"].to_numpy(float))
    expected_directional = selected_directional * np.power(
        10.0, runs["alpha"].to_numpy(float) - selected_alpha
    )
    expected_repulsion = selected_repulsion * np.power(
        10.0, runs["beta"].to_numpy(float) - selected_beta
    )
    conversions = {
        "w_dir": expected_w_dir,
        "w_reg": expected_w_reg,
        "directional_force": expected_directional,
        "repulsion_strength": expected_repulsion,
        "alpha_noise": runs["alpha"].to_numpy(float) - selected_alpha,
        "beta_noise": runs["beta"].to_numpy(float) - selected_beta,
    }
    for column, expected_values in conversions.items():
        if not np.allclose(runs[column].to_numpy(float), expected_values, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Bootstrap {column} does not match the recorded alpha/beta transformation.")
    first = runs.sort_values("bootstrap_index").iloc[0]
    if not (
        np.isclose(float(first["alpha"]), selected_alpha, rtol=0.0, atol=1e-12)
        and np.isclose(float(first["beta"]), selected_beta, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Bootstrap index 0 must reproduce the unperturbed selected HPO candidate.")

    sample_counts = positions.groupby("bootstrap_index").size()
    lonlat_counts = lonlat.groupby("bootstrap_index").size()
    if sample_counts.size != n_bootstrap or sample_counts.nunique() != 1:
        raise ValueError("Bootstrap position samples do not have a stable node count per run.")
    node_count = int(sample_counts.iloc[0])
    if lonlat_counts.size != n_bootstrap or not (lonlat_counts == node_count).all():
        raise ValueError("Bootstrap lon/lat samples do not match position sample counts.")

    sample_labels = set(positions["label"].astype(str))
    if set(lonlat["label"].astype(str)) != sample_labels:
        raise ValueError("Bootstrap position and lon/lat node labels differ.")
    if set(ellipses["label"].astype(str)) != sample_labels:
        raise ValueError("Bootstrap ellipse labels do not match the sampled nodes.")
    if len(ellipses) != node_count * 3 or set(ellipses["confidence_level"].astype(float)) != {0.85, 0.9, 0.95}:
        raise ValueError("Bootstrap ellipse summary must contain 85%, 90%, and 95% rows per node.")
    if set(kde_status["label"].astype(str)) != sample_labels or len(kde_status) != node_count:
        raise ValueError("Bootstrap KDE status does not contain exactly one row per node.")

    calibration_labels = [str(label) for label in config.get("calibration_labels", [])]
    if len(calibration_labels) != 3 or config.get("anchor_align_label") not in calibration_labels:
        raise ValueError("Bootstrap output must use three calibration anchors including anchor_align.")
    drift_labels = set(anchor_drift["label"].astype(str))
    if drift_labels != set(calibration_labels):
        raise ValueError("Bootstrap anchor-drift rows do not match calibration anchors.")
    if float(anchor_drift["max_anchor_drift_sim"].max()) > 1e-6:
        raise ValueError("Bootstrap calibration anchors drifted beyond tolerance.")
    kde_by_label = kde_status.set_index("label")["kde_status"].astype(str).to_dict()
    if any(kde_by_label.get(label) != "degenerate" for label in calibration_labels):
        raise ValueError("Fixed calibration anchors must be marked as degenerate in KDE output.")
    non_anchor_status = kde_status[~kde_status["label"].astype(str).isin(calibration_labels)]["kde_status"]
    if not (non_anchor_status.astype(str) == "ok").all():
        raise ValueError("Formal bootstrap KDE contains failed non-anchor distributions.")

    finite_columns = {
        "run parameters": (runs, sorted(required_run_columns)),
        "position samples": (positions, ["x_y_up_sim", "y_y_up_sim"]),
        "lon/lat samples": (lonlat, ["lon", "lat"]),
        "ellipse summary": (
            ellipses,
            [
                "mean_x_y_up_sim",
                "mean_y_y_up_sim",
                "cov_xx",
                "cov_xy",
                "cov_yy",
                "ellipse_width_sim",
                "ellipse_height_sim",
                "ellipse_angle_deg",
            ],
        ),
    }
    for label, (frame, columns) in finite_columns.items():
        if not np.isfinite(frame[columns].to_numpy(float)).all():
            raise ValueError(f"Bootstrap {label} contains non-finite values.")
    return config


def _validate_bfgs_source(bfgs_outdir: Path) -> dict:
    config_path = bfgs_outdir / "bfgs_experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing formal BFGS config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(bfgs_outdir / "bfgs_runs_by_seed.csv")
    expected_seeds = set(range(100))
    if set(runs["seed"].astype(int)) != expected_seeds or len(runs) != 100:
        raise ValueError("Formal BFGS source must contain exactly seeds 0-99.")
    if (runs["status"] != "ok").any() or int(config.get("failure_count", -1)) != 0:
        raise ValueError("Formal BFGS source contains failed runs.")
    if not np.isfinite(runs[[*CORE_METRICS, "objective_final", "gradient_norm_inf"]].to_numpy(float)).all():
        raise ValueError("Formal BFGS source contains non-finite numerical results.")
    if not np.allclose(
        [float(config.get("alpha", np.nan)), float(config.get("beta", np.nan))],
        [0.5, -0.5],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Formal BFGS source is not the selected alpha=0.5, beta=-0.5 run.")
    return config


def _export_tables(as_outdir: Path, outdir: Path) -> None:
    export_random_layout_summary(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_progressive_chain_summary(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distonly_vs_distdir_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distdir_vs_distdiranch_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distdiranch_vs_full_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_smacof_vs_distonly_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_dc_smacof_vs_distdir_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_overall_model_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_smacof_dc_smacof_baseline_full_statistics(as_outdir=as_outdir, outdir=outdir, overwrite=True)


def _write_readme(
    paper_results: Path,
    as_outdir: Path,
    dc_hpo_outdir: Path,
    bfgs_outdir: Path,
    bootstrap_outdir: Path | None = None,
) -> Path:
    def display_path(value: Path) -> str:
        resolved = value.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(resolved)

    path = paper_results / "README.md"
    lines = [
                "# Current Paper Results Snapshot",
                "",
                f"Snapshot date: {date.today().isoformat()}",
                "",
                "This directory contains the current numerical results used for the paper.",
                "Copied experiment files are preserved byte-for-byte; paper tables and metric",
                "figures are regenerated from the formal Progressive AS output.",
                "",
                "## Formal Sources",
                "",
                f"- Progressive AS: `{display_path(as_outdir)}`",
                f"- DC-SMACOF HPO: `{display_path(dc_hpo_outdir)}`",
                f"- HPO-selected SciPy-BFGS: `{display_path(bfgs_outdir)}`",
    ]
    if bootstrap_outdir is not None:
        lines.append(f"- PhysicsSim positional stability: `{display_path(bootstrap_outdir)}`")
    lines.extend(
            [
                "",
                "## Current Selected Hyperparameters",
                "",
                "- PhysicsSim: alpha = 1.0, beta = -0.5.",
                "- DC-SMACOF: alpha = -2.0, w_weight = 1.0, v_weight = 0.01.",
                "- DC-SMACOF direction target: Wang current pairwise distance.",
                "- Repeated direction observations: vector consensus by undirected pair.",
                "- SciPy-BFGS: alpha = 0.5, beta = -0.5, selected by its independent HPO.",
                "",
                "## Formal Progressive AS",
                "",
                "- Physics variants and baselines use seeds 0-99.",
                "- Random+Align uses 1,000 layouts.",
                "- The held-out evaluation set contains eight site-position test points.",
                "- All 1,600 runs completed successfully.",
                "",
                "## Visualization Status",
                "",
                "Root-level metric table figures are regenerated from the current tables.",
                "Representative Section 6.5 visualization subdirectories predate this rerun and",
                "must be regenerated before they are used for revised DC-SMACOF claims.",
                "",
            ]
    )
    if bootstrap_outdir is not None:
        lines.extend(
            [
                "## Positional Stability Outputs",
                "",
                "The `07_bootstrap_stability` directory contains the validated formal",
                "PhysicsSim parameter-perturbation repeated-simulation outputs and figures.",
                "It is not an observation-resampling bootstrap analysis.",
                "",
            ]
        )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return path


def _refresh_manifest(
    *,
    paper_results: Path,
    source_map: dict[str, str],
    previous_sources: dict[str, str],
    derived_source: str,
) -> None:
    manifest_path = paper_results / "manifest_sha256.csv"
    rows = []
    for path in sorted(paper_results.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        snapshot = path.relative_to(paper_results).as_posix()
        if snapshot in source_map:
            source = source_map[snapshot]
        elif snapshot.startswith(("05_paper_tables/", "Result chapter table/")):
            source = derived_source
        elif snapshot.startswith("06_paper_figures/") and path.parent == paper_results / "06_paper_figures":
            source = derived_source
        elif snapshot == "README.md" or snapshot.endswith("summary_statistics_metadata.json"):
            source = derived_source
        else:
            source = previous_sources.get(snapshot, "retained from previous paper_results snapshot")
        rows.append(
            {
                "snapshot_path": snapshot,
                "source_path": source,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snapshot_path", "source_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def update_paper_results(
    *,
    as_outdir: Path,
    dc_hpo_outdir: Path,
    bfgs_outdir: Path,
    paper_results: Path,
    bootstrap_outdir: Path | None = None,
) -> None:
    config = _validate_formal_sources(as_outdir, dc_hpo_outdir)
    _validate_bfgs_source(bfgs_outdir)
    if bootstrap_outdir is not None:
        _validate_bootstrap_source(bootstrap_outdir, config)
    paper_results.mkdir(parents=True, exist_ok=True)
    manifest_path = paper_results / "manifest_sha256.csv"
    previous_sources: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            previous_sources = {
                row["snapshot_path"].replace("\\", "/"): row["source_path"]
                for row in csv.DictReader(handle)
            }
    source_map: dict[str, str] = {}

    for source in sorted(dc_hpo_outdir.iterdir()):
        if source.is_file():
            _copy_file(source, paper_results / "02_dc_smacof_hpo" / source.name, source_map, paper_results)
    for name in AS_MAIN_FILES:
        _copy_file(as_outdir / name, paper_results / "03_progressive_as_main" / name, source_map, paper_results)
    for name in AS_SUPPLEMENTARY_FILES:
        _copy_file(as_outdir / name, paper_results / "04_progressive_as_supplementary" / name, source_map, paper_results)
    bfgs_destination = paper_results / "08_scipy_bfgs"
    _clear_flat_directory(bfgs_destination)
    for name in BFGS_FORMAL_FILES:
        _copy_file(bfgs_outdir / name, bfgs_destination / name, source_map, paper_results)
    if bootstrap_outdir is not None:
        bootstrap_destination = paper_results / "07_bootstrap_stability"
        _clear_flat_directory(bootstrap_destination)
        for name in BOOTSTRAP_FORMAL_FILES:
            _copy_file(bootstrap_outdir / name, bootstrap_destination / name, source_map, paper_results)

    provenance_sources = {
        PROJECT_ROOT / "library/config.py": paper_results / "00_provenance/code_configuration/config.py",
        PROJECT_ROOT / "MDS_model/directed_mds_model.py": paper_results / "00_provenance/code_configuration/directed_mds_model.py",
        PROJECT_ROOT / "library/model_cmp.py": paper_results / "00_provenance/code_configuration/model_cmp.py",
        PROJECT_ROOT / "run_paper_script/ch5_ablation_progressive.py": paper_results / "00_provenance/code_configuration/ch5_ablation_progressive.py",
        PROJECT_ROOT / "run_paper_script/ch5_dc_smacof_hparam.py": paper_results / "00_provenance/code_configuration/ch5_dc_smacof_hparam.py",
        PROJECT_ROOT / "library/scipy_objective.py": paper_results / "00_provenance/code_configuration/scipy_objective.py",
        PROJECT_ROOT / "library/scipy_minimizer.py": paper_results / "00_provenance/code_configuration/scipy_minimizer.py",
        PROJECT_ROOT / "run_paper_script/ch5_scipy_bfgs.py": paper_results / "00_provenance/code_configuration/ch5_scipy_bfgs.py",
        PROJECT_ROOT / "data/direction_edges_verified.csv": paper_results / "00_provenance/input_data/direction_edges_verified.csv",
        PROJECT_ROOT / "data/distance_edges_verified.csv": paper_results / "00_provenance/input_data/distance_edges_verified.csv",
        PROJECT_ROOT / "data/site_rmse_points.csv": paper_results / "00_provenance/input_data/site_rmse_points.csv",
    }
    if bootstrap_outdir is not None:
        provenance_sources.update(
            {
                PROJECT_ROOT / "library/bootstrap_and_visualization.py": paper_results / "00_provenance/code_configuration/bootstrap_and_visualization.py",
                PROJECT_ROOT / "run_paper_script/ch5_bootstrap_stability.py": paper_results / "00_provenance/code_configuration/ch5_bootstrap_stability.py",
            }
        )
    for source, destination in provenance_sources.items():
        _copy_file(source, destination, source_map, paper_results)

    metadata_path = paper_results / "03_progressive_as_main/summary_statistics_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_runs": "04_progressive_as_supplementary/progressive_runs_by_seed.csv",
                "summary_files": ["progressive_summary.csv", "random_align_summary.csv"],
                "std_definition": "sample standard deviation (ddof=1; n=1 uses 0.0)",
                "se_definition": "sample SD / sqrt(n)",
                "ci95_definition": "2,000-resample percentile bootstrap confidence interval for the mean",
                "model_simulation_rerun": True,
                "formal_as_source": str(as_outdir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table_dir = paper_results / "05_paper_tables"
    with tempfile.TemporaryDirectory(prefix="paper_supporting_tables_") as tmp:
        supporting_table_dir = Path(tmp)
        _export_tables(as_outdir, supporting_table_dir)
        export_figures(table_dir=supporting_table_dir, outdir=paper_results / "06_paper_figures")
        _clear_flat_directory(table_dir)
        export_result_chapter_tables(
            paper_table_dir=supporting_table_dir,
            outdir=table_dir,
            overwrite=True,
        )
        export_physics_bfgs_comparison(
            as_outdir=as_outdir,
            bfgs_outdir=bfgs_outdir,
            outdir=table_dir,
            overwrite=True,
        )
    _remove_flat_directory(paper_results / "Result chapter table")
    _write_readme(paper_results, as_outdir, dc_hpo_outdir, bfgs_outdir, bootstrap_outdir)
    _refresh_manifest(
        paper_results=paper_results,
        source_map=source_map,
        previous_sources=previous_sources,
        derived_source=f"derived from {as_outdir.resolve()} and {bfgs_outdir.resolve()}",
    )

    written_config = json.loads(
        (paper_results / "03_progressive_as_main/progressive_config.json").read_text(encoding="utf-8")
    )
    if written_config != config:
        raise RuntimeError("Copied Progressive AS config changed during paper-results refresh.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh paper_results/current from formal experiment outputs.")
    parser.add_argument("--as-outdir", default=str(DEFAULT_AS_OUTDIR))
    parser.add_argument("--dc-hpo-outdir", default=str(DEFAULT_DC_HPO_OUTDIR))
    parser.add_argument("--bfgs-outdir", default=str(DEFAULT_BFGS_OUTDIR))
    parser.add_argument("--paper-results", default=str(DEFAULT_PAPER_RESULTS))
    parser.add_argument(
        "--bootstrap-outdir",
        default=None,
        help="Optional validated formal (>=300-run) PhysicsSim positional-stability output.",
    )
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()
    if not args.confirm_overwrite:
        raise SystemExit("Refusing to update paper_results without --confirm-overwrite.")
    update_paper_results(
        as_outdir=Path(args.as_outdir),
        dc_hpo_outdir=Path(args.dc_hpo_outdir),
        bfgs_outdir=Path(args.bfgs_outdir),
        paper_results=Path(args.paper_results),
        bootstrap_outdir=Path(args.bootstrap_outdir) if args.bootstrap_outdir else None,
    )
    print(f"[Updated] {Path(args.paper_results)}")


if __name__ == "__main__":
    main()
