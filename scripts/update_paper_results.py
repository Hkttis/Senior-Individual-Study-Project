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


DEFAULT_AS_OUTDIR = (
    PROJECT_ROOT
    / "outputs"
    / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
)
DEFAULT_DC_HPO_OUTDIR = (
    PROJECT_ROOT / "outputs" / "ch5_dc_smacof_hparam_wang_current_alpha_-4_0_seed0_9_20260721"
)
DEFAULT_PAPER_RESULTS = PROJECT_ROOT / "paper_results" / "current"

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
CORE_METRICS = ["E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_test_km"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _copy_file(source: Path, destination: Path, source_map: dict[str, str], paper_results: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing paper-results source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_map[destination.relative_to(paper_results).as_posix()] = str(source.resolve())


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


def _write_readme(paper_results: Path, as_outdir: Path, dc_hpo_outdir: Path) -> Path:
    path = paper_results / "README.md"
    path.write_text(
        "\n".join(
            [
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
                f"- Progressive AS: `{as_outdir.relative_to(PROJECT_ROOT).as_posix()}`",
                f"- DC-SMACOF HPO: `{dc_hpo_outdir.relative_to(PROJECT_ROOT).as_posix()}`",
                "",
                "## Current Selected Hyperparameters",
                "",
                "- PhysicsSim: alpha = 1.0, beta = -0.5.",
                "- DC-SMACOF: alpha = -2.0, w_weight = 1.0, v_weight = 0.01.",
                "- DC-SMACOF direction target: Wang current pairwise distance.",
                "- Repeated direction observations: vector consensus by undirected pair.",
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
        ),
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


def update_paper_results(*, as_outdir: Path, dc_hpo_outdir: Path, paper_results: Path) -> None:
    config = _validate_formal_sources(as_outdir, dc_hpo_outdir)
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

    provenance_sources = {
        PROJECT_ROOT / "library/config.py": paper_results / "00_provenance/code_configuration/config.py",
        PROJECT_ROOT / "MDS_model/directed_mds_model.py": paper_results / "00_provenance/code_configuration/directed_mds_model.py",
        PROJECT_ROOT / "library/model_cmp.py": paper_results / "00_provenance/code_configuration/model_cmp.py",
        PROJECT_ROOT / "run_paper_script/ch5_ablation_progressive.py": paper_results / "00_provenance/code_configuration/ch5_ablation_progressive.py",
        PROJECT_ROOT / "run_paper_script/ch5_dc_smacof_hparam.py": paper_results / "00_provenance/code_configuration/ch5_dc_smacof_hparam.py",
        PROJECT_ROOT / "data/direction_edges_verified.csv": paper_results / "00_provenance/input_data/direction_edges_verified.csv",
        PROJECT_ROOT / "data/distance_edges_verified.csv": paper_results / "00_provenance/input_data/distance_edges_verified.csv",
        PROJECT_ROOT / "data/site_rmse_points.csv": paper_results / "00_provenance/input_data/site_rmse_points.csv",
    }
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
    _remove_flat_directory(paper_results / "Result chapter table")
    _write_readme(paper_results, as_outdir, dc_hpo_outdir)
    _refresh_manifest(
        paper_results=paper_results,
        source_map=source_map,
        previous_sources=previous_sources,
        derived_source=f"derived from {as_outdir.resolve()}",
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
    parser.add_argument("--paper-results", default=str(DEFAULT_PAPER_RESULTS))
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()
    if not args.confirm_overwrite:
        raise SystemExit("Refusing to update paper_results without --confirm-overwrite.")
    update_paper_results(
        as_outdir=Path(args.as_outdir),
        dc_hpo_outdir=Path(args.dc_hpo_outdir),
        paper_results=Path(args.paper_results),
    )
    print(f"[Updated] {Path(args.paper_results)}")


if __name__ == "__main__":
    main()
