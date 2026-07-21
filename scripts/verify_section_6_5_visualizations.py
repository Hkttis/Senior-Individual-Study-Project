"""Verify and snapshot Section 6.5 visualization outputs.

The checker validates that the visualization metadata, plotted representative
seeds, saved coordinates, and displayed metrics are all consistent with the
formal progressive AS results. It can also copy the checked visualization files
into paper_results/current and refresh the snapshot manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim
from scripts.create_section_6_5_visual_prototype import (
    DEFAULT_VARIANTS,
    _distance_edge_errors,
    _load_position_matrix,
    _wrong_direction_nodes,
)


DEFAULT_AS_OUTDIR = PROJECT_ROOT / "outputs" / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-0.5_100seeds_random1000"
DEFAULT_REPRESENTATIVE_DIR = PROJECT_ROOT / "outputs" / "ch6_section_6_5_full_smacof_dc_representative"
DEFAULT_VIS_DIR = PROJECT_ROOT / "outputs" / "ch6_section_6_5_visual_prototype"
DEFAULT_PAPER_RESULTS = PROJECT_ROOT / "paper_results" / "current"

CORE_METRICS = ("RMSE_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae")
VIS_FILES = (
    "section_6_5_three_model_visualization_prototype.png",
    "section_6_5_three_model_visualization_prototype.svg",
    "section_6_5_overlay_large.png",
    "section_6_5_overlay_large.svg",
    "section_6_5_error_map_large.png",
    "section_6_5_error_map_large.svg",
    "section_6_5_three_model_visualization_prototype.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _norm_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _assert_close(label: str, actual: float, expected: float, failures: list[str], *, atol: float, rtol: float) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=rtol):
        failures.append(f"{label}: {actual} != {expected}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sources(as_outdir: Path, representative_dir: Path, vis_dir: Path, failures: list[str]) -> tuple[dict, dict]:
    vis_meta = _load_json(vis_dir / "section_6_5_three_model_visualization_prototype.json")
    rep_meta = _load_json(representative_dir / "representative_selection.json")
    if _norm_path(vis_meta["source_progressive_as"]) != as_outdir.resolve():
        failures.append("visualization metadata source_progressive_as does not match --as-outdir")
    if _norm_path(vis_meta["source_representative_dir"]) != representative_dir.resolve():
        failures.append("visualization metadata source_representative_dir does not match --representative-dir")
    if _norm_path(rep_meta["source_progressive_as"]) != as_outdir.resolve():
        failures.append("representative metadata source_progressive_as does not match --as-outdir")
    for name in VIS_FILES:
        path = vis_dir / name
        if not path.exists():
            failures.append(f"missing visualization output: {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty visualization output: {path}")
    return vis_meta, rep_meta


def verify_section_6_5_visualizations(
    *,
    as_outdir: Path,
    representative_dir: Path,
    vis_dir: Path,
    expected_variants: list[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-9,
) -> tuple[list[str], dict]:
    failures: list[str] = []
    as_outdir = as_outdir.resolve()
    representative_dir = representative_dir.resolve()
    vis_dir = vis_dir.resolve()
    vis_meta, rep_meta = _verify_sources(as_outdir, representative_dir, vis_dir, failures)

    variants = list(vis_meta.get("variants", DEFAULT_VARIANTS))
    if expected_variants is not None and variants != expected_variants:
        failures.append(f"unexpected visualization variants: {variants}; expected {expected_variants}")

    rep_by_variant = {row["variant"]: row for row in rep_meta["selections"]}
    metric_rows = pd.read_csv(as_outdir / "progressive_runs_by_seed.csv", encoding="utf-8-sig")
    config = _load_json(as_outdir / "progressive_config.json")
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    del graph, edges
    data_sim = data_Li2sim(distance_data)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    directional_data = uploading_directional_data()
    anchor_label = get_anchor_align_label()
    anchor_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, config.get("refer_pos_sim", refer_pos_sim))

    if anchor_label not in anchor_labels:
        failures.append(f"anchor_align label is not in anchor labels: {anchor_label}")
    for label in anchor_labels + test_labels:
        if label not in targets:
            failures.append(f"missing projected target position for site label: {label}")

    checks: dict[str, dict] = {}
    for variant in variants:
        if variant not in rep_by_variant:
            failures.append(f"missing representative selection for {variant}")
            continue
        rep = rep_by_variant[variant]
        seed = int(rep["seed"])
        if int(vis_meta["seeds"].get(variant, -1)) != seed:
            failures.append(f"visualization metadata seed mismatch for {variant}")

        source_row = metric_rows[(metric_rows["variant"] == variant) & (metric_rows["seed"] == seed)]
        if len(source_row) != 1:
            failures.append(f"expected one AS metric row for {variant} seed {seed}, got {len(source_row)}")
            continue
        source = source_row.iloc[0].to_dict()

        points = _load_position_matrix(as_outdir, rep, vertice)
        recomputed = _evaluate(
            variant,
            seed,
            points,
            vertice,
            dni,
            data_sim,
            directional_data,
            test_labels,
            targets,
            distance_data,
        )
        for metric in CORE_METRICS:
            _assert_close(f"{variant} representative rerun {metric}", rep["rerun_metrics"][metric], source[metric], failures, atol=atol, rtol=rtol)
            _assert_close(f"{variant} visualization metadata {metric}", vis_meta["metrics"][variant][metric], source[metric], failures, atol=atol, rtol=rtol)
            _assert_close(f"{variant} recomputed {metric}", recomputed[metric], source[metric], failures, atol=atol, rtol=rtol)
        _assert_close(
            f"{variant} overlay RMSE",
            rep["test_rmse_overlay_km"],
            source["RMSE_test_km"],
            failures,
            atol=atol,
            rtol=rtol,
        )

        edge_errors = _distance_edge_errors(points, data_sim, dni)
        wrong_nodes = _wrong_direction_nodes(points, vertice, dni)
        if len(edge_errors) != len(distance_data):
            failures.append(f"{variant} visual edge-error count mismatch: {len(edge_errors)} != {len(distance_data)}")
        if not all(np.isfinite(error) and error >= 0.0 for *_ij, error in edge_errors):
            failures.append(f"{variant} visual edge errors contain non-finite or negative values")
        checks[variant] = {
            "seed": seed,
            "metrics": {metric: float(source[metric]) for metric in CORE_METRICS},
            "n_positions": int(len(points)),
            "n_distance_edges_visualized": int(len(edge_errors)),
            "n_direction_violation_nodes_visualized": int(len(wrong_nodes)),
            "position_source_sha256": _sha256(as_outdir / "progressive_final_positions_y_up_sim.csv"),
        }
    return failures, checks


def _refresh_manifest(paper_results: Path) -> None:
    manifest_path = paper_results / "manifest_sha256.csv"
    source_by_snapshot: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                source_by_snapshot[row["snapshot_path"].replace("\\", "/")] = row.get("source_path", "")
    rows = []
    for path in sorted(paper_results.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        snapshot = path.relative_to(paper_results).as_posix()
        rows.append({
            "snapshot_path": snapshot,
            "source_path": source_by_snapshot.get(snapshot, str(path.resolve())),
            "sha256": _sha256(path),
            "size_bytes": str(path.stat().st_size),
        })
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snapshot_path", "source_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def copy_visualizations_to_paper_results(*, vis_dir: Path, paper_results: Path, paper_subdir: str = "section_6_5_visualizations") -> Path:
    destination = paper_results / "06_paper_figures" / paper_subdir
    destination.mkdir(parents=True, exist_ok=True)
    for name in VIS_FILES:
        shutil.copy2(vis_dir / name, destination / name)
    _refresh_manifest(paper_results)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-outdir", default=str(DEFAULT_AS_OUTDIR))
    parser.add_argument("--representative-dir", default=str(DEFAULT_REPRESENTATIVE_DIR))
    parser.add_argument("--vis-dir", default=str(DEFAULT_VIS_DIR))
    parser.add_argument("--paper-results", default=str(DEFAULT_PAPER_RESULTS))
    parser.add_argument("--expected-variants", default="", help="Comma-separated expected variants. Empty means accept the visualization metadata variants.")
    parser.add_argument("--paper-subdir", default="section_6_5_visualizations", help="Subdirectory under paper_results/current/06_paper_figures for copied outputs.")
    parser.add_argument("--no-copy", action="store_true", help="Only verify consistency; do not copy files into paper_results/current.")
    args = parser.parse_args()
    expected_variants = [value.strip() for value in args.expected_variants.split(",") if value.strip()] or None

    failures, checks = verify_section_6_5_visualizations(
        as_outdir=Path(args.as_outdir),
        representative_dir=Path(args.representative_dir),
        vis_dir=Path(args.vis_dir),
        expected_variants=expected_variants,
    )
    if failures:
        print("[FAIL] Section 6.5 visualization consistency check failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    report_path = Path(args.vis_dir) / "section_6_5_visualization_consistency_report.json"
    report_path.write_text(json.dumps({"status": "ok", "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Section 6.5 visualizations are consistent with formal model results.")
    print(f"[Saved] {report_path}")
    if not args.no_copy:
        destination = copy_visualizations_to_paper_results(
            vis_dir=Path(args.vis_dir),
            paper_results=Path(args.paper_results),
            paper_subdir=args.paper_subdir,
        )
        print(f"[Copied] {destination}")
        print(f"[Updated] {Path(args.paper_results) / 'manifest_sha256.csv'}")


if __name__ == "__main__":
    main()
