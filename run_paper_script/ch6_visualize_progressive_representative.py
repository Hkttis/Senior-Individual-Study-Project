"""Create verified representative visualizations from formal progressive AS outputs.

The script selects one representative seed per requested variant using the
recorded AS metrics, reruns that exact configuration, verifies all metrics and
final positions, then exports an error map and a test-only ground-truth overlay.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, OUTPUT_DIR, refer_pos_screen, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.units import data_Li2sim
from library.visualization import ground_truth_comparison, visualize_error_map_official
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from run_paper_script.ch5_ablation_progressive import (
    PHYSICS_VARIANTS,
    _evaluate,
    _run_dc_smacof,
    _run_physics,
    _run_random,
    _run_smacof,
    _target_positions_sim,
)


SELECTION_METRICS = ("E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_test_km")
DEFAULT_VARIANTS = (
    "PhysicsSim-DistOnly",
    "PhysicsSim-DistDir",
    "PhysicsSim-DistDirAnch",
    "PhysicsSim-Full",
    "SMACOF",
    "DC-SMACOF",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progressive-outdir",
        default="outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-0.5_100seeds_random1000",
        help="Formal progressive AS output directory.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/ch6_progressive_representative",
        help="New directory for representative figures and verification records.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated AS variants to visualize.",
    )
    parser.add_argument("--include-random", action="store_true", help="Also visualize one Random+Align representative run.")
    parser.add_argument("--skip-errormap", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--no-wait", action="store_true", help="Save Pygame figures without waiting for a window close.")
    parser.add_argument("--verify-abs-tol", type=float, default=1e-6)
    parser.add_argument("--verify-rel-tol", type=float, default=1e-6)
    return parser.parse_args()


def _select_representative_seed(group: pd.DataFrame) -> dict:
    ok = group[group["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful AS rows for variant={group['variant'].iloc[0]!r}")
    metrics = ok.loc[:, SELECTION_METRICS].astype(float)
    median = metrics.median()
    mad = (metrics - median).abs().median().replace(0.0, 1.0)
    distance = np.sqrt((((metrics - median) / mad) ** 2).sum(axis=1))
    row = ok.loc[distance.idxmin()]
    return {
        "variant": str(row["variant"]),
        "seed": int(row["seed"]),
        "selection_metrics": {metric: float(row[metric]) for metric in SELECTION_METRICS},
        "median_vector": {metric: float(median[metric]) for metric in SELECTION_METRICS},
        "mad_vector": {metric: float(mad[metric]) for metric in SELECTION_METRICS},
        "standardized_distance": float(distance.loc[row.name]),
    }


def _verify_rerun(recorded: pd.Series, rerun_metrics: dict, *, abs_tol: float, rel_tol: float) -> list[dict]:
    rows = []
    failed = []
    for metric in SELECTION_METRICS:
        expected = float(recorded[metric])
        actual = float(rerun_metrics[metric])
        abs_diff = abs(actual - expected)
        rel_diff = abs_diff / max(abs(expected), 1e-12)
        ok = abs_diff <= abs_tol or rel_diff <= rel_tol
        rows.append({
            "metric": metric,
            "as_metric": expected,
            "rerun_metric": actual,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "ok": bool(ok),
        })
        if not ok:
            failed.append(metric)
    if failed:
        raise ValueError(
            f"Rerun metrics do not match formal AS for variant={recorded['variant']!r}, "
            f"seed={int(recorded['seed'])}: {failed}"
        )
    return rows


def _flip_y_up_for_display(points_y_up, anchor_index: int):
    target_x, target_y = map(float, refer_pos_screen)
    points = np.asarray(points_y_up, dtype=float)
    flipped = points.copy()
    flipped[:, 1] = 2.0 * target_y - flipped[:, 1]
    flipped += np.asarray([target_x, target_y]) - flipped[anchor_index]
    return flipped.tolist()


def _rerun_variant(
    variant,
    seed,
    config,
    graph,
    vertice,
    dni,
    edges,
    distance_data,
    calibration_labels,
    calibration_lonlat,
    anchor_label,
    targets,
):
    rp_sim = config.get("refer_pos_sim", refer_pos_sim)
    if variant in PHYSICS_VARIANTS:
        _v, _d, points, _extras = _run_physics(
            PHYSICS_VARIANTS[variant],
            seed,
            calibration_labels,
            calibration_lonlat,
            anchor_label,
            rp_sim,
            float(config["alpha"]),
            float(config["beta"]),
        )
        return points
    if variant == "SMACOF":
        return _run_smacof(seed, graph, vertice, dni, edges, calibration_labels, calibration_lonlat, anchor_label, rp_sim)
    if variant == "DC-SMACOF":
        dc_params = config["dc_smacof_hpo"]
        return _run_dc_smacof(seed, vertice, dni, anchor_label, rp_sim, dc_params)
    if variant == "Random+Align":
        points, _attempts = _run_random(seed, vertice, dni, calibration_labels, targets, anchor_label, rp_sim)
        return points
    raise ValueError(f"Unsupported progressive AS variant: {variant!r}")


def main():
    args = _parse_args()
    as_outdir = Path(args.progressive_outdir)
    config_path = as_outdir / "progressive_config.json"
    runs_path = as_outdir / "progressive_runs_by_seed.csv"
    positions_path = as_outdir / "progressive_final_positions_y_up_sim.csv"
    for required in (config_path, runs_path, positions_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing formal progressive AS file: {required}")

    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    calibration_labels = list(config.get("calibration_labels") or get_anchor_labels())
    anchor_label = str(config.get("anchor_align_label") or get_anchor_align_label())
    test_labels = list(config.get("test_labels") or get_test_site_labels())
    if anchor_label not in calibration_labels:
        raise ValueError("Progressive AS configuration does not include anchor_align in calibration_labels.")
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, config.get("refer_pos_sim", refer_pos_sim))
    data_sim = data_Li2sim(distance_data)
    directional_data = uploading_directional_data()
    runs = pd.read_csv(runs_path)
    saved_positions = pd.read_csv(positions_path)

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if args.include_random:
        variants.append("Random+Align")
    variants = list(dict.fromkeys(variants))
    missing_variants = sorted(set(variants) - set(runs["variant"]))
    if missing_variants:
        raise ValueError(f"Requested variants are absent from progressive AS results: {missing_variants}")

    verification_rows = []
    selections = []
    for variant in variants:
        group = runs[runs["variant"] == variant]
        selection = _select_representative_seed(group)
        seed = selection["seed"]
        recorded = group[(group["seed"] == seed) & (group["status"] == "ok")]
        if len(recorded) != 1:
            raise ValueError(f"Expected one successful recorded row for {variant=} {seed=}")
        recorded = recorded.iloc[0]
        points_y_up = _rerun_variant(
            variant, seed, config, graph, vertice, dni, edges, distance_data,
            calibration_labels, calibration_lonlat, anchor_label, targets,
        )
        rerun_metrics = _evaluate(
            variant, seed, points_y_up, vertice, dni, data_sim, directional_data,
            test_labels, targets, distance_data,
        )
        metric_rows = _verify_rerun(
            recorded, rerun_metrics, abs_tol=args.verify_abs_tol, rel_tol=args.verify_rel_tol,
        )
        for row in metric_rows:
            row.update({"variant": variant, "seed": seed})
        verification_rows.extend(metric_rows)

        saved = saved_positions[(saved_positions["variant"] == variant) & (saved_positions["seed"] == seed)]
        if len(saved) != len(vertice):
            raise ValueError(f"Saved AS positions are incomplete for {variant=} {seed=}")
        saved_by_label = saved.set_index("label").loc[vertice]
        saved_matrix = saved_by_label[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        rerun_matrix = np.asarray(points_y_up, dtype=float)
        max_position_delta = float(np.max(np.abs(saved_matrix - rerun_matrix)))
        if not np.allclose(saved_matrix, rerun_matrix, atol=args.verify_abs_tol, rtol=args.verify_rel_tol):
            raise ValueError(f"Rerun final positions do not match saved AS positions for {variant=} {seed=}")

        points_y_down = _flip_y_up_for_display(points_y_up, dni[anchor_label])
        safe_variant = variant.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("+", "plus")
        prefix = f"progressive_AS_{safe_variant}_seed{seed}_"
        wrong_dir = wrong_directions_nonflip(deepcopy(points_y_up), vertice, dni)
        if not args.skip_errormap:
            visualize_error_map_official(
                deepcopy(points_y_down), vertice, dni, distance_data, wrong_dir,
                file_name=prefix, wait=not args.no_wait, output_dir=outdir,
                title=f"Error Map: {variant} (representative seed {seed})",
            )
        overlay = None
        if not args.skip_overlay:
            overlay = ground_truth_comparison(
                vertice, dni, data_sim, deepcopy(gt_lonlat), points_y_down[dni[anchor_label]],
                deepcopy(points_y_down), prefix, wait=not args.no_wait,
                eval_labels=test_labels, output_dir=outdir,
                title=f"Ground-Truth Overlap: {variant} (representative seed {seed})",
            )
            if not np.isclose(overlay["rmse_km"], rerun_metrics["RMSE_test_km"], rtol=args.verify_rel_tol, atol=args.verify_abs_tol):
                raise ValueError(
                    f"Test-only overlay RMSE does not match AS RMSE for {variant=} {seed=}: "
                    f"{overlay['rmse_km']} != {rerun_metrics['RMSE_test_km']}"
                )

        selection.update({
            "max_final_position_delta_sim": max_position_delta,
            "rerun_metrics": {metric: float(rerun_metrics[metric]) for metric in SELECTION_METRICS},
            "test_rmse_overlay_km": None if overlay is None else float(overlay["rmse_km"]),
            "test_labels": test_labels,
            "error_map_file": None if args.skip_errormap else f"{prefix}error_map_full.png",
            "ground_truth_overlay_file": None if args.skip_overlay else f"{prefix}Overlap.png",
        })
        selections.append(selection)

    pd.DataFrame(verification_rows).to_csv(outdir / "representative_rerun_verification.csv", index=False, encoding="utf-8-sig")
    (outdir / "representative_selection.json").write_text(
        json.dumps({"source_progressive_as": str(as_outdir), "selections": selections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Saved] {outdir / 'representative_selection.json'}")
    print(f"[Saved] {outdir / 'representative_rerun_verification.csv'}")


if __name__ == "__main__":
    main()
