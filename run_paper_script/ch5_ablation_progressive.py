"""Progressive-information ablation study with Random+Align null baseline."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    Li2km,
    km2pix,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters, lcc_transformation_with_anchor
from library.initialization import generate_CHEN_initial_positions
from library.metrics import (
    alignment_and_scaling,
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
    procrustes_align_by_fixed_points,
)
from library.model_cmp import get_dc_smacof_direction_method_metadata, run_directed_MDS
from library.physics import main_physics_simulation
from library.progressive_alignment import (
    anchored_similarity_procrustes,
    place_in_anchor_frame,
    sample_non_degenerate_unit_square_layout,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from MDS_model.stress_majorization_mds_model import stress_majorization
from run_paper_script.ch5_ablation_study import (
    _bootstrap_ci_mean,
    _dc_smacof_weights_from_alpha,
    _load_selected_dc_smacof_params,
    _load_selected_hpo_params,
    _series_stats,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _weights_from_alpha_beta
from scripts.evaluate_repulsion_layout import (
    TAU_MULTIPLIERS,
    _convex_hull_area,
    _gini,
    _nearest_neighbor_distances_km,
    _pairwise_distances_km,
    _topology_metrics,
)


PHYSICS_VARIANTS = {
    "PhysicsSim-DistOnly": {"direction": False, "anchors": False, "repulsion": False, "procrustes": True},
    "PhysicsSim-DistDir": {"direction": True, "anchors": False, "repulsion": False, "procrustes": False},
    "PhysicsSim-DistDirAnch": {"direction": True, "anchors": True, "repulsion": False, "procrustes": False},
    "PhysicsSim-Full": {"direction": True, "anchors": True, "repulsion": True, "procrustes": False},
}
PAIRED_COMPARISONS = [
    ("PhysicsSim-DistDir", "PhysicsSim-DistOnly", "direction_given_distance"),
    ("PhysicsSim-DistDirAnch", "PhysicsSim-DistDir", "optimization_anchors_given_distance_direction"),
    ("PhysicsSim-Full", "PhysicsSim-DistDirAnch", "repulsion_given_distance_direction_anchors"),
]
METRICS = [
    "RMSE_test_km", "MAE_test_km", "median_error_km", "E_distance_stress", "E_direction_vr",
    "E_direction_mae", "crowding_violation_rate_tau_0p1", "collapse_node_rate_tau_0p1",
    "nnd_q05_km", "distance_edge_crossing_rate",
]
LOWER_IS_BETTER = {
    "RMSE_test_km", "MAE_test_km", "median_error_km", "E_distance_stress", "E_direction_vr",
    "E_direction_mae", "crowding_violation_rate_tau_0p1", "collapse_node_rate_tau_0p1",
    "distance_edge_crossing_rate",
}


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_physics_hpo_config(hpo_outdir: str | Path) -> tuple[Path, dict]:
    """Resolve a direct HPO directory or a manual-candidate directory to its HPO config."""
    hpo_path = Path(hpo_outdir)
    direct_config = hpo_path / "gridsearch_config.json"
    if direct_config.exists():
        return direct_config, _load_json(direct_config)

    candidate_path = hpo_path / "selected_candidate_summary.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Neither gridsearch_config.json nor selected_candidate_summary.csv found in {hpo_path}"
        )
    candidate = pd.read_csv(candidate_path)
    if candidate.empty or "source_hpo_outdir" not in candidate.columns:
        raise ValueError(f"Manual PhysicsSim candidate has no source_hpo_outdir: {candidate_path}")
    source = Path(str(candidate.iloc[0]["source_hpo_outdir"]))
    if not source.is_absolute():
        source = Path.cwd() / source
    source_config = source / "gridsearch_config.json"
    return source_config, _load_json(source_config)


def _as_lcc_mapping(value, keys: Sequence[str]) -> dict[str, float]:
    if isinstance(value, dict):
        return {key: float(value[key]) for key in keys}
    if isinstance(value, (list, tuple)) and len(value) == len(keys):
        return {key: float(item) for key, item in zip(keys, value)}
    raise ValueError(f"Invalid LCC provenance value: {value!r}")


def _assert_same_lcc(label: str, recorded_bounds, recorded_params) -> None:
    expected_bounds = _as_lcc_mapping(
        get_lcc_bounds(), ("lon_min", "lon_max", "lat_min", "lat_max")
    )
    expected_params = _as_lcc_mapping(get_lcc_parameters(), ("lat_1", "lat_2", "lon_0"))
    actual_bounds = _as_lcc_mapping(recorded_bounds, ("lon_min", "lon_max", "lat_min", "lat_max"))
    actual_params = _as_lcc_mapping(recorded_params, ("lat_1", "lat_2", "lon_0"))
    for key, expected in {**expected_bounds, **expected_params}.items():
        actual = (actual_bounds | actual_params)[key]
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{label} uses a different LCC parameter for {key}: {actual} != current {expected}. "
                "Do not mix outputs from different projections."
            )


def _validate_input_provenance(hpo_outdir, dc_hpo_outdir, calibration_labels, anchor_label, test_labels) -> dict:
    physics_config_path, physics_config = _resolve_physics_hpo_config(hpo_outdir)
    _assert_same_lcc("PhysicsSim HPO", physics_config["lcc_bounds"], physics_config["lcc_parameters"])
    if list(physics_config.get("anchor_labels", [])) != list(calibration_labels):
        raise ValueError("PhysicsSim HPO anchor labels do not match the current site-point roles.")
    if list(physics_config.get("test_labels", [])) != list(test_labels):
        raise ValueError("PhysicsSim HPO test labels do not match the current site-point roles.")

    dc_config_path = Path(dc_hpo_outdir) / "dc_smacof_hparam_config.json"
    dc_config = _load_json(dc_config_path)
    _assert_same_lcc("DC-SMACOF HPO", dc_config["lcc_bounds"], dc_config["lcc_parameters"])
    if list(dc_config.get("anchor_labels", [])) != list(calibration_labels):
        raise ValueError("DC-SMACOF HPO anchor labels do not match the current site-point roles.")
    if dc_config.get("anchor_align_label") != anchor_label:
        raise ValueError("DC-SMACOF HPO anchor_align label does not match the current site-point roles.")
    return {
        "physics_hpo_config": str(physics_config_path),
        "dc_smacof_hpo_config": str(dc_config_path),
        "lcc_matches_current_data": True,
        "site_roles_match_current_data": True,
    }


def _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos) -> dict[str, np.ndarray]:
    projected = lcc_transformation_with_anchor(dni, gt_lonlat, anchor_label=anchor_label)
    targets = {}
    for label, index in dni.items():
        x_km, y_km = projected[index]
        if x_km is not None and y_km is not None:
            targets[label] = np.asarray([float(refer_pos[0]) + x_km * km2pix, float(refer_pos[1]) + y_km * km2pix])
    return targets


def _physics_forces(spec, alpha, beta):
    w_dir, w_reg, spring, directional, repulsion = _weights_from_alpha_beta(
        alpha, beta, 1.0, SPRING_STIFFNESS_BASE, DIRECTIONAL_FORCE_MAGNITUDE_BASE, REPULSION_STRENGTH_BASE
    )
    return w_dir, w_reg, spring, directional if spec["direction"] else 0.0, repulsion if spec["repulsion"] else 0.0


def _rigid_procrustes(points, calibration_labels, calibration_lonlat, dni, anchor_label, refer_pos):
    return np.asarray(
        procrustes_align_by_fixed_points(
            deepcopy(points), list(calibration_labels), list(calibration_lonlat), dni,
            refer_pos=refer_pos, anchor_label=anchor_label,
        ),
        dtype=float,
    )


def _run_physics(spec, seed, calibration_labels, calibration_lonlat, anchor_label, refer_pos, alpha, beta):
    np.random.seed(seed)
    fixed_labels = list(calibration_labels) if spec["anchors"] else []
    fixed_lonlat = list(calibration_lonlat) if spec["anchors"] else []
    vertice, dni, data_li, initial, fixed_positions = generate_CHEN_initial_positions(
        list(refer_pos), fixed_labels, fixed_lonlat, anchor_label=anchor_label
    )
    _w_dir, _w_reg, spring, directional, repulsion = _physics_forces(spec, alpha, beta)
    _wrong, _history, _positions, final = main_physics_simulation(
        vertice, dni, data_Li2sim(data_li), initial, uploading_directional_data(), fixed_positions,
        spring, repulsion, directional, plot=False,
    )
    aligned = place_in_anchor_frame(final, dni, anchor_label, refer_pos)
    if spec["procrustes"]:
        aligned = _rigid_procrustes(aligned, calibration_labels, calibration_lonlat, dni, anchor_label, refer_pos)
    return vertice, dni, aligned, {"w_dir": _w_dir, "w_reg": _w_reg, "spring_stiffness": spring,
                                   "directional_force": directional, "repulsion_strength": repulsion}


def _run_smacof(seed, graph, vertice, dni, edges, calibration_labels, calibration_lonlat, anchor_label, refer_pos):
    np.random.seed(seed)
    pos_li, _history, _all = stress_majorization(graph, dni, vertice, edges)
    framed = alignment_and_scaling(pos_li, vertice, dni, refer_pos, y_down=False, anchor_label=anchor_label)
    return _rigid_procrustes(framed, calibration_labels, calibration_lonlat, dni, anchor_label, refer_pos)


def _run_dc_smacof(seed, vertice, dni, anchor_label, refer_pos, dc_params):
    np.random.seed(seed)
    history = run_directed_MDS(vis=False, w_weight_value=dc_params["w_weight"], v_weight_value=dc_params["v_weight"])
    return np.asarray(alignment_and_scaling(history[-1], vertice, dni, refer_pos, y_down=False, anchor_label=anchor_label), dtype=float)


def _run_random(seed, vertice, dni, calibration_labels, target_positions, anchor_label, refer_pos):
    points, attempts = sample_non_degenerate_unit_square_layout(
        len(vertice), dni, calibration_labels, np.random.default_rng(seed)
    )
    return anchored_similarity_procrustes(
        points, dni, calibration_labels, target_positions, anchor_label, refer_pos, allow_scaling=True
    ), attempts


def _layout_metrics(points, vertice, dni, distance_data):
    points_km = np.asarray(pos_matrix_sim2km(points.tolist()), dtype=float)
    pairwise = _pairwise_distances_km(points_km)
    nnd = _nearest_neighbor_distances_km(points_km)
    target_median_km = float(np.median([float(row[2]) * Li2km for row in distance_data]))
    result = {
        "nnd_q05_km": float(np.quantile(nnd, 0.05)),
        "radius_gyration_km": float(np.sqrt(np.mean(np.sum((points_km - points_km.mean(axis=0)) ** 2, axis=1)))),
        "convex_hull_area_km2": _convex_hull_area(points_km),
    }
    for multiplier in TAU_MULTIPLIERS:
        suffix = str(multiplier).replace(".", "p")
        tau = target_median_km * multiplier
        result[f"crowding_violation_rate_tau_{suffix}"] = float(np.mean(pairwise < tau))
        result[f"collapse_node_rate_tau_{suffix}"] = float(np.mean(nnd < tau))
    result.update(_topology_metrics(points_km, vertice, [(row[0], row[1]) for row in distance_data], target_median_km * 0.05))
    result["nnd_cv"] = float(nnd.std(ddof=0) / nnd.mean())
    result["nnd_gini"] = _gini(nnd)
    return result


def _evaluate(variant, seed, points, vertice, dni, data_sim, directional_data, test_labels, targets, distance_data):
    errors = np.asarray([np.linalg.norm(points[dni[label]] - targets[label]) / km2pix for label in test_labels], dtype=float)
    result = {
        "variant": variant, "seed": int(seed), "status": "ok", "error": "",
        "RMSE_test_km": float(np.sqrt(np.mean(errors ** 2))), "MAE_test_km": float(errors.mean()),
        "median_error_km": float(np.median(errors)),
        "E_distance_stress": float(calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)),
        "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
        "E_direction_mae": float(mean_angular_error_violations(points, directional_data, dni)),
    }
    result.update(_layout_metrics(points, vertice, dni, distance_data))
    return result


def _summary(runs):
    rows = []
    for variant, group in runs[runs.status == "ok"].groupby("variant"):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append({
                "variant": variant, "metric": metric, **_series_stats(values),
                "q05": float(np.quantile(values, 0.05)), "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
            })
    return pd.DataFrame(rows)


def _paired(runs):
    rows = []
    for left, right, comparison in PAIRED_COMPARISONS:
        left_df = runs[(runs.variant == left) & (runs.status == "ok")].set_index("seed")
        right_df = runs[(runs.variant == right) & (runs.status == "ok")].set_index("seed")
        seeds = sorted(set(left_df.index).intersection(right_df.index))
        for metric in METRICS:
            diff = left_df.loc[seeds, metric].to_numpy(float) - right_df.loc[seeds, metric].to_numpy(float)
            lo, hi = _bootstrap_ci_mean(diff)
            rows.append({"comparison": comparison, "left_variant": left, "right_variant": right, "metric": metric,
                         "diff_definition": "left_minus_right", "n_pairs": len(diff), "paired_diff_mean": float(diff.mean()),
                         "paired_diff_median": float(np.median(diff)), "paired_diff_ci95_lo": lo, "paired_diff_ci95_hi": hi,
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def _random_percentiles(runs):
    random_rows = runs[(runs.variant == "Random+Align") & (runs.status == "ok")]
    rows = []
    for variant, group in runs[(runs.variant != "Random+Align") & (runs.status == "ok")].groupby("variant"):
        for metric in METRICS:
            observed = group[metric].to_numpy(float)
            null = random_rows[metric].to_numpy(float)
            lower = metric in LOWER_IS_BETTER
            percentile = [float(np.mean(null >= value) if lower else np.mean(null <= value)) for value in observed]
            rows.append({"variant": variant, "metric": metric, "n_model_runs": len(observed), "n_random_runs": len(null),
                         "mean_model_percentile_vs_random": float(np.mean(percentile)), "median_model_percentile_vs_random": float(np.median(percentile))})
    return pd.DataFrame(rows)


def run_progressive_ablation(*, hpo_outdir, dc_hpo_outdir, seeds, random_runs, outdir, include_random=True):
    if include_random and random_runs < 1:
        raise ValueError("--random-runs must be at least 1 when Random+Align is enabled.")
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    alpha, beta = _load_selected_hpo_params(hpo_outdir)
    dc_params = _load_selected_dc_smacof_params(dc_hpo_outdir)
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    calibration_labels, anchor_label, test_labels = get_anchor_labels(), get_anchor_align_label(), get_test_site_labels()
    if len(calibration_labels) != 3 or anchor_label not in calibration_labels:
        raise ValueError("Progressive AS requires three calibration anchors including anchor_align.")
    input_validation = _validate_input_provenance(
        hpo_outdir, dc_hpo_outdir, calibration_labels, anchor_label, test_labels
    )
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, DEFAULT_REFER_POS_SIM)
    data_sim, directional_data = data_Li2sim(distance_data), uploading_directional_data()
    dc_direction_method, _ = get_dc_smacof_direction_method_metadata(directional_data, dni)
    rows, positions = [], []

    def record(variant, seed, points, extras=None):
        row = _evaluate(variant, seed, points, vertice, dni, data_sim, directional_data, test_labels, targets, distance_data)
        row.update(extras or {})
        rows.append(row)
        positions.extend({"variant": variant, "seed": seed, "label": label, "x_y_up_sim": float(points[index, 0]), "y_y_up_sim": float(points[index, 1])} for index, label in enumerate(vertice))

    for seed in seeds:
        for variant, spec in PHYSICS_VARIANTS.items():
            try:
                _v, _d, points, extras = _run_physics(spec, seed, calibration_labels, calibration_lonlat, anchor_label, DEFAULT_REFER_POS_SIM, alpha, beta)
                record(variant, seed, points, extras)
            except Exception as exc:
                rows.append({"variant": variant, "seed": seed, "status": "failed", "error": repr(exc)})
        for variant, runner in (("SMACOF", _run_smacof_baseline), ("DC-SMACOF", _run_dc_smacof_baseline)):
            try:
                points = runner(seed, graph, vertice, dni, edges, calibration_labels, calibration_lonlat, anchor_label, DEFAULT_REFER_POS_SIM) if variant == "SMACOF" else _run_dc_smacof(seed, vertice, dni, anchor_label, DEFAULT_REFER_POS_SIM, dc_params)
                record(variant, seed, points)
            except Exception as exc:
                rows.append({"variant": variant, "seed": seed, "status": "failed", "error": repr(exc)})
    if include_random:
        for seed in range(random_runs):
            try:
                points, attempts = _run_random(seed, vertice, dni, calibration_labels, targets, anchor_label, DEFAULT_REFER_POS_SIM)
                record("Random+Align", seed, points, {"random_rejection_attempts": attempts})
            except Exception as exc:
                rows.append({"variant": "Random+Align", "seed": seed, "status": "failed", "error": repr(exc)})
    runs = pd.DataFrame(rows)
    final_positions = pd.DataFrame(positions)
    summary, paired, percentiles = _summary(runs), _paired(runs), _random_percentiles(runs) if include_random else pd.DataFrame()
    runs.to_csv(outdir / "progressive_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    final_positions.to_csv(outdir / "progressive_final_positions_y_up_sim.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "progressive_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(outdir / "progressive_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    percentiles.to_csv(outdir / "random_align_percentiles.csv", index=False, encoding="utf-8-sig")
    random_rows = runs[runs.variant == "Random+Align"].copy()
    random_summary = summary[summary.variant == "Random+Align"].copy()
    status_summary = runs.groupby(["variant", "status"], dropna=False).size().reset_index(name="n_runs")
    random_rows.to_csv(outdir / "random_align_runs.csv", index=False, encoding="utf-8-sig")
    random_summary.to_csv(outdir / "random_align_summary.csv", index=False, encoding="utf-8-sig")
    status_summary.to_csv(outdir / "progressive_run_status.csv", index=False, encoding="utf-8-sig")
    config = {"hpo_outdir": str(hpo_outdir), "dc_hpo_outdir": str(dc_hpo_outdir), "alpha": alpha, "beta": beta, "dc_smacof_hpo": dc_params, "dc_smacof_direction_method": dc_direction_method, "seeds": list(seeds), "random_runs": random_runs,
              "calibration_labels": calibration_labels, "anchor_align_label": anchor_label, "test_labels": test_labels,
              "lcc_bounds": _as_lcc_mapping(get_lcc_bounds(), ("lon_min", "lon_max", "lat_min", "lat_max")), "lcc_parameters": _as_lcc_mapping(get_lcc_parameters(), ("lat_1", "lat_2", "lon_0")), "lcc_standard_parallel_rule": "lat_1=lat_min+(lat_max-lat_min)/6; lat_2=lat_max-(lat_max-lat_min)/6", "lcc_bounds_source": FILE_PATHS["ground_truth_path"],
              "random_rejection": {"min_anchor_distance_unit_square": 0.05, "min_anchor_triangle_area_unit_square": 0.005}, "physics_variants": PHYSICS_VARIANTS,
              "input_validation": input_validation, "failure_count": int((runs["status"] != "ok").sum()),
              "alignment_protocol": {"Random+Align": "anchor_frame+rotation_reflection_scaling", "PhysicsSim-DistOnly": "anchor_frame+rotation_reflection", "SMACOF": "anchor_frame+rotation_reflection", "PhysicsSim-DistDir": "anchor_frame", "PhysicsSim-DistDirAnch": "anchor_frame", "PhysicsSim-Full": "anchor_frame", "DC-SMACOF": "anchor_frame"}}
    (outdir / "progressive_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if config["failure_count"]:
        raise RuntimeError(
            f"Progressive AS completed with {config['failure_count']} failed runs; outputs were saved but are incomplete."
        )
    return {"runs": runs, "summary": summary, "paired": paired, "outdir": outdir}


def _run_smacof_baseline(seed, graph, vertice, dni, edges, calibration_labels, calibration_lonlat, anchor_label, refer_pos):
    return _run_smacof(seed, graph, vertice, dni, edges, calibration_labels, calibration_lonlat, anchor_label, refer_pos)


def _run_dc_smacof_baseline(*_args, **_kwargs):
    raise AssertionError("DC-SMACOF is dispatched directly to preserve its no-Procrustes contract.")


def main():
    parser = argparse.ArgumentParser(description="Run the progressive-information AS experiment.")
    parser.add_argument("--hpo-outdir", required=True)
    parser.add_argument("--dc-hpo-outdir", required=True)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--random-runs", type=int, default=1000)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    result = run_progressive_ablation(hpo_outdir=args.hpo_outdir, dc_hpo_outdir=args.dc_hpo_outdir, seeds=_parse_seeds(args.seeds), random_runs=args.random_runs, outdir=args.outdir, include_random=not args.no_random)
    print(f"[Saved] {result['outdir']}")


if __name__ == "__main__":
    main()
