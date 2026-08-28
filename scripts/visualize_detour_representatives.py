"""Create verified representative configuration plots from formal detour outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from library.config import FILE_PATHS, km2pix
from library.data_io import load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth
from library.metrics import calculate_kruskals_stress, direction_violation_rate, mean_angular_error_violations
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _scale_sim_distance_data
from scripts.create_section_6_5_visual_prototype import (
    NODE_HANDLES,
    OVERLAY_HANDLES,
    _cjk_font,
    _combined_overlay_extent,
    _distance_edge_errors,
    _draw_error_map,
    _draw_overlay,
    _panel_extent,
    _relax_annotations,
    _style_axis,
    _wrong_direction_nodes,
)


DEFAULT_SOURCE = "outputs/ch5_detour_factor_sensitivity_formal_13scenarios_hpo10_final100_20260825"
DEFAULT_KAPPAS = (1.0, 0.975, 0.95, 0.85, 0.75, 0.70)
SELECTION_METRICS = (
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "RMSE_final_test_km",
)


def _scenario_name(kappa: float) -> str:
    return f"scenario_kappa_{kappa:.3f}".replace(".", "p")


def _select_representative_run(runs: pd.DataFrame) -> tuple[pd.Series, dict]:
    metrics = runs.loc[:, SELECTION_METRICS].astype(float)
    valid = np.isfinite(metrics).all(axis=1)
    if not valid.any():
        raise ValueError("No successful finite run is available for representative selection.")
    metrics = metrics.loc[valid]
    median = metrics.median()
    mad = (metrics - median).abs().median().replace(0.0, 1.0)
    distance = np.sqrt((((metrics - median) / mad) ** 2).sum(axis=1))
    row = runs.loc[distance.idxmin()]
    details = {
        "median_vector": {metric: float(median[metric]) for metric in SELECTION_METRICS},
        "mad_vector": {metric: float(mad[metric]) for metric in SELECTION_METRICS},
        "standardized_distance": float(distance.loc[row.name]),
    }
    return row, details


def _overlay_rmse(points: np.ndarray, targets: dict[str, np.ndarray], labels: list[str], dni: dict[str, int]) -> float:
    errors = [float(np.linalg.norm(points[dni[label]] - targets[label]) / km2pix) for label in labels]
    return float(np.sqrt(np.mean(np.square(errors))))


def _verify_saved_metrics(
    row: pd.Series,
    points: np.ndarray,
    data_sim: list,
    directional_data: list,
    targets: dict[str, np.ndarray],
    tests: list[str],
    dni: dict[str, int],
) -> dict[str, float]:
    recomputed = {
        "E_distance_stress": float(calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)),
        "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
        "E_direction_mae": float(mean_angular_error_violations(points, directional_data, dni)),
        "RMSE_final_test_km": _overlay_rmse(points, targets, tests, dni),
    }
    for metric, value in recomputed.items():
        expected = float(row[metric])
        if not np.isclose(value, expected, rtol=1e-9, atol=1e-8):
            raise ValueError(f"Saved {metric} mismatch for seed {int(row['seed'])}: {value} != {expected}")
    return recomputed


def _draw_overlay_figure(
    case: dict,
    targets: dict,
    vertices: list[str],
    dni: dict[str, int],
    outdir: Path,
    *,
    shared_extent: tuple[float, float, float, float],
    shared_norm: colors.Normalize,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    record = {
        "variant": f"kappa={case['kappa']:.3f}",
        "rerun_metrics": {
            "RMSE_test_km": case["metrics"]["RMSE_final_test_km"],
            "E_distance_stress": case["metrics"]["E_distance_stress"],
            "E_direction_vr": case["metrics"]["E_direction_vr"],
        },
    }
    annotations = _draw_overlay(
        ax,
        case["points"],
        targets,
        record,
        vertices,
        dni,
        case["anchor_labels"],
        case["test_labels"],
        shared_norm,
        plt.get_cmap("plasma"),
        _cjk_font(12.5),
        draw_title=False,
    )
    _style_axis(ax, shared_extent)
    ax.set_title(
        f"Ground-truth overlay | kappa={case['kappa']:.3f} (detour ratio={case['detour_ratio']:.3f})\n"
        f"Representative seed {case['seed']} | alpha={case['alpha']:g}, beta={case['beta']:g} | "
        f"RMSE={case['metrics']['RMSE_final_test_km']:.1f} km",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    _relax_annotations(fig, ax, annotations, iterations=180, max_offset=70.0)
    scalar = cm.ScalarMappable(norm=shared_norm, cmap=plt.get_cmap("plasma"))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.027, pad=0.025)
    colorbar.set_label("Held-out test-site error (km)", fontsize=12)
    fig.legend(handles=OVERLAY_HANDLES, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.subplots_adjust(bottom=0.10, top=0.87, left=0.03, right=0.95)
    stem = f"{_scenario_name(case['kappa'])}_ground_truth_overlay"
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _draw_error_figure(
    case: dict,
    vertices: list[str],
    dni: dict[str, int],
    outdir: Path,
    *,
    shared_extent: tuple[float, float, float, float],
    shared_norm: colors.Normalize,
) -> None:
    edge_errors = case["edge_errors"]
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    wrong_nodes = _wrong_direction_nodes(case["points"], vertices, dni)
    annotations = _draw_error_map(
        ax,
        case["points"],
        edge_errors,
        wrong_nodes,
        shared_norm,
        cmap,
        vertices,
        _cjk_font(11.8),
        clip_labels=False,
    )
    _style_axis(ax, shared_extent)
    ax.set_title(
        f"Constraint-error visualization | kappa={case['kappa']:.3f} (detour ratio={case['detour_ratio']:.3f})\n"
        f"Representative seed {case['seed']} | Stress={case['metrics']['E_distance_stress']:.4f} | "
        f"Violation Rate={case['metrics']['E_direction_vr']:.4f} | "
        f"Mean Angular Error={case['metrics']['E_direction_mae']:.4f} rad",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    _relax_annotations(fig, ax, annotations, iterations=260, max_offset=75.0)
    scalar = cm.ScalarMappable(norm=shared_norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Distance-edge relative error", fontsize=12)
    fig.legend(handles=NODE_HANDLES, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(bottom=0.09, top=0.86, left=0.025, right=0.95)
    stem = f"{_scenario_name(case['kappa'])}_constraint_error"
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def visualize_detour_representatives(source: Path, outdir: Path, kappas: list[float]) -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    _, vertices, dni, _, data_li = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertices, dni)
    directional_data = uploading_directional_data()
    summary = pd.read_csv(source / "detour_scenario_summary.csv")
    cases = []
    for kappa in kappas:
        matches = summary[np.isclose(summary["kappa"].to_numpy(float), kappa)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one detour scenario for kappa={kappa}.")
        scenario_summary = matches.iloc[0]
        scenario_dir = source / "scenarios" / _scenario_name(kappa)
        config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
        selected, selection_details = _select_representative_run(runs)
        seed = int(selected["seed"])
        saved = pd.read_csv(scenario_dir / "selected_final_positions_y_up_sim.csv")
        frame = saved[saved["seed"].astype(int) == seed].sort_values("node_idx")
        if frame["label"].astype(str).tolist() != vertices:
            raise ValueError(f"Saved positions have invalid node ordering for kappa={kappa}, seed={seed}.")
        points = frame[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        anchors = list(config["anchor_labels"])
        tests = list(config["test_labels"])
        frame_anchor = str(config["final_frame_anchor_label"])
        targets = _target_positions_sim(dni, gt_lonlat, frame_anchor, config["refer_pos_sim"])
        scaled_data = _scale_sim_distance_data(data_Li2sim(data_li), kappa)
        metrics = _verify_saved_metrics(selected, points, scaled_data, directional_data, targets, tests, dni)
        case = {
            "kappa": float(kappa),
            "detour_ratio": float(1.0 / kappa),
            "seed": seed,
            "alpha": float(scenario_summary["selected_alpha"]),
            "beta": float(scenario_summary["selected_beta"]),
            "scenario_mean_rmse_km": float(scenario_summary["RMSE_final_test_km_mean"]),
            "anchor_labels": anchors,
            "test_labels": tests,
            "frame_anchor": frame_anchor,
            "metrics": metrics,
            "selection": selection_details,
            "points": points,
        }
        case["targets"] = targets
        case["edge_errors"] = _distance_edge_errors(points, scaled_data, dni)
        cases.append(case)

    model_points = {_scenario_name(case["kappa"]): case["points"] for case in cases}
    shared_targets = cases[0]["targets"]
    shared_overlay_extent = _combined_overlay_extent(
        model_points,
        shared_targets,
        dni,
        cases[0]["anchor_labels"],
        cases[0]["test_labels"],
        pad_frac=0.075,
    )
    all_points = np.vstack([case["points"] for case in cases])
    shared_error_extent = _panel_extent(all_points, [], pad_frac=0.12)
    all_test_errors = [
        float(np.linalg.norm(case["points"][dni[label]] - case["targets"][label]) / km2pix)
        for case in cases
        for label in case["test_labels"]
    ]
    overlay_norm = colors.Normalize(vmin=0.0, vmax=max(all_test_errors))
    all_edge_errors = [error for case in cases for _, _, error in case["edge_errors"]]
    edge_norm = colors.Normalize(vmin=0.0, vmax=max(float(np.quantile(all_edge_errors, 0.95)), 0.03))
    for case in cases:
        _draw_overlay_figure(
            case,
            case["targets"],
            vertices,
            dni,
            outdir,
            shared_extent=shared_overlay_extent,
            shared_norm=overlay_norm,
        )
        _draw_error_figure(
            case,
            vertices,
            dni,
            outdir,
            shared_extent=shared_error_extent,
            shared_norm=edge_norm,
        )

    records = [
        {key: value for key, value in case.items() if key not in {"points", "targets", "edge_errors"}}
        for case in cases
    ]
    payload = {
        "source": str(source.resolve()),
        "selection_metrics": list(SELECTION_METRICS),
        "selection_rule": "minimum Euclidean distance to metric-wise median after MAD standardization",
        "model_rerun": False,
        "shared_coordinate_extent_across_scenarios": True,
        "shared_color_scales_across_scenarios": True,
        "verification": "All four metrics were independently recomputed from each saved position matrix.",
        "cases": records,
    }
    (outdir / "detour_representative_verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "kappa": case["kappa"],
                "detour_ratio": case["detour_ratio"],
                "seed": case["seed"],
                "alpha": case["alpha"],
                "beta": case["beta"],
                "scenario_mean_rmse_km": case["scenario_mean_rmse_km"],
                **case["metrics"],
            }
            for case in cases
        ]
    ).to_csv(outdir / "detour_representative_summary.csv", index=False, encoding="utf-8-sig")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--kappas", default=",".join(str(value) for value in DEFAULT_KAPPAS))
    parser.add_argument("--outdir", default="outputs/ch6_detour_representative_visualizations_20260826")
    args = parser.parse_args()
    kappas = [float(value.strip()) for value in args.kappas.split(",") if value.strip()]
    result = visualize_detour_representatives(Path(args.source), Path(args.outdir), kappas)
    print(f"[Saved] {len(result['cases'])} overlay figures and {len(result['cases'])} constraint-error figures to {args.outdir}")
    for case in result["cases"]:
        print(
            f"[OK] kappa={case['kappa']:.3f}, seed={case['seed']}, "
            f"representative RMSE={case['metrics']['RMSE_final_test_km']:.2f} km, "
            f"scenario mean={case['scenario_mean_rmse_km']:.2f} km"
        )


if __name__ == "__main__":
    main()
