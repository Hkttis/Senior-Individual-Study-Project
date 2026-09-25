"""Create verified representative spatial reconstructions for all tolerance scenarios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from library.config import FILE_PATHS, km2pix, refer_pos_sim, theta_thr_4dir, theta_thr_8dir
from library.data_io import load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth
from library.directions import DIR4_SIM, DIR8_UNIT_SIM
from library.metrics import calculate_kruskals_stress, direction_violation_rate, mean_angular_error_violations
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
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
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "ch5_direction_tolerance_sensitivity_hpo10_final100_20260908"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "ch6_direction_tolerance_spatial_reconstructions_20260909"
SELECTION_METRICS = (
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "RMSE_final_test_km",
)


def _scenario_name(numerator: int) -> str:
    return f"scenario_theta_{numerator}of6"


def _select_representative_run(runs: pd.DataFrame) -> tuple[pd.Series, dict]:
    metrics = runs.loc[:, SELECTION_METRICS].astype(float)
    valid = np.isfinite(metrics).all(axis=1)
    if int(valid.sum()) != 100:
        raise ValueError("Representative selection requires 100 finite formal runs per scenario.")
    metrics = metrics.loc[valid]
    median = metrics.median()
    mad = (metrics - median).abs().median().replace(0.0, 1.0)
    distance = np.sqrt((((metrics - median) / mad) ** 2).sum(axis=1))
    row = runs.loc[distance.idxmin()]
    return row, {
        "median_vector": {metric: float(median[metric]) for metric in SELECTION_METRICS},
        "mad_vector": {metric: float(mad[metric]) for metric in SELECTION_METRICS},
        "standardized_distance": float(distance.loc[row.name]),
        "selection_rule": "minimum four-metric MAD-standardized distance to the scenario median profile",
    }


def _overlay_rmse(
    points: np.ndarray,
    targets: dict[str, np.ndarray],
    labels: list[str],
    dni: dict[str, int],
) -> float:
    errors = [float(np.linalg.norm(points[dni[label]] - targets[label]) / km2pix) for label in labels]
    return float(np.sqrt(np.mean(np.square(errors))))


def _wrong_nodes_for_tolerance(
    points: np.ndarray,
    directional_data: list,
    dni: dict[str, int],
    tolerance_scale: float,
) -> set[int]:
    wrong_nodes: set[int] = set()
    for row in directional_data:
        if row is None or len(row) < 3:
            continue
        source, target, direction_name = row[0], row[1], str(row[2]).strip()
        if source not in dni or target not in dni or direction_name not in DIR8_UNIT_SIM:
            continue
        source_idx, target_idx = dni[source], dni[target]
        displacement = points[target_idx] - points[source_idx]
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-9:
            continue
        observed = displacement / distance
        expected = np.asarray(DIR8_UNIT_SIM[direction_name], dtype=float)
        angle = abs(math.atan2(
            float(observed[0] * expected[1] - observed[1] * expected[0]),
            float(np.dot(observed, expected)),
        ))
        half_width = (theta_thr_4dir if direction_name in DIR4_SIM else theta_thr_8dir) * tolerance_scale
        if angle > half_width:
            wrong_nodes.update((source_idx, target_idx))
    return wrong_nodes


def _verify_metrics(
    row: pd.Series,
    points: np.ndarray,
    data_sim: list,
    directional_data: list,
    targets: dict[str, np.ndarray],
    tests: list[str],
    dni: dict[str, int],
    tolerance_scale: float,
) -> dict[str, float]:
    recomputed = {
        "E_distance_stress": float(
            calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)
        ),
        "E_direction_vr": float(
            direction_violation_rate(
                points,
                directional_data,
                dni,
                direction_tolerance_scale=tolerance_scale,
            )
        ),
        "E_direction_mae": float(
            mean_angular_error_violations(
                points,
                directional_data,
                dni,
                direction_tolerance_scale=tolerance_scale,
            )
        ),
        "RMSE_final_test_km": _overlay_rmse(points, targets, tests, dni),
    }
    for metric, value in recomputed.items():
        expected = float(row[metric])
        if not np.isclose(value, expected, rtol=1e-9, atol=1e-8):
            raise ValueError(
                f"Metric mismatch for seed {int(row['seed'])}, {metric}: {value} != {expected}"
            )
    return recomputed


def _draw_case(
    case: dict,
    *,
    vertices: list[str],
    dni: dict[str, int],
    overlay_extent: tuple[float, float, float, float],
    error_extent: tuple[float, float, float, float],
    overlay_norm: colors.Normalize,
    edge_norm: colors.Normalize,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.7), constrained_layout=True)
    record = {
        "variant": f"λ={case['numerator']}/6",
        "rerun_metrics": {
            "RMSE_test_km": case["metrics"]["RMSE_final_test_km"],
            "E_distance_stress": case["metrics"]["E_distance_stress"],
            "E_direction_vr": case["metrics"]["E_direction_vr"],
        },
    }
    overlay_annotations = _draw_overlay(
        axes[0],
        case["points"],
        case["targets"],
        record,
        vertices,
        dni,
        case["anchor_labels"],
        case["test_labels"],
        overlay_norm,
        plt.get_cmap("plasma"),
        _cjk_font(10.8),
        draw_title=False,
    )
    error_annotations = _draw_error_map(
        axes[1],
        case["points"],
        case["edge_errors"],
        case["wrong_nodes"],
        edge_norm,
        plt.get_cmap("cividis"),
        vertices,
        _cjk_font(9.8),
        clip_labels=False,
    )
    _style_axis(axes[0], overlay_extent)
    _style_axis(axes[1], error_extent)
    _relax_annotations(fig, axes[0], overlay_annotations, iterations=180, max_offset=72.0)
    _relax_annotations(fig, axes[1], error_annotations, iterations=260, max_offset=78.0)

    axes[0].set_title("(a) Ground-truth overlay", fontsize=16, fontweight="bold", pad=14)
    axes[1].set_title("(b) Constraint-error visualization", fontsize=16, fontweight="bold", pad=14)
    fig.suptitle(
        f"Direction tolerance λ={case['numerator']}/6 | "
        f"cardinal/diagonal half-width={case['cardinal_deg']:g}°/{case['diagonal_deg']:g}°\n"
        f"Representative seed {case['seed']} | α={case['alpha']:g}, β={case['beta']:g} | "
        f"RMSE={case['metrics']['RMSE_final_test_km']:.1f} km, "
        f"Stress={case['metrics']['E_distance_stress']:.3f}, "
        f"VR={case['metrics']['E_direction_vr']:.3f}, "
        f"MAE={case['metrics']['E_direction_mae']:.3f} rad",
        fontsize=15,
        fontweight="bold",
    )

    overlay_map = cm.ScalarMappable(norm=overlay_norm, cmap=plt.get_cmap("plasma"))
    overlay_map.set_array([])
    overlay_bar = fig.colorbar(overlay_map, ax=axes[0], fraction=0.025, pad=0.012)
    overlay_bar.set_label("Held-out test-site error (km)", fontsize=10.5)
    edge_map = cm.ScalarMappable(norm=edge_norm, cmap=plt.get_cmap("cividis"))
    edge_map.set_array([])
    edge_bar = fig.colorbar(edge_map, ax=axes[1], fraction=0.025, pad=0.012)
    edge_bar.set_label("Distance-edge relative error", fontsize=10.5)
    fig.legend(
        handles=OVERLAY_HANDLES + NODE_HANDLES,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    stem = outdir / f"lambda_{case['numerator']}of6_spatial_reconstruction"
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_direction_tolerance_representatives(source: Path, outdir: Path) -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    _, vertices, dni, _, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)
    gt_lonlat = uploading_ground_truth(vertices, dni)
    directional_data = uploading_directional_data()
    summary = pd.read_csv(source / "direction_tolerance_scenario_summary.csv", encoding="utf-8-sig")
    experiment_config = json.loads(
        (source / "direction_tolerance_experiment_config.json").read_text(encoding="utf-8")
    )
    common_anchors = list(experiment_config["anchor_labels"])
    common_tests = list(experiment_config["test_labels"])
    common_frame_anchor = str(experiment_config["final_frame_anchor_label"])
    cases: list[dict] = []

    for numerator in range(7):
        summary_row = summary.loc[summary["tolerance_numerator"].eq(numerator)]
        if len(summary_row) != 1:
            raise ValueError(f"Expected one summary row for lambda={numerator}/6.")
        summary_row = summary_row.iloc[0]
        scenario_dir = source / "scenarios" / _scenario_name(numerator)
        config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv", encoding="utf-8-sig")
        selected, selection = _select_representative_run(runs)
        seed = int(selected["seed"])
        positions = pd.read_csv(
            scenario_dir / "selected_final_positions_y_up_sim.csv", encoding="utf-8-sig"
        )
        frame = positions.loc[positions["seed"].astype(int).eq(seed)].sort_values("node_idx")
        if len(frame) != len(vertices) or frame["label"].astype(str).tolist() != vertices:
            raise ValueError(f"Invalid position matrix for lambda={numerator}/6, seed={seed}.")
        points = frame[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        anchors = list(config.get("anchor_labels", common_anchors))
        tests = list(config.get("test_labels", common_tests))
        frame_anchor = str(config.get("final_frame_anchor_label", common_frame_anchor))
        if anchors != common_anchors or tests != common_tests or frame_anchor != common_frame_anchor:
            raise ValueError(f"Registered anchor/test split differs for lambda={numerator}/6.")
        targets = _target_positions_sim(
            dni,
            gt_lonlat,
            frame_anchor,
            config.get("refer_pos_sim", refer_pos_sim),
        )
        scale = float(config["direction_tolerance_scale"])
        metrics = _verify_metrics(
            selected,
            points,
            data_sim,
            directional_data,
            targets,
            tests,
            dni,
            scale,
        )
        cases.append(
            {
                "numerator": numerator,
                "tolerance_scale": scale,
                "cardinal_deg": float(summary_row["cardinal_half_width_deg"]),
                "diagonal_deg": float(summary_row["diagonal_half_width_deg"]),
                "seed": seed,
                "alpha": float(summary_row["selected_alpha"]),
                "beta": float(summary_row["selected_beta"]),
                "scenario_mean_rmse_km": float(summary_row["RMSE_final_test_km_mean"]),
                "anchor_labels": anchors,
                "test_labels": tests,
                "metrics": metrics,
                "selection": selection,
                "points": points,
                "targets": targets,
                "edge_errors": _distance_edge_errors(points, data_sim, dni),
                "wrong_nodes": _wrong_nodes_for_tolerance(
                    points, directional_data, dni, scale
                ),
            }
        )

    model_points = {_scenario_name(case["numerator"]): case["points"] for case in cases}
    overlay_extent = _combined_overlay_extent(
        model_points,
        cases[0]["targets"],
        dni,
        cases[0]["anchor_labels"],
        cases[0]["test_labels"],
        pad_frac=0.09,
    )
    error_extent = _panel_extent(np.vstack([case["points"] for case in cases]), [], pad_frac=0.14)
    test_errors = [
        float(np.linalg.norm(case["points"][dni[label]] - case["targets"][label]) / km2pix)
        for case in cases
        for label in case["test_labels"]
    ]
    overlay_norm = colors.Normalize(vmin=0.0, vmax=max(test_errors))
    edge_errors = [error for case in cases for _, _, error in case["edge_errors"]]
    edge_norm = colors.Normalize(vmin=0.0, vmax=max(float(np.quantile(edge_errors, 0.95)), 0.03))

    for case in cases:
        _draw_case(
            case,
            vertices=vertices,
            dni=dni,
            overlay_extent=overlay_extent,
            error_extent=error_extent,
            overlay_norm=overlay_norm,
            edge_norm=edge_norm,
            outdir=outdir,
        )

    records = []
    for case in cases:
        records.append(
            {
                "tolerance_numerator": case["numerator"],
                "direction_tolerance_scale": case["tolerance_scale"],
                "cardinal_half_width_deg": case["cardinal_deg"],
                "diagonal_half_width_deg": case["diagonal_deg"],
                "seed": case["seed"],
                "alpha": case["alpha"],
                "beta": case["beta"],
                "scenario_mean_rmse_km": case["scenario_mean_rmse_km"],
                **case["metrics"],
                "n_direction_violation_nodes": len(case["wrong_nodes"]),
                "standardized_representative_distance": case["selection"]["standardized_distance"],
            }
        )
    pd.DataFrame(records).to_csv(
        outdir / "direction_tolerance_representative_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "source": str(source.resolve()),
        "model_rerun": False,
        "selection_metrics": list(SELECTION_METRICS),
        "selection_rule": "minimum four-metric MAD-standardized distance to each scenario median profile",
        "direction_violation_policy": "each panel uses its own registered direction-tolerance scale",
        "shared_coordinate_extent_across_scenarios": True,
        "shared_color_scales_across_scenarios": True,
        "verification": "All four displayed metrics were independently recomputed from the selected saved position matrix.",
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"points", "targets", "edge_errors", "wrong_nodes"}
            }
            for case in cases
        ],
    }
    (outdir / "direction_tolerance_representative_verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    result = visualize_direction_tolerance_representatives(args.source, args.outdir)
    print(f"[Saved] {len(result['cases'])} verified spatial-reconstruction figures to {args.outdir}")
    for case in result["cases"]:
        metrics = case["metrics"]
        print(
            f"[OK] lambda={case['numerator']}/6, seed={case['seed']}, "
            f"RMSE={metrics['RMSE_final_test_km']:.2f} km, "
            f"Stress={metrics['E_distance_stress']:.4f}, "
            f"VR={metrics['E_direction_vr']:.4f}, "
            f"MAE={metrics['E_direction_mae']:.4f} rad"
        )


if __name__ == "__main__":
    main()
