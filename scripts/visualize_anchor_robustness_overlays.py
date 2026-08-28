"""Visualize representative anchor-split runs without rerunning any model."""

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

from library.config import FILE_PATHS, km2pix, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth
from library.metrics import calculate_kruskals_stress, direction_violation_rate
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from scripts.create_section_6_5_visual_prototype import (
    OVERLAY_HANDLES,
    NODE_HANDLES,
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


DEFAULT_SOURCE = "outputs/ch5_anchor_split_robustness_formal_45splits_hpo3_final10_20260824"
DEFAULT_SPLITS = ("split_018", "split_005", "split_040", "split_008", "split_039", "split_037")
SELECTION_METRICS = ("E_distance_stress", "E_direction_vr", "RMSE_final_test_km")


def _select_representative_run(runs: pd.DataFrame) -> pd.Series:
    metrics = runs.loc[:, SELECTION_METRICS].astype(float)
    valid = np.isfinite(metrics).all(axis=1)
    if not valid.any():
        raise ValueError("No successful finite run is available for representative selection.")
    metrics = metrics.loc[valid]
    median = metrics.median()
    mad = (metrics - median).abs().median().replace(0.0, 1.0)
    distance = np.sqrt((((metrics - median) / mad) ** 2).sum(axis=1))
    return runs.loc[distance.idxmin()]


def _rebase_points(points: np.ndarray, frame_anchor: str, common_targets: dict[str, np.ndarray]) -> np.ndarray:
    if frame_anchor not in common_targets:
        raise ValueError(f"Frame anchor {frame_anchor!r} does not have ground-truth coordinates.")
    return np.asarray(points, dtype=float) + common_targets[frame_anchor] - np.asarray(refer_pos_sim, dtype=float)


def _overlay_rmse(points: np.ndarray, targets: dict[str, np.ndarray], labels: list[str], dni: dict[str, int]) -> float:
    errors = [float(np.linalg.norm(points[dni[label]] - targets[label]) / km2pix) for label in labels]
    return float(np.sqrt(np.mean(np.square(errors))))


def _constraint_metrics(points: np.ndarray, data_sim: list, directional_data: list, dni: dict[str, int]) -> tuple[float, float]:
    stress = float(calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim))
    violation_rate = float(direction_violation_rate(points, directional_data, dni))
    return stress, violation_rate


def _case_kind(split_id: str, row: pd.Series) -> str:
    if bool(row["is_original_split"]):
        return "Original anchors"
    if float(row["RMSE_final_test_mean_km"]) < 150.0:
        return "Low RMSE; boundary" if bool(row["selected_on_grid_boundary"]) else "Low RMSE; interior"
    return "High RMSE; boundary" if bool(row["selected_on_grid_boundary"]) else "High RMSE; interior"


def create_anchor_robustness_overlays(source: Path, outdir: Path, split_ids: list[str]) -> dict:
    summary = pd.read_csv(source / "anchor_split_summary.csv").set_index("split_id")
    _, vertice, dni, _, distance_data = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(distance_data)
    directional_data = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(vertice, dni)
    original = summary[summary["is_original_split"].astype(bool)]
    if len(original) != 1:
        raise ValueError("Exactly one original anchor split is required to establish the common frame.")
    common_anchor = str(original.iloc[0]["final_frame_anchor"])
    targets = _target_positions_sim(dni, gt_lonlat, common_anchor, refer_pos_sim)
    cases = []

    for split_id in split_ids:
        if split_id not in summary.index:
            raise ValueError(f"Unknown anchor split: {split_id}")
        split_dir = source / "splits" / split_id
        cfg = json.loads((split_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(split_dir / "selected_final_runs_by_seed.csv")
        selected = _select_representative_run(runs)
        seed = int(selected["seed"])
        positions = pd.read_csv(split_dir / "selected_final_positions_y_up_sim.csv")
        rows = positions[positions["seed"] == seed].set_index("label")
        if len(rows) != len(vertice) or set(rows.index) != set(vertice):
            raise ValueError(f"Incomplete saved position matrix for {split_id} seed {seed}.")
        raw_points = rows.loc[vertice, ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        points = _rebase_points(raw_points, str(cfg["final_frame_anchor_label"]), targets)
        tests = list(cfg["test_labels"])
        anchors = list(cfg["anchor_labels"])
        rmse = _overlay_rmse(points, targets, tests, dni)
        expected = float(selected["RMSE_final_test_km"])
        if not np.isclose(rmse, expected, rtol=1e-9, atol=1e-8):
            raise ValueError(f"Overlay RMSE mismatch for {split_id} seed {seed}: {rmse} != {expected}")
        stress, violation_rate = _constraint_metrics(points, data_sim, directional_data, dni)
        for name, actual, reference in (
            ("Stress", stress, float(selected["E_distance_stress"])),
            ("Violation Rate", violation_rate, float(selected["E_direction_vr"])),
        ):
            if not np.isclose(actual, reference, rtol=1e-9, atol=1e-8):
                raise ValueError(f"{name} mismatch for {split_id} seed {seed}: {actual} != {reference}")
        split_row = summary.loc[split_id]
        cases.append(
            {
                "split_id": split_id,
                "case_kind": _case_kind(split_id, split_row),
                "seed": seed,
                "alpha": float(split_row["selected_alpha"]),
                "beta": float(split_row["selected_beta"]),
                "split_mean_rmse_km": float(split_row["RMSE_final_test_mean_km"]),
                "representative_rmse_km": rmse,
                "stress": stress,
                "violation_rate": violation_rate,
                "anchors": anchors,
                "test_labels": tests,
                "frame_anchor": str(cfg["final_frame_anchor_label"]),
                "points": points,
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    all_labels = list(dict.fromkeys(label for case in cases for label in case["test_labels"] + case["anchors"]))
    all_points = {case["split_id"]: case["points"] for case in cases}
    extent = _combined_overlay_extent(all_points, targets, dni, [], all_labels, pad_frac=0.085)
    errors = [
        float(np.linalg.norm(case["points"][dni[label]] - targets[label]) / km2pix)
        for case in cases
        for label in case["test_labels"]
    ]
    norm = colors.Normalize(vmin=0.0, vmax=max(errors))
    cmap = plt.get_cmap("plasma")
    label_font = _cjk_font(10.5)
    edge_errors = {
        case["split_id"]: _distance_edge_errors(case["points"], data_sim, dni)
        for case in cases
    }
    edge_error_values = [error for values in edge_errors.values() for _, _, error in values]
    edge_norm = colors.Normalize(vmin=0.0, vmax=max(float(np.quantile(edge_error_values, 0.95)), 0.03))
    edge_cmap = plt.get_cmap("RdYlGn_r")

    for case in cases:
        fig, ax = plt.subplots(figsize=(9.5, 8.0))
        _draw_case(fig, ax, case, targets, vertice, dni, norm, cmap, label_font, extent)
        fig.legend(handles=OVERLAY_HANDLES, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=0.13, top=0.87, left=0.03, right=0.97)
        fig.savefig(outdir / f"{case['split_id']}_ground_truth_overlay.png", dpi=220, bbox_inches="tight")
        fig.savefig(outdir / f"{case['split_id']}_ground_truth_overlay.svg", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.5, 8.7))
        wrong_nodes = _wrong_direction_nodes(case["points"], vertice, dni)
        annotations = _draw_error_map(
            ax,
            case["points"],
            edge_errors[case["split_id"]],
            wrong_nodes,
            edge_norm,
            edge_cmap,
            vertice,
            _cjk_font(10.2),
            clip_labels=False,
        )
        _style_axis(ax, _panel_extent(case["points"], [], pad_frac=0.19))
        ax.set_title(
            f"{case['case_kind']} | {case['split_id']} | alpha={case['alpha']:g}, beta={case['beta']:g}\n"
            f"Stress={case['stress']:.4f} | Violation Rate={case['violation_rate']:.4f} | "
            f"RMSE={case['representative_rmse_km']:.1f} km",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        _relax_annotations(fig, ax, annotations, iterations=220, max_offset=62.0)
        scalar = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.025, pad=0.025)
        colorbar.set_label("Distance-edge relative error", fontsize=10)
        fig.legend(handles=NODE_HANDLES, loc="lower center", ncol=2, frameon=False, fontsize=10)
        fig.subplots_adjust(bottom=0.10, top=0.86, left=0.025, right=0.96)
        fig.savefig(outdir / f"{case['split_id']}_constraint_error_map.png", dpi=240, bbox_inches="tight")
        fig.savefig(outdir / f"{case['split_id']}_constraint_error_map.svg", bbox_inches="tight")
        plt.close(fig)

    records = [{key: value for key, value in case.items() if key != "points"} for case in cases]
    payload = {
        "source": str(source.resolve()),
        "common_coordinate_frame_anchor": common_anchor,
        "selection_metrics": list(SELECTION_METRICS),
        "selection_rule": "minimum Euclidean distance to metric-wise median after MAD standardization",
        "no_model_rerun": True,
        "verification": "overlay test-only RMSE, Stress, and Violation Rate equal the saved formal run metrics",
        "cases": records,
    }
    (outdir / "representative_overlay_verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(outdir / "representative_overlay_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[Saved] {len(records)} separate overlay figures and {len(records)} separate constraint-error figures")
    for case in records:
        print(
            f"[OK] {case['split_id']} seed={case['seed']} alpha={case['alpha']:g} beta={case['beta']:g} "
            f"representative_RMSE={case['representative_rmse_km']:.2f} km "
            f"split_mean_RMSE={case['split_mean_rmse_km']:.2f} km"
        )
    return payload


def _draw_case(fig, ax, case, targets, vertice, dni, norm, cmap, label_font, extent):
    record = {
        "variant": case["split_id"],
        "rerun_metrics": {
            "RMSE_test_km": case["representative_rmse_km"],
            "E_distance_stress": case["stress"],
            "E_direction_vr": case["violation_rate"],
        },
    }
    annotations = _draw_overlay(
        ax,
        case["points"],
        targets,
        record,
        vertice,
        dni,
        case["anchors"],
        case["test_labels"],
        norm,
        cmap,
        label_font,
        draw_title=False,
    )
    _style_axis(ax, extent)
    ax.set_title(
        f"{case['case_kind']} | {case['split_id']} | alpha={case['alpha']:g}, beta={case['beta']:g}\n"
        f"Representative RMSE={case['representative_rmse_km']:.1f} km | "
        f"Split mean={case['split_mean_rmse_km']:.1f} km",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    _relax_annotations(fig, ax, annotations, iterations=90, max_offset=44.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--split-ids", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default="outputs/ch6_anchor_robustness_representative_overlays_20260825")
    args = parser.parse_args()
    split_ids = [item.strip() for item in args.split_ids.split(",") if item.strip()]
    if not split_ids:
        raise ValueError("At least one split ID is required.")
    create_anchor_robustness_overlays(Path(args.source), Path(args.outdir), split_ids)


if __name__ == "__main__":
    main()
