"""Export review plots for ablation study outputs.

Usage
-----
python -m scripts.export_ablation_review --ablation-outdir outputs/ch5_ablation_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import FILE_PATHS, OUTPUT_DIR, km2pix, refer_pos_sim

_MPLCONFIGDIR = Path(OUTPUT_DIR) / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from library.anchor_frame import px_list_to_km_list
from library.data_io import get_anchor_labels, get_test_site_labels, load_ini_data_from_csv, load_site_points
from library.geometry import get_lcc_bounds, get_lcc_parameters, lcc_transformation_with_anchor


def _assert_lcc_matches_ablation_config(ablation_outdir: Path) -> None:
    config_path = ablation_outdir / "ablation_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing ablation config for LCC check: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "lcc_bounds" not in config or "lcc_parameters" not in config:
        raise ValueError(
            f"{config_path} does not record lcc_bounds/lcc_parameters; "
            "refusing to draw review plots because current LCC may differ from the experiment."
        )

    current_bounds = dict(zip(["lon_min", "lon_max", "lat_min", "lat_max"], map(float, get_lcc_bounds())))
    current_params = dict(zip(["lat_1", "lat_2", "lon_0"], map(float, get_lcc_parameters())))
    recorded_bounds = {key: float(value) for key, value in config["lcc_bounds"].items()}
    recorded_params = {key: float(value) for key, value in config["lcc_parameters"].items()}
    if recorded_bounds != current_bounds or recorded_params != current_params:
        raise ValueError(
            "Current LCC parameters differ from ablation_config.json. "
            f"recorded_bounds={recorded_bounds}, current_bounds={current_bounds}, "
            f"recorded_params={recorded_params}, current_params={current_params}"
        )


def _load_gt_km():
    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    site_rows = load_site_points()
    site = {row["name"]: row for row in site_rows}
    anchors = get_anchor_labels()
    tests = get_test_site_labels()
    gt_lonlat = [(0.0, 0.0) for _ in vertice]
    for label in anchors + tests:
        gt_lonlat[dni[label]] = (float(site[label]["lon"]), float(site[label]["lat"]))
    gt_km = lcc_transformation_with_anchor(dni, gt_lonlat, anchor_label=anchors[0])
    return vertice, dni, anchors, tests, gt_km


def _plot_variant_map(df_pos: pd.DataFrame, variant: str, seed: int, out_png: Path) -> None:
    _vertice, _dni, anchors, tests, gt_km = _load_gt_km()
    subset = df_pos[(df_pos["variant"] == variant) & (df_pos["seed"] == seed)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for variant={variant!r}, seed={seed}")

    pred_by_label = {
        row["label"]: (float(row["x_y_up_sim"]), float(row["y_y_up_sim"]))
        for _, row in subset.iterrows()
    }
    pred_sim = [pred_by_label[label] for label in _vertice]
    pred_km = px_list_to_km_list(pred_sim, tuple(refer_pos_sim), km2pix)

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"anchor": "#d62728", "test": "#1f77b4"}
    used_labels: set[str] = set()

    for label in anchors + tests:
        idx = _dni[label]
        role = "anchor" if label in anchors else "test"
        color = colors[role]
        gx, gy = gt_km[idx]
        px, py = pred_km[idx]
        gt_label = f"GT {role}"
        pred_label = f"Pred {role}"
        ax.scatter(gx, gy, marker="o", s=70, color=color, label=gt_label if gt_label not in used_labels else None)
        used_labels.add(gt_label)
        ax.scatter(px, py, marker="x", s=74, color=color, label=pred_label if pred_label not in used_labels else None)
        used_labels.add(pred_label)
        ax.plot([gx, px], [gy, py], color=color, alpha=0.28, linewidth=1)
        ax.text(px + 4, py + 4, label, fontsize=8, fontproperties=font)

    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.axvline(0, color="#999999", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x km, anchored at " + anchors[0], fontproperties=font)
    ax.set_ylabel("y km", fontproperties=font)
    ax.set_title(f"Ablation result: {variant}, seed={seed}", fontproperties=font)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_metric_bars(df_runs: pd.DataFrame, out_png: Path) -> None:
    ok = df_runs[df_runs["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful ablation rows found.")
    group = ok.groupby("variant", as_index=False)[
        [
            "RMSE_test_km",
            "E_distance_stress",
            "E_direction_vr",
            "E_direction_mae",
            "min_pairwise_distance_km",
            "median_pairwise_distance_km",
        ]
    ].mean()
    group = group.sort_values("RMSE_test_km")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    metrics = [
        ("RMSE_test_km", "RMSE test (km)"),
        ("E_distance_stress", "Kruskal stress"),
        ("E_direction_vr", "Direction violation rate"),
        ("E_direction_mae", "Direction MAE"),
        ("min_pairwise_distance_km", "Min pairwise distance (km)"),
        ("median_pairwise_distance_km", "Median pairwise distance (km)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(group["variant"], group[col], color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def export_ablation_review(ablation_outdir: str | Path, seed: int = 0) -> Path:
    ablation_outdir = Path(ablation_outdir)
    _assert_lcc_matches_ablation_config(ablation_outdir)
    review_dir = ablation_outdir / "ablation_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    positions_path = ablation_outdir / "ablation_final_positions_y_up_sim.csv"
    runs_path = ablation_outdir / "ablation_runs_by_seed.csv"
    if not positions_path.exists():
        raise FileNotFoundError(f"Missing positions CSV: {positions_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs CSV: {runs_path}")

    df_pos = pd.read_csv(positions_path)
    df_runs = pd.read_csv(runs_path)
    variants = sorted(df_pos[df_pos["seed"] == seed]["variant"].unique())
    for variant in variants:
        safe_variant = variant.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_png = review_dir / f"{safe_variant}_seed{seed}_map.png"
        _plot_variant_map(df_pos, variant, seed, out_png)
        print(f"[Saved] {out_png}")

    metric_png = review_dir / "ablation_metrics_barplot.png"
    _plot_metric_bars(df_runs, metric_png)
    print(f"[Saved] {metric_png}")
    return review_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ablation model-result review plots.")
    parser.add_argument("--ablation-outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_ablation_review(args.ablation_outdir, seed=args.seed)


if __name__ == "__main__":
    main()
