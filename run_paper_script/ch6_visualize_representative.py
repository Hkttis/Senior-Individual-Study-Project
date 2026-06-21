"""run_paper_script.ch6_visualize_representative

Load model outputs, select representative run per model,
then produce representative error-map / overlay figures.

Usage
-----
Run from the physics_simulation project root.
Default fixed anchors come from data/site_rmse_points.csv (use_role=anchor).

python -m run_paper_script.paper_run ch6-representative

Preferred for formal ablation outputs:
python -m run_paper_script.paper_run ch6-representative --ablation-outdir outputs/ch5_ablation_lcc_sitebounds_alpha_1_beta_-0.5_100seeds

Or pass explicit history CSVs:

python -m run_paper_script.paper_run ch6-representative `
   --csv-sm "results_data/all_pos_sm_px_data.csv" `
   --csv-dm "results_data/all_pos_dm_px_data.csv" `
   --csv-ph "results_data/all_pos_sm_ph_data.csv"
"""

from __future__ import annotations
import argparse, json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos,
    refer_pos_sim,
    refer_pos_screen,
)
from library.data_io import (
    load_ini_data_from_csv, uploading_ground_truth,
    uploading_directional_data, _read_model_csv,
    get_anchor_labels, get_anchor_align_label,
)
from library.units import data_Li2sim
from library.coordinates import flipping_y
from library.metrics import alignment_and_scaling, procrustes_align_by_fixed_points
from library.model_cmp import select_representative_run
from library.visualization import (
    plot_three_model_convergence_pygame_pixelaware,
    plot_three_model_direction_convergence,
    visualize_error_map_official,
    ground_truth_comparison,
)
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from MDS_model.stress_majorization_mds_model import stress_majorization
from library.initialization import generate_CHEN_initial_positions
from library.physics import main_physics_simulation
from library.model_cmp import run_directed_MDS
from run_paper_script.ch5_ablation_study import (
    PHYSICS_VARIANTS,
    _evaluate_positions,
    _variant_forces,
)


REPRESENTATIVE_METRICS = ["E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_test_km"]
DEFAULT_VERIFY_ABS_TOL = 1e-6
DEFAULT_VERIFY_REL_TOL = 1e-6


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", type=str, default="")
    p.add_argument("--ablation-outdir", type=str, default="", help="Use formal ablation output CSVs instead of legacy history CSVs.")
    p.add_argument("--outdir", type=str, default="", help="Output directory for representative figures/history.")
    p.add_argument("--csv-sm", type=str, default=FILE_PATHS["save_all_pos_sm_px_data"])
    p.add_argument("--csv-dm", type=str, default=FILE_PATHS["save_all_pos_dm_px_data"])
    p.add_argument("--csv-ph", type=str, default=FILE_PATHS["save_all_pos_ph_px_data"])
    p.add_argument("--skip-convergence", action="store_true")
    p.add_argument("--skip-errormap", action="store_true")
    p.add_argument("--skip-overlay", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="Save figures and exit without waiting for windows to close.")
    p.add_argument("--verify-abs-tol", type=float, default=DEFAULT_VERIFY_ABS_TOL)
    p.add_argument("--verify-rel-tol", type=float, default=DEFAULT_VERIFY_REL_TOL)
    return p.parse_args()


def _select_representative_seed_from_as(group: pd.DataFrame) -> dict:
    ok = group[group["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful AS rows for variant={group['variant'].iloc[0]!r}")
    metrics = ok[REPRESENTATIVE_METRICS].astype(float)
    median = metrics.median()
    mad = (metrics - median).abs().median().replace(0, 1.0)
    distances = (((metrics - median) / mad) ** 2).sum(axis=1) ** 0.5
    chosen_idx = distances.idxmin()
    row = ok.loc[chosen_idx]
    return {
        "seed": int(row["seed"]),
        "selection_scope": "one representative seed per AS variant",
        "metrics": {metric: float(row[metric]) for metric in REPRESENTATIVE_METRICS},
        "median_vector": {metric: float(median[metric]) for metric in REPRESENTATIVE_METRICS},
        "mad_vector": {metric: float(mad[metric]) for metric in REPRESENTATIVE_METRICS},
        "std_distance": float(distances.loc[chosen_idx]),
    }


def _verify_rerun_matches_as_metrics(rep: dict, rerun_metrics: dict, *, abs_tol: float, rel_tol: float) -> dict:
    diffs = {}
    failures = []
    for metric in REPRESENTATIVE_METRICS:
        expected = float(rep["metrics"][metric])
        actual = float(rerun_metrics[metric])
        abs_diff = abs(actual - expected)
        rel_diff = abs_diff / max(abs(expected), 1e-12)
        ok = abs_diff <= abs_tol or rel_diff <= rel_tol
        diffs[metric] = {
            "as_metric": expected,
            "rerun_metric": actual,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "ok": bool(ok),
        }
        if not ok:
            failures.append(metric)
    if failures:
        raise ValueError(
            "Representative rerun metrics do not match AS metrics for "
            f"variant={rep.get('variant')!r}, seed={rep.get('seed')}: {failures}. "
            f"diffs={diffs}"
        )
    return diffs


def _as_config_value(config: dict, key: str, default):
    value = config.get(key, default)
    if isinstance(default, float):
        return float(value)
    return value


def _align_smacof_history(pos_history_li, vertice, dni, anchor_labels, anchor_lonlat, rp_sim):
    out = []
    for frame in pos_history_li:
        px = alignment_and_scaling(frame, vertice, dni, rp_sim, y_down=False, anchor_label=anchor_labels[0])
        px = procrustes_align_by_fixed_points(
            deepcopy(px),
            list(anchor_labels),
            list(anchor_lonlat),
            dni,
            refer_pos=rp_sim,
            anchor_label=anchor_labels[0],
        )
        out.append([[float(x), float(y)] for x, y in px])
    return out


def _align_dc_smacof_history(pos_history_li, vertice, dni, anchor_labels, rp_sim):
    return [
        [
            [float(x), float(y)]
            for x, y in alignment_and_scaling(
                frame,
                vertice,
                dni,
                rp_sim,
                y_down=False,
                anchor_label=anchor_labels[0],
            )
        ]
        for frame in pos_history_li
    ]


def _rerun_as_variant_history(
    *,
    variant: str,
    seed: int,
    config: dict,
    graph,
    vertice,
    dni,
    edges,
    anchor_labels,
    anchor_lonlat,
) -> list[list[list[float]]]:
    np.random.seed(seed)
    rp_sim = config.get("refer_pos_sim", refer_pos_sim)

    if variant in PHYSICS_VARIANTS:
        alpha = float(config["alpha"])
        beta = float(config["beta"])
        _w_dir, _w_reg, spring, directional, repulsion = _variant_forces(
            variant,
            alpha=alpha,
            beta=beta,
            w_dis=_as_config_value(config, "w_dis", 1.0),
            base_spring_stiffness=_as_config_value(config, "base_spring_stiffness", SPRING_STIFFNESS_BASE),
            base_directional_force=_as_config_value(config, "base_directional_force", DIRECTIONAL_FORCE_MAGNITUDE_BASE),
            base_repulsion_strength=_as_config_value(config, "base_repulsion_strength", REPULSION_STRENGTH_BASE),
        )
        vtx, v_dni, data_li, pos_init, fixed_positions_list = generate_CHEN_initial_positions(
            list(rp_sim),
            list(anchor_labels),
            list(anchor_lonlat),
            anchor_label=anchor_labels[0],
        )
        if vtx != vertice or v_dni != dni:
            raise ValueError("PhysicsSim rerun node order differs from AS data; refusing to compare histories.")
        directional_data = uploading_directional_data()
        _wrong, _stress_history, pos_history, pos_final = main_physics_simulation(
            vertice,
            dni,
            data_Li2sim(data_li),
            pos_init,
            directional_data,
            fixed_positions_list,
            spring,
            repulsion,
            directional,
            plot=False,
        )
        history = list(pos_history)
        if not history or not np.allclose(np.asarray(history[-1], dtype=float), np.asarray(pos_final, dtype=float)):
            history.append(pos_final)
        return [[[float(x), float(y)] for x, y in frame] for frame in history]

    if variant == "SMACOF":
        _pos_li, _stress_history, pos_history_li = stress_majorization(graph, dni, vertice, edges)
        return _align_smacof_history(pos_history_li, vertice, dni, anchor_labels, anchor_lonlat, rp_sim)

    if variant == "DC-SMACOF":
        dc_params = config.get("dc_smacof_hpo", {}) or {}
        pos_history_li = run_directed_MDS(
            vis=False,
            w_weight_value=dc_params.get("w_weight"),
            v_weight_value=dc_params.get("v_weight"),
        )
        return _align_dc_smacof_history(pos_history_li, vertice, dni, anchor_labels, rp_sim)

    raise ValueError(f"Unsupported AS variant for history rerun: {variant}")


def _write_history_csv(path: Path, variant: str, seed: int, vertice, history) -> None:
    rows = []
    for frame_idx, frame in enumerate(history):
        for label, pos in zip(vertice, frame):
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "frame": int(frame_idx),
                    "label": label,
                    "x_y_up_sim": float(pos[0]),
                    "y_y_up_sim": float(pos[1]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _as_convergence_metrics_for_history(
    *,
    variant: str,
    seed: int,
    history,
    vertice,
    dni,
    data_sim,
    directional_data,
    anchor_labels,
    anchor_lonlat,
    test_labels,
    test_lonlat,
    rp_sim,
) -> pd.DataFrame:
    rows = []
    for frame_idx, frame in enumerate(history):
        metrics, _positions = _evaluate_positions(
            model="PhysicsSim" if variant in PHYSICS_VARIANTS else variant,
            variant=variant,
            seed=int(seed),
            pos_y_up_sim=frame,
            vertice=vertice,
            dni=dni,
            data_sim=data_sim,
            directional_data=directional_data,
            anchor_labels=anchor_labels,
            anchor_lonlat=anchor_lonlat,
            test_labels=test_labels,
            test_lonlat=test_lonlat,
            refer_pos_sim=rp_sim,
            spring_stiffness=float("nan"),
            directional_force=float("nan"),
            repulsion_strength=float("nan"),
        )
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "frame": int(frame_idx),
                "E_distance_stress": metrics["E_distance_stress"],
                "E_direction_vr": metrics["E_direction_vr"],
                "E_direction_mae": metrics["E_direction_mae"],
                "RMSE_test_km": metrics["RMSE_test_km"],
            }
        )
    return pd.DataFrame(rows)


def _plot_as_convergence(df_metrics: pd.DataFrame, out_png: Path) -> None:
    import os
    mpl_config_dir = Path(OUTPUT_DIR) / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_titles = [
        ("E_distance_stress", "Kruskal stress"),
        ("E_direction_vr", "Direction violation rate"),
        ("E_direction_mae", "Direction MAE"),
        ("RMSE_test_km", "Test RMSE (km)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.ravel(), metric_titles):
        for variant, group in df_metrics.groupby("variant"):
            ax.plot(group["frame"], group[metric], label=variant, linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def _flip_for_display_keep_anchor(pos_yup, anchor_idx, target_anchor_xy):
    """
    將 y-up 座標翻成畫圖用座標，但保持 anchor 固定在 target_anchor_xy。
    只做翻轉與平移，不做任何縮放。
    """
    tx, ty = float(target_anchor_xy[0]), float(target_anchor_xy[1])

    # 先繞 y = ty 這條水平線翻轉，而不是繞整個 screen height
    flipped = [[float(x), 2.0 * ty - float(y)] for x, y in pos_yup]

    # 保險起見，再做一次平移，確保 anchor 精準落在 target_anchor_xy
    dx = tx - flipped[anchor_idx][0]
    dy = ty - flipped[anchor_idx][1]

    return [[x + dx, y + dy] for x, y in flipped]


def _run_from_ablation_outdir(args, ablation_outdir: Path) -> None:
    from scripts.export_ablation_review import _assert_lcc_matches_ablation_config

    _assert_lcc_matches_ablation_config(ablation_outdir)
    config_path = ablation_outdir / "ablation_config.json"
    runs_path = ablation_outdir / "ablation_runs_by_seed.csv"
    positions_path = ablation_outdir / "ablation_final_positions_y_up_sim.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing AS config JSON: {config_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing AS runs CSV: {runs_path}")
    if not positions_path.exists():
        raise FileNotFoundError(f"Missing AS positions CSV: {positions_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_labels = list(config.get("anchor_labels") or get_anchor_labels())
    test_labels = list(config.get("test_labels") or [])
    if not test_labels:
        from library.data_io import get_test_site_labels

        test_labels = get_test_site_labels()
    anchor_lonlat = [tuple(gt_lonlat[dni[label]]) for label in anchor_labels]
    test_lonlat = [tuple(gt_lonlat[dni[label]]) for label in test_labels]
    data_sim = data_Li2sim(data)
    directional_data = uploading_directional_data()
    rp_sim = config.get("refer_pos_sim", refer_pos_sim)
    anchor_align_label = get_anchor_align_label()
    df_runs = pd.read_csv(runs_path)
    df_pos = pd.read_csv(positions_path)

    outdir = Path(args.outdir) if args.outdir else Path(OUTPUT_DIR) / "ch6_representative_from_ablation"
    outdir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    all_convergence_metrics = []
    for variant, group in df_runs.groupby("variant"):
        rep = _select_representative_seed_from_as(group)
        rep["variant"] = variant
        seed = rep["seed"]
        history = _rerun_as_variant_history(
            variant=variant,
            seed=seed,
            config=config,
            graph=graph,
            vertice=vertice,
            dni=dni,
            edges=edges,
            anchor_labels=anchor_labels,
            anchor_lonlat=anchor_lonlat,
        )
        safe_variant = variant.replace("/", "_").replace("\\", "_").replace(" ", "_")
        history_path = outdir / f"{safe_variant}_seed{seed}_pos_history_y_up_sim.csv"
        _write_history_csv(history_path, variant, seed, vertice, history)
        metrics_df = _as_convergence_metrics_for_history(
            variant=variant,
            seed=seed,
            history=history,
            vertice=vertice,
            dni=dni,
            data_sim=data_sim,
            directional_data=directional_data,
            anchor_labels=anchor_labels,
            anchor_lonlat=anchor_lonlat,
            test_labels=test_labels,
            test_lonlat=test_lonlat,
            rp_sim=rp_sim,
        )
        all_convergence_metrics.append(metrics_df)

        subset = df_pos[(df_pos["variant"] == variant) & (df_pos["seed"] == seed)]
        if subset.empty:
            raise ValueError(f"No AS positions for variant={variant!r}, seed={seed}")
        pred_by_label = {
            row["label"]: [float(row["x_y_up_sim"]), float(row["y_y_up_sim"])]
            for _, row in subset.iterrows()
        }
        final_yup = history[-1]
        as_final = [pred_by_label[label] for label in vertice]
        rep["rerun_final_delta_rmse_sim"] = float(
            np.sqrt(np.mean(np.sum((np.asarray(final_yup, dtype=float) - np.asarray(as_final, dtype=float)) ** 2, axis=1)))
        )
        final_metrics = metrics_df.iloc[-1].to_dict()
        rep["rerun_final_metrics"] = {
            metric: float(final_metrics[metric]) for metric in REPRESENTATIVE_METRICS
        }
        rep["rerun_metric_diffs"] = _verify_rerun_matches_as_metrics(
            rep,
            rep["rerun_final_metrics"],
            abs_tol=float(args.verify_abs_tol),
            rel_tol=float(args.verify_rel_tol),
        )
        rep["history_csv"] = str(history_path)
        wrong_dir = wrong_directions_nonflip(deepcopy(final_yup), vertice, dni)
        final_ydown = _flip_for_display_keep_anchor(
            deepcopy(final_yup),
            anchor_idx=dni[anchor_align_label],
            target_anchor_xy=tuple(refer_pos_screen),
        )
        if not args.skip_errormap:
            visualize_error_map_official(
                deepcopy(final_ydown),
                vertice,
                dni,
                data,
                wrong_dir,
                zoom_area=None,
                file_name=f"AS_{safe_variant}_seed{seed}_",
                wait=not args.no_wait,
            )
        if not args.skip_overlay:
            ground_truth_comparison(
                vertice,
                dni,
                data_sim,
                deepcopy(gt_lonlat),
                final_ydown[dni[anchor_align_label]],
                deepcopy(final_ydown),
                file_name=f"AS_{safe_variant}_seed{seed}_",
                wait=not args.no_wait,
            )
        meta[variant] = rep

    if all_convergence_metrics and not args.skip_convergence:
        df_convergence = pd.concat(all_convergence_metrics, ignore_index=True)
        convergence_csv = outdir / "representative_as_convergence_metrics.csv"
        df_convergence.to_csv(convergence_csv, index=False, encoding="utf-8-sig")
        convergence_png = outdir / "representative_as_convergence_metrics.png"
        _plot_as_convergence(df_convergence, convergence_png)
        print(f"[Saved] {convergence_csv}")
        print(f"[Saved] {convergence_png}")

    meta_path = outdir / "representative_runs_from_ablation_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {meta_path}")
    print("[INFO] AS mode selects representative seeds from AS metrics, reruns those seeds, and uses rerun histories for convergence.")

def main():
    args = _parse_args()
    if args.ablation_outdir:
        _run_from_ablation_outdir(args, Path(args.ablation_outdir))
        return

    fixed_labels = [x.strip() for x in args.fixed.split(",") if x.strip()] or get_anchor_labels()
    anchor_align_label = get_anchor_align_label()

    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    directional_data = uploading_directional_data()
    fixed_lonlat = [tuple(gt_lonlat[dni[n]]) for n in fixed_labels]
    data_sim = data_Li2sim(data)

    # --- Load CSV histories ---
    sm_all, _ = _read_model_csv(args.csv_sm)
    dm_all, _ = _read_model_csv(args.csv_dm)
    ph_all, _ = _read_model_csv(args.csv_ph)

    # --- Select representative runs ---
    rep = {}
    for name, all_hist in [("SMACOF", sm_all), ("DC-SMACOF", dm_all), ("PhysicsSim", ph_all)]:
        rep[name] = select_representative_run(
            all_hist, dni, gt_lonlat, directional_data,
            refer_pos_screen, data_sim,
            model_name=name, verbose=True,
        )

    # --- Save metadata ---
    outdir = Path(args.outdir) if args.outdir else Path(OUTPUT_DIR) / "ch6_representative"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name in rep:
        r = rep[name]
        meta[name] = {
            "run_idx": r["run_idx"],
            "metrics": r["metrics"],
            "median_vector": r["median_vector"],
            "mad_vector": r["mad_vector"],
            "std_distance": r["std_distance"],
        }
    meta_path = outdir / "representative_runs_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {meta_path}")

    # ===================================================================
    # 6.2 Convergence (Stress+RMSE and VR+MAE)
    # ===================================================================
    if not args.skip_convergence:
        hist_ph = rep["PhysicsSim"]["history"]
        hist_dm = rep["DC-SMACOF"]["history"]
        hist_sm = rep["SMACOF"]["history"]

        plot_three_model_convergence_pygame_pixelaware(
            hist_ph, hist_dm, hist_sm,
            vertice=vertice, dni=dni, data=data,
            ground_truth_positions=gt_lonlat,
            fixed_point_labels=fixed_labels,
            fixed_point_lonlat=fixed_lonlat,
            refer_pos=tuple(refer_pos_screen),
            orientation="north-up",
            pre_process=True,   # ← CSV data 已是 pixel/y-up
        )

        plot_three_model_direction_convergence(
            hist_ph, hist_dm, hist_sm,
            vertice=vertice, dni=dni, data=data,
            ground_truth_positions=gt_lonlat,
            directional_data=directional_data,
            fixed_point_labels=fixed_labels,
            fixed_point_lonlat=fixed_lonlat,
            refer_pos=tuple(refer_pos_screen),
            orientation="north-up",
            pre_process=True,
            bin_size_sm_vr=25,
            bin_size_sm_mae=25,
            bin_size_dm_vr=5,
            bin_size_dm_mae=5,
            bin_size_ph_vr=15,
            bin_size_ph_mae=15,
        )

    # ===================================================================
    # 6.3 Error map  &  6.4 Overlay  (per model)
    # ===================================================================
    for name in ["SMACOF", "DC-SMACOF", "PhysicsSim"]:
        final_yup = rep[name]["final_pos"]

        # wrong_directions needs y-up
        wrong_dir = wrong_directions_nonflip(deepcopy(final_yup), vertice, dni)

        final_ydown = _flip_for_display_keep_anchor(
            deepcopy(final_yup),
            anchor_idx=dni[anchor_align_label],
            target_anchor_xy=tuple(refer_pos_screen),
        )

        if not args.skip_errormap:
            visualize_error_map_official(
                deepcopy(final_ydown), vertice, dni, data, wrong_dir,
                zoom_area=None, file_name=f"{name}_",
                wait=not args.no_wait,
            )

        if not args.skip_overlay:
            ground_truth_comparison(
                vertice, dni, data_sim, deepcopy(gt_lonlat),
                final_ydown[dni[anchor_align_label]],
                deepcopy(final_ydown),
                file_name=f"{name}_",
                wait=not args.no_wait,
            )


if __name__ == "__main__":
    main()
