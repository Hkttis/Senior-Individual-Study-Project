"""run_paper_script.ch5_ablation_study

Ablation study using the selected Full-model HPO parameters.

Variants:
  - PhysicsSim-Full
  - PhysicsSim-NoRep
  - PhysicsSim-NoDir
  - PhysicsSim-DistOnly
  - SMACOF
  - DC-SMACOF

Outputs include metrics and the final model positions for each variant/seed.
The script also exports summary statistics and paired comparisons.

Usage
-----
python -m run_paper_script.paper_run ch5-ablation --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_3x3_seed0_final --seeds 0 --outdir outputs/ch5_ablation_smoke
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import (
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters
from library.initialization import generate_CHEN_initial_positions
from library.metrics import (
    alignment_and_scaling,
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
    procrustes_align_by_fixed_points,
)
from library.model_cmp import run_directed_MDS
from library.physics import main_physics_simulation
from library.units import data_Li2sim, pos_matrix_sim2km
from MDS_model.stress_majorization_mds_model import stress_majorization
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _rmse_labels_km,
    _weights_from_alpha_beta,
)


PHYSICS_VARIANTS = {
    "PhysicsSim-Full": {"use_direction": True, "use_repulsion": True},
    "PhysicsSim-NoRep": {"use_direction": True, "use_repulsion": False},
    "PhysicsSim-NoDir": {"use_direction": False, "use_repulsion": True},
    "PhysicsSim-DistOnly": {"use_direction": False, "use_repulsion": False},
}

METRIC_COLS = [
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "RMSE_test_km",
    "min_pairwise_distance_km",
    "median_pairwise_distance_km",
]

PAIRED_METRIC_COLS = [
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "RMSE_test_km",
]

PAIRED_COMPARISONS = [
    ("PhysicsSim-Full", "PhysicsSim-NoRep", "repulsion_given_direction"),
    ("PhysicsSim-Full", "PhysicsSim-NoDir", "direction_given_repulsion"),
    ("PhysicsSim-NoRep", "PhysicsSim-DistOnly", "direction_without_repulsion"),
    ("PhysicsSim-NoDir", "PhysicsSim-DistOnly", "repulsion_without_direction"),
    ("PhysicsSim-DistOnly", "SMACOF", "dist_only_vs_smacof"),
    ("PhysicsSim-NoRep", "DC-SMACOF", "direction_info_matched"),
]


def _parse_seed_list(raw: str) -> List[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _load_selected_hpo_params(hpo_outdir: str | Path) -> tuple[float, float]:
    hpo_outdir = Path(hpo_outdir)
    candidate_csv = hpo_outdir / "selected_candidate_summary.csv"
    if candidate_csv.exists():
        df = pd.read_csv(candidate_csv)
        if df.empty:
            raise ValueError(f"selected_candidate_summary.csv is empty: {candidate_csv}")
        return float(df.iloc[0]["alpha"]), float(df.iloc[0]["beta"])

    summary_path = hpo_outdir / "selected_final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Neither selected_candidate_summary.csv nor selected_final_summary.json found in {hpo_outdir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["alpha"]), float(summary["beta"])


def _dc_smacof_weights_from_alpha(alpha: float, w_weight: float = 1.0) -> tuple[float, float]:
    distance_weight = float(w_weight)
    direction_weight = float(distance_weight * math.pow(10.0, float(alpha)))
    return distance_weight, direction_weight


def _load_selected_dc_smacof_params(
    dc_hpo_outdir: str | Path | None = None,
    dc_alpha: float | None = None,
) -> dict:
    if dc_alpha is not None:
        w_weight_value, v_weight_value = _dc_smacof_weights_from_alpha(float(dc_alpha))
        return {
            "source": "manual_cli_alpha",
            "hpo_outdir": str(dc_hpo_outdir) if dc_hpo_outdir else "",
            "alpha": float(dc_alpha),
            "w_weight": w_weight_value,
            "v_weight": v_weight_value,
        }

    if not dc_hpo_outdir:
        default_alpha = -0.5
        w_weight_value, v_weight_value = _dc_smacof_weights_from_alpha(default_alpha)
        return {
            "source": "default_selected_alpha",
            "hpo_outdir": "",
            "alpha": default_alpha,
            "w_weight": w_weight_value,
            "v_weight": v_weight_value,
        }

    dc_hpo_path = Path(dc_hpo_outdir)
    candidate_csv = dc_hpo_path / "dc_smacof_selected_candidate.csv"
    if not candidate_csv.exists():
        raise FileNotFoundError(
            f"DC-SMACOF HPO folder has no selected candidate: {candidate_csv}. "
            "If the Pareto front was small, choose one manually with --dc-alpha."
        )
    df = pd.read_csv(candidate_csv)
    if df.empty:
        raise ValueError(f"dc_smacof_selected_candidate.csv is empty: {candidate_csv}")
    row = df.iloc[0]
    alpha = float(row["alpha"])
    w_weight_value = float(row.get("w_weight", 1.0))
    v_weight_value = float(row.get("v_weight", _dc_smacof_weights_from_alpha(alpha, w_weight_value)[1]))
    return {
        "source": "dc_hpo_selected_candidate",
        "hpo_outdir": str(dc_hpo_path),
        "alpha": alpha,
        "w_weight": w_weight_value,
        "v_weight": v_weight_value,
    }


def _variant_forces(
    variant: str,
    *,
    alpha: float,
    beta: float,
    w_dis: float,
    base_spring_stiffness: float,
    base_directional_force: float,
    base_repulsion_strength: float,
) -> tuple[float, float, float, float, float]:
    w_dir, w_reg, spring, directional, repulsion = _weights_from_alpha_beta(
        alpha,
        beta,
        w_dis,
        base_spring_stiffness,
        base_directional_force,
        base_repulsion_strength,
    )
    spec = PHYSICS_VARIANTS[variant]
    if not spec["use_direction"]:
        directional = 0.0
    if not spec["use_repulsion"]:
        repulsion = 0.0
    return w_dir, w_reg, spring, directional, repulsion


def _bootstrap_ci_mean(values: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def _series_stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    q25, q75 = np.percentile(arr, [25, 75])
    ci_lo, ci_hi = _bootstrap_ci_mean(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "se": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan"),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


def _pairwise_distance_stats_km(pos_y_up_sim: np.ndarray) -> tuple[float, float]:
    pos_km = np.asarray(pos_matrix_sim2km(pos_y_up_sim.tolist()), dtype=float)
    distances: list[float] = []
    for i in range(len(pos_km)):
        for j in range(i + 1, len(pos_km)):
            distances.append(float(np.linalg.norm(pos_km[i] - pos_km[j])))
    if not distances:
        return float("nan"), float("nan")
    arr = np.asarray(distances, dtype=float)
    return float(arr.min()), float(np.median(arr))


def _evaluate_positions(
    *,
    model: str,
    variant: str,
    seed: int,
    pos_y_up_sim: Sequence[Sequence[float]],
    vertice: Sequence[str],
    dni: Dict[str, int],
    data_sim: Sequence[Sequence[object]],
    directional_data: Sequence[Sequence[str]],
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
    test_labels: Sequence[str],
    test_lonlat: Sequence[Tuple[float, float]],
    refer_pos_sim: Sequence[float],
    spring_stiffness: float,
    directional_force: float,
    repulsion_strength: float,
    status: str = "ok",
    error: str = "",
) -> tuple[dict, list[dict]]:
    pos = np.asarray([(float(p[0]), float(p[1])) for p in pos_y_up_sim], dtype=float)
    e_distance = float(calculate_kruskals_stress(dni, pos_matrix_sim2km(pos.tolist()), data_sim))
    e_direction = float(direction_violation_rate(pos, directional_data, dni))
    e_mae = float(mean_angular_error_violations(pos, directional_data, dni))
    min_pairwise_km, median_pairwise_km = _pairwise_distance_stats_km(pos)
    rmse_test = float(
        _rmse_labels_km(
            pos_y_up_sim=pos,
            dni=dni,
            refer_pos_sim=refer_pos_sim,
            gt_labels=list(anchor_labels) + list(test_labels),
            gt_lonlat=list(anchor_lonlat) + list(test_lonlat),
            eval_labels=test_labels,
            anchor_label_for_frame=anchor_labels[0],
        )
    )

    metrics = {
        "model": model,
        "variant": variant,
        "seed": int(seed),
        "status": status,
        "error": error,
        "spring_stiffness": float(spring_stiffness),
        "directional_force": float(directional_force),
        "repulsion_strength": float(repulsion_strength),
        "E_distance_stress": e_distance,
        "E_direction_vr": e_direction,
        "E_direction_mae": e_mae,
        "RMSE_test_km": rmse_test,
        "min_pairwise_distance_km": min_pairwise_km,
        "median_pairwise_distance_km": median_pairwise_km,
    }
    positions = [
        {
            "model": model,
            "variant": variant,
            "seed": int(seed),
            "label": label,
            "x_y_up_sim": float(pos[dni[label]][0]),
            "y_y_up_sim": float(pos[dni[label]][1]),
        }
        for label in vertice
    ]
    return metrics, positions


def _failure_row(
    *,
    model: str,
    variant: str,
    seed: int,
    spring_stiffness: float = float("nan"),
    directional_force: float = float("nan"),
    repulsion_strength: float = float("nan"),
    error: Exception,
) -> dict:
    return {
        "model": model,
        "variant": variant,
        "seed": int(seed),
        "status": "failed",
        "error": str(error),
        "spring_stiffness": float(spring_stiffness),
        "directional_force": float(directional_force),
        "repulsion_strength": float(repulsion_strength),
        "E_distance_stress": float("nan"),
        "E_direction_vr": float("nan"),
        "E_direction_mae": float("nan"),
        "RMSE_test_km": float("nan"),
        "min_pairwise_distance_km": float("nan"),
        "median_pairwise_distance_km": float("nan"),
    }


def _build_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    ok = df_runs[df_runs["status"] == "ok"]
    for (model, variant), group in ok.groupby(["model", "variant"]):
        row = {"model": model, "variant": variant}
        for col in METRIC_COLS:
            stats = _series_stats(group[col].tolist())
            for key, value in stats.items():
                row[f"{col}_{key}"] = value
        summary_rows.append(row)
    if not summary_rows:
        return pd.DataFrame()
    return pd.DataFrame(summary_rows).sort_values(["model", "variant"]).reset_index(drop=True)


def _build_paired_comparisons(df_runs: pd.DataFrame) -> pd.DataFrame:
    ok = df_runs[df_runs["status"] == "ok"].copy()
    rows: list[dict] = []
    for left, right, comparison in PAIRED_COMPARISONS:
        left_df = ok[ok["variant"] == left].set_index("seed")
        right_df = ok[ok["variant"] == right].set_index("seed")
        common_seeds = sorted(set(left_df.index).intersection(right_df.index))
        if not common_seeds:
            continue
        for metric in PAIRED_METRIC_COLS:
            diffs = np.asarray(
                [float(left_df.loc[seed, metric]) - float(right_df.loc[seed, metric]) for seed in common_seeds],
                dtype=float,
            )
            diffs = diffs[np.isfinite(diffs)]
            stats = _series_stats(diffs)
            rows.append(
                {
                    "comparison": comparison,
                    "left_variant": left,
                    "right_variant": right,
                    "metric": metric,
                    "diff_definition": "left_minus_right",
                    "lower_is_better": True,
                    "n_pairs": int(diffs.size),
                    "paired_diff_mean": stats["mean"],
                    "paired_diff_std": stats["std"],
                    "paired_diff_se": stats["se"],
                    "paired_diff_median": stats["median"],
                    "paired_diff_iqr": stats["iqr"],
                    "paired_diff_ci95_lo": stats["ci95_lo"],
                    "paired_diff_ci95_hi": stats["ci95_hi"],
                    "left_better_win_rate": float(np.mean(diffs < 0.0)) if diffs.size else float("nan"),
                    "tie_rate": float(np.mean(diffs == 0.0)) if diffs.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _run_physics_variant(
    *,
    variant: str,
    seed: int,
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
    refer_pos_sim: Sequence[float],
    spring_stiffness: float,
    directional_force: float,
    repulsion_strength: float,
) -> tuple[List[str], Dict[str, int], np.ndarray]:
    np.random.seed(seed)
    vertice, dni, data_li, pos_init, fixed_positions_list = generate_CHEN_initial_positions(
        list(refer_pos_sim),
        list(anchor_labels),
        list(anchor_lonlat),
        anchor_label=anchor_labels[0],
    )
    directional_data = uploading_directional_data()
    _wrong, _stress_history, _pos_history, pos_final = main_physics_simulation(
        vertice,
        dni,
        data_Li2sim(data_li),
        pos_init,
        directional_data,
        fixed_positions_list,
        spring_stiffness,
        repulsion_strength,
        directional_force,
        plot=False,
    )
    return vertice, dni, np.asarray([(float(p[0]), float(p[1])) for p in pos_final], dtype=float)


def _run_smacof_baseline(
    *,
    seed: int,
    graph,
    vertice,
    dni,
    edges,
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
    refer_pos_sim: Sequence[float],
) -> np.ndarray:
    np.random.seed(seed)
    pos_li, _stress_history, _pos_history = stress_majorization(graph, dni, vertice, edges)
    pos_px = alignment_and_scaling(pos_li, vertice, dni, refer_pos_sim, y_down=False)
    pos_px = procrustes_align_by_fixed_points(
        deepcopy(pos_px),
        list(anchor_labels),
        list(anchor_lonlat),
        dni,
        refer_pos=refer_pos_sim,
        anchor_label=anchor_labels[0],
    )
    return np.asarray(pos_px, dtype=float)


def _run_directed_mds_baseline(
    *,
    seed: int,
    vertice,
    dni,
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
    refer_pos_sim: Sequence[float],
    dc_w_weight: float | None = None,
    dc_v_weight: float | None = None,
) -> np.ndarray:
    np.random.seed(seed)
    pos_history_li = run_directed_MDS(
        vis=False,
        w_weight_value=dc_w_weight,
        v_weight_value=dc_v_weight,
    )
    pos_li = pos_history_li[-1]
    # DC-SMACOF already uses directional information, so do not apply
    # Procrustes rotation/reflection by anchors here.
    pos_px = alignment_and_scaling(
        pos_li,
        vertice,
        dni,
        refer_pos_sim,
        y_down=False,
        anchor_label=anchor_labels[0],
    )
    return np.asarray(pos_px, dtype=float)


def run_ablation_study(
    *,
    hpo_outdir: str | Path,
    seeds: Sequence[int],
    outdir: str | Path | None = None,
    include_baselines: bool = True,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
    dc_hpo_outdir: str | Path | None = None,
    dc_alpha: float | None = None,
    overwrite: bool = False,
) -> dict:
    outdir_path = Path(outdir) if outdir else Path(OUTPUT_DIR) / "ch5_ablation_study"
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Ablation outdir already exists and is not empty: {outdir_path}. "
            "Choose a new --outdir or pass --overwrite intentionally."
        )
    outdir_path.mkdir(parents=True, exist_ok=True)
    alpha, beta = _load_selected_hpo_params(hpo_outdir)
    dc_params = _load_selected_dc_smacof_params(dc_hpo_outdir, dc_alpha)

    graph, vertice, dni, edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)
    directional_data = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    anchor_lonlat = [tuple(gt_lonlat[dni[label]]) for label in anchor_labels]
    test_lonlat = [tuple(gt_lonlat[dni[label]]) for label in test_labels]

    run_rows: list[dict] = []
    position_rows: list[dict] = []

    for seed in seeds:
        for variant in PHYSICS_VARIANTS:
            w_dir, w_reg, spring, directional, repulsion = _variant_forces(
                variant,
                alpha=alpha,
                beta=beta,
                w_dis=w_dis,
                base_spring_stiffness=base_spring_stiffness,
                base_directional_force=base_directional_force,
                base_repulsion_strength=base_repulsion_strength,
            )
            try:
                vtx, v_dni, pos = _run_physics_variant(
                    variant=variant,
                    seed=int(seed),
                    anchor_labels=anchor_labels,
                    anchor_lonlat=anchor_lonlat,
                    refer_pos_sim=refer_pos_sim,
                    spring_stiffness=spring,
                    directional_force=directional,
                    repulsion_strength=repulsion,
                )
                metrics, positions = _evaluate_positions(
                    model="PhysicsSim",
                    variant=variant,
                    seed=int(seed),
                    pos_y_up_sim=pos,
                    vertice=vtx,
                    dni=v_dni,
                    data_sim=data_sim,
                    directional_data=directional_data,
                    anchor_labels=anchor_labels,
                    anchor_lonlat=anchor_lonlat,
                    test_labels=test_labels,
                    test_lonlat=test_lonlat,
                    refer_pos_sim=refer_pos_sim,
                    spring_stiffness=spring,
                    directional_force=directional,
                    repulsion_strength=repulsion,
                )
                metrics["w_dir"] = w_dir
                metrics["w_reg"] = w_reg
                run_rows.append(metrics)
                position_rows.extend(positions)
            except Exception as exc:
                run_rows.append(
                    _failure_row(
                        model="PhysicsSim",
                        variant=variant,
                        seed=int(seed),
                        spring_stiffness=spring,
                        directional_force=directional,
                        repulsion_strength=repulsion,
                        error=exc,
                    )
                )

        if include_baselines:
            for model, runner in [
                ("SMACOF", _run_smacof_baseline),
                ("DC-SMACOF", _run_directed_mds_baseline),
            ]:
                try:
                    if model == "SMACOF":
                        pos = runner(
                            seed=int(seed),
                            graph=graph,
                            vertice=vertice,
                            dni=dni,
                            edges=edges,
                            anchor_labels=anchor_labels,
                            anchor_lonlat=anchor_lonlat,
                            refer_pos_sim=refer_pos_sim,
                        )
                    else:
                        pos = runner(
                            seed=int(seed),
                            vertice=vertice,
                            dni=dni,
                            anchor_labels=anchor_labels,
                            anchor_lonlat=anchor_lonlat,
                            refer_pos_sim=refer_pos_sim,
                            dc_w_weight=dc_params["w_weight"],
                            dc_v_weight=dc_params["v_weight"],
                        )
                    metrics, positions = _evaluate_positions(
                        model=model,
                        variant=model,
                        seed=int(seed),
                        pos_y_up_sim=pos,
                        vertice=vertice,
                        dni=dni,
                        data_sim=data_sim,
                        directional_data=directional_data,
                        anchor_labels=anchor_labels,
                        anchor_lonlat=anchor_lonlat,
                        test_labels=test_labels,
                        test_lonlat=test_lonlat,
                        refer_pos_sim=refer_pos_sim,
                        spring_stiffness=float("nan"),
                        directional_force=float("nan"),
                        repulsion_strength=float("nan"),
                    )
                    run_rows.append(metrics)
                    position_rows.extend(positions)
                except Exception as exc:
                    run_rows.append(_failure_row(model=model, variant=model, seed=int(seed), error=exc))

    df_runs = pd.DataFrame(run_rows)
    df_positions = pd.DataFrame(position_rows)
    df_summary = _build_summary(df_runs)
    df_paired = _build_paired_comparisons(df_runs)

    df_runs.to_csv(outdir_path / "ablation_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir_path / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    df_paired.to_csv(outdir_path / "ablation_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    df_positions.to_csv(outdir_path / "ablation_final_positions_y_up_sim.csv", index=False, encoding="utf-8-sig")
    config = {
        "hpo_outdir": str(hpo_outdir),
        "alpha": alpha,
        "beta": beta,
        "dc_smacof_hpo": dc_params,
        "seeds": list(map(int, seeds)),
        "anchor_labels": list(anchor_labels),
        "test_labels": list(test_labels),
        "lcc_bounds": dict(
            zip(["lon_min", "lon_max", "lat_min", "lat_max"], map(float, get_lcc_bounds()))
        ),
        "lcc_parameters": dict(zip(["lat_1", "lat_2", "lon_0"], map(float, get_lcc_parameters()))),
        "lcc_standard_parallel_rule": "lat_1=lat_min+(lat_max-lat_min)/6; lat_2=lat_max-(lat_max-lat_min)/6",
        "lcc_bounds_source": FILE_PATHS["ground_truth_path"],
        "include_baselines": bool(include_baselines),
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
        "refer_pos_sim": list(map(float, refer_pos_sim)),
        "physics_variants": PHYSICS_VARIANTS,
        "outputs": {
            "runs": "ablation_runs_by_seed.csv",
            "summary": "ablation_summary.csv",
            "paired_comparisons": "ablation_paired_comparisons.csv",
            "final_positions": "ablation_final_positions_y_up_sim.csv",
        },
        "metrics": METRIC_COLS,
        "paired_metrics": PAIRED_METRIC_COLS,
        "summary_only_diagnostic_metrics": [
            "min_pairwise_distance_km",
            "median_pairwise_distance_km",
        ],
        "paired_comparisons": [
            {"left_variant": left, "right_variant": right, "comparison": comparison}
            for left, right, comparison in PAIRED_COMPARISONS
        ],
    }
    (outdir_path / "ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[Saved] {outdir_path / 'ablation_runs_by_seed.csv'}")
    print(f"[Saved] {outdir_path / 'ablation_summary.csv'}")
    print(f"[Saved] {outdir_path / 'ablation_paired_comparisons.csv'}")
    print(f"[Saved] {outdir_path / 'ablation_final_positions_y_up_sim.csv'}")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    return {"df_runs": df_runs, "df_summary": df_summary, "df_paired": df_paired, "outdir": outdir_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhysicsSim ablation variants and baseline comparison.")
    parser.add_argument(
        "--hpo-outdir",
        required=True,
        help="HPO output directory with selected_candidate_summary.csv or selected_final_summary.json.",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--outdir", default="", help="Output directory.")
    parser.add_argument("--no-baselines", action="store_true", help="Run only PhysicsSim variants.")
    parser.add_argument(
        "--dc-hpo-outdir",
        default="",
        help="DC-SMACOF HPO output directory with dc_smacof_selected_candidate.csv.",
    )
    parser.add_argument(
        "--dc-alpha",
        type=float,
        default=None,
        help="Manual DC-SMACOF alpha=log10(v_weight/w_weight), used when no selected DC-HPO candidate exists.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty outdir.")
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ablation_study(
        hpo_outdir=args.hpo_outdir,
        seeds=_parse_seed_list(args.seeds),
        outdir=args.outdir,
        include_baselines=not args.no_baselines,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        refer_pos_sim=DEFAULT_REFER_POS_SIM,
        dc_hpo_outdir=args.dc_hpo_outdir or None,
        dc_alpha=args.dc_alpha,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
