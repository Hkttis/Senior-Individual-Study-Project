"""PhysicsSim positional-stability experiment using the selected HPO candidate.

The procedure repeats PhysicsSim with random initial positions and Gaussian
perturbations in the selected HPO alpha/beta space. It is a parameter-perturbation repeated-simulation analysis, not
an observation-resampling bootstrap.

Run from the physics_simulation project root, for example:

python -m run_paper_script.paper_run ch5-bootstrap \
  --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2_manual_alpha_1_beta_-0.5 \
  --n-bootstrap 300 --alpha-jitter 0.05 --beta-jitter 0.05 \
  --outdir outputs/ch5_bootstrap_physics_alpha_1_beta_-0.5_jitter0p05_300runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.anchor_frame import px_list_to_km_list
from library.bootstrap_and_visualization import (
    bootstrap_dynamics,
    plot_appendix_hdr_panels,
    plot_appendix_stability_overview,
    plot_ellipse_outline_map,
    plot_hdr_small_multiples,
    plot_kde_combined,
    plot_multi_ellipses,
    plot_relative_stability_map,
)
from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2pix,
    refer_pos_sim,
)
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    uploading_ground_truth,
)
from library.geometry import (
    get_lcc_bounds,
    get_lcc_parameters,
    inverse_lcc_transformation,
    lcc_transformation_with_anchor,
)
from library.initialization import load_ini_data_from_csv
from run_paper_script.ch5_ablation_progressive import (
    _as_lcc_mapping,
    _assert_same_lcc,
    _resolve_physics_hpo_config,
)
from run_paper_script.ch5_ablation_study import (
    _load_selected_hpo_params,
    _variant_forces,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_physics_hpo_selection(hpo_outdir: str | Path) -> dict:
    hpo_path = Path(hpo_outdir)
    alpha, beta = _load_selected_hpo_params(hpo_path)
    w_dir, w_reg, spring, directional, repulsion = _variant_forces(
        "PhysicsSim-Full",
        alpha=float(alpha),
        beta=float(beta),
        w_dis=1.0,
        base_spring_stiffness=SPRING_STIFFNESS_BASE,
        base_directional_force=DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        base_repulsion_strength=REPULSION_STRENGTH_BASE,
    )
    candidate_path = hpo_path / "selected_candidate_summary.csv"
    summary_path = hpo_path / "selected_final_summary.json"
    source_path = candidate_path if candidate_path.exists() else summary_path
    if not source_path.exists():
        raise FileNotFoundError(f"Selected PhysicsSim HPO result is missing from {hpo_path}")

    if candidate_path.exists():
        candidate = pd.read_csv(candidate_path)
        if candidate.empty:
            raise ValueError(f"Selected PhysicsSim HPO result is empty: {candidate_path}")
        row = candidate.iloc[0]
        expected = {
            "spring_stiffness": spring,
            "directional_force": directional,
            "repulsion_strength": repulsion,
        }
        for column, value in expected.items():
            if column in candidate.columns and not np.isclose(float(row[column]), float(value), rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Selected HPO {column} does not match alpha/beta conversion: {row[column]} != {value}"
                )

    return {
        "hpo_outdir": str(hpo_path),
        "selected_result_file": str(source_path),
        "selected_result_sha256": _sha256(source_path),
        "alpha": float(alpha),
        "beta": float(beta),
        "w_dis": 1.0,
        "w_dir": float(w_dir),
        "w_reg": float(w_reg),
        "spring_stiffness": float(spring),
        "directional_force": float(directional),
        "repulsion_strength": float(repulsion),
    }


def _validate_hpo_provenance(
    hpo_outdir: str | Path,
    calibration_labels: Sequence[str],
    test_labels: Sequence[str],
) -> dict:
    config_path, config = _resolve_physics_hpo_config(hpo_outdir)
    _assert_same_lcc("PhysicsSim HPO", config["lcc_bounds"], config["lcc_parameters"])
    if list(config.get("anchor_labels", [])) != list(calibration_labels):
        raise ValueError("PhysicsSim HPO anchor labels do not match current site-point roles.")
    if list(config.get("test_labels", [])) != list(test_labels):
        raise ValueError("PhysicsSim HPO test labels do not match current site-point roles.")
    return {
        "physics_hpo_config": str(config_path),
        "physics_hpo_config_sha256": _sha256(config_path),
        "lcc_matches_current_data": True,
        "site_roles_match_current_data": True,
    }


def _target_positions_sim(dni, gt_lonlat, anchor_label: str) -> dict[str, np.ndarray]:
    projected = lcc_transformation_with_anchor(dni, gt_lonlat, anchor_label=anchor_label)
    targets: dict[str, np.ndarray] = {}
    for label, index in dni.items():
        x_km, y_km = projected[index]
        if x_km is None or y_km is None:
            continue
        targets[label] = np.asarray(
            [float(refer_pos_sim[0]) + float(x_km) * km2pix, float(refer_pos_sim[1]) + float(y_km) * km2pix],
            dtype=float,
        )
    return targets


def _samples_to_lonlat(samples: np.ndarray, dni, gt_lonlat, anchor_label: str) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    flat = values.reshape((-1, 2))
    positions_km = px_list_to_km_list(flat, tuple(refer_pos_sim), km2pix)
    origin_lonlat = gt_lonlat[dni[anchor_label]]
    lonlat = np.asarray(inverse_lcc_transformation(positions_km, origin_lonlat), dtype=float)
    return lonlat.reshape(values.shape)


def _write_outputs(
    *,
    outdir: Path,
    samples: np.ndarray,
    samples_lonlat: np.ndarray,
    vertice: Sequence[str],
    run_metadata: Sequence[dict],
    ellipse_summary: Sequence[dict],
    kde_status: Sequence[dict],
    stability_summary: Sequence[dict],
    hdr_selection: Sequence[dict],
    empirical_stability_summary: Sequence[dict],
    appendix_hdr_selection: Sequence[dict],
) -> dict[str, str]:
    run_df = pd.DataFrame(run_metadata)
    run_df.to_csv(outdir / "bootstrap_run_parameters.csv", index=False, encoding="utf-8-sig")

    sample_rows = []
    lonlat_rows = []
    for bootstrap_index, metadata in enumerate(run_metadata):
        for node_index, label in enumerate(vertice):
            sample_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "simulation_seed": metadata["simulation_seed"],
                    "label": label,
                    "x_y_up_sim": float(samples[bootstrap_index, node_index, 0]),
                    "y_y_up_sim": float(samples[bootstrap_index, node_index, 1]),
                }
            )
            lonlat_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "simulation_seed": metadata["simulation_seed"],
                    "label": label,
                    "lon": float(samples_lonlat[bootstrap_index, node_index, 0]),
                    "lat": float(samples_lonlat[bootstrap_index, node_index, 1]),
                }
            )
    pd.DataFrame(sample_rows).to_csv(
        outdir / "bootstrap_samples_y_up_sim.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(lonlat_rows).to_csv(
        outdir / "bootstrap_samples_lonlat.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ellipse_summary).to_csv(
        outdir / "bootstrap_ellipse_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(kde_status).to_csv(outdir / "bootstrap_kde_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stability_summary).to_csv(
        outdir / "bootstrap_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(hdr_selection).to_csv(
        outdir / "bootstrap_hdr_selection.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(empirical_stability_summary).to_csv(
        outdir / "bootstrap_empirical_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(appendix_hdr_selection).to_csv(
        outdir / "bootstrap_appendix_hdr_selection.csv", index=False, encoding="utf-8-sig"
    )

    artifact_names = [
        "bootstrap_run_parameters.csv",
        "bootstrap_samples_y_up_sim.csv",
        "bootstrap_samples_lonlat.csv",
        "bootstrap_ellipse_summary.csv",
        "bootstrap_kde_status.csv",
        "confidence_ellipses.png",
        "confidence_ellipses.svg",
        "combined_kde_density.png",
        "combined_kde_density.svg",
        "bootstrap_stability_summary.csv",
        "bootstrap_hdr_selection.csv",
        "relative_stability_map.png",
        "relative_stability_map.svg",
        "dispersion_ellipses_95_outline.png",
        "dispersion_ellipses_95_outline.svg",
        "stability_hdr_small_multiples.png",
        "stability_hdr_small_multiples.svg",
        "bootstrap_empirical_stability_summary.csv",
        "bootstrap_appendix_hdr_selection.csv",
        "appendix_stability_overview.png",
        "appendix_stability_overview.svg",
        "appendix_stability_hdr.png",
        "appendix_stability_hdr.svg",
    ]
    return {name: _sha256(outdir / name) for name in artifact_names}


def run_bootstrap_stability(
    *,
    hpo_outdir: str | Path,
    outdir: str | Path,
    n_bootstrap: int = 300,
    alpha_jitter: float = 0.10,
    beta_jitter: float = 0.10,
    seed_start: int = 0,
    jitter_seed: int = 0,
    fixed_labels: Sequence[str] | None = None,
    overwrite: bool = False,
    kde_grid_size: int = 200,
) -> dict:
    output_path = Path(outdir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. Choose a new --outdir or pass --overwrite intentionally."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    selection = _load_physics_hpo_selection(hpo_outdir)
    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    calibration_labels = list(fixed_labels) if fixed_labels else get_anchor_labels()
    anchor_label = get_anchor_align_label()
    test_labels = get_test_site_labels()
    if len(calibration_labels) != 3 or anchor_label not in calibration_labels:
        raise ValueError("Formal PhysicsSim stability analysis requires three calibration anchors including anchor_align.")
    provenance = _validate_hpo_provenance(hpo_outdir, calibration_labels, test_labels)
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]

    samples, run_vertice, run_dni, run_metadata = bootstrap_dynamics(
        int(n_bootstrap),
        float(alpha_jitter),
        float(beta_jitter),
        calibration_labels,
        calibration_lonlat,
        alpha=selection["alpha"],
        beta=selection["beta"],
        w_dis=selection["w_dis"],
        spring_stiffness=selection["spring_stiffness"],
        repulsion_strength=selection["repulsion_strength"],
        directional_force_magnitude=selection["directional_force"],
        anchor_label=anchor_label,
        seed_start=int(seed_start),
        jitter_seed=int(jitter_seed),
        return_run_metadata=True,
    )
    if list(run_vertice) != list(vertice) or dict(run_dni) != dict(dni):
        raise ValueError("Bootstrap node ordering does not match the formal input graph.")

    targets = _target_positions_sim(dni, gt_lonlat, anchor_label)
    anchor_drift_rows = []
    for label in calibration_labels:
        drift = np.linalg.norm(samples[:, dni[label], :] - targets[label], axis=1)
        anchor_drift_rows.append(
            {
                "label": label,
                "max_anchor_drift_sim": float(np.max(drift)),
                "mean_anchor_drift_sim": float(np.mean(drift)),
            }
        )
    max_anchor_drift = max(row["max_anchor_drift_sim"] for row in anchor_drift_rows)
    if max_anchor_drift > 1e-6:
        raise ValueError(f"Calibration anchor drift exceeded tolerance: {max_anchor_drift}")
    pd.DataFrame(anchor_drift_rows).to_csv(
        output_path / "bootstrap_anchor_drift.csv", index=False, encoding="utf-8-sig"
    )

    ellipse_paths = [output_path / "confidence_ellipses.png", output_path / "confidence_ellipses.svg"]
    kde_paths = [output_path / "combined_kde_density.png", output_path / "combined_kde_density.svg"]
    ellipse_summary = plot_multi_ellipses(samples, vertice, ellipse_paths)
    kde_status = plot_kde_combined(samples, vertice, kde_paths, grid_size=int(kde_grid_size))
    stability_summary = plot_relative_stability_map(
        samples,
        vertice,
        [output_path / "relative_stability_map.png", output_path / "relative_stability_map.svg"],
    )
    plot_ellipse_outline_map(
        samples,
        vertice,
        [output_path / "dispersion_ellipses_95_outline.png", output_path / "dispersion_ellipses_95_outline.svg"],
    )
    hdr_selection = plot_hdr_small_multiples(
        samples,
        vertice,
        [output_path / "stability_hdr_small_multiples.png", output_path / "stability_hdr_small_multiples.svg"],
    )
    empirical_stability_summary = plot_appendix_stability_overview(
        samples,
        vertice,
        [output_path / "appendix_stability_overview.png", output_path / "appendix_stability_overview.svg"],
        anchor_labels=calibration_labels,
        test_labels=test_labels,
    )
    appendix_hdr_selection = plot_appendix_hdr_panels(
        samples,
        vertice,
        [output_path / "appendix_stability_hdr.png", output_path / "appendix_stability_hdr.svg"],
        anchor_labels=calibration_labels,
    )
    samples_lonlat = _samples_to_lonlat(samples, dni, gt_lonlat, anchor_label)
    artifact_sha256 = _write_outputs(
        outdir=output_path,
        samples=samples,
        samples_lonlat=samples_lonlat,
        vertice=vertice,
        run_metadata=run_metadata,
        ellipse_summary=ellipse_summary,
        kde_status=kde_status,
        stability_summary=stability_summary,
        hdr_selection=hdr_selection,
        empirical_stability_summary=empirical_stability_summary,
        appendix_hdr_selection=appendix_hdr_selection,
    )
    artifact_sha256["bootstrap_anchor_drift.csv"] = _sha256(output_path / "bootstrap_anchor_drift.csv")

    config = {
        "experiment": "PhysicsSim positional stability under random initialization and HPO alpha/beta perturbation",
        "method_classification": "parameter_perturbation_repeated_simulation_not_observation_resampling",
        "hpo_selection": selection,
        "input_validation": provenance,
        "n_bootstrap": int(n_bootstrap),
        "seed_start": int(seed_start),
        "jitter_seed": int(jitter_seed),
        "perturbation_space": "HPO log10 alpha/beta with w_dis fixed at 1",
        "alpha_jitter_sd": float(alpha_jitter),
        "beta_jitter_sd": float(beta_jitter),
        "jitter_distribution": {
            "alpha": "selected alpha + Normal(mean=0, sd=alpha_jitter_sd)",
            "beta": "selected beta + Normal(mean=0, sd=beta_jitter_sd)",
            "w_dis": "fixed at the selected HPO value (1.0)",
            "spring_stiffness": "fixed at the selected HPO value",
            "directional_force": "selected force * 10^(sampled alpha - selected alpha)",
            "repulsion_strength": "selected force * 10^(sampled beta - selected beta)",
            "bootstrap_index_0": "unperturbed selected HPO alpha/beta and forces",
        },
        "calibration_labels": calibration_labels,
        "anchor_align_label": anchor_label,
        "test_labels": test_labels,
        "coordinate_frames": {
            "bootstrap_samples_y_up_sim.csv": "north-up simulation coordinates",
            "bootstrap_samples_lonlat.csv": "WGS84 longitude/latitude via inverse LCC",
        },
        "kde_degenerate_tolerance_sim": 1e-6,
        "kde_status_counts": {
            str(key): int(value)
            for key, value in pd.DataFrame(kde_status)["kde_status"].value_counts().items()
        },
        "refer_pos_sim": [float(value) for value in refer_pos_sim],
        "lcc_bounds": _as_lcc_mapping(get_lcc_bounds(), ("lon_min", "lon_max", "lat_min", "lat_max")),
        "lcc_parameters": _as_lcc_mapping(get_lcc_parameters(), ("lat_1", "lat_2", "lon_0")),
        "ground_truth_path": FILE_PATHS["ground_truth_path"],
        "distance_data_path": FILE_PATHS["chen_data"],
        "direction_data_path": FILE_PATHS["directional_data"],
        "max_anchor_drift_sim": float(max_anchor_drift),
        "failure_count": 0,
        "artifact_sha256": artifact_sha256,
    }
    (output_path / "bootstrap_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"outdir": output_path, "config": config, "samples": samples}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal PhysicsSim positional-stability analysis.")
    parser.add_argument("--hpo-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--alpha-jitter", type=float, default=0.05, help="Gaussian SD in log10 alpha space.")
    parser.add_argument("--beta-jitter", type=float, default=0.05, help="Gaussian SD in log10 beta space.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jitter-seed", type=int, default=0)
    parser.add_argument("--fixed", default="", help="Optional comma-separated calibration anchors; normally omitted.")
    parser.add_argument("--kde-grid-size", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixed_labels = [value.strip() for value in args.fixed.split(",") if value.strip()] or None
    result = run_bootstrap_stability(
        hpo_outdir=args.hpo_outdir,
        outdir=args.outdir,
        n_bootstrap=args.n_bootstrap,
        alpha_jitter=args.alpha_jitter,
        beta_jitter=args.beta_jitter,
        seed_start=args.seed_start,
        jitter_seed=args.jitter_seed,
        fixed_labels=fixed_labels,
        overwrite=args.overwrite,
        kde_grid_size=args.kde_grid_size,
    )
    print(f"[Saved] {result['outdir']}")


if __name__ == "__main__":
    main()
