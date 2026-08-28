from __future__ import annotations

from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from scipy.stats import chi2, gaussian_kde
from tqdm import trange

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2pix,
    refer_pos_sim,
)
from library.data_io import uploading_directional_data
from library.initialization import generate_CHEN_initial_positions
from library.physics import main_physics_simulation
from library.units import data_Li2sim


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_BOOTSTRAP_OUTPUT_DIR = OUTPUT_DIR / "ch5_bootstrap_stability"
ELLIPSES_FILE = str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "confidence_ellipses.png")
KDE_COMBINED_FILE = str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "combined_kde_density.png")


def _run_once(
    seed: int,
    fixed_point_labels: Sequence[str] = ("鄯善", "都護治/烏壘"),
    fixed_points_lonlat: Sequence[tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
    *,
    spring_stiffness: float = SPRING_STIFFNESS_BASE,
    repulsion_strength: float = REPULSION_STRENGTH_BASE,
    directional_force_magnitude: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    anchor_label: str | None = None,
):
    """Run one headless PhysicsSim realization in the north-up simulation frame."""
    forces = np.asarray(
        [spring_stiffness, repulsion_strength, directional_force_magnitude], dtype=float
    )
    if not np.all(np.isfinite(forces)) or np.any(forces <= 0.0):
        raise ValueError("PhysicsSim force parameters must be positive and finite.")

    np.random.seed(int(seed))
    vertice, dni, data, pos_matrix, fixed_pos = generate_CHEN_initial_positions(
        list(refer_pos_sim),
        fixed_point_labels=list(fixed_point_labels),
        fixed_points_lonlat=list(fixed_points_lonlat),
        anchor_label=anchor_label,
    )
    directional_data = uploading_directional_data()
    _wrong, _stress_history, _pos_history, final_pos = main_physics_simulation(
        vertice,
        dni,
        data_Li2sim(data),
        deepcopy(pos_matrix),
        directional_data,
        fixed_pos,
        float(spring_stiffness),
        float(repulsion_strength),
        float(directional_force_magnitude),
        plot=False,
    )
    result = np.asarray(final_pos, dtype=float)
    if result.shape != (len(vertice), 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"PhysicsSim returned invalid bootstrap positions for seed {seed}.")
    return result, vertice, dni


def bootstrap_dynamics(
    N_BOOTSTRAP: int,
    ALPHA_JITTER: float,
    BETA_JITTER: float,
    fixed_point_labels: Sequence[str],
    fixed_points_lonlat: Sequence[tuple[float, float]],
    *,
    alpha: float,
    beta: float,
    w_dis: float = 1.0,
    spring_stiffness: float = SPRING_STIFFNESS_BASE,
    repulsion_strength: float = REPULSION_STRENGTH_BASE,
    directional_force_magnitude: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    anchor_label: str | None = None,
    seed_start: int = 0,
    jitter_seed: int = 0,
    return_run_metadata: bool = False,
):
    """Run repeated PhysicsSim realizations with random starts and force perturbations.

    This is a parameter-perturbation repeated-simulation procedure. It does not
    resample the historical distance or direction observations.
    """
    n_bootstrap = int(N_BOOTSTRAP)
    if n_bootstrap < 2:
        raise ValueError("N_BOOTSTRAP must be at least 2.")
    if float(ALPHA_JITTER) < 0.0 or float(BETA_JITTER) < 0.0:
        raise ValueError("Bootstrap jitter values must be non-negative.")
    center_values = np.asarray([alpha, beta, w_dis], dtype=float)
    if not np.all(np.isfinite(center_values)) or float(w_dis) <= 0.0:
        raise ValueError("HPO alpha, beta, and w_dis must be finite; w_dis must be positive.")
    if len(fixed_point_labels) != len(fixed_points_lonlat):
        raise ValueError("fixed_point_labels and fixed_points_lonlat must have equal length.")

    rng = np.random.default_rng(int(jitter_seed))
    samples = None
    vertice = None
    dni = None
    run_metadata: list[dict] = []

    for bootstrap_index in trange(n_bootstrap, desc="Bootstrap"):
        if bootstrap_index == 0:
            sampled_alpha = float(alpha)
            sampled_beta = float(beta)
        else:
            sampled_alpha = float(alpha) + float(rng.normal(0.0, float(ALPHA_JITTER)))
            sampled_beta = float(beta) + float(rng.normal(0.0, float(BETA_JITTER)))

        alpha_noise = sampled_alpha - float(alpha)
        beta_noise = sampled_beta - float(beta)
        sampled_w_dir = float(w_dis) * float(10.0 ** sampled_alpha)
        sampled_w_reg = float(w_dis) * float(10.0 ** sampled_beta)
        sampled_directional_force = float(directional_force_magnitude) * float(10.0 ** alpha_noise)
        sampled_repulsion_strength = float(repulsion_strength) * float(10.0 ** beta_noise)

        simulation_seed = int(seed_start) + bootstrap_index
        pos, run_vertice, run_dni = _run_once(
            simulation_seed,
            fixed_point_labels,
            fixed_points_lonlat,
            spring_stiffness=float(spring_stiffness),
            repulsion_strength=sampled_repulsion_strength,
            directional_force_magnitude=sampled_directional_force,
            anchor_label=anchor_label,
        )
        if samples is None:
            vertice = list(run_vertice)
            dni = dict(run_dni)
            samples = np.zeros((n_bootstrap, len(vertice), 2), dtype=float)
        elif list(run_vertice) != vertice or dict(run_dni) != dni:
            raise ValueError("Node ordering changed between bootstrap realizations.")
        samples[bootstrap_index] = pos
        run_metadata.append(
            {
                "bootstrap_index": bootstrap_index,
                "simulation_seed": simulation_seed,
                "alpha": sampled_alpha,
                "beta": sampled_beta,
                "alpha_noise": alpha_noise,
                "beta_noise": beta_noise,
                "w_dis": float(w_dis),
                "w_dir": sampled_w_dir,
                "w_reg": sampled_w_reg,
                "spring_stiffness": float(spring_stiffness),
                "directional_force": sampled_directional_force,
                "repulsion_strength": sampled_repulsion_strength,
                "status": "ok",
            }
        )

    assert samples is not None and vertice is not None and dni is not None
    if return_run_metadata:
        return samples, vertice, dni, run_metadata
    return samples, vertice, dni


def confidence_ellipse_summary(
    samples: np.ndarray,
    vertice: Sequence[str],
    confidence_levels: Sequence[float] = (0.95, 0.90, 0.85),
) -> list[dict]:
    """Return reproducible numeric ellipse parameters for every node and level."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("samples must have shape (B, N_nodes, 2).")
    if values.shape[0] < 2:
        raise ValueError("At least two realizations are required for covariance ellipses.")
    if values.shape[1] != len(vertice):
        raise ValueError("vertice length does not match samples.")

    rows: list[dict] = []
    for index, name in enumerate(vertice):
        node_samples = values[:, index, :]
        mean = node_samples.mean(axis=0)
        cov = np.atleast_2d(np.cov(node_samples, rowvar=False, ddof=1))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]
        angle = float(np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])))
        for level in confidence_levels:
            if not 0.0 < float(level) < 1.0:
                raise ValueError(f"Invalid confidence level: {level}")
            width, height = 2.0 * np.sqrt(eigenvalues * chi2.ppf(float(level), df=2))
            rows.append(
                {
                    "label": name,
                    "confidence_level": float(level),
                    "mean_x_y_up_sim": float(mean[0]),
                    "mean_y_y_up_sim": float(mean[1]),
                    "cov_xx": float(cov[0, 0]),
                    "cov_xy": float(cov[0, 1]),
                    "cov_yy": float(cov[1, 1]),
                    "ellipse_width_sim": float(width),
                    "ellipse_height_sim": float(height),
                    "ellipse_angle_deg": angle,
                }
            )
    return rows


def positional_stability_summary(samples: np.ndarray, vertice: Sequence[str]) -> list[dict]:
    """Summarize run-to-run positional spread without assuming a confidence interval for the mean."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[1] != len(vertice):
        raise ValueError("samples must have shape (B, len(vertice), 2).")
    if values.shape[0] < 2:
        raise ValueError("At least two realizations are required for stability summaries.")

    rows: list[dict] = []
    for index, name in enumerate(vertice):
        node_samples = values[:, index, :]
        mean = node_samples.mean(axis=0)
        cov = np.atleast_2d(np.cov(node_samples, rowvar=False, ddof=1))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]
        major_sd = float(np.sqrt(eigenvalues[0]))
        minor_sd = float(np.sqrt(eigenvalues[1]))
        rows.append(
            {
                "label": name,
                "mean_x_y_up_sim": float(mean[0]),
                "mean_y_y_up_sim": float(mean[1]),
                "radial_sd_sim": float(np.sqrt(np.trace(cov))),
                "radial_sd_km": float(np.sqrt(np.trace(cov)) / km2pix),
                "major_axis_sd_sim": major_sd,
                "minor_axis_sd_sim": minor_sd,
                "major_axis_angle_deg": float(np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))),
                "ellipse_area_95_sim2": float(np.pi * chi2.ppf(0.95, df=2) * major_sd * minor_sd),
            }
        )
    return rows


def empirical_positional_stability_summary(samples: np.ndarray, vertice: Sequence[str]) -> list[dict]:
    """Summarize positional spread by empirical radial quantiles around the spatial median."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[1] != len(vertice):
        raise ValueError("samples must have shape (B, len(vertice), 2).")
    if values.shape[0] < 2:
        raise ValueError("At least two realizations are required for stability summaries.")

    rows: list[dict] = []
    for index, name in enumerate(vertice):
        node_samples = values[:, index, :]
        center = np.median(node_samples, axis=0)
        radial = np.linalg.norm(node_samples - center, axis=1)
        rows.append(
            {
                "label": name,
                "median_x_y_up_sim": float(center[0]),
                "median_y_y_up_sim": float(center[1]),
                "radial_q50_sim": float(np.quantile(radial, 0.50)),
                "radial_q95_sim": float(np.quantile(radial, 0.95)),
                "radial_max_sim": float(np.max(radial)),
                "radial_q50_km": float(np.quantile(radial, 0.50) / km2pix),
                "radial_q95_km": float(np.quantile(radial, 0.95) / km2pix),
                "radial_max_km": float(np.max(radial) / km2pix),
            }
        )
    return rows


def _add_relative_layout_scale(rows: list[dict]) -> float:
    centers = np.asarray([[row["median_x_y_up_sim"], row["median_y_y_up_sim"]] for row in rows], dtype=float)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    distances = pairwise[np.triu_indices(len(centers), k=1)]
    positive = distances[np.isfinite(distances) & (distances > 1e-9)]
    if positive.size == 0:
        raise ValueError("Median reconstructed configuration has no positive pairwise scale.")
    layout_scale_sim = float(np.mean(positive))
    for row in rows:
        row["layout_scale_sim"] = layout_scale_sim
        row["radial_q50_layout_pct"] = float(100.0 * row["radial_q50_sim"] / layout_scale_sim)
        row["radial_q95_layout_pct"] = float(100.0 * row["radial_q95_sim"] / layout_scale_sim)
        row["radial_max_layout_pct"] = float(100.0 * row["radial_max_sim"] / layout_scale_sim)
    return layout_scale_sim


def plot_appendix_stability_overview(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path],
    *,
    anchor_labels: Sequence[str] = (),
    test_labels: Sequence[str] = (),
) -> list[dict]:
    """Plot an appendix-ready node ranking based on empirical positional spread."""
    rows = empirical_positional_stability_summary(samples, vertice)
    _add_relative_layout_scale(rows)
    anchors = set(anchor_labels)
    tests = set(test_labels)
    active = sorted((row for row in rows if row["label"] not in anchors), key=lambda row: row["radial_q95_layout_pct"], reverse=True)
    for rank, row in enumerate(active, start=1):
        row["variability_rank_desc"] = rank

    fig, ax_rank = plt.subplots(figsize=(10.5, 10.5))
    y = np.arange(len(active))
    q50 = np.asarray([row["radial_q50_layout_pct"] for row in active])
    q95 = np.asarray([row["radial_q95_layout_pct"] for row in active])
    row_colors = np.asarray(["#C46A24" if row["label"] in tests else "#3E6F91" for row in active])
    ax_rank.hlines(y, q50, q95, color=row_colors, linewidth=2.0, alpha=0.48)
    ax_rank.scatter(q50, y, marker="o", s=34, facecolor="white", edgecolor=row_colors, linewidth=1.2,
                    label="Median radial displacement", zorder=3)
    test_mask = np.asarray([row["label"] in tests for row in active])
    if np.any(~test_mask):
        ax_rank.scatter(q95[~test_mask], y[~test_mask], marker="D", s=42, color="#3E6F91",
                        label="95th percentile: other node", zorder=3)
    if np.any(test_mask):
        ax_rank.scatter(q95[test_mask], y[test_mask], marker="s", s=42, color="#C46A24",
                        label="95th percentile: held-out test site", zorder=3)
    ax_rank.set_yticks(y, [f"{row['variability_rank_desc']:>2}. {row['label']}" for row in active], fontsize=9.2)
    for tick, row in zip(ax_rank.get_yticklabels(), active):
        if row["label"] in tests:
            tick.set_color("#A54D14")
            tick.set_fontweight("bold")
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Radial displacement relative to the reconstructed layout scale (%)")
    ax_rank.set_title("Empirical layout variability by reconstructed node", fontsize=15, pad=28)
    ax_rank.grid(axis="x", alpha=0.2, linewidth=0.6)
    ax_rank.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax_rank.text(
        0.0, 1.012,
        "Layout scale = mean pairwise separation in the median reconstructed configuration. Anchors are fixed and omitted.",
        transform=ax_rank.transAxes, fontsize=9.2, color="#444444",
    )
    fig.text(
        0.5, 0.018,
        "This diagnostic evaluates coarse configuration stability; it does not estimate precise historical locations or confidence intervals for true locations.",
        ha="center", fontsize=9, color="#444444",
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.075, top=0.90)
    for path in _normalise_output_paths(output_paths, str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "appendix_stability_overview.png")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_appendix_hdr_panels(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path],
    *,
    anchor_labels: Sequence[str] = (),
    grid_size: int = 120,
    selected_labels: Sequence[str] | None = None,
    ncols: int = 3,
    figure_title: str = "Selected node distributions across the coarse-layout variability range",
) -> list[dict]:
    """Plot automatic or explicitly selected empirical node distributions."""
    values = np.asarray(samples, dtype=float)
    anchors = set(anchor_labels)
    summary = empirical_positional_stability_summary(values, vertice)
    _add_relative_layout_scale(summary)
    active = sorted((row for row in summary if row["label"] not in anchors), key=lambda row: row["radial_q95_layout_pct"], reverse=True)
    rank_by_label = {row["label"]: rank for rank, row in enumerate(active, start=1)}
    row_by_label = {row["label"]: row for row in active}
    if selected_labels is None:
        indices = np.unique(np.linspace(0, len(active) - 1, min(6, len(active)), dtype=int))
        selected = [active[index] for index in indices]
    else:
        requested = list(selected_labels)
        if len(requested) != len(set(requested)):
            raise ValueError("selected_labels must not contain duplicates.")
        unavailable = [label for label in requested if label not in row_by_label]
        if unavailable:
            raise ValueError(f"Selected HDR labels are unavailable or fixed anchors: {unavailable}")
        selected = [row_by_label[label] for label in requested]
    if not selected:
        raise ValueError("At least one non-anchor node must be selected for HDR panels.")
    if int(ncols) < 1:
        raise ValueError("ncols must be at least 1.")

    panel_cols = min(int(ncols), len(selected))
    panel_rows = int(ceil(len(selected) / panel_cols))
    fig_width = 5.6 if panel_cols == 1 else 4.15 * panel_cols
    fig_height = 5.4 if panel_rows == 1 else 3.65 * panel_rows + 0.6
    fig, axes = plt.subplots(panel_rows, panel_cols, figsize=(fig_width, fig_height), squeeze=False)
    output_rows: list[dict] = []
    for ax, row in zip(axes.ravel(), selected):
        index = list(vertice).index(row["label"])
        node_samples = values[:, index, :]
        kde = gaussian_kde(node_samples.T)
        lower = node_samples.min(axis=0)
        upper = node_samples.max(axis=0)
        padding = np.maximum((upper - lower) * 0.15, 2.0)
        xx, yy = np.mgrid[
            lower[0] - padding[0] : upper[0] + padding[0] : complex(grid_size),
            lower[1] - padding[1] : upper[1] + padding[1] : complex(grid_size),
        ]
        zz = np.asarray(kde(np.vstack([xx.ravel(), yy.ravel()])), dtype=float).reshape(xx.shape)
        threshold_95 = _highest_density_threshold(zz, 0.95)
        threshold_50 = _highest_density_threshold(zz, 0.50)
        ax.scatter(node_samples[:, 0], node_samples[:, 1], s=5, color="#5B8DB8", alpha=0.15, rasterized=True)
        if threshold_95 < threshold_50:
            ax.contour(xx, yy, zz, levels=[threshold_95, threshold_50], colors=["#4677A6", "#C23B3B"], linewidths=[1.1, 1.6])
        ax.plot(row["median_x_y_up_sim"], row["median_y_y_up_sim"], marker="x", color="black", ms=5)
        ax.set_title(
            f"Rank {rank_by_label[row['label']]}: {row['label']}\nmedian={row['radial_q50_layout_pct']:.1f}%; q95={row['radial_q95_layout_pct']:.1f}% of layout scale",
            fontsize=9.5 if len(selected) <= 6 else 8.0,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.12, linewidth=0.5)
        output_rows.append({"variability_rank_desc": rank_by_label[row["label"]], **row,
                            "hdr_50_threshold": threshold_50, "hdr_95_threshold": threshold_95})
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.suptitle(figure_title, fontsize=14 if len(selected) <= 6 else 17)
    for row_index, row_axes in enumerate(axes):
        for col_index, ax in enumerate(row_axes):
            if row_index == panel_rows - 1:
                ax.set_xlabel("x (simulation units)", fontsize=8.5)
            if col_index == 0:
                ax.set_ylabel("y (simulation units)", fontsize=8.5)
    fig.text(0.5, 0.018, "Red: 50% HDR; blue: 95% HDR; x: coordinate-wise median. Panels are independently scaled.",
             ha="center", fontsize=9)
    top = 0.88 if len(selected) <= 6 else 0.955
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.06, top=top, hspace=0.48, wspace=0.30)
    for path in _normalise_output_paths(output_paths, str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "appendix_stability_hdr.png")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_rows


def plot_relative_stability_map(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[dict]:
    """Plot node-wise relative positional spread without overlapping uncertainty regions."""
    rows = positional_stability_summary(samples, vertice)
    spread = np.asarray([row["radial_sd_km"] for row in rows], dtype=float)
    positive = spread[spread > 1e-9]
    vmax = float(np.percentile(positive, 95)) if positive.size else 1.0
    vmax = max(vmax, 1e-9)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    means = np.asarray([[row["mean_x_y_up_sim"], row["mean_y_y_up_sim"]] for row in rows])
    if xlim is None:
        margin_x = max(float(np.ptp(means[:, 0])) * 0.12, 25.0)
        xlim = (float(means[:, 0].min() - margin_x), float(means[:, 0].max() + margin_x))
    if ylim is None:
        margin_y = max(float(np.ptp(means[:, 1])) * 0.18, 25.0)
        ylim = (float(means[:, 1].min() - margin_y), float(means[:, 1].max() + margin_y))
    sizes = 24.0 + 150.0 * np.sqrt(np.clip(spread / vmax, 0.0, 1.0))
    scatter = ax.scatter(
        means[:, 0], means[:, 1], c=spread, s=sizes, cmap="viridis", vmin=0.0, vmax=vmax,
        edgecolors="white", linewidths=0.5, zorder=3,
    )
    center_x = float(np.median(means[:, 0]))
    label_items = []
    for row in rows:
        point = np.asarray([row["mean_x_y_up_sim"], row["mean_y_y_up_sim"]])
        side = -1.0 if point[0] <= center_x else 1.0
        label_items.append({"row": row, "point": point, "side": side, "label_y": float(point[1])})
    for side in (-1.0, 1.0):
        side_items = sorted((item for item in label_items if item["side"] == side), key=lambda item: item["label_y"])
        for index in range(1, len(side_items)):
            side_items[index]["label_y"] = max(side_items[index]["label_y"], side_items[index - 1]["label_y"] + 14.0)
        if side_items and side_items[-1]["label_y"] > ylim[1] - 8.0:
            shift = side_items[-1]["label_y"] - (ylim[1] - 8.0)
            for item in side_items:
                item["label_y"] -= shift
        for index in range(len(side_items) - 2, -1, -1):
            side_items[index]["label_y"] = min(side_items[index]["label_y"], side_items[index + 1]["label_y"] - 14.0)
        if side_items and side_items[0]["label_y"] < ylim[0] + 8.0:
            shift = (ylim[0] + 8.0) - side_items[0]["label_y"]
            for item in side_items:
                item["label_y"] += shift
        for item in side_items:
            row = item["row"]
            point = item["point"]
            label_x = float(point[0] + side * 12.0)
            ax.annotate(
                row["label"], xy=point, xytext=(label_x, item["label_y"]), textcoords="data", fontsize=7,
                ha="left" if side > 0 else "right", va="center", color="#222222", zorder=4,
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.35, "alpha": 0.7},
            )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03, shrink=0.78)
    colorbar.set_label("Radial positional SD (km)")
    ax.set(xlim=xlim, ylim=ylim, xlabel="x (simulation units, east-positive)",
           ylabel="y (simulation units, north-positive)", title="PhysicsSim relative positional stability")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linewidth=0.5)
    plt.tight_layout()
    for path in _normalise_output_paths(output_paths, str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "relative_stability_map.png")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_ellipse_outline_map(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path],
    *,
    confidence_level: float = 0.95,
    xlim: tuple[float, float] = (0.0, 1200.0),
    ylim: tuple[float, float] = (0.0, 750.0),
) -> None:
    """Plot one unfilled covariance ellipse per node for a less occluded overview."""
    rows = confidence_ellipse_summary(samples, vertice, (confidence_level,))
    fig, ax = plt.subplots(figsize=(12, 7.5))
    cmap = plt.get_cmap("tab20")
    for index, row in enumerate(rows):
        color = cmap(index % 20)
        if row["ellipse_width_sim"] > 0.0 and row["ellipse_height_sim"] > 0.0:
            ax.add_patch(Ellipse(
                (row["mean_x_y_up_sim"], row["mean_y_y_up_sim"]),
                row["ellipse_width_sim"], row["ellipse_height_sim"], angle=row["ellipse_angle_deg"],
                facecolor="none", edgecolor=color, linewidth=0.9, alpha=0.65,
            ))
        ax.plot(row["mean_x_y_up_sim"], row["mean_y_y_up_sim"], "o", color=color, ms=2.8)
    ax.set(xlim=xlim, ylim=ylim, xlabel="x (simulation units, east-positive)",
           ylabel="y (simulation units, north-positive)",
           title=f"PhysicsSim positional dispersion ellipses ({confidence_level:.0%})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linewidth=0.5)
    plt.tight_layout()
    for path in _normalise_output_paths(output_paths, str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "dispersion_ellipses_95_outline.png")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _highest_density_threshold(density: np.ndarray, mass: float) -> float:
    flat = np.asarray(density, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat >= 0.0)]
    if flat.size == 0 or float(flat.sum()) <= 0.0:
        raise ValueError("KDE grid has no finite positive mass.")
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return float(ordered[min(int(np.searchsorted(cumulative, mass)), ordered.size - 1)])


def plot_hdr_small_multiples(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path],
    *,
    n_panels: int = 6,
    grid_size: int = 100,
) -> list[dict]:
    """Show representative low-to-high spread nodes with 50% and 95% KDE HDRs."""
    values = np.asarray(samples, dtype=float)
    summary = positional_stability_summary(values, vertice)
    active = sorted((row for row in summary if row["radial_sd_sim"] > 1e-6), key=lambda row: row["radial_sd_sim"])
    count = min(int(n_panels), len(active))
    if count < 1:
        raise ValueError("No non-degenerate node distributions are available for HDR panels.")
    selected = [active[index] for index in np.linspace(0, len(active) - 1, count, dtype=int)]

    fig, axes = plt.subplots(2, int(np.ceil(count / 2)), figsize=(12, 7), squeeze=False)
    selected_rows: list[dict] = []
    for panel, (ax, row) in enumerate(zip(axes.ravel(), selected)):
        index = list(vertice).index(row["label"])
        node_samples = values[:, index, :]
        kde = gaussian_kde(node_samples.T)
        lower = node_samples.min(axis=0)
        upper = node_samples.max(axis=0)
        padding = np.maximum((upper - lower) * 0.15, 2.0)
        xx, yy = np.mgrid[
            lower[0] - padding[0] : upper[0] + padding[0] : complex(grid_size),
            lower[1] - padding[1] : upper[1] + padding[1] : complex(grid_size),
        ]
        zz = np.asarray(kde(np.vstack([xx.ravel(), yy.ravel()])), dtype=float).reshape(xx.shape)
        threshold_95 = _highest_density_threshold(zz, 0.95)
        threshold_50 = _highest_density_threshold(zz, 0.50)
        ax.scatter(node_samples[:, 0], node_samples[:, 1], s=5, color="#3B6EA8", alpha=0.22)
        if threshold_95 < threshold_50:
            ax.contour(xx, yy, zz, levels=[threshold_95, threshold_50], colors=["#5B8DB8", "#C23B3B"], linewidths=[1.0, 1.5])
        ax.plot(row["mean_x_y_up_sim"], row["mean_y_y_up_sim"], marker="x", color="black", ms=5)
        ax.set_title(f"{row['label']} | radial SD={row['radial_sd_km']:.1f} km", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12, linewidth=0.5)
        selected_rows.append({"panel": panel + 1, **row, "hdr_50_threshold": threshold_50, "hdr_95_threshold": threshold_95})
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    fig.suptitle("Representative positional distributions: 50% and 95% highest-density regions", fontsize=14)
    fig.supxlabel("x (simulation units, east-positive)")
    fig.supylabel("y (simulation units, north-positive)")
    plt.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    for path in _normalise_output_paths(output_paths, str(DEFAULT_BOOTSTRAP_OUTPUT_DIR / "stability_hdr_small_multiples.png")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selected_rows


def _normalise_output_paths(output_paths: str | Path | Iterable[str | Path] | None, default_path: str) -> list[Path]:
    if output_paths is None:
        paths = [Path(default_path)]
    elif isinstance(output_paths, (str, Path)):
        paths = [Path(output_paths)]
    else:
        paths = [Path(path) for path in output_paths]
    if not paths:
        raise ValueError("At least one output path is required.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def plot_multi_ellipses(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path] | None = None,
    *,
    xlim: tuple[float, float] = (0.0, 1200.0),
    ylim: tuple[float, float] = (0.0, 750.0),
) -> list[dict]:
    summary = confidence_ellipse_summary(samples, vertice)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(vertice)}

    for index, name in enumerate(vertice):
        node_rows = [row for row in summary if row["label"] == name]
        for row, alpha in zip(node_rows, (0.12, 0.20, 0.30)):
            if row["ellipse_width_sim"] <= 0.0 or row["ellipse_height_sim"] <= 0.0:
                continue
            ax.add_patch(
                Ellipse(
                    xy=(row["mean_x_y_up_sim"], row["mean_y_y_up_sim"]),
                    width=row["ellipse_width_sim"],
                    height=row["ellipse_height_sim"],
                    angle=row["ellipse_angle_deg"],
                    facecolor=colors[name],
                    edgecolor=colors[name],
                    lw=0.8,
                    alpha=alpha,
                    zorder=2,
                )
            )
        mean = np.asarray(samples, dtype=float)[:, index, :].mean(axis=0)
        ax.plot(mean[0], mean[1], marker="o", color=colors[name], zorder=3, ms=3)
        ax.text(mean[0] + 3.0, mean[1] + 3.0, name, color=colors[name], fontsize=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (simulation units, east-positive)")
    ax.set_ylabel("y (simulation units, north-positive)")
    ax.set_title("PhysicsSim positional uncertainty ellipses (95%, 90%, 85%)")
    plt.tight_layout()
    paths = _normalise_output_paths(output_paths, ELLIPSES_FILE)
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_kde_combined(
    samples: np.ndarray,
    vertice: Sequence[str],
    output_paths: str | Path | Iterable[str | Path] | None = None,
    *,
    xlim: tuple[float, float] = (0.0, 1200.0),
    ylim: tuple[float, float] = (0.0, 750.0),
    grid_size: int = 200,
    degenerate_tolerance: float = 1e-6,
) -> list[dict]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("samples must have shape (B, N_nodes, 2).")
    if values.shape[1] != len(vertice):
        raise ValueError("vertice length does not match samples.")
    if int(grid_size) < 20:
        raise ValueError("grid_size must be at least 20.")
    if float(degenerate_tolerance) < 0.0:
        raise ValueError("degenerate_tolerance must be non-negative.")

    xx, yy = np.mgrid[
        xlim[0] : xlim[1] : complex(int(grid_size)),
        ylim[0] : ylim[1] : complex(int(grid_size)),
    ]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    fitted: list[tuple[gaussian_kde | None, np.ndarray | None, str]] = []
    all_densities: list[np.ndarray] = []
    status_rows: list[dict] = []

    for index, name in enumerate(vertice):
        node_samples = values[:, index, :]
        coordinate_span = float(np.max(np.ptp(node_samples, axis=0)))
        if (
            node_samples.shape[0] < 3
            or np.unique(node_samples, axis=0).shape[0] < 3
            or coordinate_span <= float(degenerate_tolerance)
        ):
            fitted.append((None, None, "degenerate"))
            status_rows.append(
                {
                    "label": name,
                    "kde_status": "degenerate",
                    "n_samples": len(node_samples),
                    "coordinate_span_sim": coordinate_span,
                }
            )
            continue
        try:
            kde = gaussian_kde(node_samples.T)
            densities = np.asarray(kde(node_samples.T), dtype=float)
            positive = densities[np.isfinite(densities) & (densities > 0.0)]
            if positive.size == 0:
                raise ValueError("KDE produced no positive finite densities.")
            fitted.append((kde, densities, "ok"))
            all_densities.append(positive)
            status_rows.append(
                {
                    "label": name,
                    "kde_status": "ok",
                    "n_samples": len(node_samples),
                    "coordinate_span_sim": coordinate_span,
                }
            )
        except Exception as exc:
            fitted.append((None, None, "singular"))
            status_rows.append(
                {
                    "label": name,
                    "kde_status": "singular",
                    "n_samples": len(node_samples),
                    "coordinate_span_sim": coordinate_span,
                    "error": repr(exc),
                }
            )

    norm = None
    if all_densities:
        combined = np.hstack(all_densities)
        vmin = float(combined.min())
        vmax = float(combined.max())
        if np.isclose(vmin, vmax):
            vmax = vmin * (1.0 + 1e-9)
        norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    label_cmap = plt.get_cmap("tab20")
    density_cmap = plt.get_cmap("viridis")
    last_scatter = None

    for index, name in enumerate(vertice):
        node_samples = values[:, index, :]
        mean_xy = node_samples.mean(axis=0)
        label_color = label_cmap(index % 20)
        kde, densities, status = fitted[index]
        if kde is not None and densities is not None and norm is not None:
            last_scatter = ax.scatter(
                node_samples[:, 0],
                node_samples[:, 1],
                c=densities,
                cmap=density_cmap,
                norm=norm,
                s=4,
                alpha=0.75,
                zorder=2,
            )
            zz = np.asarray(kde(grid_coords), dtype=float).reshape(xx.shape)
            if np.nanmax(zz) > np.nanmin(zz):
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=5,
                    colors=[label_color],
                    linewidths=0.6,
                    alpha=0.65,
                    zorder=1,
                )
        else:
            ax.scatter(node_samples[:, 0], node_samples[:, 1], s=5, color=[label_color], alpha=0.7, zorder=2)
        ax.plot(mean_xy[0], mean_xy[1], marker="x", color=label_color, ms=5, mew=1.0, zorder=3)
        ax.text(mean_xy[0] + 3.0, mean_xy[1] + 3.0, name, color=label_color, fontsize=7, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (simulation units, east-positive)")
    ax.set_ylabel("y (simulation units, north-positive)")
    ax.set_title("PhysicsSim combined positional KDE map")
    if last_scatter is not None:
        colorbar = fig.colorbar(last_scatter, ax=ax, fraction=0.035, pad=0.03, shrink=0.75)
        colorbar.set_label("KDE density (shared scale)")
    plt.tight_layout()
    paths = _normalise_output_paths(output_paths, KDE_COMBINED_FILE)
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return status_rows
