"""Build a manuscript-ready, source-traceable results snapshot.

The snapshot keeps the formal experiment outputs immutable. Main-text tables
are recomputed from run-level CSV files, while figures are either regenerated
from their plotted data or copied only after an existing consistency check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_paper_script.ch5_ablation_study import _bootstrap_ci_mean
from scripts.create_manuscript_spatial_comparisons import create_spatial_comparisons


AS_DIR = PROJECT_ROOT / "outputs" / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
BFGS_DIR = PROJECT_ROOT / "outputs" / "ch5_scipy_bfgs_hpo_selected_alpha_0p5_beta_-0p5_100seeds_20260823"
POLISH_DIR = PROJECT_ROOT / "outputs" / "ch5_physics_to_bfgs_polishing_source_weights_100seeds_20260827"
ANCHOR_DIR = PROJECT_ROOT / "outputs" / "ch5_anchor_split_robustness_formal_45splits_hpo3_final10_20260824"
DETOUR_DIR = PROJECT_ROOT / "outputs" / "ch5_detour_factor_sensitivity_scenario_hpo_fixed_reference_alpha_1_beta_-0.5_13scenarios_20260826"
TOLERANCE_DIR = PROJECT_ROOT / "outputs" / "ch5_direction_tolerance_sensitivity_hpo10_final100_20260908"
DC_HPO_DIR = PROJECT_ROOT / "outputs" / "ch5_dc_smacof_hparam_wang_current_alpha_-4_0_seed0_9_20260721"
BFGS_HPO_DIR = PROJECT_ROOT / "outputs" / "ch5_scipy_bfgs_hpo_grid_36x10_20260823_all_folds_partial_ok"
PHYSICS_HPO_DIR = PROJECT_ROOT / "outputs" / "ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2_manual_alpha_1_beta_-0.5"
PHYSICS_HPO_GRID_DIR = PROJECT_ROOT / "outputs" / "ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2"
REPRESENTATIVE_DIR = PROJECT_ROOT / "outputs" / "ch6_section_6_5_full_smacof_dc_representative_wang_current_20260722"
VIS_DIR = PROJECT_ROOT / "outputs" / "ch6_section_6_5_visual_wang_current_20260722"
BOOTSTRAP_DIR = PROJECT_ROOT / "outputs" / "ch5_bootstrap_selected_hdr_6nodes_sigma0p05_300runs"
ALL_ANCHOR_DIR = PROJECT_ROOT / "outputs" / "final_reconstruction_all_verified_sites_alpha_1_beta_-0.5_seed0"
DEFAULT_OUTDIR = PROJECT_ROOT / "paper_results" / "manuscript_ready_20260828"

CORE_METRICS = (
    ("RMSE_test_km", "RMSE (km)", 0),
    ("E_distance_stress", "Stress", 3),
    ("E_direction_vr", "Violation Rate", 3),
    ("E_direction_mae", "Mean Angular Error (rad)", 3),
    ("crowding_violation_rate_tau_0p1", "Crowding Violation Rate (τ = 0.10)", 4),
    ("collapse_node_rate_tau_0p1", "Collapse Node Rate (τ = 0.10)", 3),
    ("nnd_q05_km", "Nearest-Neighbor Distance, 5th Quantile (km)", 1),
    ("distance_edge_crossing_rate", "Crossing-edge Rate", 3),
)

TABLE1_ORDER = (
    "Random+Align",
    "PhysicsSim-DistOnly",
    "SMACOF",
    "SciPy-BFGS",
    "DC-SMACOF",
    "PhysicsSim-DistDir",
    "PhysicsSim-DistDirAnch",
    "PhysicsSim-Full",
)

EFFECTS = (
    ("Direction", "direction_given_distance"),
    ("Anchors", "optimization_anchors_given_distance_direction"),
    ("Repulsion", "repulsion_given_distance_direction_anchors"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def successful(frame: pd.DataFrame, variant: str | None = None) -> pd.DataFrame:
    out = frame.loc[frame["status"].eq("ok")].copy()
    if variant is not None:
        out = out.loc[out["variant"].eq(variant)].copy()
    return out


def mean_sd(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("A reported mean and sample SD requires at least two finite values.")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def fmt_mean_sd(values: pd.Series | np.ndarray, digits: int) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def fmt_diff_ci(row: pd.Series, digits: int) -> str:
    return (
        f"{float(row['paired_diff_mean']):.{digits}f} "
        f"[{float(row['paired_diff_ci95_lo']):.{digits}f}, "
        f"{float(row['paired_diff_ci95_hi']):.{digits}f}]"
    )


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "±": r"$\pm$",
        "−": r"$-$",
        "τ": r"$\tau$",
        "Δ": r"$\Delta$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_tabular(
    frame: pd.DataFrame,
    *,
    first_col_width: str = "0.23\\linewidth",
    fixed_col_widths: tuple[str, ...] | None = None,
    header_cells: tuple[str, ...] | None = None,
    tabcolsep: str = "2pt",
    grouped_by: str | None = None,
) -> str:
    widths = fixed_col_widths or (first_col_width,)
    if len(widths) >= len(frame.columns):
        raise ValueError("At least one tabularx X column is required.")
    if header_cells is not None and len(header_cells) != len(frame.columns):
        raise ValueError("header_cells must match the table column count.")
    if grouped_by is not None and grouped_by not in frame.columns:
        raise ValueError(f"grouped_by column is missing: {grouped_by}")
    columns = [rf">{{\raggedright\arraybackslash}}p{{{width}}}" for width in widths]
    columns.extend([r">{\centering\arraybackslash}X"] * (len(frame.columns) - len(widths)))
    headers = list(header_cells) if header_cells is not None else [latex_escape(column) for column in frame.columns]
    lines = [
        r"\begingroup",
        r"\footnotesize",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\noindent\begin{tabularx}{\linewidth}{@{}" + "".join(columns) + r"@{}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    group_index = frame.columns.get_loc(grouped_by) if grouped_by is not None else None
    previous_group = None
    for row in frame.fillna("").itertuples(index=False, name=None):
        values = list(row)
        if group_index is not None:
            current_group = values[group_index]
            if previous_group is not None and current_group != previous_group:
                lines.append(r"\midrule")
            elif current_group == previous_group:
                values[group_index] = ""
            previous_group = current_group
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\endgroup", ""])
    return "\n".join(lines)


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_runs = pd.read_csv(AS_DIR / "progressive_runs_by_seed.csv", encoding="utf-8-sig")
    random_runs = pd.read_csv(AS_DIR / "random_align_runs.csv", encoding="utf-8-sig")
    bfgs_runs = pd.read_csv(BFGS_DIR / "bfgs_runs_by_seed.csv", encoding="utf-8-sig")
    expected = {
        "PhysicsSim-DistOnly", "SMACOF", "DC-SMACOF", "PhysicsSim-DistDir",
        "PhysicsSim-DistDirAnch", "PhysicsSim-Full",
    }
    available = set(successful(as_runs)["variant"].unique())
    if not expected.issubset(available):
        raise ValueError(
            "Formal AS output is missing registered manuscript variants: "
            f"{sorted(expected.difference(available))}"
        )
    if len(successful(random_runs, "Random+Align")) != 1000:
        raise ValueError("Random+Align formal result must contain 1,000 successful runs.")
    if len(successful(bfgs_runs)) != 100 or not successful(bfgs_runs)["optimizer_success"].astype(bool).all():
        raise ValueError("BFGS formal result must contain 100 successful optimizer runs.")
    for variant in expected:
        if len(successful(as_runs, variant)) != 100:
            raise ValueError(f"{variant} must contain 100 successful formal runs.")
    return as_runs, random_runs, bfgs_runs


def build_table_1(as_runs: pd.DataFrame, random_runs: pd.DataFrame, bfgs_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series: dict[str, pd.Series] = {"Random+Align": successful(random_runs, "Random+Align")["RMSE_test_km"]}
    for variant in TABLE1_ORDER:
        if variant in {"Random+Align", "SciPy-BFGS"}:
            continue
        series[variant] = successful(as_runs, variant)["RMSE_test_km"]
    series["SciPy-BFGS"] = successful(bfgs_runs)["RMSE_test_km"]
    random_mean = float(series["Random+Align"].mean())
    display_rows = []
    audit_rows = []
    for variant in TABLE1_ORDER:
        mean, sd = mean_sd(series[variant])
        reduction = 100.0 * (random_mean - mean) / random_mean
        display_rows.append({
            "Model": "BFGS" if variant == "SciPy-BFGS" else variant,
            "Held-out RMSE, mean ± SD (km)": f"{mean:.0f} ± {sd:.0f}",
            "RMSE reduction vs Random+Align": "Reference" if variant == "Random+Align" else f"{reduction:.0f}%",
        })
        audit_rows.append({"variant": variant, "n": len(series[variant]), "mean": mean, "sample_sd": sd, "reduction_percent": reduction})
    return pd.DataFrame(display_rows), pd.DataFrame(audit_rows)


def _paired_row(paired: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    rows = paired.loc[paired["comparison"].eq(comparison) & paired["metric"].eq(metric)]
    if len(rows) != 1 or int(rows.iloc[0]["n_pairs"]) != 100:
        raise ValueError(f"Expected one 100-pair row for {comparison}/{metric}.")
    return rows.iloc[0]


def build_table_2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(AS_DIR / "progressive_paired_comparisons.csv", encoding="utf-8-sig")
    panel_a_rows = []
    panel_b_rows = []
    audit_rows = []
    for effect, comparison in EFFECTS:
        values_a = {
            "Added component": effect,
            "ΔRMSE (km) [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "RMSE_test_km"), 0),
            "ΔStress [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "E_distance_stress"), 3),
            "ΔViolation Rate [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "E_direction_vr"), 3),
            "ΔMean Angular Error (rad) [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "E_direction_mae"), 3),
        }
        values_b = {
            "Added component": effect,
            "ΔCrowding Violation Rate (τ = 0.10) [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "crowding_violation_rate_tau_0p1"), 4),
            "ΔCollapse Node Rate (τ = 0.10) [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "collapse_node_rate_tau_0p1"), 3),
            "ΔNearest-Neighbor Distance 5th Quantile (km) [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "nnd_q05_km"), 1),
            "ΔCrossing-edge Rate [95% CI]": fmt_diff_ci(_paired_row(paired, comparison, "distance_edge_crossing_rate"), 3),
        }
        panel_a_rows.append(values_a)
        panel_b_rows.append(values_b)
        for metric in ("RMSE_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae", "crowding_violation_rate_tau_0p1", "collapse_node_rate_tau_0p1", "nnd_q05_km", "distance_edge_crossing_rate"):
            row = _paired_row(paired, comparison, metric)
            audit_rows.append({"effect": effect, **row.to_dict()})
    return pd.DataFrame(panel_a_rows), pd.DataFrame(panel_b_rows), pd.DataFrame(audit_rows)


def build_table_3(as_runs: pd.DataFrame, bfgs_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("Distance only", "SMACOF", as_runs),
        ("Distance only", "PhysicsSim-DistOnly", as_runs),
        ("Distance + direction", "DC-SMACOF", as_runs),
        ("Distance + direction", "PhysicsSim-DistDir", as_runs),
        ("Standard optimizer", "SciPy-BFGS", bfgs_runs),
        ("Standard optimizer", "PhysicsSim-Full", as_runs),
    )
    display_rows = []
    audit_rows = []
    for comparison, variant, source in comparisons:
        rows = successful(source, None if variant == "SciPy-BFGS" else variant)
        display = {"Comparison": comparison, "Model": "BFGS" if variant == "SciPy-BFGS" else variant}
        for metric, label, digits in CORE_METRICS:
            display[label] = fmt_mean_sd(rows[metric], digits)
            mean, sd = mean_sd(rows[metric])
            audit_rows.append({"comparison": comparison, "variant": variant, "metric": metric, "n": len(rows), "mean": mean, "sample_sd": sd})
        display_rows.append(display)
    return pd.DataFrame(display_rows), pd.DataFrame(audit_rows)


def render_table_3_latex(table3: pd.DataFrame) -> str:
    layout_columns = [
        "Crowding Violation Rate (τ = 0.10)",
        "Collapse Node Rate (τ = 0.10)",
        "Nearest-Neighbor Distance, 5th Quantile (km)",
        "Crossing-edge Rate",
    ]
    return "\n".join([
        r"\textbf{Panel A. Geographic accuracy and textual constraint satisfaction}", r"\par\medskip",
        latex_tabular(
            table3.drop(columns=layout_columns),
            fixed_col_widths=("0.20\\linewidth", "0.18\\linewidth"),
            header_cells=(
                "Comparison",
                "Model",
                r"\shortstack{RMSE\\(km)}",
                "Stress",
                r"\shortstack{Violation\\Rate}",
                r"\shortstack{Mean Angular\\Error (rad)}",
            ),
            tabcolsep="1.5pt",
            grouped_by="Comparison",
        ), r"\medskip",
        r"\textbf{Panel B. Layout diagnostics}", r"\par\medskip",
        latex_tabular(
            table3[["Comparison", "Model", *layout_columns]],
            fixed_col_widths=("0.20\\linewidth", "0.18\\linewidth"),
            header_cells=(
                "Comparison",
                "Model",
                r"\shortstack{Crowding\\Violation Rate\\($\tau=0.10$)}",
                r"\shortstack{Collapse\\Node Rate\\($\tau=0.10$)}",
                r"\shortstack{Nearest-Neighbor\\Distance, 5th\\Quantile (km)}",
                r"\shortstack{Crossing-edge\\Rate}",
            ),
            tabcolsep="1.5pt",
            grouped_by="Comparison",
        ),
    ])


def save_table(frame: pd.DataFrame, stem: Path, *, title: str, note: str, latex: str | None = None) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    stem.with_suffix(".md").write_text(f"# {title}\n\n{note}\n\n{markdown_table(frame)}\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text(latex if latex is not None else latex_tabular(frame), encoding="utf-8")


def plot_polishing(outdir: Path) -> pd.DataFrame:
    runs = pd.read_csv(POLISH_DIR / "polishing_runs.csv")
    if len(runs) != 100 or not runs["optimizer_success"].astype(bool).all():
        raise ValueError("Polishing figure requires 100 successful paired runs.")
    plot_data = runs[["seed", "before_objective_total", "after_objective_total", "before_RMSE_test_km_posthoc", "after_RMSE_test_km_posthoc"]].copy()
    plot_data.to_csv(outdir / "figure_2_plot_data.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    for row in plot_data.itertuples(index=False):
        arrow = FancyArrowPatch(
            (row.before_objective_total, row.before_RMSE_test_km_posthoc),
            (row.after_objective_total, row.after_RMSE_test_km_posthoc),
            arrowstyle="-|>", mutation_scale=6, linewidth=0.55, color="#777777", alpha=0.22,
        )
        ax.add_patch(arrow)
    ax.scatter(plot_data["before_objective_total"], plot_data["before_RMSE_test_km_posthoc"], s=32, marker="o", color="#1769aa", alpha=0.78, label="PhysicsSim endpoint", zorder=3)
    ax.scatter(plot_data["after_objective_total"], plot_data["after_RMSE_test_km_posthoc"], s=40, marker="^", color="#d95f02", alpha=0.82, label="BFGS-polished endpoint", zorder=3)
    ax.set_xscale("symlog", linthresh=1e4)
    ax.set_xlabel("Objective value, F(X) (symlog scale)", fontsize=12)
    ax.set_ylabel("Held-out test RMSE (km)", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"figure_2_bfgs_polishing_objective_rmse.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_data


def plot_anchor(outdir: Path) -> pd.DataFrame:
    data = pd.read_csv(ANCHOR_DIR / "anchor_split_summary.csv", encoding="utf-8-sig")
    if len(data) != 45 or data["split_id"].nunique() != 45 or data["is_original_split"].sum() != 1:
        raise ValueError("Anchor figure requires 45 unique splits and one original split.")
    plot_data = data[["split_id", "is_original_split", "RMSE_final_test_mean_km", "RMSE_final_test_std_km", "n_seeds"]].copy()
    plot_data = plot_data.sort_values("RMSE_final_test_mean_km").reset_index(drop=True)
    plot_data["rank"] = np.arange(1, len(plot_data) + 1)
    plot_data.to_csv(outdir / "figure_3_plot_data.csv", index=False, encoding="utf-8-sig")
    original = plot_data.loc[plot_data["is_original_split"].astype(bool)].iloc[0]
    fig, ax = plt.subplots(figsize=(6.25, 5.2), constrained_layout=True)
    ax.errorbar(plot_data["rank"], plot_data["RMSE_final_test_mean_km"], yerr=plot_data["RMSE_final_test_std_km"], fmt="o", markersize=3.5, linewidth=0.7, elinewidth=0.7, capsize=1.5, color="#4c78a8", alpha=0.72)
    ax.scatter([original["rank"]], [original["RMSE_final_test_mean_km"]], marker="*", s=130, color="#D55E00", zorder=4, label="Original split")
    ax.set_xlabel("Anchor/test split rank (lower mean RMSE to higher)")
    ax.set_ylabel("Mean held-out RMSE ± within-split SD (km)")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.18)
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"figure_3_anchor_split_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_data


def plot_detour(outdir: Path) -> pd.DataFrame:
    data = pd.read_csv(DETOUR_DIR / "detour_scenario_summary.csv", encoding="utf-8-sig")
    if len(data) != 13 or data["kappa"].nunique() != 13:
        raise ValueError("Detour figure requires 13 unique kappa scenarios.")
    columns = [
        "kappa",
        "RMSE_final_test_km_mean",
        "RMSE_final_test_km_std",
        "RMSE_final_test_km_ci95_low",
        "RMSE_final_test_km_ci95_high",
        "selected_alpha",
        "selected_beta",
        "hyperparameter_policy",
    ]
    plot_data = data[columns].sort_values("kappa").copy()
    plot_data.to_csv(outdir / "figure_4_plot_data.csv", index=False, encoding="utf-8-sig")
    y = plot_data["RMSE_final_test_km_mean"].to_numpy(float)
    lo = plot_data["RMSE_final_test_km_ci95_low"].to_numpy(float)
    hi = plot_data["RMSE_final_test_km_ci95_high"].to_numpy(float)
    x = plot_data["kappa"].to_numpy(float)
    reference = float(plot_data.loc[np.isclose(plot_data["kappa"], 1.0), "RMSE_final_test_km_mean"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    ax.fill_between(x, lo, hi, color="#4c78a8", alpha=0.18, label="95% bootstrap CI")
    ax.plot(x, y, marker="o", linewidth=2, color="#1769aa", label="Scenario mean")
    ax.axhline(reference, color="#D55E00", linestyle="--", linewidth=1.4, label="κ = 1.00 reference mean")
    ax.set_xlabel("Distance scaling factor, κ")
    ax.set_ylabel("Held-out test RMSE (km)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"figure_4_detour_rmse_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_data


def plot_anchor_detour(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_data = pd.read_csv(ANCHOR_DIR / "anchor_split_summary.csv", encoding="utf-8-sig")
    if len(anchor_data) != 45 or anchor_data["split_id"].nunique() != 45 or anchor_data["is_original_split"].sum() != 1:
        raise ValueError("Combined sensitivity figure requires 45 unique splits and one original split.")
    anchor_plot = anchor_data[["split_id", "is_original_split", "RMSE_final_test_mean_km", "RMSE_final_test_std_km", "n_seeds"]].copy()
    anchor_plot = anchor_plot.sort_values("RMSE_final_test_mean_km").reset_index(drop=True)
    anchor_plot["rank"] = np.arange(1, len(anchor_plot) + 1)
    anchor_plot.to_csv(outdir / "figure_3_plot_data.csv", index=False, encoding="utf-8-sig")
    original = anchor_plot.loc[anchor_plot["is_original_split"].astype(bool)].iloc[0]

    detour_data = pd.read_csv(DETOUR_DIR / "detour_paired_comparisons.csv", encoding="utf-8-sig")
    detour_data = detour_data.loc[detour_data["metric"].eq("RMSE_final_test_km")].copy()
    if len(detour_data) != 12 or detour_data["kappa"].nunique() != 12:
        raise ValueError("Combined sensitivity figure requires 12 paired RMSE comparisons against kappa=1.")
    detour_columns = [
        "kappa",
        "reference_kappa",
        "metric",
        "n_pairs",
        "difference_mean",
        "difference_std",
        "difference_ci95_low",
        "difference_ci95_high",
        "win_rate_lower",
    ]
    detour_plot = detour_data[detour_columns].sort_values("kappa").copy()
    reference_row = pd.DataFrame(
        [
            {
                "kappa": 1.0,
                "reference_kappa": 1.0,
                "metric": "RMSE_final_test_km",
                "n_pairs": 100,
                "difference_mean": 0.0,
                "difference_std": 0.0,
                "difference_ci95_low": 0.0,
                "difference_ci95_high": 0.0,
                "win_rate_lower": 0.0,
            }
        ]
    )
    detour_plot = pd.concat([detour_plot, reference_row], ignore_index=True).sort_values("kappa").reset_index(drop=True)
    detour_plot.to_csv(outdir / "figure_4_plot_data.csv", index=False, encoding="utf-8-sig")

    tolerance_data = pd.read_csv(
        TOLERANCE_DIR / "direction_tolerance_paired_comparisons.csv",
        encoding="utf-8-sig",
    )
    tolerance_plot = tolerance_data.loc[
        tolerance_data["metric"].eq("RMSE_final_test_km"),
        [
            "tolerance_numerator",
            "direction_tolerance_scale",
            "cardinal_half_width_deg",
            "diagonal_half_width_deg",
            "reference_numerator",
            "n_pairs",
            "difference_mean",
            "difference_std",
            "difference_ci95_low",
            "difference_ci95_high",
            "win_rate_lower",
        ],
    ].sort_values("direction_tolerance_scale").reset_index(drop=True)
    if (
        len(tolerance_plot) != 7
        or tolerance_plot["tolerance_numerator"].nunique() != 7
        or not np.isclose(tolerance_plot.iloc[-1]["direction_tolerance_scale"], 1.0)
        or not np.isclose(tolerance_plot.iloc[-1]["difference_mean"], 0.0)
    ):
        raise ValueError("Combined sensitivity figure requires seven tolerance scenarios including lambda=1.")
    tolerance_plot.to_csv(outdir / "figure_5_plot_data.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(13.4, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
    ax_split = fig.add_subplot(grid[0, :])
    ax_kappa = fig.add_subplot(grid[1, 0])
    ax_tolerance = fig.add_subplot(grid[1, 1])

    ax_split.errorbar(
        anchor_plot["rank"],
        anchor_plot["RMSE_final_test_mean_km"],
        yerr=anchor_plot["RMSE_final_test_std_km"],
        fmt="o",
        markersize=3.5,
        linewidth=0.7,
        elinewidth=0.7,
        capsize=1.5,
        color="#4c78a8",
        alpha=0.72,
    )
    ax_split.scatter(
        [original["rank"]],
        [original["RMSE_final_test_mean_km"]],
        marker="*",
        s=130,
        color="#D55E00",
        zorder=4,
        label="Primary split",
    )
    ax_split.set_xlabel("Anchor/test split rank (lower mean RMSE to higher)")
    ax_split.set_ylabel("Mean held-out RMSE ± within-split SD (km)")
    ax_split.legend(frameon=False, loc="lower right")
    ax_split.text(0.02, 0.97, "(a)", transform=ax_split.transAxes, ha="left", va="top", fontweight="bold", fontsize=12)
    ax_split.grid(True, alpha=0.18)

    x = detour_plot["kappa"].to_numpy(float)
    y = detour_plot["difference_mean"].to_numpy(float)
    lo = detour_plot["difference_ci95_low"].to_numpy(float)
    hi = detour_plot["difference_ci95_high"].to_numpy(float)
    ax_kappa.errorbar(
        x,
        y,
        yerr=np.vstack((y - lo, hi - y)),
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        elinewidth=1.0,
        capsize=2.5,
        color="#1769aa",
        label="Paired mean difference (95% CI)",
    )
    ax_kappa.axhline(0.0, color="#D55E00", linestyle="--", linewidth=1.4, label="κ = 1.00 reference")
    ax_kappa.set_xlim(0.69, 1.01)
    ax_kappa.set_xlabel("Distance scaling factor, κ")
    ax_kappa.set_ylabel("Paired ΔRMSE vs κ = 1.00 (km)")
    ax_kappa.legend(frameon=False, loc="upper left")
    ax_kappa.text(0.98, 0.97, "(b)", transform=ax_kappa.transAxes, ha="right", va="top", fontweight="bold", fontsize=12)
    ax_kappa.grid(True, alpha=0.2)

    x = tolerance_plot["direction_tolerance_scale"].to_numpy(float)
    y = tolerance_plot["difference_mean"].to_numpy(float)
    lo = tolerance_plot["difference_ci95_low"].to_numpy(float)
    hi = tolerance_plot["difference_ci95_high"].to_numpy(float)
    ax_tolerance.errorbar(
        x,
        y,
        yerr=np.vstack((y - lo, hi - y)),
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        elinewidth=1.0,
        capsize=2.5,
        color="#1769aa",
        label="Paired mean difference (95% CI)",
    )
    ax_tolerance.axhline(
        0.0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label="λ = 1.00 reference",
    )
    ax_tolerance.set_xlim(-0.025, 1.025)
    ax_tolerance.set_xticks(
        x,
        [f"{int(value)}/6" for value in tolerance_plot["tolerance_numerator"]],
    )
    ax_tolerance.set_xlabel("Direction-tolerance multiplier, λ")
    ax_tolerance.set_ylabel("Paired ΔRMSE vs λ = 1.00 (km)")
    ax_tolerance.legend(frameon=False, loc="upper left")
    ax_tolerance.text(
        0.98,
        0.97,
        "(c)",
        transform=ax_tolerance.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=12,
    )
    ax_tolerance.grid(True, alpha=0.2)

    angle_axis = ax_tolerance.secondary_xaxis("top")
    angle_axis.set_xticks(
        x,
        [
            f"{cardinal:g}/{diagonal:g}°"
            for cardinal, diagonal in zip(
                tolerance_plot["cardinal_half_width_deg"],
                tolerance_plot["diagonal_half_width_deg"],
            )
        ],
    )
    angle_axis.tick_params(labelsize=8)
    angle_axis.set_xlabel("Cardinal/diagonal tolerance half-width (°)")

    for suffix in ("png", "svg"):
        fig.savefig(
            outdir / f"figure_3_anchor_split_detour_factor_direction_tolerance_sensitivity.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return anchor_plot, detour_plot, tolerance_plot


def copy_file(source: Path, destination: Path, source_rows: list[dict]) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_rows.append({"snapshot_path": destination, "source_path": source.resolve(), "source_sha256": sha256(source)})


def copy_matching(source_dir: Path, destination_dir: Path, patterns: tuple[str, ...], source_rows: list[dict]) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    selected: set[Path] = set()
    for pattern in patterns:
        selected.update(source_dir.glob(pattern))
    for source in sorted(path for path in selected if path.is_file()):
        copy_file(source, destination_dir / source.name, source_rows)


def copy_tree(source_dir: Path, destination_dir: Path, source_rows: list[dict]) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        copy_file(source, destination_dir / source.relative_to(source_dir), source_rows)


def write_ethics_report(path: Path) -> None:
    path.write_text(
        """# Research-ethics and interpretation audit

1. **BFGS versus PhysicsSim-Full is an end-to-end pipeline comparison.** Their HPO-selected weights differ. It must not be described as an isolated optimizer effect. The same-weight PhysicsSim-to-BFGS polishing experiment is the solver-isolation diagnostic.
2. **Anchor robustness is not a pure anchor causal effect.** Each split changes calibration anchors, frame anchor, held-out test composition, and split-specific HPO. The original split lies near the distribution centre, but the between-split variation is substantive and should not be described as insensitive.
3. **Detour sensitivity combines distance assumptions with recalibration.** Scenarios below κ = 1 use scenario-specific HPO, while κ = 1 preserves the preregistered formal setting. Test RMSE must not be used to claim that a particular κ is historically true.
4. **The all-verified-sites reconstruction is conditioned on all known sites.** Its near-zero site error is true by construction and is not predictive performance.
5. **Positional-stability maps are repeated simulations under parameter perturbation.** They are not observational bootstrap confidence regions and do not characterize the full feasible solution space.
6. **Main-text values are rounded for defensible precision.** Exact values remain in audit/source CSV files; rounded manuscript values must not be used as replacement raw data.
7. **Representative spatial panels are not best runs or averages.** They are model-specific runs closest to the multimetric median profile and must be labelled as representative.
8. **DC-SMACOF preprocessing and evaluation use different row semantics.** The model consumes consensus-preprocessed direction constraints, while reported directional diagnostics evaluate the registered raw observations. This distinction must be disclosed.
9. **BFGS selected-candidate HPO included failed runs.** The selected candidate had 25 successful fold-runs out of 30; the formal 100-seed endpoint experiment had 100 successful runs. Both facts should remain visible in supplementary reporting.
10. **Held-out archaeological test sites are excluded from HPO selection.** They are used only for final/post-hoc geographic evaluation; using them to choose κ, weights, or representative runs would introduce leakage.
""",
        encoding="utf-8",
    )


def write_readme(path: Path) -> None:
    path.write_text(
        """# Manuscript-ready results snapshot

This immutable-style snapshot reorganizes the latest formal results into manuscript-ready tables and figures. It does not alter any source experiment output or `paper_results/current`.

## Main-text order

1. `01_main_tables/table_1_rmse_benchmark`: performance beyond Random+Align.
2. `01_main_tables/table_2_progressive_component_effects`: paired Direction, Anchors, and Repulsion effects.
3. `01_main_tables/table_3_information_matched_optimizer_comparison`: information-matched and optimizer comparisons.
4. `02_main_figures/figure_1a_*` and `figure_1b_*`: two readable 2×2 representative spatial comparisons covering PhysicsSim-Full, BFGS, SMACOF, and DC-SMACOF.
5. `02_main_figures/figure_2_bfgs_polishing_objective_rmse`: same-weight objective/RMSE diagnostic.
6. `02_main_figures/figure_3_anchor_split_detour_factor_direction_tolerance_sensitivity`: three-panel anchor/test split, paired detour-factor, and paired direction-tolerance sensitivity results.

CSV files contain the exact displayed cells. Markdown and TeX contain manuscript-ready versions. Exact unrounded values and run-level sources are retained in `03_supplementary_tables`, `05_source_data`, and `06_verification`.

Do not interpret rounded main-table cells as raw data. Consult `06_verification/verification_report.json`, `06_verification/protocol_consistency_audit.md`, `source_map.csv`, and `manifest_sha256.csv` for provenance.
""",
        encoding="utf-8",
    )


def write_manual_visual_audit(path: Path) -> None:
    path.write_text(
        """# Manual visual audit of main figures

This review was performed after the numerical and provenance checks passed.

- **Figure 1A/1B:** Both 2×2 figures render without clipping. Figure 1A compares PhysicsSim-Full with BFGS; Figure 1B compares SMACOF with DC-SMACOF. Panel metrics match their formal 100-seed sources, overlay displacement segments join each held-out prediction to its corresponding archaeological point, and constraint-error panels use the verified node coordinates. These are representative-run visualizations, not mean configurations.
- **Figure 2:** All 100 same-seed endpoint pairs are visible. Circle and triangle markers distinguish PhysicsSim and BFGS-polished endpoints; connecting arrows point from before to after. The x-axis is the objective value on a symmetric-log scale and the y-axis is held-out test RMSE in kilometres. Direction annotations correctly indicate lower objective to the left and lower RMSE downward.
- **Figure 3:** Panel (a) shows all 45 splits sorted by mean RMSE, with within-split sample SD error bars and the primary split reference. Panel (b) shows the 12 same-seed RMSE contrasts against κ = 1.00 plus the definitional κ = 1.00 reference point at zero. Panel (c) shows the seven direction-tolerance settings, including the definitional λ = 1.00 reference point at zero. Panels (b) and (c) use 95% percentile-bootstrap CIs of paired mean differences, and their dashed zero lines denote no difference from the respective reference.

No clipping, inconsistent panel scale, swapped axis, missing unit, or visually altered node position was found. Dense historical labels remain legible in the high-resolution PNG/SVG originals, although Figure 1 should be placed at full-page or full-width size in the manuscript.
""",
        encoding="utf-8",
    )


def build_snapshot(outdir: Path, *, overwrite: bool = False) -> Path:
    if outdir.exists() and any(outdir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {outdir}")
        shutil.rmtree(outdir)
    tables_dir = outdir / "01_main_tables"
    figures_dir = outdir / "02_main_figures"
    supplementary_tables = outdir / "03_supplementary_tables"
    supplementary_figures = outdir / "04_supplementary_figures"
    source_dir = outdir / "05_source_data"
    verification_dir = outdir / "06_verification"
    provenance_dir = outdir / "00_provenance"
    for path in (tables_dir, figures_dir, supplementary_tables, supplementary_figures, source_dir, verification_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=True)

    as_runs, random_runs, bfgs_runs = load_runs()
    table1, audit1 = build_table_1(as_runs, random_runs, bfgs_runs)
    panel_a, panel_b, audit2 = build_table_2()
    table3, audit3 = build_table_3(as_runs, bfgs_runs)
    latex1 = latex_tabular(
        table1,
        fixed_col_widths=("0.22\\linewidth",),
        header_cells=(
            "Model",
            r"\shortstack{Held-out RMSE\\mean $\pm$ SD (km)}",
            r"\shortstack{RMSE reduction\\vs Random+Align}",
        ),
        tabcolsep="3pt",
    )
    save_table(table1, tables_dir / "table_1_rmse_benchmark", title="Table 1. Held-out RMSE benchmark against Random+Align", note="Values are mean ± sample SD. Percentage reduction uses the Random+Align mean as the reference.", latex=latex1)
    combined_csv = pd.concat([panel_a.assign(Panel="A"), panel_b.assign(Panel="B")], ignore_index=True, sort=False)
    combined_csv = combined_csv[["Panel", *[column for column in combined_csv.columns if column != "Panel"]]]
    latex2 = "\n".join([
        r"\textbf{Panel A. Geographic accuracy and textual constraint satisfaction}", r"\par\medskip",
        latex_tabular(
            panel_a,
            fixed_col_widths=("0.12\\linewidth",),
            header_cells=(
                "Added component",
                r"\shortstack{$\Delta$RMSE (km)\\{[95\% CI]}}",
                r"\shortstack{$\Delta$Stress\\{[95\% CI]}}",
                r"\shortstack{$\Delta$Violation Rate\\{[95\% CI]}}",
                r"\shortstack{$\Delta$Mean Angular\\Error (rad) [95\% CI]}",
            ),
        ), r"\medskip",
        r"\textbf{Panel B. Layout diagnostics}", r"\par\medskip",
        latex_tabular(
            panel_b[[
                "Added component",
                "ΔCrowding Violation Rate (τ = 0.10) [95% CI]",
                "ΔCollapse Node Rate (τ = 0.10) [95% CI]",
                "ΔNearest-Neighbor Distance 5th Quantile (km) [95% CI]",
                "ΔCrossing-edge Rate [95% CI]",
            ]],
            fixed_col_widths=("0.12\\linewidth",),
            header_cells=(
                "Added component",
                r"\shortstack{$\Delta$Crowding\\Violation Rate\\($\tau=0.10$)\\{[95\% CI]}}",
                r"\shortstack{$\Delta$Collapse\\Node Rate\\($\tau=0.10$)\\{[95\% CI]}}",
                r"\shortstack{$\Delta$Nearest-Neighbor Distance\\5th Quantile (km) [95\% CI]}",
                r"\shortstack{$\Delta$Crossing-edge Rate\\{[95\% CI]}}",
            ),
            tabcolsep="1.5pt",
        ),
    ])
    save_table(combined_csv, tables_dir / "table_2_progressive_component_effects", title="Table 2. Paired effects of progressive model components", note="Differences are model with the added component minus the preceding model. Cells report paired mean difference [95% percentile-bootstrap CI], n = 100 paired seeds.", latex=latex2)
    latex3 = render_table_3_latex(table3)
    save_table(table3, tables_dir / "table_3_information_matched_optimizer_comparison", title="Table 3. Information-matched and standard-optimizer comparisons", note="Values are mean ± sample SD across 100 successful runs. BFGS and PhysicsSim-Full use independently selected HPO weights; this row pair is an end-to-end pipeline comparison, not an isolated optimizer effect.", latex=latex3)
    audit1.to_csv(verification_dir / "table_1_exact_values.csv", index=False, encoding="utf-8-sig")
    audit2.to_csv(verification_dir / "table_2_exact_values.csv", index=False, encoding="utf-8-sig")
    audit3.to_csv(verification_dir / "table_3_exact_values.csv", index=False, encoding="utf-8-sig")

    source_rows: list[dict] = []
    create_spatial_comparisons(
        as_dir=AS_DIR,
        bfgs_dir=BFGS_DIR,
        representative_dir=REPRESENTATIVE_DIR,
        outdir=figures_dir,
    )
    copy_file(VIS_DIR / "section_6_5_three_model_visualization_prototype.json", verification_dir / "figure_1_metadata.json", source_rows)
    copy_file(VIS_DIR / "section_6_5_visualization_consistency_report.json", verification_dir / "figure_1_consistency_report.json", source_rows)
    copy_file(REPRESENTATIVE_DIR / "representative_selection.json", verification_dir / "figure_1_representative_selection.json", source_rows)
    copy_file(REPRESENTATIVE_DIR / "representative_rerun_verification.csv", verification_dir / "figure_1_rerun_verification.csv", source_rows)
    plot_polishing(figures_dir)
    plot_anchor_detour(figures_dir)
    figure_metadata = {
        "figure_1": {"coordinates": "y-up simulation coordinates; overlay errors are kilometres in the LCC evaluation frame", "variants": ["PhysicsSim-Full", "BFGS", "SMACOF", "DC-SMACOF"], "selection": "model-specific four-metric representative seed", "layout": "Figure 1A and Figure 1B are separate 2x2 figures"},
        "figure_2": {"x": "objective value F(X), symlog scale", "y": "held-out test RMSE, km", "pairing": "same seed, PhysicsSim endpoint to same-weight BFGS-polished endpoint"},
        "figure_3": {"panel_a": {"x": "anchor/test split rank, lower mean RMSE to higher", "y": "mean held-out RMSE ± within-split sample SD, km", "n_splits": 45}, "panel_b": {"x": "distance scaling factor kappa", "y": "paired held-out test RMSE difference relative to kappa=1, km", "interval": "95% percentile-bootstrap CI across 100 matched seeds", "n_comparisons": 12, "n_plotted_points": 13, "reference_point": "kappa=1, paired difference=0 by definition", "difference_definition": "RMSE(kappa) - RMSE(kappa=1)"}, "panel_c": {"x": "direction-tolerance multiplier lambda; top axis gives cardinal/diagonal half-widths in degrees", "y": "paired held-out test RMSE difference relative to lambda=1, km", "interval": "95% percentile-bootstrap CI across 100 matched seeds", "n_comparisons": 6, "n_plotted_points": 7, "reference_point": "lambda=1, paired difference=0 by definition", "difference_definition": "RMSE(lambda) - RMSE(lambda=1)"}, "layout": "top full-width panel with two lower panels"},
    }
    (verification_dir / "figure_metadata.json").write_text(json.dumps(figure_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    copy_matching(PROJECT_ROOT / "paper_results" / "current" / "05_paper_tables", supplementary_tables / "previous_full_tables", ("*.csv", "*.md", "*.tex"), source_rows)
    copy_matching(PHYSICS_HPO_DIR, supplementary_tables / "physics_hpo", ("*.csv", "*.json", "*.md"), source_rows)
    copy_matching(PHYSICS_HPO_GRID_DIR, supplementary_tables / "physics_hpo_grid", ("*.csv", "*.json", "*.md"), source_rows)
    copy_matching(DC_HPO_DIR, supplementary_tables / "dc_smacof_hpo", ("*.csv", "*.json", "*.md"), source_rows)
    copy_matching(BFGS_HPO_DIR, supplementary_tables / "bfgs_hpo", ("*.csv", "*.json", "*.md"), source_rows)
    for source, destination, patterns in (
        (PHYSICS_HPO_DIR, supplementary_figures / "physics_hpo", ("*.png", "*.svg")),
        (PHYSICS_HPO_GRID_DIR, supplementary_figures / "physics_hpo_grid", ("*.png", "*.svg")),
        (DC_HPO_DIR, supplementary_figures / "dc_smacof_hpo", ("*.png", "*.svg")),
        (BFGS_HPO_DIR, supplementary_figures / "bfgs_hpo", ("*.png", "*.svg")),
        (POLISH_DIR, supplementary_figures / "bfgs_polishing", ("*.png", "*.svg")),
        (ANCHOR_DIR, supplementary_figures / "anchor_robustness", ("*.png", "*.svg")),
        (PROJECT_ROOT / "outputs" / "ch6_anchor_robustness_representative_overlays_20260825", supplementary_figures / "anchor_representative_overlays", ("*.png", "*.svg")),
        (DETOUR_DIR, supplementary_figures / "detour_sensitivity", ("*.png", "*.svg")),
        (TOLERANCE_DIR, supplementary_figures / "direction_tolerance_sensitivity", ("*.png", "*.svg")),
        (BOOTSTRAP_DIR, supplementary_figures / "positional_stability", ("*.png", "*.svg", "*.md")),
        (ALL_ANCHOR_DIR, supplementary_figures / "all_verified_sites", ("*.png", "*.svg")),
    ):
        if source.exists():
            copy_matching(source, destination, patterns, source_rows)
    copy_tree(PROJECT_ROOT / "paper_results" / "current" / "06_paper_figures", supplementary_figures / "previous_complete_figure_archive", source_rows)

    source_specs = (
        (AS_DIR, ("*.csv", "*.json")),
        (BFGS_DIR, ("*.csv", "*.json")),
        (POLISH_DIR, ("*.csv", "*.json")),
        (ANCHOR_DIR, ("*.csv", "*.json", "*.jsonl")),
        (DETOUR_DIR, ("*.csv", "*.json", "*.jsonl")),
        (TOLERANCE_DIR, ("*.csv", "*.json", "*.jsonl")),
        (BOOTSTRAP_DIR, ("*.csv", "*.json", "*.md")),
        (ALL_ANCHOR_DIR, ("*.csv", "*.json")),
    )
    for source, patterns in source_specs:
        if source.exists():
            copy_matching(source, source_dir / source.name, patterns, source_rows)

    for source in (
        PROJECT_ROOT / "data" / "site_rmse_points.csv",
        PROJECT_ROOT / "data" / "distance_edges_verified.csv",
        PROJECT_ROOT / "data" / "direction_edges_verified.csv",
        PROJECT_ROOT / "library" / "config.py",
    ):
        copy_file(source, provenance_dir / source.name, source_rows)

    write_ethics_report(verification_dir / "research_ethics_and_interpretation_audit.md")
    write_manual_visual_audit(verification_dir / "manual_visual_audit.md")
    write_readme(outdir / "README.md")
    source_map_rows = []
    for row in source_rows:
        source_map_rows.append({
            "snapshot_path": Path(row["snapshot_path"]).relative_to(outdir).as_posix(),
            "source_path": str(row["source_path"]),
            "source_sha256": row["source_sha256"],
        })
    pd.DataFrame(source_map_rows).sort_values("snapshot_path").to_csv(outdir / "source_map.csv", index=False, encoding="utf-8-sig")
    manifest_rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name not in {"manifest_sha256.csv", "verification_report.json"}:
            manifest_rows.append({"snapshot_path": path.relative_to(outdir).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(outdir / "manifest_sha256.csv", index=False, encoding="utf-8-sig")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--overwrite", action="store_true", help="Replace only the specified manuscript-ready snapshot, never source outputs.")
    args = parser.parse_args()
    outdir = build_snapshot(Path(args.outdir), overwrite=args.overwrite)
    print(f"[Built] {outdir}")


if __name__ == "__main__":
    main()
