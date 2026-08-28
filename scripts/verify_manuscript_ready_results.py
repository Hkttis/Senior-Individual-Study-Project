"""Verify the manuscript-ready snapshot against immutable formal outputs."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_manuscript_ready_results import (
    ANCHOR_DIR,
    AS_DIR,
    BFGS_DIR,
    DEFAULT_OUTDIR,
    DETOUR_DIR,
    POLISH_DIR,
    REPRESENTATIVE_DIR,
    VIS_DIR,
    build_table_1,
    build_table_2,
    build_table_3,
    load_runs,
    sha256,
)
from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim
from scripts.create_manuscript_spatial_comparisons import select_bfgs_representative
from scripts.verify_section_6_5_visualizations import verify_section_6_5_visualizations


def _assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str, failures: list[str]) -> None:
    actual = actual.fillna("").astype(str)
    expected = expected.fillna("").astype(str)
    if list(actual.columns) != list(expected.columns) or not actual.equals(expected):
        failures.append(f"{label} differs from values recomputed from formal run-level sources")


def _check_render_file(path: Path, failures: list[str]) -> None:
    if not path.exists() or path.stat().st_size < 1000:
        failures.append(f"missing or unexpectedly small figure: {path}")
        return
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            width, height = image.size
        if width < 1200 or height < 800:
            failures.append(f"figure resolution is too small: {path} ({width}x{height})")
    elif path.suffix.lower() == ".svg":
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            failures.append(f"invalid SVG {path}: {exc}")


def _check_table_formats(outdir: Path, failures: list[str]) -> None:
    for stem in (
        "table_1_rmse_benchmark",
        "table_2_progressive_component_effects",
        "table_3_information_matched_optimizer_comparison",
    ):
        csv_path = outdir / "01_main_tables" / f"{stem}.csv"
        md_path = csv_path.with_suffix(".md")
        tex_path = csv_path.with_suffix(".tex")
        for path in (csv_path, md_path, tex_path):
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"missing manuscript table format: {path}")
        if md_path.exists() and "|" not in md_path.read_text(encoding="utf-8"):
            failures.append(f"Markdown table is malformed: {md_path}")
        if tex_path.exists() and "\\begin{tabular}" not in tex_path.read_text(encoding="utf-8"):
            failures.append(f"LaTeX table is malformed: {tex_path}")


def verify_snapshot(outdir: Path) -> tuple[list[str], dict]:
    failures: list[str] = []
    checks: dict[str, object] = {}
    if not outdir.exists():
        return [f"snapshot does not exist: {outdir}"], checks

    as_runs, random_runs, bfgs_runs = load_runs()
    expected1, audit1 = build_table_1(as_runs, random_runs, bfgs_runs)
    panel_a, panel_b, audit2 = build_table_2()
    expected2 = pd.concat([panel_a.assign(Panel="A"), panel_b.assign(Panel="B")], ignore_index=True, sort=False)
    expected2 = expected2[["Panel", *[column for column in expected2.columns if column != "Panel"]]]
    expected3, audit3 = build_table_3(as_runs, bfgs_runs)
    for name, expected in (
        ("table_1_rmse_benchmark", expected1),
        ("table_2_progressive_component_effects", expected2),
        ("table_3_information_matched_optimizer_comparison", expected3),
    ):
        path = outdir / "01_main_tables" / f"{name}.csv"
        if not path.exists():
            failures.append(f"missing main table: {path}")
        else:
            _assert_frame(pd.read_csv(path, dtype=str, keep_default_na=False), expected, name, failures)
    exact_specs = (
        (outdir / "06_verification" / "table_1_exact_values.csv", audit1),
        (outdir / "06_verification" / "table_2_exact_values.csv", audit2),
        (outdir / "06_verification" / "table_3_exact_values.csv", audit3),
    )
    for path, expected in exact_specs:
        if not path.exists():
            failures.append(f"missing exact-value audit table: {path}")
        else:
            actual = pd.read_csv(path)
            if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
                failures.append(f"exact-value audit schema differs: {path}")
            else:
                for column in expected.columns:
                    if pd.api.types.is_numeric_dtype(expected[column]):
                        if not np.allclose(actual[column].to_numpy(float), expected[column].to_numpy(float), atol=1e-12, rtol=1e-12, equal_nan=True):
                            failures.append(f"exact-value mismatch in {path.name}/{column}")
                    elif not actual[column].fillna("").astype(str).equals(expected[column].fillna("").astype(str)):
                        failures.append(f"exact-value label mismatch in {path.name}/{column}")
    checks["main_tables"] = {"status": "ok" if not any("table" in item for item in failures) else "failed", "source": "recomputed from formal run-level CSV files"}
    _check_table_formats(outdir, failures)

    plot2 = pd.read_csv(outdir / "02_main_figures" / "figure_2_plot_data.csv")
    source2 = pd.read_csv(POLISH_DIR / "polishing_runs.csv")[["seed", "before_objective_total", "after_objective_total", "before_RMSE_test_km_posthoc", "after_RMSE_test_km_posthoc"]]
    if (
        not plot2["seed"].equals(source2["seed"])
        or not np.allclose(
            plot2.drop(columns="seed").to_numpy(float),
            source2.drop(columns="seed").to_numpy(float),
            atol=1e-9,
            rtol=1e-12,
        )
    ):
        failures.append("Figure 2 plotted data differ from polishing_runs.csv")
    plot3 = pd.read_csv(outdir / "02_main_figures" / "figure_3_plot_data.csv")
    source3 = pd.read_csv(ANCHOR_DIR / "anchor_split_summary.csv", encoding="utf-8-sig")[["split_id", "is_original_split", "RMSE_final_test_mean_km", "RMSE_final_test_std_km", "n_seeds"]].sort_values("RMSE_final_test_mean_km").reset_index(drop=True)
    source3["rank"] = np.arange(1, len(source3) + 1)
    if list(plot3.columns) != list(source3.columns) or not np.allclose(plot3.select_dtypes(include=[np.number]), source3.select_dtypes(include=[np.number]), atol=1e-12, rtol=1e-12) or not plot3["split_id"].equals(source3["split_id"]):
        failures.append("Figure 3 plotted data differ from anchor_split_summary.csv")
    plot4 = pd.read_csv(outdir / "02_main_figures" / "figure_4_plot_data.csv")
    source4 = pd.read_csv(DETOUR_DIR / "detour_scenario_summary.csv", encoding="utf-8-sig")[list(plot4.columns)].sort_values("kappa").reset_index(drop=True)
    for column in plot4.columns:
        if pd.api.types.is_numeric_dtype(source4[column]):
            if not np.allclose(plot4[column], source4[column], atol=1e-12, rtol=1e-12, equal_nan=True):
                failures.append(f"Figure 4 plotted data mismatch: {column}")
        elif not plot4[column].fillna("").astype(str).equals(source4[column].fillna("").astype(str)):
            failures.append(f"Figure 4 plotted labels mismatch: {column}")
    visual_failures, visual_checks = verify_section_6_5_visualizations(
        as_outdir=AS_DIR,
        representative_dir=REPRESENTATIVE_DIR,
        vis_dir=VIS_DIR,
        expected_variants=["PhysicsSim-Full", "SMACOF", "DC-SMACOF"],
    )
    failures.extend(f"Figure 1: {failure}" for failure in visual_failures)
    figure_dir = outdir / "02_main_figures"
    selection_path = figure_dir / "figure_1_bfgs_representative_selection.json"
    plotted_path = figure_dir / "figure_1_plotted_positions_y_up_sim.csv"
    if not selection_path.exists() or not plotted_path.exists():
        failures.append("Figure 1 BFGS selection or plotted-position audit file is missing")
        bfgs_selection = None
    else:
        bfgs_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected_selection = select_bfgs_representative(bfgs_runs)
        if bfgs_selection != expected_selection:
            failures.append("Figure 1 BFGS representative selection differs from the formal 100-seed median-profile rule")
        plotted = pd.read_csv(plotted_path, encoding="utf-8-sig")
        if len(plotted) != 140 or plotted.groupby("variant").size().to_dict() != {
            "BFGS": 35,
            "DC-SMACOF": 35,
            "PhysicsSim-Full": 35,
            "SMACOF": 35,
        }:
            failures.append("Figure 1 plotted-position audit must contain 35 nodes for each of four models")
        as_positions = pd.read_csv(AS_DIR / "progressive_final_positions_y_up_sim.csv", encoding="utf-8-sig")
        bfgs_positions = pd.read_csv(BFGS_DIR / "bfgs_final_positions_y_up_sim.csv", encoding="utf-8-sig")
        for variant, source_variant, source in (
            ("PhysicsSim-Full", "PhysicsSim-Full", as_positions),
            ("SMACOF", "SMACOF", as_positions),
            ("DC-SMACOF", "DC-SMACOF", as_positions),
            ("BFGS", "SciPy-BFGS", bfgs_positions),
        ):
            shown = plotted.loc[plotted["variant"].eq(variant)].sort_values("label").reset_index(drop=True)
            seed = int(shown["seed"].iloc[0]) if len(shown) else -1
            formal = source.loc[source["variant"].eq(source_variant) & source["seed"].eq(seed)].sort_values("label").reset_index(drop=True)
            if (
                len(shown) != len(formal)
                or not shown["label"].equals(formal["label"])
                or not np.allclose(
                    shown[["x_y_up_sim", "y_y_up_sim"]],
                    formal[["x_y_up_sim", "y_y_up_sim"]],
                    atol=1e-12,
                    rtol=1e-12,
                )
            ):
                failures.append(f"Figure 1 plotted positions differ from formal source for {variant} seed {seed}")

        graph, vertices, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
        del graph, edges
        gt_lonlat = uploading_ground_truth(vertices, dni)
        directional_data = uploading_directional_data()
        as_config = json.loads((AS_DIR / "progressive_config.json").read_text(encoding="utf-8"))
        tests = list(as_config["test_labels"])
        targets = _target_positions_sim(dni, gt_lonlat, str(as_config["anchor_align_label"]), as_config.get("refer_pos_sim", refer_pos_sim))
        bfgs_points_frame = plotted.loc[plotted["variant"].eq("BFGS")].set_index("label").loc[vertices]
        bfgs_points = bfgs_points_frame[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        recomputed = _evaluate(
            "SciPy-BFGS",
            int(bfgs_selection["seed"]),
            bfgs_points,
            vertices,
            dni,
            data_Li2sim(distance_data),
            directional_data,
            tests,
            targets,
            distance_data,
        )
        for metric in ("RMSE_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae"):
            if not np.isclose(recomputed[metric], bfgs_selection["rerun_metrics"][metric], atol=1e-9, rtol=1e-9):
                failures.append(f"Figure 1 BFGS recomputed metric mismatch: {metric}")
        visual_checks["BFGS"] = {
            "seed": int(bfgs_selection["seed"]),
            "metrics": {metric: float(recomputed[metric]) for metric in ("RMSE_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae")},
            "n_positions": int(len(bfgs_points)),
            "position_source_sha256": sha256(BFGS_DIR / "bfgs_final_positions_y_up_sim.csv"),
        }
    checks["figure_1_representative_models"] = visual_checks
    for figure in (
        "figure_1a_physics_full_vs_bfgs_spatial_reconstruction",
        "figure_1b_smacof_vs_dc_smacof_spatial_reconstruction",
        "figure_2_bfgs_polishing_objective_rmse",
        "figure_3_anchor_split_sensitivity",
        "figure_4_detour_rmse_sensitivity",
    ):
        for suffix in (".png", ".svg"):
            _check_render_file(outdir / "02_main_figures" / f"{figure}{suffix}", failures)
    metadata_path = outdir / "06_verification" / "figure_metadata.json"
    if not metadata_path.exists():
        failures.append("missing figure axis/unit metadata")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_phrases = ("km", "symlog", "kappa", "representative")
        text = json.dumps(metadata, ensure_ascii=False).lower()
        for phrase in required_phrases:
            if phrase not in text:
                failures.append(f"figure metadata omits required coordinate/unit concept: {phrase}")
    checks["main_figures"] = {
        "figure_1a_1b": "four model-specific representative runs; all plotted positions match formal sources and BFGS metrics were independently recomputed",
        "figure_2": "all 100 paired endpoints exactly copied from polishing_runs.csv",
        "figure_3": "all 45 split means and within-split SDs exactly copied from anchor summary",
        "figure_4": "all 13 kappa means and bootstrap bounds exactly copied from detour summary",
    }

    source_map_path = outdir / "source_map.csv"
    if not source_map_path.exists():
        failures.append("missing source_map.csv")
    else:
        source_map = pd.read_csv(source_map_path, encoding="utf-8-sig")
        for row in source_map.itertuples(index=False):
            snapshot = outdir / row.snapshot_path
            source = Path(row.source_path)
            if not snapshot.exists() or not source.exists():
                failures.append(f"missing provenance endpoint: {row.snapshot_path}")
            elif sha256(snapshot) != row.source_sha256 or sha256(source) != row.source_sha256:
                failures.append(f"source-copy SHA-256 mismatch: {row.snapshot_path}")
    manifest_path = outdir / "manifest_sha256.csv"
    if not manifest_path.exists():
        failures.append("missing manifest_sha256.csv")
    else:
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
        for row in manifest.itertuples(index=False):
            path = outdir / row.snapshot_path
            if not path.exists() or sha256(path) != row.sha256 or path.stat().st_size != int(row.size_bytes):
                failures.append(f"snapshot manifest mismatch: {row.snapshot_path}")
    ethics = outdir / "06_verification" / "research_ethics_and_interpretation_audit.md"
    if not ethics.exists() or len(ethics.read_text(encoding="utf-8")) < 1000:
        failures.append("research-ethics audit is missing or incomplete")
    manual_audit = outdir / "06_verification" / "manual_visual_audit.md"
    if not manual_audit.exists() or len(manual_audit.read_text(encoding="utf-8")) < 500:
        failures.append("manual visual audit is missing or incomplete")
    checks["provenance"] = "all copied files checked with SHA-256 against their registered sources"
    return failures, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()
    outdir = Path(args.paper_results)
    failures, checks = verify_snapshot(outdir)
    report = {"status": "failed" if failures else "ok", "failures": failures, "checks": checks}
    report_path = outdir / "06_verification" / "verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("[FAIL] Manuscript-ready results verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("[OK] Tables, plotted data, representative positions, units, and provenance are consistent with formal outputs.")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
