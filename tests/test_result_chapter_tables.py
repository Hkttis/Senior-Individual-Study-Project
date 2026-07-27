import pandas as pd

from scripts.export_result_chapter_tables import (
    SECTION_6_1_METRICS,
    SECTION_6_2_1_DIRECTION_METRICS,
    export_section_6_1_random_vs_physics_full,
    export_section_6_1_rmse_reduction_vs_random,
    export_section_6_2_1_distonly_vs_distdir,
    export_section_6_4_overall_model_comparison,
)


def _write_sources(folder):
    folder.mkdir()
    random_values = {
        "RMSE (km)": "662.745 ± 156.103",
        "Stress": "0.757824 ± 0.0512478",
        "Violation Rate": "0.591409 ± 0.0688178",
        "Mean Angular Error (rad)": "0.915555 ± 0.107076",
        "Crossing-edge rate": "0.231821 ± 0.0416826",
    }
    pd.DataFrame(
        [{"metric": metric, "Random+Align": value} for metric, value in random_values.items()]
    ).to_csv(folder / "table_random_layout_mean_sd.csv", index=False)
    pd.DataFrame(
        [
            {"variant": "PhysicsSim-DistOnly", "RMSE (km)": "547.976 ± 143.439"},
            {"variant": "PhysicsSim-DistDir", "RMSE (km)": "214.643 ± 45.0897"},
            {"variant": "PhysicsSim-DistDirAnch", "RMSE (km)": "195.074 ± 45.636"},
            {"variant": "PhysicsSim-Full", "RMSE (km)": "184.617 ± 45.2686"},
        ]
    ).to_csv(folder / "table_progressive_chain_mean_sd.csv", index=False)
    pd.DataFrame([{"model": "SMACOF", "RMSE (km)": "520.311 ± 141.548"}]).to_csv(
        folder / "table_smacof_vs_distonly_information_matched_comparison.csv", index=False
    )
    pd.DataFrame([{"model": "DC-SMACOF", "RMSE (km)": "248.252 ± 25.5509"}]).to_csv(
        folder / "table_dc_smacof_vs_distdir_information_matched_comparison.csv", index=False
    )
    full = {"model": "PhysicsSim-Full"}
    full.update(
        {
            "RMSE (km)": "184.617 ± 45.2686",
            "Stress": "0.0599776 ± 0.0166382",
            "Violation Rate": "0.0293182 ± 0.0168767",
            "Mean Angular Error (rad)": "0.117678 ± 0.0290839",
            "Crossing-edge rate": "0.0159238 ± 0.00470645",
        }
    )
    pd.DataFrame([full]).to_csv(folder / "table_overall_model_comparison_comparison.csv", index=False)


def test_section_6_1_tables_keep_requested_columns_and_export_latex(tmp_path):
    sources = tmp_path / "sources"
    outdir = tmp_path / "tables"
    _write_sources(sources)

    comparison_paths = export_section_6_1_random_vs_physics_full(
        paper_table_dir=sources, outdir=outdir
    )
    reduction_paths = export_section_6_1_rmse_reduction_vs_random(
        paper_table_dir=sources, outdir=outdir
    )

    comparison = pd.read_csv(comparison_paths["csv"])
    assert comparison.columns.tolist() == ["Model", *SECTION_6_1_METRICS]
    assert comparison["Model"].tolist() == ["Random+Align", "PhysicsSim-Full"]

    reduction = pd.read_csv(reduction_paths["csv"])
    assert reduction.columns.tolist() == [
        "Model",
        "RMSE, mean ± SD (km)",
        "RMSE reduction vs Random+Align",
    ]
    dc = reduction[reduction["Model"] == "DC-SMACOF"].iloc[0]
    assert dc["RMSE, mean ± SD (km)"] == "248.252 ± 25.5509"
    assert dc["RMSE reduction vs Random+Align"] == "62.54%"
    assert "62.54\\%" in reduction_paths["latex"].read_text(encoding="utf-8")
    reduction_latex = reduction_paths["latex"].read_text(encoding="utf-8")
    assert r"\shortstack{RMSE reduction vs\\Random+Align}" in reduction_latex
    assert r"p{0.340\linewidth}" in reduction_latex
    assert r"662.745 $\pm$ 156.103" in reduction_latex
    assert r"\renewcommand{\arraystretch}{1.30}" in reduction_latex
    assert r"PhysicsSim-DistDirAnch" in reduction_latex
    assert r"PhysicsSim-\\DistDirAnch" not in reduction_latex
    comparison_latex = comparison_paths["latex"].read_text(encoding="utf-8")
    assert r"\shortstack{Mean\\Angular\\Error\\(rad)}" in comparison_latex
    assert r"\renewcommand{\arraystretch}{1.45}" in comparison_latex
    assert comparison_latex.count(r"\begin{longtable}") == 1


def test_curated_paired_table_uses_exact_columns_and_paper_label(tmp_path):
    sources = tmp_path / "sources"
    outdir = tmp_path / "tables"
    sources.mkdir()
    rows = []
    for model in ("PhysicsSim-DistOnly", "PhysicsSim-DistDir", "DistDir - DistOnly paired"):
        row = {"model": model}
        row.update({metric: "1 ± 0.1" for metric in SECTION_6_2_1_DIRECTION_METRICS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        sources / "table_distonly_vs_distdir_paired_comparison.csv",
        index=False,
    )

    paths = export_section_6_2_1_distonly_vs_distdir(
        paper_table_dir=sources,
        outdir=outdir,
    )

    table = pd.read_csv(paths["csv"])
    assert table.columns.tolist() == ["Model", *SECTION_6_2_1_DIRECTION_METRICS]
    assert table["Model"].tolist()[-1] == "Paired difference: DistDir − DistOnly"
    latex = paths["latex"].read_text(encoding="utf-8")
    assert "Panel A. Site accuracy and constraint satisfaction" in latex
    assert "Panel B. Layout and topology diagnostics" in latex
    assert latex.count(r"\begin{longtable}") == 2
    assert latex.count("% DATA_ROWS_BEGIN") == 2
    assert r"\addlinespace[4pt]" in latex
    assert r"\vspace{0.7\baselineskip}" in latex


def test_overall_table_starts_on_a_fresh_page_and_keeps_two_panels(tmp_path):
    sources = tmp_path / "sources"
    outdir = tmp_path / "tables"
    sources.mkdir()
    rows = []
    for model in ("SMACOF", "DC-SMACOF", "PhysicsSim-Full"):
        row = {"model": model}
        row.update({metric: "1 簣 0.1" for metric in SECTION_6_2_1_DIRECTION_METRICS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        sources / "table_overall_model_comparison_comparison.csv",
        index=False,
    )

    paths = export_section_6_4_overall_model_comparison(
        paper_table_dir=sources,
        outdir=outdir,
    )

    latex = paths["latex"].read_text(encoding="utf-8")
    assert latex.startswith("\\newpage\n")
    assert latex.count(r"\begin{longtable}") == 2
    assert latex.count("% DATA_ROWS_BEGIN") == 2
