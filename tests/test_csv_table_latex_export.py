import pandas as pd

from scripts.export_csv_tables_to_latex import (
    dataframe_to_longtable,
    export_csv_tables_to_latex,
)


def test_dataframe_to_longtable_keeps_columns_and_escapes_latex():
    frame = pd.DataFrame(
        [{"Model_name": "A&B", "Reduction": "62.54%", "Value": "1 ± 0.2"}]
    )

    latex = dataframe_to_longtable(frame)

    assert r"Model\_name & Reduction & Value \\" in latex
    assert r"A\&B & 62.54\% & 1 ± 0.2 \\" in latex
    assert r"\begin{longtable}" in latex


def test_directory_export_creates_one_tex_per_csv_and_preserves_existing(tmp_path):
    pd.DataFrame([{"A": 1}]).to_csv(tmp_path / "first.csv", index=False)
    pd.DataFrame([{"B": 2}]).to_csv(tmp_path / "second.csv", index=False)
    custom = tmp_path / "first.tex"
    custom.write_text("custom layout", encoding="utf-8")

    outputs = export_csv_tables_to_latex(tmp_path, overwrite=False)

    assert set(outputs) == {"first", "second"}
    assert custom.read_text(encoding="utf-8") == "custom layout"
    assert (tmp_path / "second.tex").exists()

    export_csv_tables_to_latex(tmp_path, overwrite=True)
    assert custom.read_text(encoding="utf-8") != "custom layout"

