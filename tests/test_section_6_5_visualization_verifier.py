import struct

from scripts.verify_section_6_5_visualizations import (
    VIS_FILES,
    _sha256,
    _verify_source_hashes,
    _verify_visual_file,
    copy_visualizations_to_paper_results,
)


def test_source_hash_verification_detects_changed_input(tmp_path):
    source = tmp_path / "progressive_runs_by_seed.csv"
    source.write_text("variant,seed\nPhysicsSim-Full,0\n", encoding="utf-8")
    metadata = {"source_sha256": {source.name: _sha256(source)}}

    failures = []
    _verify_source_hashes(metadata, [source], "visualization", failures)
    assert failures == []

    source.write_text("variant,seed\nPhysicsSim-Full,1\n", encoding="utf-8")
    failures = []
    _verify_source_hashes(metadata, [source], "visualization", failures)
    assert failures == [f"visualization source SHA-256 mismatch: {source.name}"]


def test_visual_file_validation_accepts_svg_and_png_headers(tmp_path):
    svg = tmp_path / "figure.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle cx="1" cy="1" r="1"/></svg>', encoding="utf-8")
    png = tmp_path / "figure.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 640, 480))

    failures = []
    _verify_visual_file(svg, failures)
    _verify_visual_file(png, failures)
    assert failures == []


def test_visual_file_validation_rejects_invalid_svg(tmp_path):
    svg = tmp_path / "figure.svg"
    svg.write_text("not svg", encoding="utf-8")

    failures = []
    _verify_visual_file(svg, failures)
    assert any("invalid SVG XML" in failure for failure in failures)


def test_verified_copy_includes_report_and_updates_readme(tmp_path):
    vis_dir = tmp_path / "visual"
    paper_results = tmp_path / "paper_results"
    vis_dir.mkdir()
    paper_results.mkdir()
    for name in VIS_FILES:
        (vis_dir / name).write_text(name, encoding="utf-8")
    report_name = "section_6_5_visualization_consistency_report.json"
    (vis_dir / report_name).write_text('{"status":"ok"}', encoding="utf-8")
    (paper_results / "README.md").write_text(
        "Representative Section 6.5 visualization subdirectories predate this rerun and\n"
        "must be regenerated before they are used for revised DC-SMACOF claims.",
        encoding="utf-8",
    )

    destination = copy_visualizations_to_paper_results(
        vis_dir=vis_dir,
        paper_results=paper_results,
    )

    assert (destination / report_name).exists()
    assert "passed scripts.verify_section_6_5_visualizations" in (
        paper_results / "README.md"
    ).read_text(encoding="utf-8")
    assert (paper_results / "manifest_sha256.csv").exists()
