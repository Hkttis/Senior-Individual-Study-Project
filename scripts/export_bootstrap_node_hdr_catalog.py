"""Export per-node KDE-HDR diagnostics from an existing bootstrap result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from library.bootstrap_and_visualization import (
    empirical_positional_stability_summary,
    plot_appendix_hdr_panels,
)


def _safe_filename(label: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(label)).strip().rstrip(".")
    return cleaned or "node"


def _load_samples(source_outdir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    sample_path = source_outdir / "bootstrap_samples_y_up_sim.csv"
    config_path = source_outdir / "bootstrap_config.json"
    frame = pd.read_csv(sample_path)
    required = {"bootstrap_index", "label", "x_y_up_sim", "y_y_up_sim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap sample file is missing columns: {sorted(missing)}")
    if frame.duplicated(["bootstrap_index", "label"]).any():
        raise ValueError("Bootstrap sample file contains duplicate run-label rows.")

    first_index = frame["bootstrap_index"].min()
    labels = frame.loc[frame["bootstrap_index"].eq(first_index), "label"].tolist()
    run_groups = list(frame.groupby("bootstrap_index", sort=True))
    samples = np.stack(
        [
            group.set_index("label").loc[labels, ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            for _, group in run_groups
        ]
    )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Bootstrap samples contain NaN or infinite coordinates.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    anchors = list(config.get("calibration_labels", []))
    return samples, labels, anchors


def export_catalog(source_outdir: str | Path, outdir: str | Path, *, grid_size: int = 100) -> Path:
    source = Path(source_outdir)
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    individual_dir = output / "individual_nodes"
    individual_dir.mkdir()

    samples, labels, anchors = _load_samples(source)
    anchors_set = set(anchors)
    summary = empirical_positional_stability_summary(samples, labels)
    selected = [
        row["label"]
        for row in sorted(
            (row for row in summary if row["label"] not in anchors_set),
            key=lambda row: row["radial_q95_sim"],
            reverse=True,
        )
    ]
    catalog_rows = plot_appendix_hdr_panels(
        samples,
        labels,
        [output / "all_non_anchor_node_hdr_catalog.png", output / "all_non_anchor_node_hdr_catalog.svg"],
        anchor_labels=anchors,
        selected_labels=selected,
        ncols=4,
        grid_size=int(grid_size),
        figure_title="All non-anchor node distributions for manual review",
    )
    rank_by_label = {row["label"]: int(row["variability_rank_desc"]) for row in catalog_rows}

    for label in selected:
        rank = rank_by_label[label]
        stem = f"rank_{rank:02d}_{_safe_filename(label)}"
        plot_appendix_hdr_panels(
            samples,
            labels,
            [individual_dir / f"{stem}.png", individual_dir / f"{stem}.svg"],
            anchor_labels=anchors,
            selected_labels=[label],
            ncols=1,
            grid_size=int(grid_size),
            figure_title="Node-level reconstruction distribution",
        )

    pd.DataFrame(catalog_rows).sort_values("variability_rank_desc").to_csv(
        output / "node_hdr_catalog_index.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "source_output": str(source),
        "n_runs": int(samples.shape[0]),
        "n_nodes": int(samples.shape[1]),
        "excluded_fixed_anchors": anchors,
        "exported_node_count": len(selected),
        "selection_purpose": "manual_review_only",
        "panels_independently_scaled": True,
    }
    (output / "catalog_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    args = parser.parse_args()
    result = export_catalog(args.source_outdir, args.outdir, grid_size=args.grid_size)
    print(f"[Saved] {result}")


if __name__ == "__main__":
    main()
