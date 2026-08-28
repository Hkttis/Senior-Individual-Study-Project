"""Verify the north-up bootstrap coordinate and KDE-HDR input pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS
from library.data_io import load_ini_data_from_csv, uploading_ground_truth
from library.directions import DIR4_SIM, SIM_Y_IS_UP
from run_paper_script.ch5_bootstrap_stability import _samples_to_lonlat
from scripts.export_bootstrap_node_hdr_catalog import _load_samples


def check_coordinate_pipeline(source_outdir: str | Path, catalog_outdir: str | Path | None = None) -> dict:
    source = Path(source_outdir)
    config = json.loads((source / "bootstrap_config.json").read_text(encoding="utf-8"))
    samples, labels, anchors = _load_samples(source)

    raw = pd.read_csv(source / "bootstrap_samples_y_up_sim.csv")
    direct = np.stack(
        [
            group.set_index("label").loc[labels, ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            for _, group in raw.groupby("bootstrap_index", sort=True)
        ]
    )
    source_diff = float(np.max(np.abs(samples - direct)))
    if source_diff != 0.0:
        raise ValueError(f"Catalog loader altered simulation coordinates: max_abs_diff={source_diff}")

    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    if list(vertice) != labels:
        raise ValueError("Bootstrap node order differs from the current formal graph.")
    gt_lonlat = uploading_ground_truth(vertice, dni)
    recomputed_lonlat = _samples_to_lonlat(samples, dni, gt_lonlat, config["anchor_align_label"])
    saved_frame = pd.read_csv(source / "bootstrap_samples_lonlat.csv")
    saved_lonlat = np.stack(
        [
            group.set_index("label").loc[labels, ["lon", "lat"]].to_numpy(float)
            for _, group in saved_frame.groupby("bootstrap_index", sort=True)
        ]
    )
    lonlat_diff = float(np.max(np.abs(recomputed_lonlat - saved_lonlat)))
    if lonlat_diff > 1e-10:
        raise ValueError(f"Saved lon/lat differs from inverse LCC output: max_abs_diff={lonlat_diff}")

    if not SIM_Y_IS_UP or not np.array_equal(DIR4_SIM["北"], np.asarray([0.0, 1.0])):
        raise ValueError("Direction-vector convention is not north-up (+y).")
    if not np.array_equal(DIR4_SIM["南"], np.asarray([0.0, -1.0])):
        raise ValueError("South direction vector is inconsistent with north-up coordinates.")

    anchor_indices = [dni[label] for label in anchors]
    anchor_y = samples[0, anchor_indices, 1]
    anchor_lat = saved_lonlat[0, anchor_indices, 1]
    order = np.argsort(anchor_lat)
    if not np.all(np.diff(anchor_y[order]) > 0.0):
        raise ValueError("Anchor simulation y does not increase with projected latitude.")

    catalog_count = None
    if catalog_outdir is not None:
        catalog = pd.read_csv(Path(catalog_outdir) / "node_hdr_catalog_index.csv")
        expected = set(labels).difference(anchors)
        if set(catalog["label"]) != expected:
            raise ValueError("HDR catalog labels differ from the non-anchor bootstrap labels.")
        ranks = sorted(catalog["variability_rank_desc"].astype(int).tolist())
        if ranks != list(range(1, len(expected) + 1)):
            raise ValueError("HDR catalog variability ranks are incomplete.")
        catalog_count = len(catalog)

    return {
        "status": "ok",
        "n_runs": int(samples.shape[0]),
        "n_nodes": int(samples.shape[1]),
        "coordinate_frame": "simulation x=east-positive, y=north-positive",
        "catalog_loader_max_abs_diff_sim": source_diff,
        "inverse_lcc_max_abs_diff_degrees": lonlat_diff,
        "anchor_count": len(anchors),
        "catalog_non_anchor_count": catalog_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outdir", required=True)
    parser.add_argument("--catalog-outdir")
    args = parser.parse_args()
    result = check_coordinate_pipeline(args.source_outdir, args.catalog_outdir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
