"""run_paper_script.ch5_bootstrap_stability

Chapter 5 — Bootstrap stability test.

Runs repeated physics simulations under small parameter jitters and visualizes
uncertainty using:
  - confidence ellipses
  - combined KDE density map

Usage
-----
Run from the physics_simulation project root.
Default fixed anchors come from data/site_rmse_points.csv (use_role=anchor).

python -m run_paper_script.paper_run ch5-bootstrap --n-bootstrap 300 --spring-jitter 0.05 --repulse-jitter 0.20

Notes
-----
This script saves all artifacts under the repo-local output folder
`<OUTPUT_DIR>/ch5/bootstrap/`.
"""

from __future__ import annotations


import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--spring-jitter", type=float, default=0.05)
    p.add_argument("--repulse-jitter", type=float, default=0.20)
    p.add_argument(
        "--fixed",
        type=str,
        default="",
        help="Comma-separated anchor labels (must exist in ground truth).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # We call the updated bootstrap helpers (see library/bootstrap_and_visualization.py).
    from library.bootstrap_and_visualization import bootstrap_dynamics, plot_multi_ellipses, plot_kde_combined
    import numpy as np

    from library.data_io import uploading_ground_truth, save_bootstrap_data, get_anchor_labels
    from library.config import refer_pos_sim, refer_pos, FILE_PATHS
    from library.initialization import load_ini_data_from_csv
    from library.coordinates import flipping_y

    fixed_point_labels = [x.strip() for x in args.fixed.split(",") if x.strip()] or get_anchor_labels()

    _graph, vertice, dni, _edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    samples, vertice, dni = bootstrap_dynamics(
        int(args.n_bootstrap),
        float(args.spring_jitter),
        float(args.repulse_jitter),
        fixed_point_labels=fixed_point_labels,
        fixed_points_lonlat=fixed_points_lonlat
    )
    # Save bootstrap samples for the interactive map.
    # `save_bootstrap_data` expects positions in the *north-up* coordinate frame,
    # so we convert our plotting samples (pygame y-down) back to y-up.
    samples_y_up = np.asarray([s for s in samples], dtype=float)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    save_bootstrap_data(vertice, dni, samples_y_up, gt_lonlat, refer_pos=refer_pos_sim)

    plot_multi_ellipses(samples_y_up, vertice)
    plot_kde_combined(samples_y_up, vertice)


if __name__ == "__main__":
    main()
