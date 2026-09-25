"""Same-weight controls for the softened exact-direction formulation.

The experiment runs (1) unanchored PhysicsSim Dist+Dir and (2) anchored BFGS
Full.  Sector and softened-exact variants share each seed's initialization and
the original sector-selected alpha/beta.  Earlier experiment code and outputs
are not modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, MIN_DISTANCE_BASE, refer_pos_sim
from library.data_io import (
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.directional_objectives import (
    DIRECTIONAL_OBJECTIVE_SECTOR,
    DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
)
from library.initialization import generate_CHEN_initial_positions
from library.physics import main_physics_simulation
from library.progressive_alignment import place_in_anchor_frame
from library.scipy_minimizer import run_bfgs
from library.scipy_objective import ObjectiveWeights, build_current_objective
from library.scipy_soft_exact_objective import build_current_softened_exact_objective
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim
from run_paper_script.ch5_sector_exact_control import OFFICIAL_METRICS


VARIANTS = (DIRECTIONAL_OBJECTIVE_SECTOR, DIRECTIONAL_OBJECTIVE_SOFT_EXACT)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_seed(block: str, seed: int, alpha: float, beta: float, delta: float) -> dict:
    _, vertices, dni, _, distances = load_ini_data_from_csv(FILE_PATHS)
    ground_truth = uploading_ground_truth(vertices, dni)
    anchors = get_anchor_labels()
    tests = get_test_site_labels()
    targets = _target_positions_sim(dni, ground_truth, "鄯善", refer_pos_sim)
    directions = uploading_directional_data()
    anchored = block == "bfgs_full"
    np.random.seed(seed)
    generated_vertices, generated_dni, _, initial, fixed = generate_CHEN_initial_positions(
        list(refer_pos_sim),
        anchors if anchored else [],
        [ground_truth[dni[label]] for label in anchors] if anchored else [],
        anchor_label="鄯善",
    )
    assert generated_vertices == vertices and generated_dni == dni
    initial = np.asarray(initial, dtype=float)
    initial_sha = hashlib.sha256(initial.tobytes()).hexdigest()
    weights = ObjectiveWeights.from_physics_hpo(alpha=alpha, beta=beta)
    rows = []

    for mode in VARIANTS:
        started = time.perf_counter()
        row = {
            "variant": mode,
            "seed": seed,
            "block": block,
            "initial_sha256": initial_sha,
            "model_anchor_count": len(fixed),
            "direction_softening_delta": delta if mode == DIRECTIONAL_OBJECTIVE_SOFT_EXACT else None,
            "status": "failed",
            "error": "",
        }
        points = None
        try:
            if anchored:
                if mode == DIRECTIONAL_OBJECTIVE_SECTOR:
                    problem = build_current_objective(weights=weights)
                else:
                    problem = build_current_softened_exact_objective(
                        weights=weights,
                        direction_softening_delta=delta,
                    )
                centered = initial - np.asarray(refer_pos_sim)
                np.testing.assert_allclose(
                    centered[problem.anchor_indices],
                    problem.anchor_coordinates,
                    atol=1e-10,
                    rtol=0.0,
                )
                y0 = problem.pack(centered)
                history = []

                def record(y: np.ndarray) -> None:
                    value, gradient = problem.fun_and_jac(y)
                    history.append(
                        {
                            "iteration": len(history),
                            "objective": value,
                            "gradient_norm_inf": float(np.linalg.norm(gradient, np.inf)),
                        }
                    )

                record(y0)
                result = run_bfgs(y0, problem, callback=record)
                row.update({key: value for key, value in result.items() if key != "y_final"})
                row["objective_initial"] = problem.fun(y0)
                row["history"] = history
                if result["y_final"] is not None:
                    final = result["y_final"]
                    record(final)
                    points = problem.unpack(final) + np.asarray(refer_pos_sim)
                    row["components_final"] = asdict(problem.components(final))
                    row["max_anchor_error_sim"] = float(
                        np.max(
                            np.abs(
                                points[problem.anchor_indices]
                                - initial[problem.anchor_indices]
                            )
                        )
                    )
                ok = result["success"]
            else:
                assert not fixed
                _, _, _, final = main_physics_simulation(
                    vertices,
                    dni,
                    data_Li2sim(distances),
                    initial.copy(),
                    directions,
                    [],
                    weights.distance,
                    0.0,
                    weights.direction,
                    plot=False,
                    directional_objective=mode,
                    direction_softening_delta=delta,
                )
                raw = np.asarray(final, dtype=float)
                row["raw_positions"] = raw.tolist()
                points = place_in_anchor_frame(raw, dni, "鄯善", refer_pos_sim)
                translation = points - raw
                assert np.allclose(translation, translation[0], atol=1e-10, rtol=0.0)
                row["alignment_translation_sim"] = translation[0].tolist()
                ok = True

            if points is not None:
                row.update(
                    _evaluate(
                        mode,
                        seed,
                        points,
                        vertices,
                        dni,
                        data_Li2sim(distances),
                        directions,
                        tests,
                        targets,
                        distances,
                    )
                )
                row["positions"] = points.tolist()
            row["status"] = "ok" if ok else "failed"
            row["error"] = "" if ok else str(row.get("failure_reason"))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
    return {
        "block": block,
        "seed": seed,
        "vertices": vertices,
        "initial_positions": initial.tolist(),
        "runs": rows,
    }


def aggregate(out: Path, blocks: list[str]) -> None:
    for block in blocks:
        folder = out / block
        rows, positions, histories, initials = [], [], [], []
        for path in sorted(folder.glob("seed_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for label, point in zip(data["vertices"], data["initial_positions"]):
                initials.append(
                    {
                        "seed": data["seed"],
                        "label": label,
                        "x_y_up_sim": point[0],
                        "y_y_up_sim": point[1],
                    }
                )
            for row in data["runs"]:
                rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("positions", "raw_positions", "history", "components_final")
                    }
                )
                for label, point in zip(data["vertices"], row.get("positions", [])):
                    positions.append(
                        {
                            "variant": row["variant"],
                            "seed": row["seed"],
                            "label": label,
                            "x_y_up_sim": point[0],
                            "y_y_up_sim": point[1],
                        }
                    )
                histories.extend(
                    {"variant": row["variant"], "seed": row["seed"], **item}
                    for item in row.get("history", [])
                )
        runs = pd.DataFrame(rows)
        runs.to_csv(folder / "runs.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(positions).to_csv(folder / "positions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(initials).to_csv(folder / "initial_positions.csv", index=False, encoding="utf-8-sig")
        if histories:
            pd.DataFrame(histories).to_csv(folder / "objective_history.csv", index=False)

        successful = runs[runs["status"] == "ok"]
        summaries, paired = [], []
        for mode, group in successful.groupby("variant"):
            for metric in OFFICIAL_METRICS:
                values = group[metric].to_numpy(float)
                summaries.append(
                    {
                        "variant": mode,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
                )
        for index, metric in enumerate(OFFICIAL_METRICS):
            if successful.empty:
                continue
            pivot = successful.pivot(index="seed", columns="variant", values=metric)
            if not set(VARIANTS).issubset(pivot.columns):
                continue
            pivot = pivot.dropna(subset=list(VARIANTS))
            if pivot.empty:
                continue
            differences = (
                pivot[DIRECTIONAL_OBJECTIVE_SOFT_EXACT]
                - pivot[DIRECTIONAL_OBJECTIVE_SECTOR]
            ).to_numpy()
            rng = np.random.default_rng(20260906 + index)
            bootstrap = rng.choice(
                differences, (10_000, len(differences)), replace=True
            ).mean(axis=1)
            lo, hi = np.quantile(bootstrap, [0.025, 0.975])
            paired.append(
                {
                    "metric": metric,
                    "n_pairs": len(differences),
                    "soft_exact_minus_sector": float(differences.mean()),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
            )
        pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False)
        pd.DataFrame(
            paired,
            columns=[
                "metric",
                "n_pairs",
                "soft_exact_minus_sector",
                "ci95_lo",
                "ci95_hi",
            ],
        ).to_csv(folder / "paired_comparison.csv", index=False)
        counts = runs.groupby(["variant", "status"]).size().to_dict()
        (folder / "completion.json").write_text(
            json.dumps({str(key): int(value) for key, value in counts.items()}, indent=2),
            encoding="utf-8",
        )
        print(block, counts, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--blocks",
        nargs="+",
        choices=["bfgs_full", "physics_distdir"],
        default=["bfgs_full", "physics_distdir"],
    )
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=-0.5)
    parser.add_argument("--direction-softening-delta", type=float, default=MIN_DISTANCE_BASE)
    args = parser.parse_args()
    if args.n_seeds < 1 or args.workers < 1:
        parser.error("--n-seeds and --workers must be positive")
    if not np.isfinite(args.direction_softening_delta) or args.direction_softening_delta <= 0:
        parser.error("--direction-softening-delta must be finite and positive")

    out = Path(args.outdir)
    protocol_path = out / "protocol.json"
    if out.exists() and any(out.iterdir()) and not protocol_path.exists():
        raise FileExistsError(
            f"Refusing to use non-empty output directory without matching protocol: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(FILE_PATHS[key]) for key in ("chen_data", "directional_data", "ground_truth_path")]
    sources += [
        root / path
        for path in (
            "library/directional_objectives.py",
            "library/scipy_objective.py",
            "library/scipy_soft_exact_objective.py",
            "library/scipy_minimizer.py",
            "library/physics.py",
            "library/initialization.py",
            "library/progressive_alignment.py",
            "library/geometry.py",
            "library/metrics.py",
            "library/config.py",
            "run_paper_script/ch5_matched_soft_exact_controls.py",
        )
    ]
    config = {
        "protocol": "same_weight_sector_vs_softened_exact_direction_controls",
        "blocks": args.blocks,
        "seeds": list(range(args.n_seeds)),
        "alpha": args.alpha,
        "beta": args.beta,
        "direction_softening_delta": args.direction_softening_delta,
        "directional_objective_formula": (
            "0.5*w_dir*sum((s^2*phi^2+delta^2*pi^2)/(s^2+delta^2))"
        ),
        "full_weights": asdict(
            ObjectiveWeights.from_physics_hpo(alpha=args.alpha, beta=args.beta)
        ),
        "distdir_repulsion_weight": 0,
        "distdir_model_anchors": 0,
        "hpo": False,
        "evaluation": "existing 8 held-out sites and sector reporting metrics; no test-site fitting",
        "bfgs": "full-memory analytic-gradient BFGS; collision C1 extension; antipodal branch remains excluded",
        "physics": "1001 updates; sector/softened-exact receive identical initial coordinates",
        "bootstrap": "10000 paired percentile replicates; seed 20260906+metric index",
        "legacy_outputs_modified": False,
        "source_sha256": {str(path.resolve()): _sha(path) for path in sources},
    }
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Refusing to reuse output with changed protocol or sources")
    protocol_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for source in sources:
        target = out / "executed_source_snapshot" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha(target) != _sha(source):
            raise ValueError(f"Existing source snapshot differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)

    jobs = []
    for block in args.blocks:
        (out / block).mkdir(exist_ok=True)
        for seed in range(args.n_seeds):
            if not (out / block / f"seed_{seed:03d}.json").exists():
                jobs.append((block, seed))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_seed,
                block,
                seed,
                args.alpha,
                args.beta,
                args.direction_softening_delta,
            ): (block, seed)
            for block, seed in jobs
        }
        for future in as_completed(futures):
            block, seed = futures[future]
            data = future.result()
            (out / block / f"seed_{seed:03d}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            statuses = ", ".join(
                f"{row['variant']}={row['status']}" for row in data["runs"]
            )
            print(f"{block} seed {seed}: {statuses}", flush=True)
    aggregate(out, args.blocks)


if __name__ == "__main__":
    main()
