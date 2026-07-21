"""Rebuild PhysicsSim HPO summaries with sample SD from existing HPO run CSVs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


FOLD_GROUP_COLUMNS = [
    "alpha", "beta", "w_dis", "w_dir", "w_reg", "fold_id", "train_labels", "train_anchor_label", "heldout_label",
]
GRID_GROUP_COLUMNS = ["alpha", "beta", "w_dis", "w_dir", "w_reg"]
METRICS = {
    "E_distance_stress": "E_distance_stress",
    "E_direction_vr": "E_direction_vr",
    "RMSE_anchor_LOO_km": "RMSE_anchor_LOO_km",
}
GRID_METRIC_COLUMNS = {
    "E_distance_stress": "E_distance_stress_mean",
    "E_direction_vr": "E_direction_vr_mean",
    "RMSE_anchor_LOO_km": "RMSE_anchor_LOO_mean_km",
}


def _sample_std(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _backup_files(backup_dir: Path, *files: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, backup_dir / file.name)


def _rebuild_final_summary(summary_path: Path, runs_path: Path) -> None:
    if not summary_path.exists() or not runs_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(runs_path)
    if "RMSE_final_test_km" in runs:
        values = runs["RMSE_final_test_km"].dropna()
        summary["RMSE_final_test_mean_km"] = float(values.mean())
        summary["RMSE_final_test_std_km"] = _sample_std(values)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_physics_hpo_statistics(*, hpo_outdir: str | Path, manual_outdir: str | Path, backup_dir: str | Path) -> dict[str, Path]:
    hpo_outdir = Path(hpo_outdir)
    manual_outdir = Path(manual_outdir)
    backup_dir = Path(backup_dir)
    runs_path = hpo_outdir / "grid_runs_by_seed.csv"
    folds_path = hpo_outdir / "grid_folds_mean_std.csv"
    grid_path = hpo_outdir / "grid_summary_cv.csv"
    pareto_path = hpo_outdir / "pareto_front_3d.csv"
    hpo_final_summary = hpo_outdir / "selected_final_summary.json"
    manual_pareto = manual_outdir / "pareto_candidates.csv"
    manual_candidate = manual_outdir / "selected_candidate_summary.csv"
    manual_final_summary = manual_outdir / "selected_final_summary.json"
    required = [runs_path, folds_path, grid_path, pareto_path, manual_pareto, manual_candidate]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HPO summary inputs: {missing}")
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Backup directory is not empty: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    _backup_files(
        backup_dir / hpo_outdir.name,
        folds_path,
        grid_path,
        pareto_path,
        *(path for path in [hpo_final_summary] if path.exists()),
    )
    _backup_files(
        backup_dir / manual_outdir.name,
        manual_pareto,
        manual_candidate,
        *(path for path in [manual_final_summary] if path.exists()),
    )

    runs = pd.read_csv(runs_path)
    fold_rows = []
    for keys, group in runs.groupby(FOLD_GROUP_COLUMNS, dropna=False, sort=False):
        row = dict(zip(FOLD_GROUP_COLUMNS, keys))
        row["n_seeds"] = int(len(group))
        row["n_failed_seeds"] = int(group["RMSE_anchor_LOO_km"].isna().sum())
        for source, target_mean, target_std in (
            ("E_distance_stress", "E_distance_stress_mean", "E_distance_stress_std"),
            ("E_direction_vr", "E_direction_vr_mean", "E_direction_vr_std"),
            ("RMSE_anchor_LOO_km", "RMSE_anchor_LOO_mean_km", "RMSE_anchor_LOO_std_km"),
        ):
            row[target_mean] = float(group[source].mean())
            row[target_std] = _sample_std(group[source])
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)

    old_grid = pd.read_csv(grid_path)
    grid_rows = []
    for keys, group in folds.groupby(GRID_GROUP_COLUMNS, dropna=False, sort=False):
        row = dict(zip(GRID_GROUP_COLUMNS, keys))
        static = old_grid
        for column, value in row.items():
            static = static[static[column] == value]
        if len(static) != 1:
            raise ValueError(f"Expected exactly one existing HPO grid row for {row}")
        existing = static.iloc[0]
        for column in ("spring_stiffness", "directional_force", "repulsion_strength", "is_pareto"):
            row[column] = existing[column]
        row["n_folds"] = int(len(group))
        row["n_seeds_per_fold"] = int(group["n_seeds"].iloc[0])
        row["n_failed_runs"] = int(group["n_failed_seeds"].sum())
        for source, target_mean, target_std in (
            ("E_distance_stress_mean", "E_distance_stress_mean", "E_distance_stress_std"),
            ("E_direction_vr_mean", "E_direction_vr_mean", "E_direction_vr_std"),
            ("RMSE_anchor_LOO_mean_km", "RMSE_anchor_LOO_mean_km", "RMSE_anchor_LOO_std_km"),
        ):
            row[target_mean] = float(group[source].mean())
            row[target_std] = _sample_std(group[source])
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows).sort_values(["alpha", "beta"]).reset_index(drop=True)
    fold_order = pd.read_csv(folds_path).columns.tolist()
    grid_order = old_grid.columns.tolist()
    folds = folds.reindex(columns=fold_order)
    grid = grid.reindex(columns=grid_order)
    pareto = grid[grid["is_pareto"]].copy()

    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")
    _rebuild_final_summary(hpo_final_summary, hpo_outdir / "selected_final_runs_by_seed.csv")
    _rebuild_final_summary(manual_final_summary, manual_outdir / "selected_final_runs_by_seed.csv")

    manual_candidates = pd.read_csv(manual_pareto)
    grid_lookup = grid.set_index(["alpha", "beta"])
    for index, row in manual_candidates.iterrows():
        matched = grid_lookup.loc[(row["alpha"], row["beta"])]
        for column in ("E_distance_stress_std", "E_direction_vr_std", "RMSE_anchor_LOO_std_km"):
            manual_candidates.loc[index, column] = matched[column]
    manual_candidates.to_csv(manual_pareto, index=False, encoding="utf-8-sig")

    selected = pd.read_csv(manual_candidate)
    for index, row in selected.iterrows():
        matched = grid_lookup.loc[(row["alpha"], row["beta"])]
        for column in ("E_distance_stress_std", "E_direction_vr_std", "RMSE_anchor_LOO_std_km"):
            selected.loc[index, column] = matched[column]
    selected.to_csv(manual_candidate, index=False, encoding="utf-8-sig")

    metadata = {
        "std_definition": "sample standard deviation (ddof=1; n=1 uses 0.0)",
        "source_runs": runs_path.name,
        "model_simulation_rerun": False,
        "original_summary_backup": str(backup_dir),
    }
    for destination in (hpo_outdir, manual_outdir):
        (destination / "hpo_summary_statistics_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"grid_summary": grid_path, "manual_candidate": manual_candidate, "backup": backup_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild PhysicsSim HPO summaries with sample SD without rerunning HPO.")
    parser.add_argument("--hpo-outdir", required=True)
    parser.add_argument("--manual-outdir", required=True)
    parser.add_argument("--backup-dir", required=True)
    args = parser.parse_args()
    paths = recompute_physics_hpo_statistics(
        hpo_outdir=args.hpo_outdir,
        manual_outdir=args.manual_outdir,
        backup_dir=args.backup_dir,
    )
    for label, path in paths.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
