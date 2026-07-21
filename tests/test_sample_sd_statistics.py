import math

import pandas as pd

from run_paper_script.ch5_ablation_study import _series_stats
from run_paper_script.ch5_ablation_progressive import METRICS
from scripts.recompute_progressive_statistics import recompute_progressive_statistics


def test_series_stats_reports_sample_standard_deviation():
    stats = _series_stats([1.0, 3.0])

    assert stats["std"] == math.sqrt(2.0)
    assert stats["se"] == 1.0


def test_recompute_progressive_statistics_uses_existing_runs_and_backs_up_original(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    rows = []
    for seed, value in enumerate((1.0, 3.0)):
        row = {"variant": "Random+Align", "seed": seed, "status": "ok", "error": ""}
        row.update({metric: value for metric in METRICS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_runs_by_seed.csv", index=False)
    pd.DataFrame({"old": ["summary"]}).to_csv(as_outdir / "progressive_summary.csv", index=False)
    pd.DataFrame({"old": ["random"]}).to_csv(as_outdir / "random_align_summary.csv", index=False)

    paths = recompute_progressive_statistics(as_outdir=as_outdir, backup_dir=tmp_path / "backup")

    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "std"] == math.sqrt(2.0)
    assert (tmp_path / "backup" / "progressive_summary.csv").exists()
    assert (tmp_path / "backup" / "random_align_summary.csv").exists()
