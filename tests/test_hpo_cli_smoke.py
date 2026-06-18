import subprocess
import sys


def _run_help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", *args, "--help"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def test_paper_run_help_lists_current_hpo_command():
    stdout = _run_help("run_paper_script.paper_run")

    assert "ch5-hparam-kfold" in stdout
    assert "ch5_hparam_anchor_loo" in stdout


def test_hpo_module_help_lists_export_loo_review_flag():
    stdout = _run_help("run_paper_script.ch5_hparam_kfold_gridsearch_pareto")

    assert "--export-loo-review" in stdout
    assert "--alpha-min" in stdout


def test_export_hpo_loo_review_help_loads():
    stdout = _run_help("scripts.export_hpo_loo_review")

    assert "--hpo-outdir" in stdout
    assert "--alpha" in stdout
