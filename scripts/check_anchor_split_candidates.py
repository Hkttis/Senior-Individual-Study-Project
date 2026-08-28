"""Validate the anchor-region workbook before anchor-split experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_paper_script.ch5_anchor_split_robustness import (
    REGION_ORDER,
    build_anchor_splits,
    load_anchor_candidate_table,
)


def check_anchor_split_candidates(path: str | Path) -> dict:
    candidates = load_anchor_candidate_table(path)
    splits = build_anchor_splits(candidates)
    eligible = candidates[candidates["anchor_eligible"]]
    counts = {
        region: int((eligible["region_class"] == region).sum())
        for region in REGION_ORDER
    }
    original_matches = [split.split_id for split in splits if split.is_original_split]
    result = {
        "n_site_points": int(len(candidates)),
        "n_eligible": int(len(eligible)),
        "eligible_by_region": counts,
        "n_valid_stratified_splits": int(len(splits)),
        "original_split_ids": original_matches,
    }
    print("Anchor split candidate workbook: OK")
    print(f"  Site points: {result['n_site_points']}")
    print(f"  Eligible sites: {result['n_eligible']}")
    print(f"  Eligible by region: {counts}")
    print(f"  Valid stratified splits: {result['n_valid_stratified_splits']}")
    if original_matches:
        print(f"  Original formal split included: {original_matches[0]}")
    else:
        print("  Original formal split is not a valid split under the entered classification.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/anchor_robustness_candidates.xlsx")
    args = parser.parse_args()
    try:
        check_anchor_split_candidates(args.candidates)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Anchor split candidate workbook: NOT READY\n  {exc}") from exc


if __name__ == "__main__":
    main()
