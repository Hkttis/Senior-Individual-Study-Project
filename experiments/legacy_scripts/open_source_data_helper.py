#!/usr/bin/env python3
"""Generate the historical-name normalization crosswalk.

Important
---------
The mapping must be extracted from the raw Han Shu file
``GPT-4_漢書_numerals_utf8.csv``.  ``方向.csv`` already contains cleaned model
names, so using it as the source would always produce zero mappings.

Existing paths are obtained from ``library.config``:

- ``resolve_path("chen_data")`` provides the model-node whitelist.
- ``resolve_path("directional_data")`` identifies the configured data folder.
- The raw Han Shu CSV is read from that same folder.
- ``name_crosswalk.csv`` is written to that same folder using UTF-8 with BOM.

The Hou Han Shu CSV is never opened.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from library import config
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from library import config


SCRIPT_VERSION = "2026-06-12-fix-raw-hanshu-source"
NORMALIZATION_SUFFIXES = frozenset({"國", "城", "王"})
HAN_RAW_FILENAME = "GPT-4_漢書_numerals_utf8.csv"
OUTPUT_FILENAME = "name_crosswalk.csv"
OUTPUT_HEADER = ("史書原始名稱", "模型標準化名稱")


def normalize_place_name(raw_name: str) -> str:
    """Strip whitespace and remove exactly one recognized trailing suffix."""

    cleaned = raw_name.strip()
    if cleaned and cleaned[-1] in NORMALIZATION_SUFFIXES:
        return cleaned[:-1]
    return cleaned


def iter_first_two_columns(csv_path: Path) -> Iterator[str]:
    """Yield non-empty values from columns 1 and 2 after skipping the header."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for row_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{csv_path} row {row_number} has fewer than two columns: {row!r}"
                )

            for raw_name in row[:2]:
                cleaned = raw_name.strip()
                if cleaned:
                    yield cleaned


def load_legal_nodes(chen_data_path: Path) -> set[str]:
    """Build the legal model-node whitelist from Chen's distance CSV."""

    legal_nodes = set(iter_first_two_columns(chen_data_path))
    if not legal_nodes:
        raise ValueError(f"No legal model nodes were found in: {chen_data_path}")
    return legal_nodes


def build_name_crosswalk(
    legal_nodes: set[str],
    han_raw_data_path: Path,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    """Build valid mappings and return rejected suffix-removal candidates."""

    candidates: set[tuple[str, str]] = set()
    mappings: set[tuple[str, str]] = set()

    for original_name in iter_first_two_columns(han_raw_data_path):
        normalized_name = normalize_place_name(original_name)

        if original_name == normalized_name:
            continue

        mapping = (original_name, normalized_name)
        candidates.add(mapping)

        if normalized_name in legal_nodes:
            mappings.add(mapping)

    accepted = sorted(mappings, key=lambda pair: (pair[1], pair[0]))
    rejected = candidates - mappings
    return accepted, rejected


def write_name_crosswalk(
    rows: Iterable[Sequence[str]],
    output_path: Path,
) -> None:
    """Write an Excel-compatible UTF-8 CSV containing a BOM."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(rows)


def get_paths_from_config() -> tuple[Path, Path, Path]:
    """Resolve the whitelist, raw Han Shu input, and output paths."""

    chen_data_path = Path(config.resolve_path("chen_data"))

    # directional_data is used only to locate the configured data directory.
    # The contents of 方向.csv are NOT read when constructing this crosswalk.
    directional_data_path = Path(config.resolve_path("directional_data"))
    configured_data_dir = directional_data_path.parent
    han_raw_data_path = configured_data_dir / HAN_RAW_FILENAME

    # If resolve_path fell back to PROJECT_ROOT/data/方向.csv, also check DATA_DIR.
    if not han_raw_data_path.is_file():
        fallback_path = Path(config.DATA_DIR) / HAN_RAW_FILENAME
        if fallback_path.is_file():
            han_raw_data_path = fallback_path
        else:
            raise FileNotFoundError(
                "Raw Han Shu CSV was not found.\n"
                f"Checked: {han_raw_data_path}\n"
                f"Checked: {fallback_path}\n"
                f"Required filename: {HAN_RAW_FILENAME}"
            )

    output_path = configured_data_dir / OUTPUT_FILENAME
    return chen_data_path, han_raw_data_path, output_path


def main() -> int:
    print(f"Script version: {SCRIPT_VERSION}")

    try:
        chen_data_path, han_raw_data_path, output_path = get_paths_from_config()
        legal_nodes = load_legal_nodes(chen_data_path)
        rows, rejected = build_name_crosswalk(legal_nodes, han_raw_data_path)

        # Never overwrite the output with a misleading header-only file.
        if not rows:
            raise ValueError(
                "No valid name mappings were found, so no CSV was written. "
                "Verify that Raw Han Shu data points to "
                f"{HAN_RAW_FILENAME}, not 方向.csv."
            )

        write_name_crosswalk(rows, output_path)

    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Legal model nodes: {len(legal_nodes)}")
    print(f"Suffix-removal candidates: {len(rows) + len(rejected)}")
    print(f"Rejected by legal-node whitelist: {len(rejected)}")
    print(f"Crosswalk mappings: {len(rows)}")
    print(f"Chen data: {chen_data_path}")
    print(f"Raw Han Shu data: {han_raw_data_path}")
    print(f"Output: {output_path}")
    print("Encoding: UTF-8 with BOM (utf-8-sig)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
