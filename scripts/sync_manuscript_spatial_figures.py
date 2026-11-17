"""Publish the English spatial figures without replacing experiment results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from shutil import copy2
from textwrap import fill
from xml.etree import ElementTree as ET


STEMS = (
    "figure_1a_physics_full_vs_bfgs_spatial_reconstruction",
    "figure_1b_smacof_vs_dc_smacof_spatial_reconstruction",
)
FIGURES = tuple(f"{stem}.{ext}" for stem in STEMS for ext in ("png", "svg"))
METADATA = "figure_1_spatial_comparison_metadata.json"
INVARIANTS = ("figure_1_plotted_positions_y_up_sim.csv", "figure_1_bfgs_representative_selection.json")
ROOT = Path(__file__).resolve().parents[1]
NAMES_PATH = ROOT / "data" / "country_display_names_yu2006.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sync(source: Path, snapshot: Path, backup: Path, png_dirs: list[Path]) -> None:
    figures_dir = snapshot / "02_main_figures"
    manifest_path = snapshot / "manifest_sha256.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    by_file = {row["snapshot_path"]: row for row in manifest}

    for name in INVARIANTS:
        if sha256(source / name) != sha256(figures_dir / name):
            raise ValueError(f"Representative results changed: {name}")
    old = json.loads((figures_dir / METADATA).read_bytes())
    new = json.loads((source / METADATA).read_bytes())
    for key in old.keys() - {"country_display_names", "source_sha256"}:
        if old[key] != new[key]:
            raise ValueError(f"Figure metadata changed unexpectedly: {key}")
    if new["country_display_names"]["sha256"] != sha256(NAMES_PATH):
        raise ValueError("Figures were not generated with the current mapping.")
    with NAMES_PATH.open(encoding="utf-8-sig", newline="") as file:
        names = [row["english_name"] for row in csv.DictReader(file)]
    parts = {part for name in names for part in fill(name, width=22, break_long_words=False).splitlines()}
    for stem in STEMS:
        svg = source / f"{stem}.svg"
        tree = ET.parse(svg, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
        labels = {entry.text.strip() for entry in tree.iter(ET.Comment)}
        if not parts <= labels:
            raise ValueError(f"Missing English label(s) in {svg.name}: {sorted(parts - labels)}")
        if not (source / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid PNG: {stem}")

    targets = [(figures_dir / name, source / name) for name in (*FIGURES, METADATA)]
    for path, original in targets:
        row = by_file[path.relative_to(snapshot).as_posix()]
        if row["sha256"] != sha256(path) or int(row["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"Snapshot manifest does not match {path}")
        row.update(sha256=sha256(original), size_bytes=str(original.stat().st_size))
    for directory in png_dirs:
        targets.extend((directory / f"{stem}.png", source / f"{stem}.png") for stem in STEMS)
    destinations = [path.resolve(strict=True) for path, _ in targets]
    if len(set(destinations)) != len(destinations) or source.resolve() in {p.parent for p in destinations}:
        raise ValueError("Duplicate targets or a target in the source directory.")

    backup.mkdir(parents=True, exist_ok=False)
    copy2(manifest_path, backup / manifest_path.name)
    for index, (path, _) in enumerate(targets):
        copy2(path, backup / f"{index:02d}_{path.name}")
    (backup / "destinations.json").write_text(json.dumps([str(p) for p in destinations], indent=2), encoding="utf-8")
    for path, original in targets:
        copy2(original, path)
        if sha256(path) != sha256(original):
            raise OSError(f"Copy verification failed: {path}")
        print(f"[Updated] {path}")
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    sync(args.source, args.snapshot, args.backup, args.png_dir)
