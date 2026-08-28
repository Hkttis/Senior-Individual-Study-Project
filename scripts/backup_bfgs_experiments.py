"""Create a read-only-style copy of long-running BFGS outputs with SHA-256 hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def backup_experiments(*, sources: Sequence[str | Path], outdir: str | Path) -> Path:
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Backup directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    resolved = [Path(source).resolve() for source in sources]
    if not resolved:
        raise ValueError("At least one --source directory is required.")
    names = [source.name for source in resolved]
    if len(set(names)) != len(names):
        raise ValueError("Backup sources must have unique directory names.")
    for source in resolved:
        if not source.is_dir():
            raise FileNotFoundError(f"Backup source directory is missing: {source}")
        shutil.copytree(source, outdir / source.name)

    rows = []
    for path in sorted(outdir.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "backup_path": path.relative_to(outdir).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = outdir / "backup_manifest_sha256.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["backup_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "BACKUP_INFO.txt").write_text(
        "\n".join(
            [
                f"Created: {datetime.now().astimezone().isoformat()}",
                "Sources:",
                *[f"- {source}" for source in resolved],
                f"Files hashed: {len(rows)}",
                "Source directories were read only and were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    manifest = backup_experiments(sources=args.source, outdir=args.outdir)
    print(f"[Backed up and hashed] {manifest}")


if __name__ == "__main__":
    main()
