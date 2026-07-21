"""Create two contact sheets from progressive representative visualization PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="Representative visualization output directory.")
    return parser.parse_args()


def _safe_variant_name(variant: str) -> str:
    return variant.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("+", "plus")


def _write_sheet(outdir: Path, selections: list[dict], suffix: str, filename: str) -> Path:
    width, image_height, label_height = 800, 500, 42
    canvas = Image.new("RGB", (width * 3, (image_height + label_height) * 2), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
    for index, item in enumerate(selections):
        variant, seed = item["variant"], int(item["seed"])
        source = outdir / f"progressive_AS_{_safe_variant_name(variant)}_seed{seed}_{suffix}.png"
        if not source.exists():
            raise FileNotFoundError(f"Missing representative image: {source}")
        x = (index % 3) * width
        y = (index // 3) * (image_height + label_height)
        image = Image.open(source).convert("RGB").resize((width, image_height))
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 12, y + 8), f"{variant} | seed {seed}", fill=(23, 32, 51), font=font)
    target = outdir / filename
    canvas.save(target)
    return target


def main():
    args = _parse_args()
    outdir = Path(args.outdir)
    metadata = json.loads((outdir / "representative_selection.json").read_text(encoding="utf-8"))
    selections = metadata["selections"]
    if len(selections) != 6:
        raise ValueError("Contact sheets require exactly six representative variants.")
    print(_write_sheet(outdir, selections, "Overlap", "representative_ground_truth_overlays_contact_sheet.png"))
    print(_write_sheet(outdir, selections, "error_map_full", "representative_error_maps_contact_sheet.png"))


if __name__ == "__main__":
    main()
