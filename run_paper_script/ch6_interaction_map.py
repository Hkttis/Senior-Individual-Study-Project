"""run_paper_script.ch6_interaction_map

Chapter 6 — Interactive overlay map.

Build a Folium HTML map that overlays:
  - reconstructed nodes (from save_vis_data)
  - bootstrap samples (from save_bootstrap_data)
  - ground truth points
  - distance-error edges (from save_err_data)

This script is a thin wrapper around `scripts.execute_interaction` but exposes
the workflow as a chapter-aligned entrypoint.

Usage
-----
Run from the physics_simulation project root.

python -m run_paper_script.paper_run ch6-map
"""

from __future__ import annotations


def main() -> None:
    from scripts.execute_interaction import build_interactive_map

    build_interactive_map()


if __name__ == "__main__":
    main()
