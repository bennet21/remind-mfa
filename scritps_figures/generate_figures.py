"""Run the numbered figure scripts in this folder end-to-end.

Run from the repo root: `uv run --no-sync python scritps_figures/generate_figures.py`.
Each figure script is run as its own process, in order, exactly as if invoked by hand
(e.g. `uv run --no-sync python scritps_figures/01_demand_by_structure.py`).
"""

import subprocess
import sys
from pathlib import Path

# Which figures to (re)generate: "all", or a list of numeric prefixes, e.g. ["01", "05"].
FIGURES = "all"

SCRIPTS_DIR = Path(__file__).parent
ALL_SCRIPTS = sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py"))


def selected_scripts() -> list[Path]:
    if FIGURES == "all":
        return ALL_SCRIPTS
    wanted = {str(f).zfill(2) for f in FIGURES}
    scripts = [s for s in ALL_SCRIPTS if s.name[:2] in wanted]
    missing = wanted - {s.name[:2] for s in scripts}
    if missing:
        raise ValueError(f"No figure script found for number(s): {sorted(missing)}")
    return scripts


def main() -> None:
    for script in selected_scripts():
        print("=" * 80)
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], cwd=SCRIPTS_DIR.parent, check=True)


if __name__ == "__main__":
    main()
