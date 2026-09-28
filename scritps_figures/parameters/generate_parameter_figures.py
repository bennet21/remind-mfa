"""Run the parameter figure scripts in this folder end-to-end.

Run from the repo root: `uv run --no-sync python scritps_figures/parameters/generate_parameter_figures.py`.
Each figure script is run as its own process, in order, exactly as if invoked by hand
(e.g. `uv run --no-sync python scritps_figures/parameters/floorspace.py`).
"""

import subprocess
import sys
from pathlib import Path

# Which parameter scripts to (re)generate: "all", or a list of script stems, e.g. ["floorspace"].
PARAMETERS = "all"

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
ALL_SCRIPTS = sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name != Path(__file__).name)


def selected_scripts() -> list[Path]:
    if PARAMETERS == "all":
        return ALL_SCRIPTS
    wanted = set(PARAMETERS)
    scripts = [s for s in ALL_SCRIPTS if s.stem in wanted]
    missing = wanted - {s.stem for s in scripts}
    if missing:
        raise ValueError(f"No parameter figure script found for: {sorted(missing)}")
    return scripts


def main() -> None:
    for script in selected_scripts():
        print("=" * 80)
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
