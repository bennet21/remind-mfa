"""Export selected figures with transparent background to a custom folder (e.g. for a poster).

Run from the repo root: `uv run --no-sync python scritps_figures/export_poster_figures.py`.
The figure scripts are run unchanged, except for module-level settings given as overrides in
`POSTER_FIGURES`. Only `plotly`'s `Figure.write_image` is intercepted: files that are not requested are
skipped, the background is made transparent and the figure is written to
`OUTPUT_DIR` instead of `FIGURES_DIR`, in `IMAGE_FORMAT`. A script is stopped once all its requested figures are written.
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

import plotly.basedatatypes

OUTPUT_DIR = Path(r"C:\Users\bennetwe\Nextcloud\Conferences\2026_IAMC_Athens\Poster\figures")

# Export format: "svg" or "png" (the figure scripts themselves always request PNG file names).
IMAGE_FORMAT = "svg"


@dataclass
class PosterFigures:
    """One run of a figure script and the figures to export from it.

    Attributes:
        script: File name of the figure script in `SCRIPTS_DIR`.
        figures: PNG names (without extension) to export.
        overrides: Values replacing module-level settings of the script, by name.
    """

    script: str
    figures: list[str]
    overrides: dict[str, object] = field(default_factory=dict)

    def run(self) -> None:
        """Execute the script with the module-level assignments in `overrides` replaced."""
        path = SCRIPTS_DIR / self.script
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        missing = set(self.overrides)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            else:
                continue
            if isinstance(target, ast.Name) and target.id in self.overrides:
                node.value = ast.parse(repr(self.overrides[target.id]), mode="eval").body
                missing.discard(target.id)
        if missing:
            raise ValueError(f"{self.script} has no module-level setting(s): {sorted(missing)}")
        code = compile(ast.fix_missing_locations(tree), str(path), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(path)})


POSTER_FIGURES = [
    PosterFigures("15_h12_region_map.py", ["fig15_h12_region_map_compact"]),
    PosterFigures("01_demand_by_structure.py", ["fig1_demand_by_structure_IND_SSP2"],
                  {"SCENARIOS": ["SSP2"], "SINGLE_REGION": "IND", "SINGLE_REGION_MAPPING": "h12"}),
    PosterFigures("01_demand_by_structure.py", ["fig1_demand_by_structure_USA_SSP2"],
                  {"SCENARIOS": ["SSP2"], "SINGLE_REGION": "USA", "SINGLE_REGION_MAPPING": "h12"}),
    PosterFigures("01_demand_by_structure.py", ["fig1_demand_by_structure_GRC_SSP2"],
                  {"SCENARIOS": ["SSP2"], "SINGLE_REGION": "GRC",
                   "SINGLE_REGION_MAPPING": "gtap155"}),
    PosterFigures("12_scenario_overview.py", ["fig12_scenario_overview", "fig12_circularity_depth"]),
    PosterFigures("11_ce_waterfall_emissions.py", ["fig11_ce_waterfall_process_emissions_h12"]),
    PosterFigures("03_production_by_ssp.py", ["fig3_production_by_ssp_agg_horizontal"]),
]

TRANSPARENT = "rgba(0,0,0,0)"
SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent


class AllExported(Exception):
    """Raised to stop a figure script once all requested figures are written."""


class TransparentExporter:
    """Replaces `Figure.write_image` to export only wanted figures, transparently, to `OUTPUT_DIR`.

    Attributes:
        wanted: PNG file names still to be exported from the current script.
    """

    def __init__(self, original_write_image):
        self.original_write_image = original_write_image
        self.wanted: set[str] = set()

    def __call__(self, figure, file, *args, **kwargs):
        name = Path(file).name
        if name not in self.wanted:
            return None
        figure.update_layout(
            paper_bgcolor=TRANSPARENT,
            plot_bgcolor=TRANSPARENT,
            legend_bgcolor=TRANSPARENT,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = (OUTPUT_DIR / name).with_suffix(f".{IMAGE_FORMAT}")
        self.original_write_image(figure, target, *args, **kwargs)
        print(f"Exported transparent figure to: {target}")
        self.wanted.discard(name)
        if not self.wanted:
            raise AllExported
        return None


def main() -> None:
    figure_class = plotly.basedatatypes.BaseFigure
    exporter = TransparentExporter(figure_class.write_image)
    figure_class.write_image = lambda figure, file, *args, **kwargs: exporter(
        figure, file, *args, **kwargs
    )

    sys.path.insert(0, str(SCRIPTS_DIR))
    for poster_figures in POSTER_FIGURES:
        print("=" * 80)
        print(f"Running {poster_figures.script} {poster_figures.overrides or ''}...")
        exporter.wanted = {f"{name}.png" for name in poster_figures.figures}
        try:
            poster_figures.run()
        except AllExported:
            pass
        if exporter.wanted:
            raise RuntimeError(f"{poster_figures.script} did not write: {sorted(exporter.wanted)}")


if __name__ == "__main__":
    main()
