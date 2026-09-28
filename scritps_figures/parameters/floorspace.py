"""Parameter figure: cement floorspace per capita under SSP1-5 plus the two circular-economy
(CE) variants.

One figure per h12 region, each with two subplots (Residential, Commercial), comparing the
floorspace-per-capita time series across the 7 scenarios. A vertical dashed line marks the last
historical year (2023).

Run from the repository root:
    uv run --no-sync python scritps_figures/parameters/floorspace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas

YLABEL = "Floorspace per capita (m²/cap)"
END_USES = [("Res", "Residential"), ("Com", "Commercial")]
SSPS = list(SSP_SOURCE_PICKLES.keys())
OUTPUT_DIR = FIGURES_DIR / "parameters"

combined_by_ssp = {
    ssp: load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"] for ssp in SSPS
}
time = combined_by_ssp[SSPS[0]].stocks["floorspace"].stock.dims["t"].items
regions = combined_by_ssp[SSPS[0]].stocks["floorspace"].stock.dims["r"].items


def floorspace(combined, region, end_use):
    fs = combined.stocks["floorspace"].stock[{"r": region, "c": end_use}].values
    population = combined.parameters["population"][{"r": region}].values
    return fs / population


def save(fig, output_name: str, width: int, height: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (OUTPUT_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_region(region: str):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[name for _, name in END_USES],
        horizontal_spacing=0.08,
    )

    for col, (end_use, _) in enumerate(END_USES, start=1):
        for ssp in SSPS:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=floorspace(combined_by_ssp[ssp], region, end_use),
                    mode="lines",
                    line={"color": SSP_COLORS[ssp], "width": 2, "dash": SSP_DASHES[ssp]},
                    name=SSP_LABELS[ssp],
                    legendgroup=ssp,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        fig.add_vline(
            x=LAST_HISTORICAL_YEAR,
            line_color="black",
            line_dash="dash",
            line_width=1,
            opacity=0.7,
            row=1,
            col=col,
        )
        fig.update_yaxes(title_text=YLABEL if col == 1 else None, showgrid=True, row=1, col=col)
        fig.update_xaxes(title_text="Year", row=1, col=col)

    fig.update_layout(
        title=REGION_DISPLAY_NAMES.get(region, region),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"size": 11}},
        margin={"l": 90, "r": 30, "t": 80},
    )
    save(fig, f"floorspace_{region}", width=1100, height=500)


for region in regions:
    plot_region(str(region))

print("END")
