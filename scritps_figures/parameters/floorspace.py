"""Parameter figure: cement floorspace per capita under SSP1-5 plus the two circular-economy
(CE) variants.

One figure per h12 region, plus a global figure (floorspace and population summed across
regions before dividing), each with two subplots (Residential, Commercial), comparing the
floorspace-per-capita time series across the 7 scenarios. A vertical dashed line marks the last
historical year (2023).

Run from the repository root:
    uv run --no-sync python scritps_figures/parameters/floorspace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    BACKGROUND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas, shared_range

YLABEL = "Floorspace per capita (m²/cap)"
END_USES = [("Res", "Residential"), ("Com", "Commercial")]
SSPS = list(SSP_SOURCE_PICKLES.keys())
OUTPUT_DIR = FIGURES_DIR / "parameters"

combined_by_ssp = {
    ssp: load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"] for ssp in SSPS
}
regions = combined_by_ssp[SSPS[0]].stocks["floorspace"].stock.dims["r"].items

_full_time = combined_by_ssp[SSPS[0]].stocks["floorspace"].stock.dims["t"].items

# Floorspace is zeroed out for a leading run of years (see
# cement_mfa_system_bottom_up.compute_floorspace_stock); skip that run rather than hardcode a year.
_raw_floorspace = combined_by_ssp[SSPS[0]].stocks["floorspace"].stock.sum_to("t").values
_display_start_index = int(np.flatnonzero(_raw_floorspace)[0])
FIRST_DISPLAY_YEAR = _full_time[_display_start_index]
time = _full_time[_display_start_index:]


def floorspace(combined, region, end_use):
    """Floorspace per capita from FIRST_DISPLAY_YEAR onward; global (region=None) sums
    floorspace and population across regions before dividing."""
    if region is not None:
        fs = combined.stocks["floorspace"].stock[{"r": region, "c": end_use}]
        population = combined.parameters["population"][{"r": region}]
    else:
        fs = combined.stocks["floorspace"].stock[{"c": end_use}].sum_to("t")
        population = combined.parameters["population"].sum_to("t")
    return (fs / population).values[_display_start_index:]


# Shared y-range per end use, spanning all regions, the global aggregate, and all scenarios, so
# every figure is visually comparable.
Y_RANGES = {
    end_use: shared_range(
        *(floorspace(combined_by_ssp[ssp], region, end_use) for ssp in SSPS for region in regions),
        *(floorspace(combined_by_ssp[ssp], None, end_use) for ssp in SSPS),
    )
    for end_use, _ in END_USES
}


def save(fig, output_name: str, width: int, height: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (OUTPUT_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_region(region, title: str, output_name: str):
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
        fig.update_yaxes(
            title_text=YLABEL if col == 1 else None,
            showgrid=True,
            range=Y_RANGES[end_use],
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Year", row=1, col=col)

    fig.update_xaxes(range=[time[0], time[-1]])

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"size": 11}},
        margin={"l": 90, "r": 30, "t": 80},
    )
    save(fig, output_name, width=1100, height=500)


for region in regions:
    region = str(region)
    plot_region(region, REGION_DISPLAY_NAMES.get(region, region), f"floorspace_{region}")

plot_region(None, "Global", "floorspace_Global")

print("END")
