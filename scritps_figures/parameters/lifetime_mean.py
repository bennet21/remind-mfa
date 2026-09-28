"""Parameter figure: cement mean lifetime under SSP1-5 plus the two circular-economy (CE)
variants.

One figure per end use (RS, RM, Com, Ind, Civ), with a 12-panel h12 region grid comparing the
lifetime-mean time series across the 7 scenarios. A vertical dashed line marks the last
historical year (2023). No global panel.

Run from the repository root:
    uv run --no-sync python scritps_figures/parameters/lifetime_mean.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    FIGURES_DIR,
    FUNCTION_DISPLAY_NAMES,
    LAST_HISTORICAL_YEAR,
    OTHER_CIV_NAME,
    OTHER_IND_NAME,
    REGION_DISPLAY_NAMES,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas, shared_range

YLABEL = "Mean lifetime (years)"
SSPS = list(SSP_SOURCE_PICKLES.keys())
OUTPUT_DIR = FIGURES_DIR / "parameters"
NCOLS = 4
END_USE_DISPLAY_NAMES = {**FUNCTION_DISPLAY_NAMES, "Ind": OTHER_IND_NAME, "Civ": OTHER_CIV_NAME}

combined_by_ssp = {
    ssp: load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"] for ssp in SSPS
}
time = combined_by_ssp[SSPS[0]].parameters["lifetime_mean"].dims["t"].items
regions = combined_by_ssp[SSPS[0]].parameters["lifetime_mean"].dims["r"].items
end_uses = combined_by_ssp[SSPS[0]].parameters["lifetime_mean"].dims["e"].items


def lifetime_mean(combined, end_use, region):
    return combined.parameters["lifetime_mean"][{"e": end_use, "r": region}].values


def save(fig, output_name: str, width: int, height: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (OUTPUT_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_end_use(end_use: str, output_name: str):
    nrows = math.ceil(len(regions) / NCOLS)
    fig = make_subplots(
        rows=nrows,
        cols=NCOLS,
        subplot_titles=[REGION_DISPLAY_NAMES.get(str(r), str(r)) for r in regions],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for index, region in enumerate(regions):
        row = index // NCOLS + 1
        col = index % NCOLS + 1
        for ssp in SSPS:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=lifetime_mean(combined_by_ssp[ssp], end_use, region),
                    mode="lines",
                    line={"color": SSP_COLORS[ssp], "width": 2, "dash": SSP_DASHES[ssp]},
                    name=SSP_LABELS[ssp],
                    legendgroup=ssp,
                    showlegend=(index == 0),
                ),
                row=row,
                col=col,
            )
        fig.add_vline(
            x=LAST_HISTORICAL_YEAR,
            line_color="black",
            line_dash="dash",
            line_width=1,
            opacity=0.7,
            row=row,
            col=col,
        )
        fig.update_yaxes(showgrid=True, row=row, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = {"size": 13}

    fig.update_xaxes(range=[time[0], time[-1]])
    fig.update_yaxes(range=shared_range(*(trace.y for trace in fig.data)))

    fig.update_layout(
        title=END_USE_DISPLAY_NAMES.get(end_use, end_use),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "font": {"size": 12},
        },
        margin={"t": 90, "l": 95, "b": 60, "r": 150},
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="Year",
                x=0.5, y=-0.05,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="top",
                font={"size": 16},
            ),
            dict(
                text=YLABEL,
                x=-0.065, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="middle",
                textangle=-90,
                font={"size": 16},
            ),
        ],
    )
    save(fig, output_name, width=1700, height=800)


for end_use in end_uses:
    plot_end_use(end_use, f"lifetime_mean_{end_use}_h12")

print("END")
