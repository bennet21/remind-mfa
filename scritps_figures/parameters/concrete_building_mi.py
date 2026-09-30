"""Parameter figure: cement concrete material intensity of buildings under SSP1-5 plus the two
circular-economy (CE) variants.

One figure per (structure, bottom-up end use) combination: a global panel (floorspace-weighted
average across all h12 regions) plus one panel per h12 region, comparing the material-intensity
time series across the 7 scenarios. A vertical dashed line marks the last historical year (2023).

Run from the repository root:
    uv run --no-sync python scritps_figures/parameters/concrete_building_mi.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    BACKGROUND,
    FIGURES_DIR,
    FUNCTION_DISPLAY_NAMES,
    LAST_HISTORICAL_YEAR,
    OTHER_STRUCTURE_KEYS,
    REGION_DISPLAY_NAMES,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
    STRUCTURE_DISPLAY_NAMES,
)
from helpers import load_mfas, shared_range
from remind_mfa.cement.cement_mfa_system_bottom_up import expand_common_to_bu

YLABEL = "Concrete material intensity (t/m²)"
SSPS = list(SSP_SOURCE_PICKLES.keys())
OUTPUT_DIR = FIGURES_DIR / "parameters"
NCOLS_REGIONAL = 4
STRUCTURES = [s for s in STRUCTURE_DISPLAY_NAMES if s not in OTHER_STRUCTURE_KEYS]
END_USES = list(FUNCTION_DISPLAY_NAMES)

combined_by_ssp = {
    ssp: load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"] for ssp in SSPS
}
time = combined_by_ssp[SSPS[0]].parameters["concrete_building_mi"].dims["t"].items
regions = combined_by_ssp[SSPS[0]].parameters["concrete_building_mi"].dims["r"].items


def concrete_mi(combined, structure, end_use, region=None):
    """Concrete material-intensity time series for one bottom-up end use; global (region=None)
    is floorspace-weighted across regions."""
    mi = combined.parameters["concrete_building_mi"][{"s": structure, "b": end_use}]  # (t, r)
    if region is not None:
        return mi[{"r": region}].values

    bu_floorspace = expand_common_to_bu(combined.stocks["floorspace"].stock, combined.parameters)[
        {"b": end_use}
    ]  # (t, r)
    weighted = (mi * bu_floorspace).sum_to("t")
    total = bu_floorspace.sum_to("t")
    return (weighted / total).values


def save(fig, output_name: str, width: int, height: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (OUTPUT_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def add_ssp_traces(fig, structure, end_use, region=None, showlegend=True, row=None, col=None):
    kwargs = {"row": row, "col": col} if row is not None else {}
    for ssp in SSPS:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=concrete_mi(combined_by_ssp[ssp], structure, end_use, region=region),
                mode="lines",
                line={"color": SSP_COLORS[ssp], "width": 2, "dash": SSP_DASHES[ssp]},
                name=SSP_LABELS[ssp],
                legendgroup=ssp,
                showlegend=showlegend,
            ),
            **kwargs,
        )


def plot_combined(structure: str, end_use: str, output_name: str):
    nrows = math.ceil(len(regions) / NCOLS_REGIONAL)
    ncols_total = NCOLS_REGIONAL + 1

    specs = [[{"rowspan": nrows}] + [{}] * NCOLS_REGIONAL]
    for _ in range(nrows - 1):
        specs.append([None] + [{}] * NCOLS_REGIONAL)

    subplot_titles = ["Global"] + [REGION_DISPLAY_NAMES.get(str(r), str(r)) for r in regions]

    fig = make_subplots(
        rows=nrows,
        cols=ncols_total,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
        column_widths=[1.5] + [1] * NCOLS_REGIONAL,
    )

    add_ssp_traces(fig, structure, end_use, region=None, showlegend=True, row=1, col=1)
    fig.add_vline(
        x=LAST_HISTORICAL_YEAR,
        line_color="black",
        line_dash="dash",
        line_width=1,
        opacity=0.7,
        row=1,
        col=1,
    )
    fig.update_yaxes(showgrid=True, row=1, col=1)

    for index, region in enumerate(regions):
        row = index // NCOLS_REGIONAL + 1
        col = index % NCOLS_REGIONAL + 2
        add_ssp_traces(fig, structure, end_use, region=region, showlegend=False, row=row, col=col)
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
        annotation.font = {"size": 12}

    fig.update_xaxes(range=[time[0], time[-1]])
    fig.update_yaxes(range=shared_range(*(trace.y for trace in fig.data)))

    title = f"{STRUCTURE_DISPLAY_NAMES[structure]} — {FUNCTION_DISPLAY_NAMES[end_use]}"

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "font": {"size": 11},
        },
        margin={"t": 100, "l": 95, "b": 60, "r": 120},
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="Year",
                x=0.5, y=-0.05,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="top",
                font={"size": 15},
            ),
            dict(
                text=YLABEL,
                x=-0.055, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="middle",
                textangle=-90,
                font={"size": 15},
            ),
        ],
    )
    save(fig, output_name, width=2100, height=850)


for structure in STRUCTURES:
    for end_use in END_USES:
        plot_combined(structure, end_use, f"concrete_building_mi_{structure}_{end_use}_h12_global")

print("END")
