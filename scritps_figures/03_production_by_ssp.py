"""Figure 3: cement production under SSP1-5 plus the two circular-economy (CE) variants.

Line plot comparing total (or per-region) cement production from the top-down MFA
for the five SSP scenarios and the SSP1_CE / SSP2_CE circular-economy variants
(drawn dashed in their parent SSP colour). A vertical dashed line marks the last
historical year (2023). Produces a global figure and a 12-panel regional figure.

Run from the repository root:
    uv run python scritps_figures/03_production_by_ssp.py
"""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    AGG_REGIONS,
    AGG_REGION_ORDER,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas

MT_PER_T = 1e-6
YLABEL = "Cement production (Mt)"
SSPS = list(SSP_SOURCE_PICKLES.keys())

combined_by_ssp = {
    ssp: load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"] for ssp in SSPS
}
time = combined_by_ssp["SSP2"].stocks["in_use"].stock.dims["t"].items
regions = combined_by_ssp["SSP2"].stocks["in_use"].inflow.dims["r"].items


def production(combined, region=None):
    region_items = [
        source_region
        for source_region, aggregate_region in AGG_REGIONS.items()
        if aggregate_region == region
    ] or [region]
    result = None
    for source_region in region_items:
        filter_dict = {"k": "cement"}
        if source_region is not None:
            filter_dict["r"] = source_region
        arr = combined.stocks["in_use"].inflow[filter_dict].sum_to("t") * MT_PER_T
        result = arr if result is None else result + arr
    return result


def add_ssp_traces(fig, region=None, showlegend=True, row=None, col=None):
    kwargs = {"row": row, "col": col} if row is not None else {}
    for ssp in SSPS:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=production(combined_by_ssp[ssp], region=region).values,
                mode="lines",
                line={"color": SSP_COLORS[ssp], "width": 2, "dash": SSP_DASHES[ssp]},
                name=SSP_LABELS[ssp],
                legendgroup=ssp,
                showlegend=showlegend,
            ),
            **kwargs,
        )


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_global(output_name: str):
    fig = go.Figure()
    add_ssp_traces(fig)
    fig.add_vline(
        x=LAST_HISTORICAL_YEAR,
        line_color="black",
        line_dash="dash",
        line_width=1,
        opacity=0.7,
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=YLABEL,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"size": 11}},
        margin={"l": 90, "r": 30},
    )
    fig.update_xaxes(title_font_size=15)
    fig.update_yaxes(title_font_size=15)
    save(fig, output_name, width=1050, height=620)


def plot_regional(output_name: str, region_items, display_names, ncols: int, include_world=False):
    plot_regions = [None, *region_items] if include_world else list(region_items)
    nrows = math.ceil(len(plot_regions) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            "World" if region is None else display_names.get(str(region), str(region))
            for region in plot_regions
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for index, region in enumerate(plot_regions):
        row = index // ncols + 1
        col = index % ncols + 1
        add_ssp_traces(fig, region=region, showlegend=(index == 0), row=row, col=col)
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

    if include_world:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            xref="x domain",
            yref="y domain",
            line={"color": "black", "width": 2},
            fillcolor="rgba(0,0,0,0)",
            row=1,
            col=1,
        )

    for annotation in fig.layout.annotations:
        annotation.font = {"size": 13}

    fig.update_layout(
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
        margin={"t": 70, "l": 95, "b": 60, "r": 150},
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


def plot_combined(output_name: str, region_items, display_names, ncols_regional: int):
    """Global panel full-height on the left and a regional grid, with x-axis from 2000."""
    nrows = math.ceil(len(region_items) / ncols_regional)
    ncols_total = ncols_regional + 1

    specs = [[{"rowspan": nrows}] + [{}] * ncols_regional]
    for _ in range(nrows - 1):
        specs.append([None] + [{}] * ncols_regional)

    subplot_titles = ["Global"] + [display_names.get(str(r), str(r)) for r in region_items]

    fig = make_subplots(
        rows=nrows,
        cols=ncols_total,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
        column_widths=[1.5] + [1] * ncols_regional,
    )

    add_ssp_traces(fig, region=None, showlegend=True, row=1, col=1)
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

    for index, region in enumerate(region_items):
        row = index // ncols_regional + 1
        col = index % ncols_regional + 2
        add_ssp_traces(fig, region=region, showlegend=False, row=row, col=col)
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

    fig.update_xaxes(range=[2000, max(time)])

    fig.update_layout(
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
        margin={"t": 70, "l": 95, "b": 60, "r": 120},
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


plot_global("fig3_production_by_ssp_global")
plot_regional("fig3_production_by_ssp_h12", regions, REGION_DISPLAY_NAMES, ncols=4)
plot_regional("fig3_production_by_ssp_agg", AGG_REGION_ORDER, {}, ncols=3, include_world=True)
plot_combined("fig3_production_by_ssp_combined_h12", regions, REGION_DISPLAY_NAMES, ncols_regional=4)
plot_combined("fig3_production_by_ssp_combined_agg", AGG_REGION_ORDER, {}, ncols_regional=3)

print("END")
