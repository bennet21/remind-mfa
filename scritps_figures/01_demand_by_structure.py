"""Figure 1: combined-MFA cement demand stacked by building structure and function.

Stacked areas show the reconciled (combined) cement demand split over the building-structure
dimension (`s`: Concrete / Masonry / Timber / Steel), with each structure further subdivided over
the end-use dimension (`e`: single-family res. / multi-family res. / commercial) via
shading. A separate grey "Other" band holds non-building (industrial + civil) cement. On top, a
black line shows the pre-reconciliation top-down total cement demand. Produces a global figure and
a 12-panel regional figure, saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/01_demand_by_structure.py
"""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    SOURCE_PICKLE,
    CACHE_DIR_CEMENT,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    AGG_REGIONS,
    AGG_REGION_ORDER,
    STRUCTURE_DISPLAY_NAMES,
    FUNCTION_DISPLAY_NAMES,
    STRUCTURE_BASE_COLORS,
    OTHER_STRUCTURE_KEYS,
    OTHER_IND_COLOR,
    OTHER_IND_NAME,
    OTHER_CIV_COLOR,
    OTHER_CIV_NAME,
    OTHER_RES_COM_MORTAR_COLOR,
    OTHER_RES_COM_MORTAR_NAME,
    COLOR_PALETTE,
)
from helpers import load_mfas, shade, shade_levels

MT_PER_T = 1e-6
TD_LINE_COLOR = "#222222"
TD_LINE_NAME = "Top-down demand (pre-reconciliation)"
YLABEL = "Cement demand (Mt)"

mfas = load_mfas(SOURCE_PICKLE, CACHE_DIR_CEMENT)
combined = mfas["combined"]
td = mfas["td"]

time = combined.stocks["in_use"].stock.dims["t"].items
regions = combined.stocks["in_use"].inflow.dims["r"].items
_all_structures = combined.stocks["in_use"].inflow.dims["s"].items
_all_functions = combined.stocks["in_use"].inflow.dims["e"].items

_buildings = [b for b in _all_structures if str(b) not in OTHER_STRUCTURE_KEYS]
_functions = [f for f in _all_functions if str(f) in FUNCTION_DISPLAY_NAMES]


def build_series() -> list[dict]:
    """Ordered stack series (bottom -> top): structure x function, then non-building 'Other' on top."""
    series = []

    levels = shade_levels(len(_functions))
    structure_legend_ranks = {"S": 10, "T": 20, "M": 30, "C": 40}
    for b in _buildings:
        base = STRUCTURE_BASE_COLORS.get(str(b), COLOR_PALETTE[0])
        structure_name = STRUCTURE_DISPLAY_NAMES.get(str(b), str(b))
        struct_rank = structure_legend_ranks.get(str(b), 100)
        for f, t in zip(_functions, levels):
            series.append(
                {
                    "selections": [{"s": b, "e": f}],
                    "color": shade(base, t),
                    "name": FUNCTION_DISPLAY_NAMES.get(str(f), str(f)),
                    "group": str(b),
                    "grouptitle": structure_name,
                    "legendrank": struct_rank,
                }
            )

    # Three grey sub-categories for non-building cement, stacked on top.
    series.append({"selections": [{"e": "Ind"}], "color": OTHER_IND_COLOR,
                   "name": OTHER_IND_NAME, "group": "other", "grouptitle": "Other cement use",
                   "legendrank": 2})
    series.append({"selections": [{"e": "Civ"}], "color": OTHER_CIV_COLOR,
                   "name": OTHER_CIV_NAME, "group": "other", "grouptitle": None,
                   "legendrank": 2})
    series.append({"selections": [{"m": "mortar", "e": "RS"}, {"m": "mortar", "e": "RM"},
                                  {"m": "mortar", "e": "Com"}],
                   "color": OTHER_RES_COM_MORTAR_COLOR,
                   "name": OTHER_RES_COM_MORTAR_NAME, "group": "other", "grouptitle": None,
                   "legendrank": 2})

    return series


SERIES = build_series()


def demand(selections, region=None):
    """Combined cement demand (Mt) summed to time, for the given selection(s)."""
    if isinstance(selections, dict):
        selections = [selections]
    region_items = [
        source_region
        for source_region, aggregate_region in AGG_REGIONS.items()
        if aggregate_region == region
    ] or [region]
    result = None
    for sel in selections:
        for source_region in region_items:
            filter_dict = {"k": "cement", **sel}
            if source_region is not None:
                filter_dict["r"] = source_region
            arr = combined.stocks["in_use"].inflow[filter_dict].sum_to("t") * MT_PER_T
            result = arr if result is None else result + arr
    return result


def td_demand(selection: dict):
    """Pre-reconciliation top-down total cement demand (Mt) summed to time."""
    region = selection.get("r")
    region_items = [
        source_region
        for source_region, aggregate_region in AGG_REGIONS.items()
        if aggregate_region == region
    ] or [region]
    result = None
    for source_region in region_items:
        filter_dict = {"k": "cement", **selection}
        if source_region is not None:
            filter_dict["r"] = source_region
        else:
            filter_dict.pop("r", None)
        arr = td.stocks["in_use"].inflow[filter_dict].sum_to("t") * MT_PER_T
        result = arr if result is None else result + arr
    return result


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_global(output_name: str):
    fig = go.Figure()

    for s in SERIES:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=demand(s["selections"]).values,
                mode="lines",
                stackgroup="struct",
                line={"color": s["color"], "width": 0.3},
                fillcolor=s["color"],
                name=s["name"],
                legendgroup=s["group"],
                legendgrouptitle_text=s["grouptitle"],
                legendrank=s["legendrank"],
            )
        )

    # Draw the top-down line last so it stays visible on top of the stacked fills. Because the
    # combined demand decomposition sums to the same total as the top-down demand, this line
    # traces the top of the stack.
    fig.add_trace(
        go.Scatter(
            x=time,
            y=td_demand({}).values,
            mode="lines",
            line={"color": TD_LINE_COLOR, "width": 2},
            name=TD_LINE_NAME,
            legendgroup="td",
            legendrank=1,
        )
    )

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
        legend={"font": {"size": 11}, "tracegroupgap": 4, "groupclick": "toggleitem"},
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

        for s in SERIES:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=demand(s["selections"], region=region).values,
                    mode="lines",
                    stackgroup=f"struct{index}",  # isolate stacking per panel
                    line={"color": s["color"], "width": 0.3},
                    fillcolor=s["color"],
                    name=s["name"],
                    legendgroup=s["group"],
                    legendgrouptitle_text=s["grouptitle"],
                    legendrank=s["legendrank"],
                    showlegend=index == 0,
                ),
                row=row,
                col=col,
            )

        # top-down line last so it stays on top of the stacked fills
        fig.add_trace(
            go.Scatter(
                x=time,
                y=td_demand({"r": region}).values,
                mode="lines",
                line={"color": TD_LINE_COLOR, "width": 1.5},
                name=TD_LINE_NAME,
                legendgroup="td",
                legendrank=1,
                showlegend=index == 0,
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
            "tracegroupgap": 6,
        },
        margin={"t": 70, "l": 95, "b": 60, "r": 220},
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


plot_global("fig1_demand_by_structure_global")
plot_regional("fig1_demand_by_structure_h1", regions, REGION_DISPLAY_NAMES, ncols=4)
plot_regional("fig1_demand_by_structure_agg", AGG_REGION_ORDER, {}, ncols=3, include_world=True)

print("END")
