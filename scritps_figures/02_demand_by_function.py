"""Figure 2: combined-MFA cement demand stacked by stock type and building function.

Stacked areas show cement demand split into residential (single-family / multi-family shaded),
commercial concrete, and a three-band "Other" (industrial, civil, res./com. mortar). On top, a
black line shows the pre-reconciliation top-down total cement demand. Produces a global figure and
a 12-panel regional figure, saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/02_demand_by_function.py
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
    STOCK_TYPE_BASE_COLORS,
    OTHER_IND_COLOR,
    OTHER_IND_NAME,
    OTHER_CIV_COLOR,
    OTHER_CIV_NAME,
    OTHER_RES_COM_MORTAR_COLOR,
    OTHER_RES_COM_MORTAR_NAME,
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

RES_COLOR = STOCK_TYPE_BASE_COLORS["Res"]
COM_COLOR = STOCK_TYPE_BASE_COLORS["Com"]

SERIES = [
    {
        "selections": [{"s": "Res", "m": "concrete", "f": "RS"}],
        "color": shade(RES_COLOR, shade_levels(2)[0]),
        "name": "Single-family res. buildings",
        "group": "res",
        "grouptitle": "Residential",
    },
    {
        "selections": [{"s": "Res", "m": "concrete", "f": "RM"}],
        "color": shade(RES_COLOR, shade_levels(2)[1]),
        "name": "Multi-family res. buildings",
        "group": "res",
        "grouptitle": None,
    },
    {
        "selections": [{"s": "Com", "m": "concrete"}],
        "color": COM_COLOR,
        "name": "Commercial",
        "group": "com",
        "grouptitle": None,
    },
    {
        "selections": [{"s": "Ind"}],
        "color": OTHER_IND_COLOR,
        "name": OTHER_IND_NAME,
        "group": "other",
        "grouptitle": "Other cement use",
    },
    {
        "selections": [{"s": "Civ"}],
        "color": OTHER_CIV_COLOR,
        "name": OTHER_CIV_NAME,
        "group": "other",
        "grouptitle": None,
    },
    {
        "selections": [{"s": "Res", "m": "mortar"}, {"s": "Com", "m": "mortar"}],
        "color": OTHER_RES_COM_MORTAR_COLOR,
        "name": OTHER_RES_COM_MORTAR_NAME,
        "group": "other",
        "grouptitle": None,
    },
]


def demand(selections, region=None):
    """Combined cement demand (Mt) summed to time, for the given selection(s)."""
    if isinstance(selections, dict):
        selections = [selections]
    result = None
    for sel in selections:
        filter_dict = {"k": "cement", **sel}
        if region is not None:
            filter_dict["r"] = region
        arr = combined.stocks["in_use"].inflow[filter_dict].sum_to("t") * MT_PER_T
        result = arr if result is None else result + arr
    return result


def td_demand(selection: dict):
    """Pre-reconciliation top-down total cement demand (Mt) summed to time."""
    return td.stocks["in_use"].inflow[{"k": "cement", **selection}].sum_to("t") * MT_PER_T


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
                stackgroup="func",
                line={"color": s["color"], "width": 0.3},
                fillcolor=s["color"],
                name=s["name"],
                legendgroup=s["group"],
                legendgrouptitle_text=s["grouptitle"],
            )
        )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=td_demand({}).values,
            mode="lines",
            line={"color": TD_LINE_COLOR, "width": 2},
            name=TD_LINE_NAME,
            legendgroup="td",
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


def plot_regional(output_name: str):
    ncols = 4
    nrows = math.ceil(len(regions) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[REGION_DISPLAY_NAMES.get(str(r), str(r)) for r in regions],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for index, region in enumerate(regions):
        row = index // ncols + 1
        col = index % ncols + 1

        for s in SERIES:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=demand(s["selections"], region=region).values,
                    mode="lines",
                    stackgroup=f"func{index}",
                    line={"color": s["color"], "width": 0.3},
                    fillcolor=s["color"],
                    name=s["name"],
                    legendgroup=s["group"],
                    legendgrouptitle_text=s["grouptitle"],
                    showlegend=index == 0,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=td_demand({"r": region}).values,
                mode="lines",
                line={"color": TD_LINE_COLOR, "width": 1.5},
                name=TD_LINE_NAME,
                legendgroup="td",
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


plot_global("fig2_demand_by_function_global")
plot_regional("fig2_demand_by_function_regional")

print("END")
