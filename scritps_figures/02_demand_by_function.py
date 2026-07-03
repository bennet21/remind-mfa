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
    OTHER_STRUCTURE_KEYS,
)
from helpers import load_mfas, shade, shade_levels

MT_PER_T = 1e-6
TD_LINE_COLOR = "#222222"
TD_LINE_NAME = "Top-down demand (pre-reconciliation)"
YLABEL = "Cement demand (Mt)"
LABEL_YEAR = 2080
MIN_FRACTION_GLOBAL = 0.04
MIN_FRACTION_REGIONAL = 0.10
STRUCTURE_SHADE_SPREAD = 0.10

mfas = load_mfas(SOURCE_PICKLE, CACHE_DIR_CEMENT)
combined = mfas["combined"]
td = mfas["td"]

time = combined.stocks["in_use"].stock.dims["t"].items
regions = combined.stocks["in_use"].inflow.dims["r"].items
_all_structures = combined.stocks["in_use"].inflow.dims["b"].items
_buildings = [b for b in _all_structures if str(b) not in OTHER_STRUCTURE_KEYS]

RES_COLOR = STOCK_TYPE_BASE_COLORS["Res"]
COM_COLOR = STOCK_TYPE_BASE_COLORS["Com"]

_res_shade_levels = shade_levels(2)

SERIES = [
    {
        "selections": [{"s": "Res", "m": "concrete", "f": "RS"}],
        "color": shade(RES_COLOR, _res_shade_levels[0]),
        "name": "Single-family res. buildings",
        "group": "res",
        "grouptitle": "Residential",
        "base_hex": RES_COLOR,
        "shade_center": _res_shade_levels[0],
    },
    {
        "selections": [{"s": "Res", "m": "concrete", "f": "RM"}],
        "color": shade(RES_COLOR, _res_shade_levels[1]),
        "name": "Multi-family res. buildings",
        "group": "res",
        "grouptitle": None,
        "base_hex": RES_COLOR,
        "shade_center": _res_shade_levels[1],
    },
    {
        "selections": [{"s": "Com", "m": "concrete"}],
        "color": COM_COLOR,
        "name": "Commercial",
        "group": "com",
        "grouptitle": None,
        "base_hex": COM_COLOR,
        "shade_center": 0.0,
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

_INLINE_LABELS = {
    "Single-family res. buildings": "Single-family<br>res. buildings",
    "Multi-family res. buildings": "Multi-family<br>res. buildings",
    "Res./com. mortar": "Res./com.<br>mortar",
}


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


def _text_color(color: str) -> str:
    if color.startswith("#"):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    else:
        parts = color.strip("rgb()").split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 160 else "#333333"


def _structure_shade_levels(center: float, n: int) -> list:
    if n == 1:
        return [center]
    return [center - STRUCTURE_SHADE_SPREAD + 2 * STRUCTURE_SHADE_SPREAD * i / (n - 1) for i in range(n)]


def _band_midpoints(region=None) -> list:
    time_list = list(time)
    t_idx = min(range(len(time_list)), key=lambda i: abs(time_list[i] - LABEL_YEAR))
    vals = [float(demand(s["selections"], region=region).values[t_idx]) for s in SERIES]
    total = sum(vals)
    result, cumulative = [], 0.0
    for s, v in zip(SERIES, vals):
        result.append({"series": s, "val": v, "total": total, "y_mid": cumulative + v / 2})
        cumulative += v
    return result


def _label_annotations(
    region=None, xref="x", yref="y", font_size=11, min_fraction=MIN_FRACTION_GLOBAL
) -> list:
    out = []
    for info in _band_midpoints(region=region):
        if info["total"] > 0 and info["val"] / info["total"] >= min_fraction:
            name = info["series"]["name"]
            out.append(dict(
                x=LABEL_YEAR, y=info["y_mid"],
                xref=xref, yref=yref,
                text=_INLINE_LABELS.get(name, name),
                showarrow=False,
                font=dict(size=font_size, color=_text_color(info["series"]["color"])),
                xanchor="center", yanchor="middle", align="center",
            ))
    return out


def _add_series_traces(fig, stackgroup, region=None, row=None, col=None):
    """Add SERIES traces; building categories expand into per-structure sub-bands."""
    add_kwargs = {"row": row, "col": col} if row is not None else {}

    for s in SERIES:
        if "base_hex" in s:
            shade_ts = _structure_shade_levels(s["shade_center"], n=len(_buildings))
            for b, shade_t in zip(_buildings, shade_ts):
                sub_sel = [{**sel, "b": b} for sel in s["selections"]]
                sub_color = shade(s["base_hex"], shade_t)
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=demand(sub_sel, region=region).values,
                        mode="lines",
                        stackgroup=stackgroup,
                        line={"color": sub_color, "width": 0.3},
                        fillcolor=sub_color,
                        showlegend=False,
                    ),
                    **add_kwargs,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=demand(s["selections"], region=region).values,
                    mode="lines",
                    stackgroup=stackgroup,
                    line={"color": s["color"], "width": 0.3},
                    fillcolor=s["color"],
                    showlegend=False,
                ),
                **add_kwargs,
            )


def plot_global(output_name: str):
    fig = go.Figure()

    _add_series_traces(fig, stackgroup="func")

    fig.add_trace(
        go.Scatter(
            x=time,
            y=td_demand({}).values,
            mode="lines",
            line={"color": TD_LINE_COLOR, "width": 2},
            name=TD_LINE_NAME,
            showlegend=False,
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
        margin={"l": 90, "r": 30},
        annotations=_label_annotations(),
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

    for s in SERIES:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker={"color": s["color"], "size": 12, "symbol": "square"},
                name=s["name"],
                showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line={"color": TD_LINE_COLOR, "width": 2},
            name=TD_LINE_NAME,
            showlegend=True,
        )
    )

    for index, region in enumerate(regions):
        row = index // ncols + 1
        col = index % ncols + 1

        _add_series_traces(fig, stackgroup=f"func{index}", region=region, row=row, col=col)

        fig.add_trace(
            go.Scatter(
                x=time,
                y=td_demand({"r": region}).values,
                mode="lines",
                line={"color": TD_LINE_COLOR, "width": 1.5},
                name=TD_LINE_NAME,
                showlegend=False,
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
        margin={"t": 70, "l": 95, "b": 130, "r": 40},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font={"size": 12},
        ),
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="Year",
                x=0.5, y=-0.06,
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

    save(fig, output_name, width=1700, height=900)


plot_global("fig2_demand_by_function_global")
plot_regional("fig2_demand_by_function_regional")

print("END")
