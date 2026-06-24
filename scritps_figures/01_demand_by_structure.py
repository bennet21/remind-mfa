"""Figure 1: combined-MFA cement demand stacked by building structure and function.

Stacked areas show the reconciled (combined) cement demand split over the building-structure
dimension (`b`: Concrete / Masonry / Timber / Steel), with each structure further subdivided over
the building-function dimension (`f`: single-family res. / multi-family res. / commercial) via
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
    STRUCTURE_DISPLAY_NAMES,
    FUNCTION_DISPLAY_NAMES,
    STRUCTURE_BASE_COLORS,
    OTHER_STRUCTURE_KEYS,
    OTHER_STRUCTURE_COLOR,
    COLOR_PALETTE,
)
from helpers import load_mfas

MT_PER_T = 1e-6
TD_LINE_COLOR = "#222222"
TD_LINE_NAME = "Top-down demand (pre-reconciliation)"
YLABEL = "Cement demand (Mt)"

# Shading range for the function split within each structure hue (darkest -> lightest).
SHADE_MIN, SHADE_MAX = -0.30, 0.45

mfas = load_mfas(SOURCE_PICKLE, CACHE_DIR_CEMENT)
combined = mfas["combined"]
td = mfas["td"]

time = combined.stocks["in_use"].stock.dims["t"].items
regions = combined.stocks["in_use"].inflow.dims["r"].items
_all_structures = combined.stocks["in_use"].inflow.dims["b"].items
_all_functions = combined.stocks["in_use"].inflow.dims["f"].items

_buildings = [b for b in _all_structures if str(b) not in OTHER_STRUCTURE_KEYS]
_other = [b for b in _all_structures if str(b) in OTHER_STRUCTURE_KEYS]
_functions = [f for f in _all_functions if str(f) in FUNCTION_DISPLAY_NAMES]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def shade(base_hex: str, t: float) -> str:
    """Blend a base colour toward black (t<0) or white (t>0); return an rgb() string."""
    rgb = _hex_to_rgb(base_hex)
    target = (255, 255, 255) if t >= 0 else (0, 0, 0)
    amount = abs(t)
    out = tuple(round(c + (tc - c) * amount) for c, tc in zip(rgb, target))
    return f"rgb({out[0]},{out[1]},{out[2]})"


def _shade_levels(n: int) -> list[float]:
    if n <= 1:
        return [0.1]
    return [SHADE_MIN + (SHADE_MAX - SHADE_MIN) * i / (n - 1) for i in range(n)]


def build_series() -> list[dict]:
    """Ordered stack series (bottom -> top): structure x function, then non-building 'Other' on top."""
    series = []

    levels = _shade_levels(len(_functions))
    for b in _buildings:
        base = STRUCTURE_BASE_COLORS.get(str(b), COLOR_PALETTE[0])
        structure_name = STRUCTURE_DISPLAY_NAMES.get(str(b), str(b))
        for f, t in zip(_functions, levels):
            series.append(
                {
                    "selection": {"b": b, "f": f},
                    "color": shade(base, t),
                    "name": FUNCTION_DISPLAY_NAMES.get(str(f), str(f)),
                    "group": str(b),
                    "grouptitle": structure_name,
                }
            )

    for b in _other:
        series.append(
            {
                "selection": {"b": b},
                "color": OTHER_STRUCTURE_COLOR,
                "name": STRUCTURE_DISPLAY_NAMES.get(str(b), str(b)),
                "group": "other",
                "grouptitle": None,
            }
        )

    return series


SERIES = build_series()


def demand(selection: dict):
    """Combined cement demand (Mt) summed to time, for the given selection."""
    return combined.stocks["in_use"].inflow[{"k": "cement", **selection}].sum_to("t") * MT_PER_T


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
                y=demand(s["selection"]).values,
                mode="lines",
                stackgroup="struct",
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
                    y=demand({**s["selection"], "r": region}).values,
                    mode="lines",
                    stackgroup=f"struct{index}",  # isolate stacking per panel
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


plot_global("fig1_demand_by_structure_global")
plot_regional("fig1_demand_by_structure_regional")

print("END")
