"""Figure 4: reconciliation of the in-use cement stock in the last historic year.

Three stacked rows show the residential, commercial and combined (residential + commercial
stacked) in-use stock. Each row has two columns: a wide panel with the 12 regions and a narrow
"World" panel with the global aggregate (they share the y-axis per row, so the global bars are
directly comparable to the regional ones). In every panel each group holds three bars: the
pre-reconciliation bottom-up estimate (left), the reconciled stock (middle) and the
pre-reconciliation top-down estimate (right). Color encodes the stock type (residential /
commercial) and is reused for the two parts of the stacked combined bar; a fill pattern encodes the
estimate. All bars show cement contained in concrete, at the last historic year. Two versions are
produced: absolute (t/capita) and relative (bottom-up = 100%).

Data sources (all reduced to cement-in-concrete by region and stock type at the last historic year):
- bottom-up: pure bottom-up in-use concrete stock incl. hibernating stock
  (`model.bu_stock`, a stock array with dims t,r,s,f,b), converted to cement via `cement_ratio`
- reconciled: reconciled combined MFA in-use stock
- top-down: pre-reconciliation top-down MFA in-use stock

Run from the repository root:
    uv run python scritps_figures/04_reconciliation_stock.py
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    SOURCE_PICKLE,
    CACHE_DIR_CEMENT,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    STOCK_TYPE_BASE_COLORS,
)
from helpers import load_mfas

H = LAST_HISTORICAL_YEAR
CEMENT_CONCRETE = {"k": "cement", "m": "concrete"}
WORLD_LABEL = "World"

STOCK_TYPE_NAMES = {"Res": "Residential", "Com": "Commercial"}

ESTIMATES = ["BU", "reconciled", "TD"]
ESTIMATE_NAMES = {"BU": "Bottom-up", "reconciled": "Reconciled", "TD": "Top-down"}
ESTIMATE_PATTERNS = {"BU": "/", "reconciled": "", "TD": "."}

# One row per group. "combined" stacks Res (bottom) + Com (top).
ROWS = ["Res", "Com", "combined"]
ROW_TITLES = {
    "Res": "Residential",
    "Com": "Commercial",
    "combined": "Combined (residential + commercial)",
}

PATTERN_KWARGS = {"fgcolor": "white", "size": 7, "solidity": 0.4}
LEGEND_GREY = "#9a9a9a"

mfas = load_mfas(SOURCE_PICKLE, CACHE_DIR_CEMENT)
td, recon, bu = mfas["td"], mfas["combined"], mfas["bu"]


def by_stocktype(arr):
    """Reduce a stock array to (r, s), summing the building dimensions (f, b) if present."""
    for extra in ("f", "b"):
        if extra in arr.dims.letters:
            arr = arr.sum_over(extra)
    return arr


# Pure bottom-up concrete stock (incl. hibernating), converted to cement-in-concrete.
# `bu` is the bu_in_use stock array (dims t,r,s,f,b); cement_ratio lives on the top-down MFA.
bu_concrete = bu[{"t": H}]
bu_cement_ratio = td.parameters["cement_ratio"][{"m": "concrete"}]

# Cement-in-concrete stock by region and stock type at the last historic year, per estimate.
STOCK = {
    "BU": by_stocktype(bu_concrete) * bu_cement_ratio,
    "reconciled": by_stocktype(recon.stocks["in_use"].stock[{"t": H, **CEMENT_CONCRETE}]),
    "TD": by_stocktype(td.stocks["in_use"].stock[{"t": H, **CEMENT_CONCRETE}]),
}
POP = td.parameters["population"]
regions = list(td.stocks["in_use"].stock.dims["r"].items)
x_labels = [REGION_DISPLAY_NAMES.get(str(r), str(r)) for r in regions]


# --- per-column value accessors: regional (per region) and global (summed over regions) ---
def regional_stock(estimate: str, key, stocktype: str) -> float:
    return STOCK[estimate][{"r": key, "s": stocktype}].values.item()


def regional_pop(key) -> float:
    return POP[{"t": H, "r": key}].values.item()


def global_stock(estimate: str, key, stocktype: str) -> float:
    return STOCK[estimate][{"s": stocktype}].sum_over("r").values.item()


def global_pop(key) -> float:
    return POP[{"t": H}].sum_over("r").values.item()


COLUMNS = [
    {"col": 1, "keys": regions, "labels": x_labels, "stock": regional_stock, "pop": regional_pop},
    {"col": 2, "keys": [WORLD_LABEL], "labels": [WORLD_LABEL], "stock": global_stock, "pop": global_pop},
]


def divisor(colspec: dict, row: str, key, relative: bool) -> float:
    """Normalization: bottom-up total of the group (relative) or population (absolute)."""
    if not relative:
        return colspec["pop"](key)
    if row == "combined":
        return colspec["stock"]("BU", key, "Res") + colspec["stock"]("BU", key, "Com")
    return colspec["stock"]("BU", key, row)


def bar_marker(stocktype: str, estimate: str) -> dict:
    return {
        "color": STOCK_TYPE_BASE_COLORS[stocktype],
        "pattern": {"shape": ESTIMATE_PATTERNS[estimate], **PATTERN_KWARGS},
        "line": {"width": 0.4, "color": "rgba(0,0,0,0.35)"},
    }


def add_data_traces(fig, relative: bool):
    """Add the grouped (and, for the combined row, stacked) bars, per column and per row."""
    for colspec in COLUMNS:
        col = colspec["col"]
        for row_idx, row in enumerate(ROWS, start=1):
            for estimate in ESTIMATES:
                offsetgroup = f"c{col}_{row}_{estimate}"
                # combined stacks Res (added first -> bottom) then Com (top)
                parts = ["Res", "Com"] if row == "combined" else [row]
                for stocktype in parts:
                    ys = []
                    for key in colspec["keys"]:
                        div = divisor(colspec, row, key, relative)
                        value = colspec["stock"](estimate, key, stocktype)
                        ys.append(value / div if div else 0.0)
                    fig.add_trace(
                        go.Bar(
                            x=colspec["labels"],
                            y=ys,
                            offsetgroup=offsetgroup,
                            alignmentgroup=f"c{col}_{row}",
                            marker=bar_marker(stocktype, estimate),
                            showlegend=False,
                        ),
                        row=row_idx,
                        col=col,
                    )


def add_legend_proxies(fig):
    """Two legend blocks via invisible (y=None) proxy bars: stock type (color) + estimate (pattern)."""
    for stocktype, name in STOCK_TYPE_NAMES.items():
        fig.add_trace(
            go.Bar(
                x=[x_labels[0]],
                y=[None],
                name=name,
                marker={"color": STOCK_TYPE_BASE_COLORS[stocktype]},
                offsetgroup="c1_Res_BU",
                alignmentgroup="c1_Res",
                legendgroup="stocktype",
                legendgrouptitle_text="Stock type",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    for estimate in ESTIMATES:
        fig.add_trace(
            go.Bar(
                x=[x_labels[0]],
                y=[None],
                name=ESTIMATE_NAMES[estimate],
                marker={
                    "color": LEGEND_GREY,
                    "pattern": {"shape": ESTIMATE_PATTERNS[estimate], **PATTERN_KWARGS},
                },
                offsetgroup="c1_Res_BU",
                alignmentgroup="c1_Res",
                legendgroup="estimate",
                legendgrouptitle_text="Estimate",
                showlegend=True,
            ),
            row=1,
            col=1,
        )


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot(relative: bool, output_name: str):
    fig = make_subplots(
        rows=len(ROWS),
        cols=2,
        shared_xaxes=True,   # share x down each column (region labels only on bottom row)
        shared_yaxes=True,   # share y across each row (global comparable to regional)
        column_widths=[0.86, 0.14],
        horizontal_spacing=0.03,
        vertical_spacing=0.055,
        subplot_titles=[
            ROW_TITLES["Res"], WORLD_LABEL,
            ROW_TITLES["Com"], "",
            ROW_TITLES["combined"], "",
        ],
    )
    add_data_traces(fig, relative)
    add_legend_proxies(fig)

    ylabel = "Cement stock (bottom-up = 100%)" if relative else "Cement stock (t/capita)"

    for annotation in fig.layout.annotations:  # subplot (row/column) titles
        annotation.font = {"size": 14}

    fig.update_layout(
        barmode="relative",
        bargap=0.25,
        bargroupgap=0.0,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": f"In-use cement stock in concrete buildings ({H})",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "font": {"size": 12},
            "tracegroupgap": 12,
            "groupclick": "toggleitem",
        },
        margin={"t": 80, "l": 100, "b": 120, "r": 230},
    )
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(tickangle=-30, tickfont={"size": 11})
    if relative:
        fig.update_yaxes(tickformat=".0%")
        for row_idx in range(1, len(ROWS) + 1):
            for col_idx in (1, 2):
                fig.add_hline(
                    y=1.0,
                    line_color="black",
                    line_dash="dash",
                    line_width=1,
                    opacity=0.6,
                    row=row_idx,
                    col=col_idx,
                )

    # single shared y-axis label centered on the left
    fig.add_annotation(
        text=ylabel,
        x=-0.07,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        font={"size": 15},
    )

    save(fig, output_name, width=1650, height=950)


plot(relative=False, output_name="fig4_reconciliation_stock_absolute")
plot(relative=True, output_name="fig4_reconciliation_stock_relative")

print("END")
