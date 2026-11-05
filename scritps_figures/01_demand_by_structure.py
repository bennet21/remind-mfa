"""Figure 1: combined-MFA cement demand stacked by building structure and function.

Stacked areas show the reconciled (combined) cement demand split over the building-structure
dimension (`s`: Concrete / Masonry / Timber / Steel), with each structure further subdivided over
the end-use dimension (`e`: single-family res. / multi-family res. / commercial) via
shading. Grey bands on top hold the cement not split by structure (mortar, industrial, civil
engineering); they are labelled directly in the world panel. Instead of a plotly legend, a
structure x use colour key for the building bands is drawn below the plot area. Produces a global figure and a 12-panel regional figure, saved as PNGs in `data/cement/output/figures`, once per scenario in
`SCENARIOS` (output filenames are suffixed with the scenario name).

Run from the repository root:
    uv run python scritps_figures/01_demand_by_structure.py
"""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    BACKGROUND,
    SCENARIOS as SCENARIO_KEYS,
    FIGURES_DIR,
    FIRST_PLOT_YEAR,
    REGION_DISPLAY_NAMES,
    AGG_REGIONS,
    AGG_REGION_ORDER,
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
from helpers import add_last_historical_line, load_figure_data, shade, shade_levels, to_h12

MT_PER_T = 1e-6
YLABEL = "Cement demand (Mt)"

# Short labels for the colour key; its axis titles ("Structure", "Use") provide the context.
KEY_STRUCTURE_LABELS = {"C": "Concrete", "M": "Masonry", "T": "Timber", "S": "Steel"}
KEY_FUNCTION_LABELS = {"RS": "Single-family", "RM": "Multi-family", "Com": "Commercial"}
# Year at which the grey bands are labelled in the world panel.
BAND_LABEL_YEAR = 2075

# Which scenarios to plot: "all", or a list of keys from constants.SCENARIOS,
# e.g. ["SSP2"].
SCENARIOS = "all"

_ALL_SCENARIOS = list(SCENARIO_KEYS)


def selected_scenarios() -> list[str]:
    if SCENARIOS == "all":
        return _ALL_SCENARIOS
    missing = set(SCENARIOS) - set(_ALL_SCENARIOS)
    if missing:
        raise ValueError(f"Unknown scenario(s): {sorted(missing)}")
    return list(SCENARIOS)


def build_series(buildings, functions) -> list[dict]:
    """Ordered stack series (bottom -> top): structure x function, then non-building 'Other' on top."""
    series = []

    levels = shade_levels(len(functions))
    for b in buildings:
        base = STRUCTURE_BASE_COLORS.get(str(b), COLOR_PALETTE[0])
        for f, t in zip(functions, levels):
            series.append(
                {
                    "selections": [{"s": b, "e": f}],
                    "color": shade(base, t),
                    "name": FUNCTION_DISPLAY_NAMES.get(str(f), str(f)),
                    "structure": str(b),
                    "function": str(f),
                }
            )

    # Three grey sub-categories for non-building cement, stacked on top.
    # Mortar first, so it sits directly above the residential/commercial building bands.
    series.append({"selections": [{"m": "mortar", "e": "RS"}, {"m": "mortar", "e": "RM"},
                                  {"m": "mortar", "e": "Com"}],
                   "color": OTHER_RES_COM_MORTAR_COLOR,
                   "name": OTHER_RES_COM_MORTAR_NAME, "structure": None, "function": None})
    series.append({"selections": [{"e": "Ind"}], "color": OTHER_IND_COLOR,
                   "name": OTHER_IND_NAME, "structure": None, "function": "Ind"})
    series.append({"selections": [{"e": "Civ"}], "color": OTHER_CIV_COLOR,
                   "name": OTHER_CIV_NAME, "structure": None, "function": "Civ"})

    return series


def demand(combined, selections, region=None):
    """Combined cement demand (Mt) summed to time, for the given selection(s).

    `combined` is the reconciled in-use inflow of new material, aggregated to H12."""
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
            arr = combined[filter_dict].sum_to("t") * MT_PER_T
            result = arr if result is None else result + arr
    return result


class ColourKey:
    """Structure x use colour grid of the building bands, drawn below the plot area.

    Rows mirror the stacking order (top of the stack first). The layout is defined in pixels and
    converted to paper coordinates, so the key keeps its size across figure sizes.
    """

    CELL_WIDTH = 125
    CELL_HEIGHT = 22
    GAP = 3
    LINE_HEIGHT = 24
    ROW_LABEL_WIDTH = 82
    AXIS_TITLE_WIDTH = 28
    TOP_OFFSET = 70  # below the plot area, leaving room for the x-axis title

    def __init__(self, series: list[dict], font_size: int = 15):
        self.font_size = font_size
        buildings = [s for s in series if s["structure"] is not None]
        self.colors = {(s["structure"], s["function"]): s["color"] for s in buildings}
        self.structures = list(dict.fromkeys(s["structure"] for s in buildings))[::-1]
        self.functions = list(dict.fromkeys(s["function"] for s in buildings))

    def height(self) -> float:
        """Vertical extent below the plot area in pixels."""
        return (self.TOP_OFFSET + 2 * self.LINE_HEIGHT
                + len(self.structures) * (self.CELL_HEIGHT + self.GAP))

    def _width(self) -> float:
        n_cols = len(self.functions)
        return (self.AXIS_TITLE_WIDTH + self.ROW_LABEL_WIDTH + n_cols * self.CELL_WIDTH
                + (n_cols - 1) * self.GAP)

    def _col_left(self, col: int) -> float:
        return self.grid_left + col * (self.CELL_WIDTH + self.GAP)

    def _row_top(self, row: int) -> float:
        return self.grid_top + row * (self.CELL_HEIGHT + self.GAP)

    def _x(self, pixels: float) -> float:
        return 0.5 + pixels / self.plot_width

    def _y(self, pixels: float) -> float:
        return -pixels / self.plot_height

    def _text(self, fig: go.Figure, text: str, x: float, y: float, xanchor: str, yanchor: str,
              bold: bool = False, angle: float = 0):
        fig.add_annotation(
            text=f"<b>{text}</b>" if bold else text,
            xref="paper", yref="paper",
            x=self._x(x), y=self._y(y),
            xanchor=xanchor, yanchor=yanchor,
            textangle=angle,
            showarrow=False,
            font={"size": self.font_size, "color": "black"},
        )

    def add_to(self, fig: go.Figure, plot_width: int, plot_height: int):
        """Draw the key, horizontally centred below a plot area of the given size (px)."""
        self.plot_width = plot_width
        self.plot_height = plot_height
        left = -self._width() / 2
        self.grid_left = left + self.AXIS_TITLE_WIDTH + self.ROW_LABEL_WIDTH
        self.grid_top = self.TOP_OFFSET + 2 * self.LINE_HEIGHT
        grid_right = self._col_left(len(self.functions) - 1) + self.CELL_WIDTH
        grid_bottom = self._row_top(len(self.structures) - 1) + self.CELL_HEIGHT

        self._text(fig, "Use", (self.grid_left + grid_right) / 2, self.TOP_OFFSET,
                   "center", "top", bold=True)
        for col, function in enumerate(self.functions):
            self._text(fig, KEY_FUNCTION_LABELS.get(function, function),
                       self._col_left(col) + self.CELL_WIDTH / 2, self.grid_top - 2,
                       "center", "bottom")
        self._text(fig, "Structure", left + self.AXIS_TITLE_WIDTH / 2,
                   (self.grid_top + grid_bottom) / 2, "center", "middle", bold=True, angle=-90)

        for row, structure in enumerate(self.structures):
            self._text(fig, KEY_STRUCTURE_LABELS.get(structure, structure), self.grid_left - 8,
                       self._row_top(row) + self.CELL_HEIGHT / 2, "right", "middle")
            for col, function in enumerate(self.functions):
                fig.add_shape(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=self._x(self._col_left(col)),
                    x1=self._x(self._col_left(col) + self.CELL_WIDTH),
                    y0=self._y(self._row_top(row) + self.CELL_HEIGHT),
                    y1=self._y(self._row_top(row)),
                    fillcolor=self.colors[(structure, function)],
                    line={"width": 0},
                )


def label_unsplit_bands(fig: go.Figure, time, series: list[dict], stack_values: list,
                        row: int | None = None, col: int | None = None):
    """Name the bands not split by structure inside the stack at `BAND_LABEL_YEAR`.

    `stack_values` holds the plotted values of each series, in stacking order.
    """
    kwargs = {"row": row, "col": col} if row is not None else {}
    year_index = list(time).index(BAND_LABEL_YEAR)
    band_bottom = 0.0
    for s, values in zip(series, stack_values):
        band_top = band_bottom + values[year_index]
        if s["structure"] is None:
            fig.add_annotation(
                text=s["name"], x=BAND_LABEL_YEAR, y=(band_bottom + band_top) / 2,
                xanchor="center", yanchor="middle", showarrow=False,
                font={"size": 13, "color": "black"}, **kwargs,
            )
        band_bottom = band_top


def add_brace(fig: go.Figure, row: int, col: int, y_bottom: float, y_top: float, label: str):
    """Curly brace right of a subplot spanning `y_bottom`..`y_top` (data units), with a label.

    x is given in domain units of the subplot (1 = right edge), y in data units.
    """
    left, width = 1.015, 0.03
    middle_x = left + width / 2
    tip_x = left + width
    y_middle = (y_bottom + y_top) / 2
    corner = 0.12 * (y_top - y_bottom) / 2
    path = (
        f"M {left},{y_bottom} Q {middle_x},{y_bottom} {middle_x},{y_bottom + corner} "
        f"L {middle_x},{y_middle - corner} Q {middle_x},{y_middle} {tip_x},{y_middle} "
        f"Q {middle_x},{y_middle} {middle_x},{y_middle + corner} "
        f"L {middle_x},{y_top - corner} Q {middle_x},{y_top} {left},{y_top}"
    )
    fig.add_shape(
        type="path", path=path, xref="x domain", yref="y",
        line={"color": "black", "width": 1.2}, row=row, col=col,
    )
    fig.add_annotation(
        text=label, x=tip_x + 0.01, y=y_middle, xref="x domain", yref="y",
        xanchor="left", yanchor="middle", showarrow=False,
        font={"size": 14, "color": "black"}, row=row, col=col,
    )


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_global(combined, time, series, output_name: str):
    fig = go.Figure()

    stack_values = [demand(combined, s["selections"]).values for s in series]
    for s, values in zip(series, stack_values):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                mode="lines",
                stackgroup="struct",
                line={"color": s["color"], "width": 0.3},
                fillcolor=s["color"],
                name=s["name"],
            )
        )

    add_last_historical_line(fig)
    label_unsplit_bands(fig, time, series, stack_values)
    key = ColourKey(series)
    plot_width, plot_height = 930, 500
    margin = {"t": 50, "l": 90, "b": round(key.height()) + 20, "r": 30}
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=YLABEL,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=margin,
    )
    fig.update_xaxes(range=[FIRST_PLOT_YEAR, max(time)], title_font_size=15)
    fig.update_yaxes(title_font_size=15)
    key.add_to(fig, plot_width, plot_height)

    save(fig, output_name, width=plot_width + margin["l"] + margin["r"],
         height=plot_height + margin["t"] + margin["b"])


def plot_regional(
    combined, time, series, output_name: str, region_items, display_names, ncols: int,
    include_world=False,
):
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
    # only the subplot titles exist at this point
    fig.update_annotations(font_size=18)

    for index, region in enumerate(plot_regions):
        row = index // ncols + 1
        col = index % ncols + 1

        stack_values = [demand(combined, s["selections"], region=region).values for s in series]
        bottom_up_top = 0.0
        total_top = 0.0
        for s, values in zip(series, stack_values):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=values,
                    mode="lines",
                    stackgroup=f"struct{index}",  # isolate stacking per panel
                    line={"color": s["color"], "width": 0.3},
                    fillcolor=s["color"],
                    name=s["name"],
                ),
                row=row,
                col=col,
            )
            total_top += values[-1]
            if s["structure"] is not None:
                bottom_up_top += values[-1]

        add_last_historical_line(fig, row=row, col=col)
        fig.update_yaxes(showgrid=True, row=row, col=col)
        if region is None:
            label_unsplit_bands(fig, time, series, stack_values, row=row, col=col)

        # braces at the last year in the rightmost panels, where the right margin leaves space
        if col == ncols:
            gap = 0.01 * total_top
            add_brace(fig, row, col, 0, bottom_up_top - gap, "Bottom-up")
            add_brace(fig, row, col, bottom_up_top + gap, total_top, "Top-down")

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

    fig.update_xaxes(range=[FIRST_PLOT_YEAR, max(time)])

    key = ColourKey(series)
    plot_width, plot_height = 1475, 670
    margin = {"t": 70, "l": 95, "b": round(key.height()) + 20, "r": 130}
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=margin,
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
                x=0, y=0.5,
                xshift=-80,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="middle",
                textangle=-90,
                font={"size": 16},
            ),
        ],
    )
    key.add_to(fig, plot_width, plot_height)

    save(fig, output_name, width=plot_width + margin["l"] + margin["r"],
         height=plot_height + margin["t"] + margin["b"])


def run(ssp: str):
    print(f"--- Scenario {ssp} ---")
    data = load_figure_data(ssp)
    combined = to_h12(data.reconciled.inflow)

    time = combined.dims["t"].items
    regions = combined.dims["r"].items
    all_structures = combined.dims["s"].items
    all_functions = combined.dims["e"].items

    buildings = [b for b in all_structures if str(b) not in OTHER_STRUCTURE_KEYS]
    functions = [f for f in all_functions if str(f) in FUNCTION_DISPLAY_NAMES]

    series = build_series(buildings, functions)

    plot_global(combined, time, series, f"fig1_demand_by_structure_global_{ssp}")
    plot_regional(
        combined, time, series, f"fig1_demand_by_structure_h12_{ssp}",
        regions, REGION_DISPLAY_NAMES, ncols=4,
    )
    plot_regional(
        combined, time, series, f"fig1_demand_by_structure_agg_{ssp}",
        AGG_REGION_ORDER, {}, ncols=3, include_world=True,
    )


for _ssp in selected_scenarios():
    run(_ssp)

print("END")
