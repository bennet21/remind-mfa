"""Figure 3: cement production under SSP1-5 plus the two circular-economy (CE) variants.

Line plot comparing total (or per-region) cement production from the top-down MFA
for the five SSP scenarios and the Ambitious / Pragmatic Circularity variants of SSP1 / SSP2
(drawn dashed in their parent SSP colour, with a light band between each SSP and its
circularity variant). A vertical dashed line marks the last
historical year. The baseline projections of the validation data are shown as grey markers.
Produces a global figure and a 12-panel regional figure.

Run from the repository root:
    uv run python scritps_figures/03_production_by_ssp.py
"""

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from plotly.subplots import make_subplots

from constants import (
    BACKGROUND,
    FIGURES_DIR,
    FIRST_PLOT_YEAR,
    LAST_HISTORICAL_YEAR,
    REGION_SHORT_NAMES,
    AGG_REGIONS,
    AGG_REGION_ORDER,
    SCENARIOS,
    SSP_COLORS,
    SSP_DASHES,
    SSP_LABELS,
)
from helpers import add_last_historical_line, load_figure_data, to_h12

GT_PER_T = 1e-9
BAND_OPACITY = 0.15
# Figure with the global panel on top: first year of the global panel, year and offset (px) of the
# label of the historical line, and minimum vertical distance of the scenario labels at the right
# end of the lines (share of the y-axis range).
GLOBAL_TOP_FIRST_YEAR = 1880
HISTORICAL_LABEL = (1960, (-40, -40))
LABEL_MIN_GAP = 0.085
# Scenarios labelled within the global panel, next to their line, instead of at its right end:
# label text, year and offset (px) of the label from the line, which connects them by a thin line.
# Without offset (None), the label is placed just above the line.
INLINE_LABELS = {"SSP2": ("SSP2", 2065, None)}
# Inline labels of the labelled aggregated figure (also labelling the lowest, ambitious circularity line).
LABELLED_INLINE_LABELS = {
    "SSP2": ("SSP2", 2070, (0, -45)),
    "SSP1_CE": ("Ambitious Circularity", 2055, (5, 95)),
}
# Font sizes of the figure with the global panel on top.
GLOBAL_TOP_FONT_SIZES = {"label": 19, "subplot_title": 19, "axis_title": 22, "tick": 16}
# Aggregated figure with direct labels: font sizes, year and offset (px) of the label of the historical line, and
# tick spacing of the x-axis (years).
LABELLED_FONT_SIZES = {"label": 22, "subplot_title": 32, "axis_title": 24, "tick": 20, "legend": 28}
LABELLED_HISTORICAL_LABEL = (2000, (-25, -110))
LABELLED_FIRST_YEAR = 1980
LABELLED_XTICKS = [LABELLED_FIRST_YEAR, LAST_HISTORICAL_YEAR, 2100]
# Frame around the World panel of the labelled figure: colour, line width and corner radius (px).
LABELLED_FRAME = {"color": "rgba(0,0,0,0.8)", "width": 2.5, "radius": 14}
# Length (px) of the line samples in the legend, which spans the width of the figure.
LABELLED_LEGEND_ITEM_WIDTH = 70
# The top margin holds the two-row legend (scenarios and validation sources).
LABELLED_MARGIN = {"t": 190, "l": 130, "b": 100, "r": 30}
# Distance (px) of the y-axis title from the plot area.
YLABEL_OFFSET = 85
YLABEL = "Cement production (Gt)"
SSPS = list(SCENARIOS)
# Validation data: baseline scenario, marker symbol and legend label of each source, the years
# marked (if available), and the sources whose projection is also drawn as a line.
VALIDATION_FILE = Path("data_in/validation/validation.mif")
VALIDATION_VARIABLE = "Material Demand|Non-Metallic Minerals|Cement"
VALIDATION_SOURCES = {
    "Arp et al 2026": ("SSP2 Baseline (BL)", "circle-open", "Arp et al. (2026), SSP2 Baseline"),
    "IEA WEO 2025": ("Stated Policies", "diamond-open", "IEA (2025), Stated Policies"),
    "WCA 2024": ("BAU", "square-open", "WCA (2024), Business as usual"),
}
VALIDATION_YEARS = [2035, 2050, 2100]
VALIDATION_LINE_SOURCES = {"Arp et al 2026"}
VALIDATION_COLOR = "rgba(80,80,80,0.7)"
VALIDATION_MARKER_SIZE = 7
GT_PER_MT = 1e-3

# reconciled in-use inflow of new material, aggregated to H12
combined_by_ssp = {ssp: to_h12(load_figure_data(ssp).reconciled.inflow) for ssp in SSPS}
time = combined_by_ssp["SSP2"].dims["t"].items
regions = combined_by_ssp["SSP2"].dims["r"].items


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
        arr = combined[filter_dict].sum_to("t") * GT_PER_T
        result = arr if result is None else result + arr
    return result


def read_validation() -> pd.DataFrame:
    """Baseline cement demand (Gt) of the validation sources, indexed by source and region, with
    one column per year."""
    data = pd.read_csv(VALIDATION_FILE, sep=";", na_values="N/A")
    is_baseline = [
        model in VALIDATION_SOURCES and scenario == VALIDATION_SOURCES[model][0]
        for model, scenario in zip(data["Model"], data["Scenario"])
    ]
    data = data[(data["Variable"] == VALIDATION_VARIABLE) & is_baseline]
    years = [column for column in data.columns if column.isdigit()]
    data = data.set_index(["Model", "Region"])[years] * GT_PER_MT
    data.columns = [int(year) for year in years]
    return data


validation = read_validation()


def validation_values(source: str, region=None) -> pd.Series:
    """Validation values of a source in a region (None: World, aggregated regions: sum of their
    H12 regions), in the years the source covers."""
    region_items = [
        source_region
        for source_region, aggregate_region in AGG_REGIONS.items()
        if aggregate_region == region
    ] or ["World" if region is None else region]
    keys = [(source, region_item) for region_item in region_items]
    if not all(key in validation.index for key in keys):
        return pd.Series(dtype=float)
    return validation.loc[keys].sum(min_count=len(keys)).dropna()


def add_validation_markers(fig, region=None, showlegend=True, kwargs=None):
    """Grey markers of the baseline projections of the validation sources in VALIDATION_YEARS.
    The projections of VALIDATION_LINE_SOURCES are also drawn as a thin line over
    all years they cover. The legend entries are marked by meta="validation legend"."""
    marker = {
        "size": VALIDATION_MARKER_SIZE,
        "color": VALIDATION_COLOR,
        "line": {"width": 1.2, "color": VALIDATION_COLOR},
    }
    line = {"color": VALIDATION_COLOR, "width": 1, "dash": "dot"}
    for source, (_, symbol, label) in VALIDATION_SOURCES.items():
        values = validation_values(source, region=region)
        marked = values[values.index.isin(VALIDATION_YEARS)]
        if values.empty:
            continue
        has_line = source in VALIDATION_LINE_SOURCES
        common = {"name": label, "legendgroup": f"validation {source}", "hoverinfo": "skip"}
        if has_line:
            fig.add_trace(
                go.Scatter(x=list(values.index), y=values.values, mode="lines", line=line,
                           showlegend=False, **common),
                **kwargs,
            )
            # legend entry only, showing line and marker
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="lines+markers", line=line,
                           marker=marker | {"symbol": symbol}, meta="validation legend",
                           showlegend=showlegend, **common),
                **kwargs,
            )
        fig.add_trace(
            go.Scatter(
                x=list(marked.index),
                y=marked.values,
                mode="markers",
                marker=marker | {"symbol": symbol},
                # markers at the last year are not cut off at the edge of the panel
                cliponaxis=False,
                meta=None if has_line else "validation legend",
                showlegend=showlegend and not has_line,
                **common,
            ),
            **kwargs,
        )


def world_y_max() -> float:
    """Upper end of the y-axis of the World panel, covering all scenarios and validation markers."""
    model_max = max(production(combined).values.max() for combined in combined_by_ssp.values())
    validation_max = pd.Series([validation_values(source).max() for source in VALIDATION_SOURCES]).max()
    return 1.05 * max(model_max, validation_max)


def add_ce_bands(fig, region=None, kwargs=None):
    """Light band between each SSP and its circularity variant, in the SSP colour."""
    for ssp in SSPS:
        ce_ssp = f"{ssp}_CE"
        if ce_ssp not in combined_by_ssp:
            continue
        base = production(combined_by_ssp[ssp], region=region).values
        ce = production(combined_by_ssp[ce_ssp], region=region).values
        red, green, blue = hex_to_rgb(SSP_COLORS[ssp])
        fig.add_trace(
            go.Scatter(
                x=[*time, *time[::-1]],
                y=[*base, *ce[::-1]],
                mode="lines",
                fill="toself",
                fillcolor=f"rgba({red},{green},{blue},{BAND_OPACITY})",
                line={"width": 0},
                hoverinfo="skip",
                legendgroup=ce_ssp,
                showlegend=False,
            ),
            **kwargs,
        )


def add_ssp_traces(fig, region=None, showlegend=True, row=None, col=None):
    kwargs = {"row": row, "col": col} if row is not None else {}
    add_ce_bands(fig, region=region, kwargs=kwargs)
    is_historical = [year <= LAST_HISTORICAL_YEAR for year in time]
    is_future = [year >= LAST_HISTORICAL_YEAR for year in time]
    historical_values = production(combined_by_ssp["SSP2"], region=region).values
    fig.add_trace(
        go.Scatter(
            x=[year for year, keep in zip(time, is_historical) if keep],
            y=historical_values[is_historical],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Historical",
            showlegend=showlegend,
        ),
        **kwargs,
    )
    for ssp in SSPS:
        fig.add_trace(
            go.Scatter(
                x=[year for year, keep in zip(time, is_future) if keep],
                y=production(combined_by_ssp[ssp], region=region).values[is_future],
                mode="lines",
                line={"color": SSP_COLORS[ssp], "width": 2, "dash": SSP_DASHES[ssp]},
                name=SSP_LABELS[ssp],
                legendgroup=ssp,
                showlegend=showlegend,
            ),
            **kwargs,
        )
    add_validation_markers(fig, region=region, showlegend=showlegend, kwargs=kwargs)


def spread_labels(targets: list[float], min_gap: float) -> list[float]:
    """Label positions close to `targets` and at least `min_gap` apart."""
    order = sorted(range(len(targets)), key=lambda index: targets[index])
    positions = [targets[index] for index in order]
    for _ in range(200):
        moved = False
        for index in range(len(positions) - 1):
            overlap = min_gap - (positions[index + 1] - positions[index])
            if overlap > 1e-9:
                positions[index] -= overlap / 2
                positions[index + 1] += overlap / 2
                moved = True
        if not moved:
            break
    spread = [0.0] * len(targets)
    for position, index in zip(positions, order):
        spread[index] = position
    return spread


def add_line_labels(
    fig,
    first_year: int,
    y_max: float,
    font_size: int = GLOBAL_TOP_FONT_SIZES["label"],
    min_gap: float = LABEL_MIN_GAP,
    historical_label: tuple[int, tuple[int, int]] = HISTORICAL_LABEL,
    end_labels: bool = True,
    inline_labels: dict[str, tuple[str, int, tuple[int, int] | None]] = INLINE_LABELS,
):
    """Labels instead of a legend for the global panel (axes x, y): the historical line is
    labelled directly, the scenarios at the right end of their lines, connected by thin lines.

    Args:
        fig: Figure with the global panel on the axes x, y.
        first_year: First year of the x-axis of the global panel.
        y_max: Upper end of the y-axis of the global panel.
        font_size: Font size of the labels.
        min_gap: Minimum vertical distance of the labels at the line ends (share of `y_max`).
        historical_label: Year and offset (px) of the label of the historical line.
        end_labels: Whether to label the scenarios not in `inline_labels` at the line ends.
        inline_labels: Scenarios labelled next to their line: label text, year and offset (px).
    """
    year, (offset_x, offset_y) = historical_label
    historical_values = production(combined_by_ssp["SSP2"]).values
    fig.add_annotation(
        x=year,
        y=float(historical_values[list(time).index(year)]),
        ax=offset_x,
        ay=offset_y,
        text="Historical",
        showarrow=True,
        arrowhead=0,
        arrowwidth=0.6,
        arrowcolor="rgba(0,0,0,0.35)",
        standoff=2,
        font={"size": font_size, "color": "black"},
    )
    end_year = max(time)
    for ssp, (text, year, offset) in inline_labels.items():
        label_style = {"showarrow": False, "yanchor": "bottom", "yshift": 6}
        if offset is not None:
            label_style = {
                "showarrow": True,
                "ax": offset[0],
                "ay": offset[1],
                "arrowhead": 0,
                "arrowwidth": 0.6,
                "arrowcolor": "rgba(0,0,0,0.35)",
                "standoff": 2,
            }
        fig.add_annotation(
            x=year,
            y=float(production(combined_by_ssp[ssp]).values[list(time).index(year)]),
            text=text,
            font={"size": font_size, "color": SSP_COLORS[ssp]},
            **label_style,
        )
    if not end_labels:
        return
    end_labelled = [ssp for ssp in SSPS if ssp not in inline_labels]
    targets = [float(production(combined_by_ssp[ssp]).values[-1]) for ssp in end_labelled]
    positions = spread_labels(targets, min_gap * y_max)
    label_x = end_year + 0.03 * (end_year - first_year)
    for ssp, target, position in zip(end_labelled, targets, positions):
        fig.add_annotation(
            x=end_year,
            y=target,
            ax=label_x,
            ay=position,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=SSP_LABELS[ssp],
            xanchor="left",
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.6,
            arrowcolor="rgba(0,0,0,0.35)",
            standoff=2,
            font={"size": font_size, "color": SSP_COLORS[ssp]},
        )


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_global(output_name: str):
    fig = go.Figure()
    add_ssp_traces(fig)
    add_last_historical_line(fig, label=True)
    fig.update_layout(
        yaxis_title=YLABEL,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"size": 11}},
        margin={"l": 90, "r": 30},
    )
    fig.update_xaxes(range=[FIRST_PLOT_YEAR, max(time)], title_font_size=15)
    fig.update_yaxes(title_font_size=15)
    save(fig, output_name, width=1050, height=620)


def plot_regional(
    output_name: str,
    region_items,
    display_names,
    ncols: int,
    include_world=False,
    width: int = 1700,
    height: int = 800,
    vertical_spacing: float = 0.12,
    bottom_margin: int = 60,
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
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.06,
    )

    for index, region in enumerate(plot_regions):
        row = index // ncols + 1
        col = index % ncols + 1
        add_ssp_traces(fig, region=region, showlegend=(index == 0), row=row, col=col)
        add_last_historical_line(fig, label=region is None, row=row, col=col)
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

    fig.update_xaxes(range=[FIRST_PLOT_YEAR, max(time)])

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
        margin={"t": 70, "l": 95, "b": bottom_margin, "r": 150},
        annotations=list(fig.layout.annotations)
        + [
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
    save(fig, output_name, width=width, height=height)


def rounded_rectangle_path(
    x0: float, x1: float, y0: float, y1: float, radius_x: float, radius_y: float
) -> str:
    """SVG path of a rectangle with rounded corners, in paper coordinates.

    Args:
        x0: Left edge.
        x1: Right edge.
        y0: Bottom edge.
        y1: Top edge.
        radius_x: Corner radius along x.
        radius_y: Corner radius along y.
    """
    return (
        f"M {x0 + radius_x},{y0} L {x1 - radius_x},{y0} Q {x1},{y0} {x1},{y0 + radius_y} "
        f"L {x1},{y1 - radius_y} Q {x1},{y1} {x1 - radius_x},{y1} "
        f"L {x0 + radius_x},{y1} Q {x0},{y1} {x0},{y1 - radius_y} "
        f"L {x0},{y0 + radius_y} Q {x0},{y0} {x0 + radius_x},{y0} Z"
    )


def plot_regional_labelled(
    output_name: str,
    region_items,
    display_names,
    ncols: int = 3,
    width: int = 1900,
    height: int = 1100,
    horizontal_spacing: float = 0.06,
    vertical_spacing: float = 0.14,
):
    """World and regions in a grid with a horizontal legend on top.

    "Historical", "SSP2" and "Ambitious Circularity" are additionally labelled directly in the
    World panel (top left); "Historical" is missing in the legend. The regions fill the remaining
    cells (left to right, top to bottom).
    """
    plot_regions = [None, *region_items]
    nrows = math.ceil(len(plot_regions) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            "World" if region is None else display_names.get(str(region), str(region))
            for region in plot_regions
        ],
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for index, region in enumerate(plot_regions):
        row = index // ncols + 1
        col = index % ncols + 1
        add_ssp_traces(fig, region=region, showlegend=(index == 0), row=row, col=col)
        add_last_historical_line(fig, row=row, col=col)
        fig.update_yaxes(showgrid=True, row=row, col=col)
    fig.update_traces(showlegend=False, selector={"name": "Historical"})

    plot_width = width - LABELLED_MARGIN["l"] - LABELLED_MARGIN["r"]
    plot_height = height - LABELLED_MARGIN["t"] - LABELLED_MARGIN["b"]
    fig.add_shape(
        type="path",
        path=rounded_rectangle_path(
            *fig.layout.xaxis.domain,
            *fig.layout.yaxis.domain,
            radius_x=LABELLED_FRAME["radius"] / plot_width,
            radius_y=LABELLED_FRAME["radius"] / plot_height,
        ),
        xref="paper",
        yref="paper",
        line={"color": LABELLED_FRAME["color"], "width": LABELLED_FRAME["width"]},
        fillcolor="rgba(0,0,0,0)",
    )
    # the lower end is below zero, as in the autoscaled regional panels
    y_max = world_y_max()
    fig.update_yaxes(range=[-0.04 * y_max, y_max], row=1, col=1)
    add_line_labels(
        fig,
        LABELLED_FIRST_YEAR,
        y_max,
        font_size=LABELLED_FONT_SIZES["label"],
        historical_label=LABELLED_HISTORICAL_LABEL,
        end_labels=False,
        inline_labels=LABELLED_INLINE_LABELS,
    )

    fig.update_xaxes(
        range=[LABELLED_FIRST_YEAR, max(time)],
        tickvals=LABELLED_XTICKS,
        tickfont={"size": LABELLED_FONT_SIZES["tick"]},
    )
    fig.update_yaxes(nticks=4, tickfont={"size": LABELLED_FONT_SIZES["tick"]})
    for annotation in fig.layout.annotations[: len(plot_regions)]:
        annotation.font = {"size": LABELLED_FONT_SIZES["subplot_title"]}

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "font": {"size": LABELLED_FONT_SIZES["legend"]},
            "itemwidth": LABELLED_LEGEND_ITEM_WIDTH,
        },
        margin=LABELLED_MARGIN,
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text=YLABEL,
                x=-YLABEL_OFFSET / plot_width, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="middle",
                textangle=-90,
                font={"size": LABELLED_FONT_SIZES["axis_title"]},
            ),
        ],
    )
    save(fig, output_name, width=width, height=height)


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
    add_last_historical_line(fig, label=True, row=1, col=1)
    fig.update_yaxes(showgrid=True, row=1, col=1)

    for index, region in enumerate(region_items):
        row = index // ncols_regional + 1
        col = index % ncols_regional + 2
        add_ssp_traces(fig, region=region, showlegend=False, row=row, col=col)
        add_last_historical_line(fig, row=row, col=col)
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


def plot_combined_top(output_name: str, region_items, display_names, ncols_regional: int):
    """Global panel (from GLOBAL_TOP_FIRST_YEAR) on top and a regional grid below (from 2000).
    The global panel spans all but the last column; its line labels fill the last column."""
    nrows_regional = math.ceil(len(region_items) / ncols_regional)

    specs = [[{"colspan": ncols_regional - 1}] + [None] * (ncols_regional - 1)]
    for _ in range(nrows_regional):
        specs.append([{}] * ncols_regional)

    subplot_titles = ["Global"] + [display_names.get(str(r), str(r)) for r in region_items]

    fig = make_subplots(
        rows=nrows_regional + 1,
        cols=ncols_regional,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.09,
        horizontal_spacing=0.055,
        row_heights=[1.6] + [1] * nrows_regional,
    )

    add_ssp_traces(fig, region=None, showlegend=False, row=1, col=1)
    add_last_historical_line(fig, label=True, row=1, col=1)
    y_max = world_y_max()
    fig.update_yaxes(showgrid=True, range=[0, y_max], row=1, col=1)
    add_line_labels(fig, GLOBAL_TOP_FIRST_YEAR, y_max)
    # key of the validation markers, in the empty upper left corner of the global panel
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=True),
        selector=lambda trace: trace.meta == "validation legend" and trace.xaxis == "x",
    )

    for index, region in enumerate(region_items):
        row = index // ncols_regional + 2
        col = index % ncols_regional + 1
        add_ssp_traces(fig, region=region, showlegend=False, row=row, col=col)
        add_last_historical_line(fig, row=row, col=col)
        fig.update_yaxes(showgrid=True, row=row, col=col)

    for annotation in fig.layout.annotations[: len(subplot_titles)]:
        annotation.font = {"size": GLOBAL_TOP_FONT_SIZES["subplot_title"]}
    # left-aligned, clear of the label of the last historical year
    fig.layout.annotations[0].update(x=fig.layout.xaxis.domain[0], xanchor="left")
    for annotation in fig.layout.annotations:
        if annotation.text == "Last historical year":
            annotation.font.size = GLOBAL_TOP_FONT_SIZES["label"]

    fig.update_xaxes(range=[2000, max(time)], dtick=50)
    fig.update_xaxes(range=[GLOBAL_TOP_FIRST_YEAR, max(time)], row=1, col=1)
    fig.update_xaxes(tickfont={"size": GLOBAL_TOP_FONT_SIZES["tick"]})
    fig.update_yaxes(nticks=4, tickfont={"size": GLOBAL_TOP_FONT_SIZES["tick"]})

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={
            "x": fig.layout.xaxis.domain[0] + 0.01,
            "xanchor": "left",
            "y": fig.layout.yaxis.domain[1] - 0.005,
            "yanchor": "top",
            "title": {"text": "Literature projections", "font": {"size": GLOBAL_TOP_FONT_SIZES["tick"]}},
            "font": {"size": GLOBAL_TOP_FONT_SIZES["tick"], "color": "rgb(80,80,80)"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"t": 60, "l": 110, "b": 100, "r": 30},
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text=YLABEL,
                x=-0.06, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="center", yanchor="middle",
                textangle=-90,
                font={"size": GLOBAL_TOP_FONT_SIZES["axis_title"]},
            ),
        ],
    )
    save(fig, output_name, width=1700, height=1300)


plot_global("fig3_production_by_ssp_global")
plot_regional("fig3_production_by_ssp_h12", regions, REGION_SHORT_NAMES, ncols=4)
plot_regional("fig3_production_by_ssp_agg", AGG_REGION_ORDER, {}, ncols=3, include_world=True)
plot_regional_labelled("fig3_production_by_ssp_agg_horizontal", AGG_REGION_ORDER, {})
plot_combined("fig3_production_by_ssp_h12_global", regions, REGION_SHORT_NAMES, ncols_regional=4)

plot_combined_top(
    "fig3_production_by_ssp_h12_global_top", regions, REGION_SHORT_NAMES, ncols_regional=4
)

print("END")
