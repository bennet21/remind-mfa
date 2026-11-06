"""Figure 14: time series of the in-use stock per capita before and after the reconciliation.

The figure shows the concrete stock (product mass) in residential and commercial buildings, the
quantity compared by the reconciliation, as estimated by
- top-down, prior (input parameters; the previous REMIND-MFA) and reconciled: in-use stock of the
  stock-driven top-down MFA
- bottom-up, prior and reconciled: stock from the floorspace inflow (inflow-driven)
- combined: stock of the final MFA used by the model, which follows the reconciled top-down stock
  in the historical period and transitions to the reconciled bottom-up stock in the future.
In addition, it shows the top-down stock of the other uses (concrete in industrial buildings and
civil engineering, and all mortar; product mass), prior and reconciled. The final MFA uses its
reconciled estimate.
Colors follow figure 10 (top-down red, bottom-up blue), the other uses are grey. The prior
estimates have the colors of their reconciled partners with a lower opacity. The stocks used by
the model are highlighted by a grey band. In the future, all unused stocks are dashed. The lines
are labelled at their right end instead of a legend (in the regional figure, in the world panel).
The bottom-up stock is only shown from BU_FIRST_YEAR on, as the floorspace stock is set to zero
before (see `StockDrivenBottomUpCementMFASystem.compute_floorspace_stock`).

Uses the scenario MAIN_SCENARIO on the region level REGION_MAPPING. The data is extracted from a
model run that computes the top-down and bottom-up MFA once with the input and once with the output
parameters of the reconciliation. The reconciliation result is read from its cache
(`<export.path>/cache/cement_reconciliation_<region mapping>.pickle`) if possible.

Run from the repository root:
    uv run python scritps_figures/14_stock_reconciliation_timeseries.py
"""

import copy
import pickle
from dataclasses import dataclass

import flodym as fd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    AGG_REGION_ORDER,
    AGG_REGIONS,
    BACKGROUND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    MAIN_SCENARIO,
    PATH_CEMENT,
    REGION_MAPPING,
    figure_cache_dir,
)
from helpers import (
    add_last_historical_line,
    cache_is_valid,
    model_to_h12_mapping,
    save_pickle,
    with_short_item_names,
)

from generate_data import build_cfg
from remind_mfa.common.common_model import CommonModel
from remind_mfa.common.helpers import init_model

RECONCILIATION_CACHE = PATH_CEMENT / "cache" / f"cement_reconciliation_{REGION_MAPPING}.pickle"
FIGURE_CACHE = figure_cache_dir(MAIN_SCENARIO, REGION_MAPPING) / "stock_timeseries.pickle"

# First year of the bottom-up stock (floorspace stock is zero before).
BU_FIRST_YEAR = 2000

# Shown period.
PLOT_YEARS = (2000, 2050)

# Residential and commercial building end uses on the top-down (u), extended (e) and bottom-up
# (b) end-use dimensions.
BUILDING_END_USES = {"u": ["Res", "Com"], "e": ["RS", "RM", "Com"], "b": ["RS", "RM", "Com"]}

# Colors of figure 10: red for top-down, blue for bottom-up; transparent before the reconciliation.
TD_COLOR = (178, 24, 43)
BU_COLOR = (33, 102, 172)
PRE_RECONCILIATION_ALPHA = 0.45


def rgba(rgb: tuple[int, int, int], alpha: float = 1.0) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


OTHER_COLOR = (110, 110, 110)

# `used_in_future`: the final MFA uses the series in the future, so it stays solid.
SERIES_STYLES = {
    "td_pre": {
        "name": "Top-down, prior",
        "color": rgba(TD_COLOR, PRE_RECONCILIATION_ALPHA),
        "used_in_future": False,
    },
    "bu_pre": {
        "name": "Bottom-up, prior",
        "color": rgba(BU_COLOR, PRE_RECONCILIATION_ALPHA),
        "used_in_future": False,
    },
    "td_post": {"name": "Top-down, reconciled", "color": rgba(TD_COLOR), "used_in_future": False},
    "bu_post": {"name": "Bottom-up, reconciled", "color": rgba(BU_COLOR), "used_in_future": False},
    "other_pre": {
        "name": "Other uses, prior",
        "color": rgba(OTHER_COLOR, PRE_RECONCILIATION_ALPHA),
        "used_in_future": False,
    },
    "other_post": {
        "name": "Other uses, reconciled",
        "color": rgba(OTHER_COLOR),
        "used_in_future": True,
    },
}
WORLD_LABEL = "World"
USED_NAME = "Combined"
USED_COLOR = "black"
# Transparent band underlying the used stock.
USED_HIGHLIGHT_COLOR = "rgba(0,0,0,0.18)"

# Labels at the right end of the lines (replace a legend). Minimum vertical distance between two
# labels as a share of the y-axis range, and padding of the y-axis range around the data.
LABEL_FONT_SIZE = 12
LABEL_MIN_GAP = {"global": 0.04, "panel": 0.075}
Y_RANGE_PADDING = 0.05

# Steps of the build-up of the global figure: shown estimates and whether the used stock is shown.
BUILD_UP_STEPS: list[tuple[tuple[str, ...], bool]] = [
    (("td_pre", "other_pre"), False),
    (("td_pre", "other_pre", "bu_pre"), False),
    (("td_pre", "other_pre", "bu_pre", "td_post", "other_post", "bu_post"), False),
    (("td_pre", "other_pre", "bu_pre", "td_post", "other_post", "bu_post"), True),
]


@dataclass
class StockTimeseriesData:
    """In-use stocks (product mass, t), dims (t, r).

    Attributes:
        td_pre: Top-down concrete stock in residential and commercial buildings with the
            pre-reconciliation parameters.
        bu_pre: Bottom-up concrete stock with the pre-reconciliation parameters.
        td_post: Top-down concrete stock in residential and commercial buildings with the
            reconciled parameters.
        bu_post: Bottom-up concrete stock with the reconciled parameters.
        used: Concrete stock in residential and commercial buildings of the final (combined) MFA.
        other_pre: Other top-down stock with the pre-reconciliation parameters.
        other_post: Other top-down stock with the reconciled parameters.
        other_used: Other stock of the final (combined) MFA.
        population: Population, dims (t, r).
    """

    td_pre: fd.FlodymArray
    bu_pre: fd.FlodymArray
    td_post: fd.FlodymArray
    bu_post: fd.FlodymArray
    used: fd.FlodymArray
    other_pre: fd.FlodymArray
    other_post: fd.FlodymArray
    other_used: fd.FlodymArray
    population: fd.FlodymArray


def sum_items(array: fd.FlodymArray, letter: str, items: list[str]) -> fd.FlodymArray:
    """Sum an array over the given items of dimension `letter`, leaving dims (t, r)."""
    total = array[{letter: items[0]}].sum_to(("t", "r"))
    for item in items[1:]:
        total = total + array[{letter: item}].sum_to(("t", "r"))
    return total


def building_concrete(stock: fd.FlodymArray) -> fd.FlodymArray:
    """Concrete stock of an MFA stock array in residential and commercial buildings, dims (t, r)."""
    stock = with_short_item_names(stock)
    if "m" in stock.dims.letters:
        stock = stock[{"m": "concrete"}]
    letter = next(letter for letter in BUILDING_END_USES if letter in stock.dims.letters)
    return sum_items(stock, letter, BUILDING_END_USES[letter])


def other_stock(stock: fd.FlodymArray) -> fd.FlodymArray:
    """Stock of an MFA stock array outside of the concrete in residential and commercial
    buildings (product mass), dims (t, r)."""
    return stock.sum_to(("t", "r")) - building_concrete(stock)


def run_projection(model, parameters: dict[str, fd.Parameter]):
    """Top-down and bottom-up future MFA with the given historical parameters, following
    `CementModel.run_with_reconciliation`."""
    model.parameters = copy.deepcopy(parameters)
    model.calculate_derived_parameters()
    model.historical_mfa = model.make_mfa(historical=True)
    model.historical_mfa.compute()
    trade_set = model.historical_mfa.trade_set
    model.extrapolate_parameters()

    td_mfa = model.make_mfa(historical=False)
    td_mfa.compute(model.get_long_term_stock(), trade_set)
    bu_mfa = model.make_bottom_up_mfa()
    bu_mfa.compute(td_mfa.stocks["use"], trade_set)
    return td_mfa, bu_mfa


def extract_figure_data() -> StockTimeseriesData:
    """Run the cement model up to the reconciliation and project the stocks with the input and
    the output parameters of the reconciliation."""
    model = init_model(cfg=build_cfg(MAIN_SCENARIO, reconcile=True, region_mapping=REGION_MAPPING))
    CommonModel.run(model, future=False)
    model.parameters = model.historical_parameters
    model.reconcile_parameters()
    reconciliation = model.parameter_reconciliation

    td_pre, bu_pre = run_projection(model, reconciliation.input_prms)
    td_post, bu_post = run_projection(model, reconciliation.output_prms)
    return StockTimeseriesData(
        td_pre=building_concrete(td_pre.stocks["use"].stock),
        bu_pre=building_concrete(bu_pre.stocks["bu_use"].stock),
        td_post=building_concrete(td_post.stocks["use"].stock),
        bu_post=building_concrete(bu_post.stocks["bu_use"].stock),
        used=building_concrete(bu_post.stocks["use"].stock),
        other_pre=other_stock(td_pre.stocks["use"].stock),
        other_post=other_stock(td_post.stocks["use"].stock),
        other_used=other_stock(bu_post.stocks["use"].stock),
        population=bu_post.parameters["population"].sum_to(("t", "r")),
    )


def load_figure_data(force_refresh: bool = False) -> StockTimeseriesData:
    """Load the figure data from the figure cache, or extract it from a model run.

    The figure cache is used if it is newer than the reconciliation cache.
    """
    if not force_refresh and cache_is_valid(RECONCILIATION_CACHE, {"cache": FIGURE_CACHE}):
        print(f"Loading figure data from cache: {FIGURE_CACHE}")
        with FIGURE_CACHE.open("rb") as file_handle:
            return pickle.load(file_handle)

    print("Building figure data cache from a model run")
    figure_data = extract_figure_data()
    save_pickle(figure_data, FIGURE_CACHE)
    return figure_data


def per_capita(
    stock: fd.FlodymArray, population: fd.FlodymArray, regions: list[str] | None = None
) -> np.ndarray:
    """Stock per capita (t/capita) of the given model regions (all if None), dims (t,)."""

    def regional_sum(array: fd.FlodymArray) -> np.ndarray:
        values = array.sum_to(("t", "r")).values
        if regions is None:
            return values.sum(axis=1)
        region_items = [str(region) for region in array.dims["r"].items]
        return values[:, [region_items.index(region) for region in regions]].sum(axis=1)

    return regional_sum(stock) / regional_sum(population)


def agg_region_members(agg_region: str) -> list[str]:
    """Model regions that belong to an aggregated region of `AGG_REGIONS`."""
    to_h12 = model_to_h12_mapping(REGION_MAPPING)
    return [region for region, h12 in to_h12.items() if AGG_REGIONS[h12] == agg_region]


def shown_values(
    data: StockTimeseriesData, regions: list[str] | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Years of PLOT_YEARS and the per-capita series of all estimates in them."""
    years = np.array(data.used.dims["t"].items)
    shown = (years >= PLOT_YEARS[0]) & (years <= PLOT_YEARS[1])
    years = years[shown]
    values = {
        key: per_capita(getattr(data, key), data.population, regions)[shown]
        for key in (*SERIES_STYLES, "used", "other_used")
    }
    for key in ("bu_pre", "bu_post"):
        values[key] = np.where(years >= BU_FIRST_YEAR, values[key], np.nan)
    return years, values


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def add_series(
    fig: go.Figure,
    years: np.ndarray,
    values: np.ndarray,
    key: str,
    subplot: dict,
):
    """Line of an estimate: solid in the historical period, dashed in the future if unused."""
    style = SERIES_STYLES[key]
    line = {"color": style["color"], "width": 2}
    if style["used_in_future"]:
        fig.add_trace(
            go.Scatter(x=years, y=values, mode="lines", name=style["name"], line=line),
            **subplot,
        )
        return
    historical = years <= LAST_HISTORICAL_YEAR
    future = years >= LAST_HISTORICAL_YEAR
    fig.add_trace(
        go.Scatter(
            x=years[historical],
            y=values[historical],
            mode="lines",
            name=style["name"],
            line=line,
        ),
        **subplot,
    )
    fig.add_trace(
        go.Scatter(
            x=years[future],
            y=values[future],
            mode="lines",
            name=style["name"],
            line={**line, "dash": "dash"},
        ),
        **subplot,
    )


def add_panel(
    fig: go.Figure,
    years: np.ndarray,
    values: dict[str, np.ndarray],
    keys: tuple[str, ...] = tuple(SERIES_STYLES),
    show_used: bool = True,
    row: int | None = None,
    col: int | None = None,
):
    """The estimates `keys` of one region and, if `show_used`, the used stocks highlighted."""
    subplot = {"row": row, "col": col} if row is not None else {}
    if show_used:
        used_keys = ("used", "other_used") if "other_post" in keys else ("used",)
        for key in used_keys:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=values[key],
                    mode="lines",
                    line={"color": USED_HIGHLIGHT_COLOR, "width": 12},
                    hoverinfo="skip",
                ),
                **subplot,
            )
    for key in keys:
        add_series(fig, years, values[key], key, subplot)
    if not show_used:
        return
    # historically, the used stock is the reconciled top-down stock, which stays visible
    future = years >= LAST_HISTORICAL_YEAR
    fig.add_trace(
        go.Scatter(
            x=years[future],
            y=values["used"][future],
            mode="lines",
            name=USED_NAME,
            line={"color": USED_COLOR, "width": 2.5},
        ),
        **subplot,
    )


def y_range(values: dict[str, np.ndarray]) -> list[float]:
    """Y-axis range spanning all shown values, with Y_RANGE_PADDING on both sides."""
    all_values = np.concatenate(list(values.values()))
    lower, upper = np.nanmin(all_values), np.nanmax(all_values)
    padding = Y_RANGE_PADDING * (upper - lower)
    return [float(lower - padding), float(upper + padding)]


def spread_labels(targets: list[float], min_gap: float, bounds: list[float]) -> list[float]:
    """Label positions close to `targets`, at least `min_gap` apart and within `bounds`."""
    order = np.argsort(targets)
    positions = np.array(targets, dtype=float)[order]
    for _ in range(200):
        moved = False
        for index in range(len(positions) - 1):
            overlap = min_gap - (positions[index + 1] - positions[index])
            if overlap > 1e-9:
                positions[index] -= overlap / 2
                positions[index + 1] += overlap / 2
                moved = True
        positions = np.clip(positions, bounds[0], bounds[1])
        if not moved:
            break
    spread = np.empty_like(positions)
    spread[order] = positions
    return list(spread)


def add_end_labels(
    fig: go.Figure,
    values: dict[str, np.ndarray],
    y_bounds: list[float],
    min_gap: float,
    keys: tuple[str, ...] = tuple(SERIES_STYLES),
    show_used: bool = True,
    xref: str = "x",
    yref: str = "y",
):
    """Label each line of `keys` (and the used stock if `show_used`) at its end (PLOT_YEARS[1]).
    Overlapping labels are spread vertically and connected to their line end by a thin line."""
    labels = [(key, SERIES_STYLES[key]["name"], SERIES_STYLES[key]["color"]) for key in keys]
    if show_used:
        labels.append(("used", USED_NAME, USED_COLOR))
    targets = [float(values[key][-1]) for key, _, _ in labels]
    gap = min_gap * (y_bounds[1] - y_bounds[0])
    positions = spread_labels(targets, gap, y_bounds)
    label_x = PLOT_YEARS[1] + 0.04 * (PLOT_YEARS[1] - PLOT_YEARS[0])
    for (key, text, color), target, position in zip(labels, targets, positions):
        fig.add_annotation(
            x=PLOT_YEARS[1],
            y=target,
            ax=label_x,
            ay=position,
            xref=xref,
            yref=yref,
            axref=xref,
            ayref=yref,
            text=f"<b>{text}</b>" if key == "used" else text,
            xanchor="left",
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.6,
            arrowcolor="rgba(0,0,0,0.35)",
            standoff=2,
            font={"size": LABEL_FONT_SIZE, "color": color},
        )


def plot_global(
    data: StockTimeseriesData,
    output_name: str,
    keys: tuple[str, ...] = tuple(SERIES_STYLES),
    show_used: bool = True,
    range_keys: tuple[str, ...] | None = None,
):
    """Global figure with the estimates `keys` and, if `show_used`, the used stock. The y-axis
    spans the series `range_keys` (default: all shown series), so that the steps of a build-up
    can share their axes."""
    years, values = shown_values(data)
    fig = go.Figure()
    add_panel(fig, years, values, keys, show_used)
    add_last_historical_line(fig, label=True)
    if range_keys is None:
        range_keys = (*keys, "used") if show_used else keys
    y_bounds = y_range({key: values[key] for key in range_keys})
    add_end_labels(fig, values, y_bounds, LABEL_MIN_GAP["global"], keys, show_used)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin={"t": 30, "l": 80, "b": 60, "r": 200},
    )
    fig.update_xaxes(
        title={"text": "Year", "font": {"size": 15}},
        range=list(PLOT_YEARS),
        tickfont={"size": 12},
    )
    fig.update_yaxes(
        title={"text": "In-use stock (t/capita)", "font": {"size": 15}},
        range=y_bounds,
        tickfont={"size": 12},
    )
    save(fig, output_name, width=1200, height=600)


def plot_agg_regions(data: StockTimeseriesData, output_name: str):
    """One panel for the world and one per aggregated region. The lines are labelled in the
    world panel, which gets extra space on its right for the labels."""
    panels = [WORLD_LABEL, *AGG_REGION_ORDER]
    n_cols = 3
    fig = make_subplots(
        rows=2,
        cols=n_cols,
        subplot_titles=panels,
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )
    for index, panel in enumerate(panels):
        row, col = divmod(index, n_cols)
        regions = None if panel == WORLD_LABEL else agg_region_members(panel)
        years, values = shown_values(data, regions)
        add_panel(fig, years, values, row=row + 1, col=col + 1)
        add_last_historical_line(fig, label=index == 0, row=row + 1, col=col + 1)
        y_bounds = y_range(values)
        fig.update_yaxes(range=y_bounds, row=row + 1, col=col + 1)
        if panel == WORLD_LABEL:
            add_end_labels(fig, values, y_bounds, LABEL_MIN_GAP["panel"])

    # Column domains with a wider gap after the first column for the labels of the world panel.
    # The subplot titles are centred above their column, so they move with it.
    column_domains = [[0.0, 0.25], [0.405, 0.685], [0.72, 1.0]]
    for index in range(2 * n_cols):
        row, col = divmod(index, n_cols)
        fig.update_xaxes(domain=column_domains[col], row=row + 1, col=col + 1)
    for index, annotation in enumerate(fig.layout.annotations[: len(panels)]):
        domain = column_domains[index % n_cols]
        annotation.x = (domain[0] + domain[1]) / 2
        annotation.font = {"size": 15}

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin={"t": 40, "l": 90, "b": 90, "r": 30},
    )
    fig.update_xaxes(range=list(PLOT_YEARS), tickfont={"size": 12})
    fig.update_yaxes(tickfont={"size": 12})
    fig.add_annotation(
        text="In-use stock (t/capita)",
        x=-0.05,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font={"size": 15},
    )
    fig.add_annotation(
        text="Year",
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15},
    )
    save(fig, output_name, width=1800, height=900)


figure_data = load_figure_data()
plot_global(figure_data, f"fig14_stock_reconciliation_timeseries_global_{REGION_MAPPING}")
plot_agg_regions(figure_data, f"fig14_stock_reconciliation_timeseries_agg_{REGION_MAPPING}")
# step-by-step build-up of the global figure (e.g. for slides), with shared axes
for step, (keys, show_used) in enumerate(BUILD_UP_STEPS, start=1):
    plot_global(
        figure_data,
        f"fig14_stock_reconciliation_timeseries_global_step{step}_{REGION_MAPPING}",
        keys,
        show_used,
        range_keys=(*BUILD_UP_STEPS[-1][0], "used", "other_used"),
    )

print("END")
