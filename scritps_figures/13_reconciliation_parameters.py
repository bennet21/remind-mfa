"""Figure 13: corrections of the reconciled parameters across all model regions.

The parameter reconciliation scales each uncertain parameter item k (region and, e.g., end use) by
a factor exp(δ_k). The figure shows the distribution of these corrections over the model regions,
with one row per parameter and item of its non-regional dimensions:
- the change factor θ' / θ of the parameter value in the last historical year. For splits, it
  includes the re-normalization after the reconciliation and therefore differs from exp(δ). The
  shaded band per row is the prior uncertainty exp(±σ), with σ the assumed log-space uncertainty
  of the item (lighter extension: range of σ across regions).
- optionally (SHOW_STANDARDIZED), a left panel with the standardized correction z = δ / σ. The
  reconciliation minimizes Σ z², so |z| shows how far a parameter was moved relative to its
  uncertainty (shaded: |z| <= 1).
The violins show the distribution over all model regions. The points show the N_POINT_REGIONS
model regions with the largest top-down in-use stock (all end uses, before the reconciliation) in
the last historical year, colored by their H12 region.

Uses the scenario MAIN_SCENARIO on the region level REGION_MAPPING_FIG13. The data is extracted
from a model run up to the reconciliation, which reads the reconciliation result from its cache
(`<export.path>/cache/cement_reconciliation_<region mapping>.pickle`) if possible.

Run from the repository root:
    uv run python scritps_figures/13_reconciliation_parameters.py
"""

import logging
import pickle
from dataclasses import dataclass

import flodym as fd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    AGG_REGION_COLORS,
    AGG_REGION_ORDER,
    AGG_REGIONS,
    BACKGROUND,
    COLORS_REMIND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    MAIN_SCENARIO,
    PATH_CEMENT,
    figure_cache_dir,
)
from helpers import SHORT_ITEM_NAMES, cache_is_valid, model_to_h12_mapping, save_pickle

from generate_data import build_cfg
from remind_mfa.cement.cement_parameter_reconciliation import CementParameterReconciliation
from remind_mfa.common.common_model import CommonModel
from remind_mfa.common.helpers import init_model

# Region level of the figure, independent of constants.REGION_MAPPING.
REGION_MAPPING_FIG13 = "gtap155"

RECONCILIATION_CACHE = PATH_CEMENT / "cache" / f"cement_reconciliation_{REGION_MAPPING_FIG13}.pickle"
FIGURE_CACHE = (
    figure_cache_dir(MAIN_SCENARIO, REGION_MAPPING_FIG13) / "reconciliation_figure_data.pickle"
)


@dataclass
class ParameterCorrection:
    """Corrections of one reconciled parameter, over its reconciled dimensions (incl. `r`).

    Attributes:
        delta: Log-correction δ.
        sigma: Assumed standard deviation σ of δ.
        change_factor: θ' / θ of the parameter in the last historical year, after the
            re-normalization of splits.
    """

    delta: fd.FlodymArray
    sigma: fd.FlodymArray
    change_factor: fd.FlodymArray


@dataclass
class ReconciliationFigureData:
    """Figure data of a reconciled model run.

    Attributes:
        corrections: Corrections of all parameters with nonzero uncertainty.
        td_stock: Top-down in-use stock (product mass, all end uses) before the reconciliation in
            the last historical year, dims (r,).
    """

    corrections: dict[str, ParameterCorrection]
    td_stock: fd.FlodymArray


def last_historical_slice(array: fd.FlodymArray) -> fd.FlodymArray:
    """Select the last historical year if `array` has a time dimension."""
    for letter in ("h", "t"):
        if letter in array.dims.letters:
            return array[{letter: LAST_HISTORICAL_YEAR}]
    return array


def extract_corrections(model) -> dict[str, ParameterCorrection]:
    """Corrections of all parameters with nonzero uncertainty of a reconciled model run."""
    reconciliation = model.parameter_reconciliation
    corrections = {}
    for prm_name, delta in reconciliation.delta.items():
        adj_dims = reconciliation.prms_adj_dims[prm_name]
        sigma = reconciliation.reshape_np_to_fd(
            np.sqrt(reconciliation.variances[prm_name]), adj_dims
        )
        input_prm = last_historical_slice(reconciliation.input_prms[prm_name])
        output_prm = last_historical_slice(reconciliation.output_prms[prm_name])
        corrections[prm_name] = ParameterCorrection(
            delta=delta,
            sigma=sigma,
            change_factor=(output_prm / input_prm).cast_to(adj_dims),
        )
    return corrections


def extract_figure_data(model) -> ReconciliationFigureData:
    """Figure data of a model run up to the reconciliation (see `run_reconciliation`)."""
    # the historical MFA is not yet replaced by the reconciled one at this point
    td_stock = model.historical_mfa.stocks["use"].stock[{"h": LAST_HISTORICAL_YEAR}]
    return ReconciliationFigureData(
        corrections=extract_corrections(model), td_stock=td_stock.sum_to(("r",))
    )


def run_reconciliation():
    """Run the cement model up to (and including) the parameter reconciliation."""
    model = init_model(cfg=build_cfg(MAIN_SCENARIO, reconcile=True, region_mapping=REGION_MAPPING_FIG13))
    CommonModel.run(model, future=False)
    model.parameters = model.historical_parameters
    model.reconcile_parameters()
    return model


def load_figure_data(force_refresh: bool = False) -> ReconciliationFigureData:
    """Load the figure data from the figure cache, or extract it from a model run.

    The figure cache is used if it is newer than the reconciliation cache.
    """
    if not force_refresh and cache_is_valid(RECONCILIATION_CACHE, {"cache": FIGURE_CACHE}):
        print(f"Loading figure data from cache: {FIGURE_CACHE}")
        with FIGURE_CACHE.open("rb") as file_handle:
            return pickle.load(file_handle)

    print("Building figure data cache from a model run up to the reconciliation")
    figure_data = extract_figure_data(run_reconciliation())
    save_pickle(figure_data, FIGURE_CACHE)
    return figure_data


def largest_stock_regions(td_stock: fd.FlodymArray, n_regions: int) -> list[str]:
    """The `n_regions` regions with the largest stock, in descending order of stock."""
    regions = [str(region) for region in td_stock.dims["r"].items]
    order = np.argsort(td_stock.values)[::-1]
    return [regions[index] for index in order[:n_regions]]


@dataclass
class CorrectionRow:
    """Corrections of one parameter item (non-regional dimensions fixed) across the regions."""

    label: str
    prm_name: str
    regions: list[str]
    sigma: np.ndarray
    z: np.ndarray
    change_factor: np.ndarray


def item_label(letter: str, item: str) -> str:
    """Short name of a dimension item; the common (c) and residential-type (d) end uses use the
    names of the end-use (u) and bottom-up end-use (b) dimensions."""
    names = SHORT_ITEM_NAMES.get({"c": "u", "d": "b"}.get(letter, letter), {})
    return names.get(item, item)


def correction_rows(corrections: dict[str, ParameterCorrection]) -> list[CorrectionRow]:
    """One row per parameter in PARAMETER_ORDER and item of its non-regional dimensions.

    Items that the reconciliation does not change (δ = 0 in all regions, e.g. mortar items or
    zero material intensities) are skipped.
    """
    rows = []
    for prm_name in PARAMETER_ORDER:
        correction = corrections[prm_name]
        delta = correction.delta
        other_dims = [dim for dim in delta.dims if dim.letter != "r"]
        regions = [str(region) for region in delta.dims["r"].items]
        slicers = CementParameterReconciliation.iter_dim_slicers(
            fd.DimensionSet(dim_list=other_dims)
        )
        for slicer in slicers:
            delta_values = delta[slicer].values
            if np.all(np.abs(delta_values) < MIN_ABS_DELTA):
                continue
            sigma_values = correction.sigma[slicer].values
            change_factor = correction.change_factor[slicer].values
            items = " ".join(item_label(letter, item) for letter, item in slicer.items())
            rows.append(
                CorrectionRow(
                    label=f"{PARAMETER_DISPLAY_NAMES[prm_name]} {items}".strip(),
                    prm_name=prm_name,
                    regions=regions,
                    sigma=sigma_values,
                    z=delta_values / sigma_values,
                    # zero input values (e.g. zero shares) have no defined change factor
                    change_factor=np.where(np.isfinite(change_factor), change_factor, np.nan),
                )
            )
    return rows


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def add_uncertainty_bands(
    fig: go.Figure, rows: list[CorrectionRow], y_positions: list[float], xref: str
):
    """Per-row bands of the prior uncertainty of the change factor, exp(±σ).

    If σ differs between the regions of a row, a lighter outer band extends from the smallest to
    the largest σ.
    """
    for row, y_position in zip(rows, y_positions):
        sigma_min, sigma_max = float(np.min(row.sigma)), float(np.max(row.sigma))
        for sigma, color in (
            (sigma_max, BAND_COLORS["1σ range"]),
            (sigma_min, BAND_COLORS["1σ"]),
        ):
            fig.add_shape(
                type="rect",
                xref=xref,
                yref="y",
                x0=np.exp(-sigma),
                x1=np.exp(sigma),
                y0=y_position - BAND_HALF_HEIGHT,
                y1=y_position + BAND_HALF_HEIGHT,
                fillcolor=color,
                line_width=0,
                layer="below",
            )


@dataclass
class PointSeries:
    """Points of one region or region group, one point per row.

    Attributes:
        name: Legend entry.
        hover_name: Name shown on hover.
        legend_group: Legend group of the series.
        legend_title: Title of the legend group.
        show_in_legend: Whether the series has its own legend entry.
        color: Marker color.
        z: Standardized correction per row.
        change_factor: Change factor per row.
        y_offset: Vertical offset of the points from the row center, in row heights.
        jitter: Maximum random vertical offset of the points, in row heights.
    """

    name: str
    hover_name: str
    legend_group: str
    legend_title: str
    show_in_legend: bool
    color: str
    z: np.ndarray
    change_factor: np.ndarray
    y_offset: float = 0.0
    jitter: float = 0.0


def country_point_series(rows: list[CorrectionRow], countries: list[str]) -> list[PointSeries]:
    """One series per country in `countries`, colored by its H12 region (one legend entry per
    H12 region)."""
    to_h12 = model_to_h12_mapping(REGION_MAPPING_FIG13)
    legend_title = f"{len(countries)} largest regions<br>by stock, colored by<br>H12 region"
    series = []
    shown_h12_regions = set()
    for country in sorted(countries, key=lambda country: (to_h12[country], country)):
        h12_region = to_h12[country]
        index = rows[0].regions.index(country)
        series.append(
            PointSeries(
                name=h12_region,
                hover_name=country,
                legend_group="countries",
                legend_title=legend_title,
                show_in_legend=h12_region not in shown_h12_regions,
                color=COLORS_REMIND[h12_region],
                z=np.array([row.z[index] for row in rows]),
                change_factor=np.array([row.change_factor[index] for row in rows]),
                jitter=JITTER,
            )
        )
        shown_h12_regions.add(h12_region)
    return series


def aggregated_point_series(
    rows: list[CorrectionRow], td_stock: fd.FlodymArray
) -> list[PointSeries]:
    """One series per aggregated region of `AGG_REGIONS`, with the mean over its model regions
    weighted by their top-down stock: the geometric mean of the change factor and the arithmetic
    mean of z (both means in log space). Regions without a defined change factor are skipped."""
    to_h12 = model_to_h12_mapping(REGION_MAPPING_FIG13)
    regions = rows[0].regions
    stock_regions = [str(region) for region in td_stock.dims["r"].items]
    weights = np.array([td_stock.values[stock_regions.index(region)] for region in regions])
    aggregated_regions = np.array([AGG_REGIONS[to_h12[region]] for region in regions])
    # spread the aggregated regions vertically within the row, first region on top
    y_offsets = np.linspace(AGG_POINT_SPREAD, -AGG_POINT_SPREAD, len(AGG_REGION_ORDER))

    series = []
    for aggregated_region, y_offset in zip(AGG_REGION_ORDER, y_offsets):
        in_region = aggregated_regions == aggregated_region
        z_means, change_factor_means = [], []
        for row in rows:
            valid = in_region & np.isfinite(row.change_factor)
            row_weights = weights[valid] / weights[valid].sum()
            z_means.append(float(np.sum(row_weights * row.z[valid])))
            change_factor_means.append(
                float(np.exp(np.sum(row_weights * np.log(row.change_factor[valid]))))
            )
        series.append(
            PointSeries(
                name=aggregated_region,
                hover_name=f"{aggregated_region} (stock-weighted mean)",
                legend_group="aggregated",
                legend_title="Stock-weighted mean",
                show_in_legend=True,
                color=AGG_REGION_COLORS[aggregated_region],
                z=np.array(z_means),
                change_factor=np.array(change_factor_means),
                y_offset=float(y_offset),
            )
        )
    return series


def add_direct_labels(
    fig: go.Figure,
    rows: list[CorrectionRow],
    y_positions: list[float],
    xref: str,
    show_violins: bool,
    show_box: bool,
):
    """Label the uncertainty bands, the violins and the boxes directly in the change-factor
    panel (replaces a legend). Each label is placed in an empty area of the plot, with an arrow
    to an example in a row of `DIRECT_LABELS`."""
    rows_by_label = {row.label: (row, y) for row, y in zip(rows, y_positions)}

    def target_x(kind: str, row: CorrectionRow) -> float:
        """Change factor the arrow of a label points to."""
        sigma_min, sigma_max = float(np.min(row.sigma)), float(np.max(row.sigma))
        match kind:
            case "band":
                return float(np.exp(0.7 * sigma_min))
            case "band_range":
                return float(np.exp(0.5 * (sigma_min + sigma_max)))
            case "violin":
                return float(np.nanpercentile(row.change_factor, DIRECT_LABEL_VIOLIN_PERCENTILE))
            case _:
                return float(np.nanmedian(row.change_factor))

    shown = {"band", "band_range"} | ({"violin"} if show_violins else set())
    shown |= {"box"} if show_box else set()
    for kind, (row_label, text, text_x, text_dy, target_dy) in DIRECT_LABELS.items():
        if kind not in shown:
            continue
        row, y_position = rows_by_label[row_label]
        text = text.format(n_regions=len(row.regions))
        fig.add_annotation(
            x=np.log10(target_x(kind, row)),
            y=y_position + target_dy,
            xref=xref,
            yref="y",
            ax=np.log10(text_x),
            ay=y_position + text_dy,
            axref=xref,
            ayref="y",
            text=text,
            xanchor="left",
            align="left",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=0.8,
            arrowcolor="black",
            standoff=1,
            font={"size": 11},
        )


def plot_corrections(
    rows: list[CorrectionRow],
    point_series: list[PointSeries],
    output_name: str,
    show_standardized: bool,
    show_violins: bool,
):
    """Plot the change factors with their prior uncertainty bands and, if `show_standardized`,
    a left panel with the standardized corrections z = δ / σ.

    If `show_violins`, the distribution over all regions is drawn as a violin per row, with a box
    of the median and the interquartile range unless single countries are shown as points. The
    `point_series` are drawn on top."""
    change_factor_title = (
        f"Change factor of the parameter value in {LAST_HISTORICAL_YEAR} (reconciled / original)"
    )
    if show_standardized:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)
        panels = ((1, "z"), (2, "change_factor"))
        change_factor_col = 2
        fig.update_xaxes(title_text="Standardized correction δ / σ", row=1, col=1)
    else:
        fig = make_subplots(rows=1, cols=1)
        panels = ((1, "change_factor"),)
        change_factor_col = 1
    change_factor_xref = "x2" if show_standardized else "x"
    show_box = not any(series.legend_group == "countries" for series in point_series)

    # y positions from top to bottom, with a gap between parameters
    y_positions = []
    y = 0.0
    for index, row in enumerate(rows):
        if index > 0 and row.prm_name != rows[index - 1].prm_name:
            y -= PARAMETER_GAP
        y_positions.append(y)
        y -= 1.0
    y_range = [min(y_positions) - 0.7, 0.7]
    rng = np.random.default_rng(0)

    if show_standardized:
        # uncertainty band |z| <= 1
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y domain",
            x0=-1,
            x1=1,
            y0=0,
            y1=1,
            fillcolor=BAND_COLORS["1σ"],
            line_width=0,
            layer="below",
        )
    add_uncertainty_bands(fig, rows, y_positions, change_factor_xref)

    # reference lines (no correction) as traces, so that they lie above the bands and below the
    # violins and points
    reference_lines = ((1, 0),) if show_standardized else ()
    for col, x_value in (*reference_lines, (change_factor_col, 1)):
        fig.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=y_range,
                mode="lines",
                line={"color": "black", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    if show_violins:
        for col, attribute in panels:
            for row, y_position in zip(rows, y_positions):
                values = getattr(row, attribute)
                fig.add_trace(
                    go.Violin(
                        x=values,
                        y=np.full(len(values), y_position),
                        orientation="h",
                        width=2 * BAND_HALF_HEIGHT,
                        spanmode="hard",
                        points=False,
                        box_visible=show_box,
                        box={
                            "fillcolor": VIOLIN_BOX_COLOR,
                            "line": {"color": "white", "width": 1},
                            "width": 0.2,
                        },
                        fillcolor=VIOLIN_FILL_COLOR,
                        line={"color": VIOLIN_LINE_COLOR, "width": 0.8},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )

    add_direct_labels(fig, rows, y_positions, change_factor_xref, show_violins, show_box)

    for series in point_series:
        for col, attribute in panels:
            fig.add_trace(
                go.Scatter(
                    x=getattr(series, attribute),
                    y=[
                        y_position + series.y_offset + rng.uniform(-series.jitter, series.jitter)
                        for y_position in y_positions
                    ],
                    mode="markers",
                    name=series.name,
                    legendgroup=series.legend_group,
                    legendgrouptitle={"text": series.legend_title},
                    showlegend=col == 1 and series.show_in_legend,
                    marker={
                        "size": POINT_SIZE,
                        "color": series.color,
                        "opacity": POINT_OPACITY,
                        "line": {"color": "white", "width": 0.8},
                    },
                    hovertemplate=f"{series.hover_name}<br>%{{x:.3f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )

    if show_standardized:
        max_abs_z = max(float(np.nanmax(np.abs(row.z))) for row in rows)
        z_limit = max(2.2, 1.1 * max_abs_z)
        fig.update_xaxes(range=[-z_limit, z_limit], zeroline=False, row=1, col=1)
    # symmetric in log space, covering all uncertainty bands and change factors
    max_abs_log = max(
        max(float(np.max(row.sigma)), float(np.nanmax(np.abs(np.log(row.change_factor)))))
        for row in rows
    )
    log10_limit = 1.1 * max_abs_log / np.log(10)
    tick_values = [tick for tick in CHANGE_FACTOR_TICKS if abs(np.log10(tick)) <= log10_limit]
    fig.update_xaxes(
        type="log",
        range=[-log10_limit, log10_limit],
        tickvals=tick_values,
        ticktext=[f"{tick:g}" for tick in tick_values],
        title_text=change_factor_title,
        row=1,
        col=change_factor_col,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e6e6e6",
        tickfont={"size": 12},
        title_font={"size": 13},
        showline=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        tickvals=y_positions,
        ticktext=[row.label for row in rows],
        tickfont={"size": 12},
        showgrid=False,
        range=y_range,
        zeroline=False,
    )
    fig.update_layout(
        template="plotly_white",
        font={"color": "black"},
        showlegend=bool(point_series),
        legend={
            "font": {"size": 12},
            "grouptitlefont": {"size": 13},
            "groupclick": "toggleitem",
            "tracegroupgap": 14,
            "itemsizing": "constant",
        },
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    width = 1350 if show_standardized else 900
    save(fig, output_name, width=width, height=90 + 19 * (len(y_positions) + 2))


# Parameters in display order: bottom-up parameters first, then top-down parameters.
PARAMETER_DISPLAY_NAMES = {
    "floorspace_pc": "Floor space per capita",
    "floorspace_net_to_gross": "Net-to-gross floor space",
    "unoccupied_residential_stock_share": "Unoccupied res. share",
    "residential_type_split": "Res. type split",
    "structure_split": "Structure split",
    "concrete_building_mi": "Concrete intensity",
    "end_use_split": "End-use split",
    "construction_material_split": "Material split",
    "cement_ratio": "Cement ratio",
    "lifetime_mean": "Lifetime",
    "cement_loss_rate": "Cement loss rate",
}
PARAMETER_ORDER = list(PARAMETER_DISPLAY_NAMES)
# Items with |δ| below this value in all regions are not shown.
MIN_ABS_DELTA = 1e-9
# Vertical gap between parameters, jitter of the country points and vertical spread of the
# aggregated-region points, in row heights.
PARAMETER_GAP = 0.6
JITTER = 0.2
AGG_POINT_SPREAD = 0.24
# Half height of the uncertainty bands, in row heights.
BAND_HALF_HEIGHT = 0.4
# Colors of the uncertainty bands, drawn from outside to inside.
BAND_COLORS = {"1σ range": "#e3e3e3", "1σ": "#c8c8c8"}
# Direct labels (replacing a legend): row of the example, text, change factor of the left edge of
# the text, and vertical offsets of the text and of the arrow tip from the row, in row heights.
# The texts lie in empty areas of the plot, away from their example, so that the arrows are steep.
DIRECT_LABELS = {
    "box": ("Floor space per capita Com", "Interquartile range<br>and median", 1.7, -2.6, 0.0),
    "band": ("Structure split RS M", "Prior uncertainty<br>exp(±σ)", 1.5, -2.0, -0.3),
    "band_range": (
        "Concrete intensity RS S",
        "Range of exp(±σ)<br>across regions",
        1.8,
        2.5,
        0.3,
    ),
    "violin": ("End-use split Com", "Distribution over<br>all {n_regions} regions", 0.35, 3.1, 0.3),
}
# Percentile of the values of the example row the arrow of the violin label points to.
DIRECT_LABEL_VIOLIN_PERCENTILE = 10
# Candidate ticks of the change-factor axis; those within the axis range are shown.
CHANGE_FACTOR_TICKS = [0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4]
VIOLIN_FILL_COLOR = "rgba(255,255,255,0.75)"
VIOLIN_LINE_COLOR = "#555555"
VIOLIN_BOX_COLOR = "#8c8c8c"
POINT_SIZE = 7
POINT_OPACITY = 0.8
# Whether to add the left panel with the standardized corrections z = δ / σ.
SHOW_STANDARDIZED = False
# Whether to draw the distribution over all regions as violins.
SHOW_VIOLINS = True
# Points drawn on top: "aggregated" (stock-weighted mean of each aggregated region),
# "largest" (the N_LARGEST_REGIONS regions with the largest top-down stock) or "none".
POINTS = "none"
N_LARGEST_REGIONS = 30


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    figure_data = load_figure_data()
    rows = correction_rows(figure_data.corrections)
    match POINTS:
        case "aggregated":
            point_series = aggregated_point_series(rows, figure_data.td_stock)
        case "largest":
            countries = largest_stock_regions(figure_data.td_stock, N_LARGEST_REGIONS)
            point_series = country_point_series(rows, countries)
        case _:
            point_series = []
    suffix = ("_standardized" if SHOW_STANDARDIZED else "") + ("_violin" if SHOW_VIOLINS else "")
    plot_corrections(
        rows,
        point_series,
        f"fig13_reconciliation_parameters{suffix}_{POINTS}_{REGION_MAPPING_FIG13}",
        show_standardized=SHOW_STANDARDIZED,
        show_violins=SHOW_VIOLINS,
    )
    print("END")
