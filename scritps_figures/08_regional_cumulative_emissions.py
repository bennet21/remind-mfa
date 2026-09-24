"""Figure 8: cumulative process CO2 emissions by aggregated region, one figure per scenario.

For each of the 5 SSP scenarios, a horizontal bar chart shows the 5 aggregated regions,
each bar split into a gray historical segment (up to and including LAST_HISTORICAL_YEAR)
and a colored future segment (2024-2100). For SSP1 and SSP2, the reduction achieved by the
corresponding _CE circular-economy variant is shown as an arrow per region.
Two variants are produced: total cumulative emissions (Gt CO2) and cumulative emissions
per capita (t CO2/cap, annual per-capita emissions summed over the years in each segment).
The total-emissions variant also shows vertical tick marks for the cumulative value up to
CUMULATIVE_MARKER_YEAR_HISTORICAL and each of CUMULATIVE_MARKER_YEARS (constants.py).
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/08_regional_cumulative_emissions.py
"""

import numpy as np
import plotly.graph_objects as go

from constants import (
    AGG_REGION_ORDER,
    AGG_REGIONS,
    COLOR_PALETTE_3,
    CUMULATIVE_MARKER_YEAR_HISTORICAL,
    CUMULATIVE_MARKER_YEARS,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    SSP_CACHE_DIRS,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas

GT_PER_T = 1e-9
HISTORICAL_COLOR = "#a0a0a0"
REGION_COLORS = dict(zip(AGG_REGION_ORDER, COLOR_PALETTE_3))


def aggregate_by_region(values: np.ndarray, region_items: np.ndarray) -> dict[str, np.ndarray]:
    """Sum a (t, r) array's source regions into aggregated regions; return one time series each."""
    result = {agg_region: None for agg_region in AGG_REGION_ORDER}
    for source_region, agg_region in AGG_REGIONS.items():
        mask = region_items == source_region
        contribution = values[:, mask].sum(axis=1)
        result[agg_region] = (
            contribution if result[agg_region] is None else result[agg_region] + contribution
        )
    return result


def load_regional_emissions(ssp: str) -> dict[str, dict[str, float]]:
    """Cumulative process CO2 emissions (Gt) per aggregated region, split hist/future."""
    mfas = load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])
    combined, td = mfas["combined"], mfas["td"]
    prm = td.parameters

    demand_arr = combined.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))
    emissions_arr = (
        demand_arr * prm["clinker_ratio"] * prm["clinker_cao_ratio"] * prm["cao_emission_factor"]
    )

    years = np.array(emissions_arr.dims["t"].items)
    future = years > LAST_HISTORICAL_YEAR
    hist = ~future
    region_items = np.array(emissions_arr.dims["r"].items)

    emissions_by_region = aggregate_by_region(emissions_arr.values, region_items)

    result = {}
    for agg_region, series in emissions_by_region.items():
        result[agg_region] = {
            "hist": series[hist].sum() * GT_PER_T,
            "future": series[future].sum() * GT_PER_T,
            "markers": {
                year: series[years <= year].sum() * GT_PER_T
                for year in [CUMULATIVE_MARKER_YEAR_HISTORICAL, *CUMULATIVE_MARKER_YEARS]
            },
        }
    return result


def load_regional_emissions_per_capita(ssp: str) -> dict[str, dict[str, float]]:
    """Cumulative per-capita process CO2 emissions (t/cap) per aggregated region, hist/future.

    Computed as the sum, over the years in each segment, of that year's regional emissions
    divided by that year's regional population.
    """
    mfas = load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])
    combined, td = mfas["combined"], mfas["td"]
    prm = td.parameters

    demand_arr = combined.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))
    emissions_arr = (
        demand_arr * prm["clinker_ratio"] * prm["clinker_cao_ratio"] * prm["cao_emission_factor"]
    )
    population_arr = prm["population"]

    years = np.array(emissions_arr.dims["t"].items)
    future = years > LAST_HISTORICAL_YEAR
    hist = ~future
    region_items = np.array(emissions_arr.dims["r"].items)

    emissions_by_region = aggregate_by_region(emissions_arr.values, region_items)
    population_by_region = aggregate_by_region(population_arr.values, region_items)

    result = {}
    for agg_region in AGG_REGION_ORDER:
        per_capita = emissions_by_region[agg_region] / population_by_region[agg_region]
        result[agg_region] = {
            "hist": per_capita[hist].sum(),
            "future": per_capita[future].sum(),
        }
    return result


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def make_marker_trace(regional_data: dict) -> go.Scatter:
    """Vertical tick marks at each region's cumulative value up to the marker years."""
    marker_years = [CUMULATIVE_MARKER_YEAR_HISTORICAL, *CUMULATIVE_MARKER_YEARS]
    xs = [regional_data[r]["markers"][year] for r in AGG_REGION_ORDER for year in marker_years]
    ys = [r for r in AGG_REGION_ORDER for _ in marker_years]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(symbol="line-ns", size=24, line=dict(width=2, color="black")),
        hoverinfo="skip",
        showlegend=False,
    )


def make_marker_annotations(regional_data: dict) -> list[dict]:
    """Year labels for the marker ticks, placed below the bottom-most region only."""
    bottom_region = AGG_REGION_ORDER[0]
    marker_years = [CUMULATIVE_MARKER_YEAR_HISTORICAL, *CUMULATIVE_MARKER_YEARS]
    return [
        dict(
            x=regional_data[bottom_region]["markers"][year],
            y=bottom_region,
            text=str(year),
            showarrow=False,
            yshift=-28,
            font=dict(size=11, color="black"),
        )
        for year in marker_years
    ]


def make_ce_annotations(regional_data: dict, ce_data: dict) -> list[dict]:
    """Arrow from each region's bar end to its CE counterpart's value."""
    annotations = []
    for r in AGG_REGION_ORDER:
        hist_val = regional_data[r]["hist"]
        annotations.append(
            dict(
                x=hist_val + ce_data[r]["future"],
                y=r,
                ax=hist_val + regional_data[r]["future"],
                ay=r,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="black",
                text="",
            )
        )
    return annotations


def plot_scenario(
    ssp: str,
    regional_data: dict,
    xaxis_title: str,
    output_prefix: str,
    ce_data: dict | None = None,
    show_markers: bool = False,
):
    hist_bar = go.Bar(
        orientation="h",
        x=[regional_data[r]["hist"] for r in AGG_REGION_ORDER],
        y=AGG_REGION_ORDER,
        marker=dict(color=HISTORICAL_COLOR),
        name=f"Historical (until {LAST_HISTORICAL_YEAR})",
    )
    future_bar = go.Bar(
        orientation="h",
        x=[regional_data[r]["future"] for r in AGG_REGION_ORDER],
        y=AGG_REGION_ORDER,
        marker=dict(color=[REGION_COLORS[r] for r in AGG_REGION_ORDER]),
        name=f"Future ({LAST_HISTORICAL_YEAR + 1}\u20132100)",
    )
    traces = [hist_bar, future_bar]
    annotations = make_ce_annotations(regional_data, ce_data) if ce_data else []
    if show_markers:
        traces.append(make_marker_trace(regional_data))
        annotations += make_marker_annotations(regional_data)
    fig = go.Figure(traces)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=SSP_LABELS[ssp], font=dict(size=16)),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=14)),
            tickfont=dict(size=12),
        ),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=140, r=40, t=50, b=60),
        bargap=0.35,
        barmode="stack",
        showlegend=False,
        annotations=annotations,
    )
    save(fig, f"{output_prefix}_{ssp}", width=700, height=400)


base_ssps = [ssp for ssp in SSP_SOURCE_PICKLES if not ssp.endswith("_CE")]

for ssp in base_ssps:
    print(f"Loading {ssp}...")
    regional_data = load_regional_emissions(ssp)
    percapita_data = load_regional_emissions_per_capita(ssp)

    ce_ssp = f"{ssp}_CE"
    ce_data = None
    ce_percapita_data = None
    if ce_ssp in SSP_SOURCE_PICKLES:
        print(f"Loading {ce_ssp}...")
        ce_data = load_regional_emissions(ce_ssp)
        ce_percapita_data = load_regional_emissions_per_capita(ce_ssp)

    plot_scenario(
        ssp,
        regional_data,
        "Cumulative process CO₂ emissions 1900–2100 (Gt CO₂)",
        "fig8_regional_cumulative_emissions",
        ce_data,
        show_markers=True,
    )
    plot_scenario(
        ssp,
        percapita_data,
        "Cumulative process CO₂ emissions per capita 1900–2100 (t CO₂/cap)",
        "fig8_regional_cumulative_emissions_percapita",
        ce_percapita_data,
    )

print("END")
