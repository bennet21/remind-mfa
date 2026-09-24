"""Figure 8: cumulative process CO2 emissions by aggregated region, one figure per scenario.

For each of the 5 SSP scenarios, a two-panel horizontal bar chart shows the 5 aggregated
regions. The left panel is per-capita cumulative emissions (t CO2/cap); the right panel is
absolute cumulative emissions (Gt CO2). Each bar is split into a gray historical segment
(FIRST_MODEL_YEAR-LAST_HISTORICAL_YEAR) and a colored future segment (2024-2100). For SSP1
and SSP2, the reduction achieved by the corresponding _CE circular-economy variant is shown
as a slim arrow (below its box) plus a lightly shaded box per region. The left panel carries
inline callouts explaining historical/future/CE savings; the right panel instead marks the
cumulative value up to CUMULATIVE_MARKER_YEAR_HISTORICAL and each of CUMULATIVE_MARKER_YEARS
(constants.py) with star markers, to avoid repeating the same explanation twice.
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/08_regional_cumulative_emissions.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    AGG_REGION_COLORS,
    AGG_REGION_ORDER,
    AGG_REGIONS,
    CUMULATIVE_MARKER_YEAR_HISTORICAL,
    CUMULATIVE_MARKER_YEARS,
    FIGURES_DIR,
    FIRST_MODEL_YEAR,
    HISTORICAL_COLOR,
    LAST_HISTORICAL_YEAR,
    SSP_CACHE_DIRS,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas, shade

GT_PER_T = 1e-9
REGION_COLORS = AGG_REGION_COLORS
ARROW_COLOR = "#1a1a1a"
CALLOUT_COLOR = "#555555"
MARKER_COLOR = "#222222"


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


def make_star_trace(regional_data: dict, xref: str, yref: str) -> go.Scatter:
    """Slim vertical tick marks at each region's cumulative value up to the marker years."""
    marker_years = [CUMULATIVE_MARKER_YEAR_HISTORICAL, *CUMULATIVE_MARKER_YEARS]
    xs = [regional_data[r]["markers"][year] for r in AGG_REGION_ORDER for year in marker_years]
    ys = [r for r in AGG_REGION_ORDER for _ in marker_years]
    return go.Scatter(
        x=xs,
        y=ys,
        xaxis=xref,
        yaxis=yref,
        mode="markers",
        marker=dict(symbol="line-ns", size=13, line=dict(width=1.4, color=MARKER_COLOR)),
        hoverinfo="skip",
        showlegend=False,
    )


def make_star_annotations(regional_data: dict, xref: str, yref: str) -> list[dict]:
    """Year labels for the marker ticks, placed above the top-most region only."""
    top_region = AGG_REGION_ORDER[-1]
    marker_years = [CUMULATIVE_MARKER_YEAR_HISTORICAL, *CUMULATIVE_MARKER_YEARS]
    return [
        dict(
            x=regional_data[top_region]["markers"][year],
            y=top_region,
            xref=xref,
            yref=yref,
            text=str(year),
            showarrow=False,
            yshift=13,
            font=dict(size=10.5, color=MARKER_COLOR),
        )
        for year in marker_years
    ]


def make_ce_annotations(regional_data: dict, ce_data: dict, xref: str, yref: str) -> list[dict]:
    """Slim arrow from each region's bar end to its CE counterpart's value, below its box."""
    annotations = []
    for r in AGG_REGION_ORDER:
        hist_val = regional_data[r]["hist"]
        annotations.append(
            dict(
                x=hist_val + ce_data[r]["future"],
                y=r,
                ax=hist_val + regional_data[r]["future"],
                ay=r,
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                yshift=-11,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.6,
                arrowcolor=ARROW_COLOR,
                text="",
            )
        )
    return annotations


def make_ce_savings_box(regional_data: dict, ce_data: dict, xref: str, yref: str) -> go.Bar:
    """Lightly shaded box spanning each region's CE savings, in a lighter tone of its own color."""
    x = []
    base = []
    y = []
    colors = []
    for r in AGG_REGION_ORDER:
        hist_val = regional_data[r]["hist"]
        start = hist_val + ce_data[r]["future"]
        end = hist_val + regional_data[r]["future"]
        x.append(end - start)
        base.append(start)
        y.append(r)
        colors.append(shade(REGION_COLORS[r], 0.55))
    return go.Bar(
        orientation="h",
        x=x,
        base=base,
        y=y,
        xaxis=xref,
        yaxis=yref,
        marker=dict(color=colors),
        showlegend=False,
        hoverinfo="skip",
    )


def make_topbar_callouts(regional_data: dict, ce_data: dict | None, xref: str, yref: str) -> list[dict]:
    """Inline callouts on the top-most region's bar explaining historical/future/CE savings."""
    top_region = AGG_REGION_ORDER[-1]
    hist_val = regional_data[top_region]["hist"]
    future_val = regional_data[top_region]["future"]

    callouts = [
        dict(
            x=hist_val / 2,
            y=top_region,
            xref=xref,
            yref=yref,
            ax=hist_val / 2,
            ay=-58,
            axref=xref,
            ayref="pixel",
            text=f"Historical<br>({FIRST_MODEL_YEAR}\u2013{LAST_HISTORICAL_YEAR})",
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=CALLOUT_COLOR,
            font=dict(size=10.5, color=CALLOUT_COLOR),
            align="center",
        ),
        dict(
            x=hist_val + future_val / 2,
            y=top_region,
            xref=xref,
            yref=yref,
            ax=hist_val + future_val / 2,
            ay=-58,
            axref=xref,
            ayref="pixel",
            text=f"Future<br>({LAST_HISTORICAL_YEAR + 1}\u20132100)",
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=CALLOUT_COLOR,
            font=dict(size=10.5, color=CALLOUT_COLOR),
            align="center",
        ),
    ]
    if ce_data:
        savings_mid = hist_val + (ce_data[top_region]["future"] + future_val) / 2
        callouts.append(
            dict(
                x=savings_mid,
                y=top_region,
                xref=xref,
                yref=yref,
                ax=savings_mid,
                ay=-58,
                axref=xref,
                ayref="pixel",
                text="CE savings",
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=CALLOUT_COLOR,
                font=dict(size=10.5, color=CALLOUT_COLOR),
                align="center",
            )
        )
    return callouts


def make_panel_bars(regional_data: dict, xref: str, yref: str) -> list[go.Bar]:
    hist_bar = go.Bar(
        orientation="h",
        x=[regional_data[r]["hist"] for r in AGG_REGION_ORDER],
        y=AGG_REGION_ORDER,
        xaxis=xref,
        yaxis=yref,
        marker=dict(color=HISTORICAL_COLOR, line=dict(color="white", width=1)),
        showlegend=False,
    )
    future_bar = go.Bar(
        orientation="h",
        x=[regional_data[r]["future"] for r in AGG_REGION_ORDER],
        y=AGG_REGION_ORDER,
        xaxis=xref,
        yaxis=yref,
        marker=dict(color=[REGION_COLORS[r] for r in AGG_REGION_ORDER], line=dict(color="white", width=1)),
        showlegend=False,
    )
    return [hist_bar, future_bar]


def plot_scenario(ssp: str, regional_data: dict, percapita_data: dict, ce_data: dict | None, ce_percapita_data: dict | None):
    """Two-panel figure: per-capita (left, with callouts) and absolute (right, with period markers)."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05)

    traces = make_panel_bars(percapita_data, "x", "y")
    annotations = make_topbar_callouts(percapita_data, ce_percapita_data, "x", "y")
    if ce_percapita_data:
        traces.append(make_ce_savings_box(percapita_data, ce_percapita_data, "x", "y"))
        annotations += make_ce_annotations(percapita_data, ce_percapita_data, "x", "y")

    traces += make_panel_bars(regional_data, "x2", "y2")
    if ce_data:
        traces.append(make_ce_savings_box(regional_data, ce_data, "x2", "y2"))
        annotations += make_ce_annotations(regional_data, ce_data, "x2", "y2")
    traces.append(make_star_trace(regional_data, "x2", "y2"))
    annotations += make_star_annotations(regional_data, "x2", "y2")

    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial, sans-serif", size=13, color="#222222"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=SSP_LABELS[ssp], font=dict(size=16), x=0, xanchor="left"),
        xaxis=dict(
            title=dict(text="Per capita 1900\u20132100 (t CO\u2082/cap)", font=dict(size=13)),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=1,
            zeroline=False,
        ),
        xaxis2=dict(
            title=dict(text="Absolute 1900\u20132100 (Gt CO\u2082)", font=dict(size=13)),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=12), ticks=""),
        yaxis2=dict(tickfont=dict(size=12), ticks="", showticklabels=False),
        margin=dict(l=140, r=30, t=100, b=55),
        bargap=0.42,
        barmode="stack",
        showlegend=False,
        annotations=annotations,
    )
    save(fig, f"fig8_regional_cumulative_emissions_{ssp}", width=1000, height=430)


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

    plot_scenario(ssp, regional_data, percapita_data, ce_data, ce_percapita_data)

print("END")
