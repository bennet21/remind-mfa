"""Figure 9: global cement demand stacked by aggregated region, one figure per scenario.

Stacked areas show market cement demand (`helpers.cement_demand`: cement going into products
plus construction losses, consumption-based) split over the 5 aggregated regions
(constants.AGG_REGION_ORDER), coloured by constants.AGG_REGION_COLORS. Global only: unlike
figures 1/2, a per-region breakdown of a single region's own demand by region doesn't make
sense, so there is no regional variant here.

Run from the repository root:
    uv run python scritps_figures/09_demand_by_region.py
"""

import numpy as np
import plotly.graph_objects as go

from constants import (
    AGG_REGION_COLORS,
    AGG_REGION_ORDER,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    SSP_CACHE_DIRS,
    SSP_SOURCE_PICKLES,
)
from helpers import aggregate_by_region, cement_demand, load_mfas

MT_PER_T = 1e-6
YLABEL = "Cement demand (Mt)"

# Which scenarios to plot: "all", or a list of keys from constants.SSP_SOURCE_PICKLES,
# e.g. ["SSP2"].
SCENARIOS = "all"

_ALL_SCENARIOS = list(SSP_SOURCE_PICKLES.keys())


def selected_scenarios() -> list[str]:
    if SCENARIOS == "all":
        return _ALL_SCENARIOS
    missing = set(SCENARIOS) - set(_ALL_SCENARIOS)
    if missing:
        raise ValueError(f"Unknown scenario(s): {sorted(missing)}")
    return list(SCENARIOS)


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_global(time, demand_by_region: dict, output_name: str):
    fig = go.Figure()

    for region in reversed(AGG_REGION_ORDER):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=demand_by_region[region],
                mode="lines",
                stackgroup="region",
                line={"color": AGG_REGION_COLORS[region], "width": 0.3},
                fillcolor=AGG_REGION_COLORS[region],
                name=region,
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
        legend={"font": {"size": 11}},
        margin={"l": 90, "r": 30},
    )
    fig.update_xaxes(title_font_size=15)
    fig.update_yaxes(title_font_size=15)

    save(fig, output_name, width=1050, height=620)


def run(ssp: str):
    print(f"--- Scenario {ssp} ---")
    combined = load_mfas(SSP_SOURCE_PICKLES[ssp], SSP_CACHE_DIRS[ssp])["combined"]

    demand_arr = cement_demand(combined) * MT_PER_T
    time = demand_arr.dims["t"].items
    region_items = np.array(demand_arr.dims["r"].items)
    demand_by_region = aggregate_by_region(demand_arr.values, region_items)

    plot_global(time, demand_by_region, f"fig9_demand_by_region_{ssp}")


for _ssp in selected_scenarios():
    run(_ssp)

print("END")
