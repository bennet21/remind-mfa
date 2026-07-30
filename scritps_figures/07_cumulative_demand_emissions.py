"""Figure 7: cumulative future cement demand and process CO2 emissions by scenario.

Horizontal bar charts comparing all 7 SSP scenarios (SSP1–5 plus SSP1_CE and SSP2_CE)
by their global cumulative value over 2024–2100. Two variants are produced: total cement
demand (Gt) and gross process CO2 emissions (Gt CO2, from clinker production). Scenarios
are sorted ascending by cumulative emissions so the lowest-emission scenario sits at the
bottom of the y-axis. CE variants are visually distinguished by diagonal hatching.
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/07_cumulative_demand_emissions.py
"""

import numpy as np
import plotly.graph_objects as go

from constants import (
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import load_mfas

GT_PER_T = 1e-9


def load_scenario_data() -> dict[str, dict[str, float]]:
    """Load all SSP scenarios and compute global cumulative demand and emissions (2024–2100)."""
    result = {}
    for ssp, pickle_path in SSP_SOURCE_PICKLES.items():
        print(f"Loading {ssp}...")
        mfas = load_mfas(pickle_path, SSP_CACHE_DIRS[ssp])
        combined, td = mfas["combined"], mfas["td"]
        prm = td.parameters

        demand_arr = combined.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))
        emissions_arr = (
            demand_arr
            * prm["clinker_ratio"]
            * prm["clinker_cao_ratio"]
            * prm["cao_emission_factor"]
        )

        years = np.array(demand_arr.dims["t"].items)
        future = years > LAST_HISTORICAL_YEAR

        result[ssp] = {
            "demand": demand_arr.values[future].sum() * GT_PER_T,
            "emissions": emissions_arr.values[future].sum() * GT_PER_T,
        }
        print(f"  demand: {result[ssp]['demand']:.1f} Gt,  emissions: {result[ssp]['emissions']:.1f} Gt CO2")
    return result


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def make_bar(sorted_ssps: list[str], scenario_data: dict, key: str) -> go.Bar:
    return go.Bar(
        orientation="h",
        x=[scenario_data[ssp][key] for ssp in sorted_ssps],
        y=[SSP_LABELS[ssp] for ssp in sorted_ssps],
        marker=dict(
            color=[SSP_COLORS[ssp] for ssp in sorted_ssps],
            pattern=dict(
                shape=["/" if ssp.endswith("_CE") else "" for ssp in sorted_ssps],
                fgcolor="white",
                size=5,
            ),
        ),
    )


def make_layout(xaxis_title: str) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=15)),
            tickfont=dict(size=13),
        ),
        yaxis=dict(tickfont=dict(size=13)),
        margin=dict(l=110, r=40, t=30, b=60),
        bargap=0.35,
        showlegend=False,
    )


def plot_demand(sorted_ssps: list[str], scenario_data: dict):
    fig = go.Figure(make_bar(sorted_ssps, scenario_data, "demand"))
    fig.update_layout(**make_layout("Cumulative cement demand 2024–2100 (Gt)"))
    save(fig, "fig7_cumulative_demand", width=700, height=450)


def plot_emissions(sorted_ssps: list[str], scenario_data: dict):
    fig = go.Figure(make_bar(sorted_ssps, scenario_data, "emissions"))
    fig.update_layout(**make_layout("Cumulative process CO₂ emissions 2024–2100 (Gt CO₂)"))
    save(fig, "fig7_cumulative_process_emissions", width=700, height=450)


scenario_data = load_scenario_data()
sorted_ssps = sorted(scenario_data, key=lambda s: scenario_data[s]["emissions"])

plot_demand(sorted_ssps, scenario_data)
plot_emissions(sorted_ssps, scenario_data)

print("END")
