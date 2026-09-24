"""Figure 7: cumulative future cement demand and process CO2 emissions by scenario.

Horizontal bar charts comparing the 5 SSP scenarios by their global cumulative value
over 2024–2100. Two variants are produced: total cement demand (Gt) and gross process
CO2 emissions (Gt CO2, from clinker production). Scenarios are ordered SSP1 (bottom) to
SSP5 (top) on the y-axis.
The reduction achieved by the SSP1_CE / SSP2_CE circular-economy variants is shown as
an arrow pointing from the SSP1/SSP2 bar end towards the corresponding CE value.
Vertical tick marks show each bar's cumulative value up to the years in
CUMULATIVE_MARKER_YEARS (constants.py).
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/07_cumulative_demand_emissions.py
"""

import numpy as np
import plotly.graph_objects as go

from constants import (
    CUMULATIVE_MARKER_YEARS,
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
            "marker_demand": {},
            "marker_emissions": {},
        }
        for year in CUMULATIVE_MARKER_YEARS:
            up_to_year = future & (years <= year)
            result[ssp]["marker_demand"][year] = demand_arr.values[up_to_year].sum() * GT_PER_T
            result[ssp]["marker_emissions"][year] = emissions_arr.values[up_to_year].sum() * GT_PER_T
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
        marker=dict(color=[SSP_COLORS[ssp] for ssp in sorted_ssps]),
    )


def make_marker_trace(sorted_ssps: list[str], scenario_data: dict, key: str) -> go.Scatter:
    """Vertical tick marks at each bar's cumulative value up to the CUMULATIVE_MARKER_YEARS."""
    xs = [
        scenario_data[ssp][f"marker_{key}"][year]
        for ssp in sorted_ssps
        for year in CUMULATIVE_MARKER_YEARS
    ]
    ys = [SSP_LABELS[ssp] for ssp in sorted_ssps for _ in CUMULATIVE_MARKER_YEARS]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(symbol="line-ns", size=24, line=dict(width=2, color="black")),
        hoverinfo="skip",
        showlegend=False,
    )


def make_marker_annotations(sorted_ssps: list[str], scenario_data: dict, key: str) -> list[dict]:
    """Year labels for the marker ticks, placed below the bottom-most bar only."""
    bottom_ssp = sorted_ssps[0]
    return [
        dict(
            x=scenario_data[bottom_ssp][f"marker_{key}"][year],
            y=SSP_LABELS[bottom_ssp],
            text=str(year),
            showarrow=False,
            yshift=-28,
            font=dict(size=11, color="black"),
        )
        for year in CUMULATIVE_MARKER_YEARS
    ]


def make_ce_annotations(sorted_ssps: list[str], scenario_data: dict, key: str) -> list[dict]:
    """Arrow from each SSP1/SSP2 bar end to its CE counterpart's value."""
    annotations = []
    for ssp in sorted_ssps:
        ce_ssp = f"{ssp}_CE"
        if ce_ssp not in scenario_data:
            continue
        annotations.append(
            dict(
                x=scenario_data[ce_ssp][key],
                y=SSP_LABELS[ssp],
                ax=scenario_data[ssp][key],
                ay=SSP_LABELS[ssp],
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


def make_layout(xaxis_title: str, annotations: list[dict]) -> dict:
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
        annotations=annotations,
    )


def plot_demand(sorted_ssps: list[str], scenario_data: dict):
    fig = go.Figure(
        [make_bar(sorted_ssps, scenario_data, "demand"), make_marker_trace(sorted_ssps, scenario_data, "demand")]
    )
    annotations = make_ce_annotations(sorted_ssps, scenario_data, "demand") + make_marker_annotations(
        sorted_ssps, scenario_data, "demand"
    )
    fig.update_layout(**make_layout("Cumulative cement demand 2024–2100 (Gt)", annotations))
    save(fig, "fig7_cumulative_demand", width=700, height=450)


def plot_emissions(sorted_ssps: list[str], scenario_data: dict):
    fig = go.Figure(
        [
            make_bar(sorted_ssps, scenario_data, "emissions"),
            make_marker_trace(sorted_ssps, scenario_data, "emissions"),
        ]
    )
    annotations = make_ce_annotations(sorted_ssps, scenario_data, "emissions") + make_marker_annotations(
        sorted_ssps, scenario_data, "emissions"
    )
    fig.update_layout(
        **make_layout("Cumulative process CO₂ emissions 2024–2100 (Gt CO₂)", annotations)
    )
    save(fig, "fig7_cumulative_process_emissions", width=700, height=450)


scenario_data = load_scenario_data()
# SSP1 at the bottom through SSP5 at the top
sorted_ssps = sorted(ssp for ssp in scenario_data if not ssp.endswith("_CE"))

plot_demand(sorted_ssps, scenario_data)
plot_emissions(sorted_ssps, scenario_data)

print("END")
