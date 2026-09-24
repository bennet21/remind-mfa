"""Figure 7: cumulative future cement demand and process CO2 emissions by scenario.

Horizontal bar charts comparing the 5 SSP scenarios by their global cumulative value over
2024-2100. Each figure combines a world per-capita value (left) and the absolute value split
by aggregated region (right), for cement demand and gross process CO2 emissions.
Scenarios are ordered SSP1 (bottom) to SSP5 (top) on the y-axis.

Demand is the market cement demand (cement going into products plus construction losses) and
emissions are the calcination CO2 of the clinker and cement kiln dust needed for it, both as
defined in `helpers.cement_demand` / `helpers.process_emissions`. Emissions are therefore
attributed to the consuming region; globally the consumption- and production-based totals are
identical, because trade cancels out.

The left panel is a single bar per scenario, coloured by SSP: world demand (or emissions)
divided by world population, summed over the future years. It carries no regional breakdown,
because per-capita values of different regions cannot be added into a stack. The regional
split is in the right panel (absolute) and in figure 8 (per region, per capita).

The reduction achieved by the SSP1_CE / SSP2_CE circular-economy variants is shown as a slim
arrow plus a lightly shaded box (in a lighter tone of the bar's own colour) pointing from the
SSP1/SSP2 bar end towards the corresponding CE value. It is explained inline via a callout on
the SSP2 bar rather than a separate legend.
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/07_cumulative_demand_emissions.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    AGG_REGION_COLORS,
    AGG_REGION_ORDER,
    CE_SHADE,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    SSP_CACHE_DIRS,
    SSP_COLORS,
    SSP_LABELS,
    SSP_SOURCE_PICKLES,
)
from helpers import aggregate_by_region, cement_demand, load_mfas, process_emissions, shade

GT_PER_T = 1e-9
FIRST_FUTURE_YEAR = LAST_HISTORICAL_YEAR + 1
CE_Y_OFFSET = 0.42
CE_BOX_WIDTH = 0.16
ARROW_COLOR = "#1a1a1a"
CALLOUT_COLOR = "#555555"


def load_scenario_data() -> dict[str, dict]:
    """Load all SSP scenarios and compute global cumulative demand and emissions (2024-2100)."""
    result = {}
    for ssp, pickle_path in SSP_SOURCE_PICKLES.items():
        print(f"Loading {ssp}...")
        combined = load_mfas(pickle_path, SSP_CACHE_DIRS[ssp])["combined"]
        # All parameters come from the reconciled MFA, so that reconciled values
        # (cement_losses in particular) are the ones used here.
        prm = combined.parameters

        demand_arr = cement_demand(combined)
        emissions_arr = process_emissions(demand_arr, prm)

        years = np.array(demand_arr.dims["t"].items)
        future = years > LAST_HISTORICAL_YEAR
        region_items = np.array(demand_arr.dims["r"].items)

        future_demand = demand_arr.values[future]
        future_emissions = emissions_arr.values[future]
        future_population = prm["population"].values[future]

        # World per capita: sum over regions first, divide second. Summing regional
        # per-capita values instead would not be a per-capita quantity.
        world_population = future_population.sum(axis=1)

        result[ssp] = {
            "demand": future_demand.sum() * GT_PER_T,
            "emissions": future_emissions.sum() * GT_PER_T,
            "demand_per_capita": (future_demand.sum(axis=1) / world_population).sum(),
            "emissions_per_capita": (future_emissions.sum(axis=1) / world_population).sum(),
            "regions": {
                "demand": {
                    region: series.sum() * GT_PER_T
                    for region, series in aggregate_by_region(future_demand, region_items).items()
                },
                "emissions": {
                    region: series.sum() * GT_PER_T
                    for region, series in aggregate_by_region(
                        future_emissions, region_items
                    ).items()
                },
            },
        }
        print(
            f"  demand: {result[ssp]['demand']:.1f} Gt"
            f" ({result[ssp]['demand_per_capita']:.1f} t/cap),"
            f"  emissions: {result[ssp]['emissions']:.1f} Gt CO2"
            f" ({result[ssp]['emissions_per_capita']:.1f} t CO2/cap)"
        )
    return result


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def make_world_bar(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[go.Bar]:
    """One plain bar per scenario, coloured by SSP: the world per-capita value."""
    return [
        go.Bar(
            orientation="h",
            x=[scenario_data[ssp][key] for ssp in sorted_ssps],
            y=list(range(len(sorted_ssps))),
            xaxis=xref,
            yaxis=yref,
            marker=dict(
                color=[SSP_COLORS[ssp] for ssp in sorted_ssps],
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    ]


def make_region_bars(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[go.Bar]:
    return [
        go.Bar(
            orientation="h",
            x=[scenario_data[ssp]["regions"][key][region] for ssp in sorted_ssps],
            y=list(range(len(sorted_ssps))),
            xaxis=xref,
            yaxis=yref,
            name=region,
            marker=dict(color=AGG_REGION_COLORS[region], line=dict(color="white", width=1)),
            showlegend=False,
            hoverinfo="skip",
        )
        for region in AGG_REGION_ORDER
    ]


def make_region_callouts(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[dict]:
    """Label the top bar's regions with vertical leader lines, replacing the legend."""
    ssp = sorted_ssps[-1]
    annotations = []
    running_total = 0.0
    for region in AGG_REGION_ORDER:
        segment = scenario_data[ssp]["regions"][key][region]
        annotations.append(
            dict(
                x=running_total + segment / 2,
                y=len(sorted_ssps) - 1,
                xref=xref,
                yref=yref,
                ax=running_total + segment / 2,
                ay=-46,
                axref=xref,
                ayref="pixel",
                text=region,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=CALLOUT_COLOR,
                font=dict(size=10.5, color=CALLOUT_COLOR),
                align="center",
            )
        )
        running_total += segment
    return annotations


def make_ce_annotations(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[dict]:
    """Slim arrow from each SSP1/SSP2 bar end to its CE counterpart's value."""
    annotations = []
    for ssp in sorted_ssps:
        ce_ssp = f"{ssp}_CE"
        if ce_ssp not in scenario_data:
            continue
        annotations.append(
            dict(
                x=scenario_data[ce_ssp][key],
                y=sorted_ssps.index(ssp) + CE_Y_OFFSET - CE_BOX_WIDTH / 2,
                ax=scenario_data[ssp][key],
                ay=sorted_ssps.index(ssp) + CE_Y_OFFSET - CE_BOX_WIDTH / 2,
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.6,
                arrowcolor=ARROW_COLOR,
                text="",
            )
        )
    return annotations


def make_ce_savings_bars(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[go.Bar]:
    """CE savings as one slim shaded box per scenario, for the non-stacked world panel."""
    x, base, y, colors = [], [], [], []
    for index, ssp in enumerate(sorted_ssps):
        ce_ssp = f"{ssp}_CE"
        if ce_ssp not in scenario_data:
            continue
        x.append(scenario_data[ssp][key] - scenario_data[ce_ssp][key])
        base.append(scenario_data[ce_ssp][key])
        y.append(index + CE_Y_OFFSET)
        colors.append(shade(SSP_COLORS[ssp], CE_SHADE))
    return [
        go.Bar(
            orientation="h",
            x=x,
            base=base,
            y=y,
            xaxis=xref,
            yaxis=yref,
            width=CE_BOX_WIDTH,
            marker=dict(color=colors),
            showlegend=False,
            hoverinfo="skip",
        )
    ]


def make_ce_savings_boxes(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[go.Bar]:
    """CE savings as lightly shaded stacked segments per aggregated region."""
    traces = []
    for region in AGG_REGION_ORDER:
        x = []
        base = []
        y = []
        for index, ssp in enumerate(sorted_ssps):
            ce_ssp = f"{ssp}_CE"
            if ce_ssp not in scenario_data:
                continue
            ce_total = scenario_data[ce_ssp][key]
            prior_savings = sum(
                scenario_data[ssp]["regions"][key][prior_region]
                - scenario_data[ce_ssp]["regions"][key][prior_region]
                for prior_region in AGG_REGION_ORDER[: AGG_REGION_ORDER.index(region)]
            )
            savings = (
                scenario_data[ssp]["regions"][key][region]
                - scenario_data[ce_ssp]["regions"][key][region]
            )
            x.append(savings)
            base.append(ce_total + prior_savings)
            y.append(index + CE_Y_OFFSET)
        traces.append(
            go.Bar(
                orientation="h",
                x=x,
                base=base,
                y=y,
                xaxis=xref,
                yaxis=yref,
                width=CE_BOX_WIDTH,
                marker=dict(color=shade(AGG_REGION_COLORS[region], CE_SHADE)),
                name=region,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def make_ce_callout(
    sorted_ssps: list[str], scenario_data: dict, key: str, xref: str, yref: str
) -> list[dict]:
    """Inline callout on the highest CE-enabled bar explaining the CE savings box/arrow."""
    ce_ssps = [ssp for ssp in sorted_ssps if f"{ssp}_CE" in scenario_data]
    if not ce_ssps:
        return []
    ssp = ce_ssps[-1]
    ce_ssp = f"{ssp}_CE"
    savings_mid = (scenario_data[ce_ssp][key] + scenario_data[ssp][key]) / 2
    return [
        dict(
            x=savings_mid,
            y=sorted_ssps.index(ssp) + CE_Y_OFFSET,
            xref=xref,
            yref=yref,
            ax=savings_mid,
            # Shorter leader than the region callouts: this bar sits mid-plot, so a long
            # line would run into the row above.
            ay=-24,
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
    ]


def make_layout(
    sorted_ssps: list[str], per_capita_title: str, absolute_title: str, annotations: list[dict]
) -> dict:
    return dict(
        template="simple_white",
        font=dict(family="Arial, sans-serif", size=13, color="#222222"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text=per_capita_title, font=dict(size=13)),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=1,
            zeroline=False,
        ),
        xaxis2=dict(
            title=dict(text=absolute_title, font=dict(size=13)),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(sorted_ssps))),
            ticktext=[SSP_LABELS[ssp] for ssp in sorted_ssps],
            tickfont=dict(size=13),
            ticks="",
            range=[-0.55, len(sorted_ssps) - 0.1],
        ),
        yaxis2=dict(tickfont=dict(size=13), ticks="", showticklabels=False),
        margin=dict(l=110, r=30, t=100, b=55),
        bargap=0.42,
        barmode="stack",
        showlegend=False,
        annotations=annotations,
    )


def plot_comparison(
    sorted_ssps: list[str],
    scenario_data: dict,
    key: str,
    per_capita_title: str,
    absolute_title: str,
    output_name: str,
):
    """Plot the world per-capita value and the region-split absolute value for one flow."""
    per_capita_key = f"{key}_per_capita"
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05)
    traces = [
        *make_world_bar(sorted_ssps, scenario_data, per_capita_key, "x", "y"),
        *make_ce_savings_bars(sorted_ssps, scenario_data, per_capita_key, "x", "y"),
        *make_region_bars(sorted_ssps, scenario_data, key, "x2", "y2"),
        *make_ce_savings_boxes(sorted_ssps, scenario_data, key, "x2", "y2"),
    ]
    for trace in traces:
        fig.add_trace(trace)
    annotations = (
        make_ce_annotations(sorted_ssps, scenario_data, per_capita_key, "x", "y")
        + make_ce_callout(sorted_ssps, scenario_data, per_capita_key, "x", "y")
        + make_region_callouts(sorted_ssps, scenario_data, key, "x2", "y2")
        + make_ce_annotations(sorted_ssps, scenario_data, key, "x2", "y2")
    )
    fig.update_layout(**make_layout(sorted_ssps, per_capita_title, absolute_title, annotations))
    save(fig, output_name, width=1000, height=430)


scenario_data = load_scenario_data()
# SSP1 at the bottom through SSP5 at the top
sorted_ssps = sorted(ssp for ssp in scenario_data if not ssp.endswith("_CE"))

plot_comparison(
    sorted_ssps,
    scenario_data,
    "demand",
    f"World per capita {FIRST_FUTURE_YEAR}–2100 (t/cap)",
    f"Absolute {FIRST_FUTURE_YEAR}–2100 (Gt)",
    "fig7_cumulative_demand",
)
plot_comparison(
    sorted_ssps,
    scenario_data,
    "emissions",
    f"World per capita {FIRST_FUTURE_YEAR}–2100 (t CO₂/cap)",
    f"Absolute {FIRST_FUTURE_YEAR}–2100 (Gt CO₂)",
    "fig7_cumulative_process_emissions",
)

print("END")
