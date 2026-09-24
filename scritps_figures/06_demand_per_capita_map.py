"""Figure 6: world maps of cumulative future cement demand under SSP2.

Choropleth maps in which each country is shaded by the cumulative future cement demand of its
REMIND H12 region over 2024-2100, from the reconciled (combined) MFA. Four variants are produced:
per-capita demand, total demand (Gt), and the corresponding gross process CO2 emissions.

Demand is the region's market cement demand (cement going into products plus construction
losses) and emissions are the calcination CO2 of the clinker and cement kiln dust needed for
it, both as defined in `helpers.cement_demand` / `helpers.process_emissions`. Emissions are
therefore attributed to the consuming region.

The per-capita variants sum the annual ratio (that year's regional value divided by that
year's regional population) over the future years. That is an average annual rate multiplied
by the number of years, not the cumulative total divided by a single population figure.

Countries are assigned to regions via the REMIND H12 regionmapping (`scritps_figures/h12.csv`).
Saved as PNGs in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/06_demand_per_capita_map.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import (
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    REGION_DISPLAY_NAMES,
    REGIONMAPPING_CSV,
    SSP_CACHE_DIRS,
    SSP_SOURCE_PICKLES,
)
from helpers import cement_demand, load_mfas, process_emissions

GT_PER_T = 1e-9
FIRST_FUTURE_YEAR = LAST_HISTORICAL_YEAR + 1

mfas = load_mfas(SSP_SOURCE_PICKLES["SSP2"], SSP_CACHE_DIRS["SSP2"])
combined = mfas["combined"]
# All parameters are taken from the reconciled MFA, so that reconciled values
# (cement_losses in particular) are the ones used here.
prm = combined.parameters

demand = cement_demand(combined)
emissions = process_emissions(demand, prm)


def sum_future_years(array) -> dict[str, float]:
    """Sum a (t, r) flodym array over the future years; return one value per region."""
    years = np.array(array.dims["t"].items)
    future = years > LAST_HISTORICAL_YEAR
    cumulative = array.values[future].sum(axis=0)
    return {str(region): value for region, value in zip(array.dims["r"].items, cumulative)}


def cumulative_per_capita_demand() -> dict[str, float]:
    """Sum of annual per-capita cement demand (t/cap) over the future years, per region."""
    return sum_future_years(demand / prm["population"])


def cumulative_total_demand() -> dict[str, float]:
    """Cumulative cement demand (Gt) over the future years, per region."""
    return sum_future_years(demand * GT_PER_T)


def cumulative_process_emissions_per_capita() -> dict[str, float]:
    """Sum of annual per-capita process emissions (t CO2/cap) over the future years, per region."""
    return sum_future_years(emissions / prm["population"])


def cumulative_process_emissions_total() -> dict[str, float]:
    """Cumulative process emissions (Gt CO2) over the future years, per region."""
    return sum_future_years(emissions * GT_PER_T)


def country_table(region_values: dict[str, float], unit: str) -> pd.DataFrame:
    mapping = pd.read_csv(REGIONMAPPING_CSV, sep=";")
    mapping = mapping[mapping["CountryCode"] != "ATA"]  # Antarctica (nominally LAM)
    mapping["value"] = mapping["RegionCode"].map(region_values)
    region_names = {code: name.replace("<br>", " ") for code, name in REGION_DISPLAY_NAMES.items()}
    mapping["hover"] = (
        mapping["RegionCode"].map(region_names).fillna(mapping["RegionCode"])
        + " ("
        + mapping["RegionCode"]
        + ")<br>"
        + mapping["value"].round(1).astype(str)
        + f" {unit}"
    )
    return mapping


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_map(region_values: dict[str, float], colorbar_title: str, unit: str, output_name: str):
    countries = country_table(region_values, unit)

    # ASCII-safe console output (Windows consoles may not encode the CO2 subscript / en dash).
    label = colorbar_title.replace("<br>", " ").replace("₂", "2").replace("–", "-")
    print(f"{label} by region:")
    for region, value in sorted(region_values.items(), key=lambda item: -item[1]):
        print(f"  {region}: {value:.1f}")

    fig = go.Figure(
        go.Choropleth(
            locations=countries["CountryCode"],
            z=countries["value"],
            text=countries["hover"],
            hoverinfo="text",
            colorscale="YlOrRd",
            marker_line_color="white",
            marker_line_width=0.3,
            colorbar={
                "title": {"text": colorbar_title, "font": {"size": 13}},
                "thickness": 15,
                "len": 0.7,
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        geo={
            "projection_type": "natural earth",
            "showframe": False,
            "showcoastlines": False,
            "landcolor": "#e6e6e6",
            "showland": True,
            "bgcolor": "rgba(0,0,0,0)",
            # Crop the (unmapped) Antarctic land mass.
            "lataxis": {"range": [-58, 88]},
        },
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    save(fig, output_name, width=1200, height=650)


plot_map(
    cumulative_per_capita_demand(),
    colorbar_title=f"Cumulative cement demand<br>{FIRST_FUTURE_YEAR}–2100 (t per capita)",
    unit="t per capita",
    output_name="fig6_demand_per_capita_map",
)
plot_map(
    cumulative_total_demand(),
    colorbar_title=f"Cumulative cement demand<br>{FIRST_FUTURE_YEAR}–2100 (Gt)",
    unit="Gt",
    output_name="fig6_demand_total_map",
)
plot_map(
    cumulative_process_emissions_per_capita(),
    colorbar_title=f"Cumulative process emissions<br>{FIRST_FUTURE_YEAR}–2100 (t CO₂ per capita)",
    unit="t CO₂ per capita",
    output_name="fig6_process_emissions_per_capita_map",
)
plot_map(
    cumulative_process_emissions_total(),
    colorbar_title=f"Cumulative process emissions<br>{FIRST_FUTURE_YEAR}–2100 (Gt CO₂)",
    unit="Gt CO₂",
    output_name="fig6_process_emissions_total_map",
)

print("END")
