"""Figure 6: world map of cumulative future cement demand per capita under SSP2.

Choropleth map in which each country is shaded by the cumulative future per-capita cement
demand of its REMIND H12 region: for every future year (2024-2100), the regional reconciled
(combined) cement demand is divided by that year's population, and the annual per-capita
values are summed. Countries are assigned to regions via the REMIND H12 regionmapping
(`scritps_figures/h12.csv`). Saved as PNG in `data/cement/output/figures`.

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
from helpers import load_mfas

COLORBAR_TITLE = "Cumulative cement demand<br>2024–2100 (t per capita)"

mfas = load_mfas(SSP_SOURCE_PICKLES["SSP2"], SSP_CACHE_DIRS["SSP2"])
combined = mfas["combined"]
td = mfas["td"]


def cumulative_per_capita_demand() -> dict[str, float]:
    """Sum of annual per-capita cement demand (t/cap) over the future years, per region."""
    demand = combined.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))
    population = td.parameters["population"]
    per_capita = demand / population

    years = np.array(per_capita.dims["t"].items)
    future = years > LAST_HISTORICAL_YEAR
    cumulative = per_capita.values[future].sum(axis=0)
    return {str(region): value for region, value in zip(per_capita.dims["r"].items, cumulative)}


def country_table(region_values: dict[str, float]) -> pd.DataFrame:
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
        + " t per capita"
    )
    return mapping


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_map(output_name: str):
    region_values = cumulative_per_capita_demand()
    countries = country_table(region_values)

    print("Cumulative per-capita cement demand 2024-2100 (t/cap) by region:")
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
                "title": {"text": COLORBAR_TITLE, "font": {"size": 13}},
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


plot_map("fig6_demand_per_capita_map")

print("END")
