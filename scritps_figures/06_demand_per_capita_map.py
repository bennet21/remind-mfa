"""Figure 6: world maps of cumulative future cement demand under SSP2.

Choropleth maps in which each country is shaded by the cumulative future cement demand of its
REMIND H12 region over 2024-2100, from the reconciled (combined) MFA. Four variants are produced:
per-capita demand (annual regional demand divided by that year's population, summed over the
future years; t/cap), total demand (annual regional demand summed over the future years; Gt),
and the corresponding process CO2 emissions (clinker demand times the clinker emission factor).
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
from helpers import load_mfas

GT_PER_T = 1e-9

mfas = load_mfas(SSP_SOURCE_PICKLES["SSP2"], SSP_CACHE_DIRS["SSP2"])
combined = mfas["combined"]
td = mfas["td"]


def regional_demand():
    """Combined cement demand (t) with dims (t, r)."""
    return combined.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))


def sum_future_years(array) -> dict[str, float]:
    """Sum a (t, r) flodym array over the future years; return one value per region."""
    years = np.array(array.dims["t"].items)
    future = years > LAST_HISTORICAL_YEAR
    cumulative = array.values[future].sum(axis=0)
    return {str(region): value for region, value in zip(array.dims["r"].items, cumulative)}


def cumulative_per_capita_demand() -> dict[str, float]:
    """Sum of annual per-capita cement demand (t/cap) over the future years, per region."""
    return sum_future_years(regional_demand() / td.parameters["population"])


def cumulative_total_demand() -> dict[str, float]:
    """Cumulative cement demand (Gt) over the future years, per region."""
    return sum_future_years(regional_demand() * GT_PER_T)


def process_emissions():
    """Process CO2 emissions (t) with dims (t, r): clinker demand times the clinker
    emission factor (CaO content times CO2 released per CaO, as in the carbonation model)."""
    prm = td.parameters
    return (
        regional_demand()
        * prm["clinker_ratio"]
        * prm["clinker_cao_ratio"]
        * prm["cao_emission_factor"]
    )


def cumulative_process_emissions_per_capita() -> dict[str, float]:
    """Sum of annual per-capita process emissions (t CO2/cap) over the future years, per region."""
    return sum_future_years(process_emissions() / td.parameters["population"])


def cumulative_process_emissions_total() -> dict[str, float]:
    """Cumulative process emissions (Gt CO2) over the future years, per region."""
    return sum_future_years(process_emissions() * GT_PER_T)


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
    colorbar_title="Cumulative cement demand<br>2024–2100 (t per capita)",
    unit="t per capita",
    output_name="fig6_demand_per_capita_map",
)
plot_map(
    cumulative_total_demand(),
    colorbar_title="Cumulative cement demand<br>2024–2100 (Gt)",
    unit="Gt",
    output_name="fig6_demand_total_map",
)
plot_map(
    cumulative_process_emissions_per_capita(),
    colorbar_title="Cumulative process emissions<br>2024–2100 (t CO₂ per capita)",
    unit="t CO₂ per capita",
    output_name="fig6_process_emissions_per_capita_map",
)
plot_map(
    cumulative_process_emissions_total(),
    colorbar_title="Cumulative process emissions<br>2024–2100 (Gt CO₂)",
    unit="Gt CO₂",
    output_name="fig6_process_emissions_total_map",
)

print("END")
