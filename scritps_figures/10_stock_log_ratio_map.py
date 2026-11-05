"""Figure 10: world map of the log ratio of the top-down to the bottom-up in-use cement stock.

Each country is shaded by ln(TD / BU) of its model region in the last historical year, before
the reconciliation. Both stocks are the concrete (product mass) in residential and commercial
buildings, the quantity compared by the reconciliation:
- top-down: in-use stock of the model run without reconciliation
- bottom-up: stock computed from floorspace with the pre-reconciliation parameters
Positive values (red) mean that the top-down stock is larger, negative values (blue) that the
bottom-up stock is larger. Values are grouped into bins whose edges are factors of TD / BU and
their reciprocals (e.g. 2 and 1/2), i.e. symmetric in ln. Three maps are produced:
residential + commercial, residential only and commercial only.

Uses the scenario MAIN_SCENARIO on the region level REGION_MAPPING_FIG10 (generate its data with
`uv run --no-sync python scritps_figures/generate_data.py --region-mapping gtap155 SSP2`).

Run from the repository root:
    uv run python scritps_figures/10_stock_log_ratio_map.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import (
    BACKGROUND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    MAIN_SCENARIO,
    MODEL_REGIONMAPPING_CSVS,
)
from helpers import load_figure_data

H = LAST_HISTORICAL_YEAR

# Region level of the map, independent of constants.REGION_MAPPING.
REGION_MAPPING_FIG10 = "gtap155"

# Inner bin edges as factors of TD / BU above 1; their reciprocals are the edges below 1. The
# outer bins are open.
FACTOR_EDGES = [1.25, 1.5, 2, 3]
FACTOR_EDGE_LABELS = [
    *[f"1/{factor:g}" for factor in reversed(FACTOR_EDGES)],
    *[f"{factor:g}" for factor in FACTOR_EDGES],
]
# Inner bin edges of ln(TD / BU), symmetric around zero.
BIN_EDGES = [-np.log(factor) for factor in reversed(FACTOR_EDGES)] + [
    np.log(factor) for factor in FACTOR_EDGES
]
# Diverging colors, one per bin (blue: BU larger, grey: about equal, red: TD larger).
BIN_COLORS = [
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#f0efec",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
]

# End-use items per map, on the top-down (u) and bottom-up (b) end-use dimensions.
STOCK_TYPES = {
    "combined": {"u": ["Res", "Com"], "b": ["RS", "RM", "Com"]},
    "Res": {"u": ["Res"], "b": ["RS", "RM"]},
    "Com": {"u": ["Com"], "b": ["Com"]},
}
STOCK_TYPE_TITLES = {
    "combined": "residential + commercial",
    "Res": "residential",
    "Com": "commercial",
}

data = load_figure_data(MAIN_SCENARIO, region_mapping=REGION_MAPPING_FIG10)
# product mass: sum of cement and non-cement constituents
td_stock = data.unreconciled.stock[{"t": H, "m": "concrete"}].sum_over("k")
bu_stock = data.reconciled.bu_concrete_stock


def sum_items(array, letter: str, items: list[str]):
    """Sum an array over the given items of dimension `letter`, leaving dims (r,)."""
    selected = [array[{letter: item}].sum_to(("r",)) for item in items]
    total = selected[0]
    for part in selected[1:]:
        total = total + part
    return total


def log_ratio(stock_type: str) -> dict[str, float]:
    """ln(TD / BU) per model region for the given stock type."""
    items = STOCK_TYPES[stock_type]
    td = sum_items(td_stock, "u", items["u"])
    bu = sum_items(bu_stock, "b", items["b"])
    values = np.log(td.values / bu.values)
    return {str(region): value for region, value in zip(td.dims["r"].items, values)}


def bin_labels() -> list[str]:
    """Legend labels of the bins as TD / BU factor ranges."""
    labels = [f"< {FACTOR_EDGE_LABELS[0]}"]
    for lower, upper in zip(FACTOR_EDGE_LABELS[:-1], FACTOR_EDGE_LABELS[1:]):
        labels.append(f"{lower} to {upper}")
    labels.append(f"> {FACTOR_EDGE_LABELS[-1]}")
    return labels


def country_table(region_values: dict[str, float]) -> pd.DataFrame:
    mapping = pd.read_csv(MODEL_REGIONMAPPING_CSVS[REGION_MAPPING_FIG10], sep=";")
    mapping = mapping[mapping["CountryCode"] != "ATA"]  # Antarctica
    mapping["value"] = mapping["RegionCode"].map(region_values)
    # countries without a model value are drawn as land without data
    mapping = mapping.dropna(subset=["value"])
    mapping["bin"] = np.digitize(mapping["value"], BIN_EDGES)
    mapping["hover"] = (
        mapping["X"]
        + " ("
        + mapping["RegionCode"]
        + ")<br>ln(TD/BU) = "
        + mapping["value"].round(2).astype(str)
        + "<br>TD/BU = "
        + np.exp(mapping["value"]).round(2).astype(str)
    )
    return mapping


def stepped_colorscale(colors: list[str]) -> list[list]:
    """Plotly colorscale with one constant color per integer bin index."""
    n_bins = len(colors)
    scale = []
    for index, color in enumerate(colors):
        scale.append([index / n_bins, color])
        scale.append([(index + 1) / n_bins, color])
    return scale


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def plot_map(stock_type: str, output_name: str):
    region_values = log_ratio(stock_type)
    countries = country_table(region_values)

    values = np.array(list(region_values.values()))
    in_center_bin = np.abs(values) < np.log(FACTOR_EDGES[0])
    print(
        f"TD/BU, {STOCK_TYPE_TITLES[stock_type]}: median {np.exp(np.median(values)):.2f}, "
        f"min {np.exp(values.min()):.2f}, max {np.exp(values.max()):.2f}, "
        f"within 1/{FACTOR_EDGES[0]:g} to {FACTOR_EDGES[0]:g} in {np.mean(in_center_bin):.0%} "
        f"of {len(values)} regions"
    )

    n_bins = len(BIN_COLORS)
    fig = go.Figure(
        go.Choropleth(
            locations=countries["CountryCode"],
            z=countries["bin"],
            zmin=-0.5,
            zmax=n_bins - 0.5,
            text=countries["hover"],
            hoverinfo="text",
            colorscale=stepped_colorscale(BIN_COLORS),
            marker_line_color="white",
            marker_line_width=0.3,
            colorbar={
                "title": {
                    "text": f"TD / BU, {STOCK_TYPE_TITLES[stock_type]}<br>"
                    f"concrete stock {H}",
                    "font": {"size": 13},
                },
                "tickvals": list(range(n_bins)),
                "ticktext": bin_labels(),
                "tickfont": {"size": 11},
                "thickness": 15,
                "len": 0.8,
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        geo={
            "projection_type": "equal earth",
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
    save(fig, output_name, width=1300, height=600)


plot_map("combined", f"fig10_stock_log_ratio_map_{REGION_MAPPING_FIG10}")
plot_map("Res", f"fig10_stock_log_ratio_map_res_{REGION_MAPPING_FIG10}")
plot_map("Com", f"fig10_stock_log_ratio_map_com_{REGION_MAPPING_FIG10}")

print("END")
