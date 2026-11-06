"""Figure 15: world map of the H12 model regions.

Each country is colored by its H12 region, using H12_REGION_COLORS (shades of the colour of the
aggregated region). Each region is named by a label placed in the sea next to it, connected to
the region by a short leader line.

Run from the repository root:
    uv run python scritps_figures/15_h12_region_map.py
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import (
    BACKGROUND,
    FIGURES_DIR,
    H12_REGION_ORDER,
    REGION_SHORT_NAMES,
    REGIONMAPPING_CSV,
)
from helpers import H12_REGION_COLORS

LINE_COLOR = "#404040"
LINE_WIDTH = 0.8
LABEL_FONT_SIZE = 13
# Points per leader line. Scattergeo connects consecutive points along great circles, so the line
# is sampled densely to follow the straight lon/lat path instead.
LINE_POINTS = 20


@dataclass
class RegionLabel:
    """Label of a region on the map.

    Attributes:
        region: H12 region code.
        position: (lon, lat) of the label end of the leader line.
        text_position: Plotly text position of the label relative to `position`.
        target: (lon, lat) point inside the region the leader line ends at.
    """

    region: str
    position: tuple[float, float]
    text_position: str
    target: tuple[float, float]


REGION_LABELS = [
    RegionLabel("USA", (-128, 30), "middle left", (-117, 35)),
    RegionLabel("CAZ", (-142, 46), "middle left", (-124, 54)),
    RegionLabel("CAZ", (108, -36), "middle left", (118, -30)),
    RegionLabel("LAM", (-85, -15), "middle left", (-70, -10)),
    RegionLabel("SSA", (2, -12), "middle left", (16, -8)),
    RegionLabel("MEA", (-20, 30), "middle left", (-5, 28)),
    RegionLabel("EUR", (-14, 46), "middle left", (0, 46)),
    RegionLabel("NEU", (-8, 60), "middle left", (8, 61)),
    RegionLabel("REF", (75, 84), "top center", (82, 70)),
    RegionLabel("CHA", (128, 25), "middle right", (114, 28)),
    RegionLabel("JPN", (150, 42), "middle right", (140, 38)),
    RegionLabel("IND", (65, 12), "middle left", (76, 18)),
    RegionLabel("OAS", (105, -13), "bottom center", (113, 0)),
]

mapping = pd.read_csv(REGIONMAPPING_CSV, sep=";")
mapping = mapping[mapping["CountryCode"] != "ATA"]  # Antarctica


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def add_region_label(fig: go.Figure, label: RegionLabel):
    """Draw the leader line from the label position to the target, then the label text."""
    fig.add_trace(
        go.Scattergeo(
            lon=np.linspace(label.position[0], label.target[0], LINE_POINTS),
            lat=np.linspace(label.position[1], label.target[1], LINE_POINTS),
            mode="lines",
            line={"color": LINE_COLOR, "width": LINE_WIDTH},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[label.position[0]],
            lat=[label.position[1]],
            mode="text",
            text=[REGION_SHORT_NAMES[label.region]],
            textposition=label.text_position,
            textfont={"size": LABEL_FONT_SIZE, "color": "black"},
            hoverinfo="skip",
            showlegend=False,
        )
    )


fig = go.Figure()
for region in H12_REGION_ORDER:
    countries = mapping[mapping["RegionCode"] == region]
    color = H12_REGION_COLORS[region]
    fig.add_trace(
        go.Choropleth(
            locations=countries["CountryCode"],
            z=[0] * len(countries),
            text=countries["X"] + " (" + region + ")",
            hoverinfo="text",
            # constant colorscale: every country of the region gets the region colour
            colorscale=[[0, color], [1, color]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.3,
            showlegend=False,
        )
    )
for label in REGION_LABELS:
    add_region_label(fig, label)
fig.update_layout(
    template="plotly_white",
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
save(fig, "fig15_h12_region_map", width=1300, height=600)

print("END")
