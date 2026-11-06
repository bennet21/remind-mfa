"""Figure 11: cumulative future process CO2 emissions by scenario, with CE waterfalls.

Vertical bars of the global gross process CO2 emissions summed over the future years
(2026-2100), split by aggregated region, as in the right panel of figure 7. Scenarios are ordered
SSP5 (left) to SSP1, with the circular-economy (CE) scenarios placed after their parent SSP:
SSP2 is followed by Pragmatic Circularity (SSP2_CE) and SSP1 by Ambitious Circularity (SSP1_CE).

Between each SSP and its CE scenario, a waterfall shows the change caused by each CE measure
(see ce_steps.py). The measures are applied cumulatively in a fixed order, grouped by the CE
aspects narrow, slow and close, which are marked by brackets below the axis. As their effects
interact, the change attributed to a measure depends on this order. A hatched step between
Pragmatic Circularity and SSP1 shows the remaining difference, caused by the SSP1 drivers
(population, GDP and floor area). Colour only encodes the regions, which are labelled left of the
SSP5 bar; the figure has no legend.

A second version splits the bars by the H12 regions instead, shown as shades of the colour of
their aggregated region.

The intermediate steps are generated with
    uv run --no-sync python scritps_figures/generate_data.py --ce-steps
Saved as PNG in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/11_ce_waterfall_emissions.py
"""

import pickle
import re
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from ce_steps import CEStepChain
from constants import (
    AGG_REGION_COLORS,
    AGG_REGION_ORDER,
    AGG_REGIONS,
    BACKGROUND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    SCENARIOS,
    SSP_LABELS,
)
from helpers import (
    ReconciledData,
    figure_cache_paths,
    load_figure_data,
    shade,
    shade_levels,
    to_h12,
)

GT_PER_T = 1e-9
FIRST_FUTURE_YEAR = LAST_HISTORICAL_YEAR + 1
# Base scenarios from left to right; a CE scenario follows its parent with its waterfall.
BASE_ORDER = ["SSP5", "SSP4", "SSP3", "SSP2", "SSP1"]
CE_OF_PARENT = {"SSP2": "SSP2_CE", "SSP1": "SSP1_CE"}
# Scenario preceded by a step for the driver difference to the scenario before it.
DRIVER_STEP_SCENARIO = "SSP1"
DRIVER_STEP_LABEL = "Driver difference"
DRIVER_ASPECT = "Drivers"
SCENARIO_BAR_WIDTH = 0.7
STEP_BAR_WIDTH = 0.5
# Space right of the last bar for the total potential arrow and its label, in bar positions.
TOTAL_POTENTIAL_SPACE = 1.9
# Savings marked by a bracket above the bars, and the total potential marked by an arrow right
# of the last bar, as (from scenario, to scenario). Relative savings refer to the first one.
SAVINGS_BRACKETS = [("SSP2", "SSP2_CE"), ("SSP1", "SSP1_CE")]
TOTAL_POTENTIAL = ("SSP2", "SSP1_CE")
# Charcoal: clearly darker than the grey of "Rest of the World" and not blue like China.
STEP_COLOR = "#404040"
CONNECTOR_COLOR = "#8c8c8c"
LABEL_COLOR = "#444444"
# Savings brackets and the total potential arrow: darker and thicker than the connectors.
SAVINGS_COLOR = "#1a1a1a"
SAVINGS_LINE_WIDTH = 1.2
TOTAL_ARROW_WIDTH = 2.2
GRID_COLOR = "#f2f2f2"

# Figure geometry in pixels, used to place the step labels and the brackets below them.
WIDTH = 1550
HEIGHT = 600
MARGIN = dict(l=75, r=20, t=30, b=55)
PLOT_HEIGHT = HEIGHT - MARGIN["t"] - MARGIN["b"]
STEP_LABEL_SHIFT = 4
STEP_LABEL_LINE_HEIGHT = 17
BRACKET_OFFSET = 8
BRACKET_TICK = 5
# Distance of the savings brackets above the highest bar they span, incl. its value label.
SAVINGS_BRACKET_OFFSET = 30
# Head room above the highest bar, as a share of its value.
Y_HEADROOM = 0.1

# Minimum vertical distance between the centres of neighbouring region labels, in pixels.
REGION_LABEL_MIN_GAP = 17


@dataclass
class RegionLevel:
    """Regions the scenario bars are split into, from bottom to top.

    Attributes:
        name: Suffix of the output file.
        labels: Label of each region, placed left of the first bar.
        colors: Colour of each region.
        of_h12: Region of each H12 region.
        label_space: Space left of the first bar for the region labels, in bar positions.
    """

    name: str
    labels: dict[str, str]
    colors: dict[str, str]
    of_h12: dict[str, str]
    label_space: float

    @property
    def order(self) -> list[str]:
        return list(self.labels)

    def aggregate(self, h12_values: dict[str, float]) -> dict[str, float]:
        """Sum values of the H12 regions to the regions of this level."""
        result = dict.fromkeys(self.order, 0.0)
        for h12_region, value in h12_values.items():
            result[self.of_h12[h12_region]] += value
        return result


AGGREGATED = RegionLevel(
    name="",
    labels={
        "Sub-Saharan Africa": "Sub-Saharan<br>Africa",
        "S & SE Asia": "S & SE Asia",
        "Rest of the World": "Rest of<br>the World",
        "China": "China",
        "OECD": "OECD",
    },
    colors=AGG_REGION_COLORS,
    of_h12=AGG_REGIONS,
    label_space=1.8,
)

# H12 regions grouped by aggregated region, in the order of AGG_REGION_ORDER.
H12_LABELS = {
    "SSA": "Sub-Saharan Africa",
    "IND": "India",
    "OAS": "Other Asia",
    "LAM": "Latin America",
    "MEA": "Middle East, N. Africa",
    "REF": "Former Soviet Union",
    "CHA": "China",
    "USA": "USA",
    "EUR": "EU 28",
    "NEU": "Non-EU Europe",
    "JPN": "Japan",
    "CAZ": "Canada, NZ, Australia",
}


def h12_colors() -> dict[str, str]:
    """Colour of each H12 region: shades of the colour of its aggregated region."""
    colors = {}
    for agg_region in AGG_REGION_ORDER:
        members = [region for region in H12_LABELS if AGG_REGIONS[region] == agg_region]
        levels = shade_levels(len(members)) if len(members) > 1 else [0.0]
        for region, level in zip(members, levels):
            colors[region] = shade(AGG_REGION_COLORS[agg_region], level)
    return colors


H12 = RegionLevel(
    name="_h12",
    labels=H12_LABELS,
    colors=h12_colors(),
    of_h12={region: region for region in H12_LABELS},
    label_space=2.7,
)


def load_reconciled(run_name: str) -> ReconciledData:
    """Reconciled figure data of a scenario or of an intermediate CE step."""
    if run_name in SCENARIOS:
        return load_figure_data(run_name).reconciled
    cache_path = figure_cache_paths(run_name)["reconciled"]
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No figure data for CE step {run_name} at {cache_path}. Run "
            "scritps_figures/generate_data.py --ce-steps first."
        )
    with cache_path.open("rb") as file_handle:
        return pickle.load(file_handle)


def cumulative_emissions_by_region(run_name: str) -> dict[str, float]:
    """Future process CO2 emissions (Gt) summed over the years, per H12 region."""
    print(f"Loading {run_name}...")
    emissions = to_h12(load_reconciled(run_name).emissions)
    future = np.array(emissions.dims["t"].items) > LAST_HISTORICAL_YEAR
    sums = emissions.values[future].sum(axis=0) * GT_PER_T
    return dict(zip(emissions.dims["r"].items, sums))


@dataclass
class ScenarioColumn:
    """A stacked bar of the cumulative emissions of a scenario, split by aggregated region."""

    label: str
    h12_regions: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.h12_regions.values())


@dataclass
class StepColumn:
    """A floating waterfall bar from the total before to the total after a step."""

    label: str
    aspect: str
    before: float
    after: float

    @property
    def is_driver_step(self) -> bool:
        return self.aspect == DRIVER_ASPECT


class CEWaterfallFigure:
    """Bar chart of the cumulative process emissions with waterfalls from SSPs to CE scenarios."""

    def __init__(self):
        self.columns: list[ScenarioColumn | StepColumn] = []
        self.regions = AGGREGATED
        for ssp in BASE_ORDER:
            scenario = ScenarioColumn(SSP_LABELS[ssp], cumulative_emissions_by_region(ssp))
            if ssp == DRIVER_STEP_SCENARIO:
                before = self.columns[-1].total
                self.columns.append(
                    StepColumn(DRIVER_STEP_LABEL, DRIVER_ASPECT, before, scenario.total)
                )
            self.columns.append(scenario)
            if ssp in CE_OF_PARENT:
                self._add_waterfall(CEStepChain(CE_OF_PARENT[ssp]))

    def _add_waterfall(self, chain: CEStepChain):
        totals = [self.columns[-1].total]
        for step_number, step in enumerate(chain.steps, start=1):
            regions = cumulative_emissions_by_region(chain.run_name(step_number))
            totals.append(sum(regions.values()))
            self.columns.append(StepColumn(step.label, step.aspect, totals[-2], totals[-1]))
        self.columns.append(ScenarioColumn(SSP_LABELS[chain.ce_scenario], regions))
        print(
            f"  {chain.ce_scenario}: "
            + ", ".join(
                f"{step.label} {after - before:+.1f}"
                for step, before, after in zip(chain.steps, totals[:-1], totals[1:])
            )
            + " Gt CO2"
        )

    def _positions(self, column_type: type) -> list[int]:
        return [index for index, column in enumerate(self.columns) if isinstance(column, column_type)]

    def region_values(self, column: ScenarioColumn) -> dict[str, float]:
        return self.regions.aggregate(column.h12_regions)

    def region_traces(self) -> list[go.Bar]:
        positions = self._positions(ScenarioColumn)
        values = [self.region_values(self.columns[index]) for index in positions]
        return [
            go.Bar(
                x=positions,
                y=[column_values[region] for column_values in values],
                width=SCENARIO_BAR_WIDTH,
                marker=dict(color=self.regions.colors[region], line=dict(color="white", width=1)),
                hoverinfo="skip",
            )
            for region in self.regions.order
        ]

    def step_traces(self) -> list[go.Bar]:
        """CE steps as solid charcoal bars, the driver step as a hatched bar."""
        traces = []
        for is_driver_step in (False, True):
            positions = [
                index
                for index in self._positions(StepColumn)
                if self.columns[index].is_driver_step == is_driver_step
            ]
            steps = [self.columns[index] for index in positions]
            if is_driver_step:
                marker = dict(
                    color="white",
                    line=dict(color=STEP_COLOR, width=1.2),
                    pattern=dict(shape="/", fgcolor=STEP_COLOR, size=6, solidity=0.25),
                )
            else:
                marker = dict(color=STEP_COLOR, line=dict(width=0))
            traces.append(
                go.Bar(
                    x=positions,
                    y=[abs(step.before - step.after) for step in steps],
                    base=[min(step.before, step.after) for step in steps],
                    width=STEP_BAR_WIDTH,
                    marker=marker,
                    hoverinfo="skip",
                )
            )
        return traces

    def connector_shapes(self) -> list[dict]:
        """Dotted lines joining the level after each bar to the next bar of the waterfall."""
        shapes = []
        for index, (left, right) in enumerate(zip(self.columns[:-1], self.columns[1:])):
            if not isinstance(right, StepColumn) and not isinstance(left, StepColumn):
                continue
            level = left.after if isinstance(left, StepColumn) else left.total
            shapes.append(
                dict(
                    type="line",
                    x0=index + self._half_width(left),
                    x1=index + 1 - self._half_width(right),
                    y0=level,
                    y1=level,
                    line=dict(color=CONNECTOR_COLOR, width=1, dash="dot"),
                )
            )
        return shapes

    @staticmethod
    def _half_width(column: ScenarioColumn | StepColumn) -> float:
        return (STEP_BAR_WIDTH if isinstance(column, StepColumn) else SCENARIO_BAR_WIDTH) / 2

    def aspect_groups(self) -> list[tuple[str, int, int]]:
        """Consecutive CE steps of the same aspect as (aspect, first index, last index)."""
        groups = []
        for index, column in enumerate(self.columns):
            if not isinstance(column, StepColumn) or column.is_driver_step:
                continue
            if groups and groups[-1][0] == column.aspect and groups[-1][2] == index - 1:
                groups[-1] = (column.aspect, groups[-1][1], index)
            else:
                groups.append((column.aspect, index, index))
        return groups

    def _waterfall_of(self, index: int) -> range:
        """Indices of the CE steps of the waterfall containing the step at `index`."""
        is_ce_step = [
            isinstance(column, StepColumn) and not column.is_driver_step for column in self.columns
        ]
        first, last = index, index
        while first > 0 and is_ce_step[first - 1]:
            first -= 1
        while last < len(self.columns) - 1 and is_ce_step[last + 1]:
            last += 1
        return range(first, last + 1)

    def bracket_level(self, index: int) -> float:
        """Height (Gt) of the aspect brackets of the waterfall containing the step at `index`,
        just below its lowest step label. All brackets of a waterfall share this height."""
        label_bottoms = []
        for step_index in self._waterfall_of(index):
            column = self.columns[step_index]
            n_lines = wrap(column.label).count("<br>") + 1
            label_height = STEP_LABEL_SHIFT + STEP_LABEL_LINE_HEIGHT * n_lines
            label_bottoms.append(
                min(column.before, column.after) - self._pixels_to_data(label_height)
            )
        return min(label_bottoms) - self._pixels_to_data(BRACKET_OFFSET)

    def bracket_shapes(self) -> list[dict]:
        """A bracket below the step labels spanning the steps of each CE aspect."""
        shapes = []
        for _, first, last in self.aspect_groups():
            y_bar = self.bracket_level(first)
            y_tick = y_bar + self._pixels_to_data(BRACKET_TICK)
            x0, x1 = first - STEP_BAR_WIDTH / 2, last + STEP_BAR_WIDTH / 2
            shapes.append(
                dict(
                    type="path",
                    path=f"M {x0},{y_tick} L {x0},{y_bar} L {x1},{y_bar} L {x1},{y_tick}",
                    line=dict(color=LABEL_COLOR, width=1),
                    fillcolor="rgba(0,0,0,0)",
                )
            )
        return shapes

    def bracket_annotations(self) -> list[dict]:
        return [
            dict(
                x=(first + last) / 2,
                y=self.bracket_level(first),
                yanchor="top",
                yshift=-3,
                text=f"<i>{aspect}</i>",
                showarrow=False,
                font=dict(size=14, color=LABEL_COLOR),
            )
            for aspect, first, last in self.aspect_groups()
        ]

    def region_annotations(self) -> list[dict]:
        """Region names left of the first bar, pointing at the middle of each segment. Labels of
        thin segments are moved apart so that they do not overlap."""
        segments = np.array(list(self.region_values(self.columns[0]).values()))
        middles = np.cumsum(segments) - segments / 2
        label_positions = spread(middles, self._pixels_to_data(REGION_LABEL_MIN_GAP))
        return [
            dict(
                x=-SCENARIO_BAR_WIDTH / 2,
                y=middle,
                ax=-24,
                ay=(middle - label_position) / self._pixels_to_data(1),
                xanchor="right",
                text=self.regions.labels[region],
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=CONNECTOR_COLOR,
                standoff=3,
                align="right",
                font=dict(size=14, color=LABEL_COLOR),
            )
            for region, middle, label_position in zip(
                self.regions.order, middles, label_positions
            )
        ]

    def value_annotations(self) -> list[dict]:
        """Totals above the scenario bars and changes above the step bars."""
        annotations = []
        for index, column in enumerate(self.columns):
            if isinstance(column, ScenarioColumn):
                y, text = column.total, f"<b>{column.total:.0f}</b>"
            else:
                change = column.after - column.before
                y, text = max(column.before, column.after), f"{change:+.1f}".replace("-", "−")
            annotations.append(
                dict(
                    x=index,
                    y=y,
                    yshift=9,
                    text=text,
                    showarrow=False,
                    font=dict(size=13, color=LABEL_COLOR),
                )
            )
        return annotations

    def tick_labels(self) -> list[str]:
        """Scenario names on the axis; the steps are labelled below their bars instead."""
        return [
            f"<b>{column.label.replace(' ', '<br>')}</b>" if isinstance(column, ScenarioColumn) else ""
            for column in self.columns
        ]

    def step_label_annotations(self) -> list[dict]:
        """Name of each step directly below its bar."""
        annotations = []
        for index, column in enumerate(self.columns):
            if not isinstance(column, StepColumn):
                continue
            text = wrap(column.label)
            annotations.append(
                dict(
                    x=index,
                    y=min(column.before, column.after),
                    yanchor="top",
                    yshift=-STEP_LABEL_SHIFT,
                    text=f"<i>{text}</i>" if column.is_driver_step else text,
                    showarrow=False,
                    font=dict(size=13, color=LABEL_COLOR),
                )
            )
        return annotations

    def _scenario_index(self, ssp: str) -> int:
        label = SSP_LABELS[ssp]
        return next(
            index
            for index, column in enumerate(self.columns)
            if isinstance(column, ScenarioColumn) and column.label == label
        )

    @property
    def y_max(self) -> float:
        return max(column.total for column in self.columns if isinstance(column, ScenarioColumn)) * (
            1 + Y_HEADROOM
        )

    def _pixels_to_data(self, pixels: float) -> float:
        return pixels * self.y_max / PLOT_HEIGHT

    def _top(self, column: ScenarioColumn | StepColumn) -> float:
        return column.total if isinstance(column, ScenarioColumn) else max(column.before, column.after)

    @staticmethod
    def savings_text(before: float, after: float) -> str:
        """Absolute and relative change from `before` to `after`."""
        return f"−{before - after:.0f} Gt CO₂ (−{(before - after) / before:.0%})"

    def savings_brackets(self) -> tuple[list[dict], list[dict]]:
        """Shapes and labels of the brackets spanning each SSP-to-CE waterfall."""
        shapes, annotations = [], []
        for start_ssp, end_ssp in SAVINGS_BRACKETS:
            first, last = self._scenario_index(start_ssp), self._scenario_index(end_ssp)
            top = max(self._top(column) for column in self.columns[first : last + 1])
            y_bar = top + self._pixels_to_data(SAVINGS_BRACKET_OFFSET)
            y_tick = y_bar - self._pixels_to_data(BRACKET_TICK)
            x0, x1 = first - SCENARIO_BAR_WIDTH / 2, last + SCENARIO_BAR_WIDTH / 2
            shapes.append(
                dict(
                    type="path",
                    path=f"M {x0},{y_tick} L {x0},{y_bar} L {x1},{y_bar} L {x1},{y_tick}",
                    line=dict(color=SAVINGS_COLOR, width=SAVINGS_LINE_WIDTH),
                    opacity=1,
                    fillcolor="rgba(0,0,0,0)",
                )
            )
            annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=y_bar,
                    yanchor="bottom",
                    yshift=2,
                    text=self.savings_text(
                        self.columns[first].total, self.columns[last].total
                    ),
                    showarrow=False,
                    font=dict(size=14, color=SAVINGS_COLOR),
                )
            )
        return shapes, annotations

    def total_potential(self) -> tuple[list[dict], list[dict]]:
        """A downward arrow right of the last bar for the total potential, with short end caps."""
        start, end = (self.columns[self._scenario_index(ssp)] for ssp in TOTAL_POTENTIAL)
        end_index = self._scenario_index(TOTAL_POTENTIAL[1])
        x_arrow = end_index + SCENARIO_BAR_WIDTH / 2 + 0.3
        shapes = [
            dict(
                type="line",
                x0=x_arrow - 0.12,
                x1=x_arrow + 0.12,
                y0=level,
                y1=level,
                line=dict(color=SAVINGS_COLOR, width=SAVINGS_LINE_WIDTH),
                opacity=1,
            )
            for level in (start.total, end.total)
        ]
        annotations = [
            dict(
                x=x_arrow,
                y=end.total,
                ax=x_arrow,
                ay=start.total,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=TOTAL_ARROW_WIDTH,
                arrowcolor=SAVINGS_COLOR,
            ),
            dict(
                x=x_arrow + 0.12,
                y=(start.total + end.total) / 2,
                xanchor="left",
                align="left",
                text=(
                    "<b>Total potential</b><br>"
                    + self.savings_text(start.total, end.total).replace(" (", "<br>(")
                ),
                showarrow=False,
                font=dict(size=14, color=SAVINGS_COLOR),
            ),
        ]
        return shapes, annotations

    def plot(self) -> go.Figure:
        fig = go.Figure([*self.region_traces(), *self.step_traces()])
        savings_shapes, savings_annotations = self.savings_brackets()
        total_shapes, total_annotations = self.total_potential()
        fig.update_layout(
            template="simple_white",
            font=dict(family="Arial, sans-serif", size=16, color="#222222"),
            barmode="stack",
            showlegend=False,
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.columns))),
                ticktext=self.tick_labels(),
                ticks="",
                tickangle=0,
                range=[
                    -0.5 - self.regions.label_space,
                    len(self.columns) - 0.5 + TOTAL_POTENTIAL_SPACE,
                ],
            ),
            yaxis=dict(
                title=dict(text=f"Process emissions {FIRST_FUTURE_YEAR}–2100 (Gt CO₂)"),
                showgrid=True,
                gridcolor=GRID_COLOR,
                range=[0, self.y_max],
            ),
            margin=MARGIN,
            shapes=(
                self.connector_shapes()
                + self.bracket_shapes()
                + savings_shapes
                + total_shapes
            ),
            annotations=(
                self.value_annotations()
                + self.region_annotations()
                + self.step_label_annotations()
                + self.bracket_annotations()
                + savings_annotations
                + total_annotations
            ),
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=BACKGROUND,
        )
        return fig


def wrap(label: str, max_line_length: int = 10) -> str:
    """Break a label into lines of at most `max_line_length` characters, at spaces and after
    slashes."""
    lines = [""]
    for piece in re.findall(r"[^ /]+/?", label):
        separator = "" if not lines[-1] or lines[-1].endswith("/") else " "
        if lines[-1] and len(lines[-1]) + len(separator) + len(piece) > max_line_length:
            lines.append(piece)
        else:
            lines[-1] = f"{lines[-1]}{separator}{piece}"
    return "<br>".join(lines)


def spread(positions: np.ndarray, min_gap: float) -> np.ndarray:
    """Move ascending positions apart until neighbours are at least `min_gap` apart, shifting
    each group of crowded positions symmetrically around its mean."""
    offsets = min_gap * np.arange(len(positions))
    # Pool adjacent violators: merge groups whose offset-free positions are out of order.
    groups: list[tuple[float, int]] = []
    for value in positions - offsets:
        groups.append((value, 1))
        while len(groups) > 1 and groups[-2][0] > groups[-1][0]:
            (last_value, last_size), (previous_value, previous_size) = groups.pop(), groups.pop()
            size = last_size + previous_size
            groups.append(((last_value * last_size + previous_value * previous_size) / size, size))
    return np.repeat([value for value, _ in groups], [size for _, size in groups]) + offsets


def save(fig: go.Figure, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


figure = CEWaterfallFigure()
for region_level in (AGGREGATED, H12):
    figure.regions = region_level
    save(
        figure.plot(),
        f"fig11_ce_waterfall_process_emissions{region_level.name}",
        width=WIDTH,
        height=HEIGHT,
    )

print("END")
