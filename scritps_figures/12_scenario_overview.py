"""Figure 12: scenario parameters by scenario, and the circularity depth of each scenario.

Left: table of the scenario parameters that differ between the scenarios, read from the scenario
CSV files (config/scenarios) along each scenario's inheritance chain. Factors are shown as the
relative change they reach in the given year, targets (→) as the value they reach, minimum shares
(≥) as the lower bound. Driver rows (d) name the SSP dataset used for population, GDP and floor
area. Values set for single regions only are not shown. The lifetime of new age-cohorts is
adjusted per region after reading the scenario factor (longer for regions with short lifetimes,
see `CementModel.get_lifetime_factor_cohort`), so its row shows the range actually applied, taken
from the model runs. Planned measures that are not implemented yet are marked as TODO.

Right: circularity depth of each scenario against its SSP. For each CE parameter, the global
value in 2100 is the population-weighted (SSP2, last historical year) mean of the regional
values of the model run. It is normalized to 0 for SSP2 and 1 for Ambitious Circularity. The
normalized values are averaged within each CE measure of ce_steps.CE_STEPS and then across the
measures, so each measure has equal weight. Planned TODO values enter the circularity depth
instead of the values of the model run. The right panel is also saved on its own.

The table lists the CE scenarios first, followed by SSP1 to SSP5.
Saved as PNG in `data/cement/output/figures`.

Run from the repository root:
    uv run python scritps_figures/12_scenario_overview.py
"""

from dataclasses import dataclass

import flodym as fd
import numpy as np
import plotly.graph_objects as go

from ce_steps import CE_STEPS, SCENARIOS_PATH
from constants import (
    BACKGROUND,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    MAIN_SCENARIO,
    SCENARIOS,
    SSP_COLORS,
    SSP_LABELS,
    figure_cache_dir,
    source_pickle,
)
from helpers import load_cached

from remind_mfa.common.helpers import ModelNames
from remind_mfa.common.scenarios import ScenarioReader

# Base scenarios from left to right, preceded by the CE scenarios in the table.
BASE_ORDER = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
CE_PARENT = {"SSP1_CE": "SSP1", "SSP2_CE": "SSP2"}
COLUMN_ORDER = list(CE_PARENT) + BASE_ORDER
# Scenarios defining 0 and 1 of the circularity depth.
DEPTH_ZERO = MAIN_SCENARIO
DEPTH_ONE = "SSP1_CE"
DEPTH_YEAR = 2100

DRIVER_PREFIX = "d"
MINUS = "−"
LABEL_COLOR = "#444444"
RULE_COLOR = "#222222"
SEPARATOR_COLOR = "#d9d9d9"
STRIPE_COLOR = "#f3f3f3"
AXIS_COLOR = "#444444"
TODO_COLOR = "#b2182b"
# Half diagonal of the diamond outline of CE markers in the circularity depth panel, in pixels.
CE_MARKER_HALF_SIZE = 10

# Figure geometry in pixels. The table is drawn with paper-referenced shapes and annotations.
LABEL_WIDTH = 205
YEAR_WIDTH = 50
SCENARIO_WIDTH = 90
HEADER_HEIGHT = 52
ROW_HEIGHT = 25
TABLE_FONT_SIZE = 14
TODO_FONT_SIZE = 11
BASELINE_WIDTH = 160
BASELINE_FONT_SIZE = 13
TABLE_RIGHT = LABEL_WIDTH + YEAR_WIDTH + SCENARIO_WIDTH * len(COLUMN_ORDER) + BASELINE_WIDTH
# Left end of the circularity depth panel incl. its y-axis title and ticks, and its width.
PANEL_LEFT = TABLE_RIGHT + 20
PANEL_AXIS_OFFSET = 35
PANEL_WIDTH = 200
# Space below the table for the x-axis tick labels of the panel.
TICK_SPACE = 35
MARGIN = dict(l=10, r=15, t=20, b=5)


@dataclass(frozen=True)
class TableRow:
    """One row of the parameter table.

    Attributes:
        label: Display name.
        parameter: Scenario parameter shown in the row.
        kind: How the values are formatted: "driver" (SSP dataset), "factor" (relative change),
            "target" (value reached), "target_share" (share reached), "min_share" (lower bound)
            or "applied_lifetime" (range of the regionally adjusted lifetime factor of new
            age-cohorts).
    """

    label: str
    parameter: str
    kind: str


@dataclass(frozen=True)
class RowGroup:
    label: str
    rows: tuple[TableRow, ...]


ROW_GROUPS = [
    RowGroup(
        "Model-wide",
        (
            TableRow("Population", "driver_scen", "driver"),
            TableRow("Trade volume", "trade_factor", "factor"),
            TableRow("Lifetime, all cohorts (nurture)", "lifetime_factor_nurture", "factor"),
            TableRow("Lifetime, new cohorts (nature)", "lifetime_factor_nature", "applied_lifetime"),
            TableRow("Clinker ratio", "clinker_ratio", "target"),
            TableRow("Cement content", "cement_ratio", "factor"),
            TableRow("Concrete reuse", "reuse_share", "target_share"),
        ),
    ),
    RowGroup(
        "Bottom-up",
        (
            TableRow("Floor area", "driver_scen", "driver"),
            TableRow("Material intensity", "concrete_building_mi", "factor"),
            TableRow("Timber share of new floor area", "min_timber_share", "min_share"),
            TableRow("Unoccupied residential stock", "unoccupied_residential_stock_share",
                     "target_share"),
        ),
    ),
    RowGroup(
        "Top-down",
        (
            TableRow("GDP per capita", "driver_scen", "driver"),
            TableRow("Stock per capita", "stock_factor", "factor"),
        ),
    ),
]

N_TABLE_ROWS = sum(len(group.rows) + 1 for group in ROW_GROUPS)
TABLE_BOTTOM = HEADER_HEIGHT + N_TABLE_ROWS * ROW_HEIGHT
WIDTH = MARGIN["l"] + PANEL_LEFT + PANEL_WIDTH + MARGIN["r"]
HEIGHT = MARGIN["t"] + TABLE_BOTTOM + TICK_SPACE + MARGIN["b"]
PLOT_WIDTH = WIDTH - MARGIN["l"] - MARGIN["r"]
PLOT_HEIGHT = HEIGHT - MARGIN["t"] - MARGIN["b"]

# Parameters that take effect through the target of another parameter, whose year applies.
YEAR_OF = {"min_timber_share": "structure_split"}

# Behaviour of the model if no scenario sets a parameter (BASE.csv and the model code), shown
# in the last column. Driver rows are left empty, as BASE sets no driver scenario.
BASELINE_DESCRIPTIONS = {
    "trade_factor": "∝ demand",
    "lifetime_factor_nurture": "current level",
    "lifetime_factor_nature": "current level",
    "clinker_ratio": "current level",
    "cement_ratio": "current level",
    "reuse_share": "none",
    "concrete_building_mi": "current level",
    "min_timber_share": "→ global mean (low GDP)",
    "unoccupied_residential_stock_share": "current; China →10%",
    "stock_factor": "saturation curve",
}


@dataclass(frozen=True)
class TodoCell:
    """Planned value of a table cell that is not implemented yet.

    Attributes:
        text: Cell text.
        value: Planned global value of the row's parameter, used for the circularity depth.
    """

    text: str
    value: float | None = None


TODO_CELLS = {
    ("Floor area", "SSP1_CE"): TodoCell(f"{DRIVER_PREFIX}SSP1_CE"),
    ("Timber share of new floor area", "SSP1"): TodoCell("≥10%", 0.1),
}


@dataclass
class ParameterSetting:
    """Effective setting of a scenario parameter after applying a scenario's inheritance chain.

    Attributes:
        value: Value set for all regions, or None if not set.
        year: Year in which an extrapolated parameter reaches its value.
    """

    value: float | str | None = None
    year: int | None = None


class ScenarioSettings:
    """Scenario parameters of a scenario, as set for all regions in the scenario CSV files."""

    def __init__(self, scenario: str):
        reader = ScenarioReader(
            name=scenario,
            base_path=str(SCENARIOS_PATH),
            model=ModelNames.CEMENT,
            dims=fd.DimensionSet(dim_list=[]),
            parameter_definitions=[],
        )
        chain = []
        name = scenario
        while name is not None:
            single = reader.read_single(name)
            single.filter_data_by_model(ModelNames.CEMENT)
            chain.insert(0, single)
            name = single.parent

        self.settings: dict[str, ParameterSetting] = {}
        for single in chain:
            for data_point in single.data:
                # values of single regions are not shown
                if "Region" in data_point.index:
                    continue
                setting = self.settings.setdefault(data_point.parameter, ParameterSetting())
                setting.value = data_point.value
                setting.year = data_point.extra.get("year", setting.year)

    def get(self, parameter: str) -> ParameterSetting:
        return self.settings.get(parameter, ParameterSetting())


def extract_scenario_parameters(model) -> dict[str, fd.FlodymArray]:
    """Parameters of a model run needed for the applied lifetimes and the circularity depth."""
    names = {name for step in CE_STEPS for name in step.parameters} - {"min_timber_share"}
    arrays = {name: model.parameters[name] for name in names}
    arrays["min_timber_share"] = model.scenario_parameters["min_timber_share"]
    arrays["lifetime_factor_cohort"] = model.parameters["lifetime_factor_cohort"]
    arrays["population"] = model.parameters["population"]
    return arrays


def load_scenario_parameters(ssp: str) -> dict[str, fd.FlodymArray]:
    return load_cached(
        source_pickle(ssp),
        figure_cache_dir(ssp) / "scenario_parameters.pickle",
        extract_scenario_parameters,
    )


def planned_value(ssp: str, parameter: str) -> float | None:
    """Planned global value of a TODO cell of `parameter` in `ssp`, if any."""
    for group in ROW_GROUPS:
        for row in group.rows:
            todo = TODO_CELLS.get((row.label, ssp))
            if row.parameter == parameter and todo is not None and todo.value is not None:
                return todo.value
    return None


class ScenarioData:
    """Scenario settings from the CSV files and parameters from the model runs."""

    def __init__(self):
        self.settings = {ssp: ScenarioSettings(ssp) for ssp in SCENARIOS}
        self.parameters = {ssp: load_scenario_parameters(ssp) for ssp in SCENARIOS}
        population = self.parameters[DEPTH_ZERO]["population"][{"t": LAST_HISTORICAL_YEAR}]
        self.region_weights = population.values / population.values.sum()

    def applied_lifetime_range(self, ssp: str, year: int) -> tuple[float, float]:
        """Smallest and largest lifetime factor of new age-cohorts over regions and end uses."""
        values = self.parameters[ssp]["lifetime_factor_cohort"][{"t": year}].values
        return float(values.min()), float(values.max())

    def global_value(self, ssp: str, parameter: str) -> float:
        """Planned value of a TODO cell, else the population-weighted mean of the regional values
        in `DEPTH_YEAR`. Dimensions other than region and time are averaged without weights."""
        planned = planned_value(ssp, parameter)
        if planned is not None:
            return planned
        array = self.parameters[ssp][parameter]
        if "t" in array.dims.letters:
            array = array[{"t": DEPTH_YEAR}]
        n_per_region = array.values.size / array.dims["r"].len
        regional = array.sum_to(("r",)).values / n_per_region
        return float((regional * self.region_weights).sum())

    def circularity_depth(self, ssp: str) -> float:
        measure_scores = []
        for step in CE_STEPS:
            parameters = [
                "lifetime_factor_cohort" if name == "lifetime_factor_nature" else name
                for name in step.parameters
            ]
            scores = []
            for name in parameters:
                zero = self.global_value(DEPTH_ZERO, name)
                one = self.global_value(DEPTH_ONE, name)
                scores.append((self.global_value(ssp, name) - zero) / (one - zero))
            measure_scores.append(np.mean(scores))
        return float(np.mean(measure_scores))


def signed_percent(value: float) -> str:
    return f"+{value:.0f}%" if value >= 0 else f"{MINUS}{-value:.0f}%"


class CellFormatter:
    """Text of the table cells. Unchanged parameters (BASE default) are left empty."""

    def __init__(self, data: ScenarioData):
        self.data = data

    def cell(self, row: TableRow, ssp: str) -> str:
        todo = TODO_CELLS.get((row.label, ssp))
        if todo is not None:
            return (
                f"<span style='color:{TODO_COLOR};font-size:{TODO_FONT_SIZE}px'>"
                f"<i>TODO: {todo.text}</i></span>"
            )
        setting = self.data.settings[ssp].get(row.parameter)
        if row.kind == "driver":
            return f"{DRIVER_PREFIX}{setting.value}"
        if row.kind == "applied_lifetime":
            return self._applied_lifetime(setting, ssp)
        return self._format(row.kind, setting.value)

    @staticmethod
    def _format(kind: str, value: float | str | None) -> str:
        if kind == "factor":
            return "" if value in (None, 1) else signed_percent((value - 1) * 100)
        if kind == "target":
            return "" if value is None else f"→{value:g}"
        if kind == "target_share":
            return "" if value in (None, 0) else f"→{value * 100:.0f}%"
        if kind == "min_share":
            return "" if value in (None, 0) else f"≥{value * 100:.0f}%"
        raise ValueError(f"Unknown row kind {kind}.")

    def _applied_lifetime(self, setting: ParameterSetting, ssp: str) -> str:
        if setting.value in (None, 1):
            return ""
        low, high = self.data.applied_lifetime_range(ssp, setting.year)
        return f"+{(low - 1) * 100:.0f}…{(high - 1) * 100:.0f}%"

    def year(self, row: TableRow) -> str:
        if row.kind == "driver":
            return ""
        years = sorted(
            {
                settings.get(YEAR_OF.get(row.parameter, row.parameter)).year
                for settings in self.data.settings.values()
                if settings.get(row.parameter).value is not None
            }
            - {None}
        )
        return "/".join(str(year) for year in years)


class ScenarioOverviewFigure:
    panel_left = PANEL_LEFT
    # Whether the circularity depth increases downwards.
    flipped = False
    # Font size of tick labels, scenario labels and the axis title (+2).
    font_size = TABLE_FONT_SIZE
    width = WIDTH
    height = HEIGHT
    panel_height = TABLE_BOTTOM

    def __init__(self, data: ScenarioData | None = None):
        self.data = ScenarioData() if data is None else data
        self.formatter = CellFormatter(self.data)
        self.depths = {ssp: self.data.circularity_depth(ssp) for ssp in COLUMN_ORDER}

    def _x(self, pixels: float) -> float:
        return pixels / (self.width - MARGIN["l"] - MARGIN["r"])

    def _y(self, pixels_from_top: float) -> float:
        return 1 - pixels_from_top / (self.height - MARGIN["t"] - MARGIN["b"])

    @staticmethod
    def _column_center(index: int) -> float:
        return LABEL_WIDTH + YEAR_WIDTH + SCENARIO_WIDTH * (index + 0.5)

    def _text(self, text: str, x: float, y: float, xanchor: str = "center", **font) -> dict:
        return dict(
            text=text,
            x=self._x(x),
            y=self._y(y),
            xref="paper",
            yref="paper",
            xanchor=xanchor,
            yanchor="middle",
            showarrow=False,
            font=dict(size=TABLE_FONT_SIZE, color=LABEL_COLOR) | font,
        )

    def _rule(self, y: float, color: str = RULE_COLOR, width: float = 1.0) -> dict:
        return dict(
            type="line",
            xref="paper",
            yref="paper",
            x0=0,
            x1=self._x(TABLE_RIGHT),
            y0=self._y(y),
            y1=self._y(y),
            line=dict(color=color, width=width),
        )

    def table(self) -> tuple[list[dict], list[dict]]:
        shapes = self._header_color_marks()
        annotations = self._header_annotations()
        shapes += [self._rule(0, width=1.5), self._rule(HEADER_HEIGHT)]

        y = HEADER_HEIGHT
        for group_index, group in enumerate(ROW_GROUPS):
            if group_index > 0:
                shapes.append(self._rule(y, color=SEPARATOR_COLOR))
            annotations.append(
                self._text(f"<b>{group.label}</b>", 0, y + ROW_HEIGHT / 2, "left",
                           color=RULE_COLOR)
            )
            y += ROW_HEIGHT
            for row_index, row in enumerate(group.rows):
                if row_index % 2 == 0:
                    shapes.append(self._stripe(y))
                center = y + ROW_HEIGHT / 2
                annotations.append(self._text(row.label, 0, center, "left"))
                annotations.append(
                    self._text(self.formatter.year(row), LABEL_WIDTH + YEAR_WIDTH / 2, center)
                )
                for index, ssp in enumerate(COLUMN_ORDER):
                    annotations.append(
                        self._text(self.formatter.cell(row, ssp), self._column_center(index),
                                   center)
                    )
                annotations.append(
                    self._text(BASELINE_DESCRIPTIONS.get(row.parameter, ""), self._baseline_center,
                               center, size=BASELINE_FONT_SIZE)
                )
                y += ROW_HEIGHT
        shapes.append(self._rule(TABLE_BOTTOM, width=1.5))
        return shapes, annotations

    def _stripe(self, top: float) -> dict:
        """Shaded background of a table row."""
        return dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            x1=self._x(TABLE_RIGHT),
            y0=self._y(top + ROW_HEIGHT),
            y1=self._y(top),
            fillcolor=STRIPE_COLOR,
            line=dict(width=0),
            layer="below",
        )

    def _header_color_marks(self) -> list[dict]:
        """Short bar in the scenario colour below each scenario name, linking the columns to
        the markers of the circularity depth panel."""
        y = HEADER_HEIGHT - 6
        return [
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=self._x(self._column_center(index) - 18),
                x1=self._x(self._column_center(index) + 18),
                y0=self._y(y),
                y1=self._y(y),
                line=dict(color=SSP_COLORS[ssp], width=3,
                          dash="dot" if ssp in CE_PARENT else "solid"),
            )
            for index, ssp in enumerate(COLUMN_ORDER)
        ]

    def _header_annotations(self) -> list[dict]:
        center = (HEADER_HEIGHT - 10) / 2
        annotations = [
            self._text("<b>Parameter</b>", 0, center, "left", color=RULE_COLOR),
            self._text("<b>Year</b>", LABEL_WIDTH + YEAR_WIDTH / 2, center, color=RULE_COLOR),
        ]
        for index, ssp in enumerate(COLUMN_ORDER):
            label = SSP_LABELS[ssp].replace(" ", "<br>")
            annotations.append(
                self._text(f"<b>{label}</b>", self._column_center(index), center,
                           color=RULE_COLOR)
            )
        annotations.append(
            self._text("<b>Baseline</b>", self._baseline_center, center, color=RULE_COLOR)
        )
        return annotations

    @property
    def _baseline_center(self) -> float:
        return TABLE_RIGHT - BASELINE_WIDTH / 2

    @staticmethod
    def _position(ssp: str) -> int:
        return BASE_ORDER.index(CE_PARENT.get(ssp, ssp))

    def depth_traces(self) -> list[go.Scatter]:
        traces = []
        for ssp in COLUMN_ORDER:
            is_ce = ssp in CE_PARENT
            traces.append(
                go.Scatter(
                    x=[self._position(ssp)],
                    y=[self.depths[ssp]],
                    mode="markers",
                    marker=dict(
                        symbol="diamond" if is_ce else "circle",
                        size=16 if is_ce else 14,
                        color="white" if is_ce else SSP_COLORS[ssp],
                        # outline of CE markers is drawn dashed by `depth_shapes`
                        line=dict(width=0),
                    ),
                    hoverinfo="skip",
                )
            )
        return traces

    def depth_shapes(self) -> list[dict]:
        """Dashed outline of the CE markers, as in the colour marks of the table header."""
        half = CE_MARKER_HALF_SIZE
        return [
            dict(
                type="path",
                xref="x", yref="y",
                xanchor=self._position(ssp), yanchor=self.depths[ssp],
                xsizemode="pixel", ysizemode="pixel",
                path=f"M 0,{half} L {half},0 L 0,{-half} L {-half},0 Z",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=SSP_COLORS[ssp], width=2.5, dash="6px,3px"),
                opacity=1,
            )
            for ssp in CE_PARENT
        ]

    def depth_annotations(self) -> list[dict]:
        annotations = []
        # gap between arrow and markers, constant in pixels
        gap = 0.04 * TABLE_BOTTOM / self.panel_height
        for ssp, parent in CE_PARENT.items():
            x = self._position(ssp)
            annotations.append(
                dict(
                    x=x, y=self.depths[ssp] - gap, ax=x, ay=self.depths[parent] + gap,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                    arrowcolor=SSP_COLORS[ssp], text="",
                )
            )
            annotations.append(
                dict(
                    x=x, y=self.depths[ssp], xref="x", yref="y", xshift=14, xanchor="left",
                    text=SSP_LABELS[ssp].replace(" ", "<br>"), showarrow=False,
                    align="left", font=dict(size=self.font_size, color=LABEL_COLOR),
                )
            )
        annotations.append(self._y_axis_arrow())
        annotations.append(self._x_axis_label())
        return annotations

    @property
    def _panel_top(self) -> float:
        """Distance of the panel's plot area from the top, leaving room for x-axis ticks above."""
        return TICK_SPACE if self.flipped else 0

    def _x_axis_label(self) -> dict:
        """Axis label left of the SSP numbers, on the level of the tick labels."""
        return dict(
            text="SSP:", x=self._x(self.panel_left + PANEL_AXIS_OFFSET),
            y=self._y(self._panel_top - 15 if self.flipped
                      else self.panel_height + TICK_SPACE / 2 - 6), xref="paper", yref="paper",
            xanchor="right", yanchor="middle", showarrow=False,
            font=dict(size=self.font_size, color=RULE_COLOR),
        )

    def _y_axis_arrow(self) -> dict:
        """Unitless y axis, drawn as an arrow pointing towards more circularity."""
        overshoot = 0 if self.flipped else 10
        tip, tail = (self.panel_height + overshoot, 0) if self.flipped else (
            -overshoot, self.panel_height)
        return dict(
            x=self._x(self.panel_left + PANEL_AXIS_OFFSET), y=self._y(tip + self._panel_top),
            ax=0, ay=tail - tip,
            xref="paper", yref="paper", axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.2,
            arrowcolor=AXIS_COLOR, text="",
        )

    def depth_range(self) -> list[float]:
        depths = self.depths.values()
        low, high = min(min(depths), 0) - 0.15, max(depths) + 0.15
        return [high, low] if self.flipped else [low, high]

    def plot(self) -> go.Figure:
        table_shapes, table_annotations = self.table()
        # the template draws shapes semi-transparent
        table_shapes = [shape | {"opacity": 1} for shape in table_shapes]
        fig = go.Figure(self.depth_traces())
        fig.update_layout(
            template="simple_white",
            font=dict(family="Arial, sans-serif", size=16, color="#222222"),
            showlegend=False,
            xaxis=dict(
                domain=[self._x(self.panel_left + PANEL_AXIS_OFFSET), 1],
                tickmode="array",
                tickvals=list(range(len(BASE_ORDER))),
                ticktext=[ssp.removeprefix("SSP") for ssp in BASE_ORDER],
                tickfont=dict(size=self.font_size),
                range=[-0.5, len(BASE_ORDER) - 0.5],
                ticks="",
                side="top" if self.flipped else "bottom",
            ),
            yaxis=dict(
                anchor="x",
                domain=[self._y(self._panel_top + self.panel_height), self._y(self._panel_top)],
                title=dict(text="Circularity depth", standoff=8,
                           font=dict(size=self.font_size + 2)),
                showline=False,
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=True,
                zerolinecolor="#bdbdbd",
                range=self.depth_range(),
            ),
            margin=MARGIN,
            shapes=table_shapes + self.depth_shapes(),
            annotations=table_annotations + self.depth_annotations(),
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=BACKGROUND,
        )
        return fig


class CircularityDepthFigure(ScenarioOverviewFigure):
    """Circularity depth panel of figure 12 without the parameter table.

    The axis is flipped (high circularity at the bottom) and the panel is wider.
    """

    panel_left = 20
    flipped = True
    font_size = 20
    height = round(0.8 * HEIGHT)
    panel_height = height - MARGIN["t"] - MARGIN["b"] - TICK_SPACE
    standalone_panel_width = 380
    width = MARGIN["l"] + panel_left + PANEL_AXIS_OFFSET + standalone_panel_width + MARGIN["r"]

    def table(self) -> tuple[list[dict], list[dict]]:
        return [], []


def save(fig: go.Figure, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


figure = ScenarioOverviewFigure()
save(figure.plot(), "fig12_scenario_overview", width=figure.width, height=figure.height)
depth_figure = CircularityDepthFigure(figure.data)
save(depth_figure.plot(), "fig12_circularity_depth", width=depth_figure.width,
     height=depth_figure.height)

print("END")
