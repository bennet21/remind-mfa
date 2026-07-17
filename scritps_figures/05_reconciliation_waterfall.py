"""Figure 5: reconciliation waterfall.

A "converging double waterfall" (bridge chart) that decomposes, via Shapley values, how each
uncertain parameter moves the two disagreeing estimates of the last-historic-year cement-in-concrete
stock toward the reconciled value:

- Left half : bottom-up initial -> per-parameter steps -> bottom-up reconciled
- Center    : bottom-up reconciled and top-down reconciled (they agree after reconciliation)
- Right half: top-down reconciled -> per-parameter steps (walked outward) -> top-down initial

Because Shapley contributions sum exactly to ``f(adjusted) - f(original)``, each side bridges cleanly
from its initial bar to the reconciled bar.

Two modes:
- ``MODE = "global"`` (default): one figure with three stacked panels (combined / Res / Com) for the
  global aggregate.
- ``MODE = "regional"``: one three-panel figure per region.

Run from the repository root:
    uv run python scritps_figures/05_reconciliation_waterfall.py
"""

import pickle

import flodym as fd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    SOURCE_PICKLE,
    CACHE_DIR_CEMENT,
    FIGURES_DIR,
    LAST_HISTORICAL_YEAR,
    CONCRETE_MASK,
    REGION_DISPLAY_NAMES,
)
from helpers import load_model

from remind_mfa.cement.cement_parameter_reconciliation import (
    CementParameterReconciliation,
    AnalyzeParameterReconciliation,
)
from remind_mfa.cement.cement_mfa_system_bottom_up import REDUCED_STOCK_TYPE

# --- configuration -----------------------------------------------------------------------------
MODE = "global"  # "global" | "regional"
MIN_PCT = 5.0  # steps contributing less than this % of the initial gap are grouped into "Other"

BU_NAME, TD_NAME = "Bottom-up", "Top-down"
BU_COLOR, TD_COLOR = "#4F8FC6", "#E07A9A"
UP_COLOR, DOWN_COLOR = "#4CAF50", "#F44336"  # step up (green) / down (red)

STOCK_TYPES = [None, "Res", "Com"]  # None = Res + Com combined
STOCK_TYPE_TITLES = {None: "Combined (residential + commercial)", "Res": "Residential", "Com": "Commercial"}

PATTERN_KWARGS = {"fgcolor": "white", "size": 7, "solidity": 0.4}


# =================================================================================================
# Analysis layer (ported from the old parameter-reconciliation-paper waterfall script)
# =================================================================================================
class ReducingDict:
    """Lazy wrapper that reduces s->u on key access, forwarding __getitem__ to the underlying dict.

    This ensures that DependencyTracker (used by AnalyzeParameterReconciliation to spy on which
    parameters are actually accessed) only records the keys the calc function uses, not all keys
    iterated up-front.
    """

    def __init__(self, prms, reduced_stock_type):
        self._prms = prms
        self._rsd = reduced_stock_type

    def __getitem__(self, key):
        val = self._prms[key]  # triggers DependencyTracker.__getitem__ when spying
        if "s" in val.dims.letters:
            val = val[{"s": self._rsd}]
        return val

    def __iter__(self):
        return iter(self._prms)

    def __len__(self):
        return len(self._prms)

    def keys(self):
        return self._prms.keys()

    def items(self):
        return ((k, self[k]) for k in self._prms)

    def values(self):
        return (self[k] for k in self._prms)


def make_fns(pr, population, region, stock_type):
    """Return (td_fn, bu_fn, pop_scalar) for a given region and stock type.

    Args:
        pr: the CementParameterReconciliation instance.
        population: population parameter at the last historic year (dims include "r").
        region: region key (e.g. "EUR"), or None for the global aggregate.
        stock_type: "Res", "Com", or None for both combined.
    """
    if region is None:
        pop_scalar = float(population.sum_over("r").values)
    else:
        pop_scalar = float(population[{"r": region}].values)

    def _select(arr):
        sel = {}
        if region is not None:
            sel["r"] = region
        if stock_type is not None:
            sel["u"] = stock_type
        if sel:
            arr = arr[sel]
        return arr.sum_to(fd.DimensionSet(dim_list=[]))

    def td_fn(prms):
        rd = ReducingDict(prms, REDUCED_STOCK_TYPE)
        concrete_stock = pr.calc_top_down_stock(rd)
        cement_in_concrete = concrete_stock * rd["cement_ratio"][CONCRETE_MASK]
        return _select(cement_in_concrete) / pop_scalar

    def bu_fn(prms):
        rd = ReducingDict(prms, REDUCED_STOCK_TYPE)
        concrete_stock = CementParameterReconciliation.calc_bottom_up_stock(rd)
        cement_in_concrete = concrete_stock * rd["cement_ratio"][CONCRETE_MASK]
        return _select(cement_in_concrete) / pop_scalar

    return td_fn, bu_fn, pop_scalar


def load_or_compute_impacts(analyzer, td_fn, bu_fn, cache_key: str):
    bu_cache = CACHE_DIR_CEMENT / f"bu_impact_{cache_key}.pkl"
    td_cache = CACHE_DIR_CEMENT / f"td_impact_{cache_key}.pkl"
    if bu_cache.exists() and td_cache.exists():
        with bu_cache.open("rb") as f:
            bu_impact = pickle.load(f)
        with td_cache.open("rb") as f:
            td_impact = pickle.load(f)
    else:
        bu_impact = analyzer.calc_parameter_impact(bu_fn)
        td_impact = analyzer.calc_parameter_impact(td_fn)
        CACHE_DIR_CEMENT.mkdir(parents=True, exist_ok=True)
        with bu_cache.open("wb") as f:
            pickle.dump(bu_impact, f)
        with td_cache.open("wb") as f:
            pickle.dump(td_impact, f)
    return bu_impact, td_impact


def group_small_impacts(adjustments, labels, min_pct, total_gap):
    """Group steps whose |contribution| is below min_pct of the initial gap into a single 'Other'."""
    grouped_adj, grouped_labels = [], []
    other_sum = 0.0
    n_other = 0
    for val, label in zip(adjustments, labels):
        pct = abs(val) / total_gap * 100
        if pct < min_pct:
            other_sum += val
            n_other += 1
        else:
            grouped_adj.append(val)
            grouped_labels.append(label)
    if n_other > 0:
        grouped_adj.append(other_sum)
        grouped_labels.append("Other")
    return grouped_adj, grouped_labels


# =================================================================================================
# Plotting layer (plotly, styled like 04_reconciliation_stock.py)
# =================================================================================================
def build_waterfall_bars(start_bu, end_bu, end_td, start_td, bu_adj, bu_labels, td_adj, td_labels):
    """Assemble the ordered bar sequence for one converging waterfall panel.

    Returns dict of parallel lists: x labels, bar bottoms, bar heights, colors, hatch flags,
    text labels, and the top-of-bar y used to draw connectors.
    """
    total_gap = abs(start_td - start_bu) or 1e-9

    x, bottoms, heights, colors, hatched, texts, tops = [], [], [], [], [], [], []

    def add_base(label, value, color):
        x.append(label)
        bottoms.append(0.0)
        heights.append(value)
        colors.append(color)
        hatched.append(True)
        texts.append(f"{value:,.2f}")
        tops.append(value)

    def add_step(label, value, current_h, reverse):
        # forward (left side): plot the adjustment directly; reverse (right side): plot its inverse
        step = -value if reverse else value
        new_h = current_h + step
        x.append(label)
        heights.append(abs(step))
        bottoms.append(min(current_h, new_h))
        colors.append(UP_COLOR if step >= 0 else DOWN_COLOR)
        hatched.append(False)
        texts.append(f"{abs(value) / total_gap * 100:.0f}%")
        tops.append(new_h)
        return new_h

    # LEFT: BU initial -> steps -> BU reconciled
    add_base(f"{BU_NAME}<br>initial", start_bu, BU_COLOR)
    h = float(start_bu)
    for label, val in zip(bu_labels, bu_adj):
        h = add_step(label, val, h, reverse=False)
    add_base(f"{BU_NAME}<br>reconciled", end_bu, BU_COLOR)

    # CENTER + RIGHT: TD reconciled -> steps walked outward -> TD initial
    add_base(f"{TD_NAME}<br>reconciled", end_td, TD_COLOR)
    h = float(end_td)
    for label, val in zip(td_labels, td_adj):
        h = add_step(label, val, h, reverse=True)
    add_base(f"{TD_NAME}<br>initial", start_td, TD_COLOR)

    return {
        "x": x,
        "bottoms": bottoms,
        "heights": heights,
        "colors": colors,
        "hatched": hatched,
        "texts": texts,
        "tops": tops,
        "total_gap": total_gap,
    }


def add_panel(fig, panel, row):
    """Add one waterfall (bars, connectors, bar-top values) to subplot (row, 1)."""
    n = len(panel["x"])
    idx = list(range(n))

    fig.add_trace(
        go.Bar(
            x=idx,
            y=panel["heights"],
            base=panel["bottoms"],
            marker={
                "color": panel["colors"],
                "pattern": {"shape": ["/" if h else "" for h in panel["hatched"]], **PATTERN_KWARGS},
                "line": {"width": 0.5, "color": "rgba(0,0,0,0.55)"},
            },
            text=panel["texts"],
            textposition="outside",
            textfont={"size": 10},
            width=0.6,
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    # dashed connectors between the top of each bar and the start of the next
    tops = panel["tops"]
    for i in range(n - 1):
        fig.add_trace(
            go.Scatter(
                x=[i + 0.3, i + 1 - 0.3],
                y=[tops[i], tops[i]],
                mode="lines",
                line={"color": "rgba(0,0,0,0.55)", "width": 0.8, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=idx,
        ticktext=panel["x"],
        tickangle=-40,
        tickfont={"size": 10},
        row=row,
        col=1,
    )


def add_legend_proxies(fig):
    """Legend explaining the bar colors/patterns via invisible proxy bars."""
    proxies = [
        (f"{BU_NAME} estimate", BU_COLOR, "/"),
        (f"{TD_NAME} estimate", TD_COLOR, "/"),
        ("Increases estimate", UP_COLOR, ""),
        ("Decreases estimate", DOWN_COLOR, ""),
    ]
    for name, color, shape in proxies:
        fig.add_trace(
            go.Bar(
                x=[0],
                y=[None],
                name=name,
                marker={"color": color, "pattern": {"shape": shape, **PATTERN_KWARGS}},
                showlegend=True,
            ),
            row=1,
            col=1,
        )


def save(fig, output_name: str, width: int, height: int):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = (FIGURES_DIR / output_name).with_suffix(".png")
    fig.write_image(png_path, width=width, height=height, scale=3)
    print(f"Saved figure to: {png_path}")


def make_figure(analyzer, pr, population, region, region_label, output_name):
    """Build and save a three-panel (combined / Res / Com) waterfall figure for one region."""
    fig = make_subplots(
        rows=len(STOCK_TYPES),
        cols=1,
        vertical_spacing=0.09,
        subplot_titles=[STOCK_TYPE_TITLES[st] for st in STOCK_TYPES],
    )

    for row, stock_type in enumerate(STOCK_TYPES, start=1):
        region_key = region if region is not None else "global"
        stock_key = stock_type if stock_type is not None else "combined"
        cache_key = f"{region_key}_{stock_key}"
        print(f"  panel: {region_label} / {STOCK_TYPE_TITLES[stock_type]}")

        td_fn, bu_fn, _ = make_fns(pr, population, region, stock_type)
        bu_impact, td_impact = load_or_compute_impacts(analyzer, td_fn, bu_fn, cache_key)

        start_bu = float(bu_fn(analyzer.original_prms).values)
        end_bu = float(bu_fn(analyzer.adjusted_prms).values)
        start_td = float(td_fn(analyzer.original_prms).values)
        end_td = float(td_fn(analyzer.adjusted_prms).values)
        print(
            f"    BU: {start_bu:.3f} -> {end_bu:.3f} | TD: {start_td:.3f} -> {end_td:.3f} "
            f"| reconciled gap = {abs(end_bu - end_td):.4f}"
        )

        total_gap = abs(start_td - start_bu) or 1e-9
        bu_adj, bu_labels = group_small_impacts(
            bu_impact.values.tolist(), list(bu_impact.dims["p"].items), MIN_PCT, total_gap
        )
        td_adj, td_labels = group_small_impacts(
            td_impact.values.tolist(), list(td_impact.dims["p"].items), MIN_PCT, total_gap
        )

        panel = build_waterfall_bars(
            start_bu, end_bu, end_td, start_td, bu_adj, bu_labels, td_adj, td_labels
        )
        add_panel(fig, panel, row)

    add_legend_proxies(fig)

    for annotation in fig.layout.annotations:  # subplot titles
        annotation.font = {"size": 14}

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": f"Parameter contributions to stock reconciliation — {region_label} ({H})",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "font": {"size": 12},
        },
        margin={"t": 80, "l": 100, "b": 90, "r": 230},
    )
    fig.update_yaxes(showgrid=True, rangemode="tozero")

    fig.add_annotation(
        text="Cement stock in concrete (t/capita)",
        x=-0.07,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        font={"size": 15},
    )

    save(fig, output_name, width=1250, height=1200)


# =================================================================================================
# Driver
# =================================================================================================
H = LAST_HISTORICAL_YEAR

print(f"Loading model: {SOURCE_PICKLE.name}")
model = load_model(SOURCE_PICKLE)

if not hasattr(model, "parameter_reconciliation") or model.parameter_reconciliation is None:
    raise SystemExit(
        "model.parameter_reconciliation is missing from the pickle — cannot build the waterfall. "
        "Re-export the model with the reconciliation object attached."
    )

pr: CementParameterReconciliation = model.parameter_reconciliation
if not hasattr(pr, "input_prms") or not hasattr(pr, "output_prms"):
    raise SystemExit("parameter_reconciliation is missing input_prms/output_prms.")

population = model.td_mfa.parameters["population"][{"t": H}]

analyzer = AnalyzeParameterReconciliation(
    pr,
    original_prms=pr.input_prms,
    adjusted_prms=pr.output_prms,
)

if MODE == "global":
    make_figure(
        analyzer, pr, population,
        region=None, region_label="Global",
        output_name="fig5_reconciliation_waterfall_global",
    )
elif MODE == "regional":
    for region in REGION_DISPLAY_NAMES:
        label = REGION_DISPLAY_NAMES[region].replace("<br>", " ")
        make_figure(
            analyzer, pr, population,
            region=region, region_label=label,
            output_name=f"fig5_reconciliation_waterfall_{region}",
        )
else:
    raise SystemExit(f"Unknown MODE: {MODE!r} (expected 'global' or 'regional')")

print("END")
