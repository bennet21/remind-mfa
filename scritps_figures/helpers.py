import pickle
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import flodym as fd
import numpy as np
import pandas as pd

from constants import (
    AGG_REGION_ORDER,
    AGG_REGIONS,
    MODEL_REGIONMAPPING_CSV,
    REGION_MAPPING,
    REGIONMAPPING_CSV,
    SSP_CACHE_DIRS,
    SSP_SOURCE_PICKLES,
    SSP_UNRECONCILED_PICKLES,
)

# Shading range shared across figure scripts (darkest -> lightest function shade).
SHADE_MIN, SHADE_MAX = -0.30, 0.45

# Short item names used by the figure scripts, per dimension letter. Items not listed keep their
# model name.
SHORT_ITEM_NAMES = {
    "e": {
        "Residential Single-Family": "RS",
        "Residential Multi-Family": "RM",
        "Commercial": "Com",
        "Industrial": "Ind",
        "Civil Engineering": "Civ",
    },
    "b": {
        "Residential Single-Family": "RS",
        "Residential Multi-Family": "RM",
        "Commercial": "Com",
    },
    "u": {
        "Residential": "Res",
        "Commercial": "Com",
        "Industrial": "Ind",
        "Civil Engineering": "Civ",
    },
    "s": {
        "Reinforced Concrete Structure": "C",
        "Masonry Structure": "M",
        "Timber Structure": "T",
        "Steel Frame Structure": "S",
        "Unspecified Structure": "U",
    },
    "m": {"Concrete": "concrete", "Mortar": "mortar"},
    "k": {"Cement": "cement", "Non-Cement": "non-cement"},
}

# Parameters of the carbonation module needed for the process emissions. They are only part of
# the model parameters if carbonation is switched on, so they are read separately.
EMISSION_PARAMETER_NAMES = ("clinker_cao_ratio", "ckd_cao_ratio", "cao_emission_factor")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def shade(base_hex: str, t: float) -> str:
    """Blend a base colour toward black (t<0) or white (t>0); return an rgb() string."""
    rgb = _hex_to_rgb(base_hex)
    target = (255, 255, 255) if t >= 0 else (0, 0, 0)
    amount = abs(t)
    out = tuple(round(c + (tc - c) * amount) for c, tc in zip(rgb, target))
    return f"rgb({out[0]},{out[1]},{out[2]})"


def shade_levels(n: int) -> list[float]:
    if n <= 1:
        return [0.1]
    return [SHADE_MIN + (SHADE_MAX - SHADE_MIN) * i / (n - 1) for i in range(n)]


@dataclass
class ReconciledData:
    """Figure data from a reconciled model run, on the level of the model regions.

    Attributes:
        inflow: New material entering the in-use stock (`construction => use`, without reuse),
            dims (t, r, e, m, s, k).
        stock: In-use stock, dims (t, r, e, m, s, k).
        demand: Market cement demand incl. construction losses (`cement_market => construction`),
            dims (t, r). Consumption-based: cement produced for export counts in the importing
            region.
        emissions: Gross process CO2 (t) from calcination for `demand`, dims (t, r).
        population: Population, dims (t, r).
        bu_stock: Pre-reconciliation bottom-up cement-in-concrete stock in the last historical
            year, incl. unoccupied residential stock, dims (r, b).
    """

    inflow: fd.FlodymArray
    stock: fd.FlodymArray
    demand: fd.FlodymArray
    emissions: fd.FlodymArray
    population: fd.FlodymArray
    bu_stock: fd.FlodymArray


@dataclass
class UnreconciledData:
    """Figure data from a model run without reconciliation (pure top-down), on the level of the
    model regions.

    Attributes:
        inflow: New material entering the in-use stock (`construction => use`, without reuse),
            dims (t, r, u, m, k).
        stock: In-use stock, dims (t, r, u, m, k).
    """

    inflow: fd.FlodymArray
    stock: fd.FlodymArray


@dataclass
class FigureData:
    """Reconciled and unreconciled figure data of one scenario."""

    reconciled: ReconciledData
    unreconciled: UnreconciledData


def with_short_item_names(array: fd.FlodymArray) -> fd.FlodymArray:
    """Copy of `array` with the item names of `SHORT_ITEM_NAMES`."""
    dims = [
        fd.Dimension(
            name=dim.name,
            letter=dim.letter,
            items=[SHORT_ITEM_NAMES.get(dim.letter, {}).get(item, item) for item in dim.items],
        )
        for dim in array.dims
    ]
    return fd.FlodymArray(
        dims=fd.DimensionSet(dim_list=dims), values=array.values.copy(), name=array.name
    )


def process_emissions(demand: fd.FlodymArray, parameters: dict) -> fd.FlodymArray:
    """Gross process CO2 (t) from calcination, with the dims of `demand`.

    Mirrors the ``clinker_production => atmosphere`` flow of the carbonation model: the CO2
    released from the CaO in clinker, plus the CO2 from the CaO in the cement kiln dust
    that is generated alongside it (``clinker_loss_rate`` is additional to the clinker that
    reaches the market, not a share of it). Applied to demand rather than to the clinker
    production flow, so emissions are attributed to the consuming region.
    """
    clinker = demand * parameters["clinker_ratio"]
    cao_per_clinker = (
        parameters["clinker_cao_ratio"]
        + parameters["clinker_loss_rate"] * parameters["ckd_cao_ratio"]
    )
    return (clinker * cao_per_clinker * parameters["cao_emission_factor"]).cast_to(demand.dims)


def read_emission_parameters(model) -> dict[str, fd.Parameter]:
    """Read the scalar emission parameters of the carbonation module from the model's input
    data (`.cs4r` files: header lines prefixed with `*`, then a single value)."""
    parameters = {}
    for name in EMISSION_PARAMETER_NAMES:
        path = Path(model.cfg.input.input_data_path) / "parameters" / f"ce_{name}.cs4r"
        lines = [line for line in path.read_text().splitlines() if not line.startswith("*")]
        parameters[name] = fd.Parameter(
            dims=fd.DimensionSet(dim_list=[]), values=np.array(float(lines[0])), name=name
        )
    return parameters


def bottom_up_cement_stock(model) -> fd.FlodymArray:
    """Pre-reconciliation bottom-up cement-in-concrete stock in the last historical year,
    dims (r, b).

    Computed from the input parameters of the reconciliation, in the same way as the bottom-up
    stock that the reconciliation compares to the top-down stock.
    """
    from remind_mfa.cement.cement_mfa_system_bottom_up import (
        StockDrivenBottomUpCementMFASystem as BU,
    )

    reconciliation = model.parameter_reconciliation
    # reduced to the reconciliation year (last historical year)
    prm = reconciliation.reduce_prms(reconciliation.input_prms)
    concrete = BU.concrete_from_floorspace(BU.calc_gross_floorspace(prm), prm).sum_to(("r", "b"))
    return concrete * prm["cement_ratio"][{"m": "Concrete"}]


def extract_reconciled(model) -> ReconciledData:
    """Figure data of a reconciled model run (`model.bu_mfa_reconciled`)."""
    mfa = model.bu_mfa_reconciled
    prm = {**mfa.parameters, **read_emission_parameters(model)}
    demand = mfa.flows["cement_market => construction"].sum_to(("t", "r"))
    return ReconciledData(
        inflow=with_short_item_names(mfa.flows["construction => use"]),
        stock=with_short_item_names(mfa.stocks["use"].stock),
        demand=demand,
        emissions=process_emissions(demand, prm),
        population=prm["population"].sum_to(("t", "r")),
        bu_stock=with_short_item_names(bottom_up_cement_stock(model)),
    )


def extract_unreconciled(model) -> UnreconciledData:
    """Figure data of a model run without reconciliation (`model.future_mfa`)."""
    mfa = model.future_mfa
    return UnreconciledData(
        inflow=with_short_item_names(mfa.flows["construction => use"]),
        stock=with_short_item_names(mfa.stocks["use"].stock),
    )


def cache_is_valid(source_pickle: Path, paths: dict[str, Path]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False

    source_mtime = source_pickle.stat().st_mtime
    return all(path.stat().st_mtime >= source_mtime for path in paths.values())


def load_model(source_pickle: Path) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with source_pickle.open("rb") as file_handle:
            return pickle.load(file_handle)


def save_pickle(obj: object, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)


def figure_cache_paths(ssp: str) -> dict[str, Path]:
    return {
        "reconciled": SSP_CACHE_DIRS[ssp] / "reconciled.pickle",
        "unreconciled": SSP_CACHE_DIRS[ssp] / "unreconciled.pickle",
    }


def load_cached[T](
    source_pickle: Path,
    cache_path: Path,
    extract: Callable[[object], T],
    force_refresh: bool = False,
) -> T:
    """Load figure data from `cache_path`, or extract it from the model pickle and cache it.

    The cache is used if it is newer than the model pickle.
    """
    if not force_refresh and cache_is_valid(source_pickle, {"cache": cache_path}):
        print(f"Loading figure data from cache: {cache_path}")
        with cache_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    print(f"Building figure data cache from source pickle: {source_pickle}")
    data = extract(load_model(source_pickle))
    save_pickle(data, cache_path)
    return data


def load_figure_data(ssp: str, force_refresh: bool = False) -> FigureData:
    """Load the reconciled and unreconciled figure data of a scenario.

    All arrays are on the level of the model regions (constants.REGION_MAPPING). Use `to_h12`
    to aggregate extensive quantities to the H12 regions.
    """
    paths = figure_cache_paths(ssp)
    return FigureData(
        reconciled=load_cached(
            SSP_SOURCE_PICKLES[ssp], paths["reconciled"], extract_reconciled, force_refresh
        ),
        unreconciled=load_cached(
            SSP_UNRECONCILED_PICKLES[ssp],
            paths["unreconciled"],
            extract_unreconciled,
            force_refresh,
        ),
    )


@cache
def model_to_h12_mapping() -> dict[str, str]:
    """H12 region of each model region (constants.REGION_MAPPING)."""
    model_regions = pd.read_csv(MODEL_REGIONMAPPING_CSV, sep=";").set_index("CountryCode")
    h12_regions = pd.read_csv(REGIONMAPPING_CSV, sep=";").set_index("CountryCode")
    pairs = (
        pd.DataFrame({"model": model_regions["RegionCode"], "h12": h12_regions["RegionCode"]})
        .dropna()
        .drop_duplicates()
    )
    if not pairs["model"].is_unique:
        raise ValueError(f"Region mapping {REGION_MAPPING} does not aggregate uniquely to H12.")
    return dict(zip(pairs["model"], pairs["h12"]))


def aggregate_regions(array: fd.FlodymArray, mapping: dict[str, str]) -> fd.FlodymArray:
    """Sum an extensive array over the regions (`r`) that `mapping` assigns to the same aggregate
    region. The aggregate regions are sorted alphabetically."""
    region_items = [str(region) for region in array.dims["r"].items]
    unmapped = sorted(set(region_items) - set(mapping))
    if unmapped:
        raise ValueError(f"Regions missing from the region mapping: {', '.join(unmapped)}")

    aggregate_items = sorted(set(mapping[region] for region in region_items))
    axis = array.dims.letters.index("r")
    target_indices = [aggregate_items.index(mapping[region]) for region in region_items]
    shape = list(array.values.shape)
    shape[axis] = len(aggregate_items)
    values = np.zeros(shape)
    np.add.at(np.moveaxis(values, axis, 0), target_indices, np.moveaxis(array.values, axis, 0))

    region_dim = array.dims["r"]
    dims = [
        fd.Dimension(name=region_dim.name, letter="r", items=aggregate_items)
        if dim.letter == "r"
        else dim
        for dim in array.dims
    ]
    return fd.FlodymArray(dims=fd.DimensionSet(dim_list=dims), values=values, name=array.name)


def to_h12(array: fd.FlodymArray) -> fd.FlodymArray:
    """Sum an extensive array of the model regions to the H12 regions."""
    return aggregate_regions(array, model_to_h12_mapping())


def shared_range(*value_arrays: np.ndarray, pad_frac: float = 0.05) -> list[float]:
    """Common axis range spanning all given value arrays, with a small padding fraction.

    Used to give every panel of a multi-panel figure (or every figure in a per-region series)
    the same axis range, so panels are visually comparable. NaNs (e.g. from weighted averages
    dividing by a zero pre-historic baseline) are ignored.
    """
    all_values = np.concatenate([np.asarray(values, dtype=float).ravel() for values in value_arrays])
    lo, hi = np.nanmin(all_values), np.nanmax(all_values)
    pad = pad_frac * (hi - lo)
    return [lo - pad, hi + pad]


def aggregate_by_region(values: np.ndarray, region_items: np.ndarray) -> dict[str, np.ndarray]:
    """Sum a (t, r) array's H12 regions into the aggregated regions of `AGG_REGIONS`.

    Returns one time series per aggregated region, in `AGG_REGION_ORDER`.
    """
    unmapped = sorted(set(str(region) for region in region_items) - set(AGG_REGIONS))
    if unmapped:
        raise ValueError(f"Regions missing from AGG_REGIONS: {', '.join(unmapped)}")

    result = {agg_region: np.zeros(values.shape[0]) for agg_region in AGG_REGION_ORDER}
    for source_region, agg_region in AGG_REGIONS.items():
        result[agg_region] += values[:, region_items == source_region].sum(axis=1)
    return result
