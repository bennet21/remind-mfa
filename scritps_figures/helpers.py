import pickle
import warnings
from pathlib import Path

import numpy as np

from constants import AGG_REGION_ORDER, AGG_REGIONS

# Shading range shared across figure scripts (darkest -> lightest function shade).
SHADE_MIN, SHADE_MAX = -0.30, 0.45


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


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "combined": cache_dir / "combined_mfa.pickle",
        "td": cache_dir / "td_mfa.pickle",
        "bu": cache_dir / "bu_mfa.pickle",
    }


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


def load_mfas(
    source_pickle: Path, cache_dir: Path, force_refresh: bool = False
) -> dict[str, object]:
    """Load the MFAs / stocks needed for the figures, caching them next to the source pickle.

    Returns a dict with:
    - "combined": reconciled bottom-up / combined MFA (carries the Structure dimension `b`)
    - "td": pre-reconciliation top-down future MFA
    - "bu": pre-reconciliation bottom-up concrete stock array (`FlodymArray`, not an MFA):
      the pure bottom-up concrete in-use stock incl. hibernating stock, dims (t, r, b, s)
      (b = bottom-up good RS/RM/Com, s = structure C/M/T/S/U)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)

    if not force_refresh and cache_is_valid(source_pickle, paths):
        print(f"Loading MFA objects from cache: {cache_dir}")
        mfas = {}
        for label, path in paths.items():
            with path.open("rb") as file_handle:
                mfas[label] = pickle.load(file_handle)
        return mfas

    print(f"Building MFA cache from source pickle: {source_pickle.name}")
    model = load_model(source_pickle)

    mfas = {
        "combined": model.bu_mfa_reconciled,
        "td": model.td_mfa,
        "bu": model.bu_stock,
    }

    for label, path in paths.items():
        with path.open("wb") as file_handle:
            pickle.dump(mfas[label], file_handle)

    return mfas


def cement_demand(mfa) -> object:
    """Regional market cement demand (t) with dims (t, r).

    The cement that ends up in products plus the construction losses that go with it, i.e.
    the model's ``market_cement => prod_product`` plus ``market_cement => sysenv``. This is
    consumption-based: cement produced for export is counted in the importing region.

    Pass the reconciled (combined) MFA, so that the reconciled ``cement_losses`` is used.
    """
    into_products = mfa.stocks["in_use"].inflow[{"k": "cement"}].sum_to(("t", "r"))
    return into_products / (1.0 - mfa.parameters["cement_losses"])


def process_emissions(demand, parameters) -> object:
    """Gross process CO2 (t) from calcination, with the dims of `demand`.

    Mirrors the ``prod_clinker => atmosphere`` flow of the carbonation model: the CO2
    released from the CaO in clinker, plus the CO2 from the CaO in the cement kiln dust
    that is generated alongside it (``clinker_losses`` is additional to the clinker that
    reaches the market, not a share of it). Applied to demand rather than to the clinker
    production flow, so emissions are attributed to the consuming region.
    """
    clinker = demand * parameters["clinker_ratio"]
    cao_per_clinker = (
        parameters["clinker_cao_ratio"] + parameters["clinker_losses"] * parameters["ckd_cao_ratio"]
    )
    return clinker * cao_per_clinker * parameters["cao_emission_factor"]


def aggregate_by_region(values: np.ndarray, region_items: np.ndarray) -> dict[str, np.ndarray]:
    """Sum a (t, r) array's source regions into the aggregated regions of `AGG_REGIONS`.

    Returns one time series per aggregated region, in `AGG_REGION_ORDER`.
    """
    unmapped = sorted(set(str(region) for region in region_items) - set(AGG_REGIONS))
    if unmapped:
        raise ValueError(f"Regions missing from AGG_REGIONS: {', '.join(unmapped)}")

    result = {agg_region: np.zeros(values.shape[0]) for agg_region in AGG_REGION_ORDER}
    for source_region, agg_region in AGG_REGIONS.items():
        result[agg_region] += values[:, region_items == source_region].sum(axis=1)
    return result
