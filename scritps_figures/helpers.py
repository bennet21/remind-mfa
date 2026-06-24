import pickle
import warnings
from pathlib import Path

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


def load_mfas(source_pickle: Path, cache_dir: Path, force_refresh: bool = False) -> dict[str, object]:
    """Load the two MFAs needed for figure 1, caching them next to the source pickle.

    Returns a dict with:
    - "combined": reconciled bottom-up / combined MFA (carries the Structure dimension `b`)
    - "td": pre-reconciliation top-down future MFA
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
    }

    for label, path in paths.items():
        with path.open("wb") as file_handle:
            pickle.dump(mfas[label], file_handle)

    return mfas
