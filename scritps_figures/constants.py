from pathlib import Path

import pyam

PATH_CEMENT = Path("data_out")
CACHE_DIR_CEMENT = Path("data/cement/output/cache")
FIGURES_DIR = Path("data/cement/output/figures")
LAST_HISTORICAL_YEAR = 2023
REGIONMAPPING_CSV = Path("scritps_figures/h12.csv")

# Cumulative-value marker years for figures 7 and 8 (absolute variants only)
CUMULATIVE_MARKER_YEARS = [2050, 2070]
CUMULATIVE_MARKER_YEAR_HISTORICAL = 1990  # figure 8 only

CEMENT_PICKLENAME = "figs_cement_SSP2_h12/model.pickle"
SOURCE_PICKLE = PATH_CEMENT / CEMENT_PICKLENAME

# Masks
CEMENT_MASK = {"k": "cement"}
CONCRETE_MASK = {"m": "concrete"}

STRUCTURE_DISPLAY_NAMES = {
    "C": "Concrete buildings",
    "M": "Masonry buildings",
    "T": "Timber buildings",
    "S": "Steel buildings",
    "U": "Unspecified",
}

# Structure item(s) excluded from the per-structure building bands and shown as the "Other" band
# instead. "U" (Unspecified structure) carries the nonzero non-building cement (industrial + civil
# goods, and all mortar). Building goods (RS/RM/Com) may also take structure "U", but their material
# intensity there is zero, so they contribute no concrete and the C/M/T/S bands capture all building
# concrete.
OTHER_STRUCTURE_KEYS = {"U"}

# Three grey shades for the Other sub-categories (darkest -> lightest), kept close together
# so they recede visually relative to the saturated building-use colors.
OTHER_IND_COLOR = "#8a8a8a"
OTHER_CIV_COLOR = "#a4a4a4"
OTHER_RES_COM_MORTAR_COLOR = "#bebebe"

OTHER_IND_NAME = "Industrial buildings"
OTHER_CIV_NAME = "Civil engineering"
OTHER_RES_COM_MORTAR_NAME = "Res./com. mortar"

# Building function split (subdivides each structure via shading).
FUNCTION_DISPLAY_NAMES = {
    "RS": "Single-family res. buildings",
    "RM": "Multi-family res. buildings",
    "Com": "Commercial buildings",
}

# One distinct, mid-tone hue per building structure; functions become shades of it.
STRUCTURE_BASE_COLORS = {
    "C": "#2C7FB8",  # blue
    "M": "#D9820B",  # orange
    "T": "#2CA25F",  # green
    "S": "#B0436B",  # rose
}

REGION_DISPLAY_NAMES = {
    "CAZ": "Canada, NZ, Australia",
    "CHA": "China",
    "EUR": "EU 28",
    "IND": "India",
    "JPN": "Japan",
    "LAM": "Latin America and<br>the Caribbean",
    "MEA": "Middle East,<br>North Africa,<br>Central Asia",
    "NEU": "Non-EU28 Europe",
    "OAS": "Other Asia",
    "REF": "Countries from the<br>Reforming Economies of<br>the Former Soviet Union",
    "SSA": "Sub-Saharan Africa",
    "USA": "USA",
}

AGG_REGIONS = {
    "CAZ": "OECD",
    "CHA": "China",
    "EUR": "OECD",
    "IND": "S & SE Asia",
    "JPN": "OECD",
    "LAM": "Rest of the World",
    "MEA": "Rest of the World",
    "NEU": "OECD",
    "OAS": "S & SE Asia",
    "REF": "Rest of the World",
    "SSA": "Sub-Saharan Africa",
    "USA": "OECD",
}

AGG_REGION_ORDER = [
    "Sub-Saharan Africa",
    "S & SE Asia",
    "Rest of the World",
    "China",
    "OECD",
]

COLOR_PALETTE_1 = [
    "#6929c4",  # Purple
    "#1192e8",  # Cyan
    "#005d5d",  # Teal
    "#9c456b",  # Magenta
    "#fa4d56",  # Red
    "#570408",  # Dark red
    "#198038",  # Green
    "#002d9c",  # Blue
    "#ee538b",  # Magenta
    "#b28600",  # Yellow
    "#009d9a",  # Teal
    "#012749",  # Dark cyan
]

COLOR_PALETTE_2 = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#0032BC",
    "#a65628",
    "#f781bf",
    "#999999",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]

# 6 hues x 2 brightness levels, chosen to stay distinct and readable on white backgrounds.
COLOR_PALETTE_3 = [
    "#4C9FD6",  # blue light
    "#1F6FA8",  # blue dark
    "#F2A541",  # orange light
    "#C77900",  # orange dark
    "#36B39A",  # teal light
    "#007F6A",  # teal dark
    "#C97DB1",  # magenta light
    "#9B4F83",  # magenta dark
    "#9FAE4C",  # olive light
    "#6F7F1E",  # olive dark
    "#9A7A52",  # brown light
    "#6B4F2A",  # brown dark
]

# 6 hues x 2 brightness levels, increased lightness differences and saturation
COLOR_PALETTE_4 = [
    "#8299FD",  # blue light
    "#4A37C2",  # blue dark
    "#CF8517",  # orange light
    "#B35A00",  # orange dark
    "#4DCCB8",  # teal light
    "#0063AF",  # teal dark
    "#CC79B7",  # magenta light
    "#7B2C65",  # magenta dark
    "#94A42A",  # olive light
    "#4D5F0A",  # olive dark
    "#B89968",  # brown light
    "#4D3412",  # brown dark
]

STOCK_TYPE_BASE_COLORS = {
    "Res": "#C0392B",  # crimson-red
    "Com": "#16A085",  # teal
}

COLORS_REMIND = {
    "CAZ": "#f58231",
    "CHA": "#3cb44b",
    "MEA": "#4363d8",
    "LAM": "#96cfc8",
    "SSA": "#911eb4",
    "JPN": "#ff9999",
    "USA": "#e6194B",
    "OAS": "#800000",
    "IND": "#808000",
    "FRA": "#000075",
    "DEU": "#f032e6",
    "REF": "#9A6324",
    "World": "#404040",
    "NEU": "#42d4f4",
    "EUR": "#ffd610",
}

COLOR_PALETTE = COLOR_PALETTE_1

SSP_PICKLENAMES = {
    "SSP1": "figs_cement_SSP1_h12/model.pickle",
    "SSP2": "figs_cement_SSP2_h12/model.pickle",
    "SSP3": "figs_cement_SSP3_h12/model.pickle",
    "SSP4": "figs_cement_SSP4_h12/model.pickle",
    "SSP5": "figs_cement_SSP5_h12/model.pickle",
    "SSP1_CE": "figs_cement_SSP1_CE_h12/model.pickle",
    "SSP2_CE": "figs_cement_SSP2_CE_h12/model.pickle",
}
SSP_SOURCE_PICKLES = {ssp: PATH_CEMENT / name for ssp, name in SSP_PICKLENAMES.items()}
SSP_CACHE_DIRS = {ssp: CACHE_DIR_CEMENT / ssp for ssp in SSP_PICKLENAMES}

SSP_PALETTE = [
    pyam.plotting.PYAM_COLORS[f"AR6-{ssp}"] for ssp in ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5")
]
SSP_COLORS = {
    ssp: color for ssp, color in zip(("SSP1", "SSP2", "SSP3", "SSP4", "SSP5"), SSP_PALETTE)
}
SSP_COLORS.update(
    {
        # Circular-economy variants share their parent SSP hue; distinguished by a dashed line.
        "SSP1_CE": SSP_COLORS["SSP1"],
        "SSP2_CE": SSP_COLORS["SSP2"],
    }
)

# Line-dash style per scenario: CE variants dashed, base SSPs solid.
SSP_DASHES = {ssp: ("dash" if ssp.endswith("_CE") else "solid") for ssp in SSP_PICKLENAMES}

# Legend labels (CE variants read as "SSPx (CE)").
SSP_LABELS = {ssp: (f"{ssp[:-3]} (CE)" if ssp.endswith("_CE") else ssp) for ssp in SSP_PICKLENAMES}
