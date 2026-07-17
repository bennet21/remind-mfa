from pathlib import Path

import flodym as fd

PATH_CEMENT = Path("data/cement/output/export/pickle")
CACHE_DIR_CEMENT = Path("data/cement/output/cache")
FIGURES_DIR = Path("data/cement/output/figures")
LAST_HISTORICAL_YEAR = 2023

CEMENT_PICKLENAME = "model_cement_SSP2_h12_2026-07-17--12-20-40.pickle"
SOURCE_PICKLE = PATH_CEMENT / CEMENT_PICKLENAME

# Masks
CEMENT_MASK = {"k": "cement"}
CONCRETE_MASK = {"m": "concrete"}
STOCK_TYPE_DIM = fd.Dimension(name="Reduced Stock Type", letter="u", items=["Res", "Com"])
STOCK_TYPE_MASK = {"s": STOCK_TYPE_DIM}
TOTAL_MASK = {**CEMENT_MASK, **CONCRETE_MASK, **STOCK_TYPE_MASK}

STRUCTURE_DISPLAY_NAMES = {
    "C": "Concrete buildings",
    "M": "Masonry buildings",
    "T": "Timber buildings",
    "S": "Steel buildings",
    "nan": "Other (industrial, civil, masonry)",
}

# Structure items treated as the non-building "Other" category (stacked first, split into sub-categories).
OTHER_STRUCTURE_KEYS = {"nan", "N/A"}

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
    "SSP1": "model_cement_SSP1_h12_2026-06-24--16-48-39.pickle",
    "SSP2": "model_cement_SSP2_h12_2026-06-24--16-50-55.pickle",
    "SSP3": "model_cement_SSP3_h12_2026-06-24--16-53-17.pickle",
    "SSP4": "model_cement_SSP4_h12_2026-06-24--16-55-36.pickle",
    "SSP5": "model_cement_SSP5_h12_2026-06-24--16-58-24.pickle",
}
SSP_SOURCE_PICKLES = {ssp: PATH_CEMENT / name for ssp, name in SSP_PICKLENAMES.items()}
SSP_CACHE_DIRS = {ssp: CACHE_DIR_CEMENT / ssp for ssp in SSP_PICKLENAMES}
SSP_COLORS = {
    "SSP1": "#2166AC",  # blue
    "SSP2": "#4DAC26",  # green
    "SSP3": "#D01C8B",  # magenta
    "SSP4": "#F1A340",  # orange
    "SSP5": "#762A83",  # purple
}