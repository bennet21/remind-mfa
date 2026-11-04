"""Regenerate the cement model pickles that the figure scripts in this folder read from data_out.

Run from the repo root: `uv run --no-sync python scritps_figures/generate_data.py`, optionally
followed by `--region-mapping <h12|gtap155>` (default: constants.REGION_MAPPING) and a subset of
the scenarios (default: constants.SCENARIOS), e.g.
`uv run --no-sync python scritps_figures/generate_data.py --region-mapping gtap155 SSP2`.
Runs the cement model twice per scenario, with config/figs_cement.toml stacked on config/default.toml:
- reconciled: parameter reconciliation on (set in the overlay), exported with RECONCILED_PREFIX.
- unreconciled: parameter reconciliation off, exported with UNRECONCILED_PREFIX.

After each run, the data needed by the figures is extracted into a small cache
(helpers.figure_cache_paths), so the figure scripts do not need to load the model pickles.

Rerun this after updating remind_mfa_data to refresh the figures' input data.
"""

import argparse
import logging

from constants import (
    PATH_CEMENT,
    RECONCILED_PREFIX,
    REGION_MAPPING,
    SCENARIOS,
    UNRECONCILED_PREFIX,
    source_pickle,
)

from helpers import extract_reconciled, extract_unreconciled, figure_cache_paths, save_pickle

from remind_mfa.common.config_loader import load_config
from remind_mfa.common.helpers import ModelNames, init_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def build_cfg(scenario: str, reconcile: bool, region_mapping: str) -> dict:
    cfg = load_config(["default", "figs_cement"], ModelNames.CEMENT)
    cfg["input"]["region_mapping"] = region_mapping
    cfg["model_switches"]["scenario"] = scenario
    cfg["model_switches"]["parameter_reconciliation"] = reconcile
    cfg["export"]["path"] = str(PATH_CEMENT)
    cfg["export"]["prefix"] = RECONCILED_PREFIX if reconcile else UNRECONCILED_PREFIX
    return cfg


def run(scenario: str, reconcile: bool, region_mapping: str) -> None:
    label = "reconciled" if reconcile else "unreconciled"
    logging.info("=" * 80)
    logging.info(f"Running {label} cement model for scenario {scenario}...")
    model = init_model(cfg=build_cfg(scenario, reconcile, region_mapping))
    model.run()
    model.export()

    expected_pickle = source_pickle(scenario, reconcile, region_mapping)
    if not expected_pickle.is_file():
        raise RuntimeError(
            f"Expected pickle {expected_pickle} was not created for scenario {scenario}. "
            "The export folder naming (prefix_model_scenario_regionmapping in "
            "common_export.py) may no longer match constants.source_pickle."
        )
    logging.info(f"Wrote {expected_pickle}")

    # extract the figure data while the model is in memory, so that the figure scripts do not
    # need to load the (large) model pickle
    cache_path = figure_cache_paths(scenario, region_mapping)[label]
    extract = extract_reconciled if reconcile else extract_unreconciled
    save_pickle(extract(model), cache_path)
    logging.info(f"Wrote {cache_path}")


def main(scenarios: list[str], region_mapping: str) -> None:
    for scenario in scenarios:
        run(scenario, reconcile=True, region_mapping=region_mapping)
        run(scenario, reconcile=False, region_mapping=region_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the input data of the cement figures.")
    parser.add_argument("scenarios", nargs="*", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--region-mapping", default=REGION_MAPPING)
    args = parser.parse_args()
    main(args.scenarios, args.region_mapping)
