"""Regenerate the cement model pickles that the figure scripts in this folder read from data_out.

Run from the repo root: `uv run --no-sync python scritps_figures/generate_data.py`.
Runs the cement model once per scenario in constants.SSP_PICKLENAMES, with config/figs_cement.toml
stacked on config/default.toml. The overlay turns on parameter reconciliation and pickle export,
so default.toml does not need to be edited to regenerate the pickles.

Rerun this after updating remind_mfa_data to refresh the figures' input data.
"""

import logging
from pathlib import Path

from constants import PATH_CEMENT, SSP_PICKLENAMES

from remind_mfa.common.config_loader import load_config
from remind_mfa.common.helpers import ModelNames, init_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def build_cfg(scenario: str) -> dict:
    cfg = load_config(["default", "figs_cement"], ModelNames.CEMENT)
    cfg["model_switches"]["scenario"] = scenario
    cfg["export"]["path"] = str(PATH_CEMENT)
    return cfg


def main() -> None:
    for scenario, pickle_name in SSP_PICKLENAMES.items():
        logging.info("=" * 80)
        logging.info(f"Running cement model for scenario {scenario}...")
        cfg = build_cfg(scenario)
        model = init_model(cfg=cfg)
        model.run()
        model.export()

        expected_pickle = Path(PATH_CEMENT) / pickle_name
        if not expected_pickle.is_file():
            raise RuntimeError(
                f"Expected pickle {expected_pickle} was not created for scenario {scenario}. "
                "The export folder naming (prefix_model_scenario_regionmapping in "
                "common_export.py) may no longer match constants.SSP_PICKLENAMES."
            )
        logging.info(f"Wrote {expected_pickle}")


if __name__ == "__main__":
    main()
