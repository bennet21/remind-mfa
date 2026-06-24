import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import questionary
from run_remind_mfa import configure_logger, read_model_config, init_model

MATERIALS = ["cement", "steel", "plastics"]
SSPS = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]


def checkbox_with_all(message: str, items: list) -> list:
    choices = [questionary.Choice("all", checked=True)] + [
        questionary.Choice(item, checked=False) for item in items
    ]
    result = questionary.checkbox(f"{message} (Space to toggle, Enter to confirm)", choices=choices).ask()
    if result is None:
        return []
    return items if "all" in result else result


def prompt_selections():
    materials = checkbox_with_all("Which materials should be calculated?", MATERIALS)
    ssps = checkbox_with_all("Which SSPs should be run?", SSPS)
    reconciliation = questionary.select(
        "Enable reconciliation?",
        choices=["on", "off", "both"],
        default="on",
    ).ask()
    return materials, ssps, reconciliation


def run_combination(material: str, ssp: str, reconciliation: bool):
    logging.info(f"Running {material} / {ssp} (reconciliation={reconciliation})")
    cfg = read_model_config(f"config/{material}.yml")
    cfg["model_switches"]["scenario"] = ssp
    if material == "cement":
        cfg["model_switches"]["parameter_reconciliation"]["do_reconcile"] = reconciliation
        cfg["model_switches"]["parameter_reconciliation"]["do_combine_mfas"] = reconciliation
    model = init_model(cfg)
    model.run()
    model.export()
    model.visualize()


if __name__ == "__main__":
    configure_logger()
    materials, ssps, reconciliation = prompt_selections()
    if not materials or not ssps:
        logging.warning("No materials or SSPs selected. Exiting.")
        sys.exit(0)
    reconciliation_values = [True, False] if reconciliation == "both" else [reconciliation == "on"]
    for material in materials:
        for ssp in ssps:
            for rec in reconciliation_values:
                run_combination(material, ssp, rec)
