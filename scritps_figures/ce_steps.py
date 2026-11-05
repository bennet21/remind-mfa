"""Step-by-step application of the circular-economy (CE) measures of a CE scenario (figure 11).

A CE scenario (e.g. SSP2_CE) differs from its parent SSP by a set of scenario parameters.
`CE_STEPS` groups these parameters into measures and orders them by the CE aspects narrow,
slow and close (Bocken et al., 2016), following the order of Pauliuk et al. Narrow comprises
more intensive use, lighter design, less clinker and cement, and timber substitution. Step k of a `CEStepChain` is the parent
scenario plus the CE rows of the parameters of steps 1..k (see
`model_switches.scenario_parameter_subset`), so the last step equals the full CE scenario. No
additional scenario files are needed: the steps are built from the existing CE scenario CSV.
Measures whose parameters the CE scenario does not set are skipped.
"""

from dataclasses import dataclass
from pathlib import Path

import flodym as fd

from remind_mfa.common.helpers import ModelNames
from remind_mfa.common.scenarios import ScenarioReader

SCENARIOS_PATH = Path("config/scenarios")


@dataclass(frozen=True)
class CEStep:
    """One CE measure of the waterfall.

    Attributes:
        label: Display name of the measure.
        aspect: CE aspect (narrow, slow or close), used for the bar colour.
        parameters: Scenario parameters the measure consists of.
    """

    label: str
    aspect: str
    parameters: tuple[str, ...]


CE_STEPS = [
    CEStep("Less vacancy", "Narrow", ("unoccupied_residential_stock_share",)),
    CEStep("Lighter design", "Narrow", ("stock_factor", "concrete_building_mi")),
    CEStep("Less clinker/cement", "Narrow", ("clinker_ratio", "cement_ratio")),
    CEStep("More timber", "Narrow", ("min_timber_share",)),
    CEStep("Longer lifetimes", "Slow", ("lifetime_factor_nurture", "lifetime_factor_nature")),
    CEStep("Concrete reuse", "Close", ("reuse_share",)),
]


class CEStepChain:
    """Ordered CE measures leading from the parent SSP of a CE scenario to the CE scenario."""

    def __init__(self, ce_scenario: str, steps: list[CEStep] = CE_STEPS):
        self.ce_scenario = ce_scenario
        reader = ScenarioReader(
            name=ce_scenario,
            base_path=str(SCENARIOS_PATH),
            model=ModelNames.CEMENT,
            dims=fd.DimensionSet(dim_list=[]),
            parameter_definitions=[],
        )
        scenario = reader.read_single(ce_scenario)
        scenario.filter_data_by_model(ModelNames.CEMENT)
        self.parent = scenario.parent
        ce_parameters = {data_point.parameter for data_point in scenario.data}
        self._check_coverage(ce_parameters, steps)
        self.steps = [
            step for step in steps if any(name in ce_parameters for name in step.parameters)
        ]

    def _check_coverage(self, ce_parameters: set[str], steps: list[CEStep]):
        covered = {name for step in steps for name in step.parameters}
        missing = sorted(ce_parameters - covered)
        if missing:
            raise ValueError(
                f"Scenario {self.ce_scenario} sets parameters {missing} that no CE step "
                "covers. Add them to a step in ce_steps.CE_STEPS."
            )

    def parameter_subset(self, step_number: int) -> list[str]:
        """Scenario parameters applied up to and including step `step_number` (1-based)."""
        return [name for step in self.steps[:step_number] for name in step.parameters]

    def intermediate_step_numbers(self) -> range:
        """Steps that need their own model run: all but the last, which is the CE scenario."""
        return range(1, len(self.steps))

    def run_name(self, step_number: int) -> str:
        """Name of the model run of step `step_number`, used for its figure data cache."""
        if step_number == 0:
            return self.parent
        if step_number == len(self.steps):
            return self.ce_scenario
        return f"{self.ce_scenario}_step{step_number}"
