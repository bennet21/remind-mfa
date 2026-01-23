from remind_mfa.common.common_mfa_system import CommonMFASystem
from remind_mfa.cement.cement_config import CementCfg
from remind_mfa.cement.cement_stock_models import CementStockModels


class InflowDrivenHistoricCementMFASystem(CommonMFASystem):

    cfg: CementCfg

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_in_use_stock()
        self.compute_flows()
        self.check_mass_balance()
        self.check_flows()

    def compute_in_use_stock(self):
        stk = self.stocks["historic_cement_in_use"]
        stk = CementStockModels.calc_cement_stock_top_down(self.parameters, stk)

    def compute_flows(self):
        flw = self.flows
        stk = self.stocks

        flw["sysenv => use"][...] = stk["historic_cement_in_use"].inflow
        flw["use => sysenv"][...] = stk["historic_cement_in_use"].outflow
