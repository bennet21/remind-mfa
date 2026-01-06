import numpy as np
from typing import List, Callable, Dict
import flodym as fd

from remind_mfa.common.common_definition import RemindMFAParameterDefinition

class ParameterReconciliation:
    """Parameter reconciliation of top-down and bottom-up models."""

    def __init__(self, prms, dims):
        self._time = 2020
        self._reduced_stock_type = fd.Dimension(name="Reduced Stock Type", letter="u", items=["Res", "Com"])
 
        self.prepare_dims(dims)
        self.prepare_prms(prms)

        self.pre_compute_sensitivity(self.calc_bottom_up_stock)

        self.bu = self.calc_bottom_up_stock(self.prms)
        self.td = self.calc_top_down_stock(self.prms)

        self.f0 = self.system_model(self.prms)

        b = np.log(self.flatten_fd_to_np(self.td)) - np.log(self.flatten_fd_to_np(self.bu)) # this should have shape 24
        pass

    def prepare_dims(self, dims: fd.DimensionSet):
        self.dims = dims.replace('s', self._reduced_stock_type)

    def prepare_prms(self, prms: Dict[str, fd.FlodymArray]):

        self.prms = {}
        for key, val in prms.items():
            self.prms[key] = val[{"s": self._reduced_stock_type}] if "s" in val.dims.letters else val

        self.bu_params = self.get_relevant_parameters(self.calc_bottom_up_stock, self.prms)
        self.td_params = self.get_relevant_parameters(self.calc_top_down_stock, self.prms)
    
    def flatten_fd_to_np(self, arr: fd.FlodymArray) -> np.ndarray:
        """Flatten a FlodymArray into a 1D numpy array."""
        return arr.values.flatten()
    
    def reshape_np_to_fd(self, flat_arr: np.ndarray, template: fd.FlodymArray) -> fd.FlodymArray:
        """Reshape a 1D numpy array back into a FlodymArray with the same shape as the template."""
        if flat_arr.size != template.values.size:
            raise ValueError("Size of flat array does not match size of template.")
        reshaped_values = flat_arr.reshape(template.values.shape)
        return fd.FlodymArray(dims=template.dims, values=reshaped_values)
    
    def rel_std(self, prm_name: str) -> fd.FlodymArray:
        """
        Get the relative standard deviation of a parameter.
        Returns a FlodymArray with the same dimensions as the parameter.
        """

        sigma = {
            "concrete_building_mi": fd.FlodymArray(
                dims = fd.DimensionSet(dim_list=[self.dims["r"]]),
                values = np.array([0.2 if self.prms["industrialized_regions"][region].values else 0.4 for region in self.dims["r"].items])
            ),
            "building_split": 0.2,
            "floorspace": 0.2
        }

        out = sigma[prm_name]
        if isinstance(out, (float, int)):
            out = fd.FlodymArray(dims=fd.DimensionSet(dim_list=[]), values=out)
        if not isinstance(out, fd.FlodymArray):
            raise ValueError(f"Relative standard deviation for parameter {prm_name} not defined.")

        out = out.cast_to(self.prms[prm_name].dims)
        return out
    
    @staticmethod
    def get_relevant_parameters(model_func: Callable, prm: List[RemindMFAParameterDefinition]) -> set:
        """
        Runs a model once to spy on which parameters are used.
        """
        # Wrap the parameters in a tracking dict
        spy_prm = DependencyTracker(prm)
        
        # Run the model
        _ = model_func(spy_prm)

        return spy_prm.accessed_keys

    def calc_bottom_up_stock(self, prm: list[fd.FlodymArray]):
        stk = prm["concrete_building_mi"] * prm["building_split"] * prm["floorspace"]

        stk = stk[{"t": self._time}]

        new_dimletters = tuple(set(stk.dims.letters) - set('f'))
        new_stk = fd.FlodymArray(dims=stk.dims[new_dimletters])
        new_stk[{'u': 'Res'}] = stk[{'f': 'RS', 'u': "Res"}] + stk[{'f': 'RM', 'u': "Res"}]
        new_stk[{'u': 'Com'}] = stk[{'f': 'Com', 'u': "Com"}]
        new_stk = new_stk.sum_over('b')
        return new_stk
    
    def calc_top_down_stock(self, prm: list[fd.FlodymArray]):
        cement_consumption = (
            (1 - prm["cement_losses"])
            * (prm["cement_production"] - prm["cement_trade"])
            * prm["stock_type_split"]
        )

        stk = fd.InflowDrivenDSM(
            dims=self.dims[cement_consumption.dims.letters],
            lifetime_model=fd.LogNormalLifetime,
            time_letter="h",
        )
        # cement stock
        stk
        stk.inflow[...] = cement_consumption
        stk.lifetime_model.set_prms(
            mean=prm["use_lifetime_mean"],
            std=prm["use_lifetime_rel_std"] * prm["use_lifetime_mean"],
        )
        stk.compute()

        # product stock
        cement_ratio = (
            prm["product_cement_content"] / prm["product_density"]
        )
        stk = stk.stock * (
            prm["product_material_split"]
            * prm["product_material_application_transform"]
            * prm["product_application_split"]
            / cement_ratio
        )

        stk = stk[{"h": self._time}]
        stk = stk[{"m": "concrete"}]
        stk = stk.sum_over("a")
        return stk
    
    def system_model(self, prm: list[fd.FlodymArray]):
        return self.calc_top_down_stock(prm) / self.calc_bottom_up_stock(prm)

    def pre_compute_sensitivity(self, f: Callable):
        """Pre-compute sensitivity matrices for parameters used in the given model function."""
        f0 = f(self.prms)

        relevant_params = self.get_relevant_parameters(f, self.prms)
        self.S_matrices = {}
        for prm_name in relevant_params:
            S_mat = self.calc_sensitivity(f, f0, prm_name)
            self.S_matrices[prm_name] = S_mat

    def calc_sensitivity(self, f: Callable, f0: fd.FlodymArray, prm_name: str):
        J = self.calc_jacobian(f, f0, prm_name)
        
        prm_flat = self.flatten_fd_to_np(self.prms[prm_name])
        f0_flat = self.flatten_fd_to_np(f0)

        prm_row = prm_flat[np.newaxis, :]
        f0_col = f0_flat[:, np.newaxis]
        scaling = prm_row / f0_col

        S = J * scaling

        return S
    
    def calc_jacobian(self, f: Callable, f0: np.ndarray, prm_name: str, epsilon=1e-5):
        """
        Use finite differences to calculate the gradient of a function with respect to one parameter.
        Expects that f0 is flat and f returns a flat array.
        """
        prm = self.prms[prm_name].values

        ouptut_dim = f0.size
        param_dim = prm.size

        J = np.zeros((ouptut_dim, param_dim))

        for i in range(param_dim):
            idx = np.unravel_index(i, prm.shape)
            val = prm[idx]

            # Calculate step size h
            h = val * epsilon if val != 0 else epsilon

            # Perform perturbation
            prm[idx] = val + h
            f_perturbed = f(self.prms)

            # Calculate gradient
            J[:, i] = self.flatten_fd_to_np(f_perturbed - f0) / h

            # Restore original value
            prm[idx] = val

        return J

    def correct_parameters(self):
        correction_factors = self.calc_correction()
        parameters = parameters * correction_factors
    
    def calc_correction(self):
        return np.exp(self.calc_log_correction())

    def calc_log_correction(self):
        d = self.O * self.B.T * self.lmda * self.S
        return d

    def calc_lambda(self):
        """Solve Aλ = b for λ."""
        A = np.zeros((self.n_dims, self.n_dims))

        for i in range(self.n_params):
            A += self.S[i] @ self.B[i] @ self.O[i] @ self.B[i].T @ self.S[i]


class DependencyTracker(dict):
    """Dictionary that tracks accessed keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_keys = set()

    def __getitem__(self, key):
        # 1. Record that this key was used
        self.accessed_keys.add(key)
        
        # 2. Return the actual value so the math doesn't crash
        return super().__getitem__(key)