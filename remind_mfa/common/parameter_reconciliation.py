import numpy as np
from typing import List, Callable, Dict, Tuple
import flodym as fd
import logging

from remind_mfa.common.common_definition import RemindMFAParameterDefinition

class ParameterReconciliation:
    """Parameter reconciliation of top-down and bottom-up models."""

    def __init__(self, cfg, prms, dims):
        self.cfg = cfg

        self._time = 2020
        self._reduced_stock_type = fd.Dimension(name="Reduced Stock Type", letter="u", items=["Res", "Com"])
        # save computation time
        self._no_correction_dim_letters = ('t', 'h') # instead of df/dx, now calculating df/dd 
        # TODO allow option to bring in knowledge that parameters and stock dimensions often do not interact
        # TODO set and skip over known sensitivity parameters


        self.prepare_dims(dims)
        self.prepare_prms(prms)

        self.bu = self.calc_bottom_up_stock(self.prms)
        self.td = self.calc_top_down_stock(self.prms)

        self.pre_compute_sensitivity(self.calc_bottom_up_stock, self.bu, denominator=True)
        self.pre_compute_sensitivity(self.calc_top_down_stock, self.td)

        self.pre_compute_lambda()
        self.calc_corrections()

        print(self.system_model(self.prms))
        self.correct_parameters()
        print(self.system_model(self.prms))


        self.f0 = self.system_model(self.prms)

    def letters_to_drop(self, dims: fd.DimensionSet) -> Tuple:
        drop_letters = tuple(dl for dl in self._no_correction_dim_letters if dl in dims.letters)
        return drop_letters

    def extract_wanted_dims(self, dims: fd.DimensionSet, drop_letters: Tuple = None) -> fd.DimensionSet:
        """Extract relevant dimensions."""
        if not drop_letters:
            drop_letters = self.letters_to_drop(dims)
        new_dims = dims
        for drop_letter in drop_letters:
            new_dims = new_dims.drop(drop_letter)
        return new_dims
        
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

    def _get_parameter_space(self, prm_name: str) -> Tuple[fd.FlodymArray, Tuple[str, ...], fd.DimensionSet]:
        prm = self.prms[prm_name]
        drop_letters = self.letters_to_drop(prm.dims)
        correction_dims = self.extract_wanted_dims(prm.dims, drop_letters)
        return prm, drop_letters, correction_dims

    def _build_template(self, dims: fd.DimensionSet) -> fd.FlodymArray:
        values = np.zeros(dims.total_size, dtype=float).reshape(dims.shape)
        return fd.FlodymArray(dims=dims, values=values)

    def _reshape_vector_to_parameter(self, prm_name: str, flat_arr: np.ndarray) -> fd.FlodymArray:
        _, _, correction_dims = self._get_parameter_space(prm_name)
        template = self._build_template(correction_dims)
        return self.reshape_np_to_fd(flat_arr, template)
    
    def rel_std(self, prm_name: str) -> fd.FlodymArray:
        """
        Get the relative standard deviation of a parameter.
        Returns a FlodymArray with the same dimensions as the parameter.
        """

        default_rel_std = 0.2

        rel_std = {
            "concrete_building_mi": fd.FlodymArray(
                dims = fd.DimensionSet(dim_list=[self.dims["r"]]),
                values = np.array([0.2 if self.prms["industrialized_regions"][region].values else 0.4 for region in self.dims["r"].items])
            ),
            "building_split": 0.2,
            "floorspace": 0.2
        }

        out = rel_std.get(prm_name)
        if out is None:
            logging.warning(
                "Relative standard deviation missing for %s; using default %f",
                prm_name,
                default_rel_std,
            )
            out = default_rel_std
        
        if isinstance(out, (float, int)):
            out = fd.FlodymArray(dims=fd.DimensionSet(dim_list=[]), values=np.array(out))
        if not isinstance(out, fd.FlodymArray):
            raise ValueError(f"Relative standard deviation for parameter {prm_name} not defined.")
        
        output_dims = self.extract_wanted_dims(self.prms[prm_name].dims)
        out = out.cast_to(output_dims)
        return out
    
    def get_sigma(self, prm_name: str) -> np.ndarray:
        """Get the standard deviation of a parameter."""
        rel_std = self.rel_std(prm_name)
        sigma = self.flatten_fd_to_np(rel_std) ** 2
        return sigma
    
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
    
    def system_model(self, prm: List[fd.FlodymArray]):
        return self.calc_top_down_stock(prm) / self.calc_bottom_up_stock(prm)

    def pre_compute_sensitivity(self, f: Callable, f0, denominator: bool = False):
        """Pre-compute sensitivity matrices for parameters used in the given model function."""
        relevant_params = self.get_relevant_parameters(f, self.prms)
        if not hasattr(self, "S_matrices") or not self.S_matrices:
            self.S_matrices = {}
    
        for prm_name in relevant_params:
            logging.info(f"Calculating sensitivity for parameter: {prm_name}")
            if prm_name in self.S_matrices:
                # TODO I can just sum the S matrices: self.S_matrices[prm_name] += S_mat
                raise ValueError(f"Sensitivity for parameter {prm_name} already computed.",
                                 "Parameters present in both bottom-up and top-down models currently not supported.")
            S_mat = self.calc_sensitivity(f, f0, prm_name, denominator=denominator)
            self.S_matrices[prm_name] = S_mat

    def calc_sensitivity(self, f: Callable, f0: fd.FlodymArray, prm_name: str, denominator: bool = False):
        J = self.calc_jacobian(f, f0, prm_name)
        
        # scaling by f0 for logarithmic sensitivity
        f0_flat = self.flatten_fd_to_np(f0)
        f0_col = f0_flat[:, np.newaxis]

        S = J / f0_col
        
        if denominator:
            return -S
        
        return S
    
    def calc_jacobian(self, f: Callable, f0: np.ndarray, prm_name: str, epsilon=1e-5):
        """
        Use finite differences to calculate the gradient of a function with respect to one parameter.
        Expects that f0 is flat and f returns a flat array.
        """
        prm, drop_letters, correction_dims = self._get_parameter_space(prm_name)
        
        output_dim = f0.size
        param_dim = correction_dims.total_size
        J = np.zeros((output_dim, param_dim))

        for flat_idx, red_idx in enumerate(np.ndindex(*correction_dims.shape)):
            full_idx = self._build_full_index(prm.dims, correction_dims, red_idx, drop_letters)
            val = prm.values[full_idx].copy()

            # Perform perturbation
            prm.values[full_idx] = val * (1 + epsilon)
            f_perturbed = f(self.prms)

            # Calculate gradient
            J[:, flat_idx] = self.flatten_fd_to_np(f_perturbed - f0) / epsilon

            # Restore original value
            prm.values[full_idx] = val

        return J
    
    @staticmethod
    def _build_full_index(prm_dims, correction_dims, red_idx, drop_letters):
        full_letters = prm_dims.letters
        red_letters = correction_dims.letters
        red_axes = dict(zip(red_letters, red_idx))
        full_idx = []
        for dl in full_letters:
            if dl in drop_letters:
                full_idx.append(slice(None))
            else:
                full_idx.append(red_axes[dl])
        return tuple(full_idx)

    def pre_compute_lambda(self):
        """Solve Aλ = b for λ."""

        b = np.log(self.flatten_fd_to_np(self.td)) - np.log(self.flatten_fd_to_np(self.bu))
        D = b.size
        A = np.zeros((D, D))
        for prm_name, S in self.S_matrices.items():
            var_vec = self.get_sigma(prm_name)
            S_weighted = S * var_vec[np.newaxis, :]
            A += S_weighted @S.T

        # solve for lambda
        self.lmda = np.linalg.solve(A, -b)
    
    def calc_corrections(self):
        self.correction_factors = {}
        for prm_name in self.S_matrices.keys():
            log_correction = self.calc_log_correction(prm_name)
            self.correction_factors[prm_name] = log_correction.apply(np.exp)

    def calc_log_correction(self, prm_name: str) -> fd.FlodymArray:
        var_vec = self.get_sigma(prm_name)
        S = self.S_matrices[prm_name]
        grad = S.T @ self.lmda
        d = var_vec * grad
        return self._reshape_vector_to_parameter(prm_name, d)

    def correct_parameters(self):
        for prm_name, c in self.correction_factors.items():
            self.prms[prm_name] = self.prms[prm_name] * c


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