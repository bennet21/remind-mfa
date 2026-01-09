import numpy as np
from typing import Callable, Tuple, Optional
import flodym as fd
import logging
import itertools

from copy import deepcopy
class ParameterReconciliation:
    """Parameter reconciliation of top-down and bottom-up models."""
    # TODO inherit from separate helper class
    # TODO pydantic?

    def __init__(self, prms: dict[str, fd.Parameter], dims: fd.DimensionSet, uncoupled: bool = False):

        self._year_of_reconciliation = 2020
        self._reduced_stock_type = fd.Dimension(name="Reduced Stock Type", letter="u", items=["Res", "Com"])
        # save computation time
        self._no_correction_dim_letters = ('t', 'h') # instead of df/dx, now calculating df/dd 
        # TODO allow option to bring in knowledge that parameters and stock dimensions often do not interact
        # NB this does not mean that all parameter dimensions are independent, only that output dimensions, if existant in parameters
        self.output_dims_are_independent = uncoupled
        # TODO set and skip over known sensitivity parameters
        # TODO don't forget to normalize!
        # TODO check if I can use [...] more for flodym arrays to avoid dimension issues

        self.prepare_dims(dims)
        self.prepare_prms(prms)
        self.correct_parameters()

    def prepare_dims(self, dims: fd.DimensionSet):
        self.dims = dims.replace('s', self._reduced_stock_type)
    
    def prepare_prms(self, prms: dict[str, fd.Parameter]):
        self.prms = {}
        self.prms_adj_dims = {}
        for key, val in prms.items():
            if "s" in val.dims.letters:
                val = val[{"s": self._reduced_stock_type}]
            self.prms[key] = val
            self.prms_adj_dims[key] = self.remove_fd_dims_if_present(val.dims, self._no_correction_dim_letters)
    
    @staticmethod
    def remove_fd_dims_if_present(dims: fd.DimensionSet, letters_to_remove: Tuple) -> fd.DimensionSet:
        new_dims = dims
        for letter in letters_to_remove:
            if letter in new_dims.letters:
                new_dims = new_dims.drop(letter)
        return new_dims

    def correct_parameters(self):
        self.original_prms = deepcopy(self.prms)

        self.td = self.calc_top_down_stock(self.prms)
        self.bu = self.calc_bottom_up_stock(self.prms)

        # Assume (without loss of generality) target function of td/bu = 1s
        self.pre_compute_sensitivity(self.calc_top_down_stock, self.td)
        self.pre_compute_sensitivity(self.calc_bottom_up_stock, self.bu, denominator=True)
        
        self.pre_compute_lambda()
        self.calc_corrections()

        # TODO see if I really want to do the mulitplication here, or rather in parameter extrapolation?
        self.updated_prms = self.prms
        for prm_name, c in self.correction_factors.items():
            self.updated_prms[prm_name] = self.prms[prm_name] * c

        print("done")

    def calc_bottom_up_stock(self, prm: dict[str, fd.FlodymArray]):
        stk = prm["concrete_building_mi"] * prm["building_split"] * prm["floorspace"]

        stk = stk[{"t": self._year_of_reconciliation}]

        # build up new stock where function (f) and stock type (s) are merged into reduced stock type (u)
        new_stk = fd.FlodymArray(dims=stk.dims.drop("f"))
        new_stk[{'u': 'Res'}] = stk[{'f': 'RS', 'u': "Res"}] + stk[{'f': 'RM', 'u': "Res"}]
        new_stk[{'u': 'Com'}] = stk[{'f': 'Com', 'u': "Com"}]
        new_stk = new_stk.sum_over('b')
        return new_stk
    
    def calc_top_down_stock(self, prm: dict[str, fd.FlodymArray]):
        cement_consumption = (
            (1 - prm["cement_losses"])
            * (prm["cement_production"] - prm["cement_trade"])
            * prm["stock_type_split"]
        )

        # cement stock
        stk = fd.InflowDrivenDSM(
            dims=self.dims[cement_consumption.dims.letters],
            lifetime_model=fd.LogNormalLifetime,
            time_letter="h",
        )
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
        
        # reduce redudant dimensions
        stk = stk[{"h": self._year_of_reconciliation}]
        stk = stk[{"m": "concrete"}]
        stk = stk.sum_over("a")
        return stk
    
    def pre_compute_sensitivity(self, f: Callable[[dict[str, fd.FlodymArray]], fd.FlodymArray], f0: fd.FlodymArray, denominator: bool = False):
        """
        Pre-compute sensitivity matrices for parameters used in the given model function.
        Pre-existing sensitivities are added to newly computed ones.
        """
        relevant_params = self.get_relevant_parameters(f, self.prms)

        # Initialize S_matrices dictionary if it doesn't exist
        if not hasattr(self, "S_matrices"):
            self.S_matrices = {}
    
        for prm_name in relevant_params:
            logging.info(f"Calculating sensitivity for parameter: {prm_name}")
            S_mat = self.calc_sensitivity(f, f0, prm_name, denominator=denominator)
            if prm_name in self.S_matrices:
                # TODO check if that makes sense
                logging.info(f"Sensitivity for parameter {prm_name} already exists; summing matrices.")
                self.S_matrices[prm_name] = self.S_matrices[prm_name] + S_mat
            else:
                self.S_matrices[prm_name] = S_mat

    @staticmethod
    def get_relevant_parameters(model_func: Callable, prms: dict[str, fd.Parameter]) -> set:
        """
        Runs a model once to spy on which parameters are used.
        """
        # Wrap the parameters in a tracking dict
        spy_prms = DependencyTracker(prms)
        
        # Run the model
        _ = model_func(spy_prms)

        return spy_prms.accessed_keys
    
    def calc_sensitivity(self, f: Callable[[dict[str, fd.FlodymArray]], fd.FlodymArray], f0: fd.FlodymArray, prm_name: str, denominator: bool = False):
        J = self.calc_jacobian(f, f0, prm_name)
        
        if self.output_dims_are_independent:
            # Convert FlodymArray Jacobian to numpy matrix and scale for logarithmic sensitivity
            S = self.flodym_jacobian_to_matrix(J / f0, f0.dims, self.prms_adj_dims[prm_name])
        else:
            f0_flat = self.flatten_fd_to_np(f0)[:, np.newaxis]
            S = J / f0_flat
        
        if denominator:
            return -S
        return S
    
    def calc_jacobian(self, f: Callable[[dict[str, fd.FlodymArray]], fd.FlodymArray], f0: fd.FlodymArray, prm_name: str, epsilon=1e-5):
        # TODO I could do everything with flodym by just introducing new parameter dimensions for output dimensions
        # matrix multiplication would then be (A*B).sum_over(dims), instead of A @ B
        if self.output_dims_are_independent:
            return self._calc_jacobian_independent(f, f0, prm_name)
        return self._calc_jacobian_full(f, f0, prm_name)
    
    def _calc_jacobian_independent(self, f: Callable[[dict[str, fd.FlodymArray]], fd.FlodymArray], f0: fd.FlodymArray, prm_name: str, epsilon=1e-5):
        # TODO why is prm not recognized as FlodymArray?
        prm = self.prms[prm_name]
        original_prm = self.original_prms[prm_name]

        # dims in parameter but NOT in output — must loop over these
        reduced_dims = self.remove_fd_dims_if_present(self.prms_adj_dims[prm_name], f0.dims.letters)
        combined_dims = self.prms_adj_dims[prm_name].union_with(f0.dims)

        if reduced_dims.total_size == 0:
            # No extra dims — single perturbation suffices
            prm[...] = original_prm * (1 + epsilon)
            f_perturbed = f(self.prms)
            prm[...] = original_prm
            J = (f_perturbed - f0) / epsilon
            return J
        
        J = fd.FlodymArray(dims=combined_dims, values=np.zeros(combined_dims.shape))

        for slicer in self.iter_dim_slicers(reduced_dims):
            val = original_prm[slicer]

            prm[slicer] = val * (1 + epsilon)
            f_perturbed = f(self.prms)
            J[slicer] = (f_perturbed - f0) / epsilon

            prm[slicer] = val
        
        return J

    def _calc_jacobian_full(self, f: Callable[[dict[str, fd.FlodymArray]], fd.FlodymArray], f0: fd.FlodymArray, prm_name: str, epsilon=1e-5):
        # TODO why is dims_to_adj not recognized as dimensionset?
        prm = self.prms[prm_name]
        original_prm = self.original_prms[prm_name]
        dims_to_adj = self.prms_adj_dims[prm_name]
             
        J = np.zeros((f0.size, dims_to_adj.total_size))

        for flat_idx, slicer in enumerate(self.iter_dim_slicers(dims_to_adj)):
            val = original_prm[slicer]

            # Perform perturbation (zero values are not corrected)
            prm[slicer] = val * (1 + epsilon)
            f_perturbed = f(self.prms)
            J[:, flat_idx] = self.flatten_fd_to_np(f_perturbed - f0) / epsilon
            
            # Restore original value
            prm[slicer] = val

        return J

    @staticmethod
    def iter_dim_slicers(dims: fd.DimensionSet):
        """
        Iterate over all element combinations of a DimensionSet, yielding dict slicers.
        
        Yields dicts like {'r': 'USA', 'u': 'Res'} for each element in the Cartesian product.
        Order matches numpy flatten (C-order): last dimension varies fastest.
        """
        items_per_dim = [d.items for d in dims]
        for dim_element in itertools.product(*items_per_dim):
            yield dict(zip(dims.letters, dim_element))
    
    def flatten_fd_to_np(self, arr: fd.FlodymArray) -> np.ndarray:
        """Flatten a FlodymArray into a 1D numpy array."""
        return arr.values.flatten()
    
    def flodym_jacobian_to_matrix(
        self,
        J: fd.FlodymArray,
        output_dims: fd.DimensionSet,
        param_dims: fd.DimensionSet,
    ) -> np.ndarray:
        """
        Convert a FlodymArray Jacobian into a 2D numpy sensitivity matrix.
        
        The Jacobian J has dimensions that are the union of output_dims and param_dims.
        Dimensions shared between output and parameter create block-diagonal structure:
        each element of the shared dimension only affects its corresponding output.
        
        Args:
            J: FlodymArray with dims = union(output_dims, param_dims)
            output_dims: Dimensions of the model output (e.g., region, stock_type)
            param_dims: Dimensions of the parameter being varied
            
        Returns:
            2D numpy array of shape (output_size, param_size)
        """
        output_size = output_dims.total_size
        param_size = param_dims.total_size
        S = np.zeros((output_size, param_size))
        
        # Identify shared vs unique dimensions
        shared_letters = set(output_dims.letters) & set(param_dims.letters)
        
        # Iterate over all output positions
        for out_idx, out_slicer in enumerate(self.iter_dim_slicers(output_dims)):
            # Iterate over all parameter positions
            for prm_idx, prm_slicer in enumerate(self.iter_dim_slicers(param_dims)):
                # Check if shared dimensions match
                # If they don't match, the sensitivity is zero (block-diagonal structure)
                shared_match = all(
                    out_slicer[letter] == prm_slicer[letter]
                    for letter in shared_letters
                )
                
                if shared_match:
                    # Build the combined slicer for J
                    # J has all dimensions from both output and param
                    j_slicer = {**out_slicer, **prm_slicer}
                    S[out_idx, prm_idx] = J[j_slicer].values.item()
        
        return S
    
    def pre_compute_lambda(self):
        """Solve Aλ = b for λ."""
        b = self.flatten_fd_to_np(self.td.apply(np.log) - self.bu.apply(np.log))
        D = b.size
        A = np.zeros((D, D))
        
        for prm_name, S in self.S_matrices.items():
            var_vec = self.get_sigma(prm_name)
            S_weighted = S * var_vec[np.newaxis, :]
            A += S_weighted @ S.T

        # solve for lambda
        self.lmda = np.linalg.solve(A, b)

    def get_sigma(self, prm_name: str) -> np.ndarray:
        rel_std = self.rel_std(prm_name)
        sigma = self.flatten_fd_to_np(rel_std) ** 2
        return sigma
    
    def rel_std(self, prm_name: str) -> fd.FlodymArray:
        """
        Get the relative standard deviation of a parameter.
        Returns a FlodymArray with the same dimensions as the parameter.
        """

        default_rel_std = 0.2

        rel_std = {
            "concrete_building_mi": fd.FlodymArray(
                dims = fd.DimensionSet(dim_list=[self.dims["r"]]),
                values = np.array([0.2 if self.prms["industrialized_regions"][{"r": region}].values else 0.4 for region in self.dims["r"].items])
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
        
        out = out.cast_to(self.prms_adj_dims[prm_name])
        return out
    
    def calc_corrections(self):
        self.correction_factors = {}
        # TODO self.S_matrices.keys() replace this with list of corrected parameters
        for prm_name in self.S_matrices.keys():
            log_correction = self.calc_log_correction(prm_name)
            self.correction_factors[prm_name] = log_correction.apply(np.exp)
    
    def calc_log_correction(self, prm_name: str) -> fd.FlodymArray:
        S = self.S_matrices[prm_name]
        grad = S.T @ self.lmda
        # TODO prepare sigma vector beforehand
        var_vec = self.get_sigma(prm_name)
        d = - var_vec * grad
        return self.reshape_np_to_fd(d, self.prms_adj_dims[prm_name])

    def reshape_np_to_fd(self, flat_arr: np.ndarray, target_dims: fd.DimensionSet) -> fd.FlodymArray:
        """Reshape a 1D numpy array back into a FlodymArray with the same shape as the template."""
        if flat_arr.size != target_dims.total_size:
            raise ValueError("Size of flat array does not match size of template.")
        reshaped_values = flat_arr.reshape(target_dims.shape)
        return fd.FlodymArray(dims=target_dims, values=reshaped_values)
    
    def system_model(self, prms: dict[str, fd.FlodymArray]) -> fd.FlodymArray:
        td = self.calc_top_down_stock(prms)
        bu = self.calc_bottom_up_stock(prms)
        return td / bu

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