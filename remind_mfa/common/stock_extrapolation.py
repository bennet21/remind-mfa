import flodym as fd
import numpy as np
from typing import Tuple, Union, Optional
from pydantic import ConfigDict

from remind_mfa.common.data_transformations import broadcast_trailing_dimensions, BoundList
from remind_mfa.common.assumptions_doc import add_assumption_doc
from remind_mfa.common.helpers import RegressOverModes, RemindMFABaseModel
from remind_mfa.common.common_config import ModelSwitches
from remind_mfa.common.fit_stocks import StockFitter


class StockExtrapolation(RemindMFABaseModel):
    """
    Class for extrapolating stocks based on historical data and GDP per capita.
    """

    model_config = ConfigDict(extra="allow")

    cfg: ModelSwitches
    """Configuration for the model."""
    historic_stocks: fd.FlodymArray
    """Historical stock data."""
    additional_stock_data: Optional[fd.FlodymArray] = None
    """additional data used for the extrapolation, like an expert guess"""
    additional_stock_data_weight: Optional[float] = 0.0
    """relative weight of the additional stock data in the regression.
    only used if additional_stock_data is not None.
    """
    dims: fd.DimensionSet
    """Dimension set for the data."""
    parameters: dict[str, fd.Parameter]
    """Parameters for the extrapolation."""
    target_dim_letters: Union[Tuple[str, ...], str] = "all"
    """Sets the dimensions of the stock extrapolation output. If "all", the output will have the same shape as historic_stocks, except for the time dimension. Defaults to "all"."""
    indep_fit_dim_letters: Union[Tuple[str, ...], str] = ()
    """indep_fit_dim_letters (Union[Tuple[str, ...]], str): Sets the dimensions across which an individual fit is performed, must be subset of target_dim_letters. If "all", all dimensions given in target_dim_letters are regressed individually. If empty (), all dimensions are regressed aggregately. Defaults to ()."""
    bound_list: BoundList = BoundList()
    """bound_list (BoundList): List of bounds for the extrapolation. Defaults to an empty BoundList."""
    do_gdppc_accumulation: bool = True
    """do_gdppc_accumulation (bool): Flag to perform GDP per capita accumulation. Defaults to True."""
    transition_smoothing: str = "critically_damped"
    """transition_smoothing (str): Method for blending between historical and future stock. Possible values are "critically_damped", "shift_zeroth_order", "none". Defaults to "critically_damped"."""
    lifetime: Optional[fd.FlodymArray] = None
    """lifetime of the stock, used to determine the number of timesteps that are used for the average slope calculation in the critically damped blend."""

    def extrapolate(self):
        """Preprocessing and extrapolation."""
        self.set_dims(self.indep_fit_dim_letters)
        self.init_arrays()
        self.calc_arrays_from_parameters_dict()
        self.set_predictor()
        self.get_pure_regression()
        self.get_pure_regression_single_predictor()
        self.fit()
        self.transform_two_predictor_regression()
        self.smooth_transition()
        # transform back to total stocks
        self.stocks[...] = self.stocks_pc * self.pop
        return self

    def set_dims(self, indep_fit_dim_letters: Tuple[str, ...]):
        """
        Check target_dim_letters.
        Set fit_dim_letters and check:
        fit_dim_letters should be the same as target_dim_letters, but without the time dimension, except if otherwise defined.
        In this case, fit_dim_letters should be a subset of target_dim_letters.
        This check cannot be performed if self.target_dim_letters or self.fit_dim_letters is None.
        """
        if self.target_dim_letters == "all":
            self.historic_dim_letters = self.historic_stocks.dims.letters
            self.target_dim_letters = ("t",) + self.historic_dim_letters[1:]
        else:
            self.historic_dim_letters = ("h",) + self.target_dim_letters[1:]

        if indep_fit_dim_letters == "all":
            # fit_dim_letters should be the same as target_dim_letters, but without the time dimension
            self.indep_fit_dim_letters = tuple(x for x in self.target_dim_letters if x != "t")
        else:
            self.indep_fit_dim_letters = indep_fit_dim_letters
            if not set(self.indep_fit_dim_letters).issubset(self.target_dim_letters):
                raise ValueError("fit_dim_letters must be subset of target_dim_letters.")
        self.get_fit_idx()

    def get_fit_idx(self):
        """Get the indices of the fit dimensions in the historic_stocks dimensions."""
        self.fit_dim_idx = tuple(
            i
            for i, x in enumerate(self.historic_stocks.dims.letters)
            if x in self.indep_fit_dim_letters
        )

    def init_arrays(self):
        """Initialize arrays for helpers and stocks in different versions:
        Only the historic part, per capita, and so on
        """
        self.historic_pop = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_gdppc = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_stocks_pc = fd.StockArray(dims=self.dims[self.historic_dim_letters])
        if self.additional_stock_data is not None:
            self.additional_stock_data_pc = fd.StockArray(dims=self.dims_out)
        self.stocks_pc = fd.StockArray(dims=self.dims_out)
        self.stocks = fd.StockArray(dims=self.dims_out)

    def calc_arrays_from_parameters_dict(self):
        """Calc drivers (GDP and population) and various variations of it"""
        self.pop = self.parameters["population"]
        self.gdppc = self.parameters["gdppc"]
        self.adapt_gdppc()
        self.historic_pop[...] = self.pop[{"t": self.dims["h"]}]
        self.historic_gdppc[...] = self.gdppc[{"t": self.dims["h"]}]
        self.historic_stocks_pc[...] = self.historic_stocks / self.historic_pop
        if self.additional_stock_data is not None:
            self.additional_stock_data_pc[...] = self.additional_stock_data / self.pop

    def adapt_gdppc(self):
        if self.do_gdppc_accumulation:
            add_assumption_doc(
                type="model assumption",
                name="Usage of cumulative GDP per capita",
                description=(
                    "Accumulated GDPpc is used for stock extrapolation to prevent "
                    "stock shrink in times of decreasing GDPpc. "
                ),
            )
            self.gdppc = self.gdppc.apply(np.maximum.accumulate, kwargs=dict(axis=0))
        self.gdppc = self.gdppc.cast_to(self.dims_out)

    def set_predictor(self):
        """wrapper for the get_predictor function using class attributes.
        get_predictor is designed to be called with other gdp values, e.g. for visualization
        """
        self.predictor = self.get_predictor(self.gdppc.values)

    def get_predictor(self, gdppc: np.ndarray) -> np.ndarray:
        """Get regression predictor: Can be either log GDP per capita or a combination of log GDP per capita and time.
        In all cases, the predictor is standardized to have a mean of 0 and a range of approximately 1.

        Args:
            gdppc (np.ndarray): GDP per capita values to use; having this as an input allows to use
              different GDP per capita values for visualization of the regression predictor, for example.

        Returns:
            np.ndarray: The predictor values for the regression
        """
        # Standardize time and GDPpc:
        # The mean is data-driven and therefore dependent on the predictor, but also unit-independent.
        # Nevertheless, this should not have an impact on the regression result, as it only shifts the values.
        # The range is hardcoded in units of years/$, which makes it independent of the predictor data.
        # This is necessary to ensure comparable regression results on different predictor data.
        # If other units are used, the range needs to be adapted accordingly.
        time = np.array(self.dims["t"].items)
        normalized_time = (time - np.mean(time)) / (2100 - 1900)
        log_gdppc = np.log10(gdppc)
        normalized_gdppc = (log_gdppc - np.mean(log_gdppc)) / (np.log10(1e5) - np.log10(1e3))

        match self.cfg.regress_over:
            case RegressOverModes.LOGGDPPC:
                return normalized_gdppc
            case RegressOverModes.LOGGDPPC_TIME:
                time = broadcast_trailing_dimensions(normalized_time, normalized_gdppc)
                predictor = np.empty(gdppc.shape, dtype=[("x1", np.float64), ("x2", np.float64)])
                predictor["x1"] = normalized_gdppc
                predictor["x2"] = time
                return predictor

    def get_pure_regression(self):
        """Regress over the chosen predictor, common for all regions.
        The extrapolation object contains the pure regression result without any correction or
        fitting to the historic stocks.
        """
        all_weights = (self.gdppc * self.pop).get_shares_over(("r",))
        historic_weights = all_weights[{"t": self.dims["h"]}]
        if self.additional_stock_data is None:
            data_to_extrapolate = self.historic_stocks_pc.values
            predictor_values = self.predictor
            weights = historic_weights.values
        else:
            # we prepend the common regression as additional "historic" data with lower weighting
            historic_in = self.historic_stocks_pc.values
            additional = self.additional_stock_data_pc.values
            data_to_extrapolate = np.concatenate([additional, historic_in], axis=0)
            weights = np.concatenate(
                [
                    all_weights.values * self.additional_stock_data_weight,
                    historic_weights.values,
                ],
                axis=0,
            )
            predictor_values = np.concatenate(
                [self.predictor, self.predictor],
                axis=0,
            )
        self.extrapolation = self.cfg.stock_extrapolation_class(
            data_to_extrapolate=data_to_extrapolate,
            predictor_values=predictor_values,
            independent_dims=self.fit_dim_idx,
            bound_list=self.bound_list,
            weights=weights,
        )
        pure_regression = self.extrapolation.regress()
        if self.additional_stock_data is not None:
            # remove prepended additional data
            pure_regression = pure_regression[self.dims_out["t"].len :, ...]

        self.export_pure_parameters()

    def get_pure_regression_single_predictor(self):
        """Get a single-predictor regression based only on GDP per capita, which is used for the stock fitting.
        Historic stocks are divided by the time-dependent extrapolation function and then regressed only over GDP per capita."""
        if self.cfg.regress_over == RegressOverModes.LOGGDPPC_TIME:
            # transform historic stocks by dividing by the time-dependent part of the regression
            prms = [self.extrapolation.fit_prms[np.newaxis, ..., i] for i in range(self.extrapolation.n_prms)]
            time_dependent_func = self.extrapolation.func(self.extrapolation.predictor_values, prms, factor = "f3")[: self.n_historic]
            data_to_extrapolate = self.extrapolation.data_to_extrapolate / time_dependent_func
            self.single_predictor = self.predictor["x1"]
            # adapt the bounds
            new_bounds = [b for b in self.bound_list.bound_list if b.var_name != "x2_growth_rate"]
            for i, b in enumerate(new_bounds):
                if b.var_name == 'x1_growth_rate':
                    new_bounds[i].var_name = "growth_rate"
                    break
            bound_list = BoundList(
                target_dims=self.dims[self.indep_fit_dim_letters],
                bound_list=new_bounds,
            )
            self.extrapolation_single_predictor = self.cfg.stock_extrapolation_class.single_predictor_cls(
                data_to_extrapolate=data_to_extrapolate,
                predictor_values=self.single_predictor,
                independent_dims=self.fit_dim_idx,
                bound_list=bound_list,
                weights=self.extrapolation.weights,
            )
            self.extrapolation_single_predictor.regress()
            self.stocks_to_fit = fd.StockArray(dims=self.dims[self.historic_dim_letters], values=data_to_extrapolate)
        else:
            self.extrapolation_single_predictor = self.extrapolation
            self.stocks_to_fit = self.historic_stocks_pc
            self.single_predictor = self.predictor

    def export_pure_parameters(self):
        """for export to csv, used in plastics
        TODO: check if still needed
        """
        if self.indep_fit_dim_letters:
            parameter_dims: fd.DimensionSet = self.dims[self.indep_fit_dim_letters]
        else:
            parameter_dims = fd.DimensionSet(dim_list=[])
        parameter_names = fd.Dimension(
            name="Parameter Names", letter="p", items=self.extrapolation.prm_names
        )
        parameter_dims = parameter_dims.expand_by([parameter_names])
        self.pure_parameters = fd.FlodymArray(
            dims=parameter_dims, values=self.extrapolation._fit_prms
        )

    def fit(self):
        """Makes region-specific alterations to the common regression parameters to better fit
        historic stock trends.
        Minimization of a penalty function is used to find a good compromise between fitting the
        historical data and keeping the regression parameters close to the pure regression.
        Details on the penalty function and the optimization can be found in the StockFitter class.
        """

        # Note that the weights do not necessarily reflect the importance of different metrics during the optimization.
        # This is due to the fact that different metrics may operate on different regimes and have different units.
        penalty_weights = {
            "data_0th_order": 0.2,
            "rel_data_0th_order": 0.25, 
            "data_1st_order": 0.4,  
            "prms": np.array(
                [
                    0.4,  # saturation_level
                    0.1,  # offset
                    0.2,  # growth_rate
                ]
            ),
        }
        # order of magnitude of a realistic, but significant change / discrepancy
        order_of_magnitude = {
            "data_0th_order": 0.1,
            "rel_data_0th_order": 0.5,
            "data_1st_order": 0.01,  # TODO
            "prms": np.array(
                [
                    0.2,  # saturation_level
                    0.2,  # offset
                    10.,  # growth_rate
                ]
            ),
        }
        penalty_weights = {k: penalty_weights[k] / order_of_magnitude[k]**2 for k in penalty_weights}
        stock_fitter = StockFitter(
            historic_stocks_pc=self.stocks_to_fit,
            extrapolation=self.extrapolation_single_predictor,
            predictor=self.single_predictor,
            dims_out=self.dims_out,
            penalty_weights=penalty_weights,
        )
        self.fitted_regression = stock_fitter.fit()

    def transform_two_predictor_regression(self):
        """If the regression was performed with two predictors (e.g. time and GDP per capita), we need to transform it back to a regression over only GDP per capita for the stock correction."""
        if self.cfg.regress_over == RegressOverModes.LOGGDPPC_TIME:
            prms = [self.extrapolation.fit_prms[np.newaxis, ..., i] for i in range(self.extrapolation.n_prms)]
            time_dependent_func = self.extrapolation.func(self.extrapolation.predictor_values, prms, factor = "f3")
            self.fitted_regression[...] = self.fitted_regression.values * time_dependent_func

    def smooth_transition(self):
        """The fit function returns a regression which only approximately continues historic trends.
        Here we create a smooth transition between the historic stocks and the fitted regression,
        which is then used as the final extrapolation result.
        """
        stocks_pc_out = np.zeros_like(self.stocks_pc.values)

        match self.transition_smoothing:
            case "none":
                stocks_pc_out[...] = self.fitted_regression.values
            case "critically_damped":
                stocks_pc_out[...] = self.critically_damped_blend(
                    self.historic_stocks_pc.values, self.fitted_regression.values
                )
                add_assumption_doc(
                    type="model assumption",
                    name="Usage of critically damped blend",
                    description=(
                        "Critically damped blending is used to smoothly transition from historic trends to the extrapolation."
                    ),
                )
            case "shift_zeroth_order":
                # match last point by adding the difference between the last historic point and the
                # corresponding prediction
                stocks_pc_out[...] = self.fitted_regression.values - (
                    self.fitted_regression.values[self.n_historic - 1, :]
                    - self.historic_stocks_pc.values[self.n_historic - 1, :]
                )
                add_assumption_doc(
                    type="model assumption",
                    name="Usage of zeroth order correction",
                    description=(
                        "Zeroth order correction is used to match the last historic point with the "
                        "extrapolated stock."
                    ),
                )
            case _:
                raise ValueError(f"Unknown stock_correction method: {self.smooth_transition}")

        stocks_pc_out[: self.n_historic, ...] = self.historic_stocks_pc.values
        self.stocks_pc.set_values(stocks_pc_out)

    def critically_damped_blend(self, historical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        Blend historical and extrapolated stock values using a critically damped system approach to ensure a smooth transition.
        In a critically damped system (e.g. spring that returns to equilibrium as quickly as possible without oscillating),
        if x0 is the initial position and v0 the initial velocity, the position x at a time t is given by:
        x(t) = (x0 + (v0 + k * x0) * t) * exp(-k * t).
        The parameter k determines how quickly the equilibrium is restored, with higher values leading to a faster transition.
        Here, we use the difference between historical and extrapolated stocks at the transition point as the initial offset (x0),
        and the difference in slopes at the transition point as the initial velocity (v0).
        Args:
            historical (np.ndarray): Historical stock data.
            prediction (np.ndarray): Extrapolated stock data from regression.
        Returns:
            np.ndarray: Stock with smooth transition from historical to prediction.
        """
        time = np.array(self.dims["t"].items)
        last_history_idx = len(historical) - 1
        last_history_year = time[last_history_idx]

        # offset between historic and prediction at transition point
        difference_0th = historical[last_history_idx, :] - prediction[last_history_idx, :]

        def avg_slope(x, y, n=1):
            """Assuming time is the first dimension, calculate the average slope over the last n historical timesteps.
            If n is 1, this is the slope between the last two historical timesteps."""
            ydiff = y[last_history_idx, :] - y[last_history_idx - n, :]
            xdiff = x[last_history_idx] - time[last_history_idx - n]
            return ydiff / xdiff

        # offset of the 1st derivative at the transition point
        difference_1st = avg_slope(time, historical) - avg_slope(time, prediction)
        # if the lifetime is given, the number of historical timesteps used for the average slope calculation is determined by the lifetime
        if self.lifetime is not None:
            lower = 3
            upper = 30
            for g in range(self.historic_stocks_pc.dims[2].len):
                Lclip = min(max(self.lifetime.values[g], lower), upper)
                alpha = (np.log(Lclip)-np.log(lower))/np.log(upper)
                avg_slope_hist = alpha * avg_slope(time, historical[:, :, g], n=1) + (1-alpha) * avg_slope(time, historical[:, :, g], n=10)
                avg_slope_pred = alpha * avg_slope(time, prediction[:, :, g], n=1) + (1-alpha) * avg_slope(time, prediction[:, :, g], n=10)
                difference_1st[:, g] = avg_slope_hist - avg_slope_pred

        approaching_time = 80
        add_assumption_doc(
            type="integer number",
            name="years for blending to regression",
            value=approaching_time,
            description=(
                "Number of years for the blending from historical to regressed in-use stocks."
                "After this time, the initial offset between historical and regressed stock is reduced to 5 percent."
            ),
        )

        # Construct time that starts at 0 in last historical year.
        time_extended = time.reshape(-1, *([1] * len(difference_0th.shape)))
        delta_t = time_extended - last_history_year

        # Amplitude decreases to 5 percent after approaching_time years
        k = 4.74 / approaching_time

        # Critically damped system solution
        correction = (difference_0th + (difference_1st + k * difference_0th) * delta_t) * np.exp(
            -k * delta_t
        )

        return prediction[...] + correction

    def _integrate_forced_damped_system(self, y0, v0, t_array, p_array, k):
        """
        Numerically integrates the forced critically damped system:
        Y'' + 2kY' + k^2Y = k^2P(t) + 2kP'(t)
        """
        # Initialize output arrays matching the shape of the prediction array
        y = np.zeros_like(p_array, dtype=float)
        v = np.zeros_like(p_array, dtype=float)
        
        # Initialize running state variables
        y_curr = y0.copy()
        v_curr = v0.copy()
        
        # Set initial conditions
        y[0, ...] = y_curr
        v[0, ...] = v_curr
        
        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i-1]
            
            # --- 1. Semi-Implicit Euler (Step A: Update Position) ---
            y_curr = y_curr + v_curr * dt
            
            # --- 2. Central Difference (Foresight for Target Velocity) ---
            p_curr = p_array[i, ...]
            
            if i < len(t_array) - 1:
                # Look ahead (i+1) and behind (i-1) to calculate the macro-trend slope
                dt_center = t_array[i+1] - t_array[i-1]
                vp_curr = (p_array[i+1, ...] - p_array[i-1, ...]) / dt_center
            else:
                # Fallback to backward difference for the very last timestep
                vp_curr = (p_array[i, ...] - p_array[i-1, ...]) / dt
                
            # --- 3. Calculate Forces at updated position ---
            # Using the rearranged ODE: Y'' = k^2(P - Y) + 2k(P' - Y')
            dv_dt = (k**2) * (p_curr - y_curr) + (2 * k) * (vp_curr - v_curr)
            
            # --- 4. Semi-Implicit Euler (Step B: Update Velocity) ---
            v_curr = v_curr + dv_dt * dt
            
            # Store the state for this timestep
            y[i, ...] = y_curr
            v[i, ...] = v_curr
            
        return y

    def critically_damped_blend(self, historical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        Blend historical and extrapolated stock values using a forced critically damped 
        system approach to ensure a smooth transition.
        
        We solve the ODE: Y'' + 2kY' + k^2Y = k^2P(t)
        where Y is the blended curve and P is the pure prediction.
        """
        time = np.array(self.dims["t"].items)
        last_history_idx = len(historical) - 1

        def avg_slope(x, y, n=1):
            """Assuming time is the first dimension, calculate the average slope over the last n historical timesteps."""
            ydiff = y[last_history_idx, :] - y[last_history_idx - n, :]
            xdiff = x[last_history_idx] - x[last_history_idx - n]
            return ydiff / xdiff

        approaching_time = 50
        add_assumption_doc(
            type="integer number",
            name="years for blending to regression",
            value=approaching_time,
            description=(
                "Number of years for the blending from historical to regressed in-use stocks. "
                "Governs the damping parameter k."
            ),
        )

        k = 4.74 / approaching_time

        # 1. Isolate the time window and prediction values we need to integrate over
        t_future = time[last_history_idx:]
        p_future = prediction[last_history_idx:]

        # 2. Set the absolute initial conditions at the transition point
        y0 = historical[last_history_idx, :]
        v0 = avg_slope(time, historical, n=1)

        # 3. Integrate to find the blended future path Y(t)
        y_future = self._integrate_forced_damped_system(y0, v0, t_future, p_future, k)

        # 4. Construct the final contiguous array
        blended_stock = prediction.copy()
        blended_stock[:last_history_idx] = historical[:last_history_idx] # Preserve exact history
        blended_stock[last_history_idx:] = y_future                      # Apply blended future

        return blended_stock

    # def critically_damped_blend(self, historical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    #     """
    #     Blend historical and extrapolated stock values using a simulated kinematic spring-damper.
        
    #     This guarantees perfect discrete C1 continuity by strictly enforcing a linear kinematic 
    #     projection on the first time step, and only applying corrective acceleration to subsequent 
    #     velocities. It tracks the target position and target velocity to smoothly merge over an 
    #     open-ended horizon.
    #     """
    #     time = np.array(self.dims["t"].items)
    #     last_history_idx = len(historical) - 1
        
    #     # 1. Preserve historical data absolutely perfectly
    #     blended = prediction.copy()
    #     blended[:last_history_idx + 1] = historical[:last_history_idx + 1]

    #     def avg_slope(x, y, n=1):
    #         """Calculate the average slope over the last n historical timesteps."""
    #         ydiff = y[last_history_idx, ...] - y[last_history_idx - n, ...]
    #         xdiff = x[last_history_idx] - x[last_history_idx - n]
    #         return ydiff / xdiff

    #     # 2. Extract initial state to ensure perfect C1 boundary
    #     y_curr = historical[last_history_idx, ...].copy()
    #     v_curr = avg_slope(time, historical, n=1).copy()

    #     approaching_time = 80
    #     # omega is the natural frequency of the spring (stiffness)
    #     omega = 4.74 / approaching_time

    #     # 3. Run the kinematic simulation forward
    #     for t_idx in range(last_history_idx + 1, len(time)):
    #         dt = time[t_idx] - time[t_idx - 1]
            
    #         # STEP A: Update Position
    #         # By applying the current velocity BEFORE calculating any new forces, 
    #         # the first step (t=1) is strictly y_hist + v_hist * dt. 
    #         # This makes the discrete derivative mathematically identical to the historical slope.
    #         y_curr = y_curr + v_curr * dt
            
    #         # STEP B: Observe the Target
    #         target_pos = prediction[t_idx, ...]
            
    #         # Calculate the target's current velocity (foresight for the damper)
    #         if t_idx < len(time) - 1:
    #             # Central difference for smoother target tracking
    #             target_vel = (prediction[t_idx+1, ...] - prediction[t_idx-1, ...]) / (time[t_idx+1] - time[t_idx-1])
    #         else:
    #             # Backward difference at the end of the array
    #             target_vel = (prediction[t_idx, ...] - prediction[t_idx-1, ...]) / dt
                
    #         # STEP C: Calculate Forces
    #         pos_error = target_pos - y_curr
    #         vel_error = target_vel - v_curr
            
    #         # Critically damped acceleration formula: a = w^2 * error + 2w * vel_error
    #         a_curr = (omega**2) * pos_error + (2 * omega) * vel_error
            
    #         # STEP D: Update Velocity for the next step
    #         v_curr = v_curr + a_curr * dt
            
    #         # Store the seamlessly blended result
    #         blended[t_idx, ...] = y_curr

    #     return blended
    
    # def critically_damped_blend(self, historical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    #     time = np.array(self.dims["t"].items)
    #     last_history_idx = len(historical) - 1
    #     last_history_year = time[last_history_idx]

        # # Helper for derivatives
        # def get_diff_derivative(y_hist, y_pred, x, n=5):
        #     """
        #     Calculates the difference in 1st and 2nd derivatives by fitting a 2nd-order 
        #     polynomial to the last n points to filter out noise.
        #     """
        #     last_idx = len(y_hist) - 1
        #     # Slice the last n points
        #     x_clip = x[last_idx - (n-1) : last_idx + 1]
        #     y_h_clip = y_hist[last_idx - (n-1) : last_idx + 1]
        #     y_p_clip = y_pred[last_idx - (n-1) : last_idx + 1]

        #     # Fit a 2nd order polynomial: y = ax^2 + bx + c
        #     # polyfit returns [a, b, c]
        #     h_params = np.polyfit(x_clip, y_h_clip, 2)
        #     p_params = np.polyfit(x_clip, y_p_clip, 2)

        #     # Derivatives of ax^2 + bx + c:
        #     # y'  = 2ax + b
        #     # y'' = 2a
            
        #     t0 = x[last_idx] # The transition time
            
        #     # History derivatives at t0
        #     h_slope = 2 * h_params[0] * t0 + h_params[1]
        #     h_accel = 2 * h_params[0]
            
        #     # Prediction derivatives at t0
        #     p_slope = 2 * p_params[0] * t0 + p_params[1]
        #     p_accel = 2 * p_params[0]
            
        #     return h_slope - p_slope, h_accel - p_accel
        
        # def get_diff_derivatives(y_h, y_p, x_vals, n1=3, n2=10):
        #     """
        #     Calculates differences in 1st and 2nd derivatives using polyfit.
        #     Reshapes data to handle multi-dimensional arrays (regions, stocks, etc.)
        #     """
        #     orig_shape = y_h.shape[1:]
            
        #     def get_vals(y, n):
        #         # Slice last n points and reshape to (n, -1) for vectorized fit
        #         y_slice = y[last_history_idx - (n-1) : last_history_idx + 1].reshape(n, -1)
        #         x_slice = x_vals[last_history_idx - (n-1) : last_history_idx + 1]
                
        #         # Use relative time (t=0 at transition) for numerical stability
        #         t_rel = x_slice - last_history_year
                
        #         # Fit y = at^2 + bt + c. y'(0)=b, y''(0)=2a
        #         coeffs = np.polyfit(t_rel, y_slice, 2)
        #         return coeffs[1].reshape(orig_shape), (2 * coeffs[0]).reshape(orig_shape)

        #     h_slope, _ = get_vals(y_h, n1)
        #     _, h_accel = get_vals(y_h, n2)
        #     p_slope, _ = get_vals(y_p, n1)
        #     _, p_accel = get_vals(y_p, n2)
            
        #     return h_slope - p_slope, h_accel - p_accel

        # # 1. Calculate the initial differences (The Error state at t=0)
        # diff_0 = historical[last_history_idx, :] - prediction[last_history_idx, :]
        # diff_1, diff_2 = get_diff_derivatives(historical, prediction, time, n1=2, n2=20)
        # diff_2 = diff_2 * 0.2

        # approaching_time = 80
        # k = 4.74 / approaching_time

        # # 2. Solve for constants to satisfy x(0), x'(0), and x''(0)
        # A = diff_0
        # B = diff_1 + k * diff_0
        # C = 0.5 * (diff_2 + 2 * k * diff_1 + k**2 * diff_0)

        # # 3. Construct time delta
        # time_extended = time.reshape(-1, *([1] * len(diff_0.shape)))
        # delta_t = time_extended - last_history_year
        
        # # We only apply the correction to the future (delta_t >= 0)
        # # Masking ensures we don't mess with historical data points
        # mask = (delta_t > 0)

        # # 4. Analytical correction: x(t) = (A + Bt + Ct^2) * exp(-kt)
        # correction = (A + B * delta_t + C * delta_t**2) * np.exp(-k * delta_t)

        # return prediction[...] + (correction * mask)

    # def _integrate_forced_damped_system(self, y0, v0, t_array, p_array, k):
    #     """
    #     Numerically integrates the forced critically damped oscillator:
    #     Y'' + 2kY' + k^2Y = k^2P(t)
    #     using the Runge-Kutta 4 (RK4) method for stability and accuracy.
    #     """
    #     # Initialize arrays for position (y) and velocity (v)
    #     y = np.zeros_like(p_array, dtype=float)
    #     v = np.zeros_like(p_array, dtype=float)
        
    #     # Set initial conditions
    #     y[0] = y0
    #     v[0] = v0
        
    #     for i in range(1, len(t_array)):
    #         dt = t_array[i] - t_array[i-1]
            
    #         y_curr = y[i-1]
    #         v_curr = v[i-1]
            
    #         # Approximate the continuous forcing function P(t) at current, next, and mid points
    #         p_curr = p_array[i-1]
    #         p_next = p_array[i]
    #         p_mid = (p_curr + p_next) / 2.0
            
    #         # State derivatives: dy/dt = v,  dv/dt = k^2 * P(t) - 2k*v - k^2*y
    #         def derivatives(y_val, v_val, p_val):
    #             dy = v_val
    #             dv = (k**2) * p_val - (2 * k) * v_val - (k**2) * y_val
    #             return dy, dv
                
    #         # RK4 steps
    #         k1_y, k1_v = derivatives(y_curr, v_curr, p_curr)
    #         k2_y, k2_v = derivatives(y_curr + 0.5 * dt * k1_y, v_curr + 0.5 * dt * k1_v, p_mid)
    #         k3_y, k3_v = derivatives(y_curr + 0.5 * dt * k2_y, v_curr + 0.5 * dt * k2_v, p_mid)
    #         k4_y, k4_v = derivatives(y_curr + dt * k3_y, v_curr + dt * k3_v, p_next)
            
    #         # Update state
    #         y[i] = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    #         v[i] = v_curr + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            
    #     return y
    
    # def _integrate_forced_damped_system(self, y0, v0, t_array, p_array, k):
    #     """
    #     Numerically integrates using the Forward Euler method:
    #     Y'' + 2kY' + k^2Y = k^2P(t) + 
    #     """
    #     y = np.zeros_like(p_array, dtype=float)
    #     v = np.zeros_like(p_array, dtype=float)
        
    #     y[0] = y0
    #     v[0] = v0
        
    #     for i in range(1, len(t_array)):
    #         dt = t_array[i] - t_array[i-1]
            
    #         # Current state
    #         y_curr = y[i-1]
    #         v_curr = v[i-1]
            
    #         p_curr = p_array[i-1]
            
    #         # Current derivatives
    #         dy_dt = v_curr
    #         # dv_dt = (k**2) * p_curr - (2 * k) * v_curr - (k**2) * y_curr

    #         vp_curr = (p_array[i] - p_array[i-1]) / dt 
    #         dv_dt = (k**2) * p_curr - (2 * k) * (v_curr-vp_curr) - (k**2) * y_curr
            
    #         # Step forward linearly
    #         y[i] = y_curr + dy_dt * dt
    #         v[i] = v_curr + dv_dt * dt
            
    #     return y
    
    # def _integrate_forced_damped_system_fixed(self, y0, v0, t_array, p_array, k):
    #     y = np.zeros_like(p_array, dtype=float)
    #     v = np.zeros_like(p_array, dtype=float)
    #     y[0] = y0
    #     v[0] = v0
        
    #     # Pre-calculate the slope of the prediction P'(t)
    #     # We use gradient for a stable estimate of the 'target velocity'
    #     p_slope = np.gradient(p_array, t_array, axis=0)
        
    #     for i in range(1, len(t_array)):
    #         dt = t_array[i] - t_array[i-1]
            
    #         y_curr, v_curr = y[i-1], v[i-1]
    #         p_curr, ps_curr = p_array[i-1], p_slope[i-1]
    #         p_next, ps_next = p_array[i], p_slope[i]
            
    #         # Midpoint estimates for RK4
    #         p_mid = (p_curr + p_next) / 2.0
    #         ps_mid = (ps_curr + ps_next) / 2.0
            
    #         def derivatives(y_val, v_val, p_val, ps_val):
    #             dy = v_val
    #             # The Magic: k^2*P + 2*k*P' cancels the damping lag!
    #             dv = (k**2 * p_val + 2 * k * ps_val) - (2 * k * v_val) - (k**2 * y_val)
    #             return dy, dv
                
    #         k1_y, k1_v = derivatives(y_curr, v_curr, p_curr, ps_curr)
    #         k2_y, k2_v = derivatives(y_curr + 0.5*dt*k1_y, v_curr + 0.5*dt*k1_v, p_mid, ps_mid)
    #         k3_y, k3_v = derivatives(y_curr + 0.5*dt*k2_y, v_curr + 0.5*dt*k2_v, p_mid, ps_mid)
    #         k4_y, k4_v = derivatives(y_curr + dt*k3_y, v_curr + dt*k3_v, p_next, ps_next)
            
    #         y[i] = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    #         v[i] = v_curr + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            
    #     return y

    # def _integrate_forced_damped_system(self, y0, v0, t_array, p_array, k):
    #     """
    #     Numerically integrates the forced critically damped system:
    #     Y'' + 2kY' + k^2Y = k^2P(t) + 2kP'(t)
    #     """
    #     # Initialize output arrays matching the shape of the prediction array
    #     y = np.zeros_like(p_array, dtype=float)
    #     v = np.zeros_like(p_array, dtype=float)
        
    #     # Initialize running state variables
    #     y_curr = y0.copy()
    #     v_curr = v0.copy()
        
    #     # Set initial conditions
    #     y[0, ...] = y_curr
    #     v[0, ...] = v_curr
        
    #     for i in range(1, len(t_array)):
    #         dt = t_array[i] - t_array[i-1]
            
    #         # --- 1. Semi-Implicit Euler (Step A: Update Position) ---
    #         y_curr = y_curr + v_curr * dt
            
    #         # --- 2. Central Difference (Foresight for Target Velocity) ---
    #         p_curr = p_array[i, ...]
            
    #         if i < len(t_array) - 1:
    #             # Look ahead (i+1) and behind (i-1) to calculate the macro-trend slope
    #             dt_center = t_array[i+1] - t_array[i-1]
    #             vp_curr = (p_array[i+1, ...] - p_array[i-1, ...]) / dt_center
    #         else:
    #             # Fallback to backward difference for the very last timestep
    #             vp_curr = (p_array[i, ...] - p_array[i-1, ...]) / dt
                
    #         # --- 3. Calculate Forces at updated position ---
    #         # Using the rearranged ODE: Y'' = k^2(P - Y) + 2k(P' - Y')
    #         dv_dt = (k**2) * (p_curr - y_curr) + (2 * k) * (vp_curr - v_curr)
            
    #         # --- 4. Semi-Implicit Euler (Step B: Update Velocity) ---
    #         v_curr = v_curr + dv_dt * dt
            
    #         # Store the state for this timestep
    #         y[i, ...] = y_curr
    #         v[i, ...] = v_curr
            
    #     return y
    
    # def _integrate_forced_damped_system(self, y0, v0, a0, t_array, p_array, k):
    #     """
    #     Numerically integrates the 3rd-order forced critically damped system:
    #     Y''' + 3kY'' + 3k^2Y' + k^3Y = k^3P + 3k^2P' + 3kP''
        
    #     This guarantees C2 continuity by tracking position, velocity, AND acceleration.
    #     """
    #     # Initialize output arrays
    #     y = np.zeros_like(p_array, dtype=float)
    #     v = np.zeros_like(p_array, dtype=float)
    #     a = np.zeros_like(p_array, dtype=float)
        
    #     # State variables
    #     y_curr = y0.copy()
    #     v_curr = v0.copy()
    #     a_curr = a0.copy()
        
    #     y[0, ...] = y_curr
    #     v[0, ...] = v_curr
    #     a[0, ...] = a_curr
        
    #     for i in range(1, len(t_array)):
    #         dt = t_array[i] - t_array[i-1]
            
    #         # --- 1. Semi-Implicit Euler Cascade (Step A & B) ---
    #         # Update position using old velocity, update velocity using old acceleration.
    #         # This locks in the C1 and C2 boundary conditions flawlessly at t=1.
    #         y_curr = y_curr + v_curr * dt
    #         v_curr = v_curr + a_curr * dt
            
    #         # --- 2. Central Difference Foresight (1st & 2nd Target Derivatives) ---
    #         p_curr = p_array[i, ...]
            
    #         if i < len(t_array) - 1:
    #             # 1st Derivative (Velocity Foresight)
    #             dt_center = t_array[i+1] - t_array[i-1]
    #             vp_curr = (p_array[i+1, ...] - p_array[i-1, ...]) / dt_center
                
    #             # 2nd Derivative (Acceleration Foresight)
    #             # Standard central difference for 2nd derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
    #             dt_avg = dt_center / 2.0
    #             ap_curr = (p_array[i+1, ...] - 2 * p_curr + p_array[i-1, ...]) / (dt_avg**2)
    #         else:
    #             # Fallbacks for the very last timestep
    #             vp_curr = (p_curr - p_array[i-1, ...]) / dt
    #             ap_curr = np.zeros_like(p_curr) # Assume no sudden acceleration at the exact end
                
    #         # --- 3. Calculate Forces (Jerk / 3rd Derivative) ---
    #         # Y''' = k^3(P - Y) + 3k^2(P' - Y') + 3k(P'' - Y'')
    #         da_dt = (k**3) * (p_curr - y_curr) + \
    #                 (3 * k**2) * (vp_curr - v_curr) + \
    #                 (3 * k) * (ap_curr - a_curr)
            
    #         # --- 4. Semi-Implicit Euler (Step C) ---
    #         a_curr = a_curr + da_dt * dt
            
    #         # Store the state
    #         y[i, ...] = y_curr
    #         v[i, ...] = v_curr
    #         a[i, ...] = a_curr
            
    #     return y


    # def critically_damped_blend(self, historical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    #     """
    #     Blend historical and extrapolated stock values using a forced critically damped 
    #     system approach to ensure a smooth transition.
        
    #     We solve the ODE: Y'' + 2kY' + k^2Y = k^2P(t)
    #     where Y is the blended curve and P is the pure prediction.
    #     """
    #     time = np.array(self.dims["t"].items)
    #     last_history_idx = len(historical) - 1

    #     def avg_slope(x, y, n=1):
    #         """Assuming time is the first dimension, calculate the average slope over the last n historical timesteps."""
    #         ydiff = y[last_history_idx, :] - y[last_history_idx - n, :]
    #         xdiff = x[last_history_idx] - x[last_history_idx - n]
    #         return ydiff / xdiff

    #     approaching_time = 50
    #     add_assumption_doc(
    #         type="integer number",
    #         name="years for blending to regression",
    #         value=approaching_time,
    #         description=(
    #             "Number of years for the blending from historical to regressed in-use stocks. "
    #             "Governs the damping parameter k."
    #         ),
    #     )

    #     k = 4.74 / approaching_time

    #     # 1. Isolate the time window and prediction values we need to integrate over
    #     t_future = time[last_history_idx:]
    #     p_future = prediction[last_history_idx:]

    #     # 2. Set the absolute initial conditions at the transition point
    #     y0 = historical[last_history_idx, :]
    #     v0 = avg_slope(time, historical, n=1)

    #     # 3. Integrate to find the blended future path Y(t)
    #     y_future = self._integrate_forced_damped_system(y0, v0, t_future, p_future, k)

    #     # 4. Construct the final contiguous array
    #     blended_stock = prediction.copy()
    #     blended_stock[:last_history_idx] = historical[:last_history_idx] # Preserve exact history
    #     blended_stock[last_history_idx:] = y_future                      # Apply blended future

    #     return blended_stock

    @property
    def dims_out(self):
        return self.dims[self.target_dim_letters]

    @property
    def n_historic(self):
        return self.dims["h"].len
