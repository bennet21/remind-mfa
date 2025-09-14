import flodym as fd
import numpy as np
from typing import Tuple, Union, Type
from scipy.interpolate import CubicHermiteSpline, BPoly

from remind_mfa.common.data_extrapolations import Extrapolation
from remind_mfa.common.data_transformations import broadcast_trailing_dimensions, BoundList
from remind_mfa.common.assumptions_doc import add_assumption_doc


class StockExtrapolation:
    """
    Class for extrapolating stocks based on historical data and GDP per capita.
    """

    def __init__(
        self,
        historic_stocks: fd.StockArray,
        dims: fd.DimensionSet,
        parameters: dict[str, fd.Parameter],
        stock_extrapolation_class: Type[Extrapolation],
        target_dim_letters: Union[Tuple[str, ...], str] = "all",
        indep_fit_dim_letters: Union[Tuple[str, ...], str] = (),
        bound_list: BoundList = BoundList(),
        do_gdppc_accumulation: bool = True,
        stock_correction: str = "gaussian_first_order",
        n_deriv: int = 5,
    ):
        """
        Initialize the StockExtrapolation class.

        Args:
            historic_stocks (fd.StockArray): Historical stock data.
            dims (fd.DimensionSet): Dimension set for the data.
            parameters (dict[str, fd.Parameter]): Parameters for the extrapolation.
            stock_extrapolation_class (Extrapolation): Class used for stock extrapolation.
            target_dim_letters (Union[Tuple[str, ...], str]): Sets the dimensions of the stock extrapolation output. If "all", the output will have the same shape as historic_stocks, except for the time dimension. Defaults to "all".
            indep_fit_dim_letters (Union[Tuple[str, ...]], str): Sets the dimensions across which an individual fit is performed, must be subset of target_dim_letters. If "all", all dimensions given in target_dim_letters are regressed individually. If empty (), all dimensions are regressed aggregately. Defaults to ().
            bound_list (BoundList): List of bounds for the extrapolation. Defaults to an empty BoundList.
            do_gdppc_accumulation (bool): Flag to perform GDP per capita accumulation. Defaults to True.
            stock_correction (str): Method for stock correction. Possible values are "gaussian_first_order", "shift_zeroth_order", "none". Defaults to "gaussian_first_order".
        """
        self.historic_stocks = historic_stocks
        self.dims = dims
        self.parameters = parameters
        self.stock_extrapolation_class = stock_extrapolation_class
        self.target_dim_letters = target_dim_letters
        self.n_deriv = n_deriv
        self.set_dims(indep_fit_dim_letters)
        self.bound_list = bound_list
        self.do_gdppc_accumulation = do_gdppc_accumulation
        self.stock_correction = stock_correction
        self.extrapolate()

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

    def extrapolate(self):
        """Preprocessing and extrapolation."""
        self.per_capita_transformation()
        self.gdp_regression()

    def per_capita_transformation(self):
        self.pop = self.parameters["population"]
        self.gdppc = self.parameters["gdppc"]
        if self.do_gdppc_accumulation:
            self.gdppc_acc = np.maximum.accumulate(self.gdppc.values, axis=0)
        self.historic_pop = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_gdppc = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_stocks_pc = fd.StockArray(dims=self.dims[self.historic_dim_letters])
        self.stocks_pc = fd.StockArray(dims=self.dims[self.target_dim_letters])
        self.stocks = fd.StockArray(dims=self.dims[self.target_dim_letters])

        self.historic_pop[...] = self.pop[{"t": self.dims["h"]}]
        self.historic_gdppc[...] = self.gdppc[{"t": self.dims["h"]}]
        self.historic_stocks_pc[...] = self.historic_stocks / self.historic_pop

    def gdp_regression(self):
        """Updates per capita stock to future by extrapolation."""

        prediction_out = self.stocks_pc.values.copy()
        historic_in = self.historic_stocks_pc.values
        if self.do_gdppc_accumulation:
            gdppc = self.gdppc_acc
            add_assumption_doc(
                type="model assumption",
                name="Usage of cumulative GDP per capita",
                description=(
                    "Accumulated GDPpc is used for stock extrapolation to prevent "
                    "stock shrink in times of decreasing GDPpc. "
                ),
            )
        else:
            gdppc = self.gdppc
        gdppc = broadcast_trailing_dimensions(gdppc, prediction_out)
        n_historic = historic_in.shape[0]

        add_assumption_doc(
            type="integer number",
            name="n years for regression derivative correction",
            value=self.n_deriv,
            description=(
                "Number of historic years used for determination of regressed and actual "
                "growth rates of ins-use stocks, which are then used for a correction "
                "reconciling the two and blending from observed to regression."
            ),
        )

        self.extrapolation = self.stock_extrapolation_class(
            data_to_extrapolate=historic_in,
            predictor_values=gdppc,
            independent_dims=self.fit_dim_idx,
            bound_list=self.bound_list,
        )
        pure_prediction = self.extrapolation.regress()

        if self.stock_correction == "gaussian_first_order":
            prediction_out[...] = self.gaussian_correction(historic_in, pure_prediction, self.n_deriv)
            add_assumption_doc(
                type="model assumption",
                name="Usage of Gaussian correction",
                description=(
                    "Gaussian correction is used to blend historic trends with the extrapolation."
                ),
            )
        elif self.stock_correction == "shift_zeroth_order":
            # match last point by adding the difference between the last historic point and the corresponding prediction
            prediction_out[...] = pure_prediction - (
                pure_prediction[n_historic - 1, :] - historic_in[n_historic - 1, :]
            )
            add_assumption_doc(
                type="model assumption",
                name="Usage of zeroth order correction",
                description=(
                    "Zeroth order correction is used to match the last historic point with the "
                    "extrapolated stock."
                ),
            )
        elif self.stock_correction == "cubic_spline":
            prediction_out[...] = self.cubic_spline_correction(historic_in, pure_prediction, self.n_deriv)
            add_assumption_doc(
                type="model assumption",
                name="Usage of Cubic Hermite spline correction",
                description=(
                    "Cubic Hermite Spline correction is used to smoothly blend historic trends "
                    "with the extrapolation."
                ),
            )
        elif self.stock_correction == "quintic_spline":
            prediction_out[...] = self.quintic_spline_correction(historic_in, pure_prediction, self.n_deriv)
            add_assumption_doc(
                type="model assumption",
                name="Usage of Quintic Hermite spline correction",
                description=(
                    "Quintic Hermite Spline correction is used to smoothly blend historic trends "
                    "with the extrapolation."
                ),
            )

        # save extrapolation data for later analysis
        self.pure_prediction = fd.FlodymArray(dims=self.stocks_pc.dims, values=pure_prediction)

        prediction_out[:n_historic, ...] = historic_in
        self.stocks_pc.set_values(prediction_out)

        # transform back to total stocks
        self.stocks[...] = self.stocks_pc * self.pop

    def gaussian_correction(
        self, historic: np.ndarray, prediction: np.ndarray, n: int = 5
    ) -> np.ndarray:
        """
        Gaussian smoothing of extrapolation between the historic and future interface to remove discontinuities
        of 0th and 1st order derivatives. Multiplies Gaussian with a Taylor expansion around
        the difference beteween historic and fit.
        Args:
            historic (np.ndarray): Historical stock data.
            prediction (np.ndarray): Predicted stock data from regression.
            n (int): Number of years for the linear regression fit. Defaults to 5.
        Returns:
            np.ndarray: Corrected stock data after applying Gaussian smoothing.
        """
        time = np.array(self.dims["t"].items)
        last_history_idx = len(historic) - 1
        last_history_year = time[last_history_idx]
        # offset between historic and prediction at transition point
        difference_0th = historic[last_history_idx, :] - prediction[last_history_idx, :]

        def lin_fit(x, y, last_idx, n=n):
            """Linear fit of the last n points."""
            x_cut = np.vstack([x[last_idx - n : last_idx], np.ones(n)]).T
            y_cut = y[last_idx - n : last_idx, :]
            y_reshaped = y_cut.reshape(n, -1).T
            slopes = [np.linalg.lstsq(x_cut, y_dim, rcond=None)[0][0] for y_dim in y_reshaped]
            slopes_reshaped = np.array(slopes).reshape(y.shape[1:])
            return slopes_reshaped

        last_historic_1st = lin_fit(time, historic, last_history_idx)
        last_prediction_1st = lin_fit(time, prediction, last_history_idx)

        # offset of the 1st derivative at the transition point
        difference_1st = (last_historic_1st - last_prediction_1st) / (
            last_history_year - time[last_history_idx - 1]
        )

        def gaussian(t, approaching_time):
            """After the approaching time, the amplitude of the gaussian has decreased to 5%."""
            a = np.sqrt(np.log(20))
            return np.exp(-((a * t / approaching_time) ** 2))

        approaching_time_0th = 50
        add_assumption_doc(
            type="integer number",
            name="years for absolute blending to regression",
            value=approaching_time_0th,
            description=(
                "Number of years for the blending from historical to regressed in-use stocks. "
            ),
        )
        approaching_time_1st = 30
        add_assumption_doc(
            type="integer number",
            name="years for derivative blending to regression",
            value=approaching_time_1st,
            description=(
                "Number of years for the blending from historical to regressed in-use stock "
                "growth rates. "
            ),
        )
        time_extended = time.reshape(-1, *([1] * len(difference_0th.shape)))
        corr0 = difference_0th * gaussian(time_extended - last_history_year, approaching_time_0th)
        corr1 = (
            difference_1st
            * (time_extended - last_history_year)
            * gaussian(time_extended - last_history_year, approaching_time_1st)
        )
        correction = corr0 + corr1

        return prediction[...] + correction

    def cubic_spline_correction(self, historic: np.ndarray, prediction: np.ndarray, n: int = 5) -> np.ndarray:
        """
        Smoothly corrects the beginning of a prediction series to connect it to a historic series.

        This function uses a Cubic Hermite Spline to create a transition. The start of the
        spline is determined by the last point and slope of the `historic` data. The end
        of the spline is chosen algorithmically from the `prediction` data by finding the
        endpoint `c` that results in the smoothest possible curve, minimizing the integral
        of the squared second derivative.

        Args:
            historic: A 1D NumPy array of historical values (y1).
            prediction: A 1D NumPy array of predicted values (y2) that follows the historic data.
                        This is the array that will be modified.

        Returns:
            A new NumPy array containing the corrected prediction, with a smooth
            transition from the end of the history.
        """
        time = np.array(self.dims["t"].items)

        # Starting point
        last_history_idx = len(historic) - 1
        x_start = time[last_history_idx]
        ys_start = historic[-1]

        # Starting point derivative
        fit_indices = np.arange(last_history_idx - n + 1, last_history_idx + 1)
        time_fit = time[fit_indices]
        historic_fit = historic[fit_indices]
        coeffs = np.polyfit(time_fit, historic_fit, 1)
        ys_prime_start = coeffs[0]

        # Construct the corrected prediction
        corrected_prediction = prediction.copy()

        # Before transition, keep historic values - same starting point across all dimensions
        corrected_prediction[:last_history_idx + 1] = historic

        # End point
        end_idxs = self.find_optimal_cubic_spline_endpoint(
            time=time,
            prediction=prediction,
            x_start=x_start,
            y_start=ys_start,
            y_prime_start=ys_prime_start,
            start_idx=last_history_idx,
            n=n
        )

        # create spline for each independent dimension (except time)
        # TODO make this agnostic of position of time dimension
        for ndi in np.ndindex(prediction.shape[1:]):
            end_idx = end_idxs[ndi]

            # x_start stays the same
            y_start = ys_start[ndi]
            y_prime_start = ys_prime_start[ndi]

            x_end = time[end_idx]
            y_end = prediction[end_idx, ndi]

            # End point derivative
            fit_indices = np.arange(end_idx - n + 1, end_idx + 1)
            time_fit = time[fit_indices]
            prediction_fit = prediction[fit_indices, ndi]
            coeffs = np.polyfit(time_fit, prediction_fit, 1)
            y_prime_end = coeffs[0]

            # Create the Hermite spline
            final_spline = CubicHermiteSpline(
                x=[x_start, x_end],
                y=np.vstack([[y_start], [y_end]]),
                dydx=np.vstack([[y_prime_start], [y_prime_end]])
            )

            # In the transition period, apply the spline
            transition_indices = np.arange(last_history_idx, end_idx + 1)
            transition_times = time[transition_indices]
            corrected_prediction[transition_indices, ndi] = final_spline(transition_times).squeeze()

        return corrected_prediction
    
    @staticmethod
    def _calculate_cubic_spline_roughness(p0, m0, p1, m1, h):
        """
        Calculates the normalized integral of the squared second derivative for a cubic Hermite spline.

        This value serves as a measure of the spline's "roughness" or "bending energy".
        A lower value means a smoother curve. The formula is derived from the coefficients
        of the cubic polynomial that defines the spline.

        Args:
            p0 (float): Start point value (y_start).
            m0 (float): Start point derivative (y_prime_start).
            p1 (float): End point value (y_end).
            m1 (float): End point derivative (y_prime_end).
            h (float): The length of the interval (x_end - x_start).

        Returns:
            float: The normalized roughness cost of the spline.
        """
        # This is the analytical solution for the integral of (P''(x))^2 dx from x0 to x1.
        term1 = 4 * (m1**2 + m0 * m1 + m0**2) / h
        term2 = 12 * (p1 - p0) * (m0 + m1) / (h**2)
        term3 = 12 * (p1 - p0)**2 / (h**3)

        integral = term1 - term2 + term3
        
        return integral / h
    
    def find_optimal_cubic_spline_endpoint(
            self,
            time: np.ndarray,
            prediction: np.ndarray,
            x_start: float,
            y_start: float,
            y_prime_start: float,
            start_idx: int,
            n: int = 5,
            search_range_years: tuple = (30, 150)
        ) -> int:
        """
        Finds the optimal endpoint for the Hermite spline by minimizing curvature.

        It iterates through a specified range of potential endpoints in the prediction
        data, creates a hypothetical spline for each, and calculates its roughness.
        The endpoint that yields the smoothest spline (minimum roughness) is chosen.

        Args:
            time: The array of time values.
            prediction: The array of predicted values.
            x_start: The starting time for the spline.
            y_start: The starting value for the spline.
            y_prime_start: The starting derivative for the spline.
            start_idx: The index in the time/prediction array corresponding to x_start.
            n: The number of points to use for linear regression to find derivatives.
            search_range_years: A tuple defining the min and max years from the start
                                to search for an optimal endpoint.

        Returns:
            The index of the optimal endpoint in the prediction array.
        """
        min_roughness = np.full_like(y_start, np.inf, dtype=float)
        optimal_end_idx = np.full_like(y_start, -1, dtype=int)

        # Define the search space in terms of array indices
        min_end_time = x_start + search_range_years[0]
        max_end_time = x_start + search_range_years[1]
        
        start_search_idx = np.searchsorted(time, min_end_time, side='left')
        end_search_idx = np.searchsorted(time, max_end_time, side='right')
        
        # Ensure the search range is valid and within array bounds
        start_search_idx = max(start_idx + n + 1, start_search_idx)
        end_search_idx = min(len(prediction) - (n + 1), end_search_idx)

        if np.any(start_search_idx >= end_search_idx):
            raise ValueError("Search range for optimal endpoint is invalid or empty.")

        for end_idx in range(start_search_idx, end_search_idx):
            # Define potential endpoint
            x_end = time[end_idx]
            y_end = prediction[end_idx]
            
            # Calculate endpoint derivative
            fit_indices = np.arange(end_idx - n + 1, end_idx + 1)
            time_fit = time[fit_indices]
            prediction_fit = prediction[fit_indices]
            coeffs_end = np.polyfit(time_fit, prediction_fit, 1)
            y_prime_end = coeffs_end[0]

            # Calculate the roughness of this potential spline
            interval_h = x_end - x_start
            roughness = self._calculate_cubic_spline_roughness(y_start, y_prime_start, y_end, y_prime_end, interval_h)

            # update minimum roughness and optimal index where applicable
            update_mask = roughness < min_roughness
            min_roughness[update_mask] = roughness[update_mask]
            optimal_end_idx[update_mask] = end_idx
        
        if np.any(optimal_end_idx == -1):
             raise RuntimeError("Could not find an optimal endpoint. Check search range and data.")

        return optimal_end_idx
    
    @staticmethod
    def derivatives_from_polyfit(x, y, idx, n):
            fit_indices = np.arange(idx - n + 1, idx + 1)
            time_fit = x[fit_indices]
            prediction_fit = y[fit_indices]
            coeffs_end = np.polyfit(time_fit, prediction_fit, 2, w=np.linspace(1, 2, len(time_fit)))
            y_prime_end = 2 * coeffs_end[0] * x[idx] + coeffs_end[1]
            y_prime2_end = 2 * coeffs_end[0]
            return y_prime_end, y_prime2_end

    def quintic_spline_correction(
            self,
            historic: np.ndarray,
            prediction: np.ndarray,
            n_start: int = 5,
            n_end: int = 5,
        ) -> np.ndarray:
        """
        Smoothly corrects the beginning of a prediction series to connect it to a historic series
        using a quintic Hermite spline (BPoly).

        This function uses `scipy.interpolate.BPoly.from_derivatives` to create a transition
        over a fixed duration. The spline is defined by the position, first derivative, and
        second derivative at its start and end points.

        Args:
            historic: A 1D NumPy array of historical values.
            prediction: A 1D NumPy array of predicted values that will be modified.
            n: The number of points to use for polynomial fitting to find derivatives.
            transition_years: The fixed duration of the spline transition in time units.

        Returns:
            A new NumPy array containing the corrected prediction, with a smooth
            transition from the end of the history.
        """
        time = np.array(self.dims["t"].items)

        # Starting point
        last_history_idx = len(historic) - 1
        x_start = time[last_history_idx]
        ys_start = historic[-1]

        # Starting point derivatives
        ys_prime_start, ys_prime2_start = self.derivatives_from_polyfit(time, historic, last_history_idx, n_start)

        # Construct the corrected prediction
        corrected_prediction = prediction.copy()

        # Before transition, keep historic values
        corrected_prediction[:last_history_idx + 1] = historic

        end_idxs = self.find_optimal_quintic_spline_endpoint(
            time=time,
            prediction=prediction,
            x_start=x_start,
            y_start=ys_start,
            y_prime_start=ys_prime_start,
            y_prime2_start=ys_prime2_start,
            n_end=n_end,
        )

        print(time[end_idxs] - x_start)

        # create spline for each independent dimension (except time)
        # TODO make this agnostic of position of time dimension
        for ndi in np.ndindex(prediction.shape[1:]):
            end_idx = end_idxs[ndi]

            # x_start stays the same
            y_start = ys_start[ndi]
            y_prime_start = ys_prime_start[ndi]
            y_prime2_start = ys_prime2_start[ndi]

            x_end = time[end_idx]
            y_end = prediction[end_idx, ndi]

            # End point derivatives
            y_prime_end, y_prime2_end = self.derivatives_from_polyfit(time, prediction[:, ndi], end_idx, n_end)

            # Create the Hermite spline
            final_spline = BPoly.from_derivatives(
                xi=[x_start, x_end],
                yi=[
                    np.vstack([y_start, y_prime_start, y_prime2_start]),
                    np.vstack([y_end, y_prime_end, y_prime2_end])
                ]
            )

            # In the transition period, apply the spline
            transition_indices = np.arange(last_history_idx, end_idx)
            transition_times = time[transition_indices]
            corrected_prediction[transition_indices, ndi] = final_spline(transition_times).squeeze()

        return corrected_prediction
    
    @staticmethod
    def _calculate_quintic_spline_jerk(p0, v0, a0, p1, v1, a1, h):
        """
        Calculates the normalized integral of the squared third derivative (jerk) for a quintic Hermite spline.

        This value is a standard measure of smoothness for motion profiles, as minimizing
        jerk reduces vibrations and mechanical stress.

        Args:
            p0 (float): Start point position.
            v0 (float): Start point velocity (1st derivative).
            a0 (float): Start point acceleration (2nd derivative).
            p1 (float): End point position.
            v1 (float): End point velocity (1st derivative).
            a1 (float): End point acceleration (2nd derivative).
            h (float): The length of the interval.

        Returns:
            float: The normalized integrated squared jerk cost of the quintic spline.
        """
        # This is the analytical solution for the integral of (P'''(x))^2 dx from 0 to h.
        h3 = h**3
        h4 = h**4
        h5 = h**5

        term1 = 36 * (v1**2 + v0**2) / h3
        term2 = 72 * v0 * v1 / h3
        term3 = (a1**2 + a0**2) * h / 5
        term4 = (a0 * a1 * h) / 10
        term5 = 3 * (a0*v1 - a1*v0) / h
        term6 = 120 * (p0 - p1)*(a0 - a1) / h3
        term7 = 360 * (p0 - p1)*(v0 + v1) / h4
        term8 = 720 * (p1 - p0)**2 / h5

        integral = term1 - term2 + term3 + term4 - term5 - term6 + term7 + term8
        
        return integral / h
    
    # @staticmethod
    # def _calculate_quintic_spline_roughness(p0, v0, a0, p1, v1, a1, h):
    #     """
    #     Calculates the integral of the squared second derivative (acceleration) for a 
    #     quintic Hermite spline. This is a measure of the spline's "bending energy".

    #     Args:
    #         p0 (float): Start point position.
    #         v0 (float): Start point velocity (1st derivative).
    #         a0 (float): Start point acceleration (2nd derivative).
    #         p1 (float): End point position.
    #         v1 (float): End point velocity (1st derivative).
    #         a1 (float): End point acceleration (2nd derivative).
    #         h (float): The length of the interval.

    #     Returns:
    #         float: The total integrated squared acceleration cost of the spline.
    #     """

    #     h2 = h**2
    #     h3 = h**3

    #     term1 = a0**2 * h + a1**2 * h + a0 * a1 * h
    #     term2 = 12 * (a0 + a1) * (p0 - p1) / h
    #     term3 = 6 * (a0 * v0 - a1 * v1)
    #     term4 = 6 * (a1 * v0 - a0 * v1)
    #     term5 = 12 * (v0**2 + v1**2) / h
    #     term6 = 24 * v0 * v1 / h
    #     term7 = 72 * (v0 + v1) * (p1 - p0) / h2
    #     term8 = 120 * (p1 - p0)**2 / h3

    #     integral = term1 + term2 - term3 + term4 + term5 - term6 + term7 + term8
        
    #     return integral / h
    
    def find_optimal_quintic_spline_endpoint(
            self,
            time: np.ndarray,
            prediction: np.ndarray,
            x_start: float,
            y_start: np.ndarray,
            y_prime_start: np.ndarray,
            y_prime2_start: np.ndarray,
            n_end: int = 5,
            search_range_years: tuple = (50, 100)
        ) -> np.ndarray:
        """
        Finds the optimal endpoint for a quintic Hermite spline by minimizing jerk.

        It iterates through a specified range of potential endpoints in the prediction
        data, creates a hypothetical quintic spline for each, and calculates its
        integrated squared third derivative (jerk). The endpoint that yields the
        smoothest spline (minimum jerk) is chosen.

        Args:
            time: The array of time values.
            prediction: The array of predicted values.
            x_start: The starting time for the spline.
            y_start: The starting value for the spline.
            y_prime_start: The starting first derivative for the spline.
            y_prime2_start: The starting second derivative for the spline.
            start_idx: The index in the time/prediction array corresponding to x_start.
            n_end: The number of points to use for polynomial regression to find derivatives.
            search_range_years: A tuple defining the min and max years from the start
                                to search for an optimal endpoint.

        Returns:
            The array of optimal endpoint indices in the prediction array.
        """
        min_jerk = np.full_like(y_start, np.inf, dtype=float)
        optimal_end_idx = np.full_like(y_start, -1, dtype=int)

        # Define the search space in terms of array indices
        min_end_time = x_start + search_range_years[0]
        max_end_time = x_start + search_range_years[1]
        
        start_search_idx = np.searchsorted(time, min_end_time, side='left')
        end_search_idx = np.searchsorted(time, max_end_time, side='right')

        if np.any(start_search_idx >= end_search_idx):
            raise ValueError("Search range for optimal endpoint is invalid.")

        # TODO vectorize this loop
        for end_idx in range(start_search_idx, end_search_idx):
            # Define potential endpoint
            x_end = time[end_idx]
            y_end = prediction[end_idx]
            
            y_prime_end, y_prime2_end = self.derivatives_from_polyfit(time, prediction, end_idx, n_end)

            # Calculate the jerk of this potential spline
            interval_h = x_end - x_start
            jerk = self._calculate_quintic_spline_jerk(y_start, y_prime_start, y_prime2_start, y_end, y_prime_end, y_prime2_end, interval_h)
            # update minimum jerk and optimal index where applicable
            update_mask = jerk < min_jerk
            min_jerk[update_mask] = jerk[update_mask]
            optimal_end_idx[update_mask] = end_idx
        
        if np.any(optimal_end_idx == -1):
             raise RuntimeError("Could not find an optimal endpoint. Check search range and data.")

        return optimal_end_idx