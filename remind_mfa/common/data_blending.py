import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Union

import flodym as fd
import numpy as np

from remind_mfa.common.assumptions_doc import add_assumption_doc


def blend(
    target_dims: fd.DimensionSet,
    y_lower: fd.FlodymArray,
    y_upper: fd.FlodymArray,
    x: Union[fd.FlodymArray, str],
    x_lower: Union[fd.FlodymArray, int, float],
    x_upper: Union[fd.FlodymArray, int, float],
    type: str = "poly_mix",
) -> fd.FlodymArray:
    """
    Blend between two arrays (y_lower, y_upper) along a dimension or variable x, using a specified blending function.

    This function interpolates (or blends) between y_lower and y_upper based on the normalized position of x between x_lower and x_upper,
    using a chosen blending curve (e.g., linear, sigmoid, hermite, quintic, etc.).

    Args:
        target_dims (fd.DimensionSet):
            The target dimensions for the output array. All input arrays and scalars are broadcast/cast to these dimensions.
        y_lower (fd.FlodymArray):
            The value (array) to use when x == x_lower (i.e., at the lower bound).
        y_upper (fd.FlodymArray):
            The value (array) to use when x == x_upper (i.e., at the upper bound).
        x (Union[fd.FlodymArray, str]):
            The variable to blend along. Can be a FlodymArray (values for each point) or a string (dimension name/letter) to use the corresponding dimension values from target_dims.
        x_lower (Union[fd.FlodymArray, int, float]):
            The lower bound for x (can be scalar or array). Where x == x_lower, the result is y_lower.
        x_upper (Union[fd.FlodymArray, int, float]):
            The upper bound for x (can be scalar or array). Where x == x_upper, the result is y_upper.
        type (str, optional):
            The blending function to use. Options include: 'linear', 'sigmoid3', 'sigmoid4', 'hermite', 'quintic', 'poly_mix', etc.
            Default is 'poly_mix'.

    Returns:
        fd.FlodymArray: The blended/interpolated array, with dimensions target_dims.

    Example:
        blend(target_dims, y_lower, y_upper, x="t", x_lower=2020, x_upper=2050, type="hermite")
        # Blends y_lower to y_upper as the 't' dimension goes from 2020 to 2050 using a Hermite curve.

    """
    if isinstance(x, str):
        x = fd.FlodymArray(dims=target_dims[(x,)], values=np.array(target_dims[x].items))
    x = x.cast_to(target_dims)
    y_lower = prepare_array(y_lower, target_dims)
    y_upper = prepare_array(y_upper, target_dims)
    x_lower = prepare_array(x_lower, target_dims)
    x_upper = prepare_array(x_upper, target_dims)

    x = (x - x_lower) / (x_upper - x_lower)
    a = fd.FlodymArray(dims=x.dims, values=blending_factor(x.values, type))
    return a * y_upper + (1 - a) * y_lower


def _linear(x):
    x = np.clip(x, 0, 1)
    return x


def _sigmoid3(x):
    return 1.0 / (1.0 + np.exp(3 - 6 * x))


def _sigmoid4(x):
    return 1.0 / (1.0 + np.exp(4 - 8 * x))


def _extrapol_sigmoid3(x):
    return (_sigmoid3(x) - _sigmoid3(0)) / (_sigmoid3(1) - _sigmoid3(0))


def _extrapol_sigmoid4(x):
    return (_sigmoid4(x) - _sigmoid4(0)) / (_sigmoid4(1) - _sigmoid4(0))


def _clamped_sigmoid3(x):
    x = np.clip(x, 0, 1)
    return _extrapol_sigmoid3(x)


def _clamped_sigmoid4(x):
    x = np.clip(x, 0, 1)
    return _extrapol_sigmoid4(x)


def _hermite(x):
    x = np.clip(x, 0, 1)
    return 3 * x**2 - 2 * x**3


def _quintic(x):
    x = np.clip(x, 0, 1)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def _poly_mix(x):
    return 0.5 * _hermite(x) + 0.5 * _quintic(x)


def _converge_quadratic(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 2


_BLEND_FUNCTIONS = {
    "linear": _linear,
    "sigmoid3": _sigmoid3,
    "sigmoid4": _sigmoid4,
    "extrapol_sigmoid3": _extrapol_sigmoid3,
    "extrapol_sigmoid4": _extrapol_sigmoid4,
    "clamped_sigmoid3": _clamped_sigmoid3,
    "clamped_sigmoid4": _clamped_sigmoid4,
    "hermite": _hermite,
    "quintic": _quintic,
    "poly_mix": _poly_mix,
    "converge_quadratic": _converge_quadratic,
}

BLEND_TYPES = list(_BLEND_FUNCTIONS)
"""Names of all available blending functions."""


def blending_factor(x: np.ndarray, type: str) -> np.ndarray:
    if type not in _BLEND_FUNCTIONS:
        raise ValueError(f"Unknown blending function {type}. Must be one of {BLEND_TYPES}")
    return _BLEND_FUNCTIONS[type](x)


def prepare_array(value: Any, target_dims: fd.DimensionSet) -> fd.FlodymArray:
    if isinstance(value, (int, float)):
        array = fd.FlodymArray(dims=target_dims)
        array[...] = value
    elif isinstance(value, fd.FlodymArray):
        array = value.cast_to(target_dims)
    else:
        raise ValueError("value must be either a FlodymArray or a scalar.")
    return array


@dataclass(frozen=True)
class TrendWindowSelection:
    """
    Local polynomial estimates of a derivative at the last time step of a time series, for a
    range of window sizes, together with their estimated bias and variance.

    Created by `select_trend_window`, which describes the method. All arrays have the candidate
    windows as first axis, followed by the spatial shape of the series.

    Attributes:
        windows (np.ndarray): Candidate window sizes in time steps. A window of size ``n`` fits
            the ``n + 1`` most recent values.
        derivative (np.ndarray): Derivative estimate for each window.
        bias (np.ndarray): Estimated bias of the derivative estimate for each window.
        variance (np.ndarray): Estimated variance of the derivative estimate for each window.
        pilot_window (np.ndarray): Window of the pilot fit from which bias and variance are
            estimated. Shape ``(spatial...)``.
        selected_window (np.ndarray): Window chosen by the search over the estimated mean
            squared error. Shape ``(spatial...)``.
    """

    windows: np.ndarray
    derivative: np.ndarray
    bias: np.ndarray
    variance: np.ndarray
    pilot_window: np.ndarray
    selected_window: np.ndarray

    @property
    def mse(self) -> np.ndarray:
        """Estimated mean squared error of the derivative estimate for each window."""
        return self.bias**2 + self.variance

    def derivative_at(self, window: int | np.ndarray | None = None) -> np.ndarray:
        """
        Derivative estimate for the given window.

        Args:
            window (int | np.ndarray | None): Window size in time steps, either a scalar or an
                array broadcastable to the spatial shape. Defaults to `selected_window`.

        Returns:
            np.ndarray: Derivative estimate, shape ``(spatial...)``.

        Raises:
            ValueError: If a window is not among the candidate `windows`.
        """
        window = self.selected_window if window is None else window
        window = np.broadcast_to(window, self.derivative.shape[1:])
        window_idx = np.searchsorted(self.windows, window)
        if np.any(window_idx >= len(self.windows)) or np.any(self.windows[window_idx] != window):
            raise ValueError(
                f"Window must be an integer between {self.windows[0]} and {self.windows[-1]}."
            )
        return np.take_along_axis(self.derivative, window_idx[np.newaxis], axis=0)[0]


def select_trend_window(
    time: np.ndarray,
    values: np.ndarray,
    derivative_order: int,
    degree: int | None = None,
) -> TrendWindowSelection:
    """
    Estimate a derivative at the last time step of a time series by local polynomial fits over
    the most recent values, and choose the window size of the fit in a data-driven way.

    The window is chosen by the refined bandwidth selector of Fan & Gijbels (1995), Section 4.1
    "Constant bandwidth", evaluated at the single point of interest, the last time step. The
    local fits use a one-sided uniform kernel, i.e. ordinary least squares over the ``n + 1``
    most recent values for a window of size ``n``.

    1. Pilot fit: A polynomial of degree ``degree + 2`` is fitted for every window. The window
       minimizing the Extended Cross-Validation criterion (Section 2) is multiplied by the
       adjusting constant for estimating the coefficient of order ``degree + 1``. The fit over
       this pilot window provides estimates of the coefficients of order ``degree + 1`` and
       ``degree + 2`` and of the noise variance.
    2. For every window, the bias of the derivative estimate of the degree ``degree`` fit is
       estimated from the pilot coefficients (Section 3, Eq. 3.3) and its variance from the pilot
       noise variance (Eq. 3.5). Their sum, the estimated mean squared error, is minimized by
       the search of Section 4.2: starting from the smallest window, the window grows by 10%
       (at least one time step) until the criterion increased three times in a row. This
       avoids large windows unless necessary, where the bias estimate extrapolates the pilot
       polynomial beyond its window.

    Deviations from the paper, owing to the evaluation at the boundary of short series:

    - The pilot window minimizes the Extended Cross-Validation criterion over all windows. The
      search of Section 4.2 stops there at the first bump caused by noise.
    - Pilot windows leave at least as many residual degrees of freedom as the pilot polynomial
      has coefficients, because a noise variance estimated from fewer values makes the
      Extended Cross-Validation criterion unreliable.
    - The moments ``s_{n,j}`` with ``j > degree + 2`` are not set to zero in the bias estimate.
      The paper does this to reduce collinearity in the interior, where these moments are of
      higher order; for one-sided windows they contribute to the leading bias term.

    Reference:
        Fan, J. and Gijbels, I. (1995). Data-driven bandwidth selection in local polynomial
        fitting: variable bandwidth and spatial adaptation. Journal of the Royal Statistical
        Society, Series B, 57(2), 371-394.

    Args:
        time (np.ndarray): 1-D array of equally spaced time values.
        values (np.ndarray): Data with time as first axis and arbitrary spatial shape
            thereafter. Each spatial element is treated as a separate series.
        derivative_order (int): Order of the derivative to estimate.
        degree (int | None): Degree of the local polynomial. Defaults to
            ``derivative_order + 1``, as recommended by the paper.

    Returns:
        TrendWindowSelection: Estimates, bias and variance for all candidate windows.

    Raises:
        ValueError: If ``degree`` is smaller than ``derivative_order``, or if the series is too
            short for the pilot fit.
    """
    degree = derivative_order + 1 if degree is None else degree
    if degree < derivative_order:
        raise ValueError(
            f"Degree {degree} must be at least the derivative order {derivative_order}."
        )
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    spatial_shape = values.shape[1:]
    series = values.reshape(len(values), -1)

    pilot_window, pilot_coefficients, pilot_noise_variance = _pilot_fit(time, series, degree)
    high_orders = np.arange(degree + 1, degree + 3)
    pilot_high_coefficients = pilot_coefficients[high_orders]

    windows = np.arange(degree, len(time))
    derivative = np.empty((len(windows), series.shape[1]))
    bias = np.empty_like(derivative)
    variance = np.empty_like(derivative)
    derivative_factor = math.factorial(derivative_order)
    for window_idx, window in enumerate(windows):
        fit = _fit_window(time, series, window, degree)
        # the coefficients beta_j = m^(j) / j! of the scaled time u = (t - t_last) / scale
        # are beta_j * scale**j; the derivative of order nu thus gets a factor nu! / scale**nu
        to_derivative = derivative_factor / fit.scale**derivative_order
        derivative[window_idx] = derivative_factor * fit.coefficients[derivative_order]

        # Eq. (3.3): bias = S_n^-1 X^T tau, with tau the next two Taylor terms of the pilot fit
        scaled_offsets = fit.design[:, 1]
        scaled_high_coefficients = pilot_high_coefficients * fit.scale ** high_orders[:, None]
        taylor_remainder = scaled_offsets[:, None] ** high_orders @ scaled_high_coefficients
        scaled_bias = fit.inverse_moments @ fit.design.T @ taylor_remainder
        bias[window_idx] = to_derivative * scaled_bias[derivative_order]

        # Eq. (3.5): for a uniform kernel, S_n^-1 S_n^* S_n^-1 reduces to S_n^-1
        inverse_moment = fit.inverse_moments[derivative_order, derivative_order]
        variance[window_idx] = to_derivative**2 * inverse_moment * pilot_noise_variance

    selected_window = _search_minimum(windows, bias**2 + variance)
    return TrendWindowSelection(
        windows=windows,
        derivative=derivative.reshape((len(windows),) + spatial_shape),
        bias=bias.reshape((len(windows),) + spatial_shape),
        variance=variance.reshape((len(windows),) + spatial_shape),
        pilot_window=pilot_window.reshape(spatial_shape),
        selected_window=selected_window.reshape(spatial_shape),
    )


def _search_minimum(
    windows: np.ndarray,
    criterion: np.ndarray,
    growth_factor: float = 1.1,
    max_consecutive_increases: int = 3,
) -> np.ndarray:
    """
    Minimize a criterion over windows by the search of Fan & Gijbels (1995), Section 4.2.

    Starting from the smallest window, the window is inflated by ``growth_factor``, by at least
    one time step, until the criterion increased ``max_consecutive_increases`` times in a row.
    The evaluated window with the smallest criterion is returned.

    Args:
        windows (np.ndarray): Consecutive integer windows, shape ``(n_windows,)``.
        criterion (np.ndarray): Criterion per window and series, shape ``(n_windows, n_series)``.
        growth_factor (float): Factor by which the window grows in each step. Defaults to 1.1.
        max_consecutive_increases (int): Number of consecutive increases after which the search
            stops. Defaults to 3.

    Returns:
        np.ndarray: Selected window per series, shape ``(n_series,)``.
    """
    grid = [windows[0]]
    while (next_window := max(grid[-1] + 1, round(growth_factor * grid[-1]))) <= windows[-1]:
        grid.append(next_window)
    grid = np.array(grid)
    grid_criterion = criterion[grid - windows[0]]

    evaluated = np.ones_like(grid_criterion, dtype=bool)
    consecutive_increases = np.zeros(criterion.shape[1], dtype=int)
    for grid_idx in range(1, len(grid)):
        evaluated[grid_idx] = evaluated[grid_idx - 1] & (
            consecutive_increases < max_consecutive_increases
        )
        increased = grid_criterion[grid_idx] > grid_criterion[grid_idx - 1]
        consecutive_increases = np.where(increased, consecutive_increases + 1, 0)
    evaluated_criterion = np.where(evaluated, grid_criterion, np.inf)
    return grid[np.argmin(evaluated_criterion, axis=0)]


class _WindowFit(NamedTuple):
    """Least squares polynomial fit over the most recent values of several series."""

    scale: float
    """Length of the window in time units, used to scale the time offsets to [-1, 0]."""
    design: np.ndarray
    """Design matrix of the scaled time offsets, shape ``(window + 1, degree + 1)``."""
    inverse_moments: np.ndarray
    """Inverse of ``design.T @ design``."""
    coefficients: np.ndarray
    """Polynomial coefficients in unscaled time, shape ``(degree + 1, n_series)``."""
    noise_variance: np.ndarray
    """Residual sum of squares per degree of freedom, shape ``(n_series,)``."""


def _fit_window(time: np.ndarray, series: np.ndarray, window: int, degree: int) -> _WindowFit:
    """Fit a polynomial to the ``window + 1`` most recent values, centered at the last time."""
    offsets = time[-window - 1 :] - time[-1]
    scale = -offsets[0]
    design = np.vander(offsets / scale, degree + 1, increasing=True)
    inverse_moments = np.linalg.inv(design.T @ design)
    window_values = series[-window - 1 :]
    scaled_coefficients = inverse_moments @ design.T @ window_values
    residuals = window_values - design @ scaled_coefficients
    residual_dof = window - degree
    if residual_dof > 0:
        noise_variance = np.sum(residuals**2, axis=0) / residual_dof
    else:
        noise_variance = np.full(series.shape[1], np.nan)
    coefficients = scaled_coefficients / scale ** np.arange(degree + 1)[:, None]
    return _WindowFit(scale, design, inverse_moments, coefficients, noise_variance)


def _pilot_fit(
    time: np.ndarray, series: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pilot stage of `select_trend_window`: fit polynomials of degree ``degree + 2`` and choose
    their window by the Extended Cross-Validation criterion (Fan & Gijbels 1995, Eq. 2.4)
    ``ECV = sigma^2 (1 + (p + 1) V_0)``, with ``V_0`` the first diagonal element of ``S_n^-1``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The pilot window per series, the pilot
        polynomial coefficients of shape ``(degree + 3, n_series)``, and the pilot noise
        variance per series.
    """
    pilot_degree = degree + 2
    n_coefficients = pilot_degree + 1
    # leave as many residual degrees of freedom as there are coefficients
    min_window = 2 * n_coefficients - 1
    max_window = len(time) - 1
    if max_window < min_window:
        raise ValueError(
            f"At least {min_window + 1} time steps are needed to select the window of a "
            f"degree {degree} fit, but got {len(time)}."
        )

    windows = np.arange(min_window, max_window + 1)
    fits = [_fit_window(time, series, window, pilot_degree) for window in windows]
    ecv = np.array(
        [fit.noise_variance * (1 + n_coefficients * fit.inverse_moments[0, 0]) for fit in fits]
    )
    ecv_window = windows[np.argmin(ecv, axis=0)]

    # uniform kernel on [-1, 0]: s_j = int u^j du, and K^2 = K
    orders = np.arange(2 * pilot_degree + 3)
    uniform_moments = (-1.0) ** orders / (orders + 1)
    adjusting_constant = _adjusting_constant(
        pilot_degree, degree + 1, uniform_moments, uniform_moments
    )
    pilot_window = np.clip(np.round(adjusting_constant * ecv_window), min_window, max_window)
    pilot_window = pilot_window.astype(int)

    pilot_idx = pilot_window - min_window
    series_idx = np.arange(series.shape[1])
    coefficients = np.stack([fit.coefficients for fit in fits])[pilot_idx, :, series_idx].T
    noise_variance = np.stack([fit.noise_variance for fit in fits])[pilot_idx, series_idx]
    return pilot_window, coefficients, noise_variance


def _adjusting_constant(
    degree: int,
    derivative_order: int,
    kernel_moments: np.ndarray,
    squared_kernel_moments: np.ndarray,
) -> float:
    """
    Ratio ``adj_{p,nu}`` of the bandwidth minimizing the mean squared error of the derivative
    estimate of order ``nu`` to the bandwidth minimizing the Extended Cross-Validation
    criterion, for a local polynomial of degree ``p`` (Fan & Gijbels 1995, Section 2).

    Args:
        degree (int): Degree ``p`` of the local polynomial.
        derivative_order (int): Derivative order ``nu``.
        kernel_moments (np.ndarray): ``int u^j K(u) du`` for ``j = 0, ..., 2p + 2``.
        squared_kernel_moments (np.ndarray): ``int u^j K(u)^2 du`` for ``j = 0, ..., 2p + 2``.

    Returns:
        float: The adjusting constant.
    """
    indices = np.add.outer(np.arange(degree + 1), np.arange(degree + 1))
    inverse_moments = np.linalg.inv(kernel_moments[indices])
    variance_factors = inverse_moments @ squared_kernel_moments[indices] @ inverse_moments
    bias_moments = kernel_moments[degree + 1 : 2 * degree + 2]
    bias_factor = (inverse_moments @ bias_moments)[derivative_order]
    residual_factor = (
        kernel_moments[2 * degree + 2] - bias_moments @ inverse_moments @ bias_moments
    ) / kernel_moments[0]
    ratio = (
        (2 * derivative_order + 1)
        * variance_factors[derivative_order, derivative_order]
        * residual_factor
        / ((degree + 1 - derivative_order) * variance_factors[0, 0] * bias_factor**2)
    )
    return ratio ** (1 / (2 * degree + 3))


class CriticallyDampedBlender:

    def __init__(
        self,
        time: Union[np.ndarray, list],
        historical: np.ndarray,
        prediction: np.ndarray,
    ):
        """
        Args:
            time (Union[np.ndarray, list]): Time values including historical and future periods.
            Same length as prediction
            historical (np.ndarray): Historical stock data with time as the first axis.
            prediction (np.ndarray): Extrapolated stock data from the regression, same shape
                as the full output (covering both historical and future period in first axis).
        """
        self.time = np.array(time)
        self.historical = historical
        self.prediction = prediction

        assert (
            self.time.shape[0] == self.prediction.shape[0]
        ), "Time and prediction must have the same length."
        assert (
            self.historical.shape[1:] == self.prediction.shape[1:]
        ), "Historical and prediction must have the same shape, except along the time dimension."
        assert (
            self.historical.shape[0] <= self.prediction.shape[0]
        ), "Historical data cannot be longer than prediction."

    def trend_window_selection(self, derivative_order: int) -> TrendWindowSelection:
        """
        Estimates of the historical trend's derivative at the last historical time step for all
        fitting windows, with estimated bias and variance. See `select_trend_window`.

        Args:
            derivative_order (int): 1 for the velocity, 2 for the acceleration.

        Returns:
            TrendWindowSelection: Estimates, bias and variance for all candidate windows.
        """
        return select_trend_window(
            self.time[: len(self.historical)], self.historical, derivative_order
        )

    def blend(
        self,
        approaching_time: float = 50,
        velocity_window: int | np.ndarray | None = None,
        acceleration_window: int | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Blend historical and extrapolated values using a forced critically damped system
        approach (PDA-controller logic) to ensure a C2-continuous transition.

        The transition is modeled as a third-order critically damped tracking system:

            Y''' + 3kY'' + 3k²Y' + k³Y = k³P(t) + 3k²P'(t) + 3kP''(t)

        where Y is the blended trajectory, P the extrapolation target, and
        k = 6.30 / approaching_time the damping parameter. The initial position is the
        last historical value; the initial velocity and acceleration are estimated from
        local polynomial fits to the recent historical trend, so position, slope, and
        curvature are all continuous at the transition point. The fitting windows are
        chosen per series by the data-driven selector in `select_trend_window`, unless
        given explicitly. The ODE is integrated with a semi-implicit Euler method;
        P''(t) is estimated with a look-ahead so the controller reacts to upcoming
        changes in P (e.g. saturation) before they occur, while P'(t) uses the plain
        local slope.

        Args:
            approaching_time (float): Characteristic timescale in years. Sets the damping
                parameter ``k = 6.30 / approaching_time`` (95% step-response convergence
                within ``approaching_time`` years), derived from solving
                ``e^{-x}(1 + x + x**2/2) = 0.05`` for ``x = k * approaching_time``.
                Must satisfy ``k * dt <= 0.5`` for
                numerical stability, i.e. ``approaching_time >= 12.6 years``. Defaults to 50.
            velocity_window (int | np.ndarray | None): Window in time steps of the fit for the
                initial velocity, scalar or per series. Defaults to the data-driven choice.
            acceleration_window (int | np.ndarray | None): Window in time steps of the fit for
                the initial acceleration, scalar or per series. Defaults to the data-driven
                choice.

        Returns:
            np.ndarray: Stock array with exact historical values preserved up to the last
            historical index and a smooth blended trajectory thereafter.
        """
        last_history_idx = len(self.historical) - 1

        # 1. Isolate the time window and prediction values we need to integrate over
        t_future = self.time[last_history_idx:]
        p_future = self.prediction[last_history_idx:]

        # 2. Set the initial conditions at the transition point
        y0 = self.historical[last_history_idx, :]
        v0 = self.trend_window_selection(derivative_order=1).derivative_at(velocity_window)
        a0 = self.trend_window_selection(derivative_order=2).derivative_at(acceleration_window)

        # 3. Integrate to find the blended future path Y(t)
        y_future = self._integrate_transition(
            y0,
            v0,
            a0,
            t_future,
            p_future,
            approaching_time,
        )

        # 4. Construct the final contiguous array
        blended_stock = self.prediction.copy()
        blended_stock[:last_history_idx] = self.historical[
            :last_history_idx
        ]  # Preserve exact history
        blended_stock[last_history_idx:] = y_future  # Apply blended future

        return blended_stock

    def _integrate_transition(
        self,
        y0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        t_array: np.ndarray,
        p_array: np.ndarray,
        approaching_time: float,
    ) -> np.ndarray:
        """
        Integrate a trajectory from an initial state (y0, v0, a0) that smoothly tracks a
        target prediction p_array using a third-order critically damped controller.

        The controller drives Y toward P via:
            Y''' + 3k·Y'' + 3k²·Y' + k³Y = k³P(t) + 3k²·P'(t) + 3k·P''(t),
            k = 6.30 / approaching_time
        integrated with a semi-implicit Euler method. P'(t) is the local slope; P''(t)
        is estimated with a look-ahead to prevent overshoot during saturation phases.

        Args:
            y0 (np.ndarray): Initial position at the transition point. Shape ``(spatial...)``.
            v0 (np.ndarray): Initial velocity (slope) at the transition point, same shape as ``y0``.
            a0 (np.ndarray): Initial acceleration (curvature) at the transition point,
                same shape as ``y0``.
            t_array (np.ndarray): 1D array of time values starting at the transition point.
            p_array (np.ndarray): Target prediction array with time as the first axis,
                shape ``(len(t_array), spatial...)``. Must be uniformly spaced in time.
            approaching_time (float): Characteristic timescale in years. Sets the damping
                parameter ``k = 6.30 / approaching_time``.

        Returns:
            np.ndarray: Integrated trajectory array of shape ``(len(t_array), spatial...)``.

        Raises:
            ValueError: If ``k * dt > 0.5``, i.e. ``approaching_time`` is too small relative
                to the time step for the integration to be numerically stable.
        """
        n_steps = len(t_array)
        dt = t_array[1] - t_array[0]

        # 6.30 is the solution to (1+x+x²/2)*exp(-x) = 0.05: the third-order critically
        # damped step response. k = 6.30 / approaching_time means 95% convergence within
        # approaching_time years.
        k = 6.30 / approaching_time

        # The semi-implicit Euler scheme diverges for k*dt above ~0.52.
        if k * dt > 0.5:
            raise ValueError(
                f"approaching_time={approaching_time} is too small for time step dt={dt}: "
                f"k*dt = {k * dt:.2f} > 0.5 makes the integration numerically unstable. "
                f"Use approaching_time >= {12.6 * dt:.1f}."
            )

        # --- Precompute predictor velocity and look-ahead acceleration ---
        vp_array, ap_array = self._calculate_derivatives(p_array, dt, n_steps, approaching_time)

        # --- Initialize state ---
        y = np.zeros_like(p_array, dtype=float)
        v = np.zeros_like(p_array, dtype=float)
        a = np.zeros_like(p_array, dtype=float)
        y[0], v[0], a[0] = y0.copy(), v0.copy(), a0.copy()
        y_curr, v_curr, a_curr = y[0].copy(), v[0].copy(), a[0].copy()

        # --- Integrate ---
        for i in range(1, n_steps):
            # 1. Compute jerk.
            da_dt = (
                k**3 * (p_array[i - 1] - y_curr)
                + 3 * k**2 * (vp_array[i - 1] - v_curr)
                + 3 * k * (ap_array[i - 1] - a_curr)
            )
            # 2. Update acceleration, velocity, and position.
            a_curr = a_curr + da_dt * dt
            v_curr = v_curr + a_curr * dt
            y_curr = y_curr + v_curr * dt
            # Store results.
            y[i], v[i], a[i] = y_curr, v_curr, a_curr

        return y

    def _calculate_derivatives(
        self,
        p_array: np.ndarray,
        dt: float,
        n_steps: int,
        approaching_time: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate P'(t) at each timestep, and P''(t + n_fwd(t)*dt) — the curvature of the
        prediction looked up n_fwd steps ahead. Only the curvature term is shifted:
        looking ahead lets it anticipate future changes in P (e.g. saturation), so the
        derivative term of the controller begins reacting before P actually flattens,
        preventing overshoot. The velocity term uses the plain local slope.

        n_fwd ramps continuously from n_fwd_max down to 0 over the first half of
        approaching_time, then stays at 0 (plain local curvature). The continuous ramp
        avoids the discrete jumps that arise from integer look-ahead steps. Both
        derivatives are computed on the raw prediction first; only the curvature is then
        sampled at the shifted position, so the ramp itself does not distort the estimate.

        Returns:
            tuple[np.ndarray, np.ndarray]: Local first derivative and look-ahead second
            derivative of the prediction, each of shape ``(n_steps, spatial...)``.
        """
        n_fwd_max = 5
        n_ramp_steps = max(1, int((approaching_time / 2) / dt))

        # Continuous look-ahead amount for each step: 5 → 0 over n_ramp_steps, then 0
        n_fwd_cont = n_fwd_max * np.maximum(0.0, 1.0 - np.arange(n_steps) / n_ramp_steps)

        # Slope and curvature of p at every step
        # (central differences; second-order one-sided at boundaries)
        vp_raw = np.gradient(p_array, dt, axis=0)
        ap_raw = np.gradient(vp_raw, dt, axis=0)

        # For each step i, look n_fwd_cont[i] steps forward in the derivative arrays
        look_pos = np.clip(np.arange(n_steps, dtype=float) + n_fwd_cont, 0, n_steps - 1)

        # Fractional interpolation between the two bracketing integer positions
        lo = look_pos.astype(int)
        hi = np.minimum(lo + 1, n_steps - 1)
        w = (look_pos - lo).reshape((-1,) + (1,) * (p_array.ndim - 1))
        ap_lookahead = (1 - w) * ap_raw[lo] + w * ap_raw[hi]
        return vp_raw, ap_lookahead
