# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Trend windows for blending, chosen on consumption

    `CriticallyDampedBlender` transitions from the historic in-use stock to the regression,
    starting with the slope and curvature of the historic stock trend, estimated by local
    polynomial fits over a window of recent years. The model chooses these windows on the stock
    curve. As the stock accumulates consumption, it is so smooth that the selector of
    Fan & Gijbels (1995) treats almost every fluctuation as signal.

    This notebook instead chooses the windows on the consumption curve of the selected model,
    region and sector (`remind_mfa.common.data_blending.select_trend_window`). Following the
    paper, a derivative of order ν is estimated with a polynomial of degree ν + 1 (Section 4;
    degree − ν must be odd), whose bias is estimated from a pilot polynomial of degree ν + 3
    (Sections 3 and 4.1). Consumption is the increase of the stock plus the smooth outflow, so a
    polynomial of degree p fitted to n years of consumption corresponds to a polynomial of degree
    p + 1 fitted to the stock over one more year:

    - the consumption level (linear fit) gives the window for the stock slope (quadratic fit),
    - the consumption slope (quadratic fit) gives the window for the stock curvature (cubic fit).

    The table below compares the windows the model chooses on the stock, the windows chosen on
    consumption and windows chosen by hand. It also lists the windows of the earlier blender,
    whose stock window decreases from 10 to 1 year for lifetimes from 3 to 30 years (logarithmic
    scale), with a linear fit for the slope and a quadratic fit for the curvature over at least
    2 years. The same windows are also used with weighted fits, whose weights decrease linearly
    with the age of the data point. The comparison at the end shows the resulting consumption of
    the future MFA for all models, sectors and regions.
    """)
    return


@app.cell
def _(mo):
    mo.md("This notebook is completely AI generated, with very little human checking!").callout(
        kind="danger", title="AI Disclaimer"
    )
    return


@app.cell
def _():
    import flodym as fd
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from dotenv import load_dotenv

    from remind_mfa.common.config_loader import load_config
    from remind_mfa.common.data_blending import CriticallyDampedBlender, select_trend_window
    from remind_mfa.common.helpers import ModelNames, init_model

    return (
        CriticallyDampedBlender,
        ModelNames,
        fd,
        go,
        init_model,
        load_config,
        load_dotenv,
        mo,
        np,
        pd,
        select_trend_window,
    )


@app.cell
def _(ModelNames, mo):
    model_picker = mo.ui.dropdown(
        options=[model_name.value for model_name in ModelNames],
        value=ModelNames.STEEL.value,
        label="Model",
    )
    model_picker
    return (model_picker,)


@app.cell
def _(ModelNames, init_model, load_config, load_dotenv, mo):
    repo_root = mo.notebook_dir().parent
    load_dotenv(repo_root / ".env")

    @mo.cache
    def run_model(model_name: str):
        """Run a model with the default configuration."""
        config = load_config(["default"], ModelNames(model_name), config_dir=repo_root / "config")
        config["input"]["input_data_path"] = str(repo_root / "data_in")
        model_run = init_model(cfg=config)
        model_run.run()
        return model_run

    return (run_model,)


@app.cell
def _(mo, model_picker, run_model):
    with mo.status.spinner(title=f"Running the {model_picker.value} model..."):
        model = run_model(model_picker.value)
    model_name = model_picker.value
    stock_extrapolation = model.stock_handler
    sector_letter = model.end_use_good_letter
    # in-use stock of the future MFA whose inflow is the consumption, and its slice for the
    # material of the model
    consumption_stocks = {
        "steel": ("in_use", {}),
        "plastics": ("in_use_dsm", {}),
        "cement": ("in_use", {"k": "cement"}),
    }
    return (
        consumption_stocks,
        model,
        model_name,
        sector_letter,
        stock_extrapolation,
    )


@app.cell
def _(mo, model, sector_letter):
    region_picker = mo.ui.dropdown(
        options=model.dims["r"].items,
        value="CHA" if "CHA" in model.dims["r"].items else model.dims["r"].items[0],
        label="Region",
    )
    sector_picker = mo.ui.dropdown(
        options=model.dims[sector_letter].items,
        value=model.dims[sector_letter].items[0],
        label="Sector",
    )
    mo.hstack([region_picker, sector_picker], justify="start")
    return region_picker, sector_picker


@app.cell
def _(
    CriticallyDampedBlender,
    model,
    np,
    region_picker,
    sector_letter,
    sector_picker,
    select_trend_window,
    stock_extrapolation,
):
    region = region_picker.value
    sector = sector_picker.value
    series_slice = {"r": region, sector_letter: sector}
    time = np.array(model.dims["t"].items)
    historic_time = np.array(model.dims["h"].items)

    historic_consumption = model.historic_mfa.stocks[model.historic_stock_name].inflow
    consumption_curve = historic_consumption[series_slice].values
    level_selection = select_trend_window(historic_time, consumption_curve, derivative_order=0)
    slope_selection = select_trend_window(historic_time, consumption_curve, derivative_order=1)

    blender = CriticallyDampedBlender(
        time=time,
        historical=stock_extrapolation.historic_stocks_pc[series_slice].values[:, np.newaxis],
        prediction=stock_extrapolation.fitted_regression[series_slice].values[:, np.newaxis],
    )
    # windows the model chooses on the stock curve, as consumption windows
    model_level_window = int(blender.trend_window_selection(1).selected_window[0]) - 1
    model_slope_window = int(blender.trend_window_selection(2).selected_window[0]) - 1
    return (
        blender,
        consumption_curve,
        historic_time,
        level_selection,
        model_level_window,
        model_slope_window,
        region,
        sector,
        series_slice,
        slope_selection,
        time,
    )


@app.cell
def _(historic_time, level_selection, mo, slope_selection):
    # the stock fit reaches one year further back than the consumption fit
    max_slider_window = min(60, len(historic_time) - 2)
    level_window_slider = mo.ui.slider(
        start=int(level_selection.windows[0]),
        stop=max_slider_window,
        value=min(int(level_selection.selected_window), max_slider_window),
        label="Window for the consumption level [years]",
        show_value=True,
        include_input=True,
    )
    slope_window_slider = mo.ui.slider(
        start=int(slope_selection.windows[0]),
        stop=max_slider_window,
        value=min(int(slope_selection.selected_window), max_slider_window),
        label="Window for the consumption slope [years]",
        show_value=True,
        include_input=True,
    )
    mo.vstack([level_window_slider, slope_window_slider])
    return level_window_slider, slope_window_slider


@app.cell
def _():
    # approaching time of the model, cf. StockExtrapolation.smooth_transition
    approaching_time = 50
    return (approaching_time,)


@app.cell
def _(model, np, series_slice):
    def lifetime_dependent_window(
        lifetime: float,
        lower_lifetime: float = 3.0,
        upper_lifetime: float = 30.0,
        min_window: int = 1,
        max_window: int = 10,
    ) -> int:
        """Stock window of the earlier blender, mapped from max_window (shortest lifetime) to
        min_window (longest lifetime) on a logarithmic scale."""
        clipped = np.clip(lifetime, lower_lifetime, upper_lifetime)
        alpha = np.log(clipped / lower_lifetime) / np.log(upper_lifetime / lower_lifetime)
        return int(np.round(max_window - alpha * (max_window - min_window)))

    # lifetime as in CommonModel.lifetime_limit, which the earlier blender received
    series_lifetime = model.lifetime_limit()[series_slice].values.item()
    lifetime_velocity_window = lifetime_dependent_window(series_lifetime)
    lifetime_acceleration_window = max(2, lifetime_velocity_window)
    return lifetime_acceleration_window, lifetime_velocity_window, series_lifetime


@app.cell
def _(
    level_selection,
    level_window_slider,
    lifetime_acceleration_window,
    lifetime_velocity_window,
    mo,
    model_level_window,
    model_slope_window,
    series_lifetime,
    slope_selection,
    slope_window_slider,
):
    megatonnes = 1e-6

    def describe(
        name: str,
        unit: str,
        selection,
        model_window: int,
        lifetime_window: int,
        chosen_window: int,
    ) -> str:
        def cell(window: int) -> str:
            return f"{window} ({selection.derivative_at(window) * megatonnes:.4g})"

        # the lifetime-based trend uses fits of one degree lower, so no estimate is comparable
        cells = [
            cell(model_window),
            cell(int(selection.selected_window)),
            str(lifetime_window),
            cell(chosen_window),
        ]
        return f"| {name} [{unit}] | " + " | ".join(cells) + " |"

    mo.md(
        "Windows in years, as consumption windows, with the estimate at the last historic year "
        "in brackets. The lifetime-based windows follow the earlier blender (lifetime "
        f"{series_lifetime:.3g} years: stock windows {lifetime_velocity_window} for the slope "
        f"and {lifetime_acceleration_window} for the curvature, with linear and quadratic fits). "
        "The weighted lifetime-based fits use the same windows.\n\n"
        "| | model (on stock) | optimal on consumption | lifetime-based | chosen |\n"
        "|---|---|---|---|---|\n"
        + describe(
            "level",
            "Mt/yr",
            level_selection,
            model_level_window,
            lifetime_velocity_window - 1,
            level_window_slider.value,
        )
        + "\n"
        + describe(
            "slope",
            "Mt/yr²",
            slope_selection,
            model_slope_window,
            lifetime_acceleration_window - 1,
            slope_window_slider.value,
        )
    )
    return (megatonnes,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Bias and variance per window

    For every window, the bias and variance of the estimated consumption level and slope at the
    last historic year are estimated from a pilot fit, a polynomial of two degrees higher over the
    pilot window (Fan & Gijbels 1995, Eqs. 3.3 and 3.5). The optimal window minimizes their sum,
    the estimated mean squared error, with the search of Section 4.2: it grows the window from the
    smallest one and stops after three consecutive increases. For windows much longer than the
    pilot window, the bias estimate extrapolates the pilot polynomial and is unreliable.
    """)
    return


@app.cell
def _(
    go,
    level_selection,
    level_window_slider,
    megatonnes,
    mo,
    pd,
    slope_selection,
    slope_window_slider,
):
    def bias_variance_view(selection, chosen_window: int, unit: str):
        """Plot and table of the estimated bias and variance per window."""
        table = pd.DataFrame(
            {
                "window [years]": selection.windows,
                f"estimate [{unit}]": selection.derivative * megatonnes,
                f"bias [{unit}]": selection.bias * megatonnes,
                f"variance [({unit})²]": selection.variance * megatonnes**2,
                f"MSE [({unit})²]": selection.mse * megatonnes**2,
            }
        )
        optimal_window = int(selection.selected_window)
        figure = go.Figure()
        figure.add_scatter(
            x=table["window [years]"],
            y=table[f"bias [{unit}]"] ** 2,
            name="bias²",
            line=dict(color="#2a78d6", width=2),
        )
        figure.add_scatter(
            x=table["window [years]"],
            y=table[f"variance [({unit})²]"],
            name="variance",
            line=dict(color="#eb6834", width=2),
        )
        figure.add_scatter(
            x=table["window [years]"],
            y=table[f"MSE [({unit})²]"],
            name="MSE = bias² + variance",
            line=dict(color="#0b0b0b", width=2, dash="dash"),
        )
        figure.add_vline(
            x=optimal_window, line=dict(color="#2a78d6", width=1), annotation_text="optimal"
        )
        figure.add_vline(
            x=chosen_window,
            line=dict(color="#eb6834", width=1, dash="dot"),
            annotation_text="chosen",
            annotation_position="bottom right",
        )
        figure.update_layout(
            xaxis=dict(title="Window [years]", range=[0, 60]),
            yaxis=dict(title=f"[({unit})²]", type="log"),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        scientific = "{:.3e}".format
        formatted_table = mo.ui.table(
            table,
            selection=None,
            page_size=20,
            format_mapping={column: scientific for column in table.columns[1:]},
            label=f"Optimal window: {optimal_window} years, pilot window: "
            f"{int(selection.pilot_window)} years",
        )
        return mo.vstack([figure, formatted_table])

    mo.ui.tabs(
        {
            "Consumption level (window of the stock slope)": bias_variance_view(
                level_selection, level_window_slider.value, "Mt/yr"
            ),
            "Consumption slope (window of the stock curvature)": bias_variance_view(
                slope_selection, slope_window_slider.value, "Mt/yr²"
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison for all models, sectors and regions

    The figures below compare the consumption of the windows of the model, the lifetime-based
    windows with and without weights and the earlier C1 blender for every region, one figure per model and
    sector. The earlier C1 blender is the second-order controller before the C2 version: it starts
    from the lifetime-based stock slope and does not match the stock curvature. The weighted
    lifetime-based fits use the lifetime-based windows, with least-squares weights that decrease
    linearly from 1 for the last historic year to 1 / (window + 1) for the oldest year. For
    windows of 1 year for the slope and 2 years for the curvature, the fit passes through all
    points, so the weights have no effect. The variant with the target curvature uses the
    lifetime-based slope. Where its slope window is longer than 1 year (lifetimes below about
    26 years), it starts from the curvature of the regression instead of the stock, so there is
    no curvature mismatch; elsewhere it equals the lifetime-based windows. All series of a model are blended at once, with one run of the future MFA
    per variant. As the consumption of a series depends only on its own in-use stock, this
    gives the same consumption as blending one series at a time. Running all models takes a few minutes the first time.
    """)
    return


@app.cell
def _(mo):
    all_models_button = mo.ui.run_button(label="Run the comparison for all models")
    all_models_button
    return (all_models_button,)


@app.cell
def _(
    CriticallyDampedBlender,
    approaching_time,
    consumption_stocks,
    fd,
    np,
    run_model,
):
    import functools

    @functools.lru_cache
    def window_variants(model_name: str) -> dict:
        """Consumption of the future MFA of a model for all variants."""
        model_run = run_model(model_name)
        handler = model_run.stock_handler
        letter = model_run.end_use_good_letter
        stock_name, material_slice = consumption_stocks[model_name]
        historic_dims = handler.historic_stocks_pc.dims
        historical = handler.historic_stocks_pc.values
        n_historic = len(historical)
        all_time = np.array(model_run.dims["t"].items)
        all_historic_time = np.array(model_run.dims["h"].items)
        all_blender = CriticallyDampedBlender(
            time=all_time, historical=historical, prediction=handler.fitted_regression.values
        )

        def consumption_of_stocks(stocks_pc_values: np.ndarray) -> fd.FlodymArray:
            """Consumption by time, region and sector for the given normalized stocks per capita,
            denormalized as in StockExtrapolation.extrapolate and CommonModel.get_long_term_stock."""
            stocks_pc = handler.stocks_pc.copy()
            stocks_pc.values[...] = stocks_pc_values
            stock_projection = stocks_pc * handler.pop * model_run.sector_specific_sat_level
            future_mfa = model_run.make_mfa(historic=False)
            future_mfa.compute(stock_projection, model_run.historic_mfa.trade_set)
            inflow = future_mfa.stocks[stock_name].inflow
            if material_slice:
                inflow = inflow[material_slice]
            return inflow.sum_to(("t", "r", letter))

        historic_inflow = model_run.historic_mfa.stocks[model_run.historic_stock_name].inflow
        historic_inflow = historic_inflow.cast_to(historic_dims)

        # lifetime-based windows and fits of the earlier blender, cf. the cell above
        lifetime = model_run.lifetime_limit().cast_to(historic_dims[historic_dims.letters[1:]])
        clipped = np.clip(lifetime.values, 3.0, 30.0)
        alpha = np.log(clipped / 3.0) / np.log(30.0 / 3.0)
        lifetime_velocity_windows = np.round(10 - alpha * 9).astype(int)
        lifetime_acceleration_windows = np.maximum(2, lifetime_velocity_windows)
        velocity = np.empty(historical.shape[1:])
        acceleration = np.empty(historical.shape[1:])
        weighted_velocity = np.empty(historical.shape[1:])
        weighted_acceleration = np.empty(historical.shape[1:])
        for series_idx in np.ndindex(historical.shape[1:]):
            series = historical[(slice(None),) + series_idx]
            for window, degree, target, weighted_target in (
                (lifetime_velocity_windows[series_idx], 1, velocity, weighted_velocity),
                (lifetime_acceleration_windows[series_idx], 2, acceleration, weighted_acceleration),
            ):
                # least-squares weights decreasing linearly with age, from 1 for the last
                # historic year to 1 / (window + 1) for the oldest year
                age = np.arange(window, -1, -1)
                weights = 1 - age / (window + 1)
                for target_array, fit_weights in ((target, None), (weighted_target, weights)):
                    # Polynomial.fit weights the unsquared residuals
                    polynomial = np.polynomial.Polynomial.fit(
                        all_historic_time[-window - 1 :],
                        series[-window - 1 :],
                        deg=degree,
                        w=None if fit_weights is None else np.sqrt(fit_weights),
                    )
                    target_array[series_idx] = polynomial.deriv(degree)(all_historic_time[-1])

        def blend_from_trend(v0: np.ndarray, a0: np.ndarray) -> np.ndarray:
            """Blend from the given initial stock slope and curvature, cf.
            CriticallyDampedBlender.blend."""
            blended = all_blender.prediction.copy()
            blended[n_historic - 1 :] = all_blender._integrate_transition(
                historical[-1],
                v0,
                a0,
                all_time[n_historic - 1 :],
                all_blender.prediction[n_historic - 1 :],
                approaching_time,
            )
            blended[: n_historic - 1] = historical[: n_historic - 1]
            return blended

        lifetime_based = blend_from_trend(velocity, acceleration)
        weighted_lifetime_based = blend_from_trend(weighted_velocity, weighted_acceleration)

        # switched: for slope windows longer than 1 year, start from the curvature of the
        # target as seen by the controller (look-ahead curvature at the transition point), so
        # there is no curvature mismatch
        future_time = all_time[n_historic - 1 :]
        _, target_acceleration = all_blender._calculate_derivatives(
            all_blender.prediction[n_historic - 1 :],
            future_time[1] - future_time[0],
            len(future_time),
            approaching_time,
        )
        switched_acceleration = np.where(
            lifetime_velocity_windows > 1, target_acceleration[0], acceleration
        )
        switched = blend_from_trend(velocity, switched_acceleration)

        def c1_transition(v0: np.ndarray) -> np.ndarray:
            """Blend of the earlier C1 blender, cf. CriticallyDampedBlender before commit
            ebd1eec: Y'' + 2kY' + k²Y = k²P + 2kP' with k = 4.74 / approaching_time, a
            look-ahead slope of P and a quadratic nudge towards P."""
            t = all_time[n_historic - 1 :]
            p = all_blender.prediction[n_historic - 1 :]
            n_steps = len(t)
            dt = t[1] - t[0]
            k = 4.74 / approaching_time
            nudge = np.minimum(1.0, ((t - t[0]) / (10 * approaching_time)) ** 2)
            # look-ahead slope of the prediction, ramping from 5 steps to 0 over half the
            # approaching time
            n_ramp_steps = max(1, int((approaching_time / 2) / dt))
            n_forward = 5 * np.maximum(0.0, 1.0 - np.arange(n_steps) / n_ramp_steps)
            look_position = np.clip(np.arange(n_steps) + n_forward, 0, n_steps - 1)
            low = look_position.astype(int)
            high = np.minimum(low + 1, n_steps - 1)
            weight = (look_position - low).reshape((-1,) + (1,) * (p.ndim - 1))
            p_slope = np.gradient(p, dt, axis=0)
            p_slope = (1 - weight) * p_slope[low] + weight * p_slope[high]
            y = np.zeros_like(p, dtype=float)
            y[0] = historical[-1]
            y_current, v_current = y[0].copy(), v0.copy()
            for i in range(1, n_steps):
                dv_dt = k**2 * (p[i] - y_current) + 2 * k * (p_slope[i] - v_current)
                v_current = v_current + dv_dt * dt
                y_current = y_current + v_current * dt
                y_current = (1 - nudge[i]) * y_current + nudge[i] * p[i]
                v_current = (y_current - y[i - 1]) / dt
                y[i] = y_current
            blended = all_blender.prediction.copy()
            blended[: n_historic - 1] = historical[: n_historic - 1]
            blended[n_historic - 1 :] = y
            return blended

        # the earlier C1 blender started from the lifetime-based stock slope
        c1_blended = c1_transition(velocity)

        return dict(
            model=model_run,
            sector_letter=letter,
            historic_consumption=historic_inflow.sum_to(("h", "r", letter)),
            consumption={
                "Windows of the model (on stock)": consumption_of_stocks(handler.stocks_pc.values),
                "Lifetime-based windows": consumption_of_stocks(lifetime_based),
                "Weighted lifetime-based windows": consumption_of_stocks(weighted_lifetime_based),
                "Earlier C1 blender": consumption_of_stocks(c1_blended),
                "Lifetime-based, target curvature for windows > 1": consumption_of_stocks(
                    switched
                ),
            },
        )

    return (window_variants,)


@app.cell
def _(
    ModelNames,
    all_models_button,
    approaching_time,
    megatonnes,
    mo,
    np,
    window_variants,
):
    from plotly.subplots import make_subplots

    mo.stop(not all_models_button.value, mo.md("Press the button to run the comparison."))

    variant_lines = {
        "Windows of the model (on stock)": dict(color="#1baf7a"),
        "Lifetime-based windows": dict(color="#9b59b6"),
        "Weighted lifetime-based windows": dict(color="#d4a017"),
        "Earlier C1 blender": dict(color="#2a78d6", dash="dash"),
        "Lifetime-based, target curvature for windows > 1": dict(color="#e0503a"),
    }
    n_columns = 4

    def region_figure(variants: dict, model_name: str, sector: str):
        """One panel per region with the consumption of all variants for one sector."""
        model_run = variants["model"]
        letter = variants["sector_letter"]
        regions = model_run.dims["r"].items
        all_time = np.array(model_run.dims["t"].items)
        all_historic_time = np.array(model_run.dims["h"].items)
        future = slice(len(all_historic_time) - 1, None)
        consumption = variants["consumption"]
        n_rows = int(np.ceil(len(regions) / n_columns))
        figure = make_subplots(
            rows=n_rows,
            cols=n_columns,
            subplot_titles=regions,
            shared_xaxes=True,
            vertical_spacing=0.06,
            horizontal_spacing=0.05,
        )
        for region_idx, region_name in enumerate(regions):
            row, col = region_idx // n_columns + 1, region_idx % n_columns + 1
            series = {"r": region_name, letter: sector}
            figure.add_scatter(
                x=all_historic_time,
                y=variants["historic_consumption"][series].values * megatonnes,
                name="Historic",
                legendgroup="Historic",
                showlegend=region_idx == 0,
                line=dict(color="#0b0b0b", width=2),
                row=row,
                col=col,
            )
            for variant_name, line in variant_lines.items():
                figure.add_scatter(
                    x=all_time[future],
                    y=consumption[variant_name][series].values[future] * megatonnes,
                    name=variant_name,
                    legendgroup=variant_name,
                    showlegend=region_idx == 0,
                    line=dict(width=1.5, **line),
                    row=row,
                    col=col,
                )
        figure.add_vline(x=all_historic_time[-1], line=dict(color="#c3c2b7", width=1))
        figure.update_xaxes(range=[1990, 2060])
        figure.update_yaxes(rangemode="tozero")
        figure.update_layout(
            title=f"{model_name.capitalize()} consumption [Mt/yr], {sector} "
            f"(approaching time {approaching_time} years)",
            height=280 * n_rows + 120,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
            margin=dict(t=140),
        )
        return figure

    region_figures = {}
    for _model_name in (name.value for name in ModelNames):
        with mo.status.spinner(title=f"Comparing the windows for {_model_name}..."):
            _variants = window_variants(_model_name)
        region_figures[_model_name] = mo.vstack(
            [
                region_figure(_variants, _model_name, _sector)
                for _sector in _variants["model"].dims[_variants["sector_letter"]].items
            ]
        )
    mo.ui.tabs(region_figures)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
