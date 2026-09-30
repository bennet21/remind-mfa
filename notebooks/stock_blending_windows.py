# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.24.2"
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

    The chart compares the consumption from the future MFA without blending, with the windows the
    model chooses on the stock, with the windows chosen on consumption, and with windows chosen by
    hand.
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
    consumption_stock_name, consumption_slice = {
        "steel": ("in_use", {}),
        "plastics": ("in_use_dsm", {}),
        "cement": ("in_use", {"k": "cement"}),
    }[model_name]
    return (
        consumption_slice,
        consumption_stock_name,
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
def _(
    consumption_slice,
    consumption_stock_name,
    fd,
    model,
    sector_letter,
    stock_extrapolation,
):
    def consumption_for(stocks_pc: fd.FlodymArray) -> fd.FlodymArray:
        """Consumption of the future MFA driven by the given normalized stocks per capita."""
        # denormalized as in StockExtrapolation.extrapolate and CommonModel.get_long_term_stock
        stock_projection = stocks_pc * stock_extrapolation.pop * model.sector_specific_sat_level
        future_mfa = model.make_mfa(historic=False)
        future_mfa.compute(stock_projection, model.historic_mfa.trade_set)
        return consumption_of(future_mfa)

    def consumption_of(future_mfa) -> fd.FlodymArray:
        """Inflow into the in-use stock of the future MFA, by time, region and sector."""
        inflow = future_mfa.stocks[consumption_stock_name].inflow
        if consumption_slice:
            inflow = inflow[consumption_slice]
        return inflow.sum_to(("t", "r", sector_letter))

    # the regression right after the last historic year,
    # cf. transition_smoothing "none" in StockExtrapolation.smooth_transition
    unblended_stocks_pc = stock_extrapolation.fitted_regression.copy()
    n_historic_years = model.dims["h"].len
    unblended_stocks_pc.values[:n_historic_years] = stock_extrapolation.historic_stocks_pc.values
    unblended_consumption = consumption_for(unblended_stocks_pc)
    model_window_consumption = consumption_of(model.future_mfa)
    return consumption_for, model_window_consumption, unblended_consumption


@app.cell
def _(blender, consumption_for, fd, model, np, series_slice, stock_extrapolation):
    # the blend of the model, cf. StockExtrapolation.smooth_transition
    approaching_time = 50
    np.testing.assert_allclose(
        blender.blend(approaching_time)[:, 0], stock_extrapolation.stocks_pc[series_slice].values
    )

    def consumption_for_windows(level_window: int, slope_window: int) -> np.ndarray:
        """Consumption of the future MFA when the selected series is blended with the stock
        windows corresponding to the given consumption windows."""
        max_stock_window = len(blender.historical) - 1
        blended = blender.blend(
            approaching_time,
            velocity_window=min(level_window + 1, max_stock_window),
            acceleration_window=min(slope_window + 1, max_stock_window),
        )[:, 0]
        stocks_pc = stock_extrapolation.stocks_pc.copy()
        stocks_pc[series_slice] = fd.FlodymArray(dims=model.dims["t",], values=blended)
        return consumption_for(stocks_pc)[series_slice].values

    return approaching_time, consumption_for_windows


@app.cell
def _(consumption_for_windows, level_selection, slope_selection):
    optimal_window_consumption = consumption_for_windows(
        int(level_selection.selected_window), int(slope_selection.selected_window)
    )
    return (optimal_window_consumption,)


@app.cell
def _(consumption_for_windows, level_window_slider, slope_window_slider):
    chosen_window_consumption = consumption_for_windows(
        level_window_slider.value, slope_window_slider.value
    )
    return (chosen_window_consumption,)


@app.cell
def _(
    level_selection,
    level_window_slider,
    mo,
    model_level_window,
    model_slope_window,
    slope_selection,
    slope_window_slider,
):
    megatonnes = 1e-6

    def describe(name: str, unit: str, selection, model_window: int, chosen_window: int) -> str:
        optimal_window = int(selection.selected_window)
        estimates = [
            selection.derivative_at(window) * megatonnes
            for window in (model_window, optimal_window, chosen_window)
        ]
        return (
            f"| {name} [{unit}] | {model_window} ({estimates[0]:.4g}) "
            f"| {optimal_window} ({estimates[1]:.4g}) | {chosen_window} ({estimates[2]:.4g}) |"
        )

    mo.md(
        "Windows in years, with the estimate at the last historic year in brackets.\n\n"
        "| | model (on stock) | optimal on consumption | chosen |\n"
        "|---|---|---|---|\n"
        + describe(
            "level", "Mt/yr", level_selection, model_level_window, level_window_slider.value
        )
        + "\n"
        + describe(
            "slope", "Mt/yr²", slope_selection, model_slope_window, slope_window_slider.value
        )
    )
    return (megatonnes,)


@app.cell
def _(
    approaching_time,
    chosen_window_consumption,
    consumption_curve,
    go,
    historic_time,
    megatonnes,
    model_name,
    model_window_consumption,
    optimal_window_consumption,
    region,
    sector,
    series_slice,
    time,
    unblended_consumption,
):
    future_years = slice(len(historic_time) - 1, None)
    consumption_figure = go.Figure()
    consumption_figure.add_scatter(
        x=historic_time,
        y=consumption_curve * megatonnes,
        name="Historic",
        line=dict(color="#0b0b0b", width=2),
    )
    consumption_variants = [
        (
            "Unblended prediction",
            unblended_consumption[series_slice].values,
            dict(color="#8a8984", dash="dash"),
        ),
        (
            "Windows of the model (on stock)",
            model_window_consumption[series_slice].values,
            dict(color="#1baf7a"),
        ),
        ("Optimal windows on consumption", optimal_window_consumption, dict(color="#2a78d6")),
        ("Chosen windows", chosen_window_consumption, dict(color="#eb6834", dash="dot")),
    ]
    for _name, _consumption, _line in consumption_variants:
        consumption_figure.add_scatter(
            x=time[future_years],
            y=_consumption[future_years] * megatonnes,
            name=_name,
            line=dict(width=2, **_line),
        )
    consumption_figure.add_vline(x=historic_time[-1], line=dict(color="#c3c2b7", width=1))
    consumption_figure.update_layout(
        title=f"{model_name.capitalize()} consumption, {region}, {sector} "
        f"(approaching time {approaching_time} years)",
        xaxis=dict(title="Year", range=[1990, 2060]),
        yaxis=dict(title="Consumption [Mt/yr]"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    consumption_figure
    return


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


if __name__ == "__main__":
    app.run()
