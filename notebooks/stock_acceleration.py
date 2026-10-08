import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Acceleration of in-use stocks

    Runs a model and plots the first and second time derivative of the in-use stock for all
    regions, historic and future. The stock is taken from the stock extrapolation step, either
    per capita or normalised by the saturation level (the quantity the critically damped
    blender operates on).

    Derivatives are central differences with a lag of $h$ years:
    $v(t) = \frac{y(t+h) - y(t-h)}{2h}$, $a(t) = \frac{v(t+h) - v(t-h)}{2h}$.
    The relative view shows the growth rate $g = v / y$ and its change $\mathrm{d}g/\mathrm{d}t$.
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
    import logging
    import os

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from remind_mfa.common.config_loader import load_config
    from remind_mfa.common.helpers import ModelNames, init_model

    return ModelNames, init_model, load_config, logging, mo, np, os, pd, px


@app.cell
def _(mo, os):
    repo_root = mo.notebook_dir().parent
    if not (repo_root / "pyproject.toml").exists():
        raise FileNotFoundError("The notebook must be located in the `notebooks` directory.")
    # config paths such as `data_in` are relative to the repository root
    os.chdir(repo_root)
    return (repo_root,)


@app.cell
def _(ModelNames, mo):
    model_selector = mo.ui.dropdown(
        options=[model.value for model in ModelNames],
        value=ModelNames.STEEL.value,
        label="Model",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(ModelNames, init_model, load_config, logging, mo, repo_root):
    @mo.cache
    def run_model(model_name: str):
        """Run a model with the default config, without export and visualization.

        Args:
            model_name: Name of the model, e.g. `"steel"`.

        Returns:
            The model after `run()`, holding the stock extrapolation in `stock_handler`.
        """
        logging.getLogger().setLevel(logging.WARNING)
        # the default config directory is resolved from the working directory at import time
        model_config = load_config(
            ["default"], ModelNames(model_name), config_dir=repo_root / "config"
        )
        model = init_model(cfg=model_config)
        model.run()
        return model

    return (run_model,)


@app.cell
def _(mo, model_selector, run_model):
    with mo.status.spinner(title=f"Running {model_selector.value} model..."):
        model = run_model(model_selector.value)
    return (model,)


@app.cell
def _(mo):
    basis_selector = mo.ui.radio(
        options=["per capita", "normalised (blender input)"],
        value="per capita",
        label="Stock basis",
    )
    quantity_selector = mo.ui.radio(
        options=["acceleration", "slope", "stock", "change of growth rate", "growth rate"],
        value="acceleration",
        label="Quantity",
    )
    lag_slider = mo.ui.slider(start=1, stop=10, value=2, label="Lag h (years)", show_value=True)
    show_difference = mo.ui.checkbox(
        value=True, label="Show difference blended − prediction (dashed)"
    )
    return basis_selector, lag_slider, quantity_selector, show_difference


@app.cell
def _(mo, model):
    # the stock has dimensions (t, r, end use); the end-use letter differs between branches
    end_use_letter = model.stock_handler.stocks_pc.dims.letters[-1]
    end_use_selector = mo.ui.multiselect(
        options=list(model.dims[end_use_letter].items),
        value=list(model.dims[end_use_letter].items),
        label="End uses",
    )
    return end_use_letter, end_use_selector


@app.cell
def _(
    basis_selector,
    end_use_selector,
    lag_slider,
    mo,
    quantity_selector,
    show_difference,
):
    mo.hstack(
        [
            basis_selector,
            quantity_selector,
            mo.vstack([lag_slider, show_difference, end_use_selector]),
        ],
        justify="start",
        gap=3,
    )
    return


@app.cell
def _(basis_selector, model):
    stock_handler = model.stock_handler
    blended_stock = stock_handler.stocks_pc
    prediction_stock = stock_handler.fitted_regression
    if basis_selector.value == "per capita":
        # undo the normalisation by the saturation level applied in `get_long_term_stock`
        saturation_level = model.sector_specific_sat_level.cast_to(blended_stock.dims)
        blended_stock = blended_stock * saturation_level
        prediction_stock = prediction_stock * saturation_level
    last_historic_year = model.dims["h"].items[-1]
    return blended_stock, last_historic_year, prediction_stock


@app.cell
def _(np):
    def central_difference(values: np.ndarray, lag: int) -> np.ndarray:
        """Central difference along the first axis with a lag of `lag` time steps of one year.

        Args:
            values: Array with time as the first axis.
            lag: Number of years between the centre and each of the two evaluation points.

        Returns:
            Array of the same shape, NaN where the stencil exceeds the time range.
        """
        derivative = np.full(values.shape, np.nan)
        derivative[lag:-lag] = (values[2 * lag :] - values[: -2 * lag]) / (2 * lag)
        return derivative

    def derive_quantity(values: np.ndarray, quantity: str, lag: int) -> np.ndarray:
        """Compute the selected quantity from a stock array with time as the first axis.

        Args:
            values: Stock values, time as the first axis with yearly steps.
            quantity: One of the options of the quantity selector.
            lag: Lag of the central differences in years.

        Returns:
            Array of the same shape as `values`.
        """
        slope = central_difference(values, lag)
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_rate = np.where(values > 0, slope / values, np.nan)
        match quantity:
            case "stock":
                return values
            case "slope":
                return slope
            case "acceleration":
                return central_difference(slope, lag)
            case "growth rate":
                return growth_rate
            case "change of growth rate":
                return central_difference(growth_rate, lag)
            case _:
                raise ValueError(f"Unknown quantity {quantity!r}")

    return (derive_quantity,)


@app.cell
def _(
    blended_stock,
    derive_quantity,
    end_use_letter,
    lag_slider,
    model,
    np,
    pd,
    prediction_stock,
    quantity_selector,
):
    def to_long_frame(values: np.ndarray, series_name: str) -> pd.DataFrame:
        """Convert a (t, r, end use) array into a long frame for plotting.

        Args:
            values: Array with dimensions ordered as in `blended_stock`.
            series_name: Label of the series, e.g. `"blended"`.

        Returns:
            Frame with columns `Time`, `Region`, `End use`, `value` and `Series`.
        """
        index = pd.MultiIndex.from_product(
            [model.dims[letter].items for letter in blended_stock.dims.letters],
            names=[model.dims[letter].name for letter in blended_stock.dims.letters],
        )
        frame = pd.DataFrame({"value": values.ravel()}, index=index).reset_index()
        frame = frame.rename(
            columns={
                model.dims["t"].name: "Time",
                model.dims["r"].name: "Region",
                model.dims[end_use_letter].name: "End use",
            }
        )
        frame["Series"] = series_name
        return frame

    if blended_stock.dims.letters != ("t", "r", end_use_letter):
        raise ValueError(f"Unexpected stock dimensions {blended_stock.dims.letters}")

    blended_quantity = derive_quantity(
        blended_stock.values, quantity_selector.value, lag_slider.value
    )
    prediction_quantity = derive_quantity(
        prediction_stock.values, quantity_selector.value, lag_slider.value
    )
    plot_frame = pd.concat(
        [
            to_long_frame(blended_quantity, "blended"),
            to_long_frame(blended_quantity - prediction_quantity, "difference"),
        ],
        ignore_index=True,
    )
    return (plot_frame,)


@app.cell
def _(
    basis_selector,
    end_use_selector,
    last_historic_year,
    model_selector,
    plot_frame,
    px,
    quantity_selector,
    show_difference,
):
    selected_series = ["blended", "difference"] if show_difference.value else ["blended"]
    selected_frame = plot_frame[
        plot_frame["End use"].isin(end_use_selector.value)
        & plot_frame["Series"].isin(selected_series)
    ]
    figure = px.line(
        selected_frame,
        x="Time",
        y="value",
        color="End use",
        line_dash="Series",
        line_dash_map={"blended": "solid", "difference": "dash"},
        facet_col="Region",
        facet_col_wrap=4,
        facet_row_spacing=0.04,
        height=1100,
        title=(
            f"{model_selector.value}: {quantity_selector.value} of in-use stock "
            f"({basis_selector.value})"
        ),
    )
    figure.update_yaxes(showticklabels=True, title_text="")
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    figure.add_vline(x=last_historic_year, line_dash="dot", line_color="grey")
    figure.add_hline(y=0, line_color="lightgrey")
    figure
    return


if __name__ == "__main__":
    app.run()
