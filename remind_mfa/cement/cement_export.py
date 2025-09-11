from plotly import colors as plc
import flodym as fd
from typing import TYPE_CHECKING
import numpy as np
import logging
import pyam

from remind_mfa.common.common_export import CommonDataExporter
from remind_mfa.common.common_cfg import CementVisualizationCfg
from remind_mfa.cement.cement_mfa_system_future import StockDrivenCementMFASystem

if TYPE_CHECKING:
    from remind_mfa.cement.cement_model import CementModel


class CementDataExporter(CommonDataExporter):
    cfg: CementVisualizationCfg

    _display_names: dict = {
        "sysenv": "System environment",
        "prod_clinker": "Production: Clinker",
        "prod_cement": "Production: Cement",
        "prod_product": "Production: Product",
        "use": "Use phase",
        "eol": "End of life",
    }

    def visualize_results(self, model: "CementModel"):
        mfa: StockDrivenCementMFASystem = model.future_mfa
        if self.cfg.consumption["do_visualize"]:
            self.visualize_consumption(mfa=mfa)
        if self.cfg.prod_clinker["do_visualize"]:
            self.visualize_prod_clinker(mfa=mfa)
        if self.cfg.prod_cement["do_visualize"]:
            self.visualize_prod_cement(mfa=mfa, regional=False)
            self.visualize_prod_cement(mfa=mfa, regional=True)
        if self.cfg.prod_product["do_visualize"]:
            self.visualize_prod_product(mfa=mfa)
        if self.cfg.use_stock["do_visualize"]:
            self.visualize_use_stock(mfa=mfa, subplots_by_stock_type=False)
            # self.visualize_use_stock(mfa=mfa, subplots_by_stock_type=True)
        if self.cfg.eol_stock["do_visualize"]:
            self.visualize_eol_stock(mfa=mfa)
        if self.cfg.sankey["do_visualize"]:
            self.visualize_sankey(mfa=mfa)
        if self.cfg.extrapolation["do_visualize"]:
            # self.visualize_extrapolation(model=model, show_extrapolation=False, show_future=False)
            # self.visualize_extrapolation(model=model, show_future=False)
            self.visualize_extrapolation(model=model)
        if self.cfg.carbonation["do_visualize"]:
            if not mfa.carbon_flow:
                logging.warning("Carbonation visualization requested, but carbonation calculation not activated.")
            else:
                self.visualize_carbonation(mfa=mfa)
        self.write_iamc(mfa)
        self.stop_and_show()

    def visualize_production(
        self, mfa: fd.MFASystem, production: fd.Flow, name: str, regional: bool = False
    ):

        x_array = None
        # intra_line_dim = "Time"
        # line_label = f"{name} Production"
        x_label = "Year"
        y_label = "Production [t]"
        linecolor_dim = None
        plot_letters = ["t"]

        if regional:
            subplot_dim = "Region"
            title = f"Regional {name} Production"
            regional_tag = "_regional"
            plot_letters += ["r"]
        else:
            subplot_dim = None
            regional_tag = "_global"
            title = f"Global {name} Production"
            
        other_letters = tuple(letter for letter in production.dims.letters if letter not in plot_letters)
        production = production.sum_over(other_letters)

        fig, ap_production = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=production,
            subplot_dim=subplot_dim,
            x_array=x_array,
            linecolor_dim=linecolor_dim,
            x_label=x_label,
            y_label=y_label,
            title=title,
            line_label="Production",
        )

        self.plot_and_save_figure(
            ap_production, f"{name}_production{regional_tag}.png", do_plot=False
        )

    def visualize_prod_clinker(self, mfa: fd.MFASystem):
        production = mfa.flows["prod_clinker => prod_cement"]
        self.visualize_production(mfa=mfa, production=production, name="Clinker")

    def visualize_prod_cement(self, mfa: fd.MFASystem, regional: bool = False):
        production = mfa.flows["prod_cement => prod_product"]
        self.visualize_production(mfa=mfa, production=production, name="Cement", regional=regional)

    def visualize_prod_product(self, mfa: fd.MFASystem):
        production = mfa.flows["prod_product => use"].sum_over("s")
        self.visualize_production(mfa=mfa, production=production, name="Product")

    def visualize_consumption(self, mfa: fd.MFASystem):
        # TODO find better way to implement cement_ratio
        cement_ratio = mfa.parameters["product_cement_content"] / mfa.parameters["product_density"]
        consumption = mfa.stocks["in_use"].inflow * cement_ratio
        plot_letters = ["t", "r", "s"]
        other_letters = tuple(letter for letter in consumption.dims.letters if letter not in plot_letters)
        consumption = consumption.sum_over(other_letters)
        sector_dim = consumption.dims.index("s")
        consumption = consumption.apply(np.cumsum, kwargs={"axis": sector_dim})
        ap = self.plotter_class(
            array=consumption,
            intra_line_dim="Time",
            subplot_dim="Region",
            linecolor_dim="Stock Type",
            chart_type="area",
            display_names=self._display_names,
            title="Cement Consumption",
        )
        fig = ap.plot()
        self.plot_and_save_figure(ap, "cement_consumption.png", do_plot=False)

    def visualize_eol_stock(self, mfa: fd.MFASystem):
        # over_gdp = self.cfg.eol_stock["over_gdp"]
        # per_capita = self.cfg.eol_stock["per_capita"]
        # stock = mfa.stocks["eol"].stock

        # self.visualize_stock(mfa, stock, over_gdp, per_capita, "EOL")
        # TODO EOL visualization does not make sense by stock type
        pass

    def visualize_use_stock(self, mfa: fd.MFASystem, subplots_by_stock_type=False):
        subplot_dim = "Stock Type" if subplots_by_stock_type else None
        cement_ratio = mfa.parameters["product_cement_content"] / mfa.parameters["product_density"]
        stock = mfa.stocks["in_use"].stock * cement_ratio
        super().visualize_use_stock(mfa, stock=stock, subplot_dim=subplot_dim)

    def visualize_stock(self, mfa: fd.MFASystem, stock, over_gdp, per_capita, name):
        population = mfa.parameters["population"]
        x_array = None

        pc_str = " pC" if per_capita else ""
        x_label = "Year"
        y_label = f"{name} Stock{pc_str}[t]"
        title = f"{name} stocks{pc_str}"
        if over_gdp:
            title = title + f" over GDP{pc_str}"

        if over_gdp:
            x_array = mfa.parameters["gdppc"]
            x_label = f"GDP/PPP{pc_str}[2005 USD]"

        # self.visualize_regional_stock(
        #     stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        # )
        self.visualize_global_stock(
            stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        )

    def visualize_global_stock(
        self, stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
    ):
        if over_gdp:
            x_array = x_array * population
            x_array = x_array.sum_over("r")
            if per_capita:
                # get global GDP per capita
                x_array = x_array / population.sum_over("r")

        self.visualize_global_stock_by_type(
            stock, x_array, population, x_label, y_label, title, per_capita
        )
        # self.visualize_global_stock_by_region(stock, x_array, x_label, y_label, title, per_capita)

    def visualize_global_stock_by_type(
        self, stock, x_array, population, x_label, y_label, title, per_capita
    ):
        plot_letters = ["t", "s"]
        stock = stock / population.sum_over("r") if per_capita else stock
        other_letters = tuple(letter for letter in stock.dims.letters if letter not in plot_letters)
        stock = stock.sum_over(other_letters)
        ap_stock = self.plotter_class(
            array=stock,
            intra_line_dim="Time",
            linecolor_dim="Stock Type",
            display_names=self._display_names,
            x_array=x_array,
            xlabel=x_label,
            ylabel=y_label,
            title=f"{title} (global by stock type)",
            area=True,
        )

        self.plot_and_save_figure(ap_stock, "use_stocks_global_by_type.png")

    def visualize_extrapolation(self, model: "CementModel", show_extrapolation: bool = True, show_future: bool = True):
        mfa = model.future_mfa
        per_capita = True  # TODO see where this shold go
        subplot_dim = "Region"
        cement_ratio = mfa.parameters["product_cement_content"] / mfa.parameters["product_density"]
        stock = mfa.stocks["in_use"].stock * cement_ratio
        population = mfa.parameters["population"]
        x_array = None

        pc_str = "pC" if per_capita else ""
        x_label = "Year"
        y_label = f"Stock{pc_str} [t]"
        title = f"Stock Extrapolation: Historic and Projected vs Pure Prediction"
        if self.cfg.use_stock["over_gdp"]:
            title = title + f" over GDP{pc_str}"
            x_label = f"GDP/PPP{pc_str} [2005 USD]"
            x_array = mfa.parameters["gdppc"]
            if not per_capita:
                x_array = x_array * population

        if subplot_dim is None:
            dimlist = ["t"]
        else:
            subplot_dimletter = next(
                dimlist.letter for dimlist in mfa.dims.dim_list if dimlist.name == subplot_dim
            )
            dimlist = ["t", subplot_dimletter]

        other_dimletters = tuple(letter for letter in stock.dims.letters if letter not in dimlist)
        stock = stock.sum_over(other_dimletters)

        if per_capita:
            stock = stock / population

        fig, ap = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=stock,
            subplot_dim=subplot_dim,
            x_array=x_array,
            x_label=x_label,
            y_label=y_label,
            title=title,
            line_label="Historic + Modelled Future",
            future_stock=show_future
        )

        # extrapolation
        if show_extrapolation:
            ap = self.plotter_class(
                array=model.stock_handler.pure_prediction,
                intra_line_dim="Time",
                subplot_dim=subplot_dim,
                x_array=x_array,
                title=title,
                fig=fig,
                line_type="dot",
                line_label="Pure Extrapolation",
            )
            fig = ap.plot()

        extrapolation_name = "_extrapolation" if show_extrapolation else ""
        future_name = "_projection" if show_future else "_historic"
        self.plot_and_save_figure(
            ap,
            f"cement_stocks{extrapolation_name}{future_name}.png",
            do_plot=False,
        )

    def visualize_carbonation(self, mfa: fd.MFASystem):
        annual_uptake = mfa.stocks["carbonated_co2"].inflow
        cumulative_uptake = mfa.stocks["carbonated_co2"].stock
        linecolor_dimletter = "Carbonation Location"
        plot_letters = ["t", "c"]
        other_dimletters = tuple(letter for letter in annual_uptake.dims.letters if letter not in plot_letters)
        annual_uptake = annual_uptake.sum_over(other_dimletters)
        annual_uptake = annual_uptake.apply(np.cumsum, kwargs={"axis": annual_uptake.dims.index("c")})

        ap = self.plotter_class(
            array=annual_uptake,
            intra_line_dim="Time",
            linecolor_dim=linecolor_dimletter,
            chart_type="area",
            display_names=self._display_names,
            x_label="Year",
            y_label="Annual Co2 Uptake [t]",
            title="Co2 Uptake from Carbonation",
        )
        fig = ap.plot()

        self.plot_and_save_figure(
            ap, "cement_carbonation_annual_uptake.png", do_plot=False
        )

    def write_iamc(self, future_mfa: fd.MFASystem):

        model = "REMIND 3.0"
        scenario = "SSP2_NPi"
        constants = {"model": model, "scenario": scenario}

        cement_ratio = future_mfa.parameters["product_cement_content"] / future_mfa.parameters["product_density"]
        reported_dims = ("t", "r", "s")

        # production (same as demand + losses as no trade is considered yet)
        # cement
        cement_prod_by_stock_type = future_mfa.flows["prod_cement => prod_product"] + future_mfa.flows["prod_cement => sysenv"]
        other_dimletters = tuple(letter for letter in cement_prod_by_stock_type.dims.letters if letter not in reported_dims)
        cement_prod_by_stock_type = cement_prod_by_stock_type.sum_over(other_dimletters).sum_over("s")
        prod_df = self.to_iamc_df(cement_prod_by_stock_type)
        prod_df["variable"] = "Production|Non-Metallic Minerals|Cement"
        cement_prod_idf = pyam.IamDataFrame(
            prod_df,
            unit="t/yr",
            **constants,
        )
        cement_prod_idf.aggregate(
            variable="Production|Non-Metallic Minerals|Cement",
            append=True,
        )

        # clinker
        clinker_prod_by_stock_type = future_mfa.flows["prod_clinker => prod_cement"] + future_mfa.flows["prod_clinker => sysenv"]
        other_dimletters = tuple(letter for letter in clinker_prod_by_stock_type.dims.letters if letter not in reported_dims)
        clinker_prod_by_stock_type = clinker_prod_by_stock_type.sum_over(other_dimletters).sum_over("s")
        clinker_prod_df = self.to_iamc_df(clinker_prod_by_stock_type)
        clinker_prod_df["variable"] = "Production|Non-Metallic Minerals|Cement Clinker"
        clinker_prod_idf = pyam.IamDataFrame(
            clinker_prod_df,
            unit="t/yr",
            **constants,
        )
        clinker_prod_idf.aggregate(
            variable="Production|Non-Metallic Minerals|Cement Clinker",
            append=True,
        )
    
        # demand
        cement_demand_by_stock_type = future_mfa.flows["prod_cement => prod_product"]
        other_dimletters = tuple(letter for letter in cement_demand_by_stock_type.dims.letters if letter not in reported_dims)
        cement_demand_by_stock_type = cement_demand_by_stock_type.sum_over(other_dimletters)
        demand_df = self.to_iamc_df(cement_demand_by_stock_type)
        demand_df["variable"] = "Material Demand|Non-Metallic Minerals|Cement|" + demand_df["Stock Type"]
        demand_df = demand_df.drop(columns=["Stock Type"])
        demand_idf = pyam.IamDataFrame(
            demand_df,
            unit="t/yr",
            **constants,
        )
        demand_idf.aggregate(
            variable="Material Demand|Non-Metallic Minerals|Cement",
            append=True,
        )

        # stocks
        cement_stock_by_stock_type = future_mfa.stocks["in_use"].stock * cement_ratio
        other_dimletters = tuple(letter for letter in cement_stock_by_stock_type.dims.letters if letter not in reported_dims)
        cement_stock_by_stock_type = cement_stock_by_stock_type.sum_over(other_dimletters)
        stock_df = self.to_iamc_df(cement_stock_by_stock_type)
        stock_df["variable"] = "Material Stock|Non-Metallic Minerals|Cement|" + stock_df["Stock Type"]
        stock_df = stock_df.drop(columns=["Stock Type"])
        stock_idf = pyam.IamDataFrame(
            stock_df,
            unit="t",
            **constants,
        )
        stock_idf.aggregate(
            variable="Material Stock|Non-Metallic Minerals|Cement",
            append=True,
        )

        # scrap
        cement_scrap_by_stock_type = future_mfa.stocks["in_use"].outflow * cement_ratio
        other_dimletters = tuple(letter for letter in cement_scrap_by_stock_type.dims.letters if letter not in reported_dims)
        cement_scrap_by_stock_type = cement_scrap_by_stock_type.sum_over(other_dimletters)
        scrap_df = self.to_iamc_df(cement_scrap_by_stock_type)
        scrap_df["variable"] = "Scrap|Non-Metallic Minerals|Cement|" + scrap_df["Stock Type"]
        scrap_df = scrap_df.drop(columns=["Stock Type"])
        scrap_idf = pyam.IamDataFrame(
            scrap_df,
            unit="t/yr",
            **constants,
        )
        scrap_idf.aggregate(
            variable="Scrap|Non-Metallic Minerals|Cement",
            append=True,
        )

        idf = pyam.concat([
            cement_prod_idf,
            clinker_prod_idf,
            demand_idf,
            stock_idf,
            scrap_idf,
        ])
        idf.aggregate_region(
            variable=idf.variable,
            region="World",
            append=True,
        )
        idf.convert_unit(current="t/yr", to="Mt/yr", inplace=True)
        idf.convert_unit(current="t", to="Mt", inplace=True)

        idf.to_excel(self.export_path(f"output_iamc.xlsx"))


    @staticmethod
    def to_iamc_df(array: fd.FlodymArray):
        time_items = list(range(2025, 2101)) # TODO: more flexible
        time_out = fd.Dimension(name="Time Out", letter="O", items=time_items)
        df = array[{"t": time_out}].to_df(dim_to_columns="Time Out", index=False)
        df = df.rename(columns={"Region": "region"})
        return df
