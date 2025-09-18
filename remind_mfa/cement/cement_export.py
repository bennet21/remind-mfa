from plotly import colors as plc
import numpy as np
import flodym as fd
from typing import TYPE_CHECKING
import numpy as np
import logging

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
                logging.warning(
                    "Carbonation visualization requested, but carbonation calculation not activated."
                )
            else:
                self.visualize_carbonation(mfa=mfa)

        if self.cfg.sd["do_visualize"]:
            # self.visualize_sd(model=model, material="concrete")
            self.visualize_sd(model=model, material="cement")
            # self.visualize_sd(model=model, material="concrete", regional=False)
            self.visualize_sd(model=model, material="cement", regional=False)
            self.visualize_sd(model=model, material="cement", regional=True, per_capita=False)
            self.visualize_sd(model=model, material="cement", regional=False, per_capita=False)
            self.visualize_top_vs_bottom(model=model)
        self.stop_and_show()

    def visualize_production(
        self, mfa: fd.MFASystem, production: fd.Flow, name: str, regional: bool = False, stacked: bool = False
    ):
        if regional and stacked:
            logging.warning("Cannot do stacked regional production plots, switching to non-stacked.")
            stacked = False

        x_array = None
        x_label = "Year"
        y_label = "Production [t]"
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
            if stacked:
                plot_letters += ["r"]

        other_letters = tuple(
            letter for letter in production.dims.letters if letter not in plot_letters
        )
        production = production.sum_over(other_letters)

        if stacked:
            region_idx = production.dims.index("r")
            production = production.apply(np.cumsum, kwargs={"axis": region_idx})
            ap_production = self.plotter_class(
                array=production,
                intra_line_dim="Time",
                linecolor_dim="Region",
                chart_type="area",
                display_names=self._display_names,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )
            fig = ap_production.plot()
        else:
            linecolor_dim = None
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
        self.visualize_production(mfa=mfa, production=production, name="Cement", regional=regional, stacked=True)

    def visualize_prod_product(self, mfa: fd.MFASystem):
        production = mfa.flows["prod_product => use"].sum_over("s")
        self.visualize_production(mfa=mfa, production=production, name="Product")

    def visualize_consumption(self, mfa: fd.MFASystem):
        cement_ratio = mfa.parameters["product_cement_content"] / mfa.parameters["product_density"]
        consumption = mfa.stocks["in_use"].inflow * cement_ratio
        plot_letters = ["t", "r", "s"]
        other_letters = tuple(
            letter for letter in consumption.dims.letters if letter not in plot_letters
        )
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

    def visualize_extrapolation(
        self, model: "CementModel", show_extrapolation: bool = True, show_future: bool = True
    ):
        mfa = model.future_mfa
        per_capita = self.cfg.use_stock["per_capita"]
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
        extrapolation = model.stock_handler.pure_prediction

        if per_capita:
            stock = stock / population
        else:
            extrapolation = extrapolation * population

        fig, ap = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=stock,
            subplot_dim=subplot_dim,
            x_array=x_array,
            x_label=x_label,
            y_label=y_label,
            title=title,
            line_label="Historic + Modelled Future",
            future_stock=show_future,
        )

        if show_extrapolation:
            ap = self.plotter_class(
                array=extrapolation,
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
        other_dimletters = tuple(
            letter for letter in annual_uptake.dims.letters if letter not in plot_letters
        )
        annual_uptake = annual_uptake.sum_over(other_dimletters)
        annual_uptake = annual_uptake.apply(
            np.cumsum, kwargs={"axis": annual_uptake.dims.index("c")}
        )

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

        self.plot_and_save_figure(ap, "cement_carbonation_annual_uptake.png", do_plot=False)

    def calculate_sd_stock(self, model: "CementModel", material="concrete") -> fd.FlodymArray:
        prm = model.parameters
        cement_ratio = model.future_mfa.parameters["product_cement_content"] / model.future_mfa.parameters["product_density"]
        # customize split for SD: no knowledge about mortar use
        pms = prm["product_material_split"]
        pms[{"m": "mortar"}][...] = 0 # set mortar to zero
        product_application_material_split = prm["product_application_split"] * pms  * prm["product_material_application_transform"]
        
        bf = prm["buildings_floorspace"]
        bf = fd.FlodymArray(dims=model.dims[("t", "r", "s", "b", "f", "m", "a")])
        big_bf = prm["buildings_floorspace"] * prm["building_split"] * product_application_material_split
        bf[{"s": "Com", "f": "Com"}][...] = big_bf[{"s": "Com", "f": "Com"}]
        bf[{"s": "Res", "f": "RS"}][...] = big_bf[{"s": "Res", "f": "RS"}]
        bf[{"s": "Res", "f": "RM"}][...] = big_bf[{"s": "Res", "f": "RM"}]
        
        stock =  bf * prm["concrete_building_mi"]
        if material == "cement":
            stock = stock * cement_ratio

        return stock
    
    def visualize_sd(self, model: "CementModel", material: str = "concrete", regional: bool = True, per_capita: bool = True):

        mfa = model.future_mfa
        cement_ratio = mfa.parameters["product_cement_content"] / mfa.parameters["product_density"]
        subplot_dim = "Region"
        linecolor_dim = "Stock Type"
        stock = mfa.stocks["in_use"].stock
        if material == "cement":
            stock = stock * cement_ratio
        stock_sd = self.calculate_sd_stock(model, material=material)
        population = mfa.parameters["population"]

        if not regional:
            subplot_dim = None
            stock = stock.sum_over("r")
            stock_sd = stock_sd.sum_over("r")
            population = population.sum_over("r")

        x_array = None

        pc_str = "pC" if per_capita else ""
        x_label = "Year"
        y_label = f"{material.capitalize()} Stock{pc_str} [t]"
        title = f"{material.capitalize()} Stock Comparison: Buttom-up SD vs Top-down Extrapolation"
        if self.cfg.sd["over_gdp"]:
            title = title + f" over GDP{pc_str}"
            x_label = f"GDP/PPP{pc_str} [2005 USD]"
            x_array = mfa.parameters["gdppc"]
            if not per_capita:
                x_array = x_array * population

        if subplot_dim is None:
            dimlist = ["t", "s"]
        else:
            subplot_dimletter = next(
                dimlist.letter for dimlist in mfa.dims.dim_list if dimlist.name == subplot_dim
            )
            dimlist = ["t", "s", subplot_dimletter]
        
        if per_capita:
            stock = stock / population
            stock_sd = stock_sd / population

        other_dimletters = tuple(letter for letter in stock.dims.letters if letter not in dimlist)
        stock = stock.sum_over(other_dimletters)

        # service demand stock
        other_dimletters_sd = tuple(letter for letter in stock_sd.dims.letters if letter not in dimlist)
        stock_sd = stock_sd.sum_over(other_dimletters_sd)

        fig, ap_final_stock = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=stock,
            linecolor_dim=linecolor_dim,
            subplot_dim=subplot_dim,
            x_array=x_array,
            x_label=x_label,
            y_label=y_label,
            title=title,
        )

        # SD
        ap_pure_prediction = self.plotter_class(
            array=stock_sd,
            intra_line_dim="Time",
            subplot_dim=subplot_dim,
            linecolor_dim=linecolor_dim,
            x_array=x_array,
            title=title,
            fig=fig,
            line_type="dot",
        )
        fig = ap_pure_prediction.plot()

        self.plot_and_save_figure(
            ap_pure_prediction,
            f"stocks_extrapolation.png",
            do_plot=False,
        )

    def visualize_top_vs_bottom(self, model: "CementModel", material="concrete"):
        mfa = model.future_mfa

        stock_sd = self.calculate_sd_stock(model, material=material).sum_over(("s", "b", "f", "m", "a"))
        stock = mfa.stocks["in_use"].stock.sum_over(("s", "m", "a"))
        gdppc = mfa.parameters["gdppc"]

        cut_time = fd.Dimension(name="CutTime", letter="p", items=np.arange(1999, 2024))
        cut_stock_sd = stock_sd[{"t": cut_time}]
        # TODO only compare material "concrete"
        cut_stock = stock[{"t": cut_time}]
        cut_gdppc = gdppc[{"t": cut_time}]

        ratio = cut_stock_sd / cut_stock
        # ratio over gdppc
        ap_ratio = self.plotter_class(
            array=ratio,
            linecolor_dim="Region",
            intra_line_dim="CutTime",
            x_array=cut_gdppc,
            xlabel="GDP/PPP [2005 USD]",
            ylabel="Ratio",
            title=f"Ratio of Bottom-Up (SD) Stock to Top-down (DSM) Stock Estimate (1990-2023)",
        )

        # fig = ap_ratio.plot()
        # fig.update_xaxes(type="log", range=[3, 5])

        self.plot_and_save_figure(ap_ratio, "ratio.png")

        # ratio over time
        ap_ratio = self.plotter_class(
            array=ratio,
            linecolor_dim="Region",
            intra_line_dim="CutTime",
            xlabel="Time",
            ylabel="Ratio",
            title=f"Ratio of Bottom-Up (SD) Stock to Top-down (DSM) Stock Estimate (1990-2023)",    
        )

        self.plot_and_save_figure(ap_ratio, "ratio_time.png")

        # top vs bottom
        ap_tb = self.plotter_class(
            array=cut_stock_sd,
            x_array=cut_stock,
            linecolor_dim="Region",
            intra_line_dim="CutTime",
            xlabel="Top-down (DSM) Stock Estimate (t)",
            ylabel="Bottom-up (SD) Stock Estimate (t)",
            title=f"Bottom-Up (SD) Stock vs Top-down (DSM) Stock Estimate (1990-2023)",    
        )

        fig = ap_tb.plot()
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

        self.plot_and_save_figure(ap_tb, "tb.png")

        cut_time = fd.Dimension(name="CutTime", letter="p", items=np.arange(2024, 2101))
        cut_stock_sd = stock_sd[{"t": cut_time}]
        cut_stock = stock[{"t": cut_time}]
        cut_gdppc = gdppc[{"t": cut_time}]

        ratio = cut_stock_sd / cut_stock

        # ratio 2024-2100 over gdppc
        ap_ratio = self.plotter_class(
            array=ratio,
            linecolor_dim="Region",
            intra_line_dim="CutTime",
            x_array=cut_gdppc,
            xlabel="GDP/PPP [2005 USD]",
            ylabel="Ratio",
            title=f"Ratio of Bottom-Up (SD) Stock to Top-down (DSM) Stock Estimate (2024-2100)",
        )

        self.plot_and_save_figure(ap_ratio, "future_ratio.png")

