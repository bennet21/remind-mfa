from typing import Optional
import flodym as fd

class CementStockModels:
    
    @staticmethod
    def calc_stock_bottom_up(prms: dict[str, fd.Parameter]):
        stk = prms["concrete_building_mi"] * prms["building_split"] * prms["floorspace"]
        return stk

    @staticmethod
    def calc_stock_bottom_up_minimal(prms: dict[str, fd.Parameter], stock_type_dimletter: str = "s"):

        stk = CementStockModels.calc_stock_bottom_up(prms)

        # build up new stock where function (f) and stock type (s) are merged into reduced stock type (u)
        new_stk = fd.FlodymArray(dims=stk.dims.drop("f"))
        new_stk[{stock_type_dimletter: 'Res'}] = stk[{'f': 'RS', stock_type_dimletter: "Res"}] + stk[{'f': 'RM', stock_type_dimletter: "Res"}]
        new_stk[{stock_type_dimletter: 'Com'}] = stk[{'f': 'Com', stock_type_dimletter: "Com"}]
        new_stk = new_stk.sum_over('b')
        return new_stk
    
    @staticmethod
    def calc_cement_stock_top_down(
        prms: dict[str, fd.Parameter],
        stk_obj: Optional[fd.Stock] = None,
        lifetime_model: fd.LifetimeModel = fd.LogNormalLifetime,
        time_letter: str = "h",
        ):
        cement_consumption = (
            (1 - prms["cement_losses"])
            * (prms["cement_production"] - prms["cement_trade"])
            * prms["stock_type_split"]
        )

        if stk_obj is None:
            stk_obj = fd.InflowDrivenDSM(
                dims=cement_consumption.dims,
                lifetime_model=lifetime_model,
                time_letter=time_letter,
            )

        # in use
        stk_obj.inflow[...] = cement_consumption
        stk_obj.lifetime_model.set_prms(
            mean=prms["use_lifetime_mean"],
            std=prms["use_lifetime_rel_std"] * prms["use_lifetime_mean"],
        )
        stk_obj.compute()
        return stk_obj
    
    def transfrom_cement_to_product_stock(prms, stk: fd.FlodymArray):
        cement_ratio = (
            prms["product_cement_content"] / prms["product_density"]
        )
        stk = stk * (
            prms["product_material_split"]
            * prms["product_material_application_transform"]
            * prms["product_application_split"]
            / cement_ratio
        )
        return stk
    
    @staticmethod
    def calc_stock_top_down_minimal(
        prms: dict[str, fd.Parameter],
        lifetime_model: fd.LifetimeModel = fd.LogNormalLifetime,
        time_letter: str = "h",
        ):

        cement_stk_obj = CementStockModels.calc_cement_stock_top_down(prms, lifetime_model=lifetime_model, time_letter=time_letter)
        product_stk = CementStockModels.transfrom_cement_to_product_stock(prms, cement_stk_obj.stock)

        stk = product_stk[{"m": "concrete"}]
        stk = stk.sum_over("a")

        return stk

