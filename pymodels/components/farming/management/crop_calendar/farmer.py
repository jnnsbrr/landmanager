"""Farmer entity type class of inseeds_farmer_management
"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license
import re
import numpy as np
import xarray as xr
from enum import Enum
import datetime

from pymodels.components import farming
from pymodels.components import base


class Farmer(farming.Farmer):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, **kwargs):
        """
        Initialize an instance of a Farmer that cultivates crops for which
        the each crop calendar is calculated.
        """
        super().__init__(**kwargs)  # must be the first line

        self.crops = CultivatedCrops(cell=self.cell)

    def update(self, t):
        # call the base class update method
        super().update(t)
        # update the crops
        self.crops.update()


class CellActivity:
    """
    Base class for any cell activities.
    """

    irrigation_systems_short = ["rainfed", "irrigated"]

    irrigation_systems_long = [
        "rainfed",
        "surface irrigated",
        "sprinkler irrigated",
        "drip irrigated",
    ]

    irrigation_systems = set(irrigation_systems_short + irrigation_systems_long)

    irrigation_pattern = re.compile(
        r"^(?:" + "|".join(map(re.escape, irrigation_systems)) + r")\s*"  # noqa
    )

    def __init__(self, cell, **kwargs):
        """
        Initialize the CellActivity object by adding the cell to the object.
        """
        self._cell = cell

        # basic time variables
        self.ndays_year = 365
        self.midday = get_midday()


    @property
    def cell(self):
        return self._cell

    def var(self, cell_attribute, avg=None):
        """Return the given attribute."""
        if avg == "monthly":
            return self.cell.output[cell_attribute].mean("time")
        elif avg == "annual" and "month" in self.cell.output.harvestc.attrs["units"]:
            return self.cell.output[cell_attribute].mean("band")
        else:
            return self.cell.output[cell_attribute]


class Crop(CellActivity):

    def __init__(self, name, parameters, column_name="cft_name", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.__dict__.update(
            parameters.query(
                f"{column_name} == '{name}' and winter_crop == {int(self.is_winter_crop)}"
            )
            .squeeze()
            .to_dict()
        )

        # initiate calculation of the sowing date (calc_sowing_date)
        self.sdate = None
        self.smonth = None
        self.sseason = None

        # initiate calculation of the harvest rule (calc_harvest_rule)
        self.harvest_rule = None
        self.harvest_rule_name = None

        # initiate calculation of the harvest options (calc_harvest_options)
        self.hdate_first = None
        self.hdate_maxrp = None
        self.hdate_last = None
        self.hdate_wetseas = None
        self.hdate_temp_base = None
        self.hdate_temp_opt = None

        # initiate calculation of the harvest date (calc_harvest_date)
        self.hdate_rf = None
        self.hdate_ir = None
        self.hreason_rf = None
        self.hreason_ir = None

    @property
    def is_winter_crop(self, temperate_cereals_only=True):
        """
        Tests if a given growing season should be classified as winter crop

        :param temperate_cereals_only: If True, only temperate cereals are classified as winter crops
        :type temperate_cereals_only: bool
        :return: True if the crop is a winter crop, False otherwise
        :rtype: bool
        """
        # initiate winter crop as False, assuming no winter crop as default
        is_winter_crop = False

        # temperate cereals is the only winter crop that has specific parameters
        if temperate_cereals_only and self.name not in ["temperate cereals"]:
            return is_winter_crop

        # get the last sowing date
        sdate = (
            self.cell.output.sdate.sel(band=f"rainfed {self.name}").isel(time=-1).item()
        )

        if sdate > 0:
            # get the last harvest date
            hdate = (
                self.cell.output.hdate.sel(band=f"rainfed {self.name}")
                .isel(time=-1)
                .item()
            )
            # get the latitude of the cell
            lat = self.cell.grid.lat.item()
            # get the lowest monthly climate temperate
            lowest_temp = self.cell.output.temp.min().item()
            grow_period = hdate - sdate if sdate <= hdate else 365 + hdate - sdate

        if lat > 0:
            if (sdate + grow_period > 365 and grow_period >= 150) and (
                -10 <= lowest_temp <= 7
            ):
                is_winter_crop = True
        else:
            if (sdate < 182 and sdate + grow_period > 182 and grow_period >= 150) and (
                -10 <= lowest_temp <= 7
            ):
                is_winter_crop = True

        return is_winter_crop

    def calc_sowing_date(self, monthly_climate, seasonality):
        """
        Calculate sowing date (Waha et al., 2012)

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param seasonality: Seasonality classification.
        :type seasonality: str

        """
        lat = self.cell.grid.lat.item()

        # Constrain first possible date for winter crop sowing
        earliest_sdate = self.initdate_sdatenh if lat >= 0 else self.initdate_sdatesh
        earliest_smonth = doy2month(earliest_sdate)
        default_doy = 1 if lat >= 0 else 182
        default_month = 0

        # What type of winter is it?
        if (min(monthly_climate["temp"]) > self.basetemp_low) and (
            seasonality in ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]
        ):
            # "Warm winter" (allowing non-vernalizing winter-sown crops)
            # sowing 2.5 months before coldest midday
            # it seems a good approximation for both India and South US)
            coldestday = self.midday[np.argmin(monthly_climate["temp"].values)]
            firstwinterdoy = (
                coldestday - 75 if coldestday - 75 > 0 else coldestday - 75 + 365
            )

        elif (min(monthly_climate["temp"]) < -10) and (
            seasonality in ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]
        ):
            # "Cold winter" (winter too harsh for winter crops, only spring sowing possible)
            firstwinterdoy = -9999

        else:
            # "Mild winter" (allowing vernalizing crops)
            firstwinterdoy = calc_doy_cross_threshold(
                monthly_climate["temp"], self.temp_fall
            )["doy_cross_down"]

        # First day of winter
        firstwintermonth = (
            default_month if firstwinterdoy == -9999 else doy2month(firstwinterdoy)
        )
        firstwinterdoy = default_doy if firstwinterdoy == -9999 else firstwinterdoy

        # First day of spring
        firstspringdoy = calc_doy_cross_threshold(
            monthly_climate["temp"], self.temp_spring
        )["doy_cross_up"]
        firstspringmonth = (
            default_month if firstspringdoy == -9999 else doy2month(firstspringdoy)
        )

        # If winter type
        if self.calcmethod_sdate == "WTYP_CALC_SDATE":

            if firstwinterdoy > earliest_sdate and firstwintermonth != default_month:
                smonth = firstwintermonth
                sdate = firstwinterdoy
                sseason = "winter"

            elif (
                firstwinterdoy <= earliest_sdate
                and min(monthly_climate["temp"]) > temp_fall
                and firstwintermonth != default_month
            ):
                smonth = earliest_smonth
                sdate = earliest_sdate
                sseason = "winter"

            else:
                smonth = firstspringmonth
                sdate = firstspringdoy
                sseason = "spring"

        else:
            if seasonality == "NO_SEASONALITY":
                smonth = default_month
                sdate = default_doy
                sseason = "spring"

            elif seasonality in ["PREC", "PRECTEMP"]:
                sdate = calcDoyWetMonth(monthly_climate["ppet"])
                smonth = doy2month(sdate)
                sseason = "spring"

            else:
                smonth = firstspringmonth
                sdate = firstspringdoy
                sseason = "spring"

        self.sdate = default_doy if smonth == default_month else sdate
        self.smonth = (
            default_month
            if self.calcmethod_sdate == "WTYP_CALC_SDATE" and sseason == "spring"
            else smonth
        )
        self.sseason = sseason

    def calc_harvest_rule(self, monthly_climate, seasonality):
        """
        Calculate harvest rule (Minoli et al., 2019).

        This method performs an agro-climatic classification of climate, based
        on monthly temperature and precipitation profiles.
        The classification is derived by intersecting the seasonality classes
        (see calcSeasonality) with the temperature of the warmest month,
        compared to crop-specific thresholds (base and optimal temperatures for
        reproductive growth):
        * t-low, temperatures always lower than the base temperature;
        * t-mid, temperatures exceed the base temperature, but are always lower
            than the optimum temperature;
        * t-high, temperatures exceed the optimum temperature.

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :return: Harvest rule classification.
        :rtype: int
        """
        temp_max = max(monthly_climate["temp"])
        temp_min = min(monthly_climate["temp"])

        if seasonality == "NO_SEASONALITY":
            if temp_max <= self.temp_base_rphase:
                harvest_rule = 1
                harvest_rule_name = "t-low_no-seas"
            elif self.temp_base_rphase < temp_max <= self.temp_opt_rphase:
                harvest_rule = 4
                harvest_rule_name = "t-mid_no-seas"
            else:
                harvest_rule = 7
                harvest_rule_name = "t-high_no-seas"

        elif seasonality == "PREC":
            if temp_max <= self.temp_base_rphase:
                harvest_rule = 2
                harvest_rule_name = "t-low_prec-seas"
            elif self.temp_base_rphase < temp_max <= self.temp_opt_rphase:
                harvest_rule = 5
                harvest_rule_name = "t-mid_prec-seas"
            else:
                harvest_rule = 8
                harvest_rule_name = "t-high_prec-seas"

        else:
            if temp_max <= self.temp_base_rphase:
                harvest_rule = 3
                harvest_rule_name = "t-low_mix-seas"
            elif self.temp_base_rphase < temp_max <= self.temp_opt_rphase:
                harvest_rule = 6
                harvest_rule_name = "t-mid_mix-seas"
            else:
                harvest_rule = 9
                harvest_rule_name = "t-high_mix-seas"

        self.harvest_rule = harvest_rule
        self.harvest_rule_name = harvest_rule_name

    def calc_harvest_options(self, monthly_climate):
        """
        Calculate harvest date vector.

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        """

        # Shortest cycle: crop lower biological limit
        hdate_first = self.sdate + self.min_growingseason
        # Medium cycle: best trade-off vegetative and reproductive growth
        hdate_maxrp = self.sdate + self.maxrp_growingseason
        # Longest cycle: crop upper biological limit
        hdate_last = self.sdate + (
            self.max_growingseason_wt
            if self.sseason == "winter"
            else self.max_growingseason_st
        )

        # End of wet season
        doy_wet1 = calc_doy_cross_threshold(monthly_climate["ppet"], self.ppet_ratio)[
            "doy_cross_down"
        ]
        doy_wet2 = calc_doy_cross_threshold(
            monthly_climate["ppet_diff"], self.ppet_ratio_diff
        )["doy_cross_down"]
        doy_wet_vec = [
            doy + self.ndays_year if doy < self.sdate and doy != -9999 else doy
            for doy in [doy_wet1, doy_wet2]
        ]
        doy_wet_first = min([doy for doy in doy_wet_vec if doy != -9999], default=-9999)
        hdate_wetseas = (
            hdate_last
            if doy_wet1 == -9999 and min(monthly_climate["ppet"]) < self.ppet_min
            else doy_wet_first + self.rphase_duration
        )

        # Warmest day of the year
        warmest_day = self.midday[monthly_climate["temp"].argmax()]
        hdate_temp_base = (
            warmest_day
            if self.sseason == "winter"
            else warmest_day + self.rphase_duration
        )

        # First hot day
        doy_exceed_opt_rp = calc_doy_cross_threshold(
            monthly_climate["temp"], 
            self.temp_opt_rphase
        )["doy_cross_up"]
        doy_exceed_opt_rp = doy_exceed_opt_rp + self.ndays_year if doy_exceed_opt_rp < self.sdate and doy_exceed_opt_rp != -9999 else doy_exceed_opt_rp

        # Last hot day
        doy_below_opt_rp = calc_doy_cross_threshold(monthly_climate["temp"], self.temp_opt_rphase)[
            "doy_cross_down"
        ]
        doy_below_opt_rp = doy_below_opt_rp + self.ndays_year if doy_below_opt_rp < self.sdate and doy_below_opt_rp != -9999 else doy_below_opt_rp

        # Winter type: First hot day; Spring type: Last hot day
        doy_opt_rp = (
            doy_exceed_opt_rp if self.sseason == "winter" else doy_below_opt_rp
        )
        hdate_temp_opt = (
            hdate_maxrp
            if doy_opt_rp == -9999
            else doy_opt_rp + (self.rphase_duration if self.sseason != "winter" else 0)
        )

        self.hdate_first = hdate_first
        self.hdate_maxrp = hdate_maxrp
        self.hdate_last = hdate_last
        # If harvest date < sowing date, it occurs the following year, so add 365 days
        self.hdate_wetseas = hdate_wetseas + self.ndays_year if hdate_wetseas < self.sdate else hdate_wetseas
        self.hdate_temp_base = (
            hdate_temp_base + self.ndays_year if hdate_temp_base < self.sdate else hdate_temp_base
        )
        self.hdate_temp_opt = (
            hdate_temp_opt + self.ndays_year if hdate_temp_opt < self.sdate else hdate_temp_opt
        )

    def calc_harvest_date(
        self,
        monthly_climate,
        seasonality
    ):
        """
        Calculate harvest date (Minoli et al., 2019).

        Rule-based estimation of the end of the crop growing period (date of
        physiological maturity), here called harvest date for simplicity.
        The assumption behind these rules is that farmers select growing seasons
        based on the mean climatic characteristics of the location in
        which they operate and on the physiological limitations (base and optimum
        temperatures for reproductive growth; sensitivity to terminal water stress)
        of the respective crop species.

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict

        """

        if seasonality == "NO_SEASONALITY":

            if self.harvest_rule == 1:
                hdate_rf = self.hdate_first
                hdate_ir = self.hdate_first
                hreason_rf = hreason_ir = "hdate_first"
            else:
                hdate_rf = self.hdate_maxrp
                hdate_ir = self.hdate_maxrp
                hreason_rf = hreason_ir = "hdate_maxrp"

        elif seasonality == "PREC":

            if self.harvest_rule == 2:
                hdate_rf = self.hdate_first
                hdate_ir = self.hdate_first
                hreason_rf = hreason_ir = "hdate_first"
            else:
                hdate_rf = min(max(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp)
                hdate_ir = self.hdate_maxrp
                hreason_rf = "hdate_wetseas" if hdate_rf == self.hdate_wetseas else "hdate_maxrp"
                hreason_ir = "hdate_maxrp"

        else:
            if self.sseason == "winter":

                if self.harvest_rule == 3:
                    hdate_rf = self.hdate_first
                    hdate_ir = self.hdate_first
                    hreason_rf = hreason_ir = "hdate_first"
                elif self.harvest_rule == 6:
                    if self.smonth == 0 and max(monthly_climate["temp"]) < self.temp_fall:
                        hdate_rf = self.hdate_first
                        hdate_ir = self.hdate_first
                        hreason_rf = hreason_ir = "hdate_first"
                    else:
                        hdate_rf = min(max(self.hdate_first, self.hdate_temp_base), self.hdate_last)
                        hdate_ir = min(max(self.hdate_first, self.hdate_temp_base), self.hdate_last)
                        hreason_rf = "hdate_temp_base" if hdate_rf == self.hdate_temp_base else "hdate_last"
                        hreason_ir = "hdate_temp_base" if hdate_ir == self.hdate_temp_base else "hdate_last"
                else:
                    hdate_rf = min(max(self.hdate_first, self.hdate_temp_opt), self.hdate_last)
                    hdate_ir = min(max(self.hdate_first, self.hdate_temp_opt), self.hdate_last)
                    hreason_rf = "hdate_temp_opt" if hdate_rf == self.hdate_temp_opt else "hdate_last"
                    hreason_ir = "hdate_temp_opt" if hdate_ir == self.hdate_temp_opt else "hdate_last"

            else:

                if self.harvest_rule == 3:
                    hdate_rf = self.hdate_first
                    hdate_ir = self.hdate_first
                    hreason_rf = hreason_ir = "hdate_first"
                elif self.harvest_rule == 6:
                    if self.smonth == 0 and max(monthly_climate["temp"]) < self.temp_spring:
                        hdate_rf = self.hdate_first
                        hdate_ir = self.hdate_first
                        hreason_rf = hreason_ir = "hdate_first"
                    elif seasonality == "PRECTEMP":
                        hdate_rf = min(max(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp)
                        hdate_ir = self.hdate_maxrp
                        hreason_rf = "hdate_wetseas" if hdate_rf == self.hdate_wetseas else "hdate_maxrp"
                        hreason_ir = "hdate_maxrp"
                    else:
                        hdate_rf = min(max(self.hdate_first, self.hdate_temp_base), max(self.hdate_first, self.hdate_wetseas), self.hdate_last)
                        hdate_ir = min(max(self.hdate_first, self.hdate_temp_base), self.hdate_last)
                        hreason_rf = "hdate_temp_base" if hdate_rf == self.hdate_temp_base else "hdate_last"
                        hreason_ir = "hdate_temp_base" if hdate_ir == self.hdate_temp_base else "hdate_last"
                else:
                    if seasonality == "PRECTEMP":
                        hdate_rf = min(max(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp)
                        hdate_ir = self.hdate_maxrp
                        hreason_rf = "hdate_wetseas" if hdate_rf == self.hdate_wetseas else "hdate_maxrp"
                        hreason_ir = "hdate_maxrp"
                    else:
                        hdate_rf = min(max(self.hdate_first, self.hdate_temp_opt), max(self.hdate_first, self.hdate_wetseas), self.hdate_last)
                        hdate_ir = min(max(self.hdate_first, self.hdate_temp_opt), self.hdate_last)
                        hreason_rf = "hdate_temp_opt" if hdate_rf == self.hdate_temp_opt else "hdate_last"
                        hreason_ir = "hdate_temp_opt" if hdate_ir == self.hdate_temp_opt else "hdate_last"


        self.hdate_rf = hdate_rf if hdate_rf <= self.ndays_year else hdate_rf - self.ndays_year
        self.hdate_ir = hdate_ir if hdate_ir <= self.ndays_year else hdate_ir - self.ndays_year
        self.hreason_rf = hreason_rf
        self.hreason_ir = hreason_ir


    def calc_phu(self, monthly_climate, hdate, vern_factor=None, phen_model="t"):
        """
        Calculate PHU requirements.

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param hdate: Maturity date (DOY).
        :type hdate: int
        :param vern_factor: Vernalization factor from calcVf(), length should be 365.
        :type vern_factor: list or numpy.ndarray
        :param phen_model: Phenological model, one of "t", "tv", "tp", "tvp".
        :type phen_model: str
        :return: Total Thermal Unit Requirements.
        :rtype: int
        """

        mdt = interpolate_monthly_to_daily(monthly_climate["temp"])

        husum = 0

        # Select days not in growing period
        hdate = hdate if self.sdate < hdate else hdate + 365
        if hdate <= 365:
            days_no_gp = list(range(1, self.sdate)) + list(range(hdate, 366))
        else:
            days_no_gp = list(range(hdate - 365, self.sdate))

        # Compute Effective Thermal Units (teff)
        if phen_model == "t":
            teff = [
                max(temp - self.basetemp_low, 0) if i not in days_no_gp else 0
                for i, temp in enumerate(mdt["y"], start=1)
            ]
        elif phen_model == "tv":
            teff = [
                max(temp - self.basetemp_low, 0) * vf if i not in days_no_gp else 0
                for i, (temp, vf) in enumerate(zip(mdt["y"], vern_factor), start=1)
            ]
        else:
            raise ValueError("Error: phen_model not declared!")

        # Total Thermal Unit Requirements
        if phen_model == "t":
            husum = int(sum(teff))
        elif phen_model == "tv":
            husum = int(sum(teff)) * -1
        else:
            raise ValueError("Error: phen_model not declared!")

        return husum


class CultivatedCrops(CellActivity):

    def __init__(self, crop_param_file="crop_calendar/crop_parameters.csv", **kwargs):
        super().__init__(**kwargs)
        # actual crop landuse
        self.parameters = base.load_csv(crop_param_file)
        # get all cultivated crops
        self.update()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    @property
    def actual_landuse(self):
        """
        Return the actual land use of the cell.
        """
        return self.cell.input.landuse.where(self.cell.input.landuse > 0).dropna(
            dim="band"
        )

    def update_landuse(self):
        """
        Update the cultivated crops based on the actual land use (what is actually
        cultivated in the cell) and for which crops is crop calendar data
        available (crop_parameters.csv).
        Crops are currently only initiated if both conditions are met and
        deleted if the crop is not cultivated anymore.
        """
        self.bands = self.actual_landuse.band.values.tolist()
        self.names = {
            CultivatedCrops.irrigation_pattern.sub("", band).strip()
            for band in self.bands  # noqa
        }
        self.calendars = {
            crop for crop in self.names if crop in self.parameters.cft_name.tolist()
        }

        for crop in self.parameters.cft_name:
            if crop in self.names and not hasattr(self, crop):
                self[crop] = Crop(
                    name=crop,
                    parameters=self.parameters,
                    cell=self.cell,
                )
            elif crop not in self.names and hasattr(self, crop):
                del self[crop]

    def calc_seasonality(self, monthly_climate, temp_min=10):
        """
        Calculate the seasonality type.

        Seasonality calculation based on average monthly climate as described
        in Waha et al. 2012.
        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param temp_min: Threshold of temperature of the coldest month
            (°C). Default is 10°C.

        """

        var_coef_prec = calc_var_coeff(monthly_climate["prec"])
        var_coef_temp = calc_var_coeff(deg2k(monthly_climate["temp"]))
        min_temp = np.nanmin(monthly_climate["temp"])

        # Determine seasonality
        if var_coef_prec <= 0.4 and var_coef_temp <= 0.01:
            return "NO_SEASONALITY"
        elif var_coef_prec > 0.4 and var_coef_temp <= 0.01:
            return "PREC"
        elif var_coef_prec > 0.4 and var_coef_temp > 0.01:
            if min_temp > temp_min:
                return "PRECTEMP"
            else:
                return "TEMPPREC"
        else:
            return "TEMP"

    @property
    def monthly_climate(self):
        """
        Return monthly climate data for the cell.
        """
        # average temperature of the cell
        temp = self.var("temp", avg="monthly")

        # average temperature of the cell
        prec = self.var("prec", avg="monthly")

        # average temperature of the cell
        ppet = (self.var("prec") / self.var("pet")).mean("time")

        # average potential evaporation of the cell
        ppet_diff = ppet - ppet.roll(band=1)


        return {"temp": temp, "prec": prec, "ppet": ppet, "ppet_diff": ppet_diff}

    def set_input(self, name, value, crop, system="short"):
        """
        Set the cell input data for the crop.

        :param name: Input data name.
        :type name: str
        :param value: Input data value.
        :type value: float
        :param crop: Crop name.
        :type crop: str
        :param system: Irrigation system of either long ("rainfed",
        "surface irrigated", "sprinkler irrigated", "drip irrigated") or short
        ("rainfed", "irrigated") or "short" for all two / "long" for all four.
        :type system: str
        """

        if system in Crop.irrigation_systems_short or system == "short":
            all_systems = Crop.irrigation_systems_short
        elif system in Crop.irrigation_systems_long or system == "long":
            all_systems = Crop.irrigation_systems_long
        else:
            raise ValueError("Invalid irrigation system")

        if system in ["short", "long"]:
            system = all_systems
        else:
            system = [system]

        self.cell.input[name].loc[
            dict(band=[f"{irrig} {crop}" for irrig in system])  # noqa
        ] = [value]

    def calc_calendar(self):
        """Calculate the crop calendar for each crop in calendars."""
        monthly_climate = self.monthly_climate
        seasonality = self.calc_seasonality(monthly_climate)

        for crop in self.calendars:
            # calculate sowing date
            self[crop].calc_sowing_date(monthly_climate, seasonality)
            # write sowing date to cell input
            self.set_input(
                name="sdate",
                value=self[crop].sdate,
                crop=crop,
                system="short"
            )
            # calculate harvest rule
            self[crop].calc_harvest_rule(monthly_climate, seasonality)

            # calculate different options for harvesting
            self[crop].calc_harvest_options(monthly_climate)

            # calculate harvest date, choose the best option
            self[crop].calc_harvest_date(monthly_climate, seasonality)

            # write phu rainfed to cell input
            self.set_input(
                name="sdate",
                value=self[crop].calc_phu(monthly_climate, self[crop].hdate_rf),
                crop=crop,
                system="rainfed"
            )
            # write phu irrigated to cell input
            self.set_input(
                name="sdate",
                value=self[crop].calc_phu(monthly_climate, self[crop].hdate_ir),
                crop=crop,
                system="irrigated"
            )

    def update(self):
        """
        Update the CultivatedCrops object.
        """
        self.update_landuse()
        self.calc_calendar()


def calc_var_coeff(x):
    """
    Calculate the coefficient of variation (CV) for a given array x.
    If the mean of x is 0, return 0 to avoid division by zero.

    :param x: array-like, input data
    :type x: array-like
    :return:coefficient of variation
    :rtype: float

    """
    mean_x = np.nanmean(x)
    return 0 if mean_x == 0 else np.nanstd(x) / mean_x


def deg2k(x):
    """
    Convert temperature from degree Celsius to Kelvin.

    :param x: temperature in degree Celsius
    :type x: float
    :return: temperature in Kelvin
    :rtype: float

    """
    return x + 273.15


def doy2month(doy=1, year=2015):
    """
    Convert day of year to month.

    :param doy: day of year
    :type doy: int
    :param year: year
    :type year: int
    :return: month
    :rtype: int

    """
    date = datetime.datetime.strptime(f"{year} {doy}", "%Y %j")
    return date.month


def interpolate_monthly_to_daily(monthly_value):
    """
    Interpolate monthly values to daily values.

    :param monthly_value: monthly values
    :type monthly_value: array-like
    :return: daily values
    :rtype
    """
    # Middle day of each month
    midday = get_midday()
    ndays_year = 365

    # Replicate values twice (two years)
    midday2 = np.concatenate((midday, midday + ndays_year))
    monthly_value2 = np.tile(monthly_value, 2)

    # Interpolate monthly to daily values
    daily_value_x = []
    daily_value_y = []

    for i in range(len(monthly_value2) - 1):
        m1 = i
        m2 = i + 1
        value = np.interp(
            np.arange(midday2[m1], midday2[m2] + 1),
            [midday2[m1], midday2[m2]],
            [monthly_value2[m1], monthly_value2[m2]],
        )
        daily_value_x.extend(np.arange(midday2[m1], midday2[m2] + 1))
        daily_value_y.extend(value)

    return {"x": np.array(daily_value_x), "y": np.array(daily_value_y)}


def calc_doy_cross_threshold(monthly_value, threshold):
    """
    Calculate day of crossing threshold

    :param monthly_value: monthly values
    :type monthly_value: array-like
    :param threshold: threshold value
    :type threshold: float
    :return: day of crossing threshold
    :rtype: dict
    """
    ndays_year = 365
    daily_value = interpolate_monthly_to_daily(monthly_value)

    # Find days when value above threshold
    is_value_above = daily_value["y"] >= threshold
    is_value_above2 = np.roll(is_value_above, 1)

    # Find days when value crosses threshold
    value_cross_threshold = is_value_above.astype(int) - is_value_above2.astype(int)
    day_cross_up = daily_value["x"][np.where(value_cross_threshold == 1)[0] + 1]
    day_cross_down = daily_value["x"][np.where(value_cross_threshold == -1)[0] + 1]

    # Convert values to 1:365
    day_cross_up = day_cross_up[day_cross_up <= ndays_year]
    day_cross_down = day_cross_down[day_cross_down <= ndays_year]
    day_cross_up = np.sort(np.unique(day_cross_up))
    day_cross_down = np.sort(np.unique(day_cross_down))

    # No crossing == -9999
    day_cross_up = day_cross_up[0] if len(day_cross_up) > 0 else -9999
    day_cross_down = day_cross_down[0] if len(day_cross_down) > 0 else -9999

    return {"doy_cross_up": day_cross_up, "doy_cross_down": day_cross_down}


def get_midday():
    """
    Standardized midday of each month.
    """
    return np.array([15, 43, 74, 104, 135, 165, 196, 227, 257, 288, 318, 349])
