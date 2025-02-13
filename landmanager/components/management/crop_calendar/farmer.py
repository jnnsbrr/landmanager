"""Farmer entity type class of landmanager_farmer_management
"""
import re
import numpy as np
import xarray as xr
import pycoupler
from enum import Enum
import datetime
from scipy.interpolate import CubicSpline

from landmanager.components import management
from landmanager.components import base


class Farmer(management.Farmer):
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
        if avg in ["monthly", "daily"]:
            return self.cell.output[cell_attribute].mean("time")
        elif avg == "annual" and (
            "month" in self.cell.output.harvestc.attrs["units"] or "day" in self.cell.output.harvestc.attrs["units"]
        ):
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
        # Early return for crops that are not temperate cereals when temperate_cereals_only is True
        if temperate_cereals_only and self.name != "temperate cereals":
            return False

        # Extract cell output once for efficiency
        cell_output = self.cell.output
        band = f"rainfed {self.name}"
        
        # Get the last sowing and harvest dates
        sdate = cell_output.sdate.sel(band=band).isel(time=-1).item()
        if sdate < 0:
            return False

        hdate = cell_output.hdate.sel(band=band).isel(time=-1).item()

        # Get the latitude of the cell and the lowest temperature
        lat = self.cell.grid.lat.item()
        lowest_temp = cell_output.temp.min().item()

        # Calculate the growing period
        grow_period = hdate - sdate if sdate <= hdate else 365 + hdate - sdate

        # Conditions for winter crop classification based on latitude and temperature
        if lat > 0:
            if (sdate + grow_period > 365 and grow_period >= 150) and (-10 <= lowest_temp <= 7): # noqa
                return True
        else:
            if (sdate < 182 and sdate + grow_period > 182 and grow_period >= 150) and (-10 <= lowest_temp <= 7): # noqa
                return True

        return False

    def calc_sowing_date(self, monthly_climate, seasonality):
        """
        Calculate sowing date (Waha et al., 2012)

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param seasonality: Seasonality classification.
        :type seasonality: str
        """
        lat = self.cell.grid.lat.item()

        # Initialize default values
        default_doy = 1 if lat >= 0 else 182
        default_month = 0

        # Constrain first possible date for winter crop sowing
        earliest_sdate = self.initdate_sdatenh if lat >= 0 else self.initdate_sdatesh
        earliest_smonth = doy2month(earliest_sdate)

        # Pre-calculate values used multiple times
        min_temp = monthly_climate["min_temp"]
        daily_temp = monthly_climate["daily_temp"]
        argmin_temp = monthly_climate["argmin_temp"]

        # Initialize firstwinterdoy with -9999 as a default
        firstwinterdoy = -9999

        # First day of spring
        firstspringdoy = calc_doy_cross_threshold(daily_temp, self.temp_spring).get("doy_cross_up", -9999)
        firstspringdoy = default_doy if firstspringdoy == -9999 else firstspringdoy
        firstspringmonth = doy2month(firstspringdoy)

        # Handle different winter types
        if (min_temp > self.basetemp_low) and (seasonality in ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]):
            # "Warm winter" (allowing non-vernalizing winter-sown crops)
            # sowing 2.5 months before coldest midday
            # it seems a good approximation for both India and South US)
            coldestday = self.midday[argmin_temp]
            firstwinterdoy = (coldestday - 75) % 365  # Ensures wrap-around for DOY

        elif (min_temp < -10) and (seasonality in ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]):
            # "Cold winter" (winter too harsh for winter crops, only spring sowing possible)
            firstwinterdoy = -9999
        else:
            # "Mild winter" (allowing vernalizing crops)
            firstwinterdoy = calc_doy_cross_threshold(daily_temp, self.temp_fall).get("doy_cross_down", -9999)

        # First day of winter
        firstwinterdoy = default_doy if firstwinterdoy == -9999 else firstwinterdoy
        firstwintermonth = doy2month(firstwinterdoy) if firstwinterdoy != -9999 else default_month

        # Determine sowing date based on calculation method
        if self.calcmethod_sdate == "WTYP_CALC_SDATE":
            if firstwinterdoy > earliest_sdate and firstwintermonth != default_month:
                smonth, sdate, sseason = firstwintermonth, firstwinterdoy, "winter"
            elif firstwinterdoy <= earliest_sdate and min_temp > self.temp_fall and firstwintermonth != default_month:
                smonth, sdate, sseason = earliest_smonth, earliest_sdate, "winter"
            else:
                smonth, sdate, sseason = firstspringmonth, firstspringdoy, "spring"
        else:
            if seasonality == "NO_SEASONALITY":
                smonth, sdate, sseason = default_month, default_doy, "spring"
            elif seasonality in ["PREC", "PRECTEMP"]:
                sdate = calc_doy_wet_month(monthly_climate["daily_ppet"])
                smonth = doy2month(sdate)
                sseason = "spring"
            else:
                smonth, sdate, sseason = firstspringmonth, firstspringdoy, "spring"

        # Final assignment (fixed smonth reset logic)
        self.sdate = sdate
        self.smonth = smonth
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

        # Mapping seasonality to corresponding rule offsets
        seasonality_mapping = {
            "NO_SEASONALITY": (1, "no-seas"),
            "PREC": (2, "prec-seas"),
        }
        base_rule, rule_name_suffix = seasonality_mapping.get(seasonality, (3, "mix-seas"))

        max_temp = monthly_climate["max_temp"]

        if max_temp <= self.temp_base_rphase:
            rule_offset = 0  # t-low
            rule_prefix = "t-low"
        elif max_temp <= self.temp_opt_rphase:
            rule_offset = 3  # t-mid
            rule_prefix = "t-mid"
        else:
            rule_offset = 6  # t-high
            rule_prefix = "t-high"

        self.harvest_rule = base_rule + rule_offset
        self.harvest_rule_name = f"{rule_prefix}_{rule_name_suffix}"

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
            self.max_growingseason_wt if self.sseason == "winter" else self.max_growingseason_st
        )

        # End of wet season
        daily_ppet = monthly_climate["daily_ppet"]
        daily_ppet_diff = monthly_climate["daily_ppet_diff"]
        doy_wet1 = calc_doy_cross_threshold(daily_ppet, self.ppet_ratio)["doy_cross_down"]
        doy_wet2 = calc_doy_cross_threshold(daily_ppet_diff, self.ppet_ratio_diff)["doy_cross_down"]

        # Adjust wet season dates if they occur before sowing date
        doy_wet_vec = np.array([doy_wet1, doy_wet2])
        doy_wet_vec[doy_wet_vec < self.sdate] += self.ndays_year
        doy_wet_vec = doy_wet_vec[doy_wet_vec != -9999]

        doy_wet_first = np.min(doy_wet_vec) if doy_wet_vec.size > 0 else -9999
        hdate_wetseas = (
            hdate_last if doy_wet1 == -9999 and monthly_climate["min_ppet"] < self.ppet_min
            else doy_wet_first + self.rphase_duration
        )

        # Warmest day of the year
        warmest_day = self.midday[monthly_climate["argmax_temp"]]
        hdate_temp_base = (
            warmest_day if self.sseason == "winter" else warmest_day + self.rphase_duration
        )

        # First and last hot day
        daily_temp = monthly_climate["daily_temp"]
        doy_exceed_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)["doy_cross_up"]
        doy_below_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)["doy_cross_down"]

        # Adjust for year boundaries
        doy_exceed_opt_rp = doy_exceed_opt_rp + self.ndays_year if doy_exceed_opt_rp < self.sdate and doy_exceed_opt_rp != -9999 else doy_exceed_opt_rp
        doy_below_opt_rp = doy_below_opt_rp + self.ndays_year if doy_below_opt_rp < self.sdate and doy_below_opt_rp != -9999 else doy_below_opt_rp

        # Winter type: First hot day; Spring type: Last hot day
        doy_opt_rp = doy_exceed_opt_rp if self.sseason == "winter" else doy_below_opt_rp
        hdate_temp_opt = (
            hdate_maxrp if doy_opt_rp == -9999 else doy_opt_rp + (self.rphase_duration if self.sseason != "winter" else 0)
        )

        # Store results, adjusting for next-year cases
        self.hdate_first = hdate_first
        self.hdate_maxrp = hdate_maxrp
        self.hdate_last = hdate_last
        self.hdate_wetseas = hdate_wetseas + self.ndays_year if hdate_wetseas < self.sdate else hdate_wetseas
        self.hdate_temp_base = hdate_temp_base + self.ndays_year if hdate_temp_base < self.sdate else hdate_temp_base
        self.hdate_temp_opt = hdate_temp_opt + self.ndays_year if hdate_temp_opt < self.sdate else hdate_temp_opt

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

        # Precompute max temperature once
        max_temp = monthly_climate["max_temp"]

        # Helper function to determine harvest date and reason
        def set_harvest_date(rule, temp_cond=None):
            if rule == 1:
                return self.hdate_first, self.hdate_first, "hdate_first"
            elif rule == 2:
                return (
                    self.hdate_first,
                    min(max(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp),
                    "hdate_wetseas" if temp_cond == self.hdate_wetseas else "hdate_maxrp"
                )
            elif rule == 3:
                return self.hdate_first, self.hdate_first, "hdate_first"
            elif rule == 6:
                if self.smonth == 0 and max_temp < temp_cond:
                    return self.hdate_first, self.hdate_first, "hdate_first"
                else:
                    return (
                        min(max(self.hdate_first, self.hdate_temp_base), self.hdate_last),
                        min(max(self.hdate_first, self.hdate_temp_base), self.hdate_last),
                        "hdate_temp_base" if temp_cond == self.hdate_temp_base else "hdate_last"
                    )
            else:
                return (
                    min(max(self.hdate_first, self.hdate_temp_opt), self.hdate_last),
                    min(max(self.hdate_first, self.hdate_temp_opt), self.hdate_last),
                    "hdate_temp_opt" if temp_cond == self.hdate_temp_opt else "hdate_last"
                )

        # Handle the different seasonality types
        if seasonality == "NO_SEASONALITY":
            hdate_rf, hdate_ir, hreason_rf = set_harvest_date(self.harvest_rule)
            hreason_ir = hreason_rf
        elif seasonality == "PREC":
            hdate_rf, hdate_ir, hreason_rf = set_harvest_date(self.harvest_rule)
            hreason_ir = "hdate_maxrp"
        else:
            if self.sseason == "winter":
                hdate_rf, hdate_ir, hreason_rf = set_harvest_date(
                    self.harvest_rule,
                    self.temp_fall if self.smonth == 0 else self.temp_spring
                )
            else:
                hdate_rf, hdate_ir, hreason_rf = set_harvest_date(
                    self.harvest_rule
                )
            hreason_ir = hreason_rf

        # Adjust the harvest dates based on year length
        self.hdate_rf = hdate_rf if hdate_rf <= self.ndays_year else hdate_rf - self.ndays_year
        self.hdate_ir = hdate_ir if hdate_ir <= self.ndays_year else hdate_ir - self.ndays_year
        self.hreason_rf = hreason_rf
        self.hreason_ir = hreason_ir

    def calc_phu(self, daily_temp, hdate, vern_factor=None, phen_model="t"):
        """
        Calculate PHU requirements.

        :param daily_temp: Dict of two arrays of interpolated monthly to daily temperatures.
        :type daily_temp: dict
        :param hdate: Maturity date (DOY).
        :type hdate: int
        :param vern_factor: Vernalization factor from calcVf(), length should be 365.
        :type vern_factor: list or numpy.ndarray
        :param phen_model: Phenological model, one of "t", "tv", "tp", "tvp".
        :type phen_model: str
        :return: Total Thermal Unit Requirements.
        :rtype: int
        """


        # Adjust hdate for cross-year growth
        hdate = hdate if self.sdate < hdate else hdate + 365

        # Determine days outside the growing period
        days = np.arange(1, 366)  # Days in the year
        if hdate <= 365:
            days_no_gp = np.concatenate((np.arange(1, self.sdate), np.arange(hdate, 366)))
        else:
            days_no_gp = np.arange(hdate - 365, self.sdate)

        # Create a boolean mask for growing period days
        grow_mask = np.isin(days, days_no_gp, invert=True)

        # Convert temperature data to a NumPy array
        temp_array = np.array(daily_temp["value"])

        # Compute Effective Thermal Units (teff) using vectorized operations
        if phen_model == "t":
            teff = np.maximum(temp_array - self.basetemp_low, 0) * grow_mask
        elif phen_model == "tv":
            teff = np.maximum(temp_array - self.basetemp_low, 0) * np.array(vern_factor) * grow_mask
        else:
            raise ValueError("Error: phen_model not declared!")

        # Compute total PHU requirement efficiently
        husum = int(teff.sum())

        # Negate if using "tv" model
        if phen_model == "tv":
            husum *= -1

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

        # Determine seasonality
        if var_coef_prec <= 0.4 and var_coef_temp <= 0.01:
            return "NO_SEASONALITY"
        elif var_coef_prec > 0.4 and var_coef_temp <= 0.01:
            return "PREC"
        elif var_coef_prec > 0.4 and var_coef_temp > 0.01:
            if monthly_climate["min_temp"] > temp_min:
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

        daily_temp = interpolate_monthly_to_daily(temp)

        min_temp = min(temp.values)
        max_temp = max(temp.values)

        argmin_temp = np.argmin(temp.values)
        argmax_temp = np.argmax(temp.values)

        # average temperature of the cell
        prec = self.var("prec", avg="monthly")

        # Fix unrealistic low PET values
        pre_pet = np.maximum(self.var("pet"), 1e-1)

        ppet = (self.var("prec") / pre_pet).mean("time", skipna=True)

        min_ppet = np.nanmin(ppet.values)
        daily_ppet = interpolate_monthly_to_daily(ppet)

        # average potential evaporation of the cell
        ppet_diff = ppet - ppet.roll(band=1)
        daily_ppet_diff = interpolate_monthly_to_daily(ppet_diff)

        return {
            "temp": temp,
            "daily_temp": daily_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "argmin_temp": argmin_temp,
            "argmax_temp": argmax_temp,
            "prec": prec,
            "ppet": ppet,
            "daily_ppet": daily_ppet,
            "min_ppet": min_ppet,
            "ppet_diff": ppet_diff,
            "daily_ppet_diff": daily_ppet_diff
        }

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

        # Determine the irrigation systems (avoid redundant checks)
        if system == "short":
            all_systems = Crop.irrigation_systems_short
        elif system == "long":
            all_systems = Crop.irrigation_systems_long
        elif system in Crop.irrigation_systems_short or system in Crop.irrigation_systems_long:
            all_systems = [system]
        else:
            raise ValueError("Invalid irrigation system")

        # Avoid unnecessary list creation and directly build the band string
        systems_str = [f"{irrig} {crop}" for irrig in all_systems]

        # Bulk assign the value, avoiding multiple indexing
        mask = self.cell.input[name]['band'].isin(systems_str)
        self.cell.input[name].values[mask] = value

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

            self.phu_rf = self[crop].calc_phu(monthly_climate["daily_temp"], self[crop].hdate_rf)

            if self[crop].hdate_rf != self[crop].hdate_ir:
                self.phu_ir = self[crop].calc_phu(monthly_climate["daily_temp"], self[crop].hdate_ir)
            else:
                self.phu_ir = self.phu_rf

            # write phu rainfed to cell input
            self.set_input(
                name="crop_phu",
                value=[[self.phu_rf], [self.phu_ir]],
                crop=crop,
                system="short"
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
    Interpolates monthly values to daily values using a periodic cubic spline.

    :param monthly_value: Array-like (length 12) of monthly values (e.g.,
        temperature).
    :return: Dictionary with "day" (day of year, length 365) and "value"
        (interpolated values, length 365).
    """
    midday = get_midday()
    ndays_year = 365
    day = np.arange(1, ndays_year + 1)  # Days of the year

    # Use periodic boundary condition to ensure smooth wrapping
    spline = CubicSpline(
        np.append(midday, midday[0] + ndays_year),  # Extend x for periodicity
        np.append(monthly_value, monthly_value[0]),  # Extend y for periodicity
        bc_type='periodic'
    )

    value = spline(day)

    return {"day": day, "value": value}



def calc_doy_cross_threshold(daily_value, threshold):
    """
    Calculate day of crossing threshold

    :param daily_value: monthly values
    :type daily_value: array-like
    :param threshold: threshold value
    :type threshold: float
    :return: day of crossing threshold
    :rtype: dict
    """
    ndays_year = 365

    # Find days when value above threshold
    is_value_above = daily_value["value"] >= threshold
    is_value_above2 = np.roll(is_value_above, 1)

    # Find days when value crosses threshold
    value_cross_threshold = is_value_above.astype(int) - is_value_above2.astype(int)
    day_cross_up = daily_value["day"][np.where(value_cross_threshold == 1)[0]]
    day_cross_down = daily_value["day"][np.where(value_cross_threshold == -1)[0]]

    # Convert values to 1:365
    day_cross_up = day_cross_up[day_cross_up <= ndays_year]
    day_cross_down = day_cross_down[day_cross_down <= ndays_year]

    day_cross_up = np.sort(np.unique(day_cross_up))
    day_cross_down = np.sort(np.unique(day_cross_down))

    # No crossing == -9999
    day_cross_up = day_cross_up[0] if len(day_cross_up) > 0 else -9999
    day_cross_down = day_cross_down[0] if len(day_cross_down) > 0 else -9999

    return {"doy_cross_up": day_cross_up, "doy_cross_down": day_cross_down}

def get_month_length():
    """
    Standardized length of each month.
    """
    return np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def get_midday(even=True):
    """
    Standardized midday of each month.
    """
    days_in_month = get_month_length()

    # Mid-month days
    midday = np.round(np.cumsum(days_in_month) - days_in_month / 2)

    if even:
        midday = midday.astype(np.int32)
    
    return midday


def calc_doy_wet_month(daily_ppet):
    """
    Calculate the first day of the 120 wettest days period.

    This function finds the first day of the 120-day period with the maximum 
    cumulative sum of the precipitation to potential evapotranspiration (P/PET) ratio.

    :param daily_ppet: Dict of numpy array of interpolated monthly to daily P/PET values.
    :type daily_ppet: dict

    :return: The day of the year (DOY) corresponding to the start of the wettest 120-day period.
    :rtype: int
    """
    doys = np.arange(1, 366)
    
    x = np.array([np.sum(daily_ppet["value"][i:i+120]) for i in range(365-120+1)])
    
    return doys[np.argmax(x)]
