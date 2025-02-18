"""Farmer entity type class of landmanager_farmer_management
"""
import re
import numpy as np
import pandas as pd
import xarray as xr
import pycoupler
from enum import Enum
import datetime
from scipy.interpolate import CubicSpline
from pycoupler.data import LPJmLDataSet

from landmanager.components import management
from landmanager.components import base


class World(management.World):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, **kwargs):
        """
        Initialize the World object by adding the cell to the object.
        """
        super().__init__(**kwargs)  # must be the first line

        # Initiate attributes
        self.calendar = None

        # Initiate the crop set with each crop and calendar
        self.crops = WorldCropSet(world=self)
        # Write the crop calendar to the input data
        self.update_input()

    def update(self, t):
        # call the base class update method
        super().update(t)
        # update the crops
        self.crops.update()
        self.update_input()

    def update_input(self):
        """
        Set the calendar data in LPJmL input for crop activities based on time
        and crop growth stages.
        """

        self._set_input("sdate", self.calendar.sdate, mask_value=-9999)
        self._set_input("crop_phu", self.calendar.crop_phu, mask_value=-9999)


    def _set_input(self, name, value, mask_value=np.nan):
        """
        Set the cell input data for the crop.

        :param name: Input data name.
        :type name: str
        :param value: Input data value.
        :type value: float
        :param mask_value: Mask value for the input data.
        :type mask_value: float
        """
        # Vectorized check against the 'band' column
        mask = xr.where(value == -9999, True, False)

        # Expand mask to match the shape of `values` (assuming dim order: [bands, cells])
        self.input[name].values = self.input[name].where(mask, value).values


class WorldActivity:
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

    def __init__(self, world, **kwargs):
        """
        Initialize the CellActivity object by adding the cell to the object.
        """
        self._world = world

        # basic time variables
        self.ndays_year = 365
        self.midday = get_midday()


    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    @property
    def world(self):
        return self._world

    def get_output(self, output, avg=None):
        """Return the given attribute."""
        if avg in ["monthly", "daily"]:
            return self.world.output[output].mean("time")
        elif avg == "annual" and (
            "month" in self.world.output.harvestc.attrs["units"] or "day" in self.world.output.harvestc.attrs["units"]
        ):
            return self.world.output[output].mean("band")
        else:
            return self.world.output[output]


class WorldCrop(WorldActivity):

    def __init__(self, name, parameters, column_name="cft_name", **kwargs):

        super().__init__(**kwargs)
        self.name = name

        # Extract relevant parameters based on crop name and winter crop classification
        param_filter = (parameters[column_name] == name) & (parameters["winter_crop"] == 0)
        winter_param = parameters[(parameters[column_name] == name) & (parameters["winter_crop"])]

        ncell = len(self.world.grid.cell)

        for param, val in parameters[param_filter].items():
            self[param] = xr.DataArray(
                np.repeat(val, ncell), dims=["cell"],
                coords={"cell": self.world.grid.cell.values},
                name=param
            )
            if not winter_param.empty:
                self[param] = self[param].where(
                    self.is_winter_crop == False,
                    winter_param[param].values
                )

        base_array = self.world.grid.cell

        # Initialize attributes efficiently
        self.sdate = xr.full_like(base_array, -9999, dtype=float)
        self.smonth = xr.full_like(base_array, -9999, dtype=float)
        self.sseason = xr.full_like(base_array, '', dtype=str)

        self.harvest_rule = xr.full_like(base_array, 0, dtype=int)
        self.harvest_rule_name = xr.full_like(base_array, '', dtype=str)

        self.hdate_first = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_maxrp = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_last = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_wetseas = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_temp_base = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_temp_opt = xr.full_like(base_array, -9999, dtype=float)

        self.hdate_rf = xr.full_like(base_array, -9999, dtype=float)
        self.hdate_ir = xr.full_like(base_array, -9999, dtype=float)
        self.hreason_rf = xr.full_like(base_array, '', dtype="<U20")
        self.hreason_ir = xr.full_like(base_array, '', dtype="<U20")

    @property
    def is_winter_crop(self, temperate_cereals_only=True):
        """
        Tests if a given growing season should be classified as winter crop

        :param temperate_cereals_only: If True, only temperate cereals are classified as winter crops
        :type temperate_cereals_only: bool
        :return: Array of booleans indicating winter crop classification for each cell
        :rtype: xr.DataArray
        """
        # Early return for crops that are not temperate cereals when temperate_cereals_only is True
        if temperate_cereals_only and self.name != "temperate cereals":
            return xr.zeros_like(self.world.grid.cell, dtype=bool)

        # Extract cell output once for efficiency
        world_output = self.world.output
        band = f"rainfed {self.name}"
        
        # Get the last sowing and harvest dates for all cells
        sdate = world_output.sdate.sel(band=band).isel(time=-1)
        hdate = world_output.hdate.sel(band=band).isel(time=-1)
        
        # Mask invalid sowing dates
        valid_sdate = sdate >= 0
        
        # Get the latitude of each cell and the lowest temperature
        lat = self.world.grid.lat
        lowest_temp = world_output.temp.min(dim=["time","band"])

        # Calculate the growing period
        grow_period = xr.where(sdate <= hdate, hdate - sdate, 365 + hdate - sdate)

        # Conditions for winter crop classification based on latitude and temperature
        winter_crop = xr.zeros_like(lat, dtype=bool)
        
        northern_cond = (lat > 0) & (sdate + grow_period > 365) & (grow_period >= 150) & (-10 <= lowest_temp) & (lowest_temp <= 7)
        southern_cond = (lat <= 0) & (sdate < 182) & (sdate + grow_period > 182) & (grow_period >= 150) & (-10 <= lowest_temp) & (lowest_temp <= 7)
        
        winter_crop = xr.where(valid_sdate & (northern_cond | southern_cond), True, False)

        return winter_crop

    def calc_sowing_date(self, monthly_climate, seasonality):
        """
        Calculate sowing date (Waha et al., 2012)

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param seasonality: Seasonality classification.
        :type seasonality: str
        """
        lat = self.world.grid.lat

        # Default values
        default_doy = xr.where(lat >= 0, 1, 182)
        default_month = xr.zeros_like(default_doy)

        # Constrain first possible date for winter crop sowing
        earliest_sdate = xr.where(lat >= 0, self.initdate_sdatenh, self.initdate_sdatesh)
        earliest_smonth = doy2month(earliest_sdate)

        # Pre-calculate values used multiple times
        min_temp = monthly_climate["min_temp"]
        daily_temp = monthly_climate["daily_temp"]
        argmin_temp = monthly_climate["argmin_temp"]

        # First day of spring
        firstspringdoy = calc_doy_cross_threshold(daily_temp, self.temp_spring).get("doy_cross_up", -9999)
        firstspringdoy = xr.where(firstspringdoy == -9999, default_doy, firstspringdoy)
        firstspringmonth = doy2month(firstspringdoy)

        # Define masks for different winter types
        warm_winter = (min_temp > self.basetemp_low) & np.isin(seasonality, ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"])
        cold_winter = (min_temp < -10) & np.isin(seasonality, ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"])
        
        # Compute first winter day
        firstwinterdoy = xr.full_like(default_doy, -9999)

        # Warm winter case: sowing 2.5 months before coldest midday
        coldestday = self.midday[argmin_temp]
        firstwinterdoy = xr.where(warm_winter, (coldestday - 75) % 365, firstwinterdoy)

        # Cold winter case: No winter sowing
        firstwinterdoy = xr.where(cold_winter, -9999, firstwinterdoy)

        # Mild winter case: threshold-based
        mild_winter = ~warm_winter & ~cold_winter
        mild_winter_doy = calc_doy_cross_threshold(daily_temp, self.temp_fall).get("doy_cross_down", -9999)
        firstwinterdoy = xr.where(mild_winter, mild_winter_doy, firstwinterdoy)

        # Ensure default values
        firstwinterdoy = xr.where(firstwinterdoy == -9999, default_doy, firstwinterdoy)
        firstwintermonth = doy2month(firstwinterdoy)

        # Determine sowing date based on calculation method
        wtyp_mask = self.calcmethod_sdate == "WTYP_CALC_SDATE"
        no_seasonality = seasonality == "NO_SEASONALITY"
        prec_based = np.isin(seasonality, ["PREC", "PRECTEMP"])

        # Apply conditions
        smonth = xr.where(wtyp_mask & (firstwinterdoy > earliest_sdate) & (firstwintermonth != default_month),
                        firstwintermonth,
                        firstspringmonth)

        sdate = xr.where(wtyp_mask & (firstwinterdoy > earliest_sdate) & (firstwintermonth != default_month),
                        firstwinterdoy,
                        firstspringdoy)

        sseason = xr.where(wtyp_mask & (firstwinterdoy > earliest_sdate) & (firstwintermonth != default_month),
                        "winter",
                        "spring")

        # Adjust for earliest sowing constraint
        smonth = xr.where(wtyp_mask & (firstwinterdoy <= earliest_sdate) & (min_temp > self.temp_fall) & (firstwintermonth != default_month),
                        earliest_smonth,
                        smonth)

        sdate = xr.where(wtyp_mask & (firstwinterdoy <= earliest_sdate) & (min_temp > self.temp_fall) & (firstwintermonth != default_month),
                        earliest_sdate,
                        sdate)

        sseason = xr.where(wtyp_mask & (firstwinterdoy <= earliest_sdate) & (min_temp > self.temp_fall) & (firstwintermonth != default_month),
                        "winter",
                        sseason)

        # Handle NO_SEASONALITY case
        smonth = xr.where(no_seasonality, default_month, smonth)
        sdate = xr.where(no_seasonality, default_doy, sdate)
        sseason = xr.where(no_seasonality, "spring", sseason)

        # Handle PREC/PRECTEMP-based sowing
        sdate_prec = calc_doy_wet_month(monthly_climate["daily_ppet"])
        smonth = xr.where(prec_based, doy2month(sdate_prec), smonth)
        sdate = xr.where(prec_based, sdate_prec, sdate)
        sseason = xr.where(prec_based, "spring", sseason)

        # Assign final values
        self.sdate.values = sdate.values
        self.smonth.values = smonth.values
        self.sseason.values = sseason.values

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
        seasonality_mapping = np.where(
            seasonality == "NO_SEASONALITY", 1, 
            np.where(seasonality == "PREC", 2, 3)
        )
        rule_name_suffix = np.where(
            seasonality == "NO_SEASONALITY", "no-seas",
            np.where(seasonality == "PREC", "prec-seas", "mix-seas")
        )

        max_temp = monthly_climate["max_temp"]

        # Vectorized rule offset computation
        rule_offset = np.select(
            [max_temp <= self.temp_base_rphase, max_temp <= self.temp_opt_rphase],
            # Corresponds to t-low and t-mid
            [0, 3],
            # t-high
            default=6,
        )

        rule_prefixes = np.select(
            [max_temp <= self.temp_base_rphase, max_temp <= self.temp_opt_rphase],
            ["t-low", "t-mid"],
            default="t-high",
        )

        # Compute final rule values
        self.harvest_rule = seasonality_mapping + rule_offset
        self.harvest_rule_name = np.core.defchararray.add(rule_prefixes, "_" + rule_name_suffix)

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
        hdate_last = self.sdate + np.where(self.sseason == "winter", self.max_growingseason_wt, self.max_growingseason_st)

        # End of wet season
        daily_ppet = monthly_climate["daily_ppet"]
        daily_ppet_diff = monthly_climate["daily_ppet_diff"]
        doy_wet1 = calc_doy_cross_threshold(daily_ppet, self.ppet_ratio)["doy_cross_down"]
        doy_wet2 = calc_doy_cross_threshold(daily_ppet_diff, self.ppet_ratio_diff)["doy_cross_down"]

        # Adjust wet season dates if they occur before sowing date
        doy_wet_vec = np.array([doy_wet1, doy_wet2])
        doy_wet_vec[(doy_wet_vec < self.sdate.values) & (doy_wet_vec != -9999)] += self.ndays_year

        masked_doy_wet_vec = np.ma.masked_equal(doy_wet_vec, -9999)
        doy_wet_first = np.ma.min(masked_doy_wet_vec, axis=0).filled(-9999)

        hdate_wetseas = np.where(
            doy_wet1 == -9999,
            np.where(
                monthly_climate["min_ppet"] >= self.ppet_min,
                hdate_last,
                hdate_first
            ),
            doy_wet_first + self.rphase_duration
        )

        # Warmest day of the year
        warmest_day = self.midday[monthly_climate["argmax_temp"]]
        hdate_temp_base = np.where(
            self.sseason == "winter",
            warmest_day,
            warmest_day + self.rphase_duration
        )

        # First and last hot day
        daily_temp = monthly_climate["daily_temp"]
        doy_exceed_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)["doy_cross_up"]
        doy_below_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)["doy_cross_down"]

        # Adjust for year boundaries
        doy_exceed_opt_rp = np.where((doy_exceed_opt_rp < self.sdate) & (doy_exceed_opt_rp != -9999), doy_exceed_opt_rp + self.ndays_year, doy_exceed_opt_rp)
        doy_below_opt_rp = np.where((doy_below_opt_rp < self.sdate) & (doy_below_opt_rp != -9999), doy_below_opt_rp + self.ndays_year, doy_below_opt_rp)

        # Winter type: First hot day; Spring type: Last hot day
        doy_opt_rp = np.where(self.sseason == "winter", doy_exceed_opt_rp, doy_below_opt_rp)
        hdate_temp_opt = np.where(
            doy_opt_rp == -9999,
            hdate_maxrp,
            np.where(
                self.sseason != "winter",
                doy_opt_rp + self.rphase_duration,
                doy_opt_rp
            )
        )

        # Store results, adjusting for next-year cases
        self.hdate_first.values = hdate_first.values
        self.hdate_maxrp.values = hdate_maxrp.values
        self.hdate_last.values = hdate_last.values
        self.hdate_wetseas.values = xr.where(
            hdate_wetseas < self.sdate,
            hdate_wetseas + self.ndays_year,
            hdate_wetseas
        ).values
        self.hdate_temp_base.values = xr.where(
            hdate_temp_base < self.sdate,
            hdate_temp_base + self.ndays_year,
            hdate_temp_base
        ).values
        self.hdate_temp_opt.values = xr.where(
            hdate_temp_opt < self.sdate,
            hdate_temp_opt + self.ndays_year,
            hdate_temp_opt
        ).values

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
        hdate = np.zeros_like(max_temp, dtype=int)
        hreason = np.empty_like(max_temp, dtype="<U20")

        hdate_rf = np.select(
            [
                (seasonality == "NO_SEASONALITY") & (self.harvest_rule == 1),
                (seasonality == "NO_SEASONALITY"),
                (seasonality == "PREC") & (self.harvest_rule == 2),
                (seasonality == "PREC"),
                (self.sseason == "winter") & (self.harvest_rule == 3),
                (self.sseason == "winter") & (self.harvest_rule == 6),
                (self.sseason == "winter"),
                (self.harvest_rule == 3),
                (self.harvest_rule == 6) & (seasonality == "PRECTEMP"),
                (self.harvest_rule == 6) & (self.smonth == 0) & (max_temp < self.temp_spring),
                (self.harvest_rule == 6),
                (seasonality == "PRECTEMP")
            ],
            [
                # NO SEASONALITY
                self.hdate_first,
                self.hdate_maxrp,
                # PREC SEASONALITY
                self.hdate_first,
                np.minimum(np.maximum(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp),
                # WINTER SOWING SEASON
                self.hdate_first,
                np.where(
                    (self.smonth == 0) & (max_temp < self.temp_fall),
                    self.hdate_temp_opt,
                    np.minimum(np.maximum(self.hdate_first, self.hdate_temp_base), self.hdate_last)
                ),
                np.minimum(np.maximum(self.hdate_first, self.hdate_temp_opt), self.hdate_last),
                # HARVEST RULE 3
                self.hdate_first,
                # HARVEST RULE 6
                np.minimum(np.maximum(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp),
                self.hdate_first,
                np.minimum.reduce([
                    np.maximum(self.hdate_first, self.hdate_temp_base),
                    np.maximum(self.hdate_first, self.hdate_wetseas),
                    self.hdate_last
                ]),
                # PRECTEMP SEASONALITY
                np.minimum(np.maximum(self.hdate_first, self.hdate_wetseas), self.hdate_maxrp)
            ],
            # ELSE
            default=np.minimum.reduce([
                np.maximum(self.hdate_first, self.hdate_temp_opt),
                np.maximum(self.hdate_first, self.hdate_wetseas),
                self.hdate_last
            ])
        )

        hdate_ir = np.select(
            [
                (seasonality == "NO_SEASONALITY") & (self.harvest_rule == 1),
                (seasonality == "NO_SEASONALITY"),
                (seasonality == "PREC") & (self.harvest_rule == 2),
                (seasonality == "PREC"),
                (self.sseason == "winter") & (self.harvest_rule == 3),
                (self.sseason == "winter") & (self.harvest_rule == 6),
                (self.sseason == "winter"),
                (self.harvest_rule == 3),
                (self.harvest_rule == 6) & (seasonality == "PRECTEMP"),
                (self.harvest_rule == 6) & (self.smonth == 0) & (max_temp < self.temp_spring),
                (self.harvest_rule == 6),
                (seasonality == "PRECTEMP")
            ],
            [
                # NO SEASONALITY
                self.hdate_first,
                self.hdate_maxrp,
                # PREC SEASONALITY
                self.hdate_first,
                self.hdate_maxrp,
                # WINTER SOWING SEASON
                self.hdate_first,
                np.where(
                    (self.smonth == 0) & (max_temp < self.temp_fall),
                    self.hdate_temp_opt,
                    np.minimum(np.maximum(self.hdate_first, self.hdate_temp_base), self.hdate_last)
                ),
                np.minimum(np.maximum(self.hdate_first, self.hdate_temp_opt), self.hdate_last),
                # HARVEST RULE 3
                self.hdate_first,
                # HARVEST RULE 6
                self.hdate_maxrp,
                self.hdate_first,
                np.minimum(np.maximum(self.hdate_first, self.hdate_temp_base), self.hdate_last),
                # PRECTEMP SEASONALITY
                self.hdate_maxrp
            ],
            # ELSE
            default=np.minimum(np.maximum(self.hdate_first, self.hdate_temp_opt), self.hdate_last)
        )

        hreason_rf = np.select(
            [
                hdate_rf == self.hdate_first,
                hdate_rf == self.hdate_maxrp,
                hdate_rf == self.hdate_wetseas,
                hdate_rf == self.hdate_temp_opt,
                hdate_rf == self.hdate_temp_base,
                hdate_rf == self.hdate_last,
            ],
            [
                "hdate_first",
                "hdate_maxrp",
                "hdate_wetseas",
                "hdate_temp_opt",
                "hdate_temp_base",
                "hdate_last"
            ],
            default=hreason
        )

        hreason_ir = np.select(
            [
                hdate_ir == self.hdate_first,
                hdate_ir == self.hdate_maxrp,
                hdate_ir == self.hdate_wetseas,
                hdate_ir == self.hdate_temp_opt,
                hdate_ir == self.hdate_temp_base,
                hdate_ir == self.hdate_last,
            ],
            [
                "hdate_first",
                "hdate_maxrp",
                "hdate_wetseas",
                "hdate_temp_opt",
                "hdate_temp_base",
                "hdate_last"
            ],
            default=hreason
        )

        # Vectorized adjustment based on year length
        hdate_rf = np.where(hdate_rf <= self.ndays_year, hdate_rf, hdate_rf - self.ndays_year)
        hdate_ir = np.where(hdate_ir <= self.ndays_year, hdate_ir, hdate_ir - self.ndays_year)

        # Assign the results to self attributes
        self.hdate_rf.values = hdate_rf
        self.hdate_ir.values = hdate_ir
        self.hreason_rf.values = hreason_rf
        self.hreason_ir.values = hreason_ir

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
        hdate = np.where(self.sdate < hdate, hdate, hdate + 365)

        husum = xr.full_like(self.sdate, 0, dtype=int)

        # Duplicate daily temperature to allow cross-year growth
        temp_values = xr.DataArray(
            xr.concat([daily_temp["value"], daily_temp["value"]], dim="day"),
            coords={"cell": self.sdate.cell, "day": np.arange(1, 2*365+1)}
        )
        day_values = xr.DataArray(
            np.arange(1, 2*365+1),
            dims=["day"]
        ).broadcast_like(temp_values)

        grow_mask = (day_values.values >= self.sdate.values[:, None]) & (day_values.values < hdate[:, None])
        # Compute Effective Thermal Units (teff) using vectorized operations
        if phen_model == "t":
            teff = np.maximum(temp_values - self.basetemp_low, 0) * grow_mask
        elif phen_model == "tv" and vern_factor is not None:
            teff = np.maximum(temp_values - self.basetemp_low, 0) * np.array(vern_factor) * grow_mask
        elif phen_model == "tv" and vern_factor is None:
            raise ValueError("Error: vern_factor not provided!")
        else:
            raise ValueError("Error: phen_model not declared!")

        # Compute total PHU requirement efficiently
        husum.values = np.sum(teff, axis=1).astype(int)

        # Negate if using "tv" model
        if phen_model == "tv":
            husum *= -1
        else:
            husum = husum.where(self.winter_crop == 1, -husum)

        return husum


class WorldCropSet(WorldActivity):

    def __init__(self, crop_param_file="crop_calendar/crop_parameters.csv", **kwargs):
        super().__init__(**kwargs)
        # actual crop landuse
        self.parameters = base.load_csv(crop_param_file)
        # get all cultivated crops
        self.update()

    @property
    def actual_landuse(self):
        """
        Return the actual land use of the cell.
        """
        return self.world.input.landuse.where(self.world.input.landuse.sum(dim="cell") > 0).dropna(dim="band")

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
            WorldCropSet.irrigation_pattern.sub("", band).strip()
            for band in self.bands  # noqa
        }
        self.calendars = {
            crop for crop in self.names if crop in self.parameters.cft_name.tolist()
        }

        for crop in self.parameters.cft_name:
            if crop in self.names and not hasattr(self, crop):
                self[crop] = WorldCrop(
                    name=crop,
                    parameters=self.parameters,
                    world=self.world,
                )
            elif crop not in self.names and hasattr(self, crop):
                del self[crop]

    def calc_seasonality(self, monthly_climate, temp_min=10):
        """
        Calculate the seasonality type for multiple spatial units.

        Seasonality calculation based on average monthly climate as described
        in Waha et al. 2012.
        :param monthly_climate: Dict of arrays of monthly climate variables (shape: (21, 12)).
        :type monthly_climate: dict
        :param temp_min: Threshold of temperature of the coldest month (°C). Default is 10°C.
        :return: Array of seasonality types for each spatial unit.
        :rtype: np.ndarray (shape: (21,))
        """
        var_coef_prec = calc_var_coeff(monthly_climate["prec"])  # Shape (21,)
        var_coef_temp = calc_var_coeff(deg2k(monthly_climate["temp"]))  # Shape (21,)
        min_temp = np.min(monthly_climate["temp"], axis=1)  # Shape (21,)

        # Create an array to store seasonality types
        seasonality = np.full(var_coef_prec.shape, "TEMP", dtype=object)

        # Apply conditions using NumPy vectorized operations
        no_seasonality = (var_coef_prec <= 0.4) & (var_coef_temp <= 0.01)
        prec_only = (var_coef_prec > 0.4) & (var_coef_temp <= 0.01)
        prec_temp = (var_coef_prec > 0.4) & (var_coef_temp > 0.01) & (min_temp > temp_min)
        temp_prec = (var_coef_prec > 0.4) & (var_coef_temp > 0.01) & (min_temp <= temp_min)

        # Assign seasonality categories
        seasonality[no_seasonality] = "NO_SEASONALITY"
        seasonality[prec_only] = "PREC"
        seasonality[prec_temp] = "PRECTEMP"
        seasonality[temp_prec] = "TEMPPREC"

        return seasonality

    def get_monthly_climate(self):
        """
        Return monthly climate data for the cell.
        """
        # Average temperature of the cell
        temp = self.get_output("temp", avg="monthly")
        daily_temp = interpolate_monthly_to_daily(temp)

        min_temp = temp.min(dim="band")
        max_temp = temp.max(dim="band")
        argmin_temp = temp.argmin(dim="band")
        argmax_temp = temp.argmax(dim="band")

        # average temperature of the cell
        prec = self.get_output("prec", avg="monthly")

        # Fix unrealistic low PET values
        pre_pet = np.maximum(self.get_output("pet"), 1e-1)

        ppet = (self.get_output("prec") / pre_pet).mean("time", skipna=True)

        daily_ppet = interpolate_monthly_to_daily(ppet)

        min_ppet = ppet.min(dim="band")

        # Average potential evaporation of the cell
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

    def calc_calendar(self):
        """Calculate the crop calendar for each crop in calendars."""
        monthly_climate = self.get_monthly_climate()
        seasonality = self.calc_seasonality(monthly_climate)

        for crop in self.calendars:
            # calculate sowing date
            self[crop].calc_sowing_date(monthly_climate, seasonality)

            # calculate harvest rule
            self[crop].calc_harvest_rule(monthly_climate, seasonality)

            # calculate different options for harvesting
            self[crop].calc_harvest_options(monthly_climate)

            # calculate harvest date, choose the best option
            self[crop].calc_harvest_date(monthly_climate, seasonality)

            self[crop].phu_rf = self[crop].calc_phu(monthly_climate["daily_temp"], self[crop].hdate_rf)
            self[crop].phu_ir = self[crop].calc_phu(monthly_climate["daily_temp"], self[crop].hdate_ir)

        self.update_calendar()
    
    def update_calendar(self):

        base_array = self.world.input.sdate
        base_array.time.values[0] = np.datetime64(
            f"{self.world.model.lpjml.sim_year}-12-31"
        )

        sdate = xr.full_like(base_array, -9999, dtype=int)
        sdate.name = "sdate"

        hdate = xr.full_like(base_array, -9999, dtype=int)
        hdate.name = "hdate"

        hreason = xr.full_like(base_array, "", dtype="<U20")
        hreason.name = "hreason"

        crop_phu = xr.full_like(base_array, -9999, dtype=int)
        crop_phu.name = "crop_phu"

        for crop in self.calendars:
            sdate.loc[{"band": [f"rainfed {crop}", f"irrigated {crop}"]}] = self[crop].sdate

            hdate.loc[{"band": f"rainfed {crop}"}] = self[crop].hdate_rf
            hdate.loc[{"band": f"irrigated {crop}"}] = self[crop].hdate_ir

            hreason.loc[{"band": f"rainfed {crop}"}] = self[crop].hreason_rf
            hreason.loc[{"band": f"irrigated {crop}"}] = self[crop].hreason_ir

            crop_phu.loc[{"band": f"rainfed {crop}"}] = self[crop].phu_rf
            crop_phu.loc[{"band": f"irrigated {crop}"}] = self[crop].phu_ir


        if self.world.calendar is None:
            self.world.calendar = LPJmLDataSet(
                {
                    "sdate": sdate,
                    "hdate": hdate,
                    "hreason": hreason,
                    "crop_phu": crop_phu
                }
            )
        else:
            self.world.calendar.sdate.values = sdate.values
            self.world.calendar.hdate.values = hdate.values
            self.world.calendar.hreason.values = hreason.values
            self.world.calendar.crop_phu.values = crop_phu.values

    def update(self):
        """
        Update the WorldCropSet object.
        """
        self.update_landuse()
        self.calc_calendar()


def calc_var_coeff(x, axis=0):
    """
    Calculate the coefficient of variation (CV) for a given array x along the specified axis.
    If the mean of x is 0, return 0 to avoid division by zero.

    :param x: array-like, input data
    :type x: np.ndarray
    :param axis: int, axis along which to compute the coefficient of variation
    :type axis: int, optional
    :return: coefficient of variation
    :rtype: np.ndarray
    """
    mean_x = np.nanmean(x, axis=1)
    std_x = np.nanstd(x, axis=1)

    return np.where(mean_x == 0, 0, std_x / mean_x)


def deg2k(x):
    """
    Convert temperature from degree Celsius to Kelvin.

    :param x: temperature in degree Celsius
    :type x: float
    :return: temperature in Kelvin
    :rtype: float

    """
    return x + 273.15


def doy2month(doy, year=2015):
    """
    Convert day of year to month for array-like input.

    :param doy: Day(s) of the year (scalar or array-like).
    :type doy: int, list, np.ndarray, or xarray.DataArray
    :param year: Year(s) (scalar or array-like, defaults to 2015).
    :type year: int, list, np.ndarray, or xarray.DataArray
    :return: Month(s) corresponding to the input day(s).
    :rtype: np.ndarray (or xarray.DataArray if input is xarray)
    """
    doy = np.asarray(doy)  # Ensure input is an array
    year = np.asarray(year) if np.ndim(year) > 0 else np.full_like(doy, year)  # Broadcast if needed

    dates = pd.to_datetime(year.astype(str) + doy.astype(str), format="%Y%j")
    return dates.month


def interpolate_monthly_to_daily(monthly_value):
    """
    Interpolates monthly values to daily values using a periodic cubic spline.

    :param monthly_value: Array-like (shape (N, 12)) of monthly values (e.g., temperature).
    :return: Dictionary with "day" (day of year, length 365) and "value"
        (interpolated values, shape (N, 365)).
    """
    midday = get_midday()
    ndays_year = 365
    day = np.arange(1, ndays_year + 1)  # Days of the year
    
    # Ensure input is an xarray DataArray
    monthly_value = monthly_value.transpose("cell", "band")  # Ensure correct order

    # Extend along the month axis for periodic boundary condition
    extended_x = np.concatenate([midday, [midday[0] + ndays_year]])  # (13,)
    extended_y = xr.concat([monthly_value, monthly_value.isel(band=0)], dim="band")  # (cell, 13)

    # Vectorized interpolation
    spline = CubicSpline(extended_x, extended_y, axis=1, bc_type="periodic")  # Axis 1 = band
    interpolated_values = spline(day)  # Result: (cell, day=365)

    # Return as an xarray.Dataset for easy handling
    return xr.Dataset({
        "day": ("day", day),
        "value": (("cell", "day"),
        interpolated_values)
    })

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
    # Ensure threshold has shape (n_cells, 1) for broadcasting
    # threshold = threshold.expand_dims("time")  # Reshape to (21, 1)

    # Find days when value is above threshold
    is_value_above = daily_value["value"] >= threshold
    is_value_above_shifted = np.roll(is_value_above, 1, axis=1)

    # Find days when value crosses threshold
    value_cross_threshold = is_value_above.astype(int) - is_value_above_shifted.astype(int)

    # Find first crossing up/down per cell
    day_cross_up = np.where(value_cross_threshold == 1, daily_value["day"], np.inf)
    day_cross_down = np.where(value_cross_threshold == -1, daily_value["day"], np.inf)

    # Get minimum crossing day per cell
    day_cross_up = np.min(day_cross_up, axis=1)
    day_cross_down = np.min(day_cross_down, axis=1)

    # Convert inf to -9999 (no crossing)
    day_cross_up[day_cross_up == np.inf] = -9999
    day_cross_down[day_cross_down == np.inf] = -9999

    return {"doy_cross_up": day_cross_up.astype(int), "doy_cross_down": day_cross_down.astype(int)}


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
