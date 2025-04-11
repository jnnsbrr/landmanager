"""Farmer entity type class of landmanager_farmer_management"""

import re
import numpy as np
import pandas as pd
import xarray as xr
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

        # Expand mask to match the shape of `values`
        #   (assuming dim order: [bands, cells])
        self.input[name].values[:] = self.input[name].where(mask, value).values[:]  # noqa


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

    irrigation_systems = set(irrigation_systems_short + irrigation_systems_long)  # noqa

    irrigation_pattern = re.compile(
        r"^(?:" + "|".join(map(re.escape, irrigation_systems)) + r")\s*"
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
            "month" in self.world.output.harvestc.attrs["units"]
            or "day" in self.world.output.harvestc.attrs["units"]
        ):
            return self.world.output[output].mean("band")
        else:
            return self.world.output[output]


class WorldCrop(WorldActivity):

    def __init__(self, name, parameters, column_name="cft_name", **kwargs):

        super().__init__(**kwargs)
        self.name = name

        # Extract relevant parameters based on crop name and winter crop
        #   classification
        self.update_parameters(is_winter_crop=self.is_winter_crop)

    def update_parameters(self, is_winter_crop):
        """
        Update crop-specific parameters based on winter crop classification.

        :param is_winter_crop: Boolean array indicating winter crop classification.  # noqa
        :type is_winter_crop: xr.DataArray
        """
        # Extract relevant parameters based on crop name and winter crop classification  # noqa
        param_filter = (self.world.crops.parameters["cft_name"] == self.name) & (  # noqa
            self.world.crops.parameters["winter_crop"] == 0
        )
        winter_param = self.world.crops.parameters[
            (self.world.crops.parameters["cft_name"] == self.name)
            & (self.world.crops.parameters["winter_crop"] == 1)
        ]

        ncell = len(self.world.grid.cell)

        # Update each parameter dynamically
        for param, val in self.world.crops.parameters[param_filter].items():
            # Create or update the parameter as an xarray DataArray
            self[param] = xr.DataArray(
                np.repeat(val, ncell),
                dims=["cell"],
                coords={"cell": self.world.grid.cell.values},
                name=param,
            )
            # Apply winter crop parameters where applicable
            if not winter_param.empty:
                self[param] = self[param].where(
                    ~is_winter_crop,  # Use spring parameters if not winter crop  # noqa
                    winter_param[param].values,
                )

    @property
    def is_winter_crop(self, temperate_cereals_only=True):
        """
        Tests if a given growing season should be classified as winter crop

        :param temperate_cereals_only: If True, only temperate cereals are
            classified as winter crops
        :type temperate_cereals_only: bool
        :return: Array of booleans indicating winter crop classification for
            each cell
        :rtype: xr.DataArray
        """
        # Early return for crops that are not temperate cereals when
        #   temperate_cereals_only is True
        if temperate_cereals_only and self.name != "temperate cereals":
            return xr.zeros_like(self.world.grid.cell, dtype=bool)

        # Extract cell output once for efficiency
        world_output = self.world.output
        band = f"rainfed {self.name}"

        # Get the last sowing and harvest dates for all cells
        # sdate = world_output.sdate.sel(band=band).isel(time=-1)
        # hdate = world_output.hdate.sel(band=band).isel(time=-1)

        sdate = world_output.sdate.mean(dim=["time"]).sel(band=band)
        hdate = world_output.hdate.mean(dim=["time"]).sel(band=band)

        # Mask invalid sowing dates
        valid_sdate = sdate >= 0

        # Get the latitude of each cell and the lowest temperature
        lat = self.world.grid.lat

        lowest_temp = world_output.temp.mean(dim=["time"]).min(dim=["band"])

        # Calculate the growing period
        grow_period = xr.where(sdate <= hdate, hdate - sdate, 365 + hdate - sdate)

        # Conditions for winter crop classification based on latitude and
        #   temperature
        winter_crop = xr.zeros_like(lat, dtype=bool)

        # filter northern and southern hemisphere cells
        northern_cond = (
            (lat > 0)
            & (sdate + grow_period > 365)
            & (grow_period >= 150)
            & (-10 <= lowest_temp)
            & (lowest_temp <= 7)
        )
        southern_cond = (
            (lat <= 0)
            & (sdate < 182)
            & (sdate + grow_period > 182)
            & (grow_period >= 150)
            & (-10 <= lowest_temp)
            & (lowest_temp <= 7)
        )
        winter_crop = xr.where(
            valid_sdate & (northern_cond | southern_cond), True, False
        )  # noqa

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
        earliest_sdate = xr.where(
            lat >= 0, self.initdate_sdatenh, self.initdate_sdatesh
        )  # noqa
        earliest_smonth = doy2month(earliest_sdate)

        # Pre-calculate values used multiple times
        min_temp = monthly_climate["min_temp"]
        daily_temp = monthly_climate["daily_temp"]
        argmin_temp = monthly_climate["argmin_temp"]

        # First day of spring
        firstspringdoy = calc_doy_cross_threshold(daily_temp, self.temp_spring).get(  # noqa
            "doy_cross_up", -9999
        )
        firstspringdoy = xr.where(firstspringdoy == -9999, default_doy, firstspringdoy)  # noqa
        firstspringmonth = doy2month(firstspringdoy)

        # Define masks for different winter types
        warm_winter = (min_temp > self.basetemp_low) & np.isin(
            seasonality, ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]
        )
        cold_winter = (min_temp < -10) & np.isin(
            seasonality, ["TEMP", "TEMPPREC", "PRECTEMP", "PREC"]
        )

        # Compute first winter day
        firstwinterdoy = xr.full_like(default_doy, -9999)

        # Warm winter case: sowing 2.5 months before coldest midday
        coldestday = self.midday[argmin_temp]
        firstwinterdoy = xr.where(warm_winter, (coldestday - 75) % 365, firstwinterdoy)  # noqa

        # Cold winter case: No winter sowing
        firstwinterdoy = xr.where(cold_winter, -9999, firstwinterdoy)

        # Mild winter case: threshold-based
        mild_winter = ~warm_winter & ~cold_winter
        mild_winter_doy = calc_doy_cross_threshold(daily_temp, self.temp_fall).get(  # noqa
            "doy_cross_down", -9999
        )
        firstwinterdoy = xr.where(mild_winter, mild_winter_doy, firstwinterdoy)

        # Ensure default values
        firstwinterdoy = xr.where(firstwinterdoy == -9999, default_doy, firstwinterdoy)  # noqa
        firstwintermonth = doy2month(firstwinterdoy)

        # Determine sowing date based on calculation method
        wtyp_mask = self.calcmethod_sdate == "WTYP_CALC_SDATE"
        no_seasonality = seasonality == "NO_SEASONALITY"
        prec_based = np.isin(seasonality, ["PREC", "PRECTEMP"])

        # Apply conditions
        smonth = xr.where(
            wtyp_mask
            & (firstwinterdoy > earliest_sdate)
            & (firstwintermonth != default_month),
            firstwintermonth,
            firstspringmonth,
        )

        sdate = xr.where(
            wtyp_mask
            & (firstwinterdoy > earliest_sdate)
            & (firstwintermonth != default_month),
            firstwinterdoy,
            firstspringdoy,
        )

        sseason = xr.where(
            wtyp_mask
            & (firstwinterdoy > earliest_sdate)
            & (firstwintermonth != default_month),
            "winter",
            "spring",
        )

        # Adjust for earliest sowing constraint
        smonth = xr.where(
            wtyp_mask
            & (firstwinterdoy <= earliest_sdate)
            & (min_temp > self.temp_fall)
            & (firstwintermonth != default_month),
            earliest_smonth,
            smonth,
        )

        sdate = xr.where(
            wtyp_mask
            & (firstwinterdoy <= earliest_sdate)
            & (min_temp > self.temp_fall)
            & (firstwintermonth != default_month),
            earliest_sdate,
            sdate,
        )

        sseason = xr.where(
            wtyp_mask
            & (firstwinterdoy <= earliest_sdate)
            & (min_temp > self.temp_fall)
            & (firstwintermonth != default_month),
            "winter",
            sseason,
        )

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
        return sdate, smonth, sseason

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
        seasonality_mapping = xr.where(
            seasonality == "NO_SEASONALITY",
            1,
            np.where(seasonality == "PREC", 2, 3),
        )
        rule_name_suffix = xr.where(
            seasonality == "NO_SEASONALITY",
            "no-seas",
            np.where(seasonality == "PREC", "prec-seas", "mix-seas"),
        )

        max_temp = monthly_climate["max_temp"]

        # Vectorized rule offset computation
        rule_offset = np.select(
            [
                max_temp <= self.temp_base_rphase,
                max_temp <= self.temp_opt_rphase,
            ],
            # Corresponds to t-low and t-mid
            [0, 3],
            # t-high
            default=6,
        )

        rule_prefixes = np.select(
            [
                max_temp <= self.temp_base_rphase,
                max_temp <= self.temp_opt_rphase,
            ],
            ["t-low", "t-mid"],
            default="t-high",
        )

        # Compute final rule values
        harvest_rule = seasonality_mapping + rule_offset
        harvest_rule_name = np.core.defchararray.add(
            rule_prefixes, "_" + rule_name_suffix
        )  # noqa

        return harvest_rule, harvest_rule_name

    def calc_harvest_options(self, monthly_climate, sdate, sseason):
        """
        Calculate harvest date vector.

        :param sdate: Sowing date (DOY).
        :type sdate: int
        :param sseason: Sowing season.
        :type sseason: xr.DataArray of type str
        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        """

        # Shortest cycle: crop lower biological limit
        hdate_first = sdate + self.min_growingseason
        # Medium cycle: best trade-off vegetative and reproductive growth
        hdate_maxrp = sdate + self.maxrp_growingseason
        # Longest cycle: crop upper biological limit
        hdate_last = sdate + np.where(
            sseason == "winter",
            self.max_growingseason_wt,
            self.max_growingseason_st,
        )

        # End of wet season
        daily_ppet = monthly_climate["daily_ppet"]
        daily_ppet_diff = monthly_climate["daily_ppet_diff"]
        doy_wet1 = calc_doy_cross_threshold(daily_ppet, self.ppet_ratio)[
            "doy_cross_down"
        ]  # noqa
        doy_wet2 = calc_doy_cross_threshold(daily_ppet_diff, self.ppet_ratio_diff)[  # noqa
            "doy_cross_down"
        ]

        # Adjust wet season dates if they occur before sowing date
        doy_wet_vec = np.array([doy_wet1, doy_wet2])
        doy_wet_vec[
            (doy_wet_vec < sdate.values) & (doy_wet_vec != -9999)
        ] += self.ndays_year  # noqa

        masked_doy_wet_vec = np.ma.masked_equal(doy_wet_vec, -9999)
        doy_wet_first = np.ma.min(masked_doy_wet_vec, axis=0).filled(-9999)

        hdate_wetseas = np.where(
            doy_wet1 == -9999,
            np.where(
                monthly_climate["min_ppet"] >= self.ppet_min,
                hdate_last,
                hdate_first,
            ),
            doy_wet_first + self.rphase_duration,
        )

        # Warmest day of the year
        warmest_day = self.midday[monthly_climate["argmax_temp"]]
        hdate_temp_base = np.where(
            sseason == "winter",
            warmest_day,
            warmest_day + self.rphase_duration,
        )

        # First and last hot day
        daily_temp = monthly_climate["daily_temp"]
        doy_exceed_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)[  # noqa
            "doy_cross_up"
        ]
        doy_below_opt_rp = calc_doy_cross_threshold(daily_temp, self.temp_opt_rphase)[  # noqa
            "doy_cross_down"
        ]

        # Adjust for year boundaries
        doy_exceed_opt_rp = np.where(
            (doy_exceed_opt_rp < sdate) & (doy_exceed_opt_rp != -9999),
            doy_exceed_opt_rp + self.ndays_year,
            doy_exceed_opt_rp,
        )
        doy_below_opt_rp = np.where(
            (doy_below_opt_rp < sdate) & (doy_below_opt_rp != -9999),
            doy_below_opt_rp + self.ndays_year,
            doy_below_opt_rp,
        )

        # Winter type: First hot day; Spring type: Last hot day
        doy_opt_rp = np.where(sseason == "winter", doy_exceed_opt_rp, doy_below_opt_rp)  # noqa
        hdate_temp_opt = np.where(
            doy_opt_rp == -9999,
            hdate_maxrp,
            np.where(
                sseason != "winter",
                doy_opt_rp + self.rphase_duration,
                doy_opt_rp,
            ),
        )

        hdate_wetseas = xr.where(
            hdate_wetseas < sdate,
            hdate_wetseas + self.ndays_year,
            hdate_wetseas,
        )
        hdate_temp_base = xr.where(
            hdate_temp_base < sdate,
            hdate_temp_base + self.ndays_year,
            hdate_temp_base,
        )
        hdate_temp_opt = xr.where(
            hdate_temp_opt < sdate,
            hdate_temp_opt + self.ndays_year,
            hdate_temp_opt,
        )

        return (
            hdate_first,
            hdate_maxrp,
            hdate_last,
            hdate_wetseas,
            hdate_temp_base,
            hdate_temp_opt,
        )

    def calc_harvest_date(
        self,
        monthly_climate,
        seasonality,
        smonth,
        sseason,
        harvest_rule,
        hdate_first,
        hdate_maxrp,
        hdate_last,
        hdate_wetseas,
        hdate_temp_base,
        hdate_temp_opt,
    ):
        """
        Calculate harvest date (Minoli et al., 2019).

        Rule-based estimation of the end of the crop growing period (date of
        physiological maturity), here called harvest date for simplicity.
        The assumption behind these rules is that farmers select growing
        seasons based on the mean climatic characteristics of the location in
        which they operate and on the physiological limitations (base and
        optimum temperatures for reproductive growth; sensitivity to terminal
        water stress) of the respective crop species.

        :param monthly_climate: Dict of arrays of monthly temperatures.
        :type monthly_climate: dict
        :param seasonality: Seasonality classification.
        :type seasonality: str
        :param smonth: Sowing month.
        :type smonth: xr.DataArray of type int
        :param sseason: Sowing season.
        :type sseason: xr.DataArray of type str
        :param harvest_rule: Harvest rule classification.
        :type harvest_rule: xr.DataArray of type int
        :param hdate_first: First possible harvest date.
        :type hdate_first: xr.DataArray of type int
        :param hdate_maxrp: Medium cycle harvest date.
        :type hdate_maxrp: xr.DataArray of type int
        :param hdate_last: Last possible harvest date.
        :type hdate_last: xr.DataArray of type int
        :param hdate_wetseas: End of wet season.
        :type hdate_wetseas: xr.DataArray of type int
        :param hdate_temp_base: First hot day.
        :type hdate_temp_base: xr.DataArray of type int
        :param hdate_temp_opt: Last hot day.
        :type hdate_temp_opt: xr.DataArray of type int
        """

        # Precompute max temperature once
        max_temp = monthly_climate["max_temp"]
        hdate_rf = hdate_ir = xr.full_like(self.world.grid.cell, 0, dtype=int)
        hreason_rf = hreason_ir = xr.full_like(self.world.grid.cell, "", dtype="<U20")  # noqa

        hdate_rf.values = np.select(
            [
                (seasonality == "NO_SEASONALITY") & (harvest_rule == 1),
                (seasonality == "NO_SEASONALITY"),
                (seasonality == "PREC") & (harvest_rule == 2),
                (seasonality == "PREC"),
                (sseason == "winter") & (harvest_rule == 3),
                (sseason == "winter") & (harvest_rule == 6),
                (sseason == "winter"),
                (harvest_rule == 3),
                (harvest_rule == 6) & (seasonality == "PRECTEMP"),
                (harvest_rule == 6) & (smonth == 0) & (max_temp < self.temp_spring),  # noqa
                (harvest_rule == 6),
                (seasonality == "PRECTEMP"),
            ],
            [
                # NO SEASONALITY
                hdate_first,
                hdate_maxrp,
                # PREC SEASONALITY
                hdate_first,
                np.minimum(np.maximum(hdate_first, hdate_wetseas), hdate_maxrp),  # noqa
                # WINTER SOWING SEASON
                hdate_first,
                np.where(
                    (smonth == 0) & (max_temp < self.temp_fall),
                    hdate_temp_opt,
                    np.minimum(np.maximum(hdate_first, hdate_temp_base), hdate_last),  # noqa
                ),
                np.minimum(np.maximum(hdate_first, hdate_temp_opt), hdate_last),  # noqa
                # HARVEST RULE 3
                hdate_first,
                # HARVEST RULE 6
                np.minimum(np.maximum(hdate_first, hdate_wetseas), hdate_maxrp),  # noqa
                hdate_first,
                np.minimum.reduce(
                    [
                        np.maximum(hdate_first, hdate_temp_base),
                        np.maximum(hdate_first, hdate_wetseas),
                        hdate_last,
                    ]
                ),
                # PRECTEMP SEASONALITY
                np.minimum(np.maximum(hdate_first, hdate_wetseas), hdate_maxrp),  # noqa
            ],
            # ELSE
            default=np.minimum.reduce(
                [
                    np.maximum(hdate_first, hdate_temp_opt),
                    np.maximum(hdate_first, hdate_wetseas),
                    hdate_last,
                ]
            ),
        )

        hdate_ir.values = np.select(
            [
                (seasonality == "NO_SEASONALITY") & (harvest_rule == 1),
                (seasonality == "NO_SEASONALITY"),
                (seasonality == "PREC") & (harvest_rule == 2),
                (seasonality == "PREC"),
                (sseason == "winter") & (harvest_rule == 3),
                (sseason == "winter") & (harvest_rule == 6),
                (sseason == "winter"),
                (harvest_rule == 3),
                (harvest_rule == 6) & (seasonality == "PRECTEMP"),
                (harvest_rule == 6) & (smonth == 0) & (max_temp < self.temp_spring),  # noqa
                (harvest_rule == 6),
                (seasonality == "PRECTEMP"),
            ],
            [
                # NO SEASONALITY
                hdate_first,
                hdate_maxrp,
                # PREC SEASONALITY
                hdate_first,
                hdate_maxrp,
                # WINTER SOWING SEASON
                hdate_first,
                np.where(
                    (smonth == 0) & (max_temp < self.temp_fall),
                    hdate_temp_opt,
                    np.minimum(np.maximum(hdate_first, hdate_temp_base), hdate_last),  # noqa
                ),
                np.minimum(np.maximum(hdate_first, hdate_temp_opt), hdate_last),  # noqa
                # HARVEST RULE 3
                hdate_first,
                # HARVEST RULE 6
                hdate_maxrp,
                hdate_first,
                np.minimum(np.maximum(hdate_first, hdate_temp_base), hdate_last),  # noqa
                # PRECTEMP SEASONALITY
                hdate_maxrp,
            ],
            # ELSE
            default=np.minimum(np.maximum(hdate_first, hdate_temp_opt), hdate_last),  # noqa
        )

        hreason_rf.values = np.select(
            [
                hdate_rf == hdate_first,
                hdate_rf == hdate_maxrp,
                hdate_rf == hdate_wetseas,
                hdate_rf == hdate_temp_opt,
                hdate_rf == hdate_temp_base,
                hdate_rf == hdate_last,
            ],
            [
                "hdate_first",
                "hdate_maxrp",
                "hdate_wetseas",
                "hdate_temp_opt",
                "hdate_temp_base",
                "hdate_last",
            ],
            default=hreason_rf,
        )

        hreason_ir.values = np.select(
            [
                hdate_ir == hdate_first,
                hdate_ir == hdate_maxrp,
                hdate_ir == hdate_wetseas,
                hdate_ir == hdate_temp_opt,
                hdate_ir == hdate_temp_base,
                hdate_ir == hdate_last,
            ],
            [
                "hdate_first",
                "hdate_maxrp",
                "hdate_wetseas",
                "hdate_temp_opt",
                "hdate_temp_base",
                "hdate_last",
            ],
            default=hreason_rf,
        )

        # Vectorized adjustment based on year length
        hdate_rf = xr.where(
            hdate_rf <= self.ndays_year, hdate_rf, hdate_rf - self.ndays_year
        )  # noqa
        hdate_ir = xr.where(
            hdate_ir <= self.ndays_year, hdate_ir, hdate_ir - self.ndays_year
        )  # noqa

        # Assign the results to self attributes
        return (hdate_rf, hdate_ir, hreason_rf, hreason_ir)

    def calc_phu(self, monthly_climate, sdate, hdate):  # noqa
        """
        Calculate PHU requirements.

        :param monthly_climate: Dict of arrays of monthly climate variables
            (shape: (21, 12)).
        :type monthly_climate: dict
        :param sdate: Sowing date (DOY).
        :type sdate: xr.DataArray of type int
        :param hdate: Maturity date (DOY).
        :type hdate: xr.DataArray of type int
        :return: Total Thermal Unit Requirements.
        :rtype: int
        """

        daily_temp = monthly_climate["daily_temp"]
        # Adjust hdate for cross-year growth
        hdate = hdate.where(sdate < hdate, hdate + 365)

        husum = xr.full_like(sdate, 0, dtype=int)

        # Duplicate daily temperature to allow cross-year growth
        temp_values = xr.DataArray(
            xr.concat([daily_temp["value"], daily_temp["value"]], dim="day"),
            coords={"cell": sdate.cell, "day": np.arange(1, 2 * 365 + 1)},
        )
        day_values = xr.DataArray(
            np.arange(1, 2 * 365 + 1), dims=["day"]
        ).broadcast_like(  # noqa
            temp_values
        )

        grow_mask = (day_values.values >= sdate.values[:, None]) & (
            day_values.values < hdate.values[:, None]
        )

        if any(self.winter_crop):
            vern_factor = self.calc_vrf(sdate, hdate, monthly_climate)
            vern_factor = vern_factor.where(self.winter_crop, 1)
        else:
            vern_factor = 1

        # Compute Effective Thermal Units (teff) using vectorized operations
        teff = np.maximum(temp_values - self.basetemp_low, 0) * grow_mask * vern_factor

        # Compute total PHU requirement efficiently
        husum.values = np.sum(teff, axis=1).astype(int)

        husum = husum.where(self.winter_crop == 0, -husum)

        return husum

    def calc_vrf(
        self,
        sdate,
        hdate,
        monthly_climate,
        vd_b=0.2,
        max_vern_months=5,
    ):  # noqa
        """
        Calculate the Vernalization Reduction Factor (VRF) for each day using
        vectorized logic.
        """

        daily_temp = monthly_climate["daily_temp"]["value"]  # (cell, 365)
        ncell = daily_temp.shape[0]

        # --- Vernalization Effectiveness ---
        veff = np.select(
            [
                (self.vern_temp_min <= daily_temp)
                & (daily_temp < self.vern_temp_opt_min),  # noqa
                (self.vern_temp_opt_min <= daily_temp)
                & (daily_temp <= self.vern_temp_opt_max),
                (self.vern_temp_opt_max < daily_temp)
                & (daily_temp < self.vern_temp_max),  # noqa
            ],
            [
                (daily_temp - self.vern_temp_min)
                / (self.vern_temp_opt_min - self.vern_temp_min),
                1.0,
                (self.vern_temp_max - daily_temp)
                / (self.vern_temp_max - self.vern_temp_opt_max),
            ],
            default=0.0,
        )
        veff = np.clip(veff, 0, 1)

        # Repeat veff to 730 days (optimized memory usage)
        veff_full = np.repeat(veff, 2, axis=1)  # (cell, 730)

        # --- Calculate vd (vernalization requirement in days) ---
        monthly_temp = monthly_climate["temp"]  # (cell, 12)
        coldest_months_idx = np.argsort(monthly_temp, axis=1)[:, :max_vern_months]  # noqa
        max_days_per_month = self.max_vern_days / max_vern_months

        month_temp = monthly_temp.values[np.arange(ncell)[:, None], coldest_months_idx]  # noqa

        temp_min = self.vern_temp_opt_min.values[:, np.newaxis]  # (ncell, 1)
        temp_max = self.vern_temp_opt_max.values[:, np.newaxis]  # (ncell, 1)

        # Vectorized month-based calculations
        days = np.where(
            (month_temp <= temp_min),
            max_days_per_month.values[:, np.newaxis],
            np.where(
                month_temp >= temp_max,
                0.0,
                max_days_per_month.values[:, np.newaxis]
                * (1 - (month_temp - temp_min) / (temp_max - temp_min)),
            ),
        )

        vd = np.round(np.sum(days, axis=1)).astype(int)  # (cell,)

        # --- Determine end of vernalization window ---
        hdate_ext = np.where(sdate < hdate, hdate, hdate + 365)

        # Track cumulative veff per cell starting from sdate
        veff_shifted = np.full((ncell, 730), 0.0)

        start_end_diff = (hdate_ext - sdate.values).astype(int)

        # Optimized slicing of veff_shifted without loop
        start_indices = sdate.values
        end_indices = start_indices + start_end_diff

        # Use np.arange for efficient indexing
        for i in range(ncell):
            veff_shifted[i, : end_indices[i] - start_indices[i]] = veff_full[
                i, start_indices[i] : end_indices[i]
            ]

        # End day of vernalization when vdsum >= vd
        no_vern_needed = vd == 0

        # --- Compute VRF with vectorized cumsum and masking ---
        vrf = np.ones((ncell, 365), dtype=float)

        # Directly assign veff to vrf_veff (no need for a loop)
        vrf_veff = np.full((ncell, 365), 0.0)
        for i in range(ncell):
            start = sdate.values[i]
            end = start + start_end_diff[i]
            vrf_veff[i, : end - start] = veff_full[i, start:end]

        vrf_cumsum = np.cumsum(vrf_veff, axis=1)  # (cell, 365)

        vmin = vd * vd_b
        with np.errstate(divide="ignore", invalid="ignore"):
            vrf_tmp = (vrf_cumsum.T - vmin).T / (vd - vmin)[:, None]
            vrf_tmp = np.clip(vrf_tmp, 0.0, 1.0)
            vrf = np.where(vrf_cumsum < vmin[:, None], 0.0, vrf_tmp)

        # Cells with no vernalization required
        vrf[no_vern_needed, :] = 1.0

        # --- Return as xarray ---
        return xr.DataArray(
            np.repeat(vrf, 2, axis=1),  # Repeat for 730 days
            dims=("cell", "day"),
            coords={"cell": sdate.cell, "day": np.arange(1, 731)},
            name="vrf",
        )


class WorldCropSet(WorldActivity):

    def __init__(self, crop_param_file="crop_calendar/crop_parameters.csv", **kwargs):  # noqa
        super().__init__(**kwargs)

        # Initiate attributes
        self.bands = []
        self.names = []
        self.calendars = {}

        # Load crop parameters
        self.parameters = base.load_csv(crop_param_file)
        self.init_world_calendar()

    @property
    def actual_landuse(self):
        """
        Return the actual land use of the cell.
        """
        return self.world.input.landuse.where(
            self.world.input.landuse.sum(dim="cell") > 0
        ).dropna(dim="band")

    def update_landuse(self):
        """
        Update the cultivated crops based on the actual land use (what is
        actually cultivated in the cell) and for which crops is crop calendar
        data available (crop_parameters.csv).
        Crops are currently only initiated if both conditions are met and
        deleted if the crop is not cultivated anymore.
        """
        self.bands = self.actual_landuse.band.values.tolist()
        self.names = {
            WorldCropSet.irrigation_pattern.sub("", band).strip()
            for band in self.bands  # noqa
        }
        self.calendars = {
            crop for crop in self.names if crop in self.parameters.cft_name.tolist()  # noqa
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
        :param monthly_climate: Dict of arrays of monthly climate variables
            (shape: (21, 12)).
        :type monthly_climate: dict
        :param temp_min: Threshold of temperature of the coldest month (°C).
            Default is 10°C.
        :return: Array of seasonality types for each spatial unit.
        :rtype: np.ndarray (shape: (21,))
        """
        var_coef_prec = calc_var_coeff(monthly_climate["prec"])
        var_coef_temp = calc_var_coeff(deg2k(monthly_climate["temp"]))
        min_temp = np.min(monthly_climate["temp"], axis=1)

        # Create an array to store seasonality types
        seasonality = xr.full_like(self.world.grid.cell, "TEMP", dtype=object)

        # Apply conditions using NumPy vectorized operations
        no_seasonality = (var_coef_prec <= 0.4) & (var_coef_temp <= 0.01)
        prec_only = (var_coef_prec > 0.4) & (var_coef_temp <= 0.01)
        prec_temp = (
            (var_coef_prec > 0.4) & (var_coef_temp > 0.01) & (min_temp > temp_min)  # noqa
        )
        temp_prec = (
            (var_coef_prec > 0.4) & (var_coef_temp > 0.01) & (min_temp <= temp_min)  # noqa
        )

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
            "daily_ppet_diff": daily_ppet_diff,
        }

    def calc_calendar(self):
        """Calculate the crop calendar for each crop in calendars."""
        monthly_climate = self.get_monthly_climate()
        seasonality = self.calc_seasonality(monthly_climate)

        self.world.calendar.time.values[:] = self.world.input.time.values[:]
        # Calculate crop calendar for each crop
        for crop in self.calendars:

            # Update crop-specific parameters based on winter crop classification  # noqa
            is_winter_crop = self[crop].is_winter_crop
            self[crop].update_parameters(is_winter_crop)

            # calculate sowing date
            crop_sdate, crop_smonth, crop_sseason = self[crop].calc_sowing_date(  # noqa
                monthly_climate=monthly_climate, seasonality=seasonality
            )
            self.world.calendar.sdate.loc[
                {"band": [f"rainfed {crop}", f"irrigated {crop}"]}
            ] = crop_sdate

            # calculate harvest rule
            crop_harvest_rule, crop_harvest_rule_name = self[crop].calc_harvest_rule(  # noqa
                monthly_climate=monthly_climate, seasonality=seasonality
            )

            # calculate different options for harvesting
            (
                crop_hdate_first,
                crop_hdate_maxrp,
                crop_hdate_last,
                crop_hdate_wetseas,
                crop_hdate_temp_base,
                crop_hdate_temp_opt,
            ) = self[crop].calc_harvest_options(
                monthly_climate=monthly_climate,
                sdate=crop_sdate,
                sseason=crop_sseason,
            )

            # calculate harvest date, choose the best option
            (
                crop_hdate_rf,
                crop_hdate_ir,
                crop_hreason_rf,
                crop_hreason_ir,
            ) = self[crop].calc_harvest_date(
                monthly_climate=monthly_climate,
                seasonality=seasonality,
                smonth=crop_smonth,
                sseason=crop_sseason,
                harvest_rule=crop_harvest_rule,
                hdate_first=crop_hdate_first,
                hdate_maxrp=crop_hdate_maxrp,
                hdate_last=crop_hdate_last,
                hdate_wetseas=crop_hdate_wetseas,
                hdate_temp_base=crop_hdate_temp_base,
                hdate_temp_opt=crop_hdate_temp_opt,
            )
            self.world.calendar.hdate.loc[{"band": f"rainfed {crop}"}] = crop_hdate_rf  # noqa
            self.world.calendar.hdate.loc[{"band": f"irrigated {crop}"}] = (
                crop_hdate_ir  # noqa
            )
            self.world.calendar.hreason.loc[{"band": f"rainfed {crop}"}] = (
                crop_hreason_rf  # noqa
            )
            self.world.calendar.hreason.loc[{"band": f"irrigated {crop}"}] = (
                crop_hreason_ir  # noqa
            )

            crop_phu_rf = self[crop].calc_phu(
                monthly_climate=monthly_climate,
                sdate=crop_sdate,
                hdate=crop_hdate_rf,
            )
            self.world.calendar.crop_phu.loc[{"band": f"rainfed {crop}"}] = crop_phu_rf  # noqa

            crop_phu_ir = self[crop].calc_phu(
                monthly_climate=monthly_climate,
                sdate=crop_sdate,
                hdate=crop_hdate_ir,
            )
            self.world.calendar.crop_phu.loc[{"band": f"irrigated {crop}"}] = (
                crop_phu_ir  # noqa
            )

    def init_world_calendar(self):
        """
        Initialize the world calendar with the calculated crop calendar data.
        """
        self.world.calendar = LPJmLDataSet(
            {
                "sdate": self.world.input.sdate.copy(),
                "hdate": xr.full_like(self.world.input.sdate, -9999, dtype=int),  # noqa
                "hreason": xr.full_like(self.world.input.sdate, "", dtype="<U20"),  # noqa
                "crop_phu": self.world.input.crop_phu.copy(),
            },
            coords={
                "cell": self.world.input.cell,
                "time": self.world.input.time,
            },
        )
        self.world.calendar.hdate.values = self.world.output.hdate.isel(
            time=[-1]
        ).values  # noqa

    def update(self):
        """
        Update the WorldCropSet object.
        """
        self.update_landuse()
        self.calc_calendar()


def calc_var_coeff(x):
    """
    Calculate the coefficient of variation (CV) for a given array x along the
    specified axis. If the mean of x is 0, return 0 to avoid division by zero.

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
    year = (
        np.asarray(year) if np.ndim(year) > 0 else np.full_like(doy, year)
    )  # Broadcast if needed

    dates = pd.to_datetime(year.astype(str) + doy.astype(str), format="%Y%j")
    return dates.month


def interpolate_monthly_to_daily(monthly_value):
    """
    Interpolates monthly values to daily values using a periodic cubic spline.

    :param monthly_value: Array-like (shape (N, 12)) of monthly values (e.g.,
        temperature).
    :return: Dictionary with "day" (day of year, length 365) and "value"
        (interpolated values, shape (N, 365)).
    """
    midday = get_midday()
    ndays_year = 365
    day = np.arange(1, ndays_year + 1)  # Days of the year

    # Ensure input is an xarray DataArray (with correct order)
    monthly_value = monthly_value.transpose("cell", "band")

    # Extend along the month axis for periodic boundary condition
    extended_x = np.concatenate([midday, [midday[0] + ndays_year]])
    extended_y = xr.concat(
        [monthly_value, monthly_value.isel(band=0)], dim="band"
    )  # (cell, 13)

    # Vectorized interpolation (Axis 1 = band)
    spline = CubicSpline(extended_x, extended_y, axis=1, bc_type="periodic")  # noqa
    interpolated_values = spline(day)  # Result: (cell, day=365)

    # Return as an xarray.Dataset for easy handling
    return xr.Dataset(
        {"day": ("day", day), "value": (("cell", "day"), interpolated_values)}
    )  # noqa


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
    # Ensure threshold has shape (n_cells, 1) for broadcasting
    # threshold = threshold.expand_dims("time")  # Reshape to (21, 1)

    # Find days when value is above threshold
    is_value_above = daily_value["value"] >= threshold
    is_value_above_shifted = np.roll(is_value_above, 1, axis=1)

    # Find days when value crosses threshold
    value_cross_threshold = is_value_above.astype(int) - is_value_above_shifted.astype(  # noqa
        int
    )

    # Find first crossing up/down per cell
    day_cross_up = np.where(value_cross_threshold == 1, daily_value["day"], np.inf)  # noqa
    day_cross_down = np.where(value_cross_threshold == -1, daily_value["day"], np.inf)  # noqa

    # Get minimum crossing day per cell
    day_cross_up = np.min(day_cross_up, axis=1)
    day_cross_down = np.min(day_cross_down, axis=1)

    # Convert inf to -9999 (no crossing)
    day_cross_up[day_cross_up == np.inf] = -9999
    day_cross_down[day_cross_down == np.inf] = -9999

    return {
        "doy_cross_up": day_cross_up.astype(int),
        "doy_cross_down": day_cross_down.astype(int),
    }


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
    cumulative sum of the precipitation to potential evapotranspiration (P/PET)
    ratio.

    :param daily_ppet: Dict of numpy array of interpolated monthly to daily
        P/PET values.
    :type daily_ppet: dict

    :return: The day of the year (DOY) corresponding to the start of the
        wettest 120-day period.
    :rtype: int
    """
    daily_values = np.asarray(daily_ppet["value"])  # Extract daily values

    n_days = daily_values.shape[1]  # Number of days in a year (e.g., 365)
    window_size = 120  # Length of the rolling window

    # Precompute the kernel for convolution
    kernel = np.ones(window_size)

    # Perform convolution with periodic wrapping
    extended_values = np.hstack([daily_values, daily_values[:, : window_size - 1]])  # noqa
    running_sum = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"),
        axis=1,
        arr=extended_values,  # noqa
    )

    # Trim the running sum to match the original data length (365 days)
    running_sum = running_sum[:, :n_days]

    # Convert indices to DOY (Day of Year)
    return np.argmax(running_sum, axis=1) + 1
