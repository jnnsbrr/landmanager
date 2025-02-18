import pandas as pd
import numpy as np
import xarray as xr
from pycoupler.data import LPJmLData

from . import Output


class Entity:
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    output_variables = Output()

    def __init__(self, model=None, **kwargs):
        """Initialize an instance of World."""
        self.model = model

    @property
    def output_table(self):
        variables = self.get_defined_outputs()

        if not variables:
            return pd.DataFrame()

        data = []
        
        for var in variables:
            var_obj = getattr(self, var, None)
            var_name = getattr(getattr(self.__class__.output_variables, var, None), "name", None)
            unit = getattr(getattr(getattr(self.__class__.output_variables, var, None), "unit", None), "symbol", None)

            if hasattr(var_obj, "dims") and "band" in var_obj.dims:
                bands = var_obj.coords["band"].values
                values = var_obj.values.flatten()  # Convert to a NumPy array for faster filtering

                # Determine the correct mask based on dtype
                if np.issubdtype(values.dtype, np.integer):
                    mask = values != -9999  # Ignore integer missing values
                elif np.issubdtype(values.dtype, np.floating):
                    mask = np.isfinite(values)  # Ignore NaN and infinities
                elif np.issubdtype(values.dtype, np.str_):
                    mask = values != ""  # Ignore empty strings
                else:
                    mask = np.ones_like(values, dtype=bool)  # Keep all other t
                    ypes
                # Collect data for all bands at once
                data.extend(
                    zip(
                        [self.model.lpjml.sim_year] * np.count_nonzero(mask),
                        [self.__class__.__name__] * np.count_nonzero(mask),
                        [var_name] * np.count_nonzero(mask),
                        values[mask],
                        [unit] * np.count_nonzero(mask),
                        bands[mask]
                    )
                )

            else:  # Scalar case
                value = var_obj

                if (isinstance(value, int) and value == -9999) or (isinstance(value, float) and not np.isfinite(value)) or (isinstance(value, str) and value == ""):
                    continue

                data.append((self.model.lpjml.sim_year, self.__class__.__name__, var_name, value, unit, None))

        return pd.DataFrame(data, columns=["year", "entity", "variable", "value", "unit", "band"])

    def get_defined_outputs(self):
        return [
            var
            for var in self.__class__.output_variables.names
            if var
            in self.model.config.coupled_config.output.to_dict()[
                self.__class__.__name__.lower()
            ]
        ]

    def update(self, t):
        pass
