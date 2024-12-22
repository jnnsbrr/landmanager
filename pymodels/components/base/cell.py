import os
import sys
import pandas as pd

from . import Entity


class Cell(Entity):
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    # property to get output variables
    @property
    def output_table(self):
        variables = self.get_defined_outputs()

        if not variables:
            return pd.DataFrame()
        else:
            df = super().output_table

            df.insert(1, "cell", [self.grid.cell.item()] * len(variables))
            df.insert(2, "lon", [self.grid.cell.lon.item()] * len(variables))
            df.insert(3, "lat", [self.grid.cell.lat.item()] * len(variables))

            if hasattr(self, "country"):
                df.insert(4, "country", [self.country.item()] * len(variables))
            if hasattr(self, "area"):
                df.insert(
                    5,
                    "area [km2]",
                    [round(self.area.item() * 1e-6, 4)] * len(variables),
                )

            return df
