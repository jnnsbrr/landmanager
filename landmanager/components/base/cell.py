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
        
        df = super().output_table

        num_rows = len(df)

        if num_rows == 0:
            return df  # Avoid unnecessary operations

        # Efficient column additions using assignment (faster than DataFrame.insert)
        df["cell"] = self.grid.cell.item()
        df["lon"] = self.grid.cell.lon.item()
        df["lat"] = self.grid.cell.lat.item()

        if hasattr(self, "country"):
            df["country"] = self.country.item()
        if hasattr(self, "area"):
            df["area [km2]"] = round(self.area.item() * 1e-6, 4)

        return df