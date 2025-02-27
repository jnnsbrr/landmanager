import pandas as pd

from . import Entity


class World(Entity):
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    @property
    def output_table(self):
        variables = self.get_defined_outputs()

        if not variables:
            return pd.DataFrame()
        else:
            df = super().output_table
            if hasattr(self, "area"):
                df.insert(
                    5,
                    "area [km2]",
                    [round(self.area.sum().item() * 1e-6, 4)] * len(variables),
                )

            return df
