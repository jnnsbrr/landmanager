"""Cell entity type class of the Crop Calendar component.
"""

import numpy as np
from enum import Enum

import pycopanlpjml as lpjml
from landmanager.components import management


class Cell(management.Cell):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, calendar=None, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # hold the input data for LPJmL on cell level
        if calendar is not None:
            self.calendar = calendar
    
    @property
    def sdate(self):
        """Get the sowing date of the crop."""
        return self.calendar.sdate

    @property
    def hdate(self):
        """Get the harvest date of the crop."""
        return self.calendar.hdate

    @property
    def hreason(self):
        """Get the harvest reason of the crop."""
        return self.calendar.hreason