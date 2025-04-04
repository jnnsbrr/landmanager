"""Cell entity type class for the management component."""

import numpy as np
from enum import Enum

import pycopanlpjml as lpjml
from landmanager.components import base


class Cell(lpjml.Cell, base.Cell):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # initialize previous soilc
        self.landuse_previous = self.landuse
        self.is_landuse_previous = self.is_landuse

    @property
    def farmers(self):
        """Return the set of all farmers."""
        farmers = [
            farmer
            for farmer in self.individuals
            if farmer.__class__.__name__ == "Farmer"  # noqa
        ]
        return farmers

    @property
    def landuse(self):
        """Get land use of the cell."""
        return self.input.landuse.isel(time=-1).sum()

    @property
    def is_landuse(self):
        """Check if the cell is under land use."""
        return self.landuse > 0

    @property
    def is_new_landuse(self):
        """Check if the cell has changed from all natural land to land use."""
        return not self.is_landuse_previous and self.is_landuse

    @property
    def is_old_landuse(self):
        """Check if the cell has changed from land use to all natural."""
        return self.is_landuse_previous and not self.is_landuse

    def update(self, t):
        """Update the agent."""
        super().update(t)

        # update the average harvest date of the cell
        self.landuse_previous = self.landuse
        self.is_landuse_previous = self.is_landuse
