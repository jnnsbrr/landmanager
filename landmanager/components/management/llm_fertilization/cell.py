"""Cell entity type class of the Crop Calendar component."""

import numpy as np
from enum import Enum

import pycopanlpjml as lpjml
from landmanager.components import management


class Cell(management.Cell):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, calendar=None, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        pass