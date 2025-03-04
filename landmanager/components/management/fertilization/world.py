"""Farmer entity type class of landmanager_farmer_management
"""
import re
import numpy as np
import pandas as pd
import xarray as xr
import pycoupler
from enum import Enum
import datetime

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
