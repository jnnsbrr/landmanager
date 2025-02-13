"""Farmer entity type class of inseeds_farmer_management
"""

import numpy as np
from enum import Enum

import pycopanlpjml as lpjml
from landmanager.components import management


class Cell(management.Cell):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        pass