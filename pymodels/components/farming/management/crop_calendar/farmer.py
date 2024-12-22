"""Farmer entity type class of inseeds_farmer_management
"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license
import numpy as np
from enum import Enum

from pymodels.components import farming
from pymodels.components import base


class Farmer(farming.Farmer):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # initialize previous cropyield
        self.cropyield_previous = self.cropyield

        data = base.load_csv('crop_calendar/crop_parameters.csv')
        breakpoint()



    def update(self, t):
        # call the base class update method
        super().update(t)

        pass
