"""The inseeds_farmer_mnagement.world class.
"""

from landmanager.components import base
import pycopanlpjml as lpjml


# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license


class World(lpjml.World, base.World):
    """World entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super().__init__(**kwargs)

    @property
    def farmers(self):
        """Return the set of all farmers."""
        farmers = [
            farmer
            for farmer in self.individuals
            if farmer.__class__.__name__ == "Farmer"  # noqa
        ]
        return farmers
