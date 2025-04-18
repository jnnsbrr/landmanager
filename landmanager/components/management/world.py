"""The management world class."""

from landmanager.components import base
import pycopanlpjml as lpjml


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
