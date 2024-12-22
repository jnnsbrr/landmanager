"""Farming component for the model."""
from pymodels.components import base

class Component(base.Component):
    """Model mixing component class for farming.
    """

    def init_farmers(self, farmer_class, **kwargs):
        """Initialize farmers."""
        farmers = []

        for cell in self.world.cells:
            if cell.output.cftfrac.sum("band") == 0:
                continue

            farmer = farmer_class(cell=cell, model=self)
            farmers.append(farmer)

        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.init_neighbourhood()

    def update_init_farmers(self, farmer_class):
        """Update initialization of farmers if land use changes."""
        farmers = []

        change=False
        for cell in self.world.cells:
            if cell.is_new_landuse:
                farmer = farmer_class(cell=cell, model=self)
                farmers.append(farmer)
                change=True
            elif cell.is_old_landuse:
                farmers.remove(cell.farmer)
                cell.farmer.deactivate()

        if change:
            # TODO: rework neighbourhood update to only update affected farmers
            farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
            for farmer in farmers_sorted:
                farmer.init_neighbourhood()

    def update(self, t):
        """Update the model."""

        farmers_sorted = sorted(self.world.farmers, key=lambda farmer: farmer.avg_hdate)

        for cell in self.world.cells:
            cell.update(t)

        for farmer in farmers_sorted:
            farmer.update(t)
