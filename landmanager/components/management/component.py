"""Management component for the model."""

from landmanager.components import base


class Component(base.Component):
    """Model mixing component class for management."""

    def init_farmers(self, farmer_class, **kwargs):
        """Initialize farmers in the world.

        :param farmer_class: farmer class to be used
        :type farmer_class: class
        :param kwargs: additional keyword arguments
        :type kwargs: dict
        """
        # init list of farmers for the world
        farmers = []

        # check if cell has land use if not skip (no farmers needed)
        for cell in self.world.cells:
            if cell.output.cftfrac.sum() == 0:
                continue

            # create representative farmer for each cell
            farmer = farmer_class(cell=cell, model=self)
            farmers.append(farmer)

        # init neighbourhood for each farmer
        # for farmer in farmers_sorted:
        #     farmer.init_neighbourhood()

    def update_farmers(self):
        """Update initialization/deactivation of farmers in the world if land
        use changes.

        :return: sorted list of farmers
        :rtype: list
        """
        farmers = []

        # get farmers class from first farmer
        farmer_class = next(farmer for farmer in self.world.farmers).__class__
        # init checker for any change in farmer composition (has land use changed?)
        change = False

        # check for new land use
        for cell in self.world.cells:
            if cell.is_new_landuse:
                farmer = farmer_class(cell=cell, model=self)
                farmers.append(farmer)
                change = True

            elif not cell.farmers:
                continue
            elif cell.is_old_landuse:
                farmers.remove(cell.farmers[0])
                cell.farmers[0].deactivate()
            else:
                farmers.append(cell.farmers[0])

        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        # if change:
        #     # TODO: rework neighbourhood update to only update affected farmers
        #     for farmer in farmers_sorted:
        #         farmer.init_neighbourhood()

        return farmers_sorted

    def update(self, t):
        """Update the model.

        :param t: current time as year
        :type t: int
        """

        # update cell state
        for cell in self.world.cells:
            cell.update(t)

        self.world.update(t)
        # update initialization (or deactivation) of farmers
        if self.world.farmers:
            farmers_sorted = self.update_farmers()

            # update farmer behaviour
            for farmer in farmers_sorted:
                farmer.update(t)
