"""Farmer entity type class of inseeds_farmer_management
"""


import numpy as np

import pycopancore.model_components.base as core
from landmanager.components import base


class Farmer(core.Individual, base.Individual):
    """Farmer (Individual) entity type mixin class."""

    # standard methods:
    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # initialize the coupled (lpjml mapped) attributes
        self.init_coupled_attributes()

        # average harvest date of the cell is used as a proxy for the order
        # of the agents making decisions in time through the year
        self.avg_hdate = self.cell_avg_hdate

        # Same applies for cropyield (as for soilc)
        self.cropyield = self.cell_cropyield

        # initialize previous cropyield
        self.cropyield_previous = self.cropyield

    def init_coupled_attributes(self):
        """Initialize the mapped variables from the LPJmL output to the farmers"""

        # get the coupling map (inseeds to lpjml names) from the configuration
        self.coupling_map = self.model.config.coupled_config.coupling_map.to_dict()

        # set control run argument
        self.control_run = self.model.config.coupled_config.control_run

        # set the mapped variables from the farmers to the LPJmL input
        for attribute, lpjml_attribute in self.coupling_map.items():
            if not isinstance(lpjml_attribute, list):
                lpjml_attribute = [lpjml_attribute]

            for single_var in lpjml_attribute:
                if len(self.cell.input[single_var].values.flatten()) > 1:
                    continue
                setattr(self, attribute, self.cell.input[single_var].item())

    def init_neighbourhood(self):
        """Initialize the neighbourhood of the agent."""
        self.neighbourhood = [
            neighbour
            for cell_neighbours in self.cell.neighbourhood
            if len(cell_neighbours.individuals) > 0
            for neighbour in cell_neighbours.individuals
        ]

    @property
    def farmers(self):
        """Return the set of all farmers in the neighbourhood."""
        return [farmer for farmer in cell.individuals if isinstance(farmer, self.__class__)]

    @property
    def cell_cropyield(self):
        """Return the average crop yield of the cell."""
        return self.cell.output.harvestc.values

    @property
    def cell_avg_hdate(self):
        """Return the average harvest date of the cell."""
        crop_idx = [
            i
            for i, item in enumerate(self.cell.output.cftfrac.band.values.tolist())
            if any(x in item for x in self.model.config.cftmap)
        ]
        if np.sum(self.cell.output.cftfrac.isel(band=crop_idx).values) == 0:
            return 365
        else:
            return np.average(
                self.cell.output.hdate,
                weights=self.cell.output.cftfrac.isel(band=crop_idx),
            )

    def set_lpjml(self, attribute):
        """Set the mapped variables from the farmers to the LPJmL input"""
        lpjml_attribute = self.coupling_map[attribute]

        if not isinstance(lpjml_attribute, list):
            lpjml_attribute = [lpjml_attribute]

        for single_var in lpjml_attribute:
            self.cell.input[single_var][:] = getattr(self, attribute)

    def update(self, t):
        """Update the agent."""
        super().update(t)

        # update the average harvest date of the cell
        self.avg_hdate = self.cell_avg_hdate

        if self.control_run:
            return
