import pycopancore.model_components.base as core

from pymodels.components import farming
from pymodels.components.farming.management import crop_calendar
from pymodels.components import lpjml


class Farmer(crop_calendar.Farmer):
    """Farmer entity type."""

    pass


class Cell(farming.Cell):
    """Cell entity type."""

    pass


class World(farming.World):
    """World entity type."""

    pass


class Model(lpjml.Component, farming.Component):
    """Model class for the InSEEDS Social model integrating the LPJmL model and
    coupling component as well as the farmer management component.
    """

    name = "InSEEDS farmer management"
    description = "InSEEDS farmer management model representing only social \
    dynamics and decision-making on the basis of the TPB"

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        # Initialize the parent classes first
        super().__init__(**kwargs)

        # Ensure self.lpjml is initialized before accessing it
        if not hasattr(self, "lpjml") or self.lpjml is None:
            raise ValueError("lpjml must be initialized in the parent class.")

        # initialize LPJmL world
        self.world = World(
            model=self,
            input=self.lpjml.read_input(),
            output=self.lpjml.read_historic_output(),  # .isel(time=[-1]),
            grid=self.lpjml.grid,
            country=self.lpjml.country,
            area=self.lpjml.terr_area,
        )
        # initialize cells
        self.init_cells(model=self, cell_class=Cell)

        # initialize farmers
        self.init_farmers(farmer_class=Farmer)

    def update(self, t):
        super().update(t)
        # self.write_output_table(
        #     file_format=self.config.coupled_config.output_settings.file_format
        # )
        self.update_lpjml(t)
