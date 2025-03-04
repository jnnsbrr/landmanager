import pycopancore.model_components.base as core
from pycopancore.data_model.variable import Variable
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)

from landmanager.components import base
from landmanager.components import management
from landmanager.components.management import crop_calendar
from landmanager.components import lpjml


class Cell(crop_calendar.Cell):
    """Cell entity type."""
    output_variables = base.Output(
        sdate=Variable(
            "sowing date",
            "sowing date of calendar crops",
            unit=DAU.doy,
        ),
        hdate=Variable(
            "harvest date",
            "harvest date of calendar crops",
            unit=DAU.doy,
        ),
        hreason=Variable(
            "harvest reason",
            "harvest reason of calendar crops",
        )
    )


class World(crop_calendar.World):
    """World entity type."""

    pass



class Model(lpjml.Component, management.Component):
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
        # self.init_cells(
        #     model=self,
        #     cell_class=Cell,
        #     world_views=["calendar"]
        # )
        # self.write_output_table(
        #     file_format=self.config.coupled_config.output_settings.file_format,
        #     init=True
        # )

    def update(self, t):
        super().update(t)
        # self.write_output_table(
        #     file_format=self.config.coupled_config.output_settings.file_format,
        #     init=False
        # )
        self.update_lpjml(t)
