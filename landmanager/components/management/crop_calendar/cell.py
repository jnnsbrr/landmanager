"""Cell entity type class of the Crop Calendar component."""

from landmanager.components import management


class Cell(management.Cell):
    """Farmer (Individual) entity type mixin class."""

    def __init__(self, calendar=None, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # hold the input data for LPJmL on cell level
        if calendar is not None:
            self.calendar = calendar

    @property
    def sdate(self):
        """Get the sowing date of the crop."""
        return self.calendar.sdate

    @property
    def hdate(self):
        """Get the harvest date of the crop."""
        return self.calendar.hdate

    @property
    def hreason(self):
        """Get the harvest reason of the crop."""
        return self.calendar.hreason

    @property
    def crop_phu(self):
        """Get the harvest reason of the crop."""
        return self.calendar.crop_phu

    @property
    def hrule(self):
        """Get the harvest rule of the crop."""
        return self.calendar.hrule

    @property
    def temp(self):
        """Get the average annual temperature."""
        return self.calendar.temp.item()

    @property
    def prec(self):
        """Get the  annual precipitation."""
        return self.calendar.prec.item()
