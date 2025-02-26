"""Output writing class."""


class Output:
    """Model mixin class."""

    def __init__(self, **kwargs):
        """Initialize the model mixin."""
        self.__dict__.update(kwargs)

    @property
    def names(self):
        """Return the model variables."""
        return list(self.__dict__.keys())
