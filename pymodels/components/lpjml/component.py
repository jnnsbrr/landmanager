import sys
import pickle
import pycopanlpjml as lpjml


# mixin for testing
class Component(lpjml.Component):

    def __init__(self, test_path=None, **kwargs):
        """Initialize an instance of LPJmLComponent."""
        super().__init__(**kwargs)

        if hasattr(sys, "_called_from_test"):
            # Define new methods for self.lpjml
            def read_input():
                """Read the input data from the LPJmL output file."""
                with open(f"{test_path}/data/lpjml_input.pkl", "rb") as inp:
                    lpjml_input = pickle.load(inp)
                return lpjml_input

            def read_output():
                """Read the output data from the LPJmL output file."""
                with open(f"{test_path}/data/lpjml_output.pkl", "rb") as out:
                    lpjml_output = pickle.load(out)
                return lpjml_output

            # Override the methods on self.lpjml
            self.lpjml.read_input = read_input
            self.lpjml.read_output = read_output
            self.lpjml.read_historic_output = read_output
