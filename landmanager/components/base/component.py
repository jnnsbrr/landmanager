import sys
import pandas as pd


class Component:
    """Model mixin class."""

    def __init__(self, **kwargs):
        """Initialize the model mixin."""
        pass

    @property
    def output_table(self):
        # get all world outputs
        df = self.world.output_table

        # get all cell outputs
        if hasattr(self.world, "cells"):
            df = pd.concat([df] + [cell.output_table for cell in self.world.cells])  # noqa

        # get all farmer outputs
        if hasattr(self.world, "farmers"):
            df = pd.concat(
                [df] + [farmer.output_table for farmer in self.world.farmers]
            )  # noqa

        return df

    def write_output_table(self, init=False, file_format="parquet"):
        if hasattr(sys, "_called_from_test"):
            return
        if file_format == "parquet":
            self.write_output_parquet(self.output_table, init)
        elif file_format == "csv":
            self.write_output_csv(self.output_table, init)
        else:
            raise ValueError(f"Output file format {file_format} not supported")

    def write_output_csv(self, df, init=False):
        """Write output data"""

        if self.lpjml.sim_year == self.config.start_coupling and init:
            header = True
            mode = "w"
        else:
            header = False
            mode = "a"

        # define the file name and header row
        file_name = f"{self.config.sim_path}/output/{self.config.sim_name}/landmanager_data.csv"  # noqa

        df.to_csv(file_name, mode=mode, header=header, index=False, sep=";")

    def write_output_parquet(self, df, init=False):
        """Write output data to Parquet file"""
        file_name = f"{self.config.sim_path}/output/{self.config.sim_name}/landmanager_data.parquet"  # noqa

        # Append mode: write new data without rewriting the file.
        if self.lpjml.sim_year == self.config.start_coupling and init:
            df.to_parquet(file_name, engine="pyarrow", index=False)
        else:
            # Read the existing data
            existing_data = pd.read_parquet(file_name)

            # Concatenate the existing data with the new data
            combined_data = pd.concat([existing_data, df], ignore_index=True)

            # Write the combined data back
            combined_data.to_parquet(file_name, engine="pyarrow", index=False)

    def update(self, t):
        """Update the model."""
        pass
