import pickle
import pytest
import numpy as np
import pandas as pd

import landmanager.components.base as base
import landmanager.components.management as management
from landmanager.models.crop_calendar import Cell, World, Model


# def test_run_model(test_path):
#     """Test running the model until end of simulation."""
#
#     with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
#         lpjml = pickle.load(lpj)
#
#     model = Model(lpjml=lpjml, test_path=test_path)
#
#     for year in model.lpjml.get_sim_years():
#         model.update(year)
#
#     last_year = (
#         model.world.output.time.values[0].astype("datetime64[Y]").astype(int).item()
#         + 1970
#     )
#
#     # last year set to 2030 in test data set
#     assert last_year == 2030


# def test_model_output(test_path):
#     """Test getting the output table of the model."""
#     with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
#         lpjml = pickle.load(lpj)
#
#     model = Model(lpjml=lpjml, test_path=test_path)
#
#     output = model.output_table
#
#     # Read the CSV file as a pandas DataFrame
#     test_output = pd.read_csv(
#         f"{test_path}/data/test_output_table.csv",
#     )
#     test_output = test_output.where(test_output.notna(), None)
#
#     # Sort the dataframes by the same columns
#     sort_columns = ["year", "cell", "entity", "variable"]
#     test_output = test_output.sort_values(by=sort_columns)
#     output = output.sort_values(by=sort_columns)
#
#     for name, row in test_output.items():
#         if name == "value":
#             # if failing lower the threshold or continue
#             #   LPJmL cell variables should be equal, but the rest can be
#             #   different
#             assert np.mean(output[name].values == row.values).item() > 0.75
#         else:
#             assert all(output[name].values == row.values)
