"""Run script for InSEEDS with LPJmL coupling"""

from pycoupler.config import read_config
from pycoupler.run import run_lpjml, check_lpjml
from pycoupler.coupler import LPJmLCoupler
from pycoupler.utils import search_country

import os

os.chdir("/p/projects/open/Jannes/repos/pymodels")

from pymodels.models.crop_calendar import Model  # noqa

# Settings ================================================================== #

config_coupled_fn = '/p/projects/copan/users/jannesbr/projects/crop_calendar/test_sim/config_coupled_test.json'

# Simulations =============================================================== #

# check if everything is set correct
# check_lpjml(config_coupled_fn)

# run lpjml simulation for coupling in the background
run_lpjml(
    config_file=config_coupled_fn,
    std_to_file=False  # write stdout and stderr to file
)

# pymodels run --------------------------------------------------------------- #

model = Model(config_file=config_coupled_fn)

for year in model.lpjml.get_sim_years():
    model.update(year)
