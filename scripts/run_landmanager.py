"""Run script for landmanager with LPJmL coupling"""

from pycoupler.config import read_config
from pycoupler.run import run_lpjml, check_lpjml

# from pycoupler.utils import search_country
from landmanager.models.crop_calendar import Model  # noqa

# Settings ================================================================== #

sim_path = "./test_sim"
model_path = ".LPJmL"
landmanager_config_file = "./landmanager/models/crop_calendar/config.yaml"

# search for country code by supplying country name
# search_country("netherlands")
country_code = "NLD"

# Configuration ============================================================= #

# create config for coupled run
config_coupled = read_config(model_path=model_path, file_name="lpjml_config.cjson")  # noqa

# set coupled run configuration
config_coupled.set_coupled(
    sim_path,
    sim_name="coupled_test",
    dependency="historic_run",
    start_year=2001,
    end_year=2100,
    coupled_year=2023,
    coupled_input=[
        "sdate",
        "crop_phu",
        "landuse",
    ],  # residue_on_field
    coupled_output=[
        "temp",
        "prec",
        "pet",
        "cftfrac",
        "harvestc",
        "sdate",
        "hdate",
        "country",
        "terr_area",
    ],
    temporal_resolution={
        "temp": "monthly",
        "prec": "monthly",
        "pet": "monthly",
    },  # noqa
)

# only for single cells runs
config_coupled.outputyear = 2003

config_coupled.fix_co2 = True
config_coupled.fix_co2_year = 2022
config_coupled.fix_climate = True
config_coupled.fix_climate_cycle = 11
config_coupled.fix_climate_year = 2013

# only for global runs = TRUE
config_coupled.river_routing = False

config_coupled.tillage_type = "read"
config_coupled.residue_treatment = "read_residue_data"
config_coupled.separate_harvests = False

# regrid by country - create new (extracted) input files and update config file
config_coupled.regrid(sim_path, country_code=country_code)

config_coupled.add_config(landmanager_config_file)


# write config (Config object) as json file
config_coupled_fn = config_coupled.to_json()


# Simulations =============================================================== #

# check if everything is set correct
check_lpjml(config_coupled_fn)

# run lpjml simulation for coupling in the background
run_lpjml(config_file=config_coupled_fn, std_to_file=False)

# landmanager run ----------------------------------------------------------- #

model = Model(config_file=config_coupled_fn)

for year in model.lpjml.get_sim_years():
    model.update(year)
