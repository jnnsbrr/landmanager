import os
import argparse
import memray

from pycoupler.coupler import LPJmLCoupler
from landmanager.models.crop_calendar import Model
from pyinstrument import Profiler


def run_landmanager(config_file):
    """Run the landmanager model with the given configuration file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} does not exist")

    # profiler = Profiler(interval=0.05)
    # profiler.start()
    # with memray.Tracker("/p/projects/open/Jannes/projects/copan_lpjml/crop_calendar/python/simulations/output_file_global_2.bin"):
    #     model = Model(config_file=config_file)
    #     for year in model.lpjml.get_sim_years():
    #         if year == model.lpjml.config.lastyear:
    #             profiler.stop()
    #             profiler.write_html("/p/projects/open/Jannes/projects/copan_lpjml/crop_calendar/python/simulations/output_test_global_2.html")
    #         model.update(year)

    model = Model(config_file=config_file)

    for year in model.lpjml.get_sim_years():
        model.update(year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the configuration file")
    args = parser.parse_args()

    run_landmanager(args.config_file)

# execute program via
# python landmanager.py /path/to/config_coupled_fn.json
