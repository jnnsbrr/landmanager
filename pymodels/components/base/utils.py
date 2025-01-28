import pandas as pd
import importlib.resources as pkg_resources


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file as a Pandas DataFrame from the library's data folder.

    :param file_path: The path to the CSV file in the data folder.
    :type file_path: str
    :return: table.
    :rtype: pd.DataFrame
    """
    # Split the file_path into parts
    parts = file_path.strip("/").split("/")

    # Extract the package (subfolder) and file name
    if len(parts) == 1:
        package, file_name = "pymodels.data", parts[0]
    else:
        package = f"pymodels.data.{'.'.join(parts[:-1])}"
        file_name = parts[-1]

    # Open and load the CSV file
    with pkg_resources.open_text(package, file_name) as file:
        return pd.read_csv(file)
