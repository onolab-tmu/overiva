"""
This will download the dataset from the web.

2019 (c) Robin Scheibler
"""
import json, os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def package_path():
    """ Compute the path to this package """
    path, this_file = os.path.split(__file__)
    return path

def info_file_path():
    """ Compute the path to the info file for the dataset """
    return os.path.join(package_path(), "info.json")

def download_dataset(dest_folder=None, force_download=False):
    """
    Download the dataset into a specified location
    """

    if dest_folder is None:
        dest_folder = package_path()

    with open(info_file_path(), "r") as f:
        info = json.load(f)

    base_url = info["files_base_url"]

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    for i in [1, 2]:
        ds = f"dataset{i}"
        ds_path = os.path.join(package_path(), ds)
        if not os.path.exists(ds_path):
            os.mkdir(ds_path)

        for type_, name in info[ds]["files"].items():
            f_url = os.path.join(base_url, name)
            fn = os.path.join(ds_path, name)

            if force_download or not os.path.exists(fn):
                urlretrieve(f_url, fn)

    return dest_folder
