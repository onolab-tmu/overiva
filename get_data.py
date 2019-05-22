import os
from pyroomacoustics.datasets.utils import download_uncompress

url_data = "https://zenodo.org/record/3066489/files/cmu_arctic_concat15.tar.gz"
temp_dir = "./temp"
samples_dir = "./samples"

def get_data():
    if os.path.exists(samples_dir):
        print(f"The samples directory ""{samples_dir}"" seems to exist already.")
        print("Delete first for re-downloading.")
    else:
        print("Downloading the samples... ", end="")
        download_uncompress(url_data, "temp")
        # change the directory name to the desired one
        dl_dir = os.listdir(temp_dir)[0]
        os.rename(os.path.join(temp_dir, dl_dir), samples_dir)
        os.rmdir(temp_dir)
        print("done.")

if __name__ == "__main__":
    get_data()
