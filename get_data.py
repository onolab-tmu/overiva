# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script can be used to download the data used in the experiments.
"""
import os
from pyroomacoustics.datasets.utils import download_uncompress

url_data = "https://zenodo.org/record/3066489/files/cmu_arctic_concat15.tar.gz"
temp_dir = "./temp"
samples_dir = "./samples"


def get_data():
    if os.path.exists(samples_dir):
        print("The samples directory " f"{samples_dir}" " seems to exist already. Delete if re-download is needed.")
    else:
        print("Downloading the samples... ", end="")
        download_uncompress(url_data, temp_dir)
        # change the directory name to the desired one
        dl_dir = os.listdir(temp_dir)[0]
        os.rename(os.path.join(temp_dir, dl_dir), samples_dir)
        os.rmdir(temp_dir)
        print("done.")


if __name__ == "__main__":
    get_data()
