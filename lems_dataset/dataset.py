import os, json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile, matlab
import pyroomacoustics as pra

import wave, struct

from .get import package_path, info_file_path, download_dataset


def read_wav(fn):
    """ Read a wav file possibly truncated """
    wf = wave.open(fn, "r")

    length = wf.getnframes()
    data = np.frombuffer(wf.readframes(length), dtype=np.int16)
    nchannels = wf.getnchannels()

    
    size_mismatch = data.shape[0] % nchannels
    if size_mismatch != 0:
        data = data[:-size_mismatch]

    audio = data.reshape((-1, nchannels))

    return wf.getframerate(), audio


class LEMSDataset(object):
    """
    A wrapper around the LEMS dataset introduced in [1]_ [2]_.

    References
    ----------

    .. [1] H. Do and H. F. Silverman, “SRP-PHAT methods of locating
        simultaneous multiple talkers using a frame of microphone array data,”
        Proc. ICASSP, pp. 125–128, Mar. 2010.

    .. [2] H. Do and H. F. Silverman, “Robust cross-correlation-based techniques
        for detecting and locating simultaneous, multiple sound sources,” Proc. ICASSP,
        Kyoto, Japan, Mar. 2012.
    """

    def __init__(self, data_dir=None, dataset_number=2, force_download=True):

        if dataset_number not in [1, 2]:
            raise ValueError("Valid datasets are dataset1 and dataset2")

        self.data_dir = download_dataset(dest_folder=data_dir, force_download=False)

        with open(info_file_path(), "r") as f:
            info = json.load(f)

        self.dataset_number = dataset_number
        ds_name = f"dataset{dataset_number}"
        base_dir = os.path.join(self.data_dir, ds_name)

        # Read audio
        self.audio_fn = os.path.join(base_dir, info[ds_name]["files"]["audio"])
        self.fs, self.audio = read_wav(self.audio_fn)

        # Read microphone locations
        self.mic_locs_fn = os.path.join(base_dir, info[ds_name]["files"]["mic_locs"])
        if self.dataset_number == 1:
            t = matlab.loadmat(self.mic_locs_fn)
            self.mic_locs = t["mic_loc"].T
        else:
            self.mic_locs = np.loadtxt(self.mic_locs_fn).T

        # talker locations
        self.talker_locs = np.array(info[ds_name]["talker_locations"]).T

        # Mapping of speaker to channel
        if self.dataset_number == 2:
            self.close_talker_mapping = (
                np.array(info[ds_name]["close_talking_channel_index"], dtype=np.int64) - 1
            )
        else:
            self.close_talker_mapping = None

    def ntalkers(self):
        return self.talker_locs.shape[1]

    def nmics(self):
        return self.mic_locs.shape[1]

    def play(self, channel):
        sd.play(self.audio[:, channel], samplerate=self.fs)

    def play_talker(self, talker_id):

        assert talker_id <= len(self.close_talker_mapping)
        assert self.dataset_number == 2, "Dataset1 has no close talking channels"

        sd.play(self.audio[:, self.close_talker_mapping[talker_id]], samplerate=self.fs)

    def stop(self):
        sd.stop()

    def duration(self):
        """ The duration of the audio in seconds """
        return self.audio.shape[0] / self.fs

    def samplerate(self):
        return self.fs

    def mic_signals(self):
        return self.audio[:, :181]

    def close_talking_signals(self):
        assert self.dataset_number == 2, "Dataset1 has no close talking channels"
        return self.audio[:, self.close_talker_mapping]

    def talker_point_cloud(self):
        return pra.experimental.PointCloud(X=self.talker_locs)

    def mic_point_cloud(self):
        return pra.experimental.PointCloud(X=self.mic_locs)

    def inter_mic_distances(self):
        iu = np.triu_indices(self.mic_locs.shape[1], k=1)
        D = np.sqrt(self.mic_point_cloud().EDM())
        return D[iu]

    def plot_points(self):
        tpc = self.talker_point_cloud()
        mpc = self.mic_point_cloud()

        ax = tpc.plot(color="b")
        mpc.plot(axes=ax, color="r")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Open the dataset")
    parser.add_argument("folder", type=str, help="Folder of the dataset")
    args = parser.parse_args()

    dataset = Dataset(args.folder)
