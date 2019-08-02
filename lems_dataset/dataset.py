import os, json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile, matlab
import pyroomacoustics as pra

import wave, struct

from .get import package_path, info_file_path, download_dataset


def read_wav(fn):
    """ Read a wav file, even if truncated """
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

    Parameters
    ----------
    data_dir: str
        An optional location where the dataset should reside
    dataset_number: int
        Chooses between dataset1 (24 channels, 5 talkers) and dataset2 (181 channels, 10 talkers)
    force_download: bool
        If set to True, the dataset files will be downloaded even if they already exist

    References
    ----------

    .. [1] H. Do and H. F. Silverman, “SRP-PHAT methods of locating
        simultaneous multiple talkers using a frame of microphone array data,”
        Proc. ICASSP, pp. 125–128, Mar. 2010.

    .. [2] H. Do and H. F. Silverman, “Robust cross-correlation-based techniques
        for detecting and locating simultaneous, multiple sound sources,” Proc. ICASSP,
        Kyoto, Japan, Mar. 2012.
    """

    def __init__(
        self,
        data_dir: str = None,
        dataset_number: int = 2,
        force_download: bool = False,
    ):

        if dataset_number not in [1, 2]:
            raise ValueError("Valid datasets are dataset1 and dataset2")

        self.data_dir = download_dataset(
            dest_folder=data_dir, force_download=force_download
        )

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
                np.array(info[ds_name]["close_talking_channel_index"], dtype=np.int64)
                - 1
            )
        else:
            self.close_talker_mapping = None

    def ntalkers(self):
        """ Number of talkers """
        return self.talker_locs.shape[1]

    def nmics(self):
        """ Number of microphones """
        return self.mic_locs.shape[1]

    def play(self, channel: int):
        """ Plays one of the channels """
        sd.play(self.audio[:, channel], samplerate=self.fs)

    def play_talker(self, talker_id: int):
        """ Plays the close-talking signal from one of the talkers """

        assert self.dataset_number == 2, "Dataset1 has no close talking channels"
        assert talker_id >= 0 and talker_id < len(
            self.close_talker_mapping
        ), "Talkers id are in 0-9"

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
