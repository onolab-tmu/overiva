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


class OnoLab6010Dataset(object):
    """
    A wrapper around the Onolab 60ch, 10 speakers dataset.

    Parameters
    ----------
    data_dir: str
        An optional location where the dataset should reside
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

    def __init__(self, data_dir: str = None, force_download: bool = False):

        self.data_dir = download_dataset(
            dest_folder=data_dir, force_download=force_download
        )

        with open(info_file_path(), "r") as f:
            info = json.load(f)

        ds_name = f"dataset"
        base_dir = os.path.join(self.data_dir, ds_name)

        # Read audio
        self.audio_fn = os.path.join(base_dir, info[ds_name]["files"]["audio"])
        self.fs, self.audio = read_wav(self.audio_fn)

        # Read segmentation info
        self.segment_labels = info["segmentation"]["labels"]
        lmt = info["segmentation"]["limits"]
        self.segment_limits = list(zip(lmt[:-1], lmt[1:]))
        self.segments = dict(
            zip(
                self.segment_labels,
                [self.audio[l:u, :] for l, u in self.segment_limits],
            )
        )
        self.segment_durations = {}
        for lbl, seg in self.segments.items():
            self.segment_durations[lbl] = seg.shape[0] / self.fs

        # Now create segments with a little silence around the talkers
        self.talker_segments = self.segment_talkers()
        self.talker_durations = {}
        for lbl, seg in self.talker_segments.items():
            self.talker_durations[lbl] = seg.shape[0] / self.fs

        self.ntalkers = info["ntalkers"]

    def ntalkers(self):
        """ Number of talkers """
        return self.talkers

    def nmics(self):
        """ Number of microphones """
        return self.audio.shape[1]

    def play(self, channel: int):
        """ Plays one of the channels """
        sd.play(self.audio[:, channel], samplerate=self.fs)

    def play_talker(self, talker_id: int):
        """ Plays the close-talking signal from one of the talkers """

        sd.play(self.segments[f"talker_{talker_id}"], samplerate=self.fs)

    def get_talker(self, talker_id: int):
        return self.segments[f"talker_{talker_id}"]

    def stop(self):
        sd.stop()

    def samplerate(self):
        return self.fs

    def get_silence(self):
        silences = []
        for lbl, (l, u) in zip(self.segment_labels, self.limits):
            if lbl.startswith("silence"):
                silences.append(self.audio[l:u, :])
        return silences

    def segment_talkers(self):
        """ Segment talkers from the full recording and adds a little bit of silence at both ends """

        labels = []
        segments = []
        for i, lbl in enumerate(self.segment_labels):
            l, u = self.segment_limits[i]
            if lbl.startswith("talker"):
                if i == 0:
                    s = 0
                else:
                    s = int(np.mean(self.segment_limits[i - 1]))

                if i == len(self.segment_labels) - 1:
                    e = u
                else:
                    e = int(np.mean(self.segment_limits[i + 1]))

                labels.append(lbl)
                segments.append(self.audio[s:e, :])

        return dict(zip(labels, segments))

    def get_mix(self, talkers: list = None, balance: bool = True):
        """ Mixes talkers together. Returns the mix and the individual speakers. """

        if talkers is None:
            talkers = self.ntalkers

        if isinstance(talkers, int):
            talkers = list(range(talkers))

        talkers = [f"talker_{t}" for t in talkers]

        local_segments = []
        min_len = self.audio.shape[0]
        for tlk, seg in self.talker_segments.items():
            if tlk in talkers:
                local_segments.append(seg)
                # keep track of shortest segment
                if seg.shape[0] < min_len:
                    min_len = seg.shape[0]

        mix = np.zeros((min_len, self.nmics()), dtype=self.audio.dtype)
        refs = np.zeros((min_len, len(talkers), self.nmics()), dtype=np.float)
        
        for i, seg in enumerate(local_segments):
            if balance:
                refs[:, i, :] = seg[:min_len, :] / np.std(seg[:min_len, :])
            else:
                refs[:, i, :] = seg[:min_len, :]

        refs *= 0.95 / np.abs(refs).max()

        mix = np.sum(refs, axis=1)

        return mix, refs
