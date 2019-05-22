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
This file contains a number of routines facilitating the simulation.
"""
import math
import numpy as np
from tkinter import Tk, Button, Label

import sounddevice as sd
import pyroomacoustics as pra


# Now come the GUI part
class PlaySoundGUI(object):
    """
    This is a class that will create a simple GUI allowing to
    play sound samples and compare them.
    """
    def __init__(self, master, fs, mix, sources, references=None):
        self.master = master
        self.fs = fs
        self.mix = mix
        self.sources = sources
        self.sources_max = np.max(np.abs(sources))
        self.references = references.copy()
        master.title("Comparator")

        if self.references is not None:
            self.references *= 0.75 / np.max(np.abs(self.references))

        nrow = 0

        self.label = Label(master, text="Listen to the output.")
        self.label.grid(row=nrow, columnspan=2)
        nrow += 1

        self.mix_button = Button(
            master, text="Mix", command=lambda: self.play(self.mix)
        )
        self.mix_button.grid(row=nrow, columnspan=2)
        nrow += 1

        self.buttons = []
        for i, source in enumerate(self.sources):
            self.buttons.append(
                Button(
                    master,
                    text="Source " + str(i + 1),
                    command=lambda src=source: self.play(src),
                )
            )

            if self.references is not None:
                self.buttons[-1].grid(row=nrow, column=1)
                ref_sig = self.references[i, :]
                self.buttons.append(
                    Button(
                        master,
                        text="Ref " + str(i + 1),
                        command=lambda rs=self.references[i, :]: self.play(rs),
                    )
                )
                self.buttons[-1].grid(row=nrow, column=0)

            else:
                self.buttons[-1].grid(row=nrow, columnspan=2)

            nrow += 1

        self.stop_button = Button(master, text="Stop", command=sd.stop)
        self.stop_button.grid(row=nrow, columnspan=2)
        nrow += 1

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(row=nrow, columnspan=2)
        nrow += 1

    def play(self, src):
        sd.play(0.75 * src / self.sources_max, samplerate=self.fs, blocking=False)


def random_layout(vol_dim, n_mic, offset=None, seed=None):

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    vol_dim = np.array(vol_dim)

    if offset is None:
        offset = np.zeros_like(vol_dim)

    vol_dim = vol_dim + np.array(offset)

    points = []
    for l, u in zip(offset, vol_dim):
        points.append(np.random.uniform(l, u, size=n_mic))

    if seed is not None:
        np.random.set_state(rng_state)

    return np.array(points)


def grid_layout(room_dim, n_mic, offset=None, seed=None):
    """
    Place the microphones equispaced on a grid
    """

    area = np.prod(room_dim[:2])
    sq_L = np.sqrt(area / n_mic)

    mic_loc = []
    wb = int(np.floor(room_dim[0] / sq_L))
    hb = n_mic // wb

    n = 0
    x, y = 0, 0
    while n < n_mic:
        mic_loc.append([x, y, 0.0])

        n = n + 1
        if n % wb == 0:
            x = 0
            y += sq_L
        else:
            x += sq_L
    mic_loc = np.array(mic_loc).T

    # center the microphones (x-y plan only)
    for i in range(2):
        mic_loc[i, :] += (room_dim[i] - mic_loc[i, :].max()) / 2

    if offset is not None:
        mic_loc += np.array([offset]).T

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        mic_loc += np.random.randn(*mic_loc.shape) * 0.025  # few centimeters
        np.random.set_state(rng_state)

    return mic_loc


def semi_circle_layout(center, angle, distance, n, rot=None, seed=None):
    """
    Places n points on a semi circle covering an angle, at a distance from center
    """

    center = np.array(center)

    angles = np.linspace(0, angle, n) + rot

    v = np.array([np.cos(angles), np.sin(angles)]) * distance

    if center.shape[0] == 3:
        v = np.concatenate([v, np.zeros((1, n))], axis=0)

    v += center[:, None]

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        v += np.random.randn(*v.shape) * 0.025  # few centimeters
        np.random.set_state(rng_state)

    return v


def gm_layout(n, centers, std=None, weights=None, seed=None):

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    if std is None:
        std = np.ones(centers.shape[1])
    else:
        std = np.array(std)

    if weights is None:
        weights = np.ones(centers.shape[1]) / centers.shape[1]

    locs = []

    rep = math.ceil(n / centers.shape[1])
    c_list = np.repeat(np.arange(centers.shape[1]), rep)[:n]

    for c in c_list:

        loc = centers[:, c] + np.random.randn(centers.shape[0]) * std
        locs.append(loc)

    if seed is not None:
        np.random.set_state(rng_state)

    return np.array(locs).T
