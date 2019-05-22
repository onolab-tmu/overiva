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
Blind Source Separation using Independent Vector Analysis with Auxiliary Function
with a principal component analysis pre-processing step used to reduce the number
of channels.
"""
import numpy as np

import pyroomacoustics as pra
from overiva import overiva

def auxiva_pca(X, n_src=None, **kwargs):

    """
    Implementation of overdetermined IVA with PCA followed by determined IVA

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    f_contrast: dict of functions
        A dictionary with two elements 'f' and 'df' containing the contrast
        function taking 3 arguments This should be a ufunc acting element-wise
        on any array
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    if n_src < n_chan:
        # compute the cov mat (n_freq, n_chan, n_chan)
        covmat = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

        # Compute EVD
        # v.shape == (n_freq, n_chan), w.shape == (n_freq, n_chan, n_chan)
        v, w = np.linalg.eigh(covmat)

        # Apply dimensionality reduction
        # new shape: (n_frames, n_freq, n_src)
        new_X = np.matmul(
            X.swapaxes(0, 1), np.conj(w[:, :, -n_src:])
        ).swapaxes(0, 1)

    else:
        new_X = X

    kwargs.pop('proj_back')
    Y = overiva(new_X, proj_back=False, **kwargs)

    z = pra.bss.projection_back(Y, X[:, :, 0])
    Y *= np.conj(z[None, :, :])

    return Y
