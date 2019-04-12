"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2019 (c) Robin Scheibler, MIT License
"""
import numpy as np

import pyroomacoustics as pra
from oiva import oiva

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
    Y = oiva(new_X, proj_back=False, **kwargs)

    z = pra.bss.projection_back(Y, X[:, :, 0])
    Y *= np.conj(z[None, :, :])

    return Y
