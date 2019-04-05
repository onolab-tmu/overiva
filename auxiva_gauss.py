"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics import stft, istft
from pyroomacoustics.bss import projection_back

def auxiva(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    model="laplace",
    return_filters=False,
    callback=None,
):

    """
    Implementation of AuxIVA algorithm for BSS presented in

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

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
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
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

    # for now, only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    I = np.eye(n_src, n_src)
    r_inv = np.zeros((n_frames, n_src))
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    X = X.swapaxes(0, 1).copy()

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = X @ np.conj(W)

    for epoch in range(n_iter):

        demix(Y, X, W)

        if callback is not None and epoch % 10 == 0:
            Y_tmp = Y.swapaxes(0, 1)
            if proj_back:
                z = projection_back(Y_tmp, X[:, :, 0].swapaxes(0, 1))
                callback(Y_tmp * np.conj(z[None, :, :]))
            else:
                callback(Y_tmp)

        # simple loop as a start
        # shape: (n_frames, n_src)
        if model == 'laplace':
            r_inv[:, :] = 1. / (2. * np.linalg.norm(Y, axis=0))
        elif model == 'gauss':
            r_inv[:, :] = n_freq / (np.linalg.norm(Y, axis=0) ** 2)

        # Update now the demixing matrix
        for s in range(n_src):
            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            V[:, :, :] = (X.swapaxes(1, 2) * r_inv[None, None, :, s]) @ np.conj(X) / n_frames

            WV = np.conj(W).swapaxes(1, 2) @ V
            rhs = I[None, :, s][[0] * WV.shape[0], :]
            W[:, :, s] = np.linalg.solve(WV, rhs)

            # normalize
            denom = np.conj(W[:, None, :, s]) @ V[:, :, :] @ W[:, :, None, s]
            W[:, :, s] /= np.sqrt(denom[:, :, 0])

    demix(Y, X, W)

    Y = Y.swapaxes(0, 1).copy()
    X = X.swapaxes(0, 1)

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
