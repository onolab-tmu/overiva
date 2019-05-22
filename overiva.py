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
Implementation of overdetermined independent vector extraction based on auxilliary function.
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def overiva(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    model="laplace",
    init_eig=False,
    return_filters=False,
    callback=None,
):

    """
    Implementation of overdetermined IVA algorithm for BSS as presented. See
    the following publication for a detailed description of the algorithm.

    R. Scheibler and N. Ono, Independent Vector Analysis with more Microphones than Sources, arXiv, 2019.
    https://arxiv.org/abs/1905.07880

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components. When
        ``n_src==nchannels``, the algorithms is identical to AuxIVA. When
        ``n_src==1``, then it is doing independent vector extraction.
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nsrc, nchannels), optional
        Initial value for demixing matrix
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    init_eig: bool, optional (default ``False``)
        If ``True``, and if ``W0 is None``, then the weights are initialized
        using the principal eigenvectors of the covariance matrix of the input
        data.
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    W_hat = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W = W_hat[:, :, :n_src]
    J = W_hat[:, :n_src, n_src:]

    def tensor_H(T):
        return np.conj(T).swapaxes(1, 2)

    def update_J_from_orth_const():
        tmp = np.matmul(tensor_H(W), Cx)
        J[:, :, :] = np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])

    # initialize A and W
    if W0 is None:

        if init_eig:
            # Initialize the demixing matrices with the principal
            # eigenvectors of the input covariance
            v, w = np.linalg.eig(Cx)
            for f in range(n_freq):
                ind = np.argsort(v[f])[-n_src:]
                W[f, :, :] = np.conj(w[f][:, ind])

        else:
            # Or with identity
            for f in range(n_freq):
                W[f, :n_src, :] = np.eye(n_src)

    else:
        W[:, :, :] = W0

    # We still need to initialize the rest of the matrix
    if n_src < n_chan:
        update_J_from_orth_const()
        for f in range(n_freq):
            W_hat[f, n_src:, n_src:] = -np.eye(n_chan - n_src)

    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    r_inv = np.zeros((n_frames, n_src))
    r = np.zeros((n_frames, n_src))

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
            r[:, :] = (2. * np.linalg.norm(Y, axis=0))
        elif model == 'gauss':
            r[:, :] = (np.linalg.norm(Y, axis=0) ** 2) / n_freq

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]

        if model == 'laplace':
            Y /= gamma[None, None, :]
            W /= gamma[None, None, :]
        elif model == 'gauss':
            g_sq = np.sqrt(gamma[None, None, :])
            Y /= g_sq
            W /= g_sq

        # ensure some numerical stability
        eps = 1e-15
        r[r < eps] = eps

        r_inv[:, :] = 1. / r

        # Update now the demixing matrix
        for s in range(n_src):
            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            V[:, :, :] = (X.swapaxes(1, 2) * r_inv[None, None, :, s]) @ np.conj(X) / n_frames

            WV = np.conj(W_hat).swapaxes(1, 2) @ V
            W[:, :, s] = np.linalg.solve(WV, eyes[:, :, s])

            # normalize
            denom = np.conj(W[:, None, :, s]) @ V[:, :, :] @ W[:, :, None, s]
            W[:, :, s] /= np.sqrt(denom[:, :, 0])

            # Update the mixing matrix according to orthogonal constraints
            if n_src < n_chan:
                update_J_from_orth_const()

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
