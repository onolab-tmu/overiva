"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

Implementation on GPU with cupy

2019 (c) Robin Scheibler, MIT License
"""
import numpy as np
import cupy as cp


def projection_back(Y, ref):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Parameters
    ----------
    Y: array_like (n_freq, n_frames, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_freq, n_frames)
        The reference signal
    """

    num = cp.sum(cp.conj(ref[:, :, None]) * Y, axis=1)
    denom = cp.sum(cp.abs(Y) ** 2, axis=1)

    c = cp.ones(num.shape, dtype=cp.complex)
    nz = denom > 0.0
    c[nz] = num[nz] / denom[nz]

    return c


def auxiva_gpu(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    return_filters=False,
    callback=None,
    gpu_id=0,
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
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence
    gpu_id: int
        The GPU index to use

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    cp.cuda.Device(gpu_id).use()

    n_frames, n_freq, n_chan = X.shape

    X_cpu = X
    X = cp.asarray(X_cpu.swapaxes(0, 1))  # new shape (n_freq, n_frames, n_chan)

    # default to determined case
    if n_src is None:
        n_src = n_chan

    # for now, only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    if W0 is None:
        W = cp.array([cp.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    eyes = cp.array([cp.eye(n_src, n_src) for f in range(n_freq)], dtype=X.dtype)
    Y = cp.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    r = cp.zeros((n_frames, n_src))
    V = cp.zeros((n_freq, n_src, n_src, n_src), dtype=X.dtype)

    # tmp var
    W_H = cp.zeros((n_freq, n_src, n_src), dtype=X.dtype)
    WV = cp.zeros((n_freq, n_src, n_src), dtype=X.dtype)
    P1 = cp.zeros((n_freq, n_src), dtype=X.dtype)
    P2 = cp.zeros((n_freq, n_src), dtype=X.dtype)

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = cp.matmul(X, cp.conj(W))

    for epoch in range(n_iter):

        demix(Y, X, W)

        # simple loop as a start
        # shape: (n_frames, n_src)
        r[:, :] = cp.linalg.norm(Y, axis=0)

        # Compute Auxiliary Variable
        cp.mean(
            (X[:, :, None, :, None] / r[None, :, :, None, None])
            * cp.conj(X[:, :, None, None, :]),
            axis=1,
            out=V,
        )

        # Update now the demixing matrix
        for s in range(n_src):
            W_H[:, :, :] = cp.conj(cp.swapaxes(W, 1, 2))
            WV[:, :, :] = cp.matmul(W_H, V[:, s, :, :])
            W[:, :, s] = cp.linalg.solve(WV, eyes[:, :, s])

            # normalize
            P1[:, :] = cp.conj(W[:, :, s])
            P2[:, :] = cp.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= cp.sqrt(cp.sum(P1 * P2, axis=1))[:, None]

    demix(Y, X, W)

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= cp.conj(z[:, None, :])

    Y_cpu = cp.asnumpy(Y.swapaxes(0, 1))

    if return_filters:
        W_cpu = cp.asnumpy(W)
        return Y_cpu, W_cpu
    else:
        return Y_cpu
