"""
Blind Source Separation using Independent Low-Rank Matrix Analysis (ILRMA)

2018 (c) Juan Azcarreta, Robin Scheibler, MIT License
"""
import numpy as np
from pyroomacoustics.bss import projection_back


def oilrma(
    X,
    n_src=None,
    n_iter=20,
    proj_back=False,
    W0=None,
    n_components=2,
    return_filters=0,
    callback=None,
):

    """
    Implementation of ILRMA algorithm without partitioning function for BSS presented in

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, September 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined Blind Source Separation
    with Independent Low-Rank Matrix Analysis*, in Audio Source Separation, S. Makino, Ed. Springer, 2018, pp.  125-156.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    n_components: int
        Number of components in the non-negative spectrum
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix W (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    # initialize the demixing matrices
    if W0 is None:

        W = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

        # initialize A and W
        v, w = np.linalg.eig(Cx)
        for f in range(n_freq):
            ind = np.argsort(v[f])[-n_src:]
            eigval = v[f][ind]
            eigvec = np.conj(w[f][:, ind])
            A[f, :, :] = eigvec * eigval[None, :]
            W[f, :, :] = eigvec / eigval[None, :]

            W[f, :n_src, :] = np.eye(n_src)
            A[f, :n_src, :] = np.eye(n_src)

    else:
        assert W0.shape == (
            n_chan,
            n_src,
        ), "Mismatch in size of initial demixing matrix"
        W = W0.copy()
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

    # initialize the nonnegative matrixes with random values
    T = np.array(np.random.rand(n_freq, n_components, n_src))
    V = np.array(np.random.rand(n_components, n_frames, n_src))
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    R = np.zeros((n_freq, n_frames, n_src))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    lambda_aux = np.zeros(n_src)
    machine_epsilon = np.finfo(float).eps

    for n in range(0, n_src):
        R[:, :, n] = np.dot(T[:, :, n], V[:, :, n])

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(W[f, :, :]))

    demix(Y, X, W)
    P = np.power(abs(Y), 2.0)

    for epoch in range(n_iter):

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:, :, 0])
                callback(Y * np.conj(z[None, :, :]))
            else:
                callback(Y)

        # simple loop as a start
        for s in range(n_src):
            iR = 1 / R[:, :, s]
            T[:, :, s] *= np.sqrt(
                np.dot(P[:, :, s].T * iR ** 2, V[:, :, s].T) / np.dot(iR, V[:, :, s].T)
            )
            T[T < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            iR = 1 / R[:, :, s]
            V[:, :, s] *= np.sqrt(
                np.dot(T[:, :, s].T, P[:, :, s].T * iR ** 2) / np.dot(T[:, :, s].T, iR)
            )
            V[V < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            # Compute Auxiliary Variable and update the demixing matrix
            for f in range(n_freq):
                U[f, s, :, :] = (
                    np.dot(X[:, f, :].T, np.conj(X[:, f, :]) / R[f, :, None, s])
                    / n_frames
                )
                W[f, :, s] = np.linalg.solve(U[f, s, :, :], A[f, :, s])
                w_Unorm = np.inner(
                    np.conj(W[f, :, s]), np.dot(U[f, s, :, :], W[f, :, s])
                )
                W[f, :, s] /= np.sqrt(w_Unorm)

                # Update the mixing matrix according to orthogonal constraints
                rhs = np.linalg.inv(
                        np.dot(np.conj(W[f].T), np.dot(Cx[f], W[f]))
                )
                np.dot(Cx[f], np.matmul(W[f], rhs), out=A[f])

        demix(Y, X, W)
        P = np.abs(Y) ** 2

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[:, :, s]))

            W[:, :, s] *= lambda_aux[s]
            A[:, :, s] /= lambda_aux[s]
            P[:, :, s] *= lambda_aux[s] ** 2
            R[:, :, s] *= lambda_aux[s] ** 2
            T[:, :, s] *= lambda_aux[s] ** 2

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
