"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def oiva(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    model='laplace',
    update_mix=False,
    return_filters=False,
    callback=None,
):

    """
    Implementation of overdetermined IVA algorithm for BSS

    Orthogonal constraints only

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
    update_mix: bool
        If set to to True, the algorithm will update the mixing matrix rather
        than demixing
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

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)
    Cx_inv = np.linalg.inv(Cx)

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

            '''
            W[f, :n_src, :] = np.eye(n_src)
            A[f, :n_src, :] = np.eye(n_src)
            '''

    else:
        assert W0.shape == (
            n_chan,
            n_src,
        ), "Mismatch in size of initial demixing matrix"
        W = W0.copy()
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(W[f, :, :]))

    def cost_func(Y, r, A):

        # need to compute L from A
        L_inv = A[:, :n_src, :]
        L = np.linalg.solve(L_inv, np.tile(I, (n_freq, 1, 1)))

        # now compute log det
        c1 = -2 * n_frames * np.sum(np.linalg.slogdet(L)[1])

        # now compute the log of activations
        c2 = n_freq * np.sum(np.log(r)) + n_freq

        return c1 + c2

    the_cost = []

    import matplotlib.pyplot as plt

    for epoch in range(n_iter):

        demix(Y, X, W)

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:, :, 0])
                callback(Y * np.conj(z[None, :, :]))
            else:
                callback(Y)

        # shape: (n_frames, n_src)
        if model == 'laplace':
            r[:, :] = np.sqrt(np.sum(np.abs(Y * np.conj(Y)), axis=1))
        elif model == 'gauss':
            r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)
        else:
            raise ValueError('Only Gauss and Laplace models are supported')

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]
        Y /= np.sqrt(gamma[None, None, :])
        W /= np.sqrt(gamma[None, None, :])
        A *= np.sqrt(gamma[None, None, :])

        eps = 1e-5
        r[r < eps] = eps

        # Compute Auxiliary Variable
        np.mean(
            (0.5 * X[:, :, None, :, None] / r[:, None, :, None, None])
            * np.conj(X[:, :, None, None, :]),
            axis=0,
            out=V,
        )

        # Update now the demixing matrix
        for s in range(n_src):
            if update_mix:
                # Update the Mixing matrix first
                # potentially less computations
                A[:, :, s] = np.matmul(V[:, s, :, :], W[:, :, s, None])[:, :, 0]

                # Update the demixing matrix according to orthogonal constraints
                rhs = np.linalg.inv(
                    np.matmul(np.conj(A.swapaxes(-2, -1)), np.matmul(Cx_inv, A))
                )
                np.matmul(Cx_inv, np.matmul(A, rhs), out=W)

                print(np.max(np.abs(np.eye(n_src) - np.dot(np.conj(W[50].T), A[50]))))

                # normalize
                P1 = np.conj(W[:, :, s])
                P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
                W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

            else:
                # Update the Demixing matrix first
                W[:, :, s] = np.linalg.solve(V[:, s, :, :], A[:, :, s])

                # normalize
                P1 = np.conj(W[:, :, s])
                P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
                W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

                # Update the mixing matrix according to orthogonal constraints
                rhs = np.linalg.inv(
                    np.matmul(np.conj(W.swapaxes(-2, -1)), np.matmul(Cx, W))
                )
                np.matmul(Cx, np.matmul(W, rhs), out=A)

    demix(Y, X, W)

    # shape: (n_frames, n_src)
    r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

    if epoch % 3 == 0:
        the_cost.append(cost_func(Y, r, A))

    plt.figure()
    plt.plot(np.arange(len(the_cost)) * 3, the_cost)
    plt.title("The cost function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Neg. log-likelihood")

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W, A
    else:
        return Y
