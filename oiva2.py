"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back

from scipy.optimize import root


def oiva2(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    f_contrast=None,
    f_contrast_args=[],
    return_filters=False,
    callback=None,
):

    """
    Implementation of overdetermined IVA algorithm for BSS

    Gaussian background only

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

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    # initialize the demixing matrices
    W_hat = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W = W_hat[:, :, :n_src]  # a view to the usefule part
    # views on blocks of W
    B = W_hat[:, :n_src, :n_src]
    C = W_hat[:, n_src:, :n_src]
    J = W_hat[:, :n_src, n_src:]

    def compute_L():
        return np.conj(B + np.matmul(J, C)).swapaxes(1, 2)

    # initialize A and W
    v, w = np.linalg.eig(Cx)
    L = compute_L()
    for f in range(n_freq):
        ind = np.argsort(v[f])[-n_src:]
        eigval = v[f][ind]
        eigvec = np.conj(w[f][:, ind])
        W[f, :, :] = eigvec / eigval[None, :]
        A = eigvec * eigval[None, :]

        W_hat[f, n_src:, n_src:] = -np.eye(n_chan - n_src)

        J[f, :, :] = np.conj(np.dot(A[n_src:, :], L[f])).T

        """
        W[f, :n_src, :] = np.eye(n_src)
        """

    # shape == (n_freq, n_src, n_src)
    C11 = 0.5 * np.mean(X[:, :, :n_src, None] * np.conj(X[:, :, None, :n_src]), axis=0)
    # shape == (n_freq, n_src, n_chan - n_src)
    C12 = 0.5 * np.mean(X[:, :, :n_src, None] * np.conj(X[:, :, None, n_src:]), axis=0)

    # other variables
    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(W[f, :, :]))

    def update_j(RX1, RX2, m):
        # J is shape (n_freq, n_src, n_chan - n_src)

        c_m = np.conj(C[:, m, :])
        L_m = np.conj(
            (
                B
                + np.matmul(J[:, :, :m], C[:, :m, :])
                + np.matmul(J[:, :, m + 1 :], C[:, m + 1 :, :])
            ).swapaxes(1, 2)
        )
        r_m = RX2[:, :, m]

        # apply the solution derived
        v = np.linalg.solve(L_m, c_m)  # shape (n_freq, n_src)
        g = np.linalg.solve(RX1, v)  # shape (n_freq, n_src)
        h = np.linalg.solve(RX1, r_m)  # shape (n_freq, n_src)
        nu = np.matmul(np.conj(g[:, None, :]), r_m[:, :, None])  # shape (n_freq, 1, 1)
        eta = np.real(
            np.matmul(np.conj(g[:, None, :]), v[:, :, None])
        )  # shape (n_freq, 1, 1)

        i1 = np.isclose(eta[:, 0, 0], -np.ones(n_freq))
        J[i1, :, m] = h[i1, :] + g[i1, :] / np.sqrt(eta[i1, :, 0])

        i2 = np.logical_not(i1)
        r1 = (1 + nu[i2, :, 0]) / eta[i2, :, 0]
        r2 = 1.0 / np.real(r1 * (1.0 + np.conj(nu[i2, :, 0])))
        beta2 = 0.5 * r1 * (-1 + np.sqrt(1 + 4 * r2))
        J[i2, :, m] = h[i2, :] + beta2 * g[i2, :]

    def cost_func(r):

        L = compute_L()

        # now compute log det
        c1 = -2 * n_frames * np.sum(np.linalg.slogdet(L)[1])

        # now compute the log of activations
        c2 = n_freq * np.sum(np.log(r)) + n_freq

        c3 = 0.0
        for s in range(n_chan - n_src):
            j_m_H = np.conj(J[:, None, :, s])
            j_m = J[:, :, None, s]
            c3 += np.real(np.matmul(j_m_H, np.matmul(C11, j_m))).sum()
            c3 -= np.real(np.matmul(j_m_H, C12[:, :, None, s])).sum()

        return c1 + c2 + c3

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
        r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

        if epoch % 3 == 0:
            the_cost.append(cost_func(r))
            print("{}: {}".format(epoch, the_cost[-1]))

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]
        Y /= np.sqrt(gamma[None, None, :])
        W /= np.sqrt(gamma[None, None, :])

        eps = 1e-5
        r[r < eps] = eps

        if epoch % 10 == 0:
            plt.figure()
            plt.plot(r)
            plt.title("r {}".format(epoch))

        if epoch % 3 == 0:
            after_scaling = cost_func(r)
            print("  after scaling: {}".format(after_scaling))

        # Compute Auxiliary Variable
        np.mean(
            (0.5 * X[:, :, None, :, None] / r[:, None, :, None, None])
            * np.conj(X[:, :, None, None, :]),
            axis=0,
            out=V,
        )

        # Update now the demixing matrix
        for s in range(n_src):
            W[:, :, s] = np.linalg.solve(
                np.matmul(np.conj(W_hat).swapaxes(1, 2), V[:, s, :, :]),
                np.tile(np.eye(n_chan)[:, s], (n_freq, 1)),
            )

            # normalize
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

        # Update the noise beamforming matrix
        for s in range(n_chan - n_src):
            update_j(C11, C12, s)

        """
        if epoch % 3 == 0:
            wha = np.matmul(np.conj(W.swapaxes(-2, -1)), A)
            const_goodness = np.linalg.norm(
                wha - np.eye(n_src)[None, :, :], axis=(1, 2)
            )
            num = len(np.where(const_goodness > 1e-1)[0])
            print(
                "Const: max err: {}, numbers > 1e-1: {}".format(
                    np.max(const_goodness), num
                )
            )
        """

    demix(Y, X, W)

    # shape: (n_frames, n_src)
    r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

    if epoch % 3 == 0:
        the_cost.append(cost_func(r))

    plt.figure()
    plt.plot(np.arange(len(the_cost)) * 3, the_cost)
    plt.title("The cost function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Neg. log-likelihood")

    if proj_back:
        print("proj back!")
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
