"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def oiva3(
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
    if W0 is None:

        W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

        W += np.eye(n_chan)[None, :, :]
        W[:, n_src:, n_src:] *= -1
        A[:, :n_src, :] += np.eye(n_src)[None, :, :]

        # initialize A and W with PCA
        """
        v,w = np.linalg.eig(Cx)
        for f in range(n_freq):
            ind = np.argsort(v[f])[-n_src:]
            eigval = v[f][ind]
            eigvec = np.conj(w[f][:,ind])
            A[f,:,:] = eigvec * eigval[None,:]
            W[f,:,:n_src] = eigvec / eigval[None,:]
        """

    else:
        assert W0.shape == (
            n_chan,
            n_src,
        ), "Mismatch in size of initial demixing matrix"
        W = W0.copy()
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

    Wout = W[:, :, :n_src]
    J = W[:, :n_src, n_src:]

    L_inv = A[:, :n_src, :].swapaxes(1, 2)
    JL_inv = A[:, n_src:, :].swapaxes(1, 2)

    I_src = np.tile(np.eye(n_src), (n_freq, 1, 1))
    I_chan = np.tile(np.eye(n_chan), (n_freq, 1, 1))
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(Wout[f, :, :]))

    def update_J(J, A):
        L_inv = np.conj(A[:, :n_src, :].swapaxes(1, 2))
        JL_inv = np.conj(A[:, :n_src, :].swapaxes(1, 2))
        J[:, :, :] = np.linalg.solve(L_inv, JL_inv)

    def cost_func(Y, r, A):

        # compute log det
        c1 = 2 * n_frames * np.sum(np.linalg.slogdet(L_inv)[1])

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
        r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

        if epoch % 3 == 0:
            the_cost.append(cost_func(Y, r, A))
            print("{}: {}".format(epoch, the_cost[-1]))

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]
        Y /= np.sqrt(gamma[None, None, :])
        Wout /= np.sqrt(gamma[None, None, :])
        A *= np.sqrt(gamma[None, None, :])

        eps = 1e-5
        r[r < eps] = eps

        if epoch % 10 == 0:
            plt.figure()
            plt.plot(r)
            plt.title("r {}".format(epoch))

        if epoch % 3 == 0:
            after_scaling = cost_func(Y, r, A)
            print("  after scaling: {}".format(after_scaling))

        # Compute Auxiliary Variable
        np.mean(
            (0.5 * X[:, :, None, :, None] / r[:, None, :, None, None])
            * np.conj(X[:, :, None, None, :]),
            axis=0,
            out=V,
        )

        # Update now the demixing matrix
        errs = []
        for s in range(n_src):
            W[:, :, s] = np.linalg.solve(
                np.matmul(np.conj(W[:, :, :].swapaxes(1, 2)), V[:, s, :, :]),
                I_chan[:, :, s],
            )

            # normalize
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

            # Update the mixing matrix according to orthogonal constraints
            u_left = np.matmul(
                np.conj(Wout.swapaxes(-2, -1)), np.matmul(Cx, Wout)
            ).swapaxes(1, 2)
            u_right = np.matmul(Cx, Wout).swapaxes(1, 2)
            A[:, :, :] = np.linalg.solve(u_left, u_right).swapaxes(1, 2)

            # Update J
            J[:, :, :] = np.linalg.solve(np.conj(L_inv), np.conj(JL_inv))

        if epoch % 3 == 0:
            wha = np.matmul(np.conj(Wout.swapaxes(-2, -1)), A)
            const_goodness = np.linalg.norm(
                wha - np.eye(n_src)[None, :, :], axis=(1, 2)
            )
            num = len(np.where(const_goodness > 1e-1)[0])
            print(
                "Const: max err: {}, numbers > 1e-1: {}".format(
                    np.max(const_goodness), num
                )
            )

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
        print("proj back!")
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W, A
    else:
        return Y
