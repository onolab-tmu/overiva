"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def tensor_diag(T):
    """
    T is a tensor of shape == (...,n,n)
    The function returns a view on the diagonals of all the squared matrices
    contained in the last dimensions.

    The returned array has shape (...,n)
    """
    assert T.ndim >= 2
    assert T.shape[-2] == T.shape[-1]
    new_shape = list(T.shape[:-1])
    new_strides = list(T.strides[:-1])
    new_strides[-1] += T.strides[-1]
    return np.lib.stride_tricks.as_strided(T, shape=new_shape, strides=new_strides)


def oiva4(
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

        # initialize A and W
        v, w = np.linalg.eig(Cx)
        for f in range(n_freq):
            ind = np.argsort(v[f])[-n_src:]
            eigval = v[f][ind]
            eigvec = np.conj(w[f][:, ind])
            A[f, :, :] = eigvec * eigval[None, :]
            W[f, :, :n_src] = eigvec / eigval[None, :]

            W[f, :n_src, :n_src] = np.eye(n_src)
            A[f, :n_src, :] = np.eye(n_src)

    else:
        assert W0.shape == (
            n_chan,
            n_src,
        ), "Mismatch in size of initial demixing matrix"
        W = W0.copy()
        A = np.zeros((n_freq, n_chan, n_src), dtype=X.dtype)

    # Views to the important bits of the demixing matrix
    Wout = W[:, :, :n_src]  # shape == (n_freq, n_chan, n_src)
    W_diag = tensor_diag(W[:, n_src:, n_src:])  # shape == (n_freq, n_chan - n_src)
    J = W[:, :n_src, n_src:]  # shape == (n_freq, n_src, n_chan - n_src)
    B = W[:, :n_src, :n_src]  # shape == (n_freq, n_src, n_src)
    C = W[:, n_src:, :n_src]  # shape == (n_freq, n_chan - n_src, n_src)

    # keep L^{-1} as a view on the top part of the mixing matrix
    L_inv = A[:, :n_src, :]

    # last part of the demixing matrix
    J[:, :, :] = np.linalg.solve(
        np.conj(L_inv).swapaxes(1, 2), np.conj(A[:, n_src:, :]).swapaxes(1, 2)
    )

    # other variables
    I = np.eye(n_src, n_src)
    I_chan = np.tile(np.eye(n_chan), (n_freq, 1, 1))
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    Z = np.zeros((n_frames, n_freq, n_chan - n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src + 1, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src + 1))

    # Compute the demixed output
    def demix():
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(Wout[f, :, :]))
            Z[:, f, :] = np.dot(X[:, f, :n_src], np.conj(J[f, :, :])) - X[:, f, n_src:]

    def update_L():
        tmp = np.conj(B + np.matmul(J, C)).swapaxes(1, 2)
        L_inv[:, :, :] = np.linalg.inv(tmp)

    def cost_func(r, A):
        """
        This should be called immediately after recomputing r
        """

        # need to compute L from A
        L_inv = A[:, :n_src, :]

        # now compute log det
        c1 = 2 * n_frames * np.sum(np.linalg.slogdet(L_inv)[1])

        # now compute the log of activations
        c2 = n_freq * (np.sum(np.log(r[:, :-1])) + 1)

        # last bit of the cost f
        c3 = n_freq * (n_chan - n_src) * (np.sum(np.log(r[:, -1])) + 1)

        return c1 + c2 + c3

    the_cost = []

    import matplotlib.pyplot as plt

    for epoch in range(n_iter):

        demix()

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:, :, 0])
                callback(Y * np.conj(z[None, :, :]))
            else:
                callback(Y)

        # shape: (n_frames, n_src-1)
        r[:, :-1] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

        # activations of background
        r[:, -1] = np.mean(np.abs(Z * np.conj(Z)), axis=(1, 2))
        # smoothing for the background
        # r[:,-1] = np.convolve(r[:,-1], np.ones(3), mode='same')

        if epoch % 3 == 0:
            the_cost.append(cost_func(r, A))
            print("{}: {}".format(epoch, the_cost[-1]))

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]
        Y /= np.sqrt(gamma[None, None, :-1])
        Wout /= np.sqrt(gamma[None, None, :-1])
        Z /= np.sqrt(gamma[-1])
        J /= np.sqrt(gamma[-1])
        W_diag /= np.sqrt(gamma[-1])

        eps = 1e-10
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
            Wout[:, :, s] = np.linalg.solve(
                np.matmul(np.conj(W.swapaxes(1, 2)), V[:, s, :, :])
                + np.eye(n_chan) * 1e-10,
                I_chan[:, :, s],
            )

            # normalize
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

        update_L()

        # Update the noise beamforming matrix
        C11 = V[:, -1, :n_src, :n_src]  # shape == (n_freq, n_src, n_src)
        C12 = V[:, -1, :n_src, n_src:]  # shape == (n_freq, n_src, n_chan - n_src)
        J[:, :, :] = np.linalg.solve(
            C11, np.matmul(L_inv, np.conj(C.swapaxes(1, 2))) + C12
        )

        # normalize the rest of the demixing matrix
        for s in range(n_src, n_chan):
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, -1, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

        update_L()

        if epoch % 10 == 0:
            plt.figure()
            plt.plot(r)
            plt.title("r {}".format(epoch))
            plt.legend(list(range(1, n_src + 1)) + ["noise"])

    demix()

    # shape: (n_frames, n_src)
    r[:, :-1] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

    # activations of background
    r[:, -1] = np.mean(np.abs(Z * np.conj(Z)), axis=(1, 2))

    if epoch % 3 == 0:
        the_cost.append(cost_func(r, A))

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
