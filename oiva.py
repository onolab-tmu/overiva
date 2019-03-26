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
    model="laplace",
    init_eig=False,
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
        n_src = X.shape[2]

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    W_hat = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W = W_hat[:, :, :n_src]
    B = W_hat[:, :n_src, :n_src]
    C = W_hat[:, n_src:, :n_src]
    J = W_hat[:, :n_src, n_src:]

    def tensor_H(T):
        return np.conj(T).swapaxes(1, 2)

    def compute_L():
        return np.conj(B + np.matmul(J, C)).swapaxes(1, 2)

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
                eigval = v[f][ind]
                eigvec = np.conj(w[f][:, ind])
                # W[f, :, :] = eigvec / eigval[None, :]
                W[f, :, :] = eigvec
                # A[f, :, :] = eigvec * eigval[None, :]

        else:
            # Or with identity
            for f in range(n_freq):
                W[f, :n_src, :] = np.eye(n_src)

    else:
        W[:, :, :] = W0

    # We still need to initialize the rest of the matrix
    update_J_from_orth_const()
    for f in range(n_freq):
        W_hat[f, n_src:, n_src:] = -np.eye(n_chan - n_src)


    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))
    small_eyes = eyes[:, :n_src, :n_src]
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(W[f, :, :]))

    def cost_func(Y, r):

        # need to compute L from A
        L_inv = np.linalg.inv(compute_L())
        L = np.linalg.solve(L_inv, small_eyes)

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
        if model == "laplace":
            r[:, :] = np.sqrt(np.sum(np.abs(Y * np.conj(Y)), axis=1))
        elif model == "gauss":
            r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)
        else:
            raise ValueError("Only Gauss and Laplace models are supported")

        # set the scale of r
        gamma = r.mean(axis=0)
        r /= gamma[None, :]
        Y /= np.sqrt(gamma[None, None, :])
        W /= np.sqrt(gamma[None, None, :])

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
            # Update the Demixing matrix first
            W[:, :, s] = np.linalg.solve(
                np.matmul(tensor_H(W_hat), V[:, s, :, :]), eyes[:, :, s]
            )

            # normalize
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

            # Update the mixing matrix according to orthogonal constraints
            update_J_from_orth_const()

    demix(Y, X, W)

    # shape: (n_frames, n_src)
    r[:, :] = np.mean(np.abs(Y * np.conj(Y)), axis=1)

    if epoch % 3 == 0:
        the_cost.append(cost_func(Y, r))

    plt.figure()
    plt.plot(np.arange(len(the_cost)) * 3, the_cost)
    plt.title("The cost function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Neg. log-likelihood")

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
