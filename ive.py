"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def ogive(
    X,
    n_iter=4000,
    step_size=1e-4,
    tol=1e-3,
    proj_back=True,
    W0=None,
    model="laplace",
    init_eig=False,
    return_filters=False,
    callback=None,
):

    """
    Implementation of Orthogonally constrained Independent Vector Analysis



    Orthogonal constraints only

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    step_size: float
        The step size of the gradient ascent
    tol: float
        Stop when the gradient is smaller than this number
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
    n_src = 1

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)
    Cx_inv = np.linalg.inv(Cx)

    w = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    a = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)

    def tensor_H(T):
        return np.conj(T).swapaxes(1, 2)

    # initialize A and W
    if W0 is None:
        if init_eig:
            # eigenvectors of the input covariance
            eigval, eigvec = np.linalg.eig(Cx)
            # lead_eigval = np.max(eigval, axis=1)
            lead_eigvec = np.zeros((n_freq, n_chan), dtype=Cx.dtype)
            for f in range(n_freq):
                ind = np.argmax(eigval[f])
                lead_eigvec[f, :] = eigvec[f, :, ind]

            # Initialize the demixing matrices with the principal
            # eigenvector
            w[:, :, 0] = np.conj(lead_eigvec)

        else:
            # Or with identity
            w[:, 0] = 1.

    else:
        w[:, :] = W0

    def update(v1, v2, C):
        v_new = C @ v1
        denom = tensor_H(v1) @ v_new
        v2[:, :, :] = v_new / denom

    def update_a_from_w():
        update(w, a, Cx)

    def update_w_from_a():
        update(a, w, Cx_inv)

    r_inv = np.zeros((n_frames, n_src))
    r = np.zeros((n_frames, n_src))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    X = X.swapaxes(0, 1).copy()

    '''
    def switching_criterion():

        b = Cx @ 
        lmb = 
    '''

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = X @ np.conj(W)

    for epoch in range(n_iter):

        demix(Y, X, w)

        if callback is not None and epoch % 50 == 0:
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
        Y /= np.sqrt(gamma[None, None, :])
        w /= np.sqrt(gamma[None, None, :])

        eps = 1e-15
        r[r < eps] = eps

        r_inv[:, :] = 1. / r

        # apply OGIVE_w update (Algorithm 3 in [1])
        update_a_from_w()

        # "Nu" in Algo 3 in [1]
        # shape (n_freq, 1, 1)
        nu = (Y.swapaxes(1, 2) * r_inv[None, None, :, 0]) @ np.conj(Y) / n_frames

        # The step
        # shape (n_freq, n_chan, 1)
        delta = a - (X.swapaxes(1, 2) * r_inv[None, None, :, 0]) @ np.conj(Y) / nu / n_frames

        w[:, :, :] += step_size * delta

        max_delta = np.max(np.linalg.norm(delta, axis=1))

        if epoch % 10 == 0:
            print(max_delta)

        if max_delta < tol:
            break

    demix(Y, X, w)

    Y = Y.swapaxes(0, 1).copy()
    X = X.swapaxes(0, 1)

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, w
    else:
        return Y
