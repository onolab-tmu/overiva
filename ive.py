"""
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
"""
import numpy as np

from pyroomacoustics.bss import projection_back


def ogive(
    X,
    n_iter=4000,
    step_size=0.1,
    tol=1e-3,
    update="demix",
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
    update: str
        Selects update of the mixing or demixing matrix, or a switching scheme,
        possible values: "mix", "demix", "switching"
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
    Cx_norm = np.linalg.norm(Cx, axis=(1, 2))

    w = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    a = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    delta = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    lambda_a = np.zeros((n_freq, 1, 1), dtype=np.float64)

    def tensor_H(T):
        return np.conj(T).swapaxes(1, 2)

    # eigenvectors of the input covariance
    eigval, eigvec = np.linalg.eig(Cx)
    lead_eigval = np.max(eigval, axis=1)
    lead_eigvec = np.zeros((n_freq, n_chan), dtype=Cx.dtype)
    for f in range(n_freq):
        ind = np.argmax(eigval[f])
        lead_eigvec[f, :] = eigvec[f, :, ind]

    # initialize A and W
    if W0 is None:
        if init_eig:

            # Initialize the demixing matrices with the principal
            # eigenvector
            w[:, :, 0] = np.conj(lead_eigvec)

        else:
            # Or with identity
            w[:, 0] = 1.0

    else:
        w[:, :] = W0

    if update == "mix":
        I_do_w = np.zeros(n_freq, dtype=np.bool)
        I_do_a = np.ones(n_freq, dtype=np.bool)
    else:  # default is "demix"
        I_do_w = np.ones(n_freq, dtype=np.bool)
        I_do_a = np.zeros(n_freq, dtype=np.bool)

    def update_a_from_w(I):
        v_new = Cx[I] @ w[I]
        lambda_w = 1. / np.real(tensor_H(w[I]) @ v_new)
        a[I, :, :] = lambda_w * v_new

    def update_w_from_a(I):
        v_new = Cx_inv[I] @ a[I]
        lambda_a[I] = 1. / np.real(tensor_H(a[I]) @ v_new)
        w[I,:,:] = lambda_a[I] * v_new

    # The very first update of a
    update_a_from_w(np.ones(n_freq, dtype=np.bool))
    update_w_from_a(np.ones(n_freq, dtype=np.bool))

    # import pdb; pdb.set_trace()

    r_inv = np.zeros((n_frames, n_src))
    r = np.zeros((n_frames, n_src))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    X_ref = X  # keep a reference to input signal
    X = X.swapaxes(0, 1).copy()  # more efficient order for processing
    X /= np.sqrt(lead_eigval[:, None, None])  # normalize amplitudes of channels

    def switching_criterion():

        a_n = a / a[:, :1, :1]
        b_n = Cx @ a_n
        lmb = b_n[:, :1, :1]
        b_n /= lmb;

        p1 = np.linalg.norm(a_n - b_n, axis=(1, 2))
        Cbb = lmb * (b_n @ tensor_H(b_n)) / np.linalg.norm(b_n, axis=(1, 2), keepdims=True) ** 2
        p2 = np.linalg.norm(Cx - Cbb, axis=(1, 2))

        kappa = p1 * p2 / Cx_norm / np.sqrt(n_chan)

        I_do_a[:] = kappa < 0.1
        I_do_w[:] = kappa >= 0.1

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = X @ np.conj(W)

    for epoch in range(n_iter):
        # compute the switching criterion
        if update == "switching" and epoch % 10 == 0:
            switching_criterion()

        # Apply the orthogonal constraints
        update_a_from_w(I_do_w)
        update_w_from_a(I_do_a)
        
        # Extract the target signal
        demix(Y, X, w)

        # Now run any necessary callback
        if callback is not None and epoch % 200 == 0:
            Y_tmp = Y.swapaxes(0, 1)
            if proj_back:
                z = projection_back(Y_tmp, X_ref[:, :, 0])
                callback(Y_tmp * np.conj(z[None, :, :]))
            else:
                callback(Y_tmp)

        # simple loop as a start
        # shape: (n_frames, n_src)
        if model == "laplace":
            r[:, :] = np.linalg.norm(Y, axis=0)

        elif model == "gauss":
            r[:, :] = (np.linalg.norm(Y, axis=0) ** 2) / n_freq

        eps = 1e-15
        r[r < eps] = eps

        r_inv[:, :] = 1.0 / r

        # Compute the score function
        psi = r_inv[None, :, :] * np.conj(Y)

        # "Nu" in Algo 3 in [1]
        # shape (n_freq, 1, 1)
        zeta = Y.swapaxes(1, 2) @ psi / n_frames

        x_psi = (X.swapaxes(1, 2) @ psi) / zeta / n_frames

        # The w-step
        # shape (n_freq, n_chan, 1)
        delta[I_do_w] = a[I_do_w] - x_psi[I_do_w]
        w[I_do_w] += step_size * delta[I_do_w]

        # The a-step
        # shape (n_freq, n_chan, 1)
        delta[I_do_a] = w[I_do_a] - (Cx_inv[I_do_a] @ x_psi[I_do_a]) * lambda_a[I_do_a]
        a[I_do_a] += step_size * delta[I_do_a]

        max_delta = np.max(np.linalg.norm(delta, axis=(1, 2)))

        if max_delta < tol:
            break

    # Apply the orthogonal constraints
    update_a_from_w(I_do_w)
    update_w_from_a(I_do_a)
        
    # Extract target
    demix(Y, X, w)

    Y = Y.swapaxes(0, 1).copy()
    X = X.swapaxes(0, 1)

    if proj_back:
        z = projection_back(Y, X_ref[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, w
    else:
        return Y
