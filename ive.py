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
Blind Source Extraction using Independent Vector Extraction via the OGIVE algorithm [1].

[1]	Z. Koldovský and P. Tichavský, “Gradient Algorithms for Complex
Non-Gaussian Independent Component/Vector Extraction, Question of Convergence,”
IEEE Trans. Signal Process., pp. 1050–1064, Dec. 2018.
"""
import os
import numpy as np

from pyroomacoustics.bss import projection_back

import matwrap


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
    Implementation of Orthogonally constrained Independent Vector Extraction
    (OGIVE) described in

    Z. Koldovský and P. Tichavský, “Gradient Algorithms for Complex
    Non-Gaussian Independent Component/Vector Extraction, Question of Convergence,”
    IEEE Trans. Signal Process., pp. 1050–1064, Dec. 2018.

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
            w[:, :, 0] = lead_eigvec

        else:
            # Or with identity
            w[:, 0] = 1.0

    else:
        w[:, :] = W0

    def update_a_from_w(I):
        v_new = Cx[I] @ w[I]
        lambda_w = 1.0 / np.real(tensor_H(w[I]) @ v_new)
        a[I, :, :] = lambda_w * v_new

    def update_w_from_a(I):
        lambda_a[:] = 0.
        v_new = Cx_inv[I] @ a[I]
        lambda_a[I] = 1.0 / np.real(tensor_H(a[I]) @ v_new)
        w[I, :, :] = lambda_a[I] * v_new

    def switching_criterion():

        a_n = a / a[:, :1, :1]
        b_n = Cx @ a_n
        lmb = b_n[:, :1, :1].copy()  # copy is important here!
        b_n /= lmb

        p1 = np.linalg.norm(a_n - b_n, axis=(1, 2)) / Cx_norm
        Cbb = (
            lmb
            * (b_n @ tensor_H(b_n))
            / np.linalg.norm(b_n, axis=(1, 2), keepdims=True) ** 2
        )
        p2 = np.linalg.norm(Cx - Cbb, axis=(1, 2))

        kappa = p1 * p2 / np.sqrt(n_chan)

        thresh = 0.1
        I_do_a[:] = kappa >= thresh
        I_do_w[:] = kappa < thresh

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = X @ np.conj(W)

    # The very first update of a
    update_a_from_w(np.ones(n_freq, dtype=np.bool))

    if update == "mix":
        I_do_w = np.zeros(n_freq, dtype=np.bool)
        I_do_a = np.ones(n_freq, dtype=np.bool)
    else:  # default is "demix"
        I_do_w = np.ones(n_freq, dtype=np.bool)
        I_do_a = np.zeros(n_freq, dtype=np.bool)

    r_inv = np.zeros((n_frames, n_src))
    r = np.zeros((n_frames, n_src))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    X_ref = X  # keep a reference to input signal
    X = X.swapaxes(0, 1).copy()  # more efficient order for processing

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
        if callback is not None and epoch % 100 == 0:
            Y_tmp = Y.swapaxes(0, 1)
            if proj_back:
                z = projection_back(Y_tmp, X_ref[:, :, 0])
                callback(Y_tmp * np.conj(z[None, :, :]))
            else:
                callback(Y_tmp)

        # simple loop as a start
        # shape: (n_frames, n_src)
        if model == "laplace":
            r[:, :] = np.linalg.norm(Y, axis=0) / np.sqrt(n_freq)

        elif model == "gauss":
            r[:, :] = (np.linalg.norm(Y, axis=0) ** 2) / n_freq

        eps = 1e-15
        r[r < eps] = eps

        r_inv[:, :] = 1.0 / r

        # Compute the score function
        psi = r_inv[None, :, :] * np.conj(Y)

        # "Nu" in Algo 3 in [1]
        # shape (n_freq, 1, 1)
        zeta = Y.swapaxes(1, 2) @ psi

        x_psi = (X.swapaxes(1, 2) @ psi) / zeta

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


def ogive_matlab_wrapper(
    X,
    n_iter=4000,
    step_size=0.1,
    tol=1e-3,
    update="demix",
    proj_back=True,
    W0=None,
    init_eig=False,
    callback=None,
    ogive_folder="./OGIVEalgorithms",
):

    """
    Wrapper around the original MATLAB implementation of Orthogonally constrained Independent Vector Extraction
    (OGIVE) by Z. Koldovský and P. Tichavský described in

    Z. Koldovský and P. Tichavský, “Gradient Algorithms for Complex
    Non-Gaussian Independent Component/Vector Extraction, Question of Convergence,”
    IEEE Trans. Signal Process., pp. 1050–1064, Dec. 2018.

    A pre-requisite is to have the MATLAB scripts from here: `here <https://asap.ite.tul.cz/wp-content/uploads/sites/3/2018/10/OGIVEalgorithms.zip>`__
    This function will automatically try to download them into the folder ``OGIVEalgorithms`` if not available.

    This function also uses the Python -> MATLAB interface provided by MathWorks.
    Please follow the `instructions <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`__
    to get started. Of course, MATLAB needs to be available for all this to work.

    This wrapper was mainly used to verify that the Python implementation runs as expected.
    We recommend to use the Python version as it is faster.

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
    init_eig: bool, optional (default ``False``)
        If ``True``, and if ``W0 is None``, then the weights are initialized
        using the principal eigenvectors of the covariance matrix of the input
        data.
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence
    ogive_folder: str
        Path to the location of the MATLAB implementation

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    if not os.path.exists(ogive_folder):
        from urllib.request import urlopen
        from io import BytesIO
        from zipfile import ZipFile
        data_url = "https://asap.ite.tul.cz/wp-content/uploads/sites/3/2018/10/OGIVEalgorithms.zip"
        zf = ZipFile(BytesIO(urlopen(data_url).read()))
        zf.extractall(ogive_folder)

    # initial callback (mixture)
    if callback is not None:
        Y = X.copy()
        if proj_back:
            z = projection_back(Y, X[:, :, 0])
            Y *= np.conj(z[None, :, :])
        callback(Y)

    n_frames, n_freq, n_chan = X.shape

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)
    Cx_inv = np.linalg.inv(Cx)
    Cx_norm = np.linalg.norm(Cx, axis=(1, 2))

    # demixing and mixing vectors
    w = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    a = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)

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
            w[:, :, 0] = lead_eigvec

        else:
            # Or with identity
            w[:, 0] = 1.0

    else:
        w[:, :] = W0

    # compute initial mixing vector from demixing vector
    v_new = Cx @ w
    lambda_w = 1.0 / np.real(tensor_H(w) @ v_new)
    a[:, :, :] = lambda_w * v_new

    with matwrap.connect_matlab() as eng:
        # add path to Zbynek's functions
        eng.addpath(ogive_folder)

        # function [w, a, shat, NumIt] = ogive_a(x, mu, aini, MaxIt, nonln)
        # [d, N, M] = size(x); shape = [microphones, samples, frequencies]
        # we need to convert the array format
        X_matlab = matwrap.ndarray_to_matlab(X.transpose([2, 0, 1]))
        # initial value for a
        aini = matwrap.ndarray_to_matlab(a[:, :, 0].T)

        if update == "switching":
            # Run the MATLAB versio no OGIVE a written by Zbynek
            w, a, shat, numit = eng.ogive_s(X_matlab, step_size, aini, n_iter, 'sign',nargout=4)
        elif update == "mix":
            # Run the MATLAB versio no OGIVE a written by Zbynek
            w, a, shat, numit = eng.ogive_a(X_matlab, step_size, aini, n_iter, 'sign', nargout=4)
        elif update == "demix":
            # Run the MATLAB versio no OGIVE w written by Zbynek
            w, a, shat, numit = eng.ogive_w(X_matlab, step_size, aini, n_iter, 'sign', nargout=4)
        else:
            raise ValueError(f"Unknown update type {update}")

        # Now convert back the output (shat, shape=(n_freq, n_frames)
        Y = np.array(shat)
        Y = Y[:, :, None].transpose([1, 0, 2]).copy()

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if callback is not None:
        callback(Y)

    return Y
