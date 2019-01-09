'''
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
'''
import numpy as np
import math

from pyroomacoustics.bss import projection_back

# A few contrast functions
f_contrasts = {
        'norm' : { 'f' : (lambda r,c,m : c * r), 'df' : (lambda r,c,m : c) },
        'cosh' : { 'f' : (lambda r,c,m : m * np.log(np.cosh(c * r))), 'df' : (lambda r,c,m : c * m * np.tanh(c * r)) }
        }

def oiva_group(X, n_src=None, n_iter=20, n_sup_iter=1, proj_back=True, W0=None,
        f_contrast=None, f_contrast_args=[],
        return_filters=False, callback=None):

    '''
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
    '''

    n_frames, n_freq, n_chan = X.shape

    assert n_chan >= n_src, 'Number of channels should be more or equal to number of sources'

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # We form n_src groups of channels
    assert n_chan % n_src == 0, 'For now, n_chan should be a multiple of n_src'
    n_groups = math.ceil(n_chan / n_src)
    groups = np.arange(n_chan).reshape((-1, n_src), order='C')

    # we are going to modify X in-place
    X_original = X
    X_g_o = np.zeros((n_groups, n_frames, n_freq, n_src), dtype=X.dtype)
    for g, grp in enumerate(groups):
        X_g_o[g,:,:,:] = X[:,:,grp]
    X = X_g_o.copy()

    # initialize the demixing matrices
    if W0 is None:
        W0 = np.tile(np.eye(n_src, dtype=X.dtype), (n_groups, n_freq, 1, 1))
    W = W0.copy()

    # W_comp will be the composition of several matrices
    W_comp = W.copy()

    # Compute the demixed output
    def demix(Y, X, W):
        # W.shape == (n_groups, n_freq, n_src, n_src)
        # X.shape == (n_groups, n_frames, n_freq, n_src)
        # Y.shape == (n_groups, n_frames, n_freq, n_src)
        for g in range(n_groups):
            #np.sum(X[g,:,:,None,:] * np.conj(W[g,None,:,:,:]), axis=-1, out=Y[g])
            for f in range(n_freq):
                Y[g,:,f,:] = np.dot(X[g,:,f,:], np.conj(W[g,f,:,:]))

    import matplotlib.pyplot as plt

    I = np.eye(n_src,n_src)
    Y = np.zeros((n_groups, n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_groups, n_freq, n_src, n_src, n_src), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))  # activations are shared by groups!
    G_r = np.zeros((n_frames, n_src))

    for sup_epoch in range(n_sup_iter):

        for epoch in range(n_iter):

            demix(Y, X, W)

            if callback is not None and epoch % 10 == 0:
                # select the first group
                if proj_back:
                    z = projection_back(Y[0], X_g_o[0,:,:,0])
                    callback(Y[0] * np.conj(z[None,:,:]))
                else:
                    callback(Y[0])

            # simple loop as a start
            # shape: (n_frames, n_src)
            r[:,:] = np.sqrt(np.mean(np.abs(Y * np.conj(Y)), axis=(0,2)))
            r[r < 1e-10] = 1e-10

            # Apply derivative of contrast function
            G_r[:,:] = 0.5 / r  # shape (n_frames, n_src)

            # Compute Auxiliary Variable
            np.mean(
                    (X[:,:,:,None,:,None] * G_r[None,:,None,:,None,None])
                    * np.conj(X[:,:,:,None,None,:]),
                    axis=1,
                    out=V,
                    )

            # Update now the demixing matrix
            for g in range(n_groups):
                for s in range(n_src):
                    W_H = np.conj(np.swapaxes(W[g], 1, 2))
                    WV = np.matmul(W_H, V[g,:,s,:,:])
                    rhs = I[None,:,s][[0] * WV.shape[0],:]
                    W[g,:,:,s] = np.linalg.solve(WV, rhs)

                    # normalize
                    P1 = np.conj(W[g,:,:,s])
                    P2 = np.sum(V[g,:,s,:,:] * W[g,:,None,:,s], axis=-1)
                    W[g,:,:,s] /= np.sqrt(
                            np.sum(P1 * P2, axis=1)
                            )[:,None]

        demix(Y, X, W)

        if callback is not None:
            # select the first group
            if proj_back:
                z = projection_back(Y[0], X_g_o[0,:,:,0])
                callback(Y[0] * np.conj(z[None,:,:]))
            else:
                callback(Y[0])


        # Now run the super iteration
        W_comp = np.matmul(W, W_comp)
        W[:,:,:,:] = W0[:,:,:,:]

        # Shuffle the channels
        X[:,:,:,:] = Y[:,:,:,:]
        X[:,:,:,0] = Y[np.arange(-1,n_groups-1),:,:,0]

    # Final step: apply PCA to each group
    Y_final = np.zeros((n_frames, n_freq, n_src), dtype=X_original.dtype)
    for s in range(n_src):
        covmat = np.mean(Y[:,None,:,:,s] * np.conj(Y[None,:,:,:,s]), axis=2)
        covmat = np.moveaxis(covmat, -1, 0)  # new shape == (n_freq, n_groups, n_groups)
        w,v = np.linalg.eigh(covmat)
        pc = v[:,:,-1]
        for f in range(n_freq):
            Y_final[:,f,s] = np.dot(Y[:,:,f,s].T, pc[f])

    if callback is not None:
        # select the first group
        if proj_back:
            z = projection_back(Y_final, X_original[:,:,0])
            callback(Y_final * np.conj(z[None,:,:]))
        else:
            callback(Y_final)
    '''
    Y_final = Y[0]
    '''

    if proj_back:
        z = projection_back(Y_final, X_original[:,:,0])
        Y_final *= np.conj(z[None,:,:])

    if return_filters:
        return Y_final, W_comp[0]
    else:
        return Y_final
