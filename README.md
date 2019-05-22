Independent Vector Analysis with more Microphones than Sources
==============================================================

This repository provides implementations and code to reproduce the results
of the paper

> R. Scheibler and N. Ono, [*"Independent Vector Analysis with more Microphones than Sources,"*](https://arxiv.org/abs/1905.07880), 2019.

Abstract
--------

We extend frequency-domain blind source separation based on independent
vector analysis to the case where there are more microphones than sources.
The signal is modelled as non-Gaussian sources in a Gaussian background. The
proposed algorithm is based on a parametrization of the demixing matrix
decreasing the number of parameters to estimate. Furthermore, orthogonal
constraints between the signal and background subspaces are imposed to
regularize the separation. The problem can then be posed as a constrained
likelihood maximization. We propose efficient alternating updates guaranteed to
converge to a stationary point of the cost function. The performance of the
algorithm is assessed on simulated signals. We find that the
separation performance is on par with that of the conventional determined
algorithm at a fraction of the computational cost.

Authors
-------

[Robin Scheibler](http://robinscheibler.org) and [Nobutaka
Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html) are with the Faculty of
System Design at [Tokyo Metropolitan University](https://www.tmu.ac.jp/english/index.html).

### Contact

    Robin Scheibler (robin[at]tmu[dot]ac[dot]jp)
    6-6 Asahigaoka
    Hino, Tokyo
    191-0065 Japan

Preliminaries
-------------

The preferred way to run the code is using [anaconda](https://www.anaconda.com/distribution/).
An `environment.yml` file is provided to install the required dependencies.

    # create the minimal environment
    conda env create -f environment.yml

    # switch to new environment
    conda activate 2019_scheibler_overiva

Test OverIVA
------------

The algorithm can be tested and compared to others using the sample
script `overiva_oneshot.py`. It can be run as follows.

    $ python ./overiva_oneshot.py --help
    The samples directory {samples_dir} seems to exist already.
    Delete first for re-downloading.
    usage: overiva_oneshot.py [-h] [--no_cb] [-b BLOCK]
                              [-a {auxiva,auxiva_pca,overiva,ilrma,ogive}]
                              [-d {laplace,gauss}] [-i {eye,eig}] [-m MICS]
                              [-s SRCS] [-n N_ITER] [--gui] [--save]

    Demonstration of blind source separation using IVA.

    optional arguments:
      -h, --help            show this help message and exit
      --no_cb               Removes callback function
      -b BLOCK, --block BLOCK
                            STFT block size
      -a {auxiva,auxiva_pca,overiva,ilrma,ogive}, --algo {auxiva,auxiva_pca,overiva,ilrma,ogive}
                            Chooses BSS method to run
      -d {laplace,gauss}, --dist {laplace,gauss}
                            IVA model distribution
      -i {eye,eig}, --init {eye,eig}
                            Initialization, eye: identity, eig: principal
                            eigenvectors
      -m MICS, --mics MICS  Number of mics
      -s SRCS, --srcs SRCS  Number of sources
      -n N_ITER, --n_iter N_ITER
                            Number of iterations
      --gui                 Creates a small GUI for easy playback of the sound
                            samples
      --save                Saves the output of the separation to wav files

For example, we can run overiva with 4 microphones and 2 sources.

    python ./overiva_oneshot.py -a overiva -m 4 -s 2

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./overiva_sim.py ./overiva_sim_config.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./overiva_sim.py ./overiva_sim_config.json -t

        # stop the workers
        ipcluster stop

3. Run the whole simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./overiva_sim.py ./overiva_sim_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_overiva_sim_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 2. and 3. from the paper are produced then by running

    python ./overiva_sim_plot.py data/<data>-<time>_overiva_sim_<flag_or_hash> -s

Data
----

For the experiment, we concatenated utterances from the CMU ARCTIC speech corpus to
obtain samples of at least 15 seconds long. The dataset thus created was stored on zenodo
with DOI [10.5281/zenodo.3066488](https://zenodo.org/record/3066489). The data is automatically
retrieved upon running the scripts, but can also be manually downloaded with the `get_data.py` script.

    python ./get_data.py

It is stored in the `samples` directory.

Use OverIVA
-----------

Our implementation of the proposed OverIVA algorithm lives in the file `overiva.py`.
It can be used simply like this.

    from overiva import overiva

    # STFT tensor, a numpy.ndarray with shape (frames, frequencies, channels)
    X = ...

    # perform separation, output Y has the same shape as X
    Y = overiva(X, n_src=2)

The function comes with docstrings.

    overiva(X, n_src=None, n_iter=20, proj_back=True, W0=None, model="laplace",
            init_eig=False, return_filters=False, callback=None,)

    Implementation of overdetermined IVA algorithm for BSS as presented. See
    the following publication for a detailed description of the algorithm.

    R. Scheibler and N. Ono, Independent Vector Analysis with more Microphones than Sources, arXiv, 2019.
    https://arxiv.org/abs/1905.07880

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components. When
        ``n_src==nchannels``, the algorithms is identical to AuxIVA. When
        ``n_src==1``, then it is doing independent vector extraction.
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

Summary of the Files in this Repo
---------------------------------

    environment.yml  # anaconda environment file

    auxiva_pca.py  # implementation of AuxIVA with PCA dim reduction step
    ive.py  # implementation of orthogonally constrained independent vector extraction (OGIVE)
    overiva.py  # implementation of the proposed overdetermined IVA
    get_data.py  # script that gets the data necessary for the experiment
    routines.py  # contains a bunch of helper routines for the simulation

    overiva_oneshot.py  # test file for source separation, with audible output
    overiva_sim.py  # script to run exhaustive simulation, used for the paper
    overiva_sim_config.json  # simulation configuration file
    overiva_sim_plot.py  # plots the figures from the output of overiva_sim.py

    data  # directory containing simulation results
    rrtools  # tools for parallel simulation
