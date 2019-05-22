Independent Vector Analysis with more Microphones than Sources
==============================================================

This repository provides implementations and code to reproduce the results
of the paper

> R. Scheibler and N. Ono, [*"Independent Vector Analysis with more Microphones than Sources,"*](https://arxiv.org/abs/1905.07880) Submitted to WASPAA 2019, 2019.

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

Reproduce the Results
---------------------

The preferred way is to use [anaconda](https://www.anaconda.com/distribution/).

    conda env create -f environment.yml

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
retrieved upon running of the scripts, but can be also manually downloaded with the `get_data.py` script.

    python ./get_data.py

It is stored in the `samples` directory.
