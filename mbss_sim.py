"""
This file contains the code to run the more systematic simulation.
"""
import argparse, json, os
import numpy as np
import pyroomacoustics as pra
import rrtools

from routines import (
    PlaySoundGUI,
    grid_layout,
    semi_circle_layout,
    random_layout,
    gm_layout,
)
from generate_samples import sampling, wav_read_center

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])


def init(parameters):
    parameters["base_dir"] = base_dir


def one_loop(args):
    global parameters

    import numpy

    np = numpy

    import pyroomacoustics

    pra = pyroomacoustics

    import sys

    sys.path.append(parameters["base_dir"])

    from routines import semi_circle_layout, random_layout, gm_layout, grid_layout
    from oiva import oiva
    from oilrma import oilrma
    from auxiva_gauss import auxiva_gauss
    from auxiva_pca import auxiva_pca
    from generate_samples import wav_read_center

    n_targets, n_mics, rt60, sinr, wav_files, seed = args

    # this is the underdetermined case. We don't do that.
    if n_mics < n_targets:
        return []

    # set MKL to only use one thread if present
    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        pass

    # set the RNG seed
    rng_state = np.random.get_state()
    np.random.seed(seed)

    # STFT parameters
    framesize = parameters["stft_params"]["framesize"]
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, framesize // 2)

    # Generate the audio signals

    # get the simulation parameters from the json file
    # Simulation parameters
    n_repeat = parameters["n_repeat"]
    fs = parameters["fs"]
    snr = parameters["snr"]

    n_interferers = parameters["n_interferers"]
    ref_mic = parameters["ref_mic"]
    room_dim = np.array(parameters["room_dim"])

    sources_var = np.ones(n_targets)
    sources_var[0] = parameters["weak_source_var"]

    # total number of sources
    n_sources = n_interferers + n_targets

    # Geometry of the room and location of sources and microphones
    interferer_locs = random_layout(
        [3.0, 5.5, 1.5], n_interferers, offset=[6.5, 1.0, 0.5], seed=1
    )

    target_locs = semi_circle_layout(
        [4.1, 3.755, 1.2],
        np.pi / 1.5,
        2.0,  # 120 degrees arc, 2 meters away
        n_targets,
        rot=0.743 * np.pi,
    )

    source_locs = np.concatenate((target_locs, interferer_locs), axis=1)

    mic_locs = np.vstack(
        (
            pra.circular_2D_array([4.1, 3.76], n_mics, np.pi / 2, 0.02),
            1.2 * np.ones((1, n_mics)),
        )
    )

    signals = wav_read_center(wav_files, seed=123)

    # Create the room itself
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        absorption=parameters["rt60_list"][rt60]["absorption"],
        max_order=parameters["rt60_list"][rt60]["max_order"],
    )

    # Place all the sound sources
    for sig, loc in zip(signals[-n_sources:, :], source_locs.T):
        room.add_source(loc, signal=sig)

    assert len(room.sources) == n_sources, (
        "Number of signals ({}) doesn"
        "t match number of sources ({})".format(signals.shape[0], n_sources)
    )

    # Place the microphone array
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=room.fs))

    # compute RIRs
    room.compute_rir()

    # Run the simulation
    premix = room.simulate(return_premix=True)

    # Normalize the signals so that they all have unit
    # variance at the reference microphone
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]

    # scale to pre-defined variance
    premix[:n_targets, :, :] *= np.sqrt(sources_var[:, None, None])

    # compute noise variance
    sigma_n = np.sqrt(10 ** (-snr / 10) * np.sum(sources_var))

    # now compute the power of interference signal needed to achieve desired SINR
    sigma_i = np.sqrt(
        np.maximum(0, 10 ** (-sinr / 10) * np.sum(sources_var) - sigma_n ** 2)
        / n_interferers
    )
    premix[n_targets:, :, :] *= sigma_i

    # Mix down the recorded signals
    mix = np.sum(premix, axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    ref = np.moveaxis(premix, 1, 2)

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(mix.T, framesize, framesize // 2, win=win_a)
    X_mics = X_all[:, :, :n_mics]

    # convergence monitoring callback
    def convergence_callback(Y, n_targets, SDR, SIR, ref, framesize, win_s, algo_name):
        from mir_eval.separation import bss_eval_sources

        y = pra.transform.synthesis(Y, framesize, framesize // 2, win=win_s)

        if algo_name not in ["oiva", "oilrma", "auxiva_pca"]:
            new_ord = np.argsort(np.std(y, axis=0))[::-1]
            y = y[:, new_ord]

        m = np.minimum(y.shape[0] - framesize // 2, ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(
            ref[:n_targets, :m, 0], y[framesize // 2 : m + framesize // 2, :n_targets].T
        )
        SDR.append(sdr.tolist())
        SIR.append(sir.tolist())

    # store results in a list, one entry per algorithm
    results = []

    # compute the initial values of SDR/SIR
    init_sdr = []
    init_sir = []
    if not parameters["monitor_convergence"]:
        convergence_callback(
            X_mics,
            n_targets,
            init_sdr,
            init_sir,
            ref,
            framesize,
            win_s,
            "init",
        )

    for name, kwargs in parameters["algorithm_kwargs"].items():

        results.append(
            {
                "algorithm": name,
                "n_targets": n_targets,
                "n_mics": n_mics,
                "rt60": rt60,
                "sinr": sinr,
                "seed": seed,
                "sdr": [],
                "sir": [],  # to store the result
            }
        )

        if parameters["monitor_convergence"]:
            def cb(Y):
                convergence_callback(
                    Y,
                    n_targets,
                    results[-1]["sdr"],
                    results[-1]["sir"],
                    ref,
                    framesize, win_s,
                    name,
                )
        else:
            cb = None
            # avoid one computation by using the initial values of sdr/sir
            results[-1]["sdr"].append(init_sdr[0])
            results[-1]["sir"].append(init_sir[0])

        if name == "auxiva":
            # Run AuxIVA
            Y = pra.bss.auxiva(X_mics, callback=cb, **kwargs)

        elif name == "ilrma":
            # Run AuxIVA
            Y = pra.bss.ilrma(X_mics, callback=cb, **kwargs)

        elif name == "auxiva_gauss":
            # Run AuxIVA
            Y = auxiva_gauss(X_mics, callback=cb, **kwargs)

        elif name == "auxiva_pca":
            # Run AuxIVA
            Y = auxiva_pca(X_mics, n_src=n_targets, callback=cb, **kwargs)

        elif name == "oiva":
            # Run BlinkIVA
            Y = oiva(X_mics, n_src=n_targets, callback=cb, **kwargs)

        elif name == "oilrma":
            # Run BlinkIVA
            Y = oilrma(X_mics, n_src=n_targets, callback=cb, **kwargs)

        else:
            continue

        # The last evaluation
        convergence_callback(
            Y,
            n_targets,
            results[-1]["sdr"],
            results[-1]["sir"],
            ref,
            framesize,
            win_s,
            name,
        )

    # restore RNG former state
    np.random.set_state(rng_state)

    return results


def generate_arguments(parameters):
    """ This will generate the list of arguments to run simulation for """

    rng_state = np.random.get_state()
    np.random.seed(parameters["seed"])

    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
        parameters["n_repeat"],
        parameters["n_interferers"] + np.max(parameters["n_targets_list"]),
        parameters["samples_list"],
        gender_balanced=True,
        seed=gen_files_seed,
    )

    args = []

    for n_targets in parameters["n_targets_list"]:
        for n_mics in parameters["n_mics_list"]:

            # we don't do underdetermined
            if n_targets > n_mics:
                continue

            for rt60 in parameters["rt60_list"].keys():
                for sinr in parameters["sinr_list"]:
                    for wav_files in all_wav_files:

                        # generate the seed for this simulation
                        seed = int(np.random.randint(2 ** 32, dtype=np.uint32))

                        # add the new combination to the list
                        args.append([n_targets, n_mics, rt60, sinr, wav_files, seed])

    np.random.set_state(rng_state)

    return args


if __name__ == "__main__":

    rrtools.run(
        one_loop,
        generate_arguments,
        func_init=init,
        base_dir=base_dir,
        results_dir="data/",
        description="Simulation for Multi-modal BSS with blinkies (ICASSP 2019)",
    )
