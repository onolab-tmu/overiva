import argparse, time
import numpy as np
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_sources

from lems_dataset import LEMSDataset
from overiva import overiva
from auxiva_pca import auxiva_pca
from mixiva import auxiva_cpp

dataset_loc = "../dataset"

algo_choices = ["auxiva", "auxiva_pca", "overiva", "ilrma", "overiva_cpp"]
model_choices = ["laplace", "gauss"]
init_choices = ["eye", "eig"]

if __name__ == "__main__":

    # Command Line Input Handling
    parser = argparse.ArgumentParser(description="Runs IVA on dataset of recordings")
    parser.add_argument("-b", "--block", type=int, default=2048, help="STFT block size")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default=algo_choices[0],
        choices=algo_choices,
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "-d",
        "--dist",
        type=str,
        default=model_choices[0],
        choices=model_choices,
        help="IVA model distribution",
    )
    parser.add_argument(
        "-i",
        "--init",
        type=str,
        default=init_choices[0],
        choices=init_choices,
        help="Initialization, eye: identity, eig: principal eigenvectors",
    )
    parser.add_argument("-m", "--mics", type=int, help="Number of mics")
    parser.add_argument(
        "-n", "--n_iter", type=int, default=51, help="Number of iterations"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Creates a small GUI for easy playback of the sound samples",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Saves the output of the separation to wav files",
    )
    args = parser.parse_args()

    # Open the dataset of 181 microphone from
    # https://web.archive.org/web/20100803034556/http://www.lems.brown.edu/array/data.html
    dataset = LEMSDataset(dataset_number=2)
    n_src = dataset.ntalkers()
    if args.mics is None:
        n_mics = dataset.nmics()
    else:
        n_mics = args.mics
    mic_signals = dataset.mic_signals()[:, :n_mics]

    assert n_mics >= n_src

    # STFT parameters
    framesize = 4096
    hop = framesize // 2
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_mics = pra.transform.analysis(
            mic_signals, framesize, hop, win=win_a
    ).astype(np.complex128)

    tic = time.perf_counter()

    convergence_callback = None

    # Run BSS
    if args.algo == "auxiva":
        # Run AuxIVA
        Y = overiva(
            X_mics,
            n_iter=args.n_iter,
            proj_back=True,
            model=args.dist,
            callback=convergence_callback,
        )
    elif args.algo == "auxiva_pca":
        # Run AuxIVA
        Y = auxiva_pca(
            X_mics,
            n_src=n_src,
            n_iter=args.n_iter,
            proj_back=True,
            model=args.dist,
            callback=convergence_callback,
        )
    elif args.algo == "overiva":
        # Run AuxIVA
        Y = overiva(
            X_mics,
            n_src=n_src,
            n_iter=args.n_iter,
            proj_back=True,
            model=args.dist,
            init_eig=(args.init == init_choices[1]),
            callback=convergence_callback,
        )
    elif args.algo == "overiva_cpp":
        # Run AuxIVA
        Y = auxiva_cpp(
            X_mics,
            n_src=n_src,
            n_iter=args.n_iter,
            proj_back=True,
            model=args.dist,
        )
    elif args.algo == "ilrma":
        # Run AuxIVA
        Y = pra.bss.ilrma(
            X_mics,
            n_iter=args.n_iter,
            n_components=2,
            proj_back=True,
            callback=convergence_callback,
        )
    else:
        raise ValueError("No such algorithm {}".format(args.algo))

    toc = time.perf_counter()

    print("Processing time: {} s".format(toc - tic))

    # Run iSTFT
    y = pra.transform.synthesis(Y, framesize, hop, win=win_s).astype(np.float64)
    y = y[framesize - hop :, :]  # cut-off front padding

    # If some of the output are uniformly zero, just add a bit of noise to compare
    for k in range(y.shape[1]):
        if np.sum(np.abs(y[:, k])) < 1e-10:
            y[:, k] = np.random.randn(y.shape[0]) * 1e-10

    # Order the output signals by average power (descending)
    new_ord = np.argsort(np.std(y, axis=0))[::-1]
    y = y[:, new_ord]

    # Compare SIR
    #############
    ref = dataset.close_talking_signals()
    m = np.minimum(y.shape[0], ref.shape[1])
    sdr, sir, sar, perm = bss_eval_sources(ref[:m, :], y[:m, :].T)

    # reorder the vector of reconstructed signals
    y_hat = y[:, perm]

    print("SDR:", sdr)
    print("SIR:", sir)

    """
    import matplotlib.pyplot as plt

    plt.figure()

    for i in range(n_src):
        plt.subplot(2, n_src, i + 1)
        plt.specgram(ref[i, :, 0] + 1e-1, NFFT=1024, Fs=room.fs)
        plt.title("Source {} (clean)".format(i))

        plt.subplot(2, n_src, i + n_src + 1)
        plt.specgram(y_hat[:, i], NFFT=1024, Fs=room.fs)
        plt.title("Source {} (separated)".format(i))

    plt.tight_layout(pad=0.5)

    plt.figure()
    a = np.array(SDR)
    b = np.array(SIR)
    for i, (sdr, sir) in enumerate(zip(a.T, b.T)):
        plt.plot(
            np.arange(a.shape[0]) * 10, sdr, label="SDR Source " + str(i), marker="*"
        )
        plt.plot(
            np.arange(a.shape[0]) * 10, sir, label="SIR Source " + str(i), marker="o"
        )
    plt.legend()
    plt.tight_layout(pad=0.5)

    if not args.gui:
        plt.show()
    else:
        plt.show(block=False)

    if args.save:
        from scipy.io import wavfile

        wavfile.write(
            "bss_iva_mix.wav",
            room.fs,
            pra.normalize(mics_signals[0, :], bits=16).astype(np.int16),
        )
        for i, sig in enumerate(y_hat):
            wavfile.write(
                "bss_iva_source{}.wav".format(i + 1),
                room.fs,
                pra.normalize(sig, bits=16).astype(np.int16),
            )
    """

    if args.gui:

        from tkinter import Tk

        # Make a simple GUI to listen to the separated samples
        root = Tk()
        my_gui = PlaySoundGUI(
            root, room.fs, mics_signals[0, :], y_hat.T, references=ref[:, :, 0]
        )
        root.mainloop()
