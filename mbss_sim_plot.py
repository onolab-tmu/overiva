import sys, argparse, os, json
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyroomacoustics as pra
from routines import grid_layout, semi_circle_layout, random_layout, gm_layout


def plot_room_setup(filename, n_mics, n_targets, parameters):
    """
    Plot the room scenario in 2D
    """

    n_interferers = parameters["n_interferers"]
    n_blinkies = parameters["n_blinkies"]
    ref_mic = parameters["ref_mic"]
    room_dim = np.array(parameters["room_dim"])

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

    if parameters["blinky_geometry"] == "gm":
        """ Normally distributed in the vicinity of each source """
        blinky_locs = gm_layout(
            n_blinkies,
            target_locs - np.c_[[0.0, 0.0, 0.5]],
            std=[0.4, 0.4, 0.05],
            seed=987,
        )

    elif parameters["blinky_geometry"] == "grid":
        """ Placed on a regular grid, with a little bit of noise added """
        blinky_locs = grid_layout(
            [3.0, 5.5], n_blinkies, offset=[1.0, 1.0, 0.7], seed=987
        )

    else:
        """ default is semi-circular """
        blinky_locs = semi_circle_layout(
            [4.1, 3.755, 1.1],
            np.pi,
            3.5,
            n_blinkies,
            rot=0.743 * np.pi - np.pi / 4,
            seed=987,
        )

    mic_locs = np.vstack(
        (
            pra.circular_2D_array([4.1, 3.76], n_mics, np.pi / 2, 0.02),
            1.2 * np.ones((1, n_mics)),
        )
    )
    all_locs = np.concatenate((mic_locs, blinky_locs), axis=1)

    # Create the room itself
    room = pra.ShoeBox(room_dim[:2])

    for loc in source_locs.T:
        room.add_source(loc[:2])

    # Place the microphone array
    room.add_microphone_array(pra.MicrophoneArray(all_locs[:2, :], fs=room.fs))

    room.plot(img_order=0)
    plt.xlim([-0.1, room_dim[0] + 0.1])
    plt.ylim([-0.1, room_dim[1] + 0.1])

    plt.savefig(filename)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Plot the data simulated by separake_near_wall"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display the plots at the end of data analysis",
    )
    parser.add_argument(
        "dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    parameters = dict()
    algorithms = dict()
    args = []
    df = None

    data_files = []

    for i, data_dir in enumerate(cli_args.dirs):

        print("Reading in", data_dir)

        # add the data file from this directory
        data_file = os.path.join(data_dir, "data.json")
        if os.path.exists(data_file):
            data_files.append(data_file)
        else:
            raise ValueError("File {} doesn" "t exist".format(data_file))

        # get the simulation config
        with open(os.path.join(data_dir, "parameters.json"), "r") as f:
            parameters = json.load(f)

    # algorithms to take in the plot
    algos = algorithms.keys()

    # check if a pickle file exists for these files
    pickle_file = ".mbss.pickle"

    if os.path.isfile(pickle_file) and pickle_flag:
        print("Reading existing pickle file...")
        # read the pickle file
        df = pd.read_pickle(pickle_file)

    else:

        # reading all data files in the directory
        records = []
        for file in data_files:
            with open(file, "r") as f:
                content = json.load(f)
                for seg in content:
                    records += seg

        # build the data table line by line
        print("Building table")
        columns = [
            "Algorithm",
            "Sources",
            "Mics",
            "RT60",
            "SINR",
            "seed",
            "Strength",
            "SDR",
            "SIR",
        ]
        table = []
        num_sources = set()

        copy_fields = ["algorithm", "n_targets", "n_mics", "rt60", "sinr", "seed"]

        for record in records:

            entry = [record[field] for field in copy_fields]

            try:

                table.append(
                    entry + ["Weak source", record["sdr"][-1][0], record["sir"][-1][0]]
                )
                table.append(
                    entry
                    + [
                        "Strong sources (avg.)",
                        np.mean(record["sdr"][-1][1:]),
                        np.mean(record["sir"][-1][1:]),
                    ]
                )
                table.append(
                    entry
                    + [
                        "Average",
                        np.mean(record["sdr"][-1]),
                        np.mean(record["sir"][-1]),
                    ]
                )
            except:
                continue

        # create a pandas frame
        print("Making PANDAS frame...")
        df = pd.DataFrame(table, columns=columns)

        df.to_pickle(pickle_file)

    # Draw the figure
    print("Plotting...")

    # sns.set(style='whitegrid')
    # sns.plotting_context(context='poster', font_scale=2.)
    # pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    df = df.replace(
        {
            "Algorithm": {
                "auxiva": "AuxIVA (Laplace)",
                "auxiva_gauss": "AuxIVA (Gauss)",
                "auxiva_pca": "AuxIVA PCA (Laplace)",
                "ilrma": "ILRMA",
                "oilrma": "od-ILRMA",
                "oiva_laplace": "od-IVA (Laplace)",
                "oiva_laplace_eig": "od-IVA (Laplace, eig. init.)",
                "oiva_gauss": "od-IVA (Gauss)",
                "oiva_gauss_eig": "od-IVA (Gauss, eig. init.)",
            }
        }
    )

    all_algos = [
        "AuxIVA (Laplace)",
        "od-IVA (Laplace)",
        "od-IVA (Laplace, eig. init.)",
        "AuxIVA PCA (Laplace)",
        "AuxIVA (Gauss)",
        "od-IVA (Gauss)",
        "od-IVA (Gauss, eig. init.)",
        "ILRMA",
        "od-ILRMA",
    ]

    sns.set(
        style="whitegrid",
        context="paper",
        font_scale=0.6,
        rc={
            #'figure.figsize': (3.39, 3.15),
            #'lines.linewidth': 1.,
            #'font.family': 'sans-serif',
            #'font.sans-serif': [u'Helvetica'],
            #'text.usetex': False,
        },
    )
    pal = sns.cubehelix_palette(
        4, start=0.5, rot=-0.5, dark=0.3, light=0.75, reverse=True, hue=1.0
    )
    sns.set_palette(pal)

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    fn_tmp = os.path.join(fig_dir, "RT60_{rt60}_SINR_{sinr}_{metric}.pdf")

    plt_kwargs = {
        "SDR": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        "SIR": {"ylim": [-0.5, 40.5], "yticks": [0, 10, 20, 30]},
    }
    fig_cols = ["Average"]
    full_width = 3.2  # inches
    aspect = 1.5  # width / height
    height = full_width / len(fig_cols) / aspect

    medians = {}

    for rt60 in parameters["rt60_list"]:
        medians[rt60] = {}
        for sinr in parameters["sinr_list"]:
            medians[rt60][sinr] = {}

            select = np.logical_and(df["RT60"] == rt60, df["SINR"] == sinr)

            for metric in ["SDR", "SIR"]:

                g = sns.catplot(
                    data=df[select],
                    x="Mics",
                    y=metric,
                    hue="Algorithm",
                    col="Strength",
                    row="Sources",
                    col_order=fig_cols,
                    hue_order=all_algos,
                    kind="box",
                    legend=False,
                    aspect=aspect,
                    height=height,
                    linewidth=0.5,
                    fliersize=0.5,
                    # size=3, aspect=0.65,
                )

                g.set(**plt_kwargs[metric])
                g.set_titles("{row_name} sources | {col_name}")

                all_artists = []

                #left_ax = g.facet_axis(2, 0)
                left_ax = g.facet_axis(0, 0)
                leg = left_ax.legend(
                    title="Algorithms",
                    frameon=True,
                    framealpha=0.85,
                    # fontsize='small',
                    loc="upper left",
                    bbox_to_anchor=[-0.05, 1.05],
                )
                leg.get_frame().set_linewidth(0.0)
                all_artists.append(leg)

                sns.despine(offset=10, trim=False, left=True, bottom=True)

                plt.tight_layout(pad=0.01)

                """
                plt.subplots_adjust(top=0.9)
                tit = g.fig.suptitle('# blinkies={}, RT60={}, SINR={}'.format(
                    parameters['n_blinkies'], rt60, sinr
                    ))
                all_artists.append(tit)
                """

                rt60_name = str(int(float(rt60) * 1000)) + "ms"
                fig_fn = fn_tmp.format(rt60=rt60_name, sinr=sinr, metric=metric)
                plt.savefig(fig_fn, bbox_extra_artists=all_artists, bbox_inches="tight")

                # also get only the median information out
                medians[rt60][sinr][metric] = []
                for sub_df in g.facet_data():
                    medians[rt60][sinr][metric].append(
                        sub_df[1].pivot_table(
                            values=metric,
                            columns="Mics",
                            index=["Algorithm", "Sources", "RT60", "SINR", "Strength"],
                            aggfunc="median",
                        )
                    )

    # fn_room_setup = os.path.join(fig_dir, 'room_setup.pdf')
    # plot_room_setup(fn_room_setup, 4, 4, parameters)

    if plot_flag:
        plt.show()
