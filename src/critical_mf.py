#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pyerrors as pe


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("pcac_mass_filenames", metavar="PCAC_MASS_FILENAME", nargs="+")
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--plot_filename", default=None)
    return parser.parse_args()


def get_consistent_metadata(data):
    original_metadata = [datum["description"] for datum in data]
    metadata = {}
    for key in (
        "group_family",
        "Nc",
        "beta",
        "valence_representation",
        "dynamical_representation",
        "Npv",
        "mpv",
    ):
        values = [metadatum[key] for metadatum in original_metadata]
        if (num_distinct := len(set(values))) != 1:
            message = f"Multiple ({num_distinct}) values found for {key}"
            raise ValueError(message)
        metadata[key] = values[0]
    return metadata


def fit_form(params, m):
    m0 = params[0]
    B = params[1]
    C = params[2]
    return B * (m - m0) ** C


def inverse_fit_form(params, mPCAC):
    m0, B, C = params
    return m0 + (mPCAC / B) ** (1 / C)


def get_smallest(target_data, key_data, skip):
    if not len(target_data) == len(key_data):
        raise ValueError("Target and key data are not the same length.")

    indices = sorted(range(len(key_data)), key=lambda k: key_data[k])
    return [target_data[index] for index in indices[skip:]]


def fit(data, skip=0):
    full_x_data = [datum["description"]["valence_masses"][0] for datum in data]
    x_data = get_smallest(full_x_data, full_x_data, skip)
    y_data = get_smallest([datum["obsdata"][0] for datum in data], full_x_data, skip)
    for datum in y_data:
        datum.gamma_method()
    return pe.fits.least_squares(
        x_data, y_data, fit_form, initial_guess=[-2.0, 1.0, 1.0], silent=True
    )


def get_description(result, args, metadata):
    return {
        "description": "Critical bare fermion mass for set of ensembles described below.",
        "input_filenames": args.pcac_mass_filenames,
        "chisquare": result.chisquare,
        "dof": result.dof,
        "method": result.method,
        **metadata,
    }


def write_result(fit_result, args, metadata):
    if not args.output_filename:
        print(fit_result.fit_parameters[0])
        return

    fit_result.fit_parameters[0].dump(
        args.output_filename, description=get_description(fit_result, args, metadata)
    )


def plot_result(data, fit_results, output_filename):
    plt.style.use("styles/paperdraft.mplstyle")
    x_data = [datum["description"]["valence_masses"][0] for datum in data]
    y_data = [datum["obsdata"][0] for datum in data]
    main_fit_result = fit_results[0]
    pe.fits.residual_plot(x_data, y_data, fit_form, main_fit_result)

    fig = plt.gcf()
    ax1, ax2 = fig.axes

    _, xmax = ax1.get_xlim()
    _, ymax = ax1.get_ylim()

    xmin = main_fit_result.fit_parameters[0].value - main_fit_result.fit_parameters[0].dvalue

    for ax in ax1, ax2:
        ax.set_xlim(xmin, xmax)
    ax1.set_ylim(0, ymax)

    x = np.linspace(xmin, xmax, 1000)
    for skip, fit_result in fit_results.items():
        label = f"Fit (omit {skip} lightest)" if skip > 0 else None
        colour = {0: "darkorange", 1: "darkgreen"}[skip]
        ax1.plot(
            x,
            fit_form([param.value for param in fit_result.fit_parameters], x),
            color=colour,
            label=label,
        )

    ax1.legend()
    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()


def main():
    args = get_args()
    data = [
        pe.input.json.load_json(filename, full_output=True, verbose=False)
        for filename in args.pcac_mass_filenames
    ]
    metadata = get_consistent_metadata(data)
    fit_result = fit(data)
    fit_result.fit_parameters[0].gamma_method()
    write_result(fit_result, args, metadata)

    fit_result_skipsmallest = fit(data, skip=1)
    plot_result(
        data,
        {0: fit_result, 1: fit_result_skipsmallest},
        args.plot_filename,
    )


if __name__ == "__main__":
    main()
