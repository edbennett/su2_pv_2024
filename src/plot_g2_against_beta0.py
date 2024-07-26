#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt

from plaquette import read_plaquette_from_flows
from plots import PlotPropRegistry, errorbar_pyerrors, save_or_show
from read import read_all_fit_results
from stats import weighted_mean_by_uncertainty
from utils import group_params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_filenames", nargs="+", metavar="beta_fit_filename")
    parser.add_argument("--plot_filename", default=None)
    parser.add_argument("--plot_styles", default="styles/paperdraft.mplstyle")
    return parser.parse_args()


def plot(fit_results):
    fig, axes = plt.subplots(
        nrows=2, layout="constrained", figsize=(3.5, 5), sharex=True
    )
    grouped_results = group_params(fit_results, ["Npv", "mpv"])
    colours = PlotPropRegistry.colours()
    markers = PlotPropRegistry("o^s")
    axes[0].set_ylabel(r"$g_{\mathrm{GF}}^2$")
    axes[1].set_ylabel(r"$\langle P \rangle$")
    axes[1].set_xlabel(r"$\beta_0$")

    for (Npv, mpv), param_results in grouped_results.items():
        beta = sorted([datum["beta"] for datum in param_results])
        g_squared = [
            datum["gGF^2"][0]
            for datum in sorted(param_results, key=lambda datum: datum["beta"])
        ]
        plaquette = [
            weighted_mean_by_uncertainty(
                [
                    read_plaquette_from_flows(source["filename"])
                    for source in datum["data_sources"]
                ]
            )
            for datum in sorted(param_results, key=lambda datum: datum["beta"])
        ]
        errorbar_pyerrors(
            axes[0],
            beta,
            g_squared,
            color=colours[(Npv, mpv)],
            dashes=(1, 4),
            marker=markers[(Npv, mpv)],
            label=f"{Npv}PV, $m_{{\\mathrm{{PV}}}} = {mpv}$",
        )
        errorbar_pyerrors(
            axes[1],
            beta,
            plaquette,
            color=colours[(Npv, mpv)],
            dashes=(1, 4),
            marker=markers[(Npv, mpv)],
            label=f"{Npv}PV, $m_{{\\mathrm{{PV}}}} = {mpv}$",
        )

    axes[0].legend(loc="best")
    return fig


def main():
    args = get_args()
    plt.style.use(args.plot_styles)
    fit_results = read_all_fit_results(args.fit_filenames)
    save_or_show(plot(fit_results), args.plot_filename)


if __name__ == "__main__":
    main()
