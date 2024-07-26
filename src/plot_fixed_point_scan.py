#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt

from names import operator_names
from plots import PlotPropRegistry, legend, save_or_show
from read import read_all_fit_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_filenames", nargs="+", metavar="beta_continuum_filename")
    parser.add_argument("--plot_filename", default=None)
    parser.add_argument("--plot_styles", default="styles/paperdraft.mplstyle")
    return parser.parse_args()


def plot(data):
    operators = set(datum["operator"] for datum in data)
    colours = PlotPropRegistry.colours()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(operators),
        figsize=(3.5 * len(operators), 5),
        layout="constrained",
        sharex=True,
        sharey="row",
    )

    axes[0][0].set_ylabel(r"$g^2_{\mathrm{GF}\star}$")
    axes[1][0].set_ylabel(r"$\gamma^*_g$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$t_{\mathrm{min}} / a^2$")

    operator_axes = PlotPropRegistry(axes.swapaxes(0, 1))

    for datum in data:
        for observable, ax in zip(
            ["g_star_squared", "gamma_star"], operator_axes[datum["operator"]]
        ):
            value = datum[f"value_{observable}"]
            uncertainty = datum[f"uncertainty_{observable}"]
            ax.errorbar(
                [datum["min_time"]],
                [value],
                yerr=[uncertainty],
                marker="s",
                color=colours[datum["max_time"]],
            )

    for operator, ax_col in operator_axes.items():
        for ax in ax_col:
            ax.text(
                0.03,
                0.03,
                operator_names[operator],
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

    legend(
        axes[0][-1],
        colours,
        "colour",
        lambda t: f"$t_{{\\mathrm{{max}}}}/a^2={t:.1f}$",
        4,
        "outside upper center",
        fig=fig,
    )

    return fig


def main():
    args = get_args()
    plt.style.use(args.plot_styles)
    fit_results = read_all_fit_results(args.fit_filenames, pyerrors=False)
    save_or_show(plot(fit_results), args.plot_filename)


if __name__ == "__main__":
    main()
