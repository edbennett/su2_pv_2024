#!/usr/bin/env python3

import argparse
import functools

import matplotlib.pyplot as plt
import numpy as np
import pyerrors as pe

from fit_beta_against_g2 import interpolating_form
from names import operator_names
from plots import PlotPropRegistry, errorbar_pyerrors, save_or_show
from perturbation_theory import add_perturbative_lines
from read import read_all_fit_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_filenames", nargs="+", metavar="beta_fit_filename")
    parser.add_argument("--plot_filename", default=None)
    parser.add_argument("--plot_styles", default="styles/paperdraft.mplstyle")
    return parser.parse_args()


def split_errors(series):
    return [datum.value for datum in series], [datum.dvalue for datum in series]


def output_fits(fit_results, filename):
    pe.input.json.dump_dict_to_json(fit_results, filename)


def plot_fit(x_values, fit_result, ax, order=4, colour=None):
    scan_x = np.linspace(min(x_values), max(x_values), 1000)
    scan_y = interpolating_form(np.asarray(fit_result, dtype=float), scan_x, n=order)
    scan_errors = pe.fits.error_band(
        scan_x, functools.partial(interpolating_form, n=order), fit_result
    )
    ax.fill_between(
        scan_x, scan_y + scan_errors, scan_y - scan_errors, color=colour, alpha=0.2
    )


def plot(fit_results):
    operators = sorted(set([datum["operator"] for datum in fit_results]))
    Npvs = sorted(set([datum["Npv"] for datum in fit_results]))

    num_columns = len(Npvs)
    num_rows = len(operators)

    fig, axes = plt.subplots(
        layout="constrained",
        figsize=(1.5 + 2 * num_columns, 1 + 1.5 * num_rows),
        ncols=num_columns,
        nrows=num_rows,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colours = PlotPropRegistry.colours()

    for ax in axes[-1]:
        ax.set_xlabel(r"$g_{\mathrm{GF}}^2(t; g_0^2)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta_{\mathrm{GF}}(t; g_0^2)$")

    for result in fit_results:
        data = read_all_fit_results(
            [source["filename"] for source in result["data_sources"]]
        )
        time = result["time"]
        gGF2 = [datum["gGF^2"][0] for datum in data]
        betaGF = [datum["betaGF"][0] for datum in data]
        errorbar_pyerrors(
            axes[operators.index(result["operator"])][Npvs.index(result["Npv"])],
            gGF2,
            betaGF,
            label=f"$t / a^2 = {time}$",
            color=colours[result["time"]],
        )

        plot_fit(
            [value.value for value in gGF2],
            result["beta_interpolation"],
            axes[operators.index(result["operator"])][Npvs.index(result["Npv"])],
            order=result["order"],
            colour=colours[result["time"]],
        )

    _, xmax = axes[0][0].get_xlim()
    for ax in axes.ravel():
        add_perturbative_lines(ax, 0, xmax, rep="adj", Nf=2, Nc=2)
        ax.axhline(0, color="black")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-5, 0.6)

    if len(operators) > 1 or len(Npvs) > 1:
        for op_idx, operator in enumerate(operators):
            for Npv_idx, Npv in enumerate(Npvs):
                ax = axes[op_idx][Npv_idx]
                ax.text(
                    0.95,
                    0.05,
                    f"{Npv}PV, {operator_names[operator]}",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                )

    axes[-1][0].legend(loc="best")

    return fig


def main():
    args = get_args()
    plt.style.use(args.plot_styles)
    fit_results = read_all_fit_results(args.fit_filenames)
    save_or_show(plot(fit_results), args.plot_filename)


if __name__ == "__main__":
    main()
