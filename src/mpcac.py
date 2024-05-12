#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import gridspec

from meson_analysis.readers import read_correlators_hirep
from meson_analysis.fits import fit_pcac, pcac_eff_mass

import numpy as np


def pcac_aic(correlator, range):
    result = fit_pcac(correlator, range, full=True)
    k = len(result.fit_parameters)
    Ncut = correlator.NT - (range[0] - range[1])
    chisquare_aug = result.chisquare_by_dof

    return result, chisquare_aug + 2 * k + 2 * Ncut


def weighted_mean(results):
    values = [result.fit_parameters[0] for result, aic in results]
    weights = [np.exp(-aic) for result, aic in results]
    result = sum([value * weight for value, weight in zip(values, weights)]) / sum(weights)
    result.gamma_method()
    return result


def add_band(ax, result):
    ax.axhline(result.value)
    ax.axhline(result.value + result.dvalue, dashes=(2, 2))
    ax.axhline(result.value + result.dvalue, dashes=(2, 2))


def plot_results(correlator, results, output_filename=None):
    plt.style.use("styles/paperdraft.mplstyle")

    meff = pcac_eff_mass(correlator)
    t, meff_value, meff_err = meff.plottable()

    result = weighted_mean(results)

    fig = plt.Figure(layout="constrained")

    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    ax0.errorbar(t, meff_value, yerr=meff_err, ls="none", capsize=1)
    add_band(ax0, result)
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$m_{\mathrm{eff}}$")

    ax1.set_ylabel(r"$m_{\mathrm{eff}}$")
    ax1.errorbar(
        list(range(len(results))),
        [result.fit_parameters[0].value for result, aic in results],
        yerr=[result.fit_parameters[0].dvalue for result, aic in results],
        ls="none",
        capsize=1,
    )
    add_band(ax1, result)
    ax1.tick_params("x", labelbottom=False)

    ax2.set_xlabel("Index")
    ax2.set_ylabel(r"$\log(p(M|D))$")
    ax2.scatter(
        list(range(len(results))),
        [-aic for result, aic in results],
    )

    fig.suptitle(f"$m_{{\\mathrm{{PCAC}}}} = {result}$")

    if output_filename:
        fig.savefig(output_filename)
        plt.close(fig)
    else:
        plt.show()


def get_pcacs_aic(correlator):
    results = []

    for tmin in range(4, correlator.NT // 2 - 1):
        for tmax in range(tmin + 1, correlator.NT // 2):
            results.append(pcac_aic(correlator, [tmin, tmax]))

    return results


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("correlator_filename")
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--plot_filename", default=None)
    parser.add_argument("--Npv", type=int, default=None)
    parser.add_argument("--mpv", type=float, default=None)
    return parser.parse_args()


def get_description(correlator):
    return {
        "description": "PCAC mass for ensemble as detailed below.",
        "input_filename": correlator.filename,
        **correlator.metadata,
    }


def main():
    args = get_args()

    correlator = read_correlators_hirep(args.correlator_filename)
    if (num_masses := len(correlator.metadata["valence_masses"])) != 1:
        message = f"This code expects 1 valence mass; {num_masses} found"
        raise ValueError(message)
    correlator.metadata["Npv"] = args.Npv
    correlator.metadata["mpv"] = args.mpv

    results = get_pcacs_aic(correlator)
    mpcac_result = weighted_mean(results)

    if args.output_filename:
        mpcac_result.dump(
            args.output_filename,
            description=get_description(correlator)
        )
    else:
        print("mPCAC =", mpcac_result)

    plot_results(correlator, results, args.plot_filename)


if __name__ == "__main__":
    main()
