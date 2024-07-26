#!/usr/bin/env python3

import numpy as np


# from https://arxiv.org/pdf/hep-ph/9701390
quadratic_casimirs = {
    "adj": lambda Nc: Nc,
    "fun": lambda Nc: (Nc ** 2 - 1) / (2 * Nc),
}

# From https://arxiv.org/pdf/1912.13302
trace_normalisations = {
    #"adj": lambda Nc: (Nc ** 2 - 4) / Nc,
    "adj": lambda Nc: Nc,
    "fun": lambda Nc: 1 / 2,
}

def perturbative_beta(x, n, rep="adj", Nf=12, Nc=3):
    # Eq. (8) of https://arxiv.org/pdf/hep-ph/9701390
    # T_F replaced with T_R for arbitrary representation
    CA = quadratic_casimirs["adj"](Nc)
    CR = quadratic_casimirs[rep](Nc)
    TR = trace_normalisations[rep](Nc)

    beta = np.asarray(
        [
            11 / 3 * CA - 4 / 3 * TR * Nf,
            34 / 3 * CA ** 2 - 4 * CR * TR * Nf - 20 / 3 * CA * TR * Nf,
            2857 / 54 * CA ** 3 + 2 * CR ** 2 * TR * Nf - 205 / 9 * CR * CA * TR * Nf - 1415 / 27 * CA ** 2 * TR * Nf + 44 / 9 * CR * TR ** 2 * Nf ** 2 + 158 / 27 * CA * TR ** 2 * Nf ** 2,
        ]
    )
    return -((4 * np.pi) ** 2) * np.sum(
        (x / (4 * np.pi) ** 2) ** (np.arange(n)[:, np.newaxis] + 2)
        * beta[:n, np.newaxis],
        axis=0,
    )


def add_perturbative_lines(ax, xmin, xmax, rep, Nf, Nc):
    analytic_range = np.linspace(xmin, xmax, 1000)
    for n_loops, dashes in [(1, (3, 1)), (2, (1, 1)), (3, (5, 1, 1, 1))]:
        ax.plot(
            analytic_range,
            perturbative_beta(analytic_range, n_loops, rep, Nf, Nc),
            dashes=dashes,
            color="grey",
            label=f"{n_loops}-loop univ." if n_loops <= 2 else f"{n_loops}-loop GF",
        )
