from glob import glob
from itertools import product
import logging

from joblib import Memory
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyerrors as pe


memory = Memory("cache")

betas = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
masses = [-2.9, -2.8, -2.7, -2.7, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pv_specs = [(0, None), (5, 0.5), (5, 1.0), (10, 0.5), (10, 1.0), (15, 0.5), (15, 1.0)]


@memory.cache
def read_single_file(filename, therm=100):
    accept_threshold = 0.2

    tlen = None
    nsteps = None
    trajectories = []
    plaquettes = []
    accepts = []
    with open(filename, "r") as f:
        for line_number, line in enumerate(f):
            if line.startswith("[MD_INT][10]MD parameters:"):
                read_tlen = float(line.split()[3].split("=")[1])
                read_nsteps = int(line.split()[4].split("=")[1])
                if tlen is not None and read_tlen != tlen:
                    raise ValueError("Inconsistent tlen")
                if nsteps is not None and read_nsteps != nsteps:
                    raise ValueError("Inconsistent nsteps")
                tlen = read_tlen
                nsteps = read_nsteps

            elif line.startswith("[MAIN][0]Trajectory #"):
                trajectory = int(line.split()[1].strip("#:."))
            elif line.startswith("[MAIN][0]Plaquette:"):
                if trajectories and trajectory < trajectories[-1]:
                    logging.warning(f"File {filename} goes backwards; skipping line {line_number} onwards")
                    break
                trajectories.append(trajectory)
                plaquettes.append(float(line.split()[1]))
            elif line.startswith("[HMC][10]Configuration"):
                accepts.append(1 if line.split()[1] == "accepted." else 0)

    if len(plaquettes) < 21:
        logging.warning(f"Skipping {filename} as only {len(plaquettes)} trajectories")
        return None

    if (accept := sum(accepts) / len(accepts)) < accept_threshold:
        logging.warning(f"Skipping {filename} as acceptance = {accept}")
        return None

    try:
        plaquette = pe.Obs([plaquettes[therm:]], [filename], idl=[trajectories[therm:]])
    except ValueError:
        logging.warning(f"Skipping {filename} as insufficient independent samples")
        return None

    plaquette.gamma_method()
    return {
        "tlen": tlen,
        "nsteps": nsteps,
        "plaquette": plaquette,
    }


def get_plaquette(npv, mpv, beta, mass, subdir=""):
    data = []
    mpv_slug = "" if npv == 0 else f"_mpv{mpv}"
    for filename in glob(f"raw_data/{subdir}/out_hmc_{npv}pv_beta{beta}_m{mass}{mpv_slug}_*"):
        datum = read_single_file(filename)
        if datum:
            data.append(datum)
    if not data:
        return

    data.sort(key=lambda d: d["plaquette"].N / list(d["plaquette"].e_tauint.values())[0])
    return data[0]["plaquette"]


def get_plaquettes(subdir=""):
    results = pl.DataFrame([
        {
            "npv": npv,
            "mpv": float(mpv if mpv is not None else np.nan),
            "beta": beta,
            "mass": mass,
            "plaquette_value": (plaquette := get_plaquette(npv, mpv, beta, mass, subdir=subdir)) or np.nan,
            "plaquette_error": plaquette.dvalue if plaquette else np.nan,
        }
        for (npv, mpv), beta, mass in product(pv_specs, betas, masses)
    ])
    return results.with_columns(
        pl.col(pl.Float32,pl.Float64).fill_nan(None)
    ).drop_nulls(subset=["plaquette_value", "plaquette_error"])


def normalise(beta):
    return (np.log(beta) - np.log(min(betas))) / (np.log(max(betas)) - np.log(min(betas)))


def cycle(iterable):
    while True:
        for element in iterable:
            yield element


def get_title(npv, mpv):
    title = f"$N_{{\\mathrm{{PV}}}}={npv}$"
    if npv > 0:
        title = f"{title}, $m_{{\\mathrm{{PV}}}}={mpv:.1f}$"
    return title


def plot_phasediagram():
    subdir = "phasediagram"
    title = r"HMC + $m=10,m+\delta m=m_{\mathrm{PV}}$"

    plaquettes = get_plaquettes(subdir=subdir)
    plot_phasediagram_threepanel(plaquettes, title=title)
    plot_phasediagram_combined(plaquettes, title=title)


def filter(plaquettes, npv, mpv, beta):
    if mpv is None:
        return plaquettes.filter(pl.col("npv") == npv).filter(pl.col("beta") == beta)
    else:
        return plaquettes.filter(pl.col("npv") == npv).filter(pl.col("mpv") == mpv).filter(pl.col("beta") == beta)


def plot_phasediagram_combined(plaquettes, title=None, file_suffix=""):
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    markers = "os^vPHD<>*p3412X+"
    for style_index, (npv, mpv) in enumerate(pv_specs):
        label = None
        for beta in betas:
            subset = filter(plaquettes, npv, mpv, beta)
            if not subset.is_empty():
                label = get_title(npv, mpv)
            ax.errorbar(
                subset["mass"],
                subset["plaquette_value"],
                yerr=subset["plaquette_error"],
                marker=markers[style_index],
                color=f"C{style_index}",
            )
        ax.plot(
            [np.nan],
            [np.nan],
            color=f"C{style_index}",
            marker=markers[style_index],
            label=label,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("$m_0$")
    ax.set_ylabel(r"$\langle P \rangle$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))
    plt.savefig(f"plots/phasediagram_overlaid{'_' if file_suffix else ''}{file_suffix}.pdf")
    plt.close(fig)


def plot_phasediagram_threepanel(plaquettes, title=None, file_suffix=""):
    colormap = mpl.colormaps["plasma"]
    plot_set = [spec for spec in pv_specs if spec in set(plaquettes[["npv", "mpv"]].rows())]
    fig, axes = plt.subplots(ncols=len(plot_set), sharey=True, figsize=(1 + 2 * len(plot_set), 4), layout="constrained")
    for (npv, mpv), ax in zip(plot_set, axes):
        ax.set_title(get_title(npv, mpv))

        for beta, marker in zip(betas, cycle("os^v<>pD")):
            subset = filter(plaquettes, npv, mpv, beta)
            ax.errorbar(
                subset["mass"],
                subset["plaquette_value"],
                yerr=subset["plaquette_error"],
                label=f"{beta}",
                ls="none",
                marker=marker,
                color=colormap(normalise(beta)),
            )
        ax.set_xlabel("$m_0$")

    for beta in betas:
        subset = filter(plaquettes, 0, None, beta).sort(by="mass")
        visible_subset = subset.filter(pl.col("mass") < 0)
        for ax in axes:
            ax.axhline(
                visible_subset["plaquette_value"][-1],
                color=colormap(normalise(beta)),
                dashes=(4, 4),
                lw=0.5,
            )

    if title:
        fig.suptitle(title)
    axes[0].set_ylabel(r"$\langle P \rangle$")
    axes[-1].legend(loc="upper left", title=r"$\beta$", ncol=2, bbox_to_anchor=(1.1, 1.0))
    plt.savefig(f"plots/phasediagram{'_' if file_suffix else ''}{file_suffix}.pdf")
    plt.close(fig)


def set_plot_defaults():
    plt.style.use("styles/paperdraft.mplstyle")


if __name__ == "__main__":
    set_plot_defaults()
    plot_phasediagram()
