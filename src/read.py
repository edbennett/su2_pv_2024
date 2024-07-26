#!/usr/bin/env python3

import gzip
import re

from flow_analysis.readers import readers

from joblib import Memory
import mpmath
import numpy as np
import pyerrors as pe
import rapidjson as json

from utils import partial_corr_mult

mpmath.mp.dps = 25
memory = Memory("cache")


def t_times_d_dt(corr, times, time_step, variant="symmetric"):
    # pyerrors can't cope if you multiply a derivative by a sequence,
    # as some elements are None

    d_corr_dt = corr.deriv(variant) / time_step
    return partial_corr_mult(times, d_corr_dt)


def normalize_coupling(corr, times, Nc, L):
    # arXiv:1208.1051 Eq. (1.3)
    # Note that the description therein has a typo:
    # theta is the Jacobi theta function, not the Jacobi elliptic function
    delta_plus_one = [
        (
            -64 * time**2 * mpmath.pi**2 / (3 * L**4)
            + mpmath.jtheta(3, 0, mpmath.exp(-(L**2) / (8 * time))) ** 4
        )
        for time in times
    ]

    # Rearrangement of arXiv:1208.1051 Eq. (1.2)
    # (to give arXiv:2402.18038 Eqs. (2) and (4))
    coefficient = [
        128 * mpmath.pi**2 / (element * 3 * (Nc**2 - 1)) for element in delta_plus_one
    ]

    return partial_corr_mult(np.asarray(coefficient, float), corr)


def get_metadata_from_filename(filename):
    Npv_s, beta_s, mpv_s, L_s = re.match(".*/out_wflow_([0-9]+)pv_beta([0-9.]+)_mpv([0-9.]+)_L([0-9]+)", filename).groups()
    Npv, beta, mpv, L = int(Npv_s), float(beta_s), float(mpv_s), int(L_s)

    return {"NT": L, "NX": L, "NY": L, "NZ": L, "Npv": Npv, "mpv": mpv, "beta": beta}


@memory.cache
def get_flows(filename, reader="hp", extra_metadata=None):
    flows = readers[reader](filename)
    if flows is None:
        return
    metadata = get_metadata_from_filename(filename)
    flows.metadata.update(metadata)
    if extra_metadata is not None:
        flows.metadata.update(extra_metadata)
    return flows


@memory.cache
def get_all_flows(filenames, reader="hp", operator="sym", extra_metadata=None):
    result = []
    for filename in filenames:
        flows = get_flows(filename, reader, extra_metadata)
        if flows is None:
            continue

        datum = {
            **flows.metadata,
            "filename": flows.filename,
            "h": flows.h,
            "t2E": flows.times**2 * flows.get_Es_pyerrors(operator=operator),
            "reader": flows.reader,
        }
        datum["gGF^2"] = normalize_coupling(
            datum["t2E"], flows.times, datum["Nc"], datum["NX"]
        )
        datum["betaGF"] = -t_times_d_dt(
            datum["gGF^2"], flows.times, flows.h, variant="improved"
        )

        for key in "t2E", "gGF^2", "betaGF":
            datum[key].gamma_method()

        result.append(datum)
    return result


def recurse_gamma(obj):
    if isinstance(obj, dict):
        recurse_gamma(obj.values())
        return
    if isinstance(obj, str):
        raise TypeError("Can't recurse into a string.")
    try:
        for value in obj:
            recurse_gamma(value)
    except TypeError:
        obj.gamma_method()


def read_fit_result(filename, pyerrors=True):
    if pyerrors:
        data = pe.input.json.load_json_dict(filename, verbose=False, full_output=True)
    else:
        with gzip.open(filename, "r") as f:
            data = json.load(f)
    data["filename"] = filename
    recurse_gamma(data["obsdata"])
    data.update(data.pop("description"))
    data.update(data.pop("obsdata"))
    if not pyerrors:
        data.update(data.pop("description"))
        data.update(data.pop("OBSDICT"))
    return data


def read_all_fit_results(filenames, pyerrors=True):
    return [read_fit_result(filename, pyerrors=pyerrors) for filename in filenames]
