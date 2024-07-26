#!/usr/bin/env python3

import argparse

import pyerrors as pe

from provenance import describe_inputs, get_consistent_metadata
from read import get_all_flows
from stats import weighted_mean
from utils import zip_combinations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_filenames", metavar="flow_filename", nargs="+")
    parser.add_argument("--reader", default="hirep")
    parser.add_argument("--operator", default="sym")
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument("--Npv", default=None, type=int)
    parser.add_argument("--mpv", default=None, type=float)
    parser.add_argument("--beta", default=None, type=float)
    return parser.parse_args()


def get_scales_at_time(flows, scale, time):
    result = []
    for flow in flows:
        index = int(time / flow["h"])
        result.append(flow[scale][index])

    return result


def linear_fit(a, x):
    return a[0] + a[1] * x


def fit_single(x_values, y_values):
    result = pe.fits.least_squares(x_values, y_values, linear_fit, silent=True)

    # Eq. (7) of 2402.18038 to compute AIC weight
    return result, result.chisquare_by_dof + 2 * len(result.fit_parameters)


def fit_scale(flows, scale, time):
    x_values = [1 / flow["NX"] ** 4 for flow in flows]
    scale_values = get_scales_at_time(flows, scale, time)
    for value in scale_values:
        value.gamma_method()
    fit_results = [
        fit_single(x_subset, scale_subset)
        for x_subset, scale_subset in zip_combinations(
            x_values, scale_values, min_count=3
        )
    ]
    return weighted_mean(fit_results)


def get_metadata(flows, operator, time):
    description = "Infinite volume extrapolation for gradient flow data."
    ensemble_keys = ["filename", "NX", "NY", "NZ", "NT", "reader"]
    consistent_keys = ["Npv", "mpv", "beta", "Nc"]
    return describe_inputs(
        flows,
        description,
        ensemble_keys,
        consistent_keys,
        operator=operator,
        time=time,
    )


def main():
    args = get_args()
    flows = get_all_flows(
        args.flow_filenames,
        reader=args.reader,
        operator=args.operator,
        extra_metadata={"Nc": 2, "Npv": args.Npv, "mpv": args.mpv, "beta": args.beta},
    )

    # Ensure a single consistent beta will be fit
    get_consistent_metadata(flows, "beta")

    result = {
        scale: fit_scale(flows, scale, args.time) for scale in ["gGF^2", "betaGF"]
    }

    if args.output_filename:
        pe.input.json.dump_dict_to_json(
            result,
            args.output_filename,
            description=get_metadata(flows, args.operator, args.time),
        )
    else:
        for observable, value in result.items():
            print(f"{observable}: {value}")


if __name__ == "__main__":
    main()
