#!/usr/bin/env python3

import pandas as pd
import pyerrors as pe


def get_data(filenames):
    data = []
    for filename in filenames:
        datum = pe.input.json.load_json(filename, full_output=True, verbose=False)
        datum["obsdata"][0].gamma_method()
        data.append(
            {
                "Npv": datum["description"]["Npv"],
                "mpv": datum["description"]["mpv"],
                "beta": datum["description"]["beta"],
                "chisquare_per_dof": datum["description"]["chisquare"] / datum["description"]["dof"],
                "value_critical_mass": datum["obsdata"][0].value,
                "error_critical_mass": datum["obsdata"][0].dvalue,
            }
        )
    return pd.DataFrame(data)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("critical_mf_filenames", metavar="CRITICAL_MF_FILENAME", nargs="+")
    parser.add_argument("--output_filename", default="/dev/stdout")
    return parser.parse_args()


def main():
    args = get_args()
    data = get_data(args.critical_mf_filenames)
    data.to_csv(args.output_filename, index=False)


if __name__ == "__main__":
    main()
