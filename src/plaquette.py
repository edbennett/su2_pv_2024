#!/usr/bin/env python3

from collections import defaultdict

from pyerrors import Obs

def read_plaquette_from_flows(filename):
    indices = defaultdict(list)
    plaquettes = defaultdict(list)

    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("[IO][0]Configuration"):
                continue
            split_line = line.split()
            cfg_filename = split_line[1].strip("[]")
            run_name = cfg_filename.split("/")[-1].split("_")[0]
            cfg_index = int(cfg_filename.split("n")[-1])
            plaquette = float(split_line[5].split("=")[-1])

            indices[run_name].append(cfg_index)
            plaquettes[run_name].append(plaquette)

    result = Obs(list(plaquettes.values()), list(plaquettes), idl=list(indices.values()))
    result.gamma_method()
    return result
