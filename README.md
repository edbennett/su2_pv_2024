# Towards the $\beta$ function of SU(2) with adjoint matter using Pauli&ndash;Villars fields&mdash;Analysis workflow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13128384.svg)](https://doi.org/10.5281/zenodo.13128384)

The workflow in this repository performs
the analyses presented in the poster
[Towards the $\beta$ function of SU(2) with adjoint matter using Pauli&ndash;Villars fields][poster]
presented at [Lattice 2024][lattice2024] in Liverpool.

Portions of it are closely based on the workflow
[Alternative analysis workflow for “Twelve flavor SU(3) gradient flow data for the continuous beta-function”][hp-pv].

## Requirements

- Conda, for example, installed from [Miniforge][miniforge]
- [Snakemake][snakemake], which may be installed using Conda
- LaTeX, for example, from [TeX Live][texlive]

## Setup

1. Install the dependencies above.
2. Clone this repository including submodules
   (or download its Zenodo release and `unzip` it)
   and `cd` into it:

   ```shellsession
   git clone --recurse-submodules https://github.com/edbennett/su2_pv_2024
   cd su2_pv_2024
   ```

3. Download and extract
   either the `raw_data_nf1.zip` or the `raw_data_nf2.zip` file from the [data release][datarelease].

## Running the workflow

The workflow is run using Snakemake:

``` shellsession
snakemake --cores 1 --use-conda
```

where the number `1`
may be replaced by
the number of CPU cores you wish to allocate to the computation.

Snakemake will automatically download and install
all required Python packages.
This requires an Internet connection;
if you are running in an HPC environment where you would need
to run the workflow without Internet access,
details on how to preinstall the environment
can be found in the [Snakemake documentation][snakemake-conda].

Using `--cores 6` on a MacBook Pro with an M1 Pro processor,
the analysis takes around 3 minutes.

## Output

Output plots are placed in the `assets/plots` directory.
When run with the `raw_data_nf1.zip` data,
this gives the plots in Figures 1 (a) and 2 (a) of the proceedings,
while with `raw_data_nf2.zip`,
it gives the plots in Figures 1 (b), 2 (b), 5, 6, and 7.

Intermediary data are placed in the `intermediary_data` directory.

## Extending the workflow

It is possible to add additional
numbers of Pauli&ndash;Villars fields,
values of $\beta$,
lattice volumes,
or operators,
by placing the relevant data files in the `data` directory
and updating the variables near the top of the file `workflow/Snakefile`.
(If additional operators are added,
an acronym will need to be defined for them
in the `operator_names` dict
in `src/names.py`.)

Other variables in `workflow/Snakefile`
control which parameter sets and ranges are included in each plot.
These will likely need to be changed
if this workflow is used to study other theories.

[datarelease]: https://doi.org/10.5281/zenodo.13128383
[hp-pv]: https://doi.org/10.5281/zenodo.13362605
[lattice2024]: https://conference.ippp.dur.ac.uk/event/1265/overview
[miniforge]: https://github.com/conda-forge/miniforge
[poster]: https://doi.org/10.5281/zenodo.13361520
[snakemake]: https://snakemake.github.io
[snakemake-conda]: https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html
[texlive]: https://tug.org/texlive/
