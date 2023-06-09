# cubicsym
This package provides various tools for modelling proteins with cubic symmetry in Rosetta and EvoDOCK. 
It can be used as a library but also contains import scripts for 2 purposes:

1. Generating cubic symmetry files for use in Rosetta and EvoDOCK
2. Generating full biological assemblies from the output of Rosetta and EvoDOCK

# Installation

The following packages must be installed: 
* PyRosetta http://www.pyrosetta.org/dow (Can be installed with Anaconda)
* MAFFT (https://mafft.cbrc.jp/alignment/software/) (can be installed with brew/apt)

Clone the cubicsym repository and ```cd``` into it. Then run the install script.
```console
git clone https://github.com/Andre-lab/cubicsym.git
cd ./cubicsym
pip setup.py install 
```

if using M1 Mac chips install mpi4py seperately with brew:
```console
brew install mpi4py
```

## Generating cubic symdef files 
```scripts/cubic_to_rosetta.py``` generates cubic symmetry files of either **Icosahedral**, **Octahedral** or **Tetrahedral symmetry**. 
The usage of the script is well documented. Use `python cubic_to_rosetta.py --help` to see more. 
A basic test can be run (inside the `cubicsym` dir) with 

```console
python scripts/cubic_to_rosetta.py --structures tests/inputs/1stm.cif
```

## Generating full biological structures 
`scripts/cubic_from_rosetta.py` generates a full biological assembly files from the output of Rosetta or EvoDOCK.  
The usage of the script is well documented. Use `python cubic_form_rosetta.py --help` to see more.
A basic test can be run with

```console
python scripts/cubic_from_rosetta.py --structures tests/inputs/1stm.cif --symm_files 
```


