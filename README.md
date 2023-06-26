# cubicsym
This package provides various tools for modelling proteins with cubic symmetry in Rosetta and EvoDOCK. 
It can be used as a library but also contains import scripts for 2 purposes:

1. Generating cubic symmetry files for use in Rosetta and EvoDOCK
2. Generating full biological assemblies from the output of Rosetta and EvoDOCK

# Installation

**NOTE: This package is installed alongside EvoDOCK!**

For a standalone install the following must be installed: 
* MAFFT (https://mafft.cbrc.jp/alignment/software/) (can be installed with Anaconda/brew/apt)
* mpi4py and its requirements (https://mpi4py.readthedocs.io/en/stable/install.html) (can be install with Anaconda/pip)
* Specifc PyRosetta branch. Obtain a license as previously. Then download one of the following pyrosetta VERSIONS from [here](https://graylab.jhu.edu/download/PyRosetta4/conda/devel/linux-64/):

The different VERSIONS:
```
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py36_0.tar.bz2	
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py37_0.tar.bz2	
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py38_0.tar.bz2
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py39_0.tar.bz2
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py310_0.tar.bz2	
pyrosetta-2023.24.post.dev+48.commits.68ccf66-py311_0.tar.bz2	
```

Untar it:

```console
tar -xf pyrosetta-2023.24.post.dev+48.commits.68ccf66-<VERSION>.tar.bz2
```

You can then add it to your PYTHONPATH:

```console
export PYTHONPATH=$PYTHONPATH:<install directory>/lib/python<VERSION>/site-packages
```

or you can move the package to the site-packages for you python environment.
Then 

Clone the cubicsym repository and ```cd``` into it. Then run the install script.
```console
git clone https://github.com/Andre-lab/cubicsym.git
cd ./cubicsym
pip setup.py install 
```

You should be all set.

## Generating cubic symdef files 
```scripts/cubic_to_rosetta.py``` generates cubic symmetry files of either **Icosahedral**, **Octahedral** or **Tetrahedral symmetry**. 
The usage of the script is well documented. Use `python cubic_to_rosetta.py --help` to see more. 

There are 2 ways to run the script. An automatic way and a manual way. 

### Automatic way: 
When using the automatic way one only needs to specify --structures and --symmetry (see options below). 

A test can be run (inside the `cubicsym` dir) with

```console
python scripts/cubic_to_rosetta.py --structures tests/inputs/1stm.cif --symmetry I --symdef_outpath tests/outputs/ --input_outpath tests/outputs/ --rosetta_repr 1 --rosetta_repr_outpath tests/outputs/ --overwrite
```

### Manual way: 
In case the automatic way fails, one can use the manual way. When using the manual way one needs to specify all or some of the following:

* --hf1 
* --hf2 
* --hf3
* --f3 
* --f21 
* --f22

'hf' stands for highest fold which corresponds to the highest symmetrical fold for the system which for an icosahedral structure is the 5-fold, for the octahedral structure the 4-fold and for the tetrahedral structure the 3-fold. 'f3' stands for the 3-fold (which all cubic structures have) and f21 and f22 the two 2-folds. Subunit numbers are parsed to each of these options to determine the symmetry as they are assigned to by the script. To see the subunit numbers, first run the script with the flag: --output_generated_structure. This will generate an output file of the full biological assembly. Look at the output in a structural program (like PyMOL or Chimera) and from it assign the subunit numbers to the options. These numbers should always be related to the main subunit.

A test to create an output structure to assign subunit numbers from can be run with:

```console
python scripts/cubic_to_rosetta.py --structures tests/inputs/7m2v.cif --symmetry I --output_generated_structure --output_generated_structure_outpath tests/outputs --overwrite  
```

This will generate the following structure: `tests/outputs/7m2v_generated.cif`. From looking at that we can assign
the subunit numbers and run another test (inside the `cubicsym` dir) with the subunit numbers specified. 

In most cases it is just enough to supply the 3-fold as this is what the script can struggle to predict sometimes:

```console
python scripts/cubic_to_rosetta.py --structures tests/inputs/7m2v.cif --symmetry I --overwrite --f3 1 28 55 --symdef_outpath tests/outputs/ --input_outpath tests/outputs/ --rosetta_repr 1 --rosetta_repr_outpath tests/outputs/
```

A test for the full description would look like this:

```console
python scripts/cubic_to_rosetta.py --structures tests/inputs/7m2v.cif --symmetry I --overwrite --hf1 1 4 7 31 10 --hf2 55 52 58 16 49 --hf3 40 46 13 28 37 --f3 1 28 55 --f21 1 52 --f22 1 37 --symdef_outpath tests/outputs/ --input_outpath tests/outputs/ --rosetta_repr 1 --rosetta_repr_outpath tests/outputs/
```

## Generating full biological structures 
`scripts/cubic_from_rosetta.py` generates a full biological assembly files from the output of Rosetta or EvoDOCK.  
The usage of the script is well documented. Use `python cubic_form_rosetta.py --help` to see more.
A basic test can be run with

```console
python scripts/cubic_from_rosetta.py --structures tests/outputs/1stm_rosetta.pdb --symmetry_files tests/outputs/1stm.symm -o tests/outputs/
```


