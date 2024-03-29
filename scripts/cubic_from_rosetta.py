#!/usr/bin/env python3
# coding=utf-8
"""
cubic_from_rosetta.py script
@Author: Mads Jeppesen
@Date: 4/6/22
"""
from cubicsym.assembly.assemblyparser import AssemblyParser
import pathlib
import textwrap
import numpy as np
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
from argparse import RawDescriptionHelpFormatter

description = textwrap.dedent("""Creates a cubic symmetry file from a Rosetta output file and its symmetry definition file.

The input structure can be of 2 types.

1): A structure consisting of the symmetrical representation of a capsid in Rosetta (multiple chains). In that case
supply both --structures and --sym_files 

2): A structure consisting of a single chain with a SYMMETRY comment (EXPERIMENTAL FOR NOW). In that case supply only
--structures 

The simplest way to run the script is as so (with r1 being a Rosetta output file and s1 the symmetry file):

python --structures <r1> --symmetry_files <s1>

or (i1 being an input file)

python --structures <i1> 

The script can be used together with openmpi to accept multiple structures and symmetry files (r1/s1, r2/s2 and r3/s3):

mpirun -n <cores> python --structures <r1> <r2> <r3> --symmetry_files  <s1> <s2> <s3>

The order is important. So the structure r1 should match the symmetry file s1 and so on. This also goes for the
other options (see below).
""")

def main():
    """Generates a capsid representation from a structure file"""
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--structures', help="Structure output from rosetta.", nargs="+", required=True)
    parser.add_argument("-o", '--out_dirs',help="Paths to the output dir", default=".", nargs="+")
    parser.add_argument('--symmetry_files', help="Symmdef files used to represent the structures in Rosetta.", nargs="+")
    parser.add_argument('--out_names',help="Name given to output file.", default='<prefix>_<repr>.<ext>', nargs="+")
    parser.add_argument('--overwrite',help="To overwrite the file or not.", action="store_true")
    parser.add_argument('--out_repr', help="Representation to output", type=str, choices=["ico", "rosetta"], default="ico")
    args = parser.parse_args()

    # handle default values
    if args.out_dirs == ".":
        args.out_dirs = ["." for _ in range(len(args.structures))]
    if args.out_names == "<prefix>_<repr>.<ext>":
        if args.out_repr == "ico":
            args.out_names = [f"{pathlib.Path(struct).stem}_{args.out_repr}.cif" for struct in args.structures]
        else:
            args.out_names = [f"{pathlib.Path(struct).stem}_{args.out_repr}.pdb" for struct in args.structures]
    if args.symmetry_files == None:
        args.symmetry_files = [None for _ in range(len(args.structures))]

    # assert equal length for all
    assert all(len(l) == len(args.structures) for l in [args.symmetry_files, args.out_names, args.out_dirs])

    # distribute to each core
    if comm.rank == 0:
        structures = np.array_split(args.structures, comm.Get_size())
        out_dirs = np.array_split(args.out_dirs, comm.Get_size())
        out_names = np.array_split(args.out_names, comm.Get_size())
        symmetry_files = np.array_split(args.symmetry_files, comm.Get_size())
    else:
        structures, out_dirs, out_names, symmetry_files = None, None, None, None
    structures = comm.scatter(structures, root=0)
    out_dirs = comm.scatter(out_dirs, root=0)
    out_names = comm.scatter(out_names, root=0)
    symmetry_files = comm.scatter(symmetry_files, root=0)

    # the actual representation generation
    parser = AssemblyParser()
    for struct, out_dir, out_name, symmdef in zip(structures, out_dirs, out_names, symmetry_files):
        name = pathlib.Path(out_dir).joinpath(out_name)
        if name.exists():
            if not args.overwrite:
                print(name, " is already generated. Script will skip it.", flush=True)
                continue
        print("Generates: ", name, flush=True)
        if args.out_repr == "ico":
            if symmdef:
                assembly = parser.from_symmetric_output_pdb_and_symmetry_file(struct, symmdef)
            else:
                assembly = parser.capsid_from_asymmetric_output(struct)
            assembly.output(str(name))
        else:
            pose = parser.rosetta_representation_from_asymmetric_output(struct, symmdef)
            pose.dump_pdb(str(name))

if __name__ == '__main__':


    main()







