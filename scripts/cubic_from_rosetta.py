#!/usr/bin/env python3
# coding=utf-8
"""
cubic_from_rosetta.py script
@Author: Mads Jeppesen
@Date: 4/6/22
"""
from cubicsym.assembly.assemblyparser import AssemblyParser
from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
from cubicsym.cubicsetup import CubicSetup
from pyrosetta import pose_from_file, init
import pathlib
import textwrap
import numpy as np
import argparse
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

multiple runs 

python --structures <r1> <r2> <r3> --symmetry_files  <s1> <s2> <s3>

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

    # the actual representation generation
    for struct, out_dir, out_name, symmdef in zip(args.structures, args.out_dirs, args.out_names, args.symmetry_files):
        name = pathlib.Path(out_dir).joinpath(out_name)
        if name.exists():
            if not args.overwrite:
                print(name, " is already generated. Script will skip it.", flush=True)
                continue
        print("Generates: ", name, flush=True)
        if args.out_repr == "ico":
            if symmdef:
                # check how many chains it has
                init("-initialize_rigid_body_dofs 1")
                pose = pose_from_file(struct)
                if pose.num_chains() < 9:
                    cs = CubicSetup(symmdef)
                    cs.make_symmetric_pose(pose)
                    assembly = CubicSymmetricAssembly.from_pose_input(pose, cs)
                else:
                    assembly = AssemblyParser().from_symmetric_output_pdb_and_symmetry_file(struct, symmdef)

            else:
                assembly = AssemblyParser().capsid_from_asymmetric_output(struct)
            assembly.output(str(name))
        else:
            pose = parser.rosetta_representation_from_asymmetric_output(struct, symmdef)
            pose.dump_pdb(str(name))

if __name__ == '__main__':


    main()







