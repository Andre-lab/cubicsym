#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates 3 and 2 fold symmetries from HF symmetry
@Author: Mads Jeppesen
@Date: 12/13/22
"""
from cubicsym.actors.symdefswapper import SymDefSwapper
from pyrosetta import init, pose_from_file
from pathlib import Path
from cubicsym.cubicsetup import CubicSetup

def create_other_symmetries(input, symdef, outdir):
    """Creates 3F and 2F symmetries"""
    init("-initialize_rigid_body_dofs")
    pose_HF = pose_from_file(input)
    cs = CubicSetup(symdef)
    cs.make_symmetric_pose(pose_HF)
    name = Path(input).stem
    sds = SymDefSwapper(pose_HF, symdef)
    # pose_3F = cs.make_asymmetric_pose(sds.create_3fold_pose_from_HFfold(pose_HF))
    # pose_2F = cs.make_asymmetric_pose(sds.create_2fold_pose_from_HFfold(pose_HF))
    # pose_3F.dump_pdb(input_outdir + "/" + name + "_3F.pdb")
    # pose_2F.dump_pdb(input_outdir + "/" + name + "_2F.pdb")
    sds.fold3F_setup.output(outdir + "/" + name + "_3F.symm")
    sds.fold2F_setup.output(outdir + "/" + name + "_2F.symm")

def main():
    import argparse
    description = "Generates 3F and 2F fold symmetries from HF symmetry"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', help="input file", type=str, required=True)
    parser.add_argument('--symdef', help="symdef file (should be HF based)", type=str, required=True)
    parser.add_argument('--outdir', help="directory to output the symmetry files", type=str, required=False, default=".")
    args = parser.parse_args()
    create_other_symmetries(args.input, args.symdef, args.outdir)

if __name__ == "__main__":
    main()