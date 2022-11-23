#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the class SymmetryMapper
@Author: Mads Jeppesen
@Date: 11/18/22
"""


# pose_org,  symdef = setup_test(name="O", file="1AEW", mute=True, return_symmetry_file=True, pymol=False)
# cs = CubicSetup(file=symdef)
# hf_chains = cs.get_HF_chains(pose_org)
# # to loose the symmetry we have to dump and reread
# hf_chains.dump_pdb("/tmp/badsfasdfsdfasfhladæfghh.pdb")
# hf_chains = pose_from_file("/tmp/badsfasdfsdfasfhladæfghh.pdb")
# f3_chains = cs.get_3fold_chains(pose)
# f2_1_chains, f2_2_chains = cs.get_2fold_chains(pose)
# out = "tests/outputs"
# hf_chains.dump_pdb(out + "/hffold.pdb")
# f3_chains.dump_pdb(out + "/3fold.pdb")
# f2_1_chains.dump_pdb(out + "/2_1fold.pdb")
# f2_2_chains.dump_pdb(out + "/2_2fold.pdb")

def test_apply():
    from simpletestlib.test import setup_test
    from pyrosetta import pose_from_file, init
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from pathlib import Path
    _, pmm, cmd, _ = setup_test(name="O", file="1AEW", mute=True, return_symmetry_file=True)
    init()
    sm = SymmetryMapper()

    # test a cn=2
    # 2CC9_c2.pdb
    pose = pose_from_file("outputs/2CC9_c2.pdb")
    sm.apply(pose, 2)

    # test a cn=3
    pose = pose_from_file("outputs/4DCL_c3.pdb")
    sm.apply(pose, 3)

    # test a cn=3
    pose = pose_from_file("outputs/af_predict_c3.pdb")
    sm.apply(pose, 3)

