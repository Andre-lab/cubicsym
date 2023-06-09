#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 1/10/23
"""

def test_right_to_left_mapping():
    from simpletestlib.test import setup_test, setup_pymol
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.symdefnormalizer import SymdefNormalizer
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.private_paths import SYMMETRICAL
    from pyrosetta import pose_from_file, Pose
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.setupaligner import SetupAligner
    from simpletestlib.test import setup_test
    from cubicsym.utilities import pose_cas_are_identical
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str, set_all_translations_to_0
    from symmetryhandler.reference_kinematics import get_dofs
    from symmetryhandler.mathfunctions import rotation_matrix
    from pathlib import Path
    import pandas as pd
    import math
    from simpletestlib.test import setup_pymol
    from shapedesign.src.utilities.score import create_score_from_name
    from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    pdbs = {"I": ("1HQK",),
            "O": ("5EKW", "3WIS"),
            "T": ("2QQY",),
              }
    pmm = setup_pymol()
    for sym, pdbs in pdbs.items():
        for pdb in pdbs:
            pose_sym_org, symdef = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=True, mute=True, return_symmetry_file=True,
                                      pymol=False)
            sds = SymDefSwapper(pose_sym_org, symdef)
            sm = SymmetryMapper()
            for cs_org, pose_compare, fold, number in zip((sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup),
                                                  (pose_sym_org, sds.create_3fold_pose_from_HFfold(pose_sym_org),
                                                   sds.create_2fold_pose_from_HFfold(pose_sym_org)), ("HF", "3F", "2F"), ("HF", "31", "21")):
                cs = CubicSetup()
                cs.load_norm_symdef(sym, fold)

                # make the cn
                if fold == "HF":
                    pose_XF = sds.foldHF_setup.get_HF_chains(pose_sym_org)
                elif fold == "3F":
                    pose_XF = sds.foldHF_setup.get_3fold_chains(pose_sym_org)
                elif fold == "2F":
                    pose_XF, _ = sds.foldHF_setup.get_2fold_chains(pose_sym_org)
                pose_XF.dump_pdb("/tmp/blabla.pdb")
                pose_XF = pose_from_file("/tmp/blabla.pdb")

                if sym == "T":
                    if fold == "HF":
                        cn = "3"
                    elif fold == "3F":
                        cn = "3"
                    else:
                        cn = "2"
                elif sym == "O":
                    if fold == "HF":
                        cn = "4"
                    elif fold == "3F":
                        cn = "3"
                    else:
                        cn = "2"
                elif sym == "I":
                    if fold == "HF":
                        cn = "5"
                    elif fold == "3F":
                        cn = "3"
                    else:
                        cn = "2"

                cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym = sm.run(model=pose_XF, cn=cn,
                                                                                                symmetry=f"{sym}", chains_allowed=None,
                                                                                                T3F= sym == "T" and fold == "3F")

                # create asymmetric pose from pose_compare
                set_jumpdof_str_str(input_pose, f"JUMP{number}fold1", "z", cs_org.get_dof_value(f"JUMP{number}fold1", "z", "translation"))
                set_jumpdof_str_str(input_pose, f"JUMP{number}fold111", "x", cs_org.get_dof_value(f"JUMP{number}fold111", "x", "translation"))
                set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1", "z", cs_org.get_dof_value(f"JUMP{number}fold1", "z", "translation"))
                set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold111", "x", cs_org.get_dof_value(f"JUMP{number}fold111", "x", "translation"))
                if sym == "T":
                    if fold == "HF":
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 56)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", 56)
                        rmsd_threshold = 0.1
                    elif fold == "3F":
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -42)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", -42)
                        rmsd_threshold = 0.2
                        # fixme: Debug
                    elif fold == "2F":
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -33.7)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", -33.7)
                        rmsd_threshold = 0.4
                if sym == "O":
                    if fold == "HF":
                        if pdb == "3WIS":
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 17)
                            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", 17)
                            rmsd_threshold = 0.4
                        else:
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 29)
                            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", 29)
                            rmsd_threshold = 0.2
                    elif fold == "3F":
                        if pdb == "3WIS":
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -30)
                            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", -30)
                            rmsd_threshold = 0.6
                        else:
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 33)
                            rmsd_threshold = 0.4
                    elif fold == "2F":
                        if pdb == "3WIS":
                            continue # we do not have enough chains to match all of them
                            # below should give you about the best approximation
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -65)
                            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", -65)
                        else:
                            continue # we do not have enough chains to match all of them
                            # below should give you about the best approximation
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -85)
                            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -85)
                            pmm.apply(input_pose)
                elif sym == "I":
                    if fold == "HF":
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 16)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", 16)
                        rmsd_threshold = 0.3
                    elif fold == "3F":
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", -18)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", -18)
                        rmsd_threshold = 0.1
                    elif fold == "2F":
                        continue # we do not have enough chains to match all of them
                        # below should give you about the best approximation
                        set_jumpdof_str_str(input_pose, f"JUMP{number}fold1_z", "angle_z", 78)
                        set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1_z", "angle_z", 78)

                # score = create_score_from_name("ref2015")
                # score(input_pose)
                # score(input_pose_flip)
                # score(pose_sym_org)
                # input_pose.pdb_info().name("input")
                # input_pose_flip.pdb_info().name("input_flip")
                # pose_sym_org.pdb_info().name("org")
                # pmm.apply(input_pose)
                # pmm.apply(input_pose_flip)
                # pmm.apply(pose_sym_org)

                rmsd = cs.rmsd_hf_map(input_pose, pose_sym_org, same_handedness=False)
                rmsd_flip = cs.rmsd_hf_map(input_pose_flip, pose_sym_org, same_handedness=False)
                assert rmsd_threshold > rmsd or rmsd_threshold > rmsd_flip