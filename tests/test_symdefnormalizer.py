#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the SymdefNormalizer class
@Author: Mads Jeppesen
@Date: 12/13/22
"""

def test_apply_but_only_benchmark_folds():
    from cubicsym.symdefnormalizer import SymdefNormalizer
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.paths import SYMMETRICAL
    from pyrosetta import pose_from_file
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.setupaligner import SetupAligner
    from simpletestlib.test import setup_test
    from cubicsym.utilities import pose_cas_are_identical
    from symmetryhandler.reference_kinematics import get_dofs
    from symmetryhandler.mathfunctions import rotation_matrix
    from pathlib import Path
    import pandas as pd
    import math
    reinitialize=False
    selection = {
    "7Q03": "2F",
    "4DCL": "HF",
    "2CC9": "HF",
    "3LEO": "HF",
    "2QQY": "2F",
    "6M8V": "2F",
    "6HSB": "HF",

    "3WIS": "3F",
    "5H46": "2F",
    "5EKW": "3F",
    "3N1I": "HF",
    "6H05": "3F",
    "7O63": "2F",
    "7OHF": "2F",

    "1HQK": "HF",
    "1T0T": "HF",
    "1X36": "HF",
    "7B3Y": "3F",
    "4V4M": "HF",
    "1JH5": "3F",
    "6ZLO": "3F"
    }
    # ------------------------------------------------------------------------ Righthandedness:
    others = {"T": ("2QQY", "3LEO", "7Q03", "6M8V", "6HSB", "4DCL", "2CC9"), # False, rest are True | false works
              "O": ("3WIS", "5EKW", "5H46", "3N1I", "6H05", "7O63", "7OHF"), # False, False, rest are True | false, false works
              "I": ("1HQK", "1T0T", "7B3Y", "4V4M", "1JH5", "6ZLO", "1X36"), # False, False, False, False, rest are True, | false, false, works
              }

    df = {"symmetry":[], "pdb_base":[], "jump":[], "dof":[], "val":[], "org_val":[], "righthanded":[], "org_righthanded":[], "rmsd":[]}
    for pdb, sym in zip(("2CC9", "1STM", "7NTN"), ("T", "I", "O")):
        # SETUP
        symloc = SYMMETRICAL.joinpath(f"{sym}/idealized")
        symdef=str(symloc.joinpath(f"symdef/native/{pdb}.symm"))
        out=f"outputs/{pdb}_norm.symm"
        pose_original_unsymmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=reinitialize, mute=True)
        pose_original_symmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=reinitialize, mute=True)
        reinitialize = False
        pose_crystal = pose_from_file(str(symloc.joinpath(f"crystal_repr/native/{pdb}_crystal.pdb")))

        # make original CubicSetups
        sds = SymDefSwapper(pose_original_symmetrized, symdef)
        cs_hf, cs_3, cs_2 = sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup

        # make normalized CubicSetups
        sn = SymdefNormalizer()
        cs_hf_norm, cs_3_norm, cs_2_norm, hf_rot_angle = sn.apply(pose_original_unsymmetrized.clone(), symdef, 15, 5)

        ##########################################################################################################
        # TEST 2: On different structure                                                                         #
        # a) Make sure that we can recapitulate the original structure from the normalized symdef file           #
        # b) Make sure that when aligned they exactly overlap in space (with the same or below rmsd)             #
        ##########################################################################################################

        reinitialize = False
        for pdb in others[sym]:
            # get the original
            pose_original_symmetrized, pmm, cmd, org_symdef = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=reinitialize, return_symmetry_file=symdef, mute=True)
            reinitialize = False
            pose_crystal = pose_from_file(str(symloc.joinpath(f"crystal_repr/native/{pdb}_crystal.pdb")))
            # make original CubicSetups
            sds = SymDefSwapper(pose_original_symmetrized, org_symdef)
            same_handedness = None
            for base, ref in zip(("HF", "3F", "2F"), (sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup)):
                # a)
                norm = CubicSetup(f"../data/{sym}/{sym}_{base}_norm.symm")
                # we need the handedness to be defined from the hf_based on as
                if same_handedness is None and norm.is_hf_based():
                    same_handedness = norm.calculate_if_rightanded() == sds.foldHF_setup.calculate_if_rightanded()
                assert same_handedness is not None
                if selection[pdb] != base:
                    continue
                pose_original_unsymmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=reinitialize, mute=True)
                # get the starting positions. If we dont have those the setupaligner can find the optimimum through unwanted means such as through -x
                if norm.is_hf_based():
                    z_start = ref.get_dof_value("JUMPHFfold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMPHFfold111", "x", "translation")
                    # angle_z_start = 0 if same_handedness else 180
                elif norm.is_3f_based():
                    z_start = ref.get_dof_value("JUMP31fold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMP31fold111", "x", "translation")
                    # angle_z_start = 0 if same_handedness else 180
                elif norm.is_2f_based():
                    z_start = ref.get_dof_value("JUMP21fold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMP21fold111", "x", "translation")
                sa = SetupAligner(norm, sds.foldHF_setup, pose_original_unsymmetrized, overlap_tol=0.05, use_hf_chain_mapping=True, same_handedness=same_handedness,
                                  x_start=x_start, z_start=z_start)
                sa.apply()
                assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_original_symmetrized, same_handedness=same_handedness), sa.final_rmsd, abs_tol=0.1)
                assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_crystal, same_handedness=same_handedness), sa.final_rmsd, abs_tol=0.5) # 1T0T has 0.2 RMSD between the symmetrial and crystal so this should be high

                # [pdb_base, jump, dof, val, org_val, righthanded, org_righthanded]
                print(base, ":", get_dofs(sa.pose_from), get_dofs(sa.pose_to))
                for j_from, dp_from in get_dofs(sa.pose_from).items():
                    df["symmetry"].append(sym)
                    df["righthanded"].append(sa.al_from.righthanded)
                    df["org_righthanded"].append(sa.al_to.righthanded)
                    df["pdb_base"].append(f"{pdb}_{sa.al_from.get_base()}")
                    df["jump"].append(j_from)
                    dof = list(dp_from.keys())[0]
                    df["dof"].append(dof)
                    df["val"].append(list(dp_from.values())[0])
                    if base == "HF":
                       df["org_val"].append(get_dofs(pose_original_symmetrized)[j_from][dof]) # approx angle_z=56 for 2QQY
                    elif base == "3F":
                        df["org_val"].append(get_dofs(sds.create_3fold_pose(pose_original_symmetrized))[j_from][dof]) # approx angle_z=-41 for 2QQY
                    elif base == "2F":
                        df["org_val"].append(get_dofs(sds.create_2fold_pose(pose_original_symmetrized))[j_from][dof])
                    df["rmsd"].append(sa.final_rmsd)

                # b)
                # fixme: uncomment:
                # pose_from = sa.al_from.calpha_superimpose_pose_hf_map(sa.pose_from, sa.pose_to, same_handedness=same_handedness)
                # assert pose_cas_are_identical(pose_from, sa.pose_to, atol=0.5)

                # VISUAL:
                # pose_from.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_from")
                # sa.pose_to.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_to")
                # pmm.apply(pose_from)
                # pmm.apply(sa.pose_to)

    pd.DataFrame(df).to_csv("outputs/normalization_info_benchmark.csv", index=False)

def test_apply():
    from cubicsym.symdefnormalizer import SymdefNormalizer
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.paths import SYMMETRICAL
    from pyrosetta import pose_from_file
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.setupaligner import SetupAligner
    from simpletestlib.test import setup_test
    from cubicsym.utilities import pose_cas_are_identical
    from symmetryhandler.reference_kinematics import get_dofs
    from symmetryhandler.mathfunctions import rotation_matrix
    from pathlib import Path
    import pandas as pd
    import math
    reinitialize=False
    # ------------------------------------------------------------------------ Righthandedness:
    others = {"T": ("2QQY", "3LEO", "7Q03", "6M8V", "6HSB", "4DCL", "2CC9"), # False, rest are True | false works
              "O": ("3WIS", "5EKW", "5H46", "3N1I", "6H05", "7O63", "7OHF"), # False, False, rest are True | false, false works
              "I": ("1HQK", "1T0T", "7B3Y", "4V4M", "1JH5", "6ZLO", "1X36"), # False, False, False, False, rest are True, | false, false, works
              }

    df = {"symmetry":[], "pdb_base":[], "jump":[], "dof":[], "val":[], "org_val":[], "righthanded":[], "org_righthanded":[], "rmsd":[]}
    for pdb, sym in zip(("1STM", "2CC9", "7NTN"), ("I", "T", "O")):
        # SETUP
        symloc = SYMMETRICAL.joinpath(f"{sym}/idealized")
        symdef=str(symloc.joinpath(f"symdef/native/{pdb}.symm"))
        out=f"outputs/{pdb}_norm.symm"
        pose_original_unsymmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=reinitialize, mute=True)
        pose_original_symmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=reinitialize, mute=True)
        reinitialize = False
        pose_crystal = pose_from_file(str(symloc.joinpath(f"crystal_repr/native/{pdb}_crystal.pdb")))

        # make original CubicSetups
        sds = SymDefSwapper(pose_original_symmetrized, symdef)
        cs_hf, cs_3, cs_2 = sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup

        # make normalized CubicSetups
        sn = SymdefNormalizer()
        cs_hf_norm, cs_3_norm, cs_2_norm, hf_rot_angle = sn.apply(pose_original_unsymmetrized.clone(), symdef, 15, 5)

        ##########################################################################################################
        # TEST 1: On the same structure:                                                                         #
        # a) Make sure that from all the normalized CubicSetups we can recapitulate the original structure       #
        # b) Make sure that when rotated around the global z they exactly overlap in space (with the same rmsd)  #
        ##########################################################################################################

        # # hf-fold:
        for norm, ref in zip((cs_hf_norm, cs_3_norm, cs_2_norm),  (cs_hf, cs_3, cs_2)):
            # break # fixme, remove
            # 3-fold:
            # a)
            sa = SetupAligner(norm, ref, pose_original_unsymmetrized, overlap_tol=0.05)
            sa.apply()
            assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_original_symmetrized), sa.final_rmsd, abs_tol=0.1)
            assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_crystal), sa.final_rmsd, abs_tol=0.1)

            # b)
            pose_from = pose_original_unsymmetrized.clone()
            sa.al_from.make_symmetric_pose(pose_from)
            R_global_z_z = rotation_matrix([0, 0, 1], hf_rot_angle)
            pose_from.rotate(R_global_z_z)
            # if they are not perfect, the rotation can perhaps create some extra rmsd difference when staring from
            # non-ideal positions? I am not sure but sometimes it is much worse than sa.final_rmsd on its own.
            # but if you can find the rmsd to within 0.1 I am sure with a bit more docking you would be able to find the perfect overlap
            # so therefore 0.1 for me satisfies the overlap condition.
            assert pose_cas_are_identical(pose_from, sa.pose_to, atol=0.3)

            # VISUAL:
            # pose_from.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_from")
            # sa.pose_to.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_to")
            # pmm.apply(pose_from)
            # pmm.apply(sa.pose_to)

            # now save these different symmetries in the data folder
            data_out = f"../data/{sym}/{sym}_{sa.al_from.get_base()}_norm.symm"
            Path(data_out).parent.mkdir(parents=True, exist_ok=True)
            norm.output(data_out, headers=("normalized=True",))

        ##########################################################################################################
        # TEST 2: On different structure                                                                         #
        # a) Make sure that we can recapitulate the original structure from the normalized symdef file           #
        # b) Make sure that when aligned they exactly overlap in space (with the same or below rmsd)             #
        ##########################################################################################################
        continue

        reinitialize = False
        for pdb in others[sym]:
            # get the original
            pose_original_symmetrized, pmm, cmd, org_symdef = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=reinitialize, return_symmetry_file=symdef, mute=True)
            reinitialize = False
            pose_crystal = pose_from_file(str(symloc.joinpath(f"crystal_repr/native/{pdb}_crystal.pdb")))
            # make original CubicSetups
            sds = SymDefSwapper(pose_original_symmetrized, org_symdef)
            same_handedness = None
            for base, ref in zip(("HF", "3F", "2F"), (sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup)):
                # a)
                norm = CubicSetup(f"../data/{sym}/{sym}_{base}_norm.symm")
                # we need the handedness to be defined from the hf_based on as
                if same_handedness is None and norm.is_hf_based():
                    same_handedness = norm.calculate_if_rightanded() == sds.foldHF_setup.calculate_if_rightanded()
                assert same_handedness is not None
                pose_original_unsymmetrized, pmm, cmd = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=reinitialize, mute=True)
                # get the starting positions. If we dont have those the setupaligner can find the optimimum through unwanted means such as through -x
                if norm.is_hf_based():
                    z_start = ref.get_dof_value("JUMPHFfold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMPHFfold111", "x", "translation")
                    # angle_z_start = 0 if same_handedness else 180
                    angle_z_start = None
                elif norm.is_3f_based():
                    z_start = ref.get_dof_value("JUMP31fold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMP31fold111", "x", "translation")
                    # angle_z_start = 0 if same_handedness else 180
                elif norm.is_2f_based():
                    z_start = ref.get_dof_value("JUMP21fold1", "z", "translation")
                    x_start = ref.get_dof_value("JUMP21fold111", "x", "translation")
                sa = SetupAligner(norm, sds.foldHF_setup, pose_original_unsymmetrized, overlap_tol=0.05, use_hf_chain_mapping=True, same_handedness=same_handedness,
                                  x_start=x_start, z_start=z_start, angle_z_start=angle_z_start )
                sa.apply()
                assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_original_symmetrized, same_handedness=same_handedness), sa.final_rmsd, abs_tol=0.1)
                assert math.isclose(sa.al_from.rmsd_hf_map(sa.pose_from, pose_crystal, same_handedness=same_handedness), sa.final_rmsd, abs_tol=0.5) # 1T0T has 0.2 RMSD between the symmetrial and crystal so this should be high

                # [pdb_base, jump, dof, val, org_val, righthanded, org_righthanded]
                print(base, ":", get_dofs(sa.pose_from), get_dofs(sa.pose_to))
                for j_from, dp_from in get_dofs(sa.pose_from).items():
                    df["symmetry"].append(sym)
                    df["righthanded"].append(sa.al_from.righthanded)
                    df["org_righthanded"].append(sa.al_to.righthanded)
                    df["pdb_base"].append(f"{pdb}_{sa.al_from.get_base()}")
                    df["jump"].append(j_from)
                    dof = list(dp_from.keys())[0]
                    df["dof"].append(dof)
                    df["val"].append(list(dp_from.values())[0])
                    if base == "HF":
                       df["org_val"].append(get_dofs(pose_original_symmetrized)[j_from][dof]) # approx angle_z=56 for 2QQY
                    elif base == "3F":
                        df["org_val"].append(get_dofs(sds.create_3fold_pose_from_HFfold(pose_original_symmetrized))[j_from][dof]) # approx angle_z=-41 for 2QQY
                    elif base == "2F":
                        df["org_val"].append(get_dofs(sds.create_2fold_pose_from_HFfold(pose_original_symmetrized))[j_from][dof])
                    df["rmsd"].append(sa.final_rmsd)

                # b)
                # fixme: uncomment:
                # pose_from = sa.al_from.calpha_superimpose_pose_hf_map(sa.pose_from, sa.pose_to, same_handedness=same_handedness)
                # assert pose_cas_are_identical(pose_from, sa.pose_to, atol=0.5)

                # VISUAL:
                # pose_from.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_from")
                # sa.pose_to.pdb_info().name(f"{pdb}_{sa.al_from.get_base()}_to")
                # pmm.apply(pose_from)
                # pmm.apply(sa.pose_to)

    pd.DataFrame(df).to_csv("outputs/normalization_info2.csv", index=False)