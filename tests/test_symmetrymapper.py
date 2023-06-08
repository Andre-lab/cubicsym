#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the class SymmetryMapper
@Author: Mads Jeppesen
@Date: 11/18/22
"""
def test_prediction_structures():
    from cubicsym.cubicsetup import CubicSetup
    from pyrosetta import pose_from_file, Pose
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.setupaligner import SetupAligner
    from simpletestlib.test import setup_test
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str, set_all_translations_to_0
    from symmetryhandler.reference_kinematics import get_dofs
    import pandas as pd
    from simpletestlib.test import setup_pymol
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from pathlib import Path
    from shapedesign.src.utilities.score import create_score_from_name
    import math
    T_selection = {
        "7Q03": 2,  # (is closest to the 2_1 in RMSD)
        "4DCL": 3,  # hf/3 (is closest to the hf in RMSD)
        "2CC9": 3,  # hf/3 (is closest to the hf in RMSD)
        "3LEO": 3,  # hf/3 (is closest to the hf in RMSD)
        "2QQY": 2,  # (is closest to the 2_1 in RMSD)
        "6M8V": 2,  # (is closest to the 2_1 in RMSD)
        "6HSB": 3  # hf/2"(is closest to the hf in RMSD)
    }
    O_selection = {
        "3WIS": 3,
        "5H46": 2,
        "5EKW": 3,
        "3N1I": 4,
        "6H05": 3,
        "7O63": 2,
        "7OHF": 2
    }

    I_selection = {
        "1HQK": 5,
        "1T0T": 5,
        "1X36": 5,
        "7B3Y": 3,
        "4V4M": 5,
        "1JH5": 3,
        "6ZLO": 3
    }
    df = {"symmetry": [], "pdb_base": [], "jump": [], "dof": [], "val": [], "org_val": [], "righthanded": [], "org_righthanded": [],
          "rmsd": [], "steps": [], "flip": []}
    pmm = setup_pymol()
    for sym, pdb2n in zip(("T", "O", "I"), (T_selection, O_selection, I_selection)):
        for pdb, cn in pdb2n.items():
            pose_sym_org, symdef = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=True, mute=True, return_symmetry_file=True,
                                              pymol=False)
            pose_sym_org_asym = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=True, mute=True, return_symmetry_file=False,
                                           pymol=False)
            model = list(Path(f"/home/mads/production/afoncubic/final/{sym}/global_from_mer/output").glob(f"{pdb}*/relaxed_model_1_multimer_v2_pred_0.pdb"))[0]
            # cn = int(model.parent.stem.split("_")[1])
            pose_XF = pose_from_file(str(model))

            if sym == "T":
                if cn == 3:
                    fold = "HF"
                elif cn == 2:
                    fold = "2F"
            elif sym == "O":
                if cn == 4:
                    fold = "HF"
                elif cn == 3:
                    fold = "3F"
                else:
                    fold = "2F"
            elif sym == "I":
                if cn == 5:
                    fold = "HF"
                elif cn == 3:
                    fold = "3F"
                else:
                    fold = "2F"

            sds = SymDefSwapper(pose_sym_org, symdef)
            if fold == "HF":
                cs_org = sds.foldHF_setup
                number = "HF"
            elif fold == "3F":
                cs_org = sds.fold3F_setup
                number = "31"
            else:
                cs_org = sds.fold2F_setup
                number = "21"

            cs = CubicSetup()
            cs.load_norm_symdef(sym, fold)

            sm = SymmetryMapper()
            cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym, x_trans, chain_used = sm.run(model=pose_XF, cn=cn,
                                                                                            symmetry=sym, chains_allowed=None,
                                                                                            T3F = sym == "T" and fold == "3F")

            # create asymmetric pose from pose_compare
            z_start = cs_org.get_dof_value(f"JUMP{number}fold1", "z", "translation")
            x_start = cs_org.get_dof_value(f"JUMP{number}fold111", "x", "translation")
            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1", "z", z_start)
            set_jumpdof_str_str(input_pose, f"JUMP{number}fold111", "x", x_start)
            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1", "z", z_start)
            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold111", "x", x_start)



            for pose, flip in zip((input_pose_asym, input_pose_flip_asym), ("", "flip")):
                sa = SetupAligner(cs, sds.foldHF_setup, use_hf_chain_mapping=True, overlap_tol=0.05, x_start=x_start, z_start=z_start,
                                  al_to_monomeric=pose_sym_org_asym, al_from_monomeric=pose,
                                  same_handedness=sds.foldHF_setup.calculate_if_rightanded(), behavior="global",
                                  assert_overlap_tol=False)
                sa.apply()
                if sa.final_rmsd < 2:
                    break

                # 1. save final structure
                score = create_score_from_name("ref2015")
                score(sa.pose_from)
                score(sa.pose_to)
                sa.pose_from.dump_pdb(f"/home/mads/projects/cubicsym/tests/outputs/global_from_mer_prediction/{pdb}_{number}.pdb")
                sa.pose_to.dump_pdb(f"/home/mads/projects/cubicsym/tests/outputs/global_from_mer_prediction/{pdb}_{number}_native.pdb")
                # 2. add to the dataframe
                for j_from, dp_from in get_dofs(sa.pose_from).items():
                    df["symmetry"].append(sym)
                    df["flip"].append(True if flip == "flip" else False)
                    df["righthanded"].append(sa.al_from.righthanded)
                    df["org_righthanded"].append(sa.al_to.righthanded)
                    df["pdb_base"].append(f"{pdb}_{sa.al_from.get_base()}")
                    df["jump"].append(j_from)
                    dof = list(dp_from.keys())[0]
                    if fold == "HF":
                        df["org_val"].append(get_dofs(pose_sym_org)[j_from][dof])
                    elif fold == "3F":
                        df["org_val"].append(get_dofs(sds.create_3fold_pose_from_HFfold(pose_sym_org))[j_from][
                                                 dof])
                    elif fold == "2F":
                        df["org_val"].append(get_dofs(sds.create_2fold_pose_from_HFfold(pose_sym_org))[j_from][dof])
                    df["dof"].append(dof)
                    df["val"].append(list(dp_from.values())[0])
                    df["rmsd"].append(sa.final_rmsd)
                    df["steps"].append(sa.steps)

            pd.DataFrame(df).to_csv("/home/mads/projects/cubicsym/tests/outputs/global_from_mer_prediction/results.csv", index=False)

def test_native_structures():
    from cubicsym.cubicsetup import CubicSetup
    from pyrosetta import pose_from_file, Pose
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.setupaligner import SetupAligner
    from simpletestlib.test import setup_test
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str, set_all_translations_to_0
    from symmetryhandler.reference_kinematics import get_dofs
    import pandas as pd
    from simpletestlib.test import setup_pymol
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    import math
    T_selection = {
    "7Q03": 2,  # (is closest to the 2_1 in RMSD)
    "4DCL": 3,  # hf/3 (is closest to the hf in RMSD)
    "2CC9": 3,  # hf/3 (is closest to the hf in RMSD)
    "3LEO": 3,  # hf/3 (is closest to the hf in RMSD)
    "2QQY": 2,  # (is closest to the 2_1 in RMSD)
    "6M8V": 2,  # (is closest to the 2_1 in RMSD)
    "6HSB": 3  # hf/2"(is closest to the hf in RMSD)
    }
    O_selection = {
    "3WIS": 3,
    "5H46": 2,
    "5EKW": 3,
    "3N1I": 4,
    "6H05": 3,
    "7O63": 2,
    "7OHF": 2
    }
    I_selection = {
    "1HQK": 5,
    "1T0T": 5,
    "1X36": 5,
    "7B3Y": 3,
    "4V4M": 5,
    "1JH5": 3,
    "6ZLO": 3
    }
    df = {"symmetry": [], "pdb_base": [], "jump": [], "dof": [], "val": [], "righthanded": [], "org_righthanded": [],
          "rmsd": []}
    pmm = setup_pymol()
    for sym, pdb2n in zip(("T", "O", "I"), (T_selection, O_selection, I_selection)):
        for pdb, cn in pdb2n.items():
            if sym == "T":
                if cn == 3:
                    fold = "HF"
                elif cn == 2:
                    fold = "2F"
            elif sym == "O":
                if cn == 4:
                    fold = "HF"
                elif cn == 3:
                    fold = "3F"
                else:
                    fold = "2F"
            elif sym == "I":
                if cn == 5:
                    fold = "HF"
                elif cn == 3:
                    fold = "3F"
                else:
                    fold = "2F"

            pose_sym_org, symdef = setup_test(name=sym, file=pdb, symmetrize=True, reinitialize=True, mute=True, return_symmetry_file=True,
                                              pymol=False)
            pose_sym_org_asym = setup_test(name=sym, file=pdb, symmetrize=False, reinitialize=True, mute=True, return_symmetry_file=False,
                                              pymol=False)

            sds = SymDefSwapper(pose_sym_org, symdef)
            if fold == "HF":
                cs_org = sds.foldHF_setup
                number = "HF"
            elif fold == "3F":
                cs_org = sds.fold3F_setup
                number = "31"
            else:
                cs_org = sds.fold2F_setup
                number = "21"


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


            sm = SymmetryMapper()
            cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym, x_trans = sm.run(model=pose_XF, cn=cn,
                                                                                            symmetry=sym, chains_allowed=None,
                                                                                            T3F= sym == "T" and fold == "3F")

            # create asymmetric pose from pose_compare
            z_start = cs_org.get_dof_value(f"JUMP{number}fold1", "z", "translation")
            x_start = cs_org.get_dof_value(f"JUMP{number}fold111", "x", "translation")
            set_jumpdof_str_str(input_pose, f"JUMP{number}fold1", "z", z_start)
            set_jumpdof_str_str(input_pose, f"JUMP{number}fold111", "x", x_start)
            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold1", "z", z_start)
            set_jumpdof_str_str(input_pose_flip, f"JUMP{number}fold111", "x", x_start)

            assert math.isclose(x_start, x_trans, abs_tol=1e-3)

            for pose in (input_pose_asym, input_pose_flip_asym):
                sa = SetupAligner(cs, sds.foldHF_setup, use_hf_chain_mapping=True, overlap_tol=0.05, x_start=x_start, z_start=z_start,
                                  al_to_monomeric=pose_sym_org_asym, al_from_monomeric=pose,
                                  same_handedness=sds.foldHF_setup.calculate_if_rightanded(), behavior="fold_rotation",
                                  assert_overlap_tol=False)
                sa.apply()
                if sa.final_rmsd < 0.5:
                    break
            # 1. save final structure
            from shapedesign.src.utilities.score import create_score_from_name
            score = create_score_from_name("ref2015")
            score(sa.pose_from)
            score(sa.pose_to)
            sa.pose_from.dump_pdb(f"/home/mads/projects/cubicsym/tests/outputs/global_from_mer_native/{pdb}_{number}.pdb")
            sa.pose_to.dump_pdb(f"/home/mads/projects/cubicsym/tests/outputs/global_from_mer_native/{pdb}_{number}_native.pdb")
            # 2. add to the dataframe
            for j_from, dp_from in get_dofs(sa.pose_from).items():
                df["symmetry"].append(sym)
                df["righthanded"].append(sa.al_from.righthanded)
                df["org_righthanded"].append(sa.al_to.righthanded)
                df["pdb_base"].append(f"{pdb}_{sa.al_from.get_base()}")
                df["jump"].append(j_from)
                dof = list(dp_from.keys())[0]
                df["dof"].append(dof)
                df["val"].append(list(dp_from.values())[0])
                df["rmsd"].append(sa.final_rmsd)

    pd.DataFrame(df).to_csv("/home/mads/projects/cubicsym/tests/outputs/global_from_mer_native/results.csv", index=False)


def test_different_rotations():
    from simpletestlib.test import setup_pymol, setup_test
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from pyrosetta import pose_from_file, init
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    import  pandas as pd
    sm = SymmetryMapper()
    pmm = setup_pymol()
    saves = []
    info = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/normalization_info.csv")
    pose_hf, symdef = setup_test(name="T", file="7Q03", symmetrize=True, reinitialize=False, mute=True, return_symmetry_file=True,
                                 pymol=False)
    for pdb in ("bad_1", "bad_2", "good_1"):
        pdb = pose_from_file(f"/home/mads/projects/cubicsym/tests/outputs/{pdb}.pdb")
        cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym = sm.run(model=pdb, cn=f"{2}", symmetry=f"{'T'}", chains_allowed=None)
        for jump, dof, val in info[info["pdb_base"] == f"{'7Q03'}_{2}F"][["jump", "dof", "val"]].values:
            # IT does not need the rotations as they are stored in rotation of the monomer
            if dof == "x":
                set_jumpdof_str_str(input_pose_flip, jump, dof, val)
            elif dof == "z":
                set_jumpdof_str_str(input_pose_flip, jump, dof, val)
            elif dof == "angle_z" and "fold1_z" in jump:
                set_jumpdof_str_str(input_pose_flip, jump, dof, val)
        for jump, dof, val in info[info["pdb_base"] == f"{'7Q03'}_{2}F"][["jump", "dof", "val"]].values:
            # IT does not need the rotations as they are stored in rotation of the monomer
            if dof == "x":
                set_jumpdof_str_str(input_pose, jump, dof, val)
            elif dof == "z":
                set_jumpdof_str_str(input_pose, jump, dof, val)
            elif dof == "angle_z" and "fold1_z" in jump:
                set_jumpdof_str_str(input_pose, jump, dof, val)
        rmsd = cs.rmsd_hf_map(input_pose, pose_hf, same_handedness=True)
        rmsd_flip = cs.rmsd_hf_map(input_pose_flip, pose_hf, same_handedness=True)
        assert rmsd < 0.1 or rmsd_flip < 0.1

def test_run_recapitulation():
    from simpletestlib.test import setup_pymol, setup_test
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.symdefnormalizer import SymdefNormalizer

    from pathlib import Path
    from cubicsym.paths import T
    from cubicsym.setupaligner import SetupAligner
    from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit
    import pandas as pd
    from pyrosetta import Pose
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from pyrosetta import pose_from_file
    pmm = setup_pymol()
    T_selection = {
        "2QQY": "2", #(is closest to the 2_1 in RMSD)
        "4DCL": "H", #hf/3 (is closest to the hf in RMSD)
        "7Q03": "2", #(is closest to the 2_1 in RMSD)
        "2CC9": "H", # hf/3 (is closest to the hf in RMSD)
        "3LEO": "H", #hf/3 (is closest to the hf in RMSD)
        "6M8V": "2", #(is closest to the 2_1 in RMSD)
        "6HSB": "H" #hf/2"(is closest to the hf in RMSD)
        }
    sm = SymmetryMapper()
    info = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/normalization_info.csv")
    for pdb, cn in T_selection.items():
        for path in Path(f"/home/mads/production/afoncubic/final/{'T'}/global_from_mer/output").glob(f"{pdb}*"):
            for model in path.glob("relaxed_model_1_multimer_v2_pred_0.pdb"):
                pose_hf, symdef = setup_test(name="T", file=pdb, symmetrize=True, reinitialize=False, mute=True, return_symmetry_file=True,
                                             pymol=False)
                pose_hf_asym = setup_test(name="T", file=pdb, symmetrize=False, reinitialize=False, mute=True, return_symmetry_file=False,
                                             pymol=False)
                sds = SymDefSwapper(pose_hf, symdef)
                cs_hf, cs_3, cs_2 = sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup

                # make the cn
                if cn == "H":
                    pose_XF = cs_hf.get_HF_chains(pose_hf)
                elif cn == "2":
                    pose_XF, _ = cs_hf.get_2fold_chains(pose_hf)
                temp_id = "/tmp/12345.pdb"
                # randomize and rotate
                pose_XF.rotate(R.random().as_matrix())
                pose_XF.translate(np.array([5,5,5]))
                pose_XF.dump_pdb(temp_id)
                pose_XF.pdb_info().obsolete(True) # now the first chain is A
                pose_XF = pose_from_file(temp_id)

                cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym = sm.run(model=pose_XF, cn=f"{3 if cn == 'H' else cn}", symmetry=f"{'T'}", chains_allowed=None)
                set_jumpdof_str_str(input_pose_flip, "JUMP21fold1", "z", sds.fold2F_setup.get_dof_value("JUMP21fold1", "z", "translation"))
                set_jumpdof_str_str(input_pose_flip, "JUMP21fold111", "x", sds.fold2F_setup.get_dof_value("JUMP21fold111", "x", "translation"))
                set_jumpdof_str_str(input_pose, "JUMP21fold1", "z", sds.fold2F_setup.get_dof_value("JUMP21fold1", "z", "translation"))
                set_jumpdof_str_str(input_pose, "JUMP21fold111", "x", sds.fold2F_setup.get_dof_value("JUMP21fold111", "x", "translation"))
                # for jump, dof, val in info[info["pdb_base"] == f"{pdb}_{cn}F"][["jump", "dof", "val"]].values:
                #     # IT does not need the rotations as they are stored in rotation of the monomer
                #     if dof == "x":
                #         set_jumpdof_str_str(input_pose_flip, jump, dof, val)
                #         start_x = val
                #     elif dof == "z":
                #         set_jumpdof_str_str(input_pose_flip, jump, dof, val)
                #         start_z = val
                #     elif dof == "angle_z" and "fold1_z" in jump:
                #         set_jumpdof_str_str(input_pose_flip, jump, dof, val)
                #         angle_z_start = val
                # for jump, dof, val in info[info["pdb_base"] == f"{pdb}_{cn}F"][["jump", "dof", "val"]].values:
                #     # IT does not need the rotations as they are stored in rotation of the monomer
                #     if dof == "x":
                #         set_jumpdof_str_str(input_pose, jump, dof, val)
                #     elif dof == "z":
                #         set_jumpdof_str_str(input_pose, jump, dof, val)
                #     elif dof == "angle_z" and "fold1_z" in jump:
                #         set_jumpdof_str_str(input_pose, jump, dof, val)

                # quick test
                # cs is always righthanded as it is created from the normalized symdefs, so theres no need to do cs.is_righthanded == cs_hf.is_righthanded as in the test_symdefnormalizer
                rmsd = cs.rmsd_hf_map(input_pose, pose_hf, same_handedness=cs_hf.calculate_if_rightanded()) # cs is always righthanded as it is created from the normalized symdefs, so theres no need to do cs.is_righthanded == cs_hf.is_righthanded as in the test_symdefnormalizer
                rmsd_flip = cs.rmsd_hf_map(input_pose_flip, pose_hf, same_handedness=cs_hf.calculate_if_rightanded())
                # assert rmsd < 0.1 or rmsd_flip < 0.1
                print(f"{pdb} WORKED!")
                sa = SetupAligner(cs, cs_hf, use_hf_chain_mapping=True, overlap_tol=0.05, x_start=sds.fold2F_setup.get_dof_value("JUMP21fold111", "x", "translation"), z_start=sds.fold2F_setup.get_dof_value("JUMP21fold1", "z", "translation"),
                                  al_to_monomeric=pose_hf_asym, al_from_monomeric=input_pose_flip_asym,
                                  same_handedness=cs_hf.calculate_if_rightanded()  # cs is always righthanded as it is created from the normalized symdefs, so theres no need to do cs.is_righthanded == cs_hf.is_righthanded as in the test_symdefnormalizer
                                  )
                # sa = SetupAligner(cs, cs_hf, use_hf_chain_mapping=True, overlap_tol=0.05, x_start=start_x, z_start=start_z, angle_z_start=angle_z_start,
                #                   al_to_monomeric=pose_hf_asym, al_from_monomeric=input_pose_flip_asym,
                #                   same_handedness=cs_hf.is_rightanded() # cs is always righthanded as it is created from the normalized symdefs, so theres no need to do cs.is_righthanded == cs_hf.is_righthanded as in the test_symdefnormalizer
                #                   )
                # todo: tmr:
                #  for all cases in cubicsym.utilities match everything. There should be very little discrepency as below, just move the z_angle
                #  a bit. write a test for it. Be carefull if you use the above randomizer as 50% of the time input_pose_flip_asym will not have the correct orientation.
                #  {'JUMP21fold1': {'z': 31.066805237647525},
                #   'JUMP21fold111': {'x': 8.273416250301478},
                #   'JUMP21fold111_x': {'angle_x': -0.1849447346782014},
                #   'JUMP21fold111_y': {'angle_y': -0.043225321050432614},
                #   'JUMP21fold111_z': {'angle_z': 0.039768572628485925},
                #   'JUMP21fold1_z': {'angle_z': -34.740637451374376}}
                sa.apply()
                ...

def test_run():
    from simpletestlib.test import setup_pymol, setup_test
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.symdefnormalizer import SymdefNormalizer
    from pathlib import Path
    from cubicsym.paths import T
    from cubicsym.setupaligner import SetupAligner
    import pandas as pd
    pmm = setup_pymol()
    T_selection = {
        # "4DCL": 3, #hf/3 (is closest to the hf in RMSD)
        "7Q03": 2, #(is closest to the 2_1 in RMSD)
        #"2QQY": 2, #(is closest to the 2_1 in RMSD)
        "2CC9": 3, # hf/3 (is closest to the hf in RMSD)
        # "3LEO": 3, #hf/3 (is closest to the hf in RMSD)
        # "7Q03": 2, #(is closest to the 2_1 in RMSD)
        # "6M8V": 2, #(is closest to the 2_1 in RMSD)
        # "6HSB": 3 #hf/2"(is closest to the hf in RMSD)
        }
    sm = SymmetryMapper()
    info = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/normalization_info.csv")
    for pdb in T_selection:
        for path in Path(f"/home/mads/production/afoncubic/final/{'T'}/global_from_mer/output").glob(f"{pdb}*"):
            for model in path.glob("relaxed_model_1_multimer_v2_pred_0.pdb"):
                # remove vvvvv
                pose_hf, symdef = setup_test(name="T", file=pdb, symmetrize=True, reinitialize=False, mute=True, return_symmetry_file=True,
                                             pymol=False)
                pose_hf_asym = setup_test(name="T", file=pdb, symmetrize=False, reinitialize=False, mute=True, return_symmetry_file=False,
                                             pymol=False)
                sds = SymDefSwapper(pose_hf, symdef)
                cs_hf, cs_3, cs_2 = sds.foldHF_setup, sds.fold3F_setup, sds.fold2F_setup
                # remove ^^^^^^^

                cn = model.parent.stem.split("_")[1]
                cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym = sm.run(model=model, cn=cn, symmetry=f"{'T'}", chains_allowed=None)
                for jump, dof, val in info[info["pdb_base"] == f"{pdb}_{2}F"][["jump", "dof", "val"]].values:
                    if len(dof) == 1: # translation
                        set_jumpdof_str_str(input_pose_flip, jump, dof, val)
                # quick test
                # sa = SetupAligner(cs, cs_hf, input_pose_flip_asym, overlap_tol=0.05)
                # sa.apply()
                ...


def test_apply():
    from simpletestlib.test import setup_test
    from pyrosetta import pose_from_file, init
    from cubicsym.alphafold.symmetrymapper import SymmetryMapper
    from pathlib import Path
    _, pmm, cmd, _ = setup_test(name="O", file="1AEW", mute=True, return_symmetry_file=True)
    init()
    sm = SymmetryMapper()

    # try some that didnt
    name = "/home/mads/production/afoncubic/final/T/031122/output/3FDD_2_1/relaxed_model_2_multimer_v2_pred_0.pdb"
    pose = pose_from_file(name)
    sm.find_combos(pose, 2)

    # test a cn=2
    # 2CC9_c2.pdb
    pose = pose_from_file("outputs/2CC9_c2.pdb")
    sm.find_combos(pose, 2)

    # test a cn=3
    pose = pose_from_file("outputs/4DCL_c3.pdb")
    sm.find_combos(pose, 3)

    # test a cn=3
    pose = pose_from_file("outputs/af_predict_c3.pdb")
    sm.find_combos(pose, 3)

# def test_run():
#     from simpletestlib.test import setup_test
#     from cubicsym.alphafold.symmetrymapper import SymmetryMapper
#     from pathlib import Path
#     _, pmm, cmd, _ = setup_test(name="O", file="1AEW", mute=True, return_symmetry_file=True)
#     dumpfolder = f"/home/mads/projects/cubicsym/tests/outputs/3LEO_3_dump"
#     model_dir = f"/home/mads/projects/cubicsym/tests/inputs/3LEO_3"
#     model_out = f"/home/mads/projects/cubicsym/tests/outputs/3LEO_3"
#     sm = SymmetryMapper(tmp_file_dir=dumpfolder)
#     sym_name, x_min, x_max = sm.run(models=Path(model_dir).glob("*"),
#                                     mer=3, model_outdir=model_out, sym_outdir="/home/mads/projects/evodock/inputs/symmetry_files",
#                                     symname="3LEO_multi", main_chains_allowed=[["B"] for _ in Path(model_dir).glob("*")],
#                                     pymolmover=pmm)
#
# def test_run_rmsd_is_good():
#     from simpletestlib.test import setup_test
#     from cubicsym.alphafold.symmetrymapper import SymmetryMapper
#     from pathlib import Path
#     path = "/home/mads/production/afoncubic/final/T"
#     for pdb_path in Path(path).glob("global_from_mer/output/*"):
#         # if not "6HSB_3" in str(pdb_path):
#         #     continue
#         dumpfolder = f"/home/mads/projects/cubicsym/tests/outputs/{pdb_path.stem}_dump"
#         Path(dumpfolder).mkdir(parents=True, exist_ok=True)
#         model_out = f"/home/mads/projects/cubicsym/tests/outputs/{pdb_path.stem}"
#         Path(model_out).mkdir(parents=True, exist_ok=True)
#         sm = SymmetryMapper(tmp_file_dir=dumpfolder)
#         mer = int(str(pdb_path).split("_")[-1])
#         model_in = [p for p in Path(pdb_path).glob("*") if any(f"pred_{i}.pdb" in str(p) for i in range(1)) and str(p.name)[:len("relaxed")] == "relaxed"]
#         if True: # skip for now
#             sym_name, x_min, x_max = sm.run(models=model_in,
#                                             mer=mer, model_outdir=model_out, sym_outdir="/home/mads/projects/evodock/inputs/symmetry_files",
#                                             symname="3LEO_multi", main_chains_allowed=[["A"] for _ in model_in])
#         # look in the output folder and compare alignemtn rsmd with nonalignment rmsd
#         # check if there are multiple populations
#         # make a variant the other way
#
#
