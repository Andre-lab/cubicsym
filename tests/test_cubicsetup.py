#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the CubicSetup class
@Author: Mads Jeppesen
@Date: 9/21/22
"""
import pytest
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_str
import importlib

# def test_read_from_pose():
#     from simpletestlib.test import setup_test
#     from cubicsym.kinematics import randomize_all_dofs
#     from cubicsym.cubicsetup import CubicSetup
#     # first we need to write down the jumpgroups, energies, anchor type, reset
#     sym_files = {"I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"], # hands: {'1STM': True, '1B5S': True, '1NQW': False, '6S44': False, '5CVZ': True}
#                  "O": ["5GU1", "3R2R", "1BG7", "1AEW", "1P3Y"], # hands: {'5GU1': True, '3R2R': True, '1BG7': True, '1AEW': True, '1P3Y': False}
#                  "T": ["1MOG", "1H0S", "7JRH", "4KIJ", "2VTY"]} # hands: {'1MOG': False, '1H0S': True, '7JRH': True, '4KIJ': True, '2VTY': False}
#     for pdb in ("7Q03", "1H0S"):
#         pose, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
#         cs = CubicSetup(pose=pose)
#         for i in range(10):
#             randomize_all_dofs(pose)
#             cs.vrts_overlap_with_pose(pose, update_and_apply_dofs=True)


def test_transform_to_full_hf():
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from simpletestlib.setup import setup_test
    for pdbid, hand in {'1NQW': False,  '1B5S': True, '6S44': False, '1STM': True}.items():
        pose, pmm, cmd, symdef = setup_test(name="I", file=pdbid, mute=True, symmetrize=True, reinitialize=False,
                                              return_symmetry_file=True)
        cs = CubicSetup(symdef)
        pose = cs.make_asymmetric_pose(pose)
        cs_new = cs.add_extra_chains()
        cs_new.make_symmetric_pose(pose)
        pose.pdb_info().name("original")
        pmm.apply(pose)

        pose.pdb_info().name("left")
        set_jumpdof_str_str(pose, jump="JUMPHFfold1_z", dof="angle_z", value=72)
        pmm.apply(pose)
        cs_new.update_dofs_from_pose(pose)
        cs_new.visualize(ip="10.8.0.18")

        # pose.pdb_info().name("right")
        # set_jumpdof_str_str(pose, jump="JUMPHFfold1_z", dof="angle_z", value=-72)
        # pmm.apply(pose)

        print("hello")

def test_transform_to_full_2f():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    for pdbid, hand in {'1STM': True,  '1NQW': False, '6S44': False, '1B5S': True }.items():
        pose, pmm, cmd, symdef = setup_test(name="I", file=pdbid, mute=True, symmetrize=True, reinitialize=False,
                                            return_symmetry_file=True)
        cs = CubicSetup(symdef)
        asympose = cs.make_asymmetric_pose(pose)

        # ass = CubicSymmetricAssembly.from_pose_input(pose, cs)
        # ass.set_server_proxy("http://10.8.0.18:9123")
        # ass.set_server_root_path("/Users/mads/mounts/mailer")
        # ass.show()

        cs_hf_based = cs.add_extra_chains()
        pose = asympose.clone()
        cs_hf_based.make_symmetric_pose(pose)
        pose.pdb_info().name("h5_original")
        pmm.apply(pose)
        cs_2f_based = cs.create_I_2fold_based_symmetry()
        pose = asympose.clone()
        cs_2f_based.make_symmetric_pose(pose)

        # cs_2f_based.update_dofs_from_pose(pose)
        # cs_2f_based.visualize(ip="10.8.0.22")

        # pose = asympose.clone()
        # cs_3f_based.make_symmetric_pose(pose)
        # pose.pdb_info().name("3f_pose")
        # pmm.apply(pose)

        pose = asympose.clone()
        cs_2f_based_new = cs_2f_based.add_extra_chains()
        cs_2f_based_new.make_symmetric_pose(pose)
        pose.pdb_info().name("h2_original")
        pmm.apply(pose)

       # lets try to transfer?

        # cs_3f_based_new.update_dofs_from_pose(pose)
        # cs_3f_based_new.visualize(ip="10.8.0.22")

        print("hello")


def test_transform_to_full_f3():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    for pdbid, hand in {'1NQW': False,  '1B5S': True, '6S44': False, '1STM': True}.items():
        pose, pmm, cmd, symdef = setup_test(name="I", file=pdbid, mute=True, symmetrize=True, reinitialize=True,
                                            return_symmetry_file=True)
        cs = CubicSetup(symdef)
        asympose = cs.make_asymmetric_pose(pose)

        # ass = CubicSymmetricAssembly.from_pose_input(pose, cs)
        # ass.set_server_proxy("http://10.8.0.22:9123")
        # ass.set_server_root_path("/Users/mads/mounts/mailer")
        # ass.show()

        pose = asympose.clone()
        cs_hf_based = cs.add_extra_chains()
        cs_hf_based.make_symmetric_pose(pose)
        pose.pdb_info().name("h5_original")
        pmm.apply(pose)
        cs_3f_based = cs.create_I_3fold_based_symmetry()
        # pose = asympose.clone()
        # cs_3f_based.make_symmetric_pose(pose)
        # pose.pdb_info().name("3f_pose")
        # pmm.apply(pose)

        pose = asympose.clone()
        cs_3f_based_new = cs_3f_based.add_extra_chains()
        cs_3f_based_new.make_symmetric_pose(pose)

        pose.pdb_info().name("h3_original")
        pmm.apply(pose)

        # lets try to transfer?


        # cs_3f_based_new.update_dofs_from_pose(pose)
        # cs_3f_based_new.visualize(ip="10.8.0.22")

        print("hello")

def cut_monomeric_pose(pose, n_resi, c_resi):
    """Cut a pose with a single chain"""
    # this cuts including the first and last index
    if c_resi > 0:
        pose.delete_residue_range_slow(pose.size() - c_resi + 1, pose.size())
    if n_resi > 0:  # else keep the N termini
        pose.delete_residue_range_slow(1, n_resi)
    return pose

def test_rmsd_7q03():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from pyrosetta import pose_from_file
    from cubicsym.private_paths import SYMMETRICAL
    sym, pdbid, fold = "T", "7Q03", "2F"
    native, pmm, cmd = setup_test(name=sym, file=pdbid, mute=True, symmetrize=True, reinitialize=False)
    norm_cs = CubicSetup()
    norm_cs.load_norm_symdef(sym, fold=fold)
    f = "/home/mads/production/evodock/results/globalfrommultimer_rest/7Q03/relax_final/out"
    pose = pose_from_file(f"{f}/input_input_relaxed_model_1_multimer_v2_pred_17_A_2_multi_up.prepack36_63_35.pdb")
    symdef = f"{f}/input_relaxed_model_1_multimer_v2_pred_17_A_2_multi_up.prepack36_63_35.symm"
    cs = CubicSetup(symdef)
    same_handedness = norm_cs.righthanded == cs.righthanded
    cs.make_symmetric_pose(pose)
    pose_temp = pose.clone()
    # pymol
    # angle, rots = cs.get_register_shift_angle()
    # jump = f"JUMP{cs.get_jumpidentifier()}fold1_z"
    # for n in range(1, rots):
    #     perturb_jumpdof_str_str(pose_temp, jump, "angle_z", angle)
    pmm.apply(pose_temp)
    native.pdb_info().name("native")
    pmm.apply(native)
    # special map
    []

    # pymol
    native_crystallic = pose_from_file(str(SYMMETRICAL.joinpath(f"{sym}/idealized/crystal_repr/native/{pdbid}_crystal.pdb")))
    rmsd = norm_cs.rmsd_hf_map(pose, native_crystallic, same_handedness, interface=False)
    rmsd_other_map = norm_cs.rmsd_hf_map(pose, native_crystallic, same_handedness, interface=False, use_map = (4, 7, 1, 2, None, 3, None))
    print("IS", rmsd, rmsd_other_map)

def test_rmsd():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from pyrosetta import pose_from_file
    from cubicsym.private_paths import SYMMETRICAL
    import pandas as pd
    from random import randint
    selection = { "T": {
        "2CC9": "HF",
        "7Q03": "2F",
        "4DCL": "HF",
        "3LEO": "HF",
        "2QQY": "2F",
        "6M8V": "2F",
        "6HSB": "HF",
    }, "O": {
        "3WIS": "3F",
        "5H46": "2F",
        "5EKW": "3F",
        "3N1I": "HF",
        "6H05": "3F",
        "7O63": "2F",
        "7OHF": "2F",
    }, "I": {
        "1HQK": "HF",
        "1T0T": "HF",
        "1X36": "HF",
        "7B3Y": "3F",
        "4V4M": "HF",
        "1JH5": "3F",
        "6ZLO": "3F"
    }}
    visualize = False
    df = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/normalization_info_benchmark.csv")
    for sym, files in selection.items():
        for pdbid, fold in files.items():
            native, pmm, cmd, symdef = setup_test(name=sym, file=pdbid, mute=True, symmetrize=False, return_symmetry_file=True, reinitialize=False)
            n = randint(3, 10)
            c = randint(3, 10)
            cs = CubicSetup(symdef)
            norm_cs = CubicSetup()
            norm_cs.load_norm_symdef(sym, fold=fold)
            same_handedness = norm_cs.righthanded == cs.righthanded
            pose = native.clone()
            # lets try to cut pose
            cut_monomeric_pose(pose, n, c)
            cs.make_symmetric_pose(native)
            norm_cs.make_symmetric_pose(pose)
            for jump, dof, val in df[df["pdb_base"] == f"{pdbid}_{fold}"][["jump", "dof", "val"]].values:
                set_jumpdof_str_str(pose, jump, dof, val)
            rmsd_in_df = df[df["pdb_base"] == f"{pdbid}_{fold}"]["rmsd"].values[0]
            native_crystallic = pose_from_file(str(SYMMETRICAL.joinpath(f"{sym}/idealized/crystal_repr/native/{pdbid}_crystal.pdb")))
            rmsd = norm_cs.rmsd_hf_map(pose, native_crystallic, same_handedness, interface=False)
            rmsd_interface = norm_cs.rmsd_hf_map(pose, native_crystallic, same_handedness, interface=True)
            try:
                if visualize:
                    pose.pdb_info().name(f"pose_{pdbid}")
                    native.pdb_info().name(f"native_{pdbid}")
                    pmm.apply(pose)
                    pmm.apply(native)
                    mapp = norm_cs.construct_residue_map_any2hf(pose, native_crystallic, same_handedness=same_handedness, interface=True)
                    mapp = {k:v for k, v in mapp.items()}
                    resis = [pose.pdb_info().pose2pdb(i).split() for i in mapp.keys()]
                    resis_ref = [native_crystallic.pdb_info().pose2pdb(i).split() for i in mapp.values()]
                    cmd.do(f"color red, pose_{pdbid} AND ( {' OR '.join([f'(resi {ri} AND chain {ch})' for ri, ch in resis])} )")
                    cmd.do(f"color red, native_{pdbid} AND ( {' OR '.join([f'(resi {ri} AND chain {ch})' for ri, ch in resis_ref])} )")
                assert rmsd_interface <= rmsd_in_df + 0.3
                assert rmsd <= rmsd_in_df + 0.3
            except AssertionError:
                pose.pdb_info().name("pose")
                native.pdb_info().name("native")
                pmm.apply(pose)
                pmm.apply(native)
                norm_cs.construct_residue_map_any2hf(pose, native_crystallic, same_handedness=same_handedness, interface=True)
                raise AssertionError



def test_calculate_if_righthanded_from_pose():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.actors.symdefswapper import SymDefSwapper
    hands = {'1STM': True, '1B5S': True, '1NQW': False, '6S44': False, '5CVZ': True,
        '5GU1': True, '3R2R': True, '1BG7': True, '1AEW': True, '1P3Y': False,
        '1MOG': False, '1H0S': True, '7JRH': True, '4KIJ': True, '2VTY': False}
    sym_files = {"I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"],
                 "O": ["5GU1", "3R2R", "1BG7", "1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S", "7JRH", "4KIJ", "2VTY"]}
    for sym, files in sym_files.items():
        for file in files:
            pose_HF, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True, reinitialize=False)
            sds = SymDefSwapper(pose_HF, symdef)
            pose_3F, pose_2F = sds.create_3fold_pose_from_HFfold(pose_HF), sds.create_2fold_pose_from_HFfold(pose_HF)
            for pose, fold, in zip((pose_3F, pose_2F, pose_HF), ("3", "2", "HF")):
                assert hands[file] == CubicSetup.calculate_if_righthanded_from_pose(pose), f"{file} is {hands[file]}"
                print(file, fold, "OK!")
            #setup.visualize(ip="10.8.0.6", suffix=f"{file}")


def test_show_multiple_symmetries():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    sym_files = {"I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"], # hands: {'1STM': True, '1B5S': True, '1NQW': False, '6S44': False, '5CVZ': True}
                 "O": ["5GU1", "3R2R", "1BG7", "1AEW", "1P3Y"], # hands: {'5GU1': True, '3R2R': True, '1BG7': True, '1AEW': True, '1P3Y': False}
                 "T": ["1MOG", "1H0S", "7JRH", "4KIJ", "2VTY"]}
    for sym, files in sym_files.items():
        for file in files:
            pose, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True, reinitialize=False)
            pmm.keep_history(True)
            pose.pdb_info().name("org")
            pmm.apply(pose)
            setup = CubicSetup(symdef=symdef)
            setup.visualize(ip="10.8.0.6", suffix=f"{file}")

def test_handedness():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    setup = CubicSetup()
    sym_files = {"I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"], # hands: {'1STM': True, '1B5S': True, '1NQW': False, '6S44': False, '5CVZ': True}
                 "O": ["5GU1", "3R2R", "1BG7", "1AEW", "1P3Y"], # hands: {'5GU1': True, '3R2R': True, '1BG7': True, '1AEW': True, '1P3Y': False}
                 "T": ["1MOG", "1H0S", "7JRH", "4KIJ", "2VTY"]} # hands: {'1MOG': False, '1H0S': True, '7JRH': True, '4KIJ': True, '2VTY': False}
    answer = {}
    for sym, files in sym_files.items():
        for file in files:
            pose, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True, pymol=False)
            setup = CubicSetup()
            setup.read_from_file(symdef)
            answer[file] = setup.calculate_if_rightanded()
    print(answer)

def test_get_chains():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    sym_files = {"I": ["1STM", "6S44"],
                 "O": ["1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S"],
                 }
    for sym, files in sym_files.items():
        for file in files:
            pose, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True)
            pmm.keep_history(True)
            pose.pdb_info().name("org")
            pmm.apply(pose)
            setup = CubicSetup(symdef=symdef)
            pose.pdb_info().name("hf")
            pmm.apply(setup.get_HF_chains(pose))
            pose.pdb_info().name("3")
            pmm.apply(setup.get_3fold_chains(pose))
            pose.pdb_info().name("2_1")
            pmm.apply(setup.get_2fold_chains(pose)[0])
            pose.pdb_info().name("2_2")
            pmm.apply(setup.get_2fold_chains(pose)[1])
    assert True # I have checked that this is consistent!

def test_chain_mapping():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from cubicsym.utilities import pose_cas_are_identical, get_chain_map
    from pyrosetta import init
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from cubicsym.actors.symdefswapper import SymDefSwapper
    sym_files = {"O": ["1AEW", "1P3Y"],
                 "T": ["1H0S", "1MOG"],
                 "I": ["1STM", "6S44"]
                 }
    for sym, files in sym_files.items():
        for file, righthanded in zip(files, (True, False)):
            posehf, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True, symmetrize=True, reinitialize=False)
            sds = SymDefSwapper(posehf, symdef)
            posehf.pdb_info().name(f"hf_{file}")
            pose3f = sds.create_3fold_pose_from_HFfold(posehf)
            pose3f.pdb_info().name(f"3f_{file}")
            pose2f = sds.create_2fold_pose_from_HFfold(posehf)
            pose2f.pdb_info().name(f"2f_{file}")
            pmm.apply(posehf)
            pmm.apply(pose3f)
            pmm.apply(pose2f)
            assert pose_cas_are_identical(posehf, pose3f, pose2f, map_chains=get_chain_map(sym, righthanded), atol=1e-1), f"{sym} and {file} does not work!"


    assert True # I have checked that this is consistent!

def test_get_chain_names():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    sym_files = {"I": ["1STM", "6S44"],
                 "O": ["1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S"],
                 }
    results = {"I": {"1STM":{}, "6S44":{}},
                 "O": {"1AEW":{}, "1P3Y":{}},
                 "T": {"1MOG":{}, "1H0S":{}}}
    for sym, files in sym_files.items():
        for file in files:
            pose, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True)
            pmm.apply(pose)
            setup = CubicSetup(symdef=symdef)
            results[sym][file]["HF"] = setup.get_HF_chain_ids()
            results[sym][file]["3"] = setup.get_3fold_chain_ids()
            results[sym][file]["2"] = setup.get_2fold_chain_ids()
    assert results == {'I': {'1STM': {'HF': ('A', 'B', 'C', 'D', 'E'),
                           '3': ('A', 'I', 'F'),
                           '2': (('A', 'H'), ('A', 'G'))},
                          '6S44': {'HF': ('A', 'B', 'C', 'D', 'E'),
                           '3': ('A', 'I', 'F'),
                           '2': (('A', 'H'), ('A', 'G'))}},
                       'O': {'1AEW': {'HF': ('A', 'B', 'C', 'D'),
                           '3': ('A', 'E', 'H'),
                           '2': (('A', 'G'), ('A', 'F'))},
                          '1P3Y': {'HF': ('A', 'B', 'C', 'D'),
                           '3': ('A', 'E', 'H'),
                           '2': (('A', 'G'), ('A', 'F'))}},
                       'T': {'1MOG': {'HF': ('A', 'B', 'C'),
                           '3': ('A', 'D', 'G'),
                           '2': (('A', 'F'), ('A', 'E'))},
                          '1H0S': {'HF': ('A', 'B', 'C'),
                           '3': ('A', 'D', 'G'),
                           '2': (('A', 'F'), ('A', 'E'))}}}

def test_cubicsetup():
    from cubicsym.cubicsetup import CubicSetup
    CubicSetup()
    assert True

def set_all_dofs_to_zero(pose):
    from symmetryhandler.kinematics import set_jumpdof, get_position_info
    for jump, dofs in get_position_info(pose).items():
        for dof, old_val in dofs.items():
            set_jumpdof(pose, jump, dof_map[dof], 0)

#fixme: change the names to the new ones
@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_independent_icosahedral_symmetries():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.kinematics import perturb_jumpdof
    from pyrosetta import Pose
    from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit

    pose, pmm, cmd, symm_file = setup_test(name="I", file="1STM", return_symmetry_file=True)
    pmm.keep_history(True)
    setup = CubicSetup()
    setup.read_from_file(symm_file)

    perturb_jumpdof(pose, "JUMPHFfold1", 3, 10)
    perturb_jumpdof(pose, "JUMPHFfold1_z", 6, 20)
    perturb_jumpdof(pose, "JUMPHFfold111_x", 4, 10)
    perturb_jumpdof(pose, "JUMPHFfold111_y", 5, 20)
    perturb_jumpdof(pose, "JUMPHFfold111_z", 6, 30)
    pmm.apply(pose)

    fold5_setup, fold3_setup, fold2_1_setup, fold2_2_setup = setup.create_independent_icosahedral_symmetries(pose)

    pose_dofs_are_0 = pose.clone()
    set_all_dofs_to_zero(pose_dofs_are_0)
    asymmetric_pose = Pose()
    extract_asymmetric_unit(pose_dofs_are_0, asymmetric_pose, False)

    # pmm.apply(pose_non_sym); pmm.apply(asymmetric_pose) #ARE IDENTICAL before symmetry = GOOD!

    # fold5
    # fold5_setup.visualize(ip="10.8.0.10", port="9123")
    fold5_setup.make_symmetric_pose(asymmetric_pose)
    pmm.apply(asymmetric_pose)

    # fold3
    # pose, pmm, cmd = setup_test("4v4m", symmetry=False)

    # fold3_setup.visualize(ip="10.8.0.10", port="9123")
    asymmetric_pose = Pose()
    extract_asymmetric_unit(pose_dofs_are_0, asymmetric_pose, False)
    fold3_setup.make_symmetric_pose(asymmetric_pose)
    pmm.apply(asymmetric_pose)

    # fold2_1
    # pose, pmm, cmd = setup_test("4v4m", symmetry=False)
    # fold2_1_setup.visualize(ip="10.8.0.10", port="9123")
    asymmetric_pose = Pose()
    extract_asymmetric_unit(pose_dofs_are_0, asymmetric_pose, False)
    fold2_1_setup.make_symmetric_pose(asymmetric_pose)
    pmm.apply(asymmetric_pose)

    # fold2_2
    # pose, pmm, cmd = setup_test("4v4m", symmetry=False)
    # fold2_2_setup.visualize(ip="10.8.0.10", port="9123")
    asymmetric_pose = Pose()
    extract_asymmetric_unit(pose_dofs_are_0, asymmetric_pose, False)
    fold2_2_setup.make_symmetric_pose(asymmetric_pose)
    pmm.apply(asymmetric_pose)

def test_new_symmetry_energy_is_the_same():
    from shapedesign.src.utilities.score import create_score_from_name
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from simpletestlib.setup import setup_test
    import numpy as np
    old_sym = "/tmp/old_symm"
    with open(old_sym, "w") as f:
        f.write("""symmetry_name /home/shared/databases/SYMMETRICAL/I/unrelaxed/native/../../idealized/symdef/native/6S44.symm
E = 60*VRTHFfold1111 + 60*(VRTHFfold1111:VRTHFfold1211) + 60*(VRTHFfold1111:VRTHFfold1311) + 60*(VRTHFfold1111:VRT3fold1111) + 30*(VRTHFfold1111:VRT2fold1111) + 30*(VRTHFfold1111:VRT3fold1211)
anchor_residue COM
virtual_coordinates_start
xyz VRTglobal 1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold11 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold111 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1111 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold12 -0.309017,0.951057,0.000000 0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold121 -0.309017,0.951057,0.000000 0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1211 -0.309017,0.951057,0.000000 0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold13 0.809017,0.587785,0.000000 0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold131 0.809017,0.587785,0.000000 0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1311 0.809017,0.587785,0.000000 0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold14 0.809017,-0.587785,0.000000 -0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold141 0.809017,-0.587785,0.000000 -0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1411 0.809017,-0.587785,0.000000 -0.587785,-0.809017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold15 -0.309017,-0.951057,0.000000 -0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold151 -0.309017,-0.951057,0.000000 -0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRTHFfold1511 -0.309017,-0.951057,0.000000 -0.951057,0.309017,0.000000 0.000000,0.000000,0.000000
xyz VRT3fold -0.859176,-0.240861,0.451444 0.240861,0.588037,0.772139 0.000000,0.000000,0.000000
xyz VRT3fold1 -0.859176,-0.240861,0.451444 0.240861,0.588037,0.772139 0.000000,0.000000,0.000000
xyz VRT3fold11 0.036427,-0.484827,-0.873851 -0.891555,-0.410786,0.190745 0.000000,0.000000,0.000000
xyz VRT3fold111 0.036427,-0.484827,-0.873851 -0.891555,-0.410786,0.190745 0.000000,0.000000,0.000000
xyz VRT3fold1111 0.036427,-0.484827,-0.873851 -0.891555,-0.410786,0.190745 0.000000,0.000000,0.000000
xyz VRT3fold12 0.859176,0.240861,-0.451444 -0.240861,-0.588037,-0.772139 0.000000,0.000000,0.000000
xyz VRT3fold121 0.859176,0.240861,-0.451444 -0.240861,-0.588037,-0.772139 0.000000,0.000000,0.000000
xyz VRT3fold1211 0.859176,0.240861,-0.451444 -0.240861,-0.588037,-0.772139 0.000000,0.000000,0.000000
xyz VRT2fold -0.472349,0.115163,0.873856 -0.115163,0.974865,-0.190724 0.000000,0.000000,0.000000
xyz VRT2fold1 -0.472349,0.115163,0.873856 -0.115163,0.974865,-0.190724 0.000000,0.000000,0.000000
xyz VRT2fold11 0.472349,-0.115163,-0.873856 0.115163,-0.974865,0.190724 0.000000,0.000000,0.000000
xyz VRT2fold111 0.472349,-0.115163,-0.873856 0.115163,-0.974865,0.190724 0.000000,0.000000,0.000000
xyz VRT2fold1111 0.472349,-0.115163,-0.873856 0.115163,-0.974865,0.190724 0.000000,0.000000,0.000000
xyz VRT2fold12 0.036437,0.891564,-0.451426 0.484818,-0.410776,-0.772149 0.000000,0.000000,0.000000
xyz VRT2fold121 0.036437,0.891564,-0.451426 0.484818,-0.410776,-0.772149 0.000000,0.000000,0.000000
xyz VRT2fold1211 0.036437,0.891564,-0.451426 0.484818,-0.410776,-0.772149 0.000000,0.000000,0.000000
virtual_coordinates_stop
connect_virtual JUMPHFfold VRTglobal VRTHFfold
connect_virtual JUMPHFfold1 VRTHFfold VRTHFfold1
connect_virtual JUMPHFfold11 VRTHFfold1 VRTHFfold11
connect_virtual JUMPHFfold111 VRTHFfold11 VRTHFfold111
connect_virtual JUMPHFfold1111 VRTHFfold111 VRTHFfold1111
connect_virtual JUMPHFfold1111_subunit VRTHFfold1111 SUBUNIT
connect_virtual JUMPHFfold12 VRTHFfold1 VRTHFfold12
connect_virtual JUMPHFfold121 VRTHFfold12 VRTHFfold121
connect_virtual JUMPHFfold1211 VRTHFfold121 VRTHFfold1211
connect_virtual JUMPHFfold1211_subunit VRTHFfold1211 SUBUNIT
connect_virtual JUMPHFfold13 VRTHFfold1 VRTHFfold13
connect_virtual JUMPHFfold131 VRTHFfold13 VRTHFfold131
connect_virtual JUMPHFfold1311 VRTHFfold131 VRTHFfold1311
connect_virtual JUMPHFfold1311_subunit VRTHFfold1311 SUBUNIT
connect_virtual JUMPHFfold14 VRTHFfold1 VRTHFfold14
connect_virtual JUMPHFfold141 VRTHFfold14 VRTHFfold141
connect_virtual JUMPHFfold1411 VRTHFfold141 VRTHFfold1411
connect_virtual JUMPHFfold1411_subunit VRTHFfold1411 SUBUNIT
connect_virtual JUMPHFfold15 VRTHFfold1 VRTHFfold15
connect_virtual JUMPHFfold151 VRTHFfold15 VRTHFfold151
connect_virtual JUMPHFfold1511 VRTHFfold151 VRTHFfold1511
connect_virtual JUMPHFfold1511_subunit VRTHFfold1511 SUBUNIT
connect_virtual JUMP3fold VRTglobal VRT3fold
connect_virtual JUMP3fold1 VRT3fold VRT3fold1
connect_virtual JUMP3fold11 VRT3fold1 VRT3fold11
connect_virtual JUMP3fold111 VRT3fold11 VRT3fold111
connect_virtual JUMP3fold1111 VRT3fold111 VRT3fold1111
connect_virtual JUMP3fold1111_subunit VRT3fold1111 SUBUNIT
connect_virtual JUMP3fold12 VRT3fold1 VRT3fold12
connect_virtual JUMP3fold121 VRT3fold12 VRT3fold121
connect_virtual JUMP3fold1211 VRT3fold121 VRT3fold1211
connect_virtual JUMP3fold1211_subunit VRT3fold1211 SUBUNIT
connect_virtual JUMP2fold VRTglobal VRT2fold
connect_virtual JUMP2fold1 VRT2fold VRT2fold1
connect_virtual JUMP2fold11 VRT2fold1 VRT2fold11
connect_virtual JUMP2fold111 VRT2fold11 VRT2fold111
connect_virtual JUMP2fold1111 VRT2fold111 VRT2fold1111
connect_virtual JUMP2fold1111_subunit VRT2fold1111 SUBUNIT
connect_virtual JUMP2fold12 VRT2fold1 VRT2fold12
connect_virtual JUMP2fold121 VRT2fold12 VRT2fold121
connect_virtual JUMP2fold1211 VRT2fold121 VRT2fold1211
connect_virtual JUMP2fold1211_subunit VRT2fold1211 SUBUNIT
set_dof JUMPHFfold1 z(67.88562749342442) angle_z(0)
set_dof JUMPHFfold111 x(24.256740140250045)
set_dof JUMPHFfold1111 angle_x(0) angle_y(0) angle_z(0)
set_dof JUMPHFfold1111_subunit angle_x(0) angle_y(0) angle_z(0)
set_jump_group JUMPGROUP1 JUMPHFfold1 JUMP3fold1 JUMP2fold1
set_jump_group JUMPGROUP2 JUMPHFfold111 JUMPHFfold121 JUMPHFfold131 JUMPHFfold141 JUMPHFfold151 JUMP3fold111 JUMP3fold121 JUMP2fold111 JUMP2fold121
set_jump_group JUMPGROUP3 JUMPHFfold1111 JUMPHFfold1211 JUMPHFfold1311 JUMPHFfold1411 JUMPHFfold1511 JUMP3fold1111 JUMP3fold1211 JUMP2fold1111 JUMP2fold1211
set_jump_group JUMPGROUP4 JUMPHFfold1111_subunit JUMPHFfold1211_subunit JUMPHFfold1311_subunit JUMPHFfold1411_subunit JUMPHFfold1511_subunit JUMP3fold1111_subunit JUMP3fold1211_subunit JUMP2fold1111_subunit JUMP2fold1211_subunit""")
    pose, pmm, cmd  = setup_test(name="I", file="6S44", symmetrize=False)
    sfxn = create_score_from_name("ref2015")
    pose_old = pose.clone()
    SetupForSymmetryMover(old_sym).apply(pose_old)
    pose_new = pose.clone()
    new_sym = "/home/mads/projects/cubicsym/tests/outputs/6S44.symm"
    SetupForSymmetryMover(new_sym).apply(pose_new)
    pmm.keep_history(True)
    pmm.apply(pose_new)
    pmm.apply(pose_old)
    # they are slightly different from 1000 of decimal so it is okay to vary slightly
    assert np.isclose(sfxn(pose_old), sfxn(pose_new), atol=1)

# fixme or delete
def test_rotations_to_2folds():
    pose, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    setup = SymmetrySetup()
    setup.read_from_file(symm_file)
    setup.visualize(ip="10.8.0.10", port="9123")
    pmm.apply(pose)
    start = time.time()
    # a, b = setup.rotations_to_2folds(pose, visualize=True, cmd=cmd)
    a, b = setup.icosahedral_angle_z(pose)
    print(time.time() - start)
    pass

@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_O_3fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    pose, pmm, cmd, symm_file = setup_test(name="O", file="7NTN", symmetrize=True, return_symmetry_file=True)
    setup = CubicSetup()
    setup.read_from_file(symm_file)
    # setup.visualize(ip="10.8.0.14", port="9123")
    ss3 = setup.create_O_3fold_based_symmetry()
    ss3.visualize(ip=get_test_options("pymol").get("ip"))  # , port="9123")
    pmm.apply(pose)
    # apply the symmetry:
    asympose = setup_test(name="O", file="7NTN", symmetrize=False, return_symmetry_file=False, pymol=False)
    ss3.make_symmetric_pose(asympose)
    asympose.pdb_info().name("3foldbased")
    pmm.apply(asympose)


@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_3fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    pose, pmm, cmd, symm_file = setup_test(name="I", file="1STM", symmetrize=True, return_symmetry_file=True)
    setup = CubicSetup()
    setup.read_from_file(symm_file)
    # setup.visualize(ip="10.8.0.14", port="9123")
    ss3 = setup.create_I_3fold_based_symmetry()
    ss3.visualize(ip=get_test_options("pymol").get("ip"))  # , port="9123")
    pmm.apply(pose)
    # apply the symmetry:
    asympose = setup_test(name="I", file="1STM", symmetrize=False, return_symmetry_file=False, pymol=False)
    ss3.make_symmetric_pose(asympose)
    asympose.pdb_info().name("3foldbased")
    pmm.apply(asympose)

@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_O_2fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    for file in ("7NTN",):
        pose, pmm, cmd, symm_file = setup_test(name="O", file=file, symmetrize=True, return_symmetry_file=True)
        setup = CubicSetup()
        setup.read_from_file(symm_file)
        pmm.apply(pose)
        # setup.visualize(ip="10.8.0.6", port="9123")
        ss2 = setup.create_O_2fold_based_symmetry()
        ss2.visualize(ip=get_test_options("pymol").get("ip"))  # , port="9123")
        pmm.apply(pose)
        # apply the symmetry:
        asympose = setup_test(name="O", file=file, symmetrize=False, return_symmetry_file=False, pymol=False)
        ss2.make_symmetric_pose(asympose)
        asympose.pdb_info().name("2foldbased")
        pmm.apply(asympose)

@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_T_3fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    for file in ("1MOG", "4CIY"):
        pose, pmm, cmd, symm_file = setup_test(name="T", file=file, symmetrize=True, return_symmetry_file=True)
        setup = CubicSetup()
        setup.read_from_file(symm_file)
        pmm.apply(pose)
        # setup.visualize(ip="10.8.0.6", port="9123")
        ss3 = setup.create_T_3fold_based_symmetry()
        ss3.visualize(ip=get_test_options("pymol").get("ip"))  # , port="9123")
        pmm.apply(pose)
        # apply the symmetry:
        asympose = setup_test(name="T", file=file, symmetrize=False, return_symmetry_file=False, pymol=False)
        ss3.make_symmetric_pose(asympose)
        asympose.pdb_info().name("3foldbased")
        pmm.apply(asympose)

@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_T_2fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    for file in ("4CIY", "1MOG"):
        pose, pmm, cmd, symm_file = setup_test(name="T", file=file, symmetrize=True, return_symmetry_file=True)
        setup = CubicSetup()
        setup.read_from_file(symm_file)
        pmm.apply(pose)
        # setup.visualize(ip="10.8.0.6", port="9123")
        ss2 = setup.create_T_2fold_based_symmetry()
        ss2.visualize(ip=get_test_options("pymol").get("ip"))  # , port="9123")
        pmm.apply(pose)
        # apply the symmetry:
        asympose = setup_test(name="T", file=file, symmetrize=False, return_symmetry_file=False, pymol=False)
        ss2.make_symmetric_pose(asympose)
        asympose.pdb_info().name("2foldbased")
        pmm.apply(asympose)


@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_create_2fold_based_symmetry():
    from simpletestlib.setup import setup_test, get_test_options
    from cubicsym.cubicsetup import CubicSetup
    for file in ("6S44", "1STM"):
        pose, pmm, cmd, symm_file = setup_test(name="I", file=file, symmetrize=True, return_symmetry_file=True)
        setup = CubicSetup()
        setup.read_from_file(symm_file)
        pmm.apply(pose)
        # setup.visualize(ip="10.8.0.6", port="9123")
        ss2 = setup.create_I_2fold_based_symmetry()
        ss2.visualize(ip=get_test_options("pymol").get("ip"))#, port="9123")
        pmm.apply(pose)
        # apply the symmetry:
        asympose = setup_test(name="I", file=file, symmetrize=False, return_symmetry_file=False, pymol=False)
        ss2.make_symmetric_pose(asympose)
        asympose.pdb_info().name("2foldbased")
        pmm.apply(asympose)

def test_vrts_overlap_with_pose():
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.cubicsetup import CubicSetup
    #for pdb in ("1STM", "6SS4"):
    #for pdb in ("1AEW", "1P3Y"):
    for pdb in ("7Q03", "1H0S"):
        pose, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        cs = CubicSetup(symdef)
        for i in range(10):
            randomize_all_dofs(pose)
            cs.vrts_overlap_with_pose(pose, update_and_apply_dofs=True)

            # pmm.apply(pose)
            # cs.visualize(ip="10.8.0.10")

def test_apply_dofs():
    """Tests that the dofs follow the pose"""
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.cubicsetup import CubicSetup
    #for pdb in ("1STM", "6SS4"):
    #for pdb in ("1AEW", "1P3Y"):
    for pdb in ("7Q03", "1H0S"):
        pose, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        cs = CubicSetup(symdef)
        for i in range(10):
            randomize_all_dofs(pose)
            cs.vrts_overlap_with_pose(pose, update_and_apply_dofs=True)
            # cs.update_dofs_from_pose(pose)
            # pmm.apply(pose)
            # cs.visualize(ip="10.8.0.6")
