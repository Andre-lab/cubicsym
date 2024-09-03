#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the SymDefSwapper class
@Author: Mads Jeppesen
@Date: 7/14/22
"""
import math

def test_all_fully_procteced():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.kinematics import randomize_all_dofs
    sym = "I"
    pdbids = {'1STM': True, '1NQW': False,  '1B5S': True, '6S44': False,}
    for pdb in pdbids.keys():
        print(f"Doing {pdb}")
        pose_mono, pmm, cmd, symdef = setup_test(name=sym, file=pdb, return_symmetry_file=True, mute=True, symmetrize=False)
        pmm.keep_history(True)
        cs = CubicSetup(symdef)
        cs_hf = cs.add_extra_chains()
        poseHF = pose_mono.clone()
        cs_hf.make_symmetric_pose(poseHF)
        symdef = cs_hf.make_symmetry_file_on_tmp_disk()
        sds = SymDefSwapper(poseHF, symdef, debug_mode=True)
        change_alot_and_transfer(poseHF, sds, pmm)


def test_all_normalized():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.kinematics import randomize_all_dofs
    sym_files = {"T": ["1MOG", "1H0S"],
                 "I": ["1STM", "6S44"],
                 "O": ["1AEW", "1P3Y"]}
    for sym, files in sym_files.items():
        for pdb in files:
            pose_mono, pmm, cmd, symdef = setup_test(name=sym, file=pdb, return_symmetry_file=True, mute=True, symmetrize=False)
            pmm.keep_history(True)

            # create HF, 3 and 2-fold independently
            cs_hf = CubicSetup()
            cs_hf.load_norm_symdef(sym, "HF")
            pose_HF = pose_mono.clone()
            cs_hf.make_symmetric_pose(pose_HF)
            randomize_all_dofs(pose_HF)
            # -----
            cs_2f = CubicSetup()
            cs_2f.load_norm_symdef(sym, "2F")
            pose_2F = pose_mono.clone()
            cs_2f.make_symmetric_pose(pose_2F)
            randomize_all_dofs(pose_2F)
            # -----
            cs_3f = CubicSetup()
            cs_3f.load_norm_symdef(sym, "3F")
            pose_3F = pose_mono.clone()
            cs_3f.make_symmetric_pose(pose_3F)
            randomize_all_dofs(pose_3F)

            for pose, cs in zip([pose_2F, pose_3F, pose_HF], [cs_2f, cs_3f, cs_hf]):
                sds = SymDefSwapper(pose, cs.make_symmetry_file_on_tmp_disk(), debug_mode=True)
                change_alot_and_transfer(pose, sds, pmm)

def test_all_symm():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    sym_files = {"I": ["1STM", "6S44"],
                 "O": ["1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S"],
                 }
    for sym, files in sym_files.items():
        for pdb in files:
            poseHF, pmm, cmd, symdef = setup_test(name=sym, file=pdb, return_symmetry_file=True, mute=True)
            pmm.keep_history(True)
            sds = SymDefSwapper(poseHF, symdef, debug_mode=True)
            change_alot_and_transfer(poseHF, sds, pmm)

def test_T_sym():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1MOG", "1H0S"):
        poseHF, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef, debug_mode=True)
        change_alot_and_transfer(poseHF, sds, pmm)

def test_O_sym():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1AEW", "1P3Y"):
        poseHF, pmm, cmd, symdef = setup_test(name="O", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef, debug_mode=True)
        poseHF.pdb_info().name("HF")
        change_alot_and_transfer(poseHF, sds, pmm)

def test_I_sym():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1STM", "6S44"):
        poseHF, pmm, cmd, symdef = setup_test(name="I", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef, debug_mode=True)
        poseHF.pdb_info().name("HF")
        change_alot_and_transfer(poseHF, sds, pmm)

def change_alot_and_transfer(pose_org_X, sds, pmm, behavior="randomize", atol=2, show_correct_structure=False):
    from cubicsym.kinematics import randomize_all_dofs, randomize_all_dofs_positive_trans
    from cubicsym.utilities import get_chain_map
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.utilities import pose_cas_are_identical
    from symmetryhandler.reference_kinematics import get_dofs
    def check_overlap(poseHF, pose3, pose2):
        sds.foldHF_setup.vrts_overlap_with_pose(poseHF)
        sds.fold3F_setup.vrts_overlap_with_pose(pose3)
        sds.fold2F_setup.vrts_overlap_with_pose(pose2)
    for i in range(12):
        if i % 3 == 0:
            pose_X = pose_org_X.clone()
            poseHF, pose3, pose2 = sds.create_remaing_folds(pose_X)
        if behavior == "randomize":
            randomize_all_dofs(pose_X)
        elif behavior == "positive_trans":
            randomize_all_dofs_positive_trans(pose_X)
        else:
            raise ValueError
        dofs_pre = get_dofs(poseHF)
        check_overlap(poseHF, pose3, pose2)
        sds.transfer_poseA2B(poseHF, pose3)
        check_overlap(poseHF, pose3, pose2)
        sds.transfer_poseA2B(pose3, pose2)
        check_overlap(poseHF, pose3, pose2)
        sds.transfer_poseA2B(pose2, poseHF)
        check_overlap(poseHF, pose3, pose2)
        dof_post = get_dofs(poseHF)
        for jump, params in dofs_pre.items():
            dof = list(dofs_pre[jump].keys())[0] # {}
            val_post = dof_post[jump][dof]
            val_pre = dofs_pre[jump][dof]
        try:
            assert math.isclose(val_post, val_pre, abs_tol=1e-3), f"{val_post} != {val_pre} of {jump}:{dof}"
        except AssertionError:
            print(f"{val_post} != {val_pre} of {jump}:{dof}")
            # raise AssertionError


        # check the dofs are the same

        poseHF.pdb_info().name("HF")
        pose3.pdb_info().name("3")
        pose2.pdb_info().name("2")
        print(f"{pose_org_X.pdb_info().name()} SUCCEDED {i}/12")
        try:
            assert pose_cas_are_identical(poseHF, pose3, pose2, map_chains=get_chain_map(CubicSetup.cubic_symmetry_from_pose(poseHF), sds.foldHF_setup.righthanded), atol=atol)
        except AssertionError:
            pmm.apply(poseHF)
            pmm.apply(pose3)
            pmm.apply(pose2)
            raise AssertionError
        else:
            if show_correct_structure:
                pmm.apply(poseHF)
                pmm.apply(pose3)
                pmm.apply(pose2)
                ...

def test_different_backbones():
    from simpletestlib.setup import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    import random
    from pathlib import Path
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover
    pose5, pmm, cmd, symm_file = setup_test(name="I", file="1STM", return_symmetry_file=True, mute=True)
    pmm.keep_history(True)
    sds = SymDefSwapper(pose5, symm_file, debug_mode=False)

    ensembles = list(Path("/home/mads/projects/evodock/benchmark/symmetries/icosahedral/1STM/subunits").glob("*"))
    anchors = []
    for i in range(10):
        pose5 = pose_from_file(str(random.choice(ensembles)))
        # remove a random set of residues
        start = random.randint(1, pose5.size())
        end = min(start + 10, pose5.size())
        DeleteRegionMover(start, end).apply(pose5)
        clone = pose5.clone()
        clone.pdb_info().name("current_mono")
        pmm.apply(clone)
        SetupForSymmetryMover(symm_file).apply(pose5)
        change_alot_and_transfer(pose5, sds, pmm)
        anchors.append(sds.foldHF_setup.get_anchor_residue(pose5))
        print("anchors", anchors)

# todo: is this still important?
def test_energy_is_the_same():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from shapedesign.src.utilities.score import create_score_from_name
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True, pymol=True)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)
    score = create_score_from_name("ref2015")
    e5 = score.score(pose5)
    e3 = score.score(pose3)
    e2 = score.score(pose2)

