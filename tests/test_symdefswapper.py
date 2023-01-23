#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the SymDefSwapper class
@Author: Mads Jeppesen
@Date: 7/14/22
"""
import time

def test_w_new_sds_vrt():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
    import random

    symdef = "/home/mads/projects/cubicsym/tests/outputs/3LEO.symm"
    poseHF, pmm, cmd = setup_test(name="T", file="3LEO", return_symmetry_file=False, mute=True,
                                  use_symdef_instead=symdef)
    pmm.keep_history(True)
    sds = SymDefSwapper(poseHF, symdef)
    poseHF.pdb_info().name("HF")

    def change_alot_and_transfer(poseHF, sds, atol=1e-1):
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1", 3, random.uniform(-1, 1))
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1_z", 6, random.uniform(-2, 20))
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold111", 1, random.uniform(-1, 10))
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold111_x", 4, random.uniform(-30, 30))
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold111_y", 5, random.uniform(-30, 30))
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold111_z", 6, random.uniform(-30, 30))
        pose3 = sds.create_3fold_pose_from_HFfold(poseHF, transfer=True)
        pose2 = sds.create_2fold_pose_from_HFfold(poseHF, transfer=True)
        sds.transfer_HFto3(poseHF, pose3)
        sds.transfer_3to2(pose3, pose2)
        sds.transfer_2toHF(pose2, poseHF)
        poseHF.pdb_info().name("HF")
        pose3.pdb_info().name("3")
        pose2.pdb_info().name("2")
        pmm.apply(poseHF)
        pmm.apply(pose3)
        pmm.apply(pose2)
        # chain_map = [(1, 1, 1), (2, 4, 7), (3, 8, 6), (4, 9, 5), (5, 6, 3), (6, 2, 4), (7, 7, 8), (8, 5, 2), (9, 3, 9)]
        # try:
        #     assert pose_cas_are_identical(poseHF, pose3, pose2, map_chains=chain_map, atol=atol)
        # except AssertionError:
        #     raise AssertionError

    for i in range(10):
        change_alot_and_transfer(poseHF,  sds)

def test_start_with_non_HF():
    from simpletestlib.test import setup_test, setup_pymol
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, perturb_jumpdof_str_str, set_jumpdof_str_str
    import random
    pmm = setup_pymol()
    import math
    fhf = {'JUMPHFfold1': [['z', 'translation', 20.14548822689412]],
           'JUMPHFfold1_z': [['z', 'rotation', 0.0]],
           'JUMPHFfold111': [['x', 'translation', 14.46937894105028]],
           'JUMPHFfold111_x': [['x', 'rotation', 0.0]],
           'JUMPHFfold111_y': [['y', 'rotation', 0.0]],
           'JUMPHFfold111_z': [['z', 'rotation', 0.0]]}
    f3 = {'JUMP31fold1': [['z', 'translation', 20.357023099436248]],
          'JUMP31fold1_z': [['z', 'rotation', 0]],
          'JUMP31fold111': [['x', 'translation', 14.17022170037703]],
          'JUMP31fold111_x': [['x', 'rotation', 0]],
          'JUMP31fold111_y': [['y', 'rotation', 0]],
          'JUMP31fold111_z': [['z', 'rotation', 0]]}
    f2 = {'JUMP21fold1': [['z', 'translation', 17.538101678972165]],
          'JUMP21fold1_z': [['z', 'rotation', 0]],
          'JUMP21fold111': [['x', 'translation', 17.539064146909325]],
          'JUMP21fold111_x': [['x', 'rotation', 0]],
          'JUMP21fold111_y': [['y', 'rotation', 0]],
          'JUMP21fold111_z': [['z', 'rotation', 0]]}
    for pdb in ("2CC9", "7Q03", "1H0S"):
        pose_mono = setup_test(name="T", file=pdb, return_symmetry_file=False, mute=True, pymol=False, symmetrize=False)
        # make HF
        cs = CubicSetup()
        cs.load_norm_symdef("T", "HF")
        pose_HF = pose_mono.clone()
        cs.make_symmetric_pose(pose_HF)
        # make the other folds
        for fold in ("2F",):
            pmm.keep_history(True)
            cs = CubicSetup()
            cs.load_norm_symdef("T", fold)
            if fold == "2F":
                cs._dofs = f2
            cs.output("/tmp/symdef.symm")
            pose_X = pose_mono.clone()
            cs.make_symmetric_pose(pose_X)
            sds = SymDefSwapper(pose_X, "/tmp/symdef.symm")
            pose_HF.pdb_info().name("HF")
            pose_X.pdb_info().name(f"{fold}")
            if fold == "2F":
                pose_HF_X = sds.make_HF_pose_from_2F(pose_X)
                rmsd = cs.CA_rmsd_hf_map(pose_X, pose_HF_X)
                assert math.isclose(rmsd, 0, abs_tol=1.e-1)
            pmm.apply(pose_HF)
            pmm.apply(pose_X)

def test_works_with_hf_start():
    from simpletestlib.test import setup_test, setup_pymol
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, perturb_jumpdof_str_str, set_jumpdof_str_str
    import random
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import get_dofs
    pmm = setup_pymol()
    pmm.keep_history(True)
    import math
    fhf = {'JUMPHFfold1': [['z', 'translation', 20.14548822689412]],
           'JUMPHFfold1_z': [['z', 'rotation', 0.0]],
           'JUMPHFfold111': [['x', 'translation', 14.46937894105028]],
           'JUMPHFfold111_x': [['x', 'rotation', 0.0]],
           'JUMPHFfold111_y': [['y', 'rotation', 0.0]],
           'JUMPHFfold111_z': [['z', 'rotation', 0.0]]}
    f3 = {'JUMP31fold1': [['z', 'translation', 20.357023099436248]],
          'JUMP31fold1_z': [['z', 'rotation', 0]],
          'JUMP31fold111': [['x', 'translation', 14.17022170037703]],
          'JUMP31fold111_x': [['x', 'rotation', 0]],
          'JUMP31fold111_y': [['y', 'rotation', 0]],
          'JUMP31fold111_z': [['z', 'rotation', 0]]}
    f2 = {'JUMP21fold1': [['z', 'translation', 17.538101678972165]],
          'JUMP21fold1_z': [['z', 'rotation', 0]],
          'JUMP21fold111': [['x', 'translation', 17.539064146909325]],
          'JUMP21fold111_x': [['x', 'rotation', 0]],
          'JUMP21fold111_y': [['y', 'rotation', 0]],
          'JUMP21fold111_z': [['z', 'rotation', 0]]}
    for pdb in ("2CC9", "7Q03", "1H0S"):
        pose_mono = setup_test(name="T", file=pdb, return_symmetry_file=False, mute=True, pymol=False, symmetrize=False)

        cs_hf = CubicSetup()
        cs_hf.load_norm_symdef("T", "HF")
        cs_hf._dofs = fhf
        pose_HF = pose_mono.clone()
        cs_hf.make_symmetric_pose(pose_HF)
        sds = SymDefSwapper(pose_HF, cs_hf.get_norm_symdef_path("T", "HF"))

        cs_2f = CubicSetup()
        cs_2f.load_norm_symdef("T", "2F")
        cs_2f._dofs = f2
        pose_2F = pose_mono.clone()
        cs_2f.make_symmetric_pose(pose_2F)
        randomize_all_dofs(pose_2F)


        cs_3f = CubicSetup()
        cs_3f.load_norm_symdef("T", "3F")
        cs_3f._dofs = f3
        pose_3F = pose_mono.clone()
        cs_3f.make_symmetric_pose(pose_3F)
        randomize_all_dofs(pose_3F)

        pose_HF_from_3F = sds.make_HF_pose_from_3F(pose_3F)
        pose_HF_from_2F = sds.make_HF_pose_from_2F(pose_2F)
        pose_3F.pdb_info().name(f"pose_3F")
        pose_2F.pdb_info().name(f"pose_2F")
        pose_HF_from_3F.pdb_info().name(f"pose_HF_from_3F")
        pose_HF_from_2F.pdb_info().name(f"pose_HF_from_2F")
        pmm.apply(pose_HF_from_3F)
        pmm.apply(pose_HF_from_2F)
        pmm.apply(pose_3F)
        pmm.apply(pose_2F)
        ...

def test_start_from_3F_and_2F():
    from simpletestlib.test import setup_test, setup_pymol
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, perturb_jumpdof_str_str, set_jumpdof_str_str
    import random
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.reference_kinematics import get_dofs
    pmm = setup_pymol()
    pmm.keep_history(True)
    import math
    fhf = {'JUMPHFfold1': [['z', 'translation', 20.14548822689412]],
           'JUMPHFfold1_z': [['z', 'rotation', 0.0]],
           'JUMPHFfold111': [['x', 'translation', 14.46937894105028]],
           'JUMPHFfold111_x': [['x', 'rotation', 0.0]],
           'JUMPHFfold111_y': [['y', 'rotation', 0.0]],
           'JUMPHFfold111_z': [['z', 'rotation', 0.0]]}
    f3 = {'JUMP31fold1': [['z', 'translation', 20.357023099436248]],
          'JUMP31fold1_z': [['z', 'rotation', 0]],
          'JUMP31fold111': [['x', 'translation', 14.17022170037703]],
          'JUMP31fold111_x': [['x', 'rotation', 0]],
          'JUMP31fold111_y': [['y', 'rotation', 0]],
          'JUMP31fold111_z': [['z', 'rotation', 0]]}
    f2 = {'JUMP21fold1': [['z', 'translation', 17.538101678972165]],
          'JUMP21fold1_z': [['z', 'rotation', 0]],
          'JUMP21fold111': [['x', 'translation', 17.539064146909325]],
          'JUMP21fold111_x': [['x', 'rotation', 0]],
          'JUMP21fold111_y': [['y', 'rotation', 0]],
          'JUMP21fold111_z': [['z', 'rotation', 0]]}
    for pdb in ("2CC9", "7Q03", "1H0S"):
        pose_mono = setup_test(name="T", file=pdb, return_symmetry_file=False, mute=True, pymol=False, symmetrize=False)

        # cs_hf = CubicSetup()
        # cs_hf.load_norm_symdef("T", "HF")
        # # cs_hf._dofs = fhf
        # pose_HF_real = pose_mono.clone()
        # cs_hf.make_symmetric_pose(pose_HF_real)
        # pose_HF_real_asym = cs_hf.make_asymmetric_pose(pose_HF_real)

        # cs = CubicSetup()
        # cs.load_norm_symdef("T", "3F")
        # cs._dofs = f3

        # cs_2f = CubicSetup()
        # cs_2f.load_norm_symdef("T", "2F")
        # cs_2f._dofs = f2
        # pose_2F = pose_mono.clone()
        # cs_2f.make_symmetric_pose(pose_2F)
        # sds = SymDefSwapper(pose_2F, cs_2f.get_norm_symdef_path("T", "2F"))
        # for i in range(1):
        #     randomize_all_dofs(pose_2F)
        #     pose_HF = sds.make_HF_pose_from_2F(pose_2F)
        #
        #     pose_HF.pdb_info().name(f"HF_from_2F_{i}")
        #     pose_2F.pdb_info().name(f"2F_{i}")
        #     pmm.apply(pose_HF)
        #     pmm.apply(pose_2F)
        # # the vrt origo overlap here in this case, but the vectors do not
        # # sds.fold2_setup.update_dofs_from_pose(pose_2F)
        # # sds.foldHF_setup.update_dofs_from_pose(pose_HF)
        # # sds.fold2_setup.visualize(ip="10.8.0.6", suffix="F2")
        # # sds.foldHF_setup.visualize(ip="10.8.0.6", suffix="HF")
        # ...

        cs_3f = CubicSetup()
        cs_3f.load_norm_symdef("T", "3F")
        # cs_3f._dofs = f3
        pose_3F = pose_mono.clone()
        cs_3f.make_symmetric_pose(pose_3F)
        sds = SymDefSwapper(pose_3F, cs_3f.get_norm_symdef_path("T", "3F"))
        randomize_all_dofs(pose_3F)
        pose_2F = sds.make_2F_pose_from_3F(pose_3F)
        pose_HF_2 = sds.make_HF_pose_from_2F(pose_2F)
        pose_HF_3 = sds.make_HF_pose_from_3F(pose_3F)

        pose_HF.pdb_info().name("HF_from_3F")
        pose_3F.pdb_info().name("3F")
        pmm.apply(pose_HF)
        pmm.apply(pose_3F)
        # the dofs overlap here in this case
        sds.fold3_setup.update_dofs_from_pose(pose_3F)
        sds.foldHF_setup.update_dofs_from_pose(pose_HF)
        sds.fold3_setup.visualize(ip="10.8.0.6", suffix="F4")
        sds.foldHF_setup.visualize(ip="10.8.0.6", suffix="HF")
        ...




        # cs = CubicSetup()
        # cs.load_norm_symdef("T", "HF")
        # cs._dofs = fhf
        # pose_HF = pose_mono.clone()
        # cs.make_symmetric_pose(pose_HF)
        #
        #
        # pose_3F = sds.create_3fold_pose_from_HFfold(pose_HF)
        # pose_2F = sds.create_2fold_pose_from_HFfold(pose_HF)
        # randomize_all_dofs(pose_HF)
        # randomize_all_dofs(pose_3F)
        # randomize_all_dofs(pose_2F)
        # pose_HF.pdb_info().name("HF")
        # pose_3F.pdb_info().name("3F")
        # pose_2F.pdb_info().name("2F")
        #
        # # poseHF -> pose3F -> pose_2F works
        # if False:
        #     sds.transfer_HFto3(pose_HF, pose_3F)
        #     sds.transfer_3to2(pose_3F, pose_2F)
        # # pose2F -> poseHF -> pose_3F works
        # if True:
        #     sds.transfer_2toHF(pose_2F, pose_HF)
        #     sds.transfer_HFto3(pose_HF, pose_3F)
        # pmm.apply(pose_HF)
        # pmm.apply(pose_3F)
        # pmm.apply(pose_2F)
        # ...

def test_normalized_symdefswapper():
    from simpletestlib.test import setup_test, setup_pymol
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, perturb_jumpdof_str_str, set_jumpdof_str_str
    import random
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.actors.symdefswapper import SymDefSwapper
    pmm = setup_pymol()
    pmm.keep_history(True)
    import math
    fhf = {'JUMPHFfold1': [['z', 'translation', 20.14548822689412]],
     'JUMPHFfold1_z': [['z', 'rotation', 0.0]],
     'JUMPHFfold111': [['x', 'translation', 14.46937894105028]],
     'JUMPHFfold111_x': [['x', 'rotation', 0.0]],
     'JUMPHFfold111_y': [['y', 'rotation', 0.0]],
     'JUMPHFfold111_z': [['z', 'rotation', 0.0]]}
    f3 = {'JUMP31fold1': [['z', 'translation', 20.357023099436248]],
     'JUMP31fold1_z': [['z', 'rotation', 0]],
     'JUMP31fold111': [['x', 'translation', 14.17022170037703]],
     'JUMP31fold111_x': [['x', 'rotation', 0]],
     'JUMP31fold111_y': [['y', 'rotation', 0]],
     'JUMP31fold111_z': [['z', 'rotation', 0]]}
    f2 = {'JUMP21fold1': [['z', 'translation', 17.538101678972165]],
     'JUMP21fold1_z': [['z', 'rotation', 0]],
     'JUMP21fold111': [['x', 'translation', 17.539064146909325]],
     'JUMP21fold111_x': [['x', 'rotation', 0]],
     'JUMP21fold111_y': [['y', 'rotation', 0]],
     'JUMP21fold111_z': [['z', 'rotation', 0]]}
    for pdb in ("2CC9", "7Q03", "1H0S"):
        pose_mono = setup_test(name="T", file=pdb, return_symmetry_file=False, mute=True, pymol=False, symmetrize=False)

        cs = CubicSetup()
        cs.load_norm_symdef("T", "HF")
        cs._dofs = fhf
        pose_HF = pose_mono.clone()
        cs.make_symmetric_pose(pose_HF)

        sds = SymDefSwapper(pose_HF, cs.get_norm_symdef_path("T", "HF"))
        pose_3F = sds.create_3fold_pose_from_HFfold(pose_HF)
        pose_2F = sds.create_2fold_pose_from_HFfold(pose_HF)
        randomize_all_dofs(pose_HF)
        randomize_all_dofs(pose_3F)
        randomize_all_dofs(pose_2F)
        pose_HF.pdb_info().name("HF")
        pose_3F.pdb_info().name("3F")
        pose_2F.pdb_info().name("2F")

        # poseHF -> pose3F -> pose_2F works
        if True:
            sds.transfer_HFto3(pose_HF, pose_3F)
            sds.transfer_3to2(pose_3F, pose_2F)
        pmm.apply(pose_HF)
        pmm.apply(pose_3F)
        pmm.apply(pose_2F)
        randomize_all_dofs(pose_HF)
        randomize_all_dofs(pose_3F)
        randomize_all_dofs(pose_2F)
        # pose2F -> poseHF -> pose_3F works
        if True:
            sds.transfer_2toHF(pose_2F, pose_HF)
            sds.transfer_HFto3(pose_HF, pose_3F)
        pmm.apply(pose_HF)
        pmm.apply(pose_3F)
        pmm.apply(pose_2F)
        ...

        # pose_2F = pose_mono.clone()
        # cs.make_symmetric_pose(pose_2F)
        # cs = CubicSetup()
        # cs.load_norm_symdef("T", "2F")
        # cs._dofs = f2
        # pose_2F = pose_mono.clone()
        # cs.make_symmetric_pose(pose_2F)
        #
        # cs = CubicSetup()
        # cs.load_norm_symdef("T", "3F")
        # cs._dofs = f3
        # pose_3F = pose_mono.clone()
        # cs.make_symmetric_pose(pose_3F)

        # pose_HF.pdb_info().name("HF")
        # pose_X.pdb_info().name(f"{fold}")
        # if fold == "2F":
        #     pose_HF_X = sds.make_HF_pose_from_2F(pose_X)
        #     perturb_jumpdof_str_str(pose_HF_X, "JUMPHFfold111_x", "angle_x", 180)
        #     rmsd = cs.CA_rmsd_hf_map(pose_X, pose_HF_X)
        #     assert math.isclose(rmsd, 0, abs_tol=1.e-1)


def test_all_symm():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    sym_files = {"I": ["1STM", "6S44"],
                 "O": ["1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S"],
                 }
    for sym, files in sym_files.items():
        for pdb in files:
            poseHF, pmm, cmd, symdef = setup_test(name=sym, file=pdb, return_symmetry_file=True, mute=True)
            pmm.keep_history(True)
            sds = SymDefSwapper(poseHF, symdef)
            change_alot_and_transfer(poseHF, sds, pmm)

def test_T_sym():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1MOG", "1H0S"):
        poseHF, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef)
        change_alot_and_transfer(poseHF, sds, pmm)

def test_O_sym():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1AEW", "1P3Y"):
        poseHF, pmm, cmd, symdef = setup_test(name="O", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef)
        poseHF.pdb_info().name("HF")
        change_alot_and_transfer(poseHF, sds, pmm)

def test_I_sym():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    for pdb in ("1STM", "6S44"):
        poseHF, pmm, cmd, symdef = setup_test(name="I", file=pdb, return_symmetry_file=True, mute=True)
        pmm.keep_history(True)
        sds = SymDefSwapper(poseHF, symdef)
        poseHF.pdb_info().name("HF")
        change_alot_and_transfer(poseHF, sds, pmm)

def change_alot_and_transfer(pose_org_HF, sds, pmm, behavior="randomize", atol=2):
    from cubicsym.kinematics import randomize_all_dofs, randomize_all_dofs_positive_trans
    from cubicsym.utilities import get_chain_map
    from cubicsym.cubicsetup import CubicSetup
    from shapedesign.src.utilities.pose import pose_cas_are_identical
    def check_overlap(poseHF, pose3, pose2):
        sds.foldHF_setup.vrts_overlap_with_pose(poseHF)
        sds.fold3_setup.vrts_overlap_with_pose(pose3)
        sds.fold2_setup.vrts_overlap_with_pose(pose2)
    for i in range(12):
        if i % 3 == 0:
            poseHF = pose_org_HF.clone()
        if behavior == "randomize":
            randomize_all_dofs(poseHF)
        elif behavior == "positive_trans":
            randomize_all_dofs_positive_trans(poseHF)
        else:
            raise ValueError
        pose3 = sds.create_3fold_pose_from_HFfold(poseHF, transfer=True)
        pose2 = sds.create_2fold_pose_from_HFfold(poseHF, transfer=True)
        check_overlap(poseHF, pose3, pose2)
        # sds.transfer_HFto3(poseHF, pose3)
        sds.transfer_poseA2B(poseHF, pose3)
        check_overlap(poseHF, pose3, pose2)
        sds.transfer_poseA2B(pose3, pose2)
        check_overlap(poseHF, pose3, pose2)
        sds.transfer_poseA2B(pose2, poseHF)
        check_overlap(poseHF, pose3, pose2)
        poseHF.pdb_info().name("HF")
        pose3.pdb_info().name("3")
        pose2.pdb_info().name("2")
        print("new")
        try:
            assert pose_cas_are_identical(poseHF, pose3, pose2, map_chains=get_chain_map(CubicSetup.cubic_symmetry_from_pose(poseHF), sds.foldHF_setup.righthanded), atol=atol)
        except AssertionError:
            pmm.apply(poseHF)
            pmm.apply(pose3)
            pmm.apply(pose2)
            raise AssertionError
        else:
            ...


def test_transfer():
    assert False

def test_dof_directions():
    # Conclusion:
    # x and z trans/rotations are in the wrong direction
    # The coordinate system is acts like there has been a 180 degree rotation around y before
    # applying any rotaions/translations
    from shapedesign.src.utilities.pymol import  setup_pymol_server
    from symmetryhandler.kinematics import perturb_jumpdof
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    # pose, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    from pyrosetta import pose_from_file, init
    init("-initialize_rigid_body_dofs 1")
    pose = pose_from_file("/home/shared/databases/SYMMETRICAL/I/idealized/input/native/1STM.cif")
    # SetupForSymmetryMover("/home/shared/databases/SYMMETRICAL/I/idealized/symdef/native/1STM.symm").apply(pose)
    name = "/tmp/symdeftmp.symm"
    with open(name, "w") as f:
        f.write("""symmetry_name /home/shared/databases/SYMMETRICAL/I/unrelaxed/native/../../idealized/symdef/native/1STM.symm
E = 1*VRTglobal
anchor_residue COM
virtual_coordinates_start
xyz VRTglobal 1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
virtual_coordinates_stop
connect_virtual JUMPHFfold VRTglobal SUBUNIT 
set_dof JUMPHFfold z
set_jump_group JUMPGROUP1 JUMPHFfold""")
    SetupForSymmetryMover(name).apply(pose)
    pose.pdb_info().name("before")
    cmd, pmm = setup_pymol_server()
    pmm.apply(pose)
    pmm.keep_history(True)
    pose.pdb_info().name("after")
    for i in range(5):
        # Translation test
        # x test: WRONG direction
        #perturb_jumpdof(pose, "JUMPHFfold", 1, 10)
        # y test: RIGHT direction
        # perturb_jumpdof(pose, "JUMPHFfold", 2, 10)
        # z test: WRONG direction
        perturb_jumpdof(pose, "JUMPHFfold", 3, 10)

        # rotation test (looking from origo through the axis it should be a right handed rotation)
        # x test: WRONG direction
        # perturb_jumpdof(pose, "JUMPHFfold", 4, 10)
        # y test: RIGHT direction
        # perturb_jumpdof(pose, "JUMPHFfold", 5, 10)
        # z test: WRONG direction
        # perturb_jumpdof(pose, "JUMPHFfold", 6, 10)
        pmm.apply(pose)
    cmd.do("run axes.py")


def test_transfer_5to3():
    from shapedesign.src.utilities.tests import setup_test
    # make pose5
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    pmm.keep_history(True)

def test_transfer_5to3():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    # make pose5
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    pmm.keep_history(True)
    # pose5.pdb_info().name("pose5_before")
    # pose3.pdb_info().name("pose3_before")
    # pmm.apply(pose5)
    # pmm.apply(pose3)
    perturb_jumpdof(pose5, "JUMPHFfold1", 3, 10) # 100 = -1, -10=1 ; -90 fails but that is probably because vrts starts to be in the opposite direciton (approx -90 for 1stm)
    perturb_jumpdof(pose5, "JUMPHFfold1", 6, -10)
    perturb_jumpdof(pose5, "JUMPHFfold111", 1, 40)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 4, -50)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 5, 50)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 6, -50)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5)
    # setup symdefswapper
    start = time.time()
    # swap from 5 to 3
    sds.transfer_IHFto3(pose5, pose3)
    pose5.pdb_info().name("pose5_after")
    pose3.pdb_info().name("pose3_after")
    pmm.apply(pose5)
    pmm.apply(pose3)
    sds.fold3_setup.update_dofs_from_pose(pose3)
    sds.fold3_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
    # sds.fold5_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)

    # assert np.isclose(sds.fold3_setup.get_vrt("VRT31fold1111").vrt_x, sds.fold5_setup.get_vrt("VRTHFfold1111").vrt_x).all()
    # assert np.isclose(sds.fold3_setup.get_vrt("VRT31fold1111").vrt_y, sds.fold5_setup.get_vrt("VRTHFfold1111").vrt_y).all()
    # assert np.isclose(sds.fold3_setup.get_vrt("VRT31fold1111").vrt_z, sds.fold5_setup.get_vrt("VRTHFfold1111").vrt_z).all()
    #
    # pmm.keep_history(True)
    # for i in range(5):
    #     perturb_jumpdof(pose3, "JUMP31fold1", 3, 10)
    #     pmm.apply(pose3)
    #



    # print("time:", start - time.time())
    # setup.visualize(ip="10.8.0.6", port="9123")
    # ss3.visualize(ip="10.8.0.6", port="9123")
    # pmm.apply(pose)
    # apply the symmetry:
    # asympose.pdb_info().name("3foldbased")
    # pmm.apply(asympose)
    ...


def test_transfer_3to2():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)

    pose3.pdb_info().name("pose3_before")
    pmm.apply(pose3)
    perturb_jumpdof(pose3, "JUMP31fold1", 3, -10)
    perturb_jumpdof(pose3, "JUMP31fold1", 6, 20)
    perturb_jumpdof(pose3, "JUMP31fold111", 1, -10)
    perturb_jumpdof(pose3, "JUMP31fold1111", 4, -50)
    perturb_jumpdof(pose3, "JUMP31fold1111", 5, 50)
    perturb_jumpdof(pose3, "JUMP31fold1111", 6, -50)
    sds.transfer_3to2(pose3, pose2)
    pose3.pdb_info().name("pose3_after")
    pose2.pdb_info().name("pose2_after")
    pmm.apply(pose3)
    pmm.apply(pose2)

    sds.fold2_setup.update_dofs_from_pose(pose2)
    # sds.fold3_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)
    sds.fold2_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)

    pmm.keep_history(True)
    for i in range(5):
        perturb_jumpdof(pose2, "JUMP21fold1", 3, 10)
        pmm.apply(pose2)

def test_delete():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    sds = SymDefSwapper(pose5, symm_file)
    # pose3 = sds.create_2fold_pose_from_5fold(pose5)
    # pose2 = sds.create_2fold_pose_from_5fold(pose5)
    # perturb_jumpdof(pose2, "JUMP21fold1", 3, 10)
    # perturb_jumpdof(pose2, "JUMP21fold1", 6, 10)
    # perturb_jumpdof(pose2, "JUMP21fold111", 1, 10)
    # perturb_jumpdof(pose2, "JUMP21fold1111", 4, -50)
    # perturb_jumpdof(pose2, "JUMP21fold1111", 5, 50)
    # perturb_jumpdof(pose2, "JUMP21fold1111", 6, -50)
    # sds.fold5_setup.update_dofs_from_pose(pose5)
    pmm.apply(pose5)
    # sds.fold5_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
    # sds.fold3_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
    sds.fold2_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)

def test_transfer_2to5():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    # setup = SymmetrySetup(symm_file)
    # sds = SymDefSwapper(pose5, symm_file)
    # setup.update_dofs_from_pose(pose5, apply_dofs=True)
    # SymmetrySetup.output_vrts_as_pdb(pose5)
    pmm.apply(pose5)
    # exit(0)


    sds = SymDefSwapper(pose5, symm_file)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)
    perturb_jumpdof(pose2, "JUMP21fold1", 3, 10)
    perturb_jumpdof(pose2, "JUMP21fold1", 6, 10)
    perturb_jumpdof(pose2, "JUMP21fold111", 1, 10)
    perturb_jumpdof(pose2, "JUMP21fold1111", 4, -50)
    perturb_jumpdof(pose2, "JUMP21fold1111", 5, 50)
    perturb_jumpdof(pose2, "JUMP21fold1111", 6, -50)
    sds.foldHF_setup.update_dofs_from_pose(pose5)
    sds.__transfer_2toIHF(pose2, pose5)
    pose2.pdb_info().name("pose2_after")
    pose5.pdb_info().name("pose5_after")
    pmm.apply(pose5)
    pmm.apply(pose2)
    # sds.fold2_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
    sds.foldHF_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
    sds.foldHF_setup.update_dofs_from_pose(pose5)
    pose5.pdb_info().name("pose5_is_messed_upno")
    pmm.apply(pose5)
    #sds.fold5_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)

    # pmm.keep_history(True)
    # for i in range(5):
    #     perturb_jumpdof(pose5, "JUMPHFfold1", 3, 10)
    #     pmm.apply(pose5)
    #

# def test_delete_this():
#     from shapedesign.src.utilities.tests import setup_test
#     from symmetryhandler.symdefswapper import SymDefSwapper
#     from symmetryhandler.symmetryhandler import SymmetrySetup
#     from shapedesign.src.utilities.kinematics import perturb_jumpdof
#     pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
#     pmm.apply(pose5)
#
#     sds = SymDefSwapper(pose5, symm_file)
#     pose2 = sds.create_2fold_pose_from_5fold(pose5)
#     sds.transfer_2to5(pose2, pose5)
#     pose2.pdb_info().name("pose2_after")
#     pose5.pdb_info().name("pose5_after")
#     pmm.apply(pose5)
#     pmm.apply(pose2)
#     # sds.fold2_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
#     # sds.fold5_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
#     sds.fold5_setup.output_vrts_as_pdb(pose5)
#     sds.fold5_setup.update_dofs_from_pose(pose5)
#     pmm.apply(pose5)
#     sds.fold5_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
#     #sds.fold5_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)
#
#     # pmm.keep_history(True)
#     # for i in range(5):
#     #     perturb_jumpdof(pose5, "JUMPHFfold1", 3, 10)
#     #     pmm.apply(pose5)
#     #

def test_transfer_5to2():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    sds = SymDefSwapper(pose5, symm_file)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)

    pose2.pdb_info().name("pose2_before")
    pmm.apply(pose5)
    perturb_jumpdof(pose5, "JUMPHFfold1", 3, -10)
    perturb_jumpdof(pose5, "JUMPHFfold1", 6, 20)
    perturb_jumpdof(pose5, "JUMPHFfold111", 1, -10)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 4, -50)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 5, 50)
    perturb_jumpdof(pose5, "JUMPHFfold1111", 6, -50)
    sds.transfer_HFto2(pose5, pose2)
    pose5.pdb_info().name("pose5_after")
    pose2.pdb_info().name("pose2_after")
    pmm.apply(pose5)
    pmm.apply(pose2)

    sds.fold2_setup.update_dofs_from_pose(pose2)
    # sds.fold3_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)
    sds.fold2_setup.visualize(ip="10.8.0.10", port="9123", apply_dofs=True)

    pmm.keep_history(True)
    for i in range(5):
        perturb_jumpdof(pose2, "JUMP21fold1", 3, 10)
        pmm.apply(pose2)

def test_fixed_symdefswapper():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from shapedesign.src.utilities.pose import get_position_info
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    pmm.keep_history(True)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5, transfer=False)
    # pose2 = sds.create_2fold_pose_from_5fold(pose5)
    # perturb_jumpdof(pose5, "JUMPHFfold1", 3, -10)
    # perturb_jumpdof(pose5, "JUMPHFfold1", 6, 20)
    # perturb_jumpdof(pose5, "JUMPHFfold111", 1, -10)
    # perturb_jumpdof(pose5, "JUMPHFfold1111", 4, -50)
    # perturb_jumpdof(pose5, "JUMPHFfold1111", 5, 50)
    # perturb_jumpdof(pose5, "JUMPHFfold1111", 6, -50)
    pose5.pdb_info().name("5")
    pose3.pdb_info().name("3")
    pmm.apply(pose5)
    pmm.apply(pose3)
    # print(get_position_info(pose3, dictionary=True))
    # pmm.apply(pose2)
    print("first time transfer")
    sds.transfer_IHFto3(pose5, pose3)
    pmm.apply(pose5)
    pmm.apply(pose3)
    sds.transfer_IHFto3(pose5, pose3)
    pmm.apply(pose5)
    pmm.apply(pose3)
    sds.transfer_IHFto3(pose5, pose3)
    pmm.apply(pose5)
    pmm.apply(pose3)
    sds.transfer_IHFto3(pose5, pose3)
    pmm.apply(pose5)
    pmm.apply(pose3)
    # sds.transfer_5to3(pose5, pose3)
    # pmm.apply(pose5)
    # pmm.apply(pose3)
    # sds.transfer_5to3(pose5, pose3)
    # pmm.apply(pose5)
    # pmm.apply(pose3)
    # print(get_position_info(pose3, dictionary=True))
    # print("second time transfer")
    # sds.transfer_5to3(pose5, pose3)
    # # sds.transfer_5to3(pose5, pose3)
    # # sds.transfer_3to2(pose3, pose2)
    # # sds.transfer_2to5(pose2, pose5)
    # # sds.transfer_5to2(pose5, pose2)
    # pmm.apply(pose5)
    # pmm.apply(pose3)
    # print(get_position_info(pose3, dictionary=True))
    # pmm.apply(sds.init_pose5)
    # pmm.apply(sds.init_pose3)
    # pmm.apply(sds.init_pose2)
    # pmm.apply(pose2)
    print(get_position_info(pose3, dictionary=True))
    print(get_position_info(pose5, dictionary=True))
    sds.foldHF_setup.update_dofs_from_pose(pose5)
    sds.fold3_setup.update_dofs_from_pose(pose3)
    sds.foldHF_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)
    sds.fold3_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)
    sds.foldHF_setup.visualize(ip="10.8.0.6", port="9123", apply_dofs=True)
    sds.fold3_setup.output_vrts_as_pdb(pose3, "/tmp/vrt3.pdb")
    sds.foldHF_setup.output_vrts_as_pdb(pose5, "/tmp/vrt5.pdb")

    # "/tmp/vrts.pdb"

def test_fixed_transfer():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from shapedesign.src.utilities.kinematics import perturb_jumpdof
    import random
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    pmm.keep_history(True)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)
    pose5.pdb_info().name("5")
    pose3.pdb_info().name("3")
    pose2.pdb_info().name("2")
    perturb_jumpdof(pose5, "JUMPHFfold1", 3, random.uniform(-10, 10))
    perturb_jumpdof(pose5, "JUMPHFfold1_z", 6, random.uniform(-20, 20))
    perturb_jumpdof(pose5, "JUMPHFfold111", 1, random.uniform(-10, 10))
    perturb_jumpdof(pose5, "JUMPHFfold111_x", 4, 180) # random.uniform(-180, 180))
    perturb_jumpdof(pose5, "JUMPHFfold111_y", 5, 180) # random.uniform(-180, 180))
    perturb_jumpdof(pose5, "JUMPHFfold111_z", 6, 180) # random.uniform(-180, 180))
    sds.transfer_IHFto3(pose5, pose3)
    sds.transfer_3to2(pose3, pose2)
    sds.__transfer_2toIHF(pose2, pose5)
    pmm.apply(pose5)
    pmm.apply(pose3)
    pmm.apply(pose2)




def test_different_backbones():
    from simpletestlib.test import setup_test
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    from shapedesign.src.utilities.pose import pose_cas_are_identical
    import random
    from pathlib import Path
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover
    pose5, pmm, cmd, symm_file = setup_test(name="I", file="1STM", return_symmetry_file=True)
    pmm.keep_history(True)
    sds = SymDefSwapper(pose5, symm_file)
    pose3 = sds.create_3fold_pose_from_HFfold(pose5)
    pose2 = sds.create_2fold_pose_from_HFfold(pose5)
    pose5.pdb_info().name("5")
    pose3.pdb_info().name("3")
    pose2.pdb_info().name("2")

    def change_alot_and_transfer(pose5, sds, atol=1e-6):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-20, 20)
        d = random.uniform(-180, 180)
        e = random.uniform(-180, 180)
        f = random.uniform(-180, 180)
        #a, b, c, d, e, f = (-5.05655459739952, 5.219965952869327, 13.536955334211747, -134.05324099693365, 163.49674082016134, -59.773915558728845)
        # a, b, c, d, e, f = (-5.05655459739952, 5.219965952869327, 13.536955334211747, -134.05324099693365, 91, -59.773915558728845)
        perturb_jumpdof(pose5, "JUMPHFfold1", 3, a)
        perturb_jumpdof(pose5, "JUMPHFfold1_z", 6, b)
        perturb_jumpdof(pose5, "JUMPHFfold111", 1, c)
        perturb_jumpdof(pose5, "JUMPHFfold111_x", 4, d)
        perturb_jumpdof(pose5, "JUMPHFfold111_y", 5, e)
        perturb_jumpdof(pose5, "JUMPHFfold111_z", 6, f)
        pose3 = sds.create_3fold_pose_from_HFfold(pose5)
        pose2 = sds.create_2fold_pose_from_HFfold(pose5)
        # assert they are all the same
        pose5.pdb_info().name("5")
        pose3.pdb_info().name("3")
        pose2.pdb_info().name("2")
        pmm.apply(pose5)
        pmm.apply(pose3)
        pmm.apply(pose2)
        chain_map = [(1, 1, 1), (2, 4, 7), (3, 8, 6), (4, 9, 5), (5, 6, 3), (6, 2, 4), (7, 7, 8), (8, 5, 2), (9, 3, 9)]
        try:
            assert pose_cas_are_identical(pose5, pose3, pose2, map_chains=chain_map, atol=atol)
            sds.transfer_IHFto3(pose5, pose3)
            assert pose_cas_are_identical(pose5, pose3, map_chains=[(i[0], i[1]) for i in chain_map], atol=atol)
            sds.transfer_3to2(pose3, pose2)
            assert pose_cas_are_identical(pose3, pose2, map_chains=[(i[1], i[2]) for i in chain_map], atol=atol)
            sds.__transfer_2toIHF(pose2, pose5)
            assert pose_cas_are_identical(pose2, pose5, map_chains=[(i[2], i[0]) for i in chain_map], atol=atol)
        except AssertionError:
            raise AssertionError
        pmm.apply(pose5)
        pmm.apply(pose3)
        pmm.apply(pose2)

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
        change_alot_and_transfer(pose5, sds)
        anchors.append(sds.foldHF_setup.get_anchor_residue(pose5))
        print("anchors", anchors)

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

def test_debug():
    from shapedesign.src.utilities.tests import setup_test
    from symmetryhandler.symdefswapper import SymDefSwapper
    from symmetryhandler.kinematics import perturb_jumpdof
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    pmm.keep_history(True)
    pose5.pdb_info().name("5")
    pmm.apply(pose5)
    perturb_jumpdof(pose5, "JUMPHFfold111_y", 5, 2)  # random.uniform(-180, 180))
    pose5.pdb_info().name("5_after_1")
    pmm.apply(pose5)
    perturb_jumpdof(pose5, "JUMPHFfold111_y", 5, 2)  # random.uniform(-180, 180))
    pose5.pdb_info().name("5_after_2")
    pmm.apply(pose5)
    sds = SymDefSwapper(pose5, symm_file)
    pose5.pdb_info().name("5_sds_construct")
    pmm.apply(pose5)
    pose5 = sds.foldHF_setup.make_asymmetric_pose(pose5, dont_reset=["JUMPHFfold111_subunit"])
    # sds.fold5_setup.reset_jumpdofs("JUMPHFfold111_subunit")
    pose5.pdb_info().name("5_sds_applied")
    sds.foldHF_setup.make_symmetric_pose(pose5)
    pmm.apply(pose5)
    # # As we reuse this function the subunit dofs might change and we want to keep them fixed
    # #
    # pose3 = sds.create_3fold_pose_from_5fold(pose5)
    # pose5.pdb_info().name("5_after")
    # pose3.pdb_info().name("3")
    # pose2.pdb_info().name("2")
    # pmm.apply(pose3)
    # pmm.apply(pose2)
    # pose2 = sds.create_2fold_pose_from_5fold(pose5)
    ...
    # chain_map = [(1, 1, 1), (2, 4, 7), (3, 8, 6), (4, 9, 5), (5, 6, 3), (6, 2, 4), (7, 7, 8), (8, 5, 2), (9, 3, 9)]
    # # assert pose_cas_are_identical(pose5, pose3, pose2, map_chains=chain_map, atol=1)

def test_jump():
    from shapedesign.src.utilities.tests import setup_test
    from pyrosetta.rosetta.core.kinematics import Stub, Jump
    from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
    from symmetryhandler.kinematics import perturb_jumpdof
    from symmetryhandler.symmetryhandler import SymmetrySetup
    pose5, pmm, cmd, symm_file = setup_test("1stm", return_symmetry_file=True)
    ss = SymmetrySetup()
    ss.read_from_file(symm_file)
    ss.update_dofs_from_pose(pose5, apply_dofs=True)
    pmm.keep_history(True)
    pose5.pdb_info().name("5")
    pmm.apply(pose5)
    def get_jump(pose5):
        stub1 = Stub(pose5.conformation().upstream_jump_stub(sym_dof_jump_num(pose5, "JUMPHFfold111_z")))
        stub2 = Stub(pose5.conformation().downstream_jump_stub(sym_dof_jump_num(pose5, "JUMPHFfold111_z")))
        return Jump(stub1, stub2)

    jump = get_jump(pose5)
    print(jump.get_rb_delta(1))
    perturb_jumpdof(pose5, "JUMPHFfold111_y", 5, 89)  # random.uniform(-180, 180))
    jump = get_jump(pose5)
    print(jump.get_rb_delta(1))
    ...


