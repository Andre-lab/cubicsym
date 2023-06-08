#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of SymmetrySlider
@Author: Mads Jeppesen
@Date: 6/7/22
"""

def test_slide_all_single_bb():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.visualization.visualizer import Visualizer
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.kinematics import randomize_all_dofs_positive_trans, translate_away, randomize_all_dofs, get_dofspecification_for_pose
    from simpletestlib.test import setup_test
    from pyrosetta import init
    from symmetryhandler.reference_kinematics import get_jumpdof_str_int, get_dofs, perturb_jumpdof_str_str
    from cloudcontactscore.cloudcontactscorecontainer import CloudContactScoreContainer
    from cubicsym.cubicdofs import CubicDofs
    import random
    import math
    random.seed(661)
    sym_files = {"I": ["1STM", "6S44"],
                 "T": ["1MOG", "1H0S"],
                 "O": ["1AEW", "1P3Y"]}

    init()
    # vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    for sym, files in sym_files.items():
        for pdb in files:
            pose_mono, pmm, cmd, symdef = setup_test(name=sym, file=pdb, return_symmetry_file=True, mute=True, symmetrize=False)
            pmm.keep_history(True)

            # create HF, 3 and 2-fold independently
            cs_hf = CubicSetup()
            cs_hf.load_norm_symdef(sym, "HF")
            pose_HF = pose_mono.clone()
            cs_hf.make_symmetric_pose(pose_HF)
            translate_away(pose_HF)
            vals_hf = [] #randomize_all_dofs_positive_trans(pose_HF, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=0, fold111_z=180, return_vals=True)
            # perturb_jumpdof_str_str(pose_HF, f"JUMPHFfold1_z", "angle_z", 180)
            # -----
            cs_2f = CubicSetup()
            cs_2f.load_norm_symdef(sym, "2F")
            pose_2F = pose_mono.clone()
            cs_2f.make_symmetric_pose(pose_2F)
            translate_away(pose_2F)
            vals_2f = [] # randomize_all_dofs_positive_trans(pose_2F, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=0, fold111_z=180, return_vals=True)
            # perturb_jumpdof_str_str(pose_2F, f"JUMP21fold1_z", "angle_z", -180)
            # -----
            cs_3f = CubicSetup()
            cs_3f.load_norm_symdef(sym, "3F")
            pose_3F = pose_mono.clone()
            cs_3f.make_symmetric_pose(pose_3F)
            translate_away(pose_3F)
            vals_3f = [] #randomize_all_dofs_positive_trans(pose_3F, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=180, fold111_z=180, return_vals=True)
            # perturb_jumpdof_str_str(pose_3F, f"JUMP31fold1_z", "angle_z", 180)

            for pose, cs, vals in zip([pose_3F, pose_2F, pose_HF], [cs_3f, cs_2f, cs_hf], [vals_3f, vals_2f, vals_hf]):
                ccsc = CloudContactScoreContainer(pose, cs.make_symmetry_file_on_tmp_disk(), CubicDofs(pose), multiple_bbs=False)
                sl = CubicSymmetrySlider(pose, cs.make_symmetry_file_on_tmp_disk(), ccsc, visualizer=None, trans_mag=2, pymolmover=None,
                                         debug_mode=True)
                jid = CubicSetup.get_jumpidentifier_from_pose(pose)
                z_pre = get_jumpdof_str_int(pose, f"JUMP{jid}fold1", 3)
                # x_pre = get_jumpdof_str_int(pose, f"JUMP{jid}fold111", 1)
                dofs_pre = get_dofs(pose)
                cmd.do("reinitialize")
                # change visualizer top visand pymolmover to pmm if you want to see the sliding
                # if pdb == "1STM" and  jid == "21":
                #     sl = CubicSymmetrySlider(pose, cs.make_symmetry_file_on_tmp_disk(), visualizer=None, trans_mag=2, pymolmover=pmm, debug_mode=True)
                # else:
                sl.apply(pose)
                z_post = get_jumpdof_str_int(pose, f"JUMP{jid}fold1", 3)

                ######################
                # Test 1: z is either the same as initially or less
                ######################
                try:
                    assert (z_pre >= z_post or math.isclose(z_pre, z_post, abs_tol=1e-2)), f"{z_pre}, {z_post}, {get_dofs(pose)}"
                except AssertionError as e:
                    print("===========================0")
                    print(f"{pdb} Failed")
                    print("base:", CubicSetup.get_base_from_pose(pose))
                    print("hit status:", sl.get_last_hit_status())
                    print("Vals given:", ", ".join(map(str, vals)))
                    print("before:", dofs_pre)
                    print("now:", get_dofs(pose))
                    print("===========================0")
                    raise AssertionError
                ######################
                # Test 2: z never goes below zero
                ######################
                try:
                    assert z_post > 0, f"{z_pre}, {z_post}, {get_dofs(pose)}"
                except AssertionError as e:
                    print(f"{pdb} Failed")
                    print("===========================0")
                    print("base:", CubicSetup.get_base_from_pose(pose))
                    print("hit status:", sl.get_last_hit_status())
                    print("Vals given:", ", ".join(map(str, vals)))
                    print("before:", dofs_pre)
                    print("now:", get_dofs(pose))
                    print("===========================0")
                    raise AssertionError

                ######################
                # Test 3: that if z is the same it is either because the folds hit each other immedially or never
                ######################
                if math.isclose(z_pre, z_post, abs_tol=1e-2):
                    data = sl.get_last_hit_status()[CubicSetup.get_base_from_pose(pose)]
                    hit = data["hit"]
                    moves = data["moves"]
                    assert (hit == False) or (moves == sl.max_slide_attemps) or (moves == 1)
                print("===========================0")
                print(f"{pdb} OK")
                print("hit status:", sl.get_last_hit_status())
                print("base:", CubicSetup.get_base_from_pose(pose))
                print("before:", dofs_pre)
                print("now:", get_dofs(pose))
                print("===========================0")

            print(f"{pdb} FINAL OK")

def test_slide_on_T_sym():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
    from shapedesign.src.visualization.visualizer import Visualizer
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    from simpletestlib.test import setup_test
    import random
    random.seed(661)
    for pdb in ("1MOG", "1H0S"):
        poseo, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        # make new pose
        pmm.keep_history(True)
        vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
        sf = SymDefSwapper(poseo, symdef, vis)
        sl = CubicSymmetrySlider(poseo, symdef, None, visualizer=vis, trans_mag=2, pymolmover=pmm)

        # rescue a HF fold
        poseHF = poseo.clone()
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1", 3, 20)
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1_z", 6, 2.5)
        sl.apply(poseHF)
        poseHF_final = poseHF.clone()

        # # rescue a 3 fold
        poseHF = poseo.clone()
        pose3 = sf.create_3fold_pose_from_HFfold(poseHF)
        pose2 = sf.create_2fold_pose_from_HFfold(poseHF)
        perturb_jumpdof_str_int(pose3, "JUMP31fold1", 3, 10)
        sf.transfer_3to2(pose3, pose2)
        sf.transfer_2toHF(pose2, poseHF)
        sl.apply(poseHF)
        pose3_final = poseHF.clone()
        print(CA_rmsd(poseo, pose3_final))

        # rescue a 2 fold
        poseHF = poseo.clone()
        pose2 = sf.create_2fold_pose_from_HFfold(poseHF)
        perturb_jumpdof_str_int(pose2, "JUMP21fold1", 3, 20)
        sf.transfer_2toHF(pose2, poseHF)
        sl.apply(poseHF)
        pose2_final = poseHF.clone()

        poseHF_final.pdb_info().name("final5")
        pose3_final.pdb_info().name("final3")
        pose2_final.pdb_info().name("final2")

        pmm.apply(poseHF_final)
        pmm.apply(pose3_final)
        pmm.apply(pose2_final)

def test_slide_on_O_sym():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
    from shapedesign.src.visualization.visualizer import Visualizer
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from simpletestlib.test import setup_test
    import random
    random.seed(661)
    for pdb in ("1AEW", "1P3Y"):
        poseo, pmm, cmd, symdef = setup_test(name="O", file=pdb, return_symmetry_file=True, mute=True)
        # make new pose
        pmm.keep_history(True)
        vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
        sf = SymDefSwapper(poseo, symdef, vis)
        sl = CubicSymmetrySlider(poseo, symdef, None, visualizer=vis, trans_mag=2, pymolmover=pmm)

        # rescue a HF fold
        poseHF = poseo.clone()
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1", 3, 20)
        perturb_jumpdof_str_int(poseHF, "JUMPHFfold1_z", 6, 2.5)
        sl.apply(poseHF)
        poseHF_final = poseHF.clone()

        # # rescue a 3 fold
        poseHF = poseo.clone()
        pose3 = sf.create_3fold_pose_from_HFfold(poseHF)
        pose2 = sf.create_2fold_pose_from_HFfold(poseHF)
        perturb_jumpdof_str_int(pose3, "JUMP31fold1", 3, 10)
        sf.transfer_3to2(pose3, pose2)
        sf.transfer_2toHF(pose2, poseHF)
        sl.apply(poseHF)
        pose3_final = poseHF.clone()
        print(CA_rmsd(poseo, pose3_final))

        # rescue a 2 fold
        poseHF = poseo.clone()
        pose2 = sf.create_2fold_pose_from_HFfold(poseHF)
        perturb_jumpdof_str_int(pose2, "JUMP21fold1", 3, 20)
        sf.transfer_2toHF(pose2, poseHF)
        sl.apply(poseHF)
        pose2_final = poseHF.clone()

        poseHF_final.pdb_info().name("final5")
        pose3_final.pdb_info().name("final3")
        pose2_final.pdb_info().name("final2")

        pmm.apply(poseHF_final)
        pmm.apply(pose3_final)
        pmm.apply(pose2_final)

def test_apply_global_slide():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.utilities.tests import setup_test
    from cubicsym.kinematics import perturb_jumpdof
    from shapedesign.src.visualization.visualizer import Visualizer
    import random
    random.seed(661)
    pose, pmm, cmd, symdef = setup_test(name="1stm", return_symmetry_file=True, mute=True)
    vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    # perturb pose
    z_trans = random.uniform(0, 100)
    perturb_jumpdof(pose, "JUMPHFfold1", 3, z_trans)
    x_trans = random.uniform(0, 100)
    perturb_jumpdof(pose, "JUMPHFfold111", 1, x_trans)
    z_rot1 = random.uniform(-20, 20)
    perturb_jumpdof(pose, "JUMPHFfold111", 6, z_rot1)
    x_rot2 = random.uniform(-180, 180)
    y_rot2 = random.uniform(-180, 180)
    z_rot2 = random.uniform(-180, 180)
    perturb_jumpdof(pose, "JUMPHFfold1111", 4, x_rot2)
    perturb_jumpdof(pose, "JUMPHFfold1111", 5, y_rot2)
    perturb_jumpdof(pose, "JUMPHFfold1111", 6, z_rot2)
    print("z_trans: ", z_trans, "x_trans: ", x_trans, "z_rot1", z_rot1, "x_rot2", x_rot2, "y_rot2", y_rot2, "z_rot2", z_rot2)
    pmm.apply(pose)
    sl = CubicSymmetrySlider(pose, None, None, visualizer=vis)
    sl.apply(pose)


def test_multiple_applies():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.utilities.tests import setup_test
    from cubicsym.kinematics import perturb_jumpdof
    from shapedesign.src.visualization.visualizer import Visualizer
    import random
    random.seed(661)
    pose, pmm, cmd, symdef = setup_test(name="1stm", return_symmetry_file=True, mute=True)
    pmm.keep_history(True)
    vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    sl = CubicSymmetrySlider(pose, symdef, None, visualizer=vis, trans_mag=1, pymolmover=pmm)

    # rescue a 5 fold
    for i in range(10):
        perturb_jumpdof(pose, "JUMPHFfold1", 3, random.uniform(-10, 10))
        perturb_jumpdof(pose, "JUMPHFfold1", 6, random.uniform(-10, 10))
        perturb_jumpdof(pose, "JUMPHFfold111", 1, random.uniform(-10, 10))
        perturb_jumpdof(pose, "JUMPHFfold1111", 4, random.uniform(-10, 10))
        perturb_jumpdof(pose, "JUMPHFfold1111", 5, random.uniform(-10, 10))
        perturb_jumpdof(pose, "JUMPHFfold1111", 6, random.uniform(-10, 10))
        sl.apply(pose)


def test_apply_local_slide():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.utilities.tests import setup_test
    from cubicsym.kinematics import perturb_jumpdof
    from shapedesign.src.visualization.visualizer import Visualizer
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    import random
    random.seed(661)
    poseo, pmm, cmd, symdef = setup_test(name="1stm", return_symmetry_file=True, mute=True)
    pmm.keep_history(True)
    vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    sf = SymDefSwapper(poseo, symdef, vis)
    sl = CubicSymmetrySlider(poseo, symdef, None, visualizer=vis, trans_mag=1)

    # rescue a 5 fold
    pose5 = poseo.clone()
    perturb_jumpdof(pose5, "JUMPHFfold1", 3, 20)
    perturb_jumpdof(pose5, "JUMPHFfold1", 6, 2.5)
    sl.apply(pose5)
    pose5_final = pose5.clone()

    # # rescue a 3 fold
    pose5 = poseo.clone()
    pose3 = sf.create_3fold_pose_from_HFfold(pose5)
    pose2 = sf.create_2fold_pose_from_HFfold(pose5)
    perturb_jumpdof(pose3, "JUMP31fold1", 3, 10)
    sf.transfer_3to2(pose3, pose2)
    sf.__transfer_2to5(pose2, pose5)
    sl.apply(pose5)
    pose3_final = pose5.clone()
    print(CA_rmsd(poseo, pose3_final))

    # rescue a 2 fold
    pose5 = poseo.clone()
    pose2 = sf.create_2fold_pose_from_HFfold(pose5)
    perturb_jumpdof(pose2, "JUMP21fold1", 3, 20)
    sf.__transfer_2to5(pose2, pose5)
    sl.apply(pose5)
    pose2_final = pose5.clone()

    pose5_final.pdb_info().name("final5")
    pose3_final.pdb_info().name("final3")
    pose2_final.pdb_info().name("final2")

    pmm.apply(pose5_final)
    pmm.apply(pose3_final)
    pmm.apply(pose2_final)

def test_apply_local_slide_from_cubicsym():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.utilities.tests import setup_test
    from cubicsym.kinematics import perturb_jumpdof
    from shapedesign.src.visualization.visualizer import Visualizer
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    import random
    random.seed(661)
    for pdb in ("1STM", "6S44",  "6ZLO", "7NO0", "6RPO", "6JJA"):
        _, pmm, cmd, _ = setup_test(name=f"{pdb}", return_symmetry_file=True, mute=True)
        # make new pose
        poseo = pose_from_file(f"/home/mads/projects/cubicsym/tests/outputs/{pdb}.pdb")
        symdef = f"/home/mads/projects/cubicsym/tests/outputs/{pdb}.symm"
        SetupForSymmetryMover(symdef).apply(poseo)
        pmm.keep_history(True)
        vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
        sf = SymDefSwapper(poseo, symdef, vis)
        sl = CubicSymmetrySlider(poseo, symdef, None, visualizer=vis, trans_mag=2, pymolmover=pmm)

        # rescue a 5 fold
        pose5 = poseo.clone()
        perturb_jumpdof(pose5, "JUMPHFfold1", 3, 20)
        perturb_jumpdof(pose5, "JUMPHFfold1_z", 6, 2.5)
        sl.apply(pose5)
        pose5_final = pose5.clone()

        # # rescue a 3 fold
        pose5 = poseo.clone()
        pose3 = sf.create_3fold_pose_from_HFfold(pose5)
        pose2 = sf.create_2fold_pose_from_HFfold(pose5)
        perturb_jumpdof(pose3, "JUMP31fold1", 3, 10)
        sf.transfer_3to2(pose3, pose2)
        sf.__transfer_2to5(pose2, pose5)
        sl.apply(pose5)
        pose3_final = pose5.clone()
        print(CA_rmsd(poseo, pose3_final))

        # rescue a 2 fold
        pose5 = poseo.clone()
        pose2 = sf.create_2fold_pose_from_HFfold(pose5)
        perturb_jumpdof(pose2, "JUMP21fold1", 3, 20)
        sf.__transfer_2to5(pose2, pose5)
        sl.apply(pose5)
        pose2_final = pose5.clone()

        pose5_final.pdb_info().name("final5")
        pose3_final.pdb_info().name("final3")
        pose2_final.pdb_info().name("final2")

        pmm.apply(pose5_final)
        pmm.apply(pose3_final)
        pmm.apply(pose2_final)

def test_cannot_go_beyound_bounds():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.visualization.visualizer import Visualizer
    from shapedesign.src.movers.cubicboundarymoverfactory import CubicBoundary
    from shapedesign.src.utilities.tests import setup_test
    from cubicsym.kinematics import perturb_jumpdof

    pose, pmm, cmd, symdef = setup_test(name=f"1stm", return_symmetry_file=True, mute=True)
    # put far out in space
    perturb_jumpdof(pose, "JUMPHFfold1", 3, 40)
    perturb_jumpdof(pose, "JUMPHFfold111", 1, 40)

    dofspecification = {"JUMPHFfold1": {
        "z": {"limit_movement": True, "min": -5, "max": 5},
        "angle_z": {"limit_movement": True, "min": -5, "max": 5},
    },
        "JUMPHFfold111": {
            "x": {"limit_movement": True, "min": -5, "max": 5},
        },
        "JUMPHFfold1111": {
            "angle_x": {"limit_movement": True, "min": -5, "max": 5},
            "angle_y": {"limit_movement": True, "min": -5, "max": 5},
            "angle_z": {"limit_movement": True, "min": -5, "max": 5}
        }
    }
    vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
    slider = CubicSymmetrySlider(pose, symdef, None, visualizer=None, trans_mag=2, pymolmover=pmm, max_slide_attempts=1000)
    pmm.keep_history(True)
    pose.pdb_info().name("hello")
    pmm.apply(pose)
    slider.apply(pose, debug=False)
    pose.pdb_info().name("hello")
    pmm.apply(pose)
    assert cb.is_within_bounds(pose)


def move_pose(pose):
    import random
    from cubicsym.kinematics import perturb_jumpdof
    perturb_jumpdof(pose, "JUMPHFfold1", 3, random.uniform(-10, 10))
    perturb_jumpdof(pose, "JUMPHFfold1", 6, random.uniform(-20, 20))
    perturb_jumpdof(pose, "JUMPHFfold111", 1, random.uniform(-10, 10))
    perturb_jumpdof(pose, "JUMPHFfold1111", 4, random.uniform(-180, 180))
    perturb_jumpdof(pose, "JUMPHFfold1111", 5, random.uniform(-180, 180))
    perturb_jumpdof(pose, "JUMPHFfold1111", 6, random.uniform(-180, 180))

def test_apply_on_different_backbones():
    from cubicsym.actors.cubicsymmetryslider import CubicSymmetrySlider
    from shapedesign.src.utilities.tests import setup_test
    from shapedesign.src.visualization.visualizer import Visualizer
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    import random
    from pathlib import Path
    random.seed(661)
    pose_input, pmm, cmd, symdef = setup_test(name=f"1stm", return_symmetry_file=True, mute=True)
    vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    slider = CubicSymmetrySlider(pose_input, symdef, None, visualizer=None, trans_mag=2, pymolmover=pmm)

    # generate an ensemble
    poses = []
    for path in Path("/home/mads/projects/evodock/inputs/subunits/1STM/4").glob("*"):
        pose = pose_from_file(str(path))
        poses.append(pose)

    # apply mock unbound docking
    for pose in poses:
        SetupForSymmetryMover(symdef).apply(pose)
        move_pose(pose)
        slider.apply(pose)

def test_sliding_with_evodock():
    import logging
    import os
    from pyrosetta import init
    from src.config_reader import EvodockConfig
    from src.differential_evolution import DifferentialEvolutionAlgorithm as DE
    from src.differential_evolution import FlexbbDifferentialEvolution as FlexbbDE
    from src.options import build_rosetta_flags
    from src.scfxn_fullatom import FAFitnessFunction
    from src.utils import get_starting_poses
    from src.dock_metric import DockMetric, SymmetryDockMetric

    def initialize_dock_metric(config, native):
        if config.syminfo:
            jump_ids, dof_ids, trans_mags = [], [], []
            for a, b, c in config.syminfo.normalize_trans_map:
                jump_ids.append(a)
                dof_ids.append(b)
                trans_mags.append(c)
            return SymmetryDockMetric(native, jump_ids=jump_ids, dof_ids=dof_ids, trans_mags=trans_mags)
        else:
            return DockMetric(native)

    MAIN_PATH = os.getcwd()

    logging.basicConfig(level=logging.ERROR)

    config_file = "/home/mads/projecrts/evodock/configs/sample_sym_dock_flexbb_debug.ini"

    config = EvodockConfig(config_file)

    init(extra_options=build_rosetta_flags(config))

    logger = logging.getLogger("evodock")
    logger.setLevel(logging.INFO)

    # --- INIT PROTEIN STRUCTURES -------------------
    pose_input = config.pose_input
    native_input = config.native_input
    # input_pose = get_pose_from_file(pose_input)
    # native_pose = get_pose_from_file(native_input)

    input_pose, native_pose = get_starting_poses(pose_input, native_input, config)

    # --- INIT METRIC CALCULATOR ---
    dockmetric = initialize_dock_metric(config, native_pose)

    # ---- INIT SCORE FUNCTION ------------------------------
    scfxn = FAFitnessFunction(input_pose, native_pose, config, dockmetric)

    # ---- START ALGORITHM ---------------------------------
    if config.docking_type_option == "Flexbb":
        alg = FlexbbDE(config, scfxn)
    else:
        alg = DE(config, scfxn)

    alg.init_population()

    # --- RUN ALGORITHM -------------------------------------
    logger.info("==============================")
    logger.info(" starts EvoDOCK : evolutionary docking process")
    best_pdb = alg.main()

    # ---- OUTPUT -------------------------------------------
    logger.info(" end EvoDOCK")
    logger.info("==============================")
    if config.out_pdb:
        name = config.out_path + "/final_docked_evo.pdb"
        best_pdb.dump_pdb(name)


