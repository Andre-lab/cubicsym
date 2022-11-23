#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of SymmetrySlider
@Author: Mads Jeppesen
@Date: 6/7/22
"""

def test_slide_on_T_sym():
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
    for pdb in ("1MOG", ):
        poseo, pmm, cmd, symdef = setup_test(name="T", file=pdb, return_symmetry_file=True, mute=True)
        # make new pose
        pmm.keep_history(True)
        vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
        sf = SymDefSwapper(poseo, symdef, vis)
        sl = CubicSymmetrySlider(poseo, symdef, visualizer=vis, trans_mag=2, pymolmover=pmm)

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
    for pdb in ("7NTN", ):
        poseo, pmm, cmd, symdef = setup_test(name="O", file=pdb, return_symmetry_file=True, mute=True)
        # make new pose
        pmm.keep_history(True)
        vis = Visualizer(name="pose", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
        sf = SymDefSwapper(poseo, symdef, vis)
        sl = CubicSymmetrySlider(poseo, symdef, visualizer=vis, trans_mag=2, pymolmover=pmm)

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
    sl = CubicSymmetrySlider(pose, None, global_slide=True, visualizer=vis)
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
    sl = CubicSymmetrySlider(pose, symdef, visualizer=vis, trans_mag=1, pymolmover=pmm)

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
    sl = CubicSymmetrySlider(poseo, symdef, visualizer=vis, trans_mag=1)

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
        sl = CubicSymmetrySlider(poseo, symdef, visualizer=vis, trans_mag=2, pymolmover=pmm)

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
    slider = CubicSymmetrySlider(pose, symdef, visualizer=None, trans_mag=2, pymolmover=pmm, cubicboundarymoverfactory=cb, max_slide_attempts=1000)
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
    slider = CubicSymmetrySlider(pose_input, symdef, visualizer=None, trans_mag=2, pymolmover=pmm)

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


