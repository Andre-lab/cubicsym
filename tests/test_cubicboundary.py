#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test code for CubicBoundaryMoverFactory
@Author: Mads Jeppesen
@Date: 9/1/22
"""
import pytest
import importlib
import random
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, is_symmetric, jump_num_sym_dof


def change_dofs(pose):
    a = random.uniform(-50, 50)
    b = random.uniform(-50, 50)
    c = random.uniform(-50, 50)
    d = random.uniform(-180, 180)
    e = random.uniform(-180, 180)
    f = random.uniform(-180, 180)
    perturb_jumpdof_str_int(pose, "JUMPHFfold1", 3, a)
    perturb_jumpdof_str_int(pose, "JUMPHFfold1_z", 6, b)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111", 1, c)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_x", 4, d)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_y", 5, e)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_z", 6, f)
    return a, b, c, d, e, f

def change_dofs_range(pose, size=10, min_=-5, max=5):
    a = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    b = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    c = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    d = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    e = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    f = random.choice([random.uniform(-size, -size - min_), random.uniform(size - max, size)])
    perturb_jumpdof_str_int(pose, "JUMPHFfold1", 3, a)
    perturb_jumpdof_str_int(pose, "JUMPHFfold1_z", 6, b)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111", 1, c)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_x", 4, d)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_y", 5, e)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_z", 6, f)
    return a, b, c, d, e, f

def create_minmover(pose, sfxn, dofspecification):
    movemap_ = MoveMap()
    movemap_.set_chi(False)
    movemap_.set_bb(False)
    # movemap_.set_jump(True)
    for jump_name, _ in dofspecification.items():
        jumpid = sym_dof_jump_num(pose, jump_name)
        movemap_.set_jump(jumpid, True)
    min_tolerance_ = 0.01
    min_type_ = "lbfgs_armijo_nonmonotone"
    nb_list_ = True
    return MinMover(movemap_, sfxn, min_type_, min_tolerance_, nb_list_)

def test_cubic_boundary_mover():
    from simpletestlib.test import setup_test
    from cubicsym.actors.cubicboundary import CubicBoundary
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, symmetrize=True)
    lb_dof, ub_dof = -5, 5
    dofspecification = {
        "JUMPHFfold1": {"z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold1_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111": {"x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_x": {"angle_x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_y": {"angle_y": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    }
    cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
    sfxn = ScoreFunctionFactory.create_score_function("ref2015")
    cb.set_constraints(pose)
    cb.turn_on_constraint_for_score(sfxn)
    assert cb.constrains_are_set_in_score(sfxn)


@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_minization_time_with_different_sd_values():
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.actors.cubicboundary import CubicBoundary
    from simpletestlib.test import setup_test
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    import time
    import pandas as pd
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, symmetrize=True)
    lb_dof, ub_dof = -5, 5
    dofspecification = {
        "JUMPHFfold1": {"z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold1_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111": {"x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_x": {"angle_x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_y": {"angle_y": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
        "JUMPHFfold111_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    }
    sfxn = ScoreFunctionFactory.create_score_function("ref2015")
    times = {}
    done_times = 100
    for i in range(done_times):
        pose_moved = pose.clone()
        randomize_all_dofs(pose_moved, fold1=5, fold1_z=60, fold111=20, fold111_x=60, fold111_y=60, fold111_z=60, return_vals=False)
        for sd in ["NO", 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
            pose_t = pose_moved.clone()
            if not sd in times:
                times[sd] = []
            while True:
                try:
                    cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification, sd=sd)
                    if sd != "NO":
                        cb.set_constraints(pose_t)
                        cb.turn_on_constraint_for_score(sfxn)
                        assert cb.constrains_are_set_in_score(sfxn)
                    minmover = create_minmover(pose_t, sfxn, dofspecification)
                    break
                except AssertionError:
                    continue
            start = time.time()
            # if not cb.all_dofs_within_bounds(pose_t):
            #     sfxn.show(pose_t)
            try:
                minmover.apply(pose_t)
            except RuntimeError:
                continue
            print(f"final score for {sd}: {sfxn.score(pose_t)}")
            times[sd].append(time.time() - start)
    plt.figure()
    sns.barplot(data=pd.DataFrame({k:[np.array(v).mean()] for k, v in times.items()})).set(
                                  title=f'Mean times for {done_times} runs',
                                  ylabel="seconds")
    plt.show()

@pytest.mark.skipif(importlib.util.find_spec("simpletestlib") is None, reason="simpletestlib is needed in order to run!")
def test_constrains():
    from cubicsym.actors.cubicboundary import CubicBoundary
    from simpletestlib.test import setup_test
    from symmetryhandler.kinematics import perturb_jumpdof
    from symmetryhandler.kinematics import get_dofs
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from cubicsym.cubicsetup import CubicSetup
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from simpletestlib.pymol import get_test_options
    import numpy as np
    pose, pmm, cmd, symdef = setup_test(name="O", file="7NTN", return_symmetry_file=True, symmetrize=False)
    setup = CubicSetup()
    setup.read_from_file(symdef)
    setup.visualize(ip=get_test_options("pymol").get("ip"))
    SetupForSymmetryMover(symdef).apply(pose)
    pmm.apply(pose)

    lb_dof, ub_dof = -5, 5
    dofspecification = {
    "JUMPHFfold1": {"z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    "JUMPHFfold1_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    "JUMPHFfold111": {"x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    "JUMPHFfold111_x": {"angle_x": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    "JUMPHFfold111_y": {"angle_y": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    "JUMPHFfold111_z": {"angle_z": {"limit_movement": True, "min": lb_dof, "max": ub_dof}},
    }
    sfxn = ScoreFunctionFactory.create_score_function("empty")
    cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
    cb.turn_on_constraint_for_score(sfxn)

    # TEST 1: try a bunch of different dof variations and check that the dofs are correct
    for i in range(100):
        pose_clone = pose.clone()
        cb.set_constraints(pose_clone)
        a, b, c, d, e, f = change_dofs(pose_clone)
        pos = get_dofs(pose_clone)
        # assert the angles are measured correctly
        posa = pos["JUMPHFfold1"]["z"]
        posb = pos["JUMPHFfold1_z"]["angle_z"]
        posc = pos["JUMPHFfold111"]["x"]
        posd = pos["JUMPHFfold111_x"]["angle_x"]
        pose_ = pos["JUMPHFfold111_y"]["angle_y"]
        posf = pos["JUMPHFfold111_z"]["angle_z"]
        if posa < 0:
            assert np.isclose(cb.convert_to_rad(180), cb.get_z_boundary_angle(pose)), f"{cb.convert_to_rad(180)} != {cb.get_z_boundary_angle(pose)}"
        else:
            assert np.isclose(posa, cb.get_z_distance(pose_clone)), f"{posa} != {cb.get_z_distance(pose_clone)}"
        if posc < 0:
            assert np.isclose(cb.convert_to_rad(180), cb.get_x_boundary_angle(pose)), f"{cb.convert_to_rad(180)} != {cb.get_x_boundary_angle(pose)}"
        else:
            assert np.isclose(posc, cb.get_x_distance(pose_clone)), f"{posc} != {cb.get_x_distance(pose_clone)}"
        assert np.isclose(posb, cb.get_angle_z(pose_clone, degrees=True)), f"{posb} != {cb.get_angle_z(pose_clone, degrees=True)}"
        assert np.isclose(posd, cb.get_com_angle_x(pose_clone, degrees=True)), f"{posd} != {cb.get_com_angle_x(pose_clone, degrees=True)}"
        assert np.isclose(pose_, cb.get_com_angle_y(pose_clone, degrees=True)), f"{pose_} != {cb.get_com_angle_y(pose_clone, degrees=True)}"
        assert np.isclose(posf, cb.get_com_angle_z(pose_clone, degrees=True)), f"{posf} != {cb.get_com_angle_z(pose_clone, degrees=True)}"
        internal_penalty = cb.get_score(pose_clone)
        score_penalty = sfxn.score(pose_clone)
        assert np.isclose(internal_penalty, score_penalty), f"{internal_penalty} != {score_penalty}, {i}, {a,b,c,d,e,f}"
        cb.get_score(pose_clone) # FOR DEBUG

    # TEST 2: Check that it is always zero when constructing the rigidbodymover
    pose_clone = pose.clone()
    cb.set_constraints(pose_clone)
    rb_mover = cb.construct_rigidbody_mover(pose_clone)
    for _ in range(1000):
        rb_mover.apply(pose_clone)
        assert np.isclose(sfxn.score(pose_clone), 0)

    # TEST 3: test it works with the minimizer and the minimizer minizes the energy
    movemap_ = MoveMap()
    movemap_.set_chi(False)
    movemap_.set_bb(False)
    movemap_.set_jump(True)
    min_tolerance_ = 0.01
    min_type_ = "lbfgs_armijo_nonmonotone"
    nb_list_ = True
    min_mover = MinMover(movemap_, sfxn, min_type_, min_tolerance_, nb_list_)
    for i in range(10):
        pose_clone = pose.clone()
        cb.set_constraints(pose_clone)
        while np.isclose(sfxn.score(pose_clone), 0):
            change_dofs_range(pose_clone, size=2*abs(ub_dof), min_=lb_dof, max=ub_dof)
            print("changing dofs until 0")
        before_score = sfxn.score(pose_clone)
        pose_before = pose_clone.clone()
        min_mover.apply(pose_clone)
        after_score = sfxn.score(pose_clone)
        assert after_score < before_score, "The minmover did not minimize the energy"

    # TEST 4: check that is minimizes for large angles of angle_z
    pmm.apply(pose)
    for angles in (-130, 130):
        pose_clone = pose.clone()
        cb.set_constraints(pose_clone)
        perturb_jumpdof(pose_clone, "JUMPHFfold1_z", 6, angles)
        pose_before = pose_clone.clone()
        before_score = sfxn.score(pose_clone)
        min_mover.apply(pose_clone)
        after_score = sfxn.score(pose_clone)
        assert after_score < before_score, "The minmover did not minimize the energy"
        pose_after = pose_clone.clone()
        pmm.keep_history(True)
        pose_before.pdb_info().name("before")
        pose_after.pdb_info().name("after")
        pmm.apply(pose_before)
        pmm.apply(pose_after)
        print(f"{i}", after_score)
        cb.get_score(pose_clone)  # FOR DEBUG

def test_construct_rigidbody_mover():
    from cubicsym.actors.cubicboundary import CubicBoundary
    from simpletestlib.test import setup_test
    from symmetryhandler.kinematics import perturb_jumpdof
    import math
    for pdb, dof_set in zip(("6S44", "1STM"), ([-67.88562749342442, -59.686581826520204, 12.312064099912234, -24.256740140250045],
    [-63.62856226628869, -29.20650770459814, 42.793500076253586, -28.269317582416896])):
        pose_org, pmm, cmd, symdef = setup_test(name="I", file=pdb, return_symmetry_file=True)
        pmm.keep_history(True)

        # TEST6 check that we can put it inside bounds
        pose = pose_org.clone()
        dofspecification = {
            "JUMPHFfold1": {"z": {"limit_movement": True, "min": -5, "max": 5}},
            "JUMPHFfold1_z": {"angle_z": {"limit_movement": True, "min": -5, "max": 5}},
            "JUMPHFfold111": {"x": {"limit_movement": True, "min": -5, "max": 5}},
            "JUMPHFfold111_x": {"angle_x": {"limit_movement": True, "min": -5, "max": 5}},
            "JUMPHFfold111_y": {"angle_y": {"limit_movement": True, "min": -5, "max": 5}},
            "JUMPHFfold111_z": {"angle_z": {"limit_movement": True, "min": -5, "max": 5}},
        }
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        _ = cb.construct_rigidbody_mover(pose)
        perturb_jumpdof(pose, "JUMPHFfold1", 3, 6)
        perturb_jumpdof(pose, "JUMPHFfold1_z", 6, 7)
        perturb_jumpdof(pose, "JUMPHFfold111", 1, 8)
        perturb_jumpdof(pose, "JUMPHFfold111_x", 4, -6)
        perturb_jumpdof(pose, "JUMPHFfold111_y", 5, -6)
        perturb_jumpdof(pose, "JUMPHFfold111_z", 6, -6)
        try:
            cb.all_dofs_within_bounds(pose)
        except AssertionError:
            assert True
        else:
            assert False
        try:
            cb.put_inside_bounds(pose)
            cb.all_dofs_within_bounds(pose)
        except AssertionError:
            assert False
        else:
            assert True


        # TEST5 but put right at the cubic symmetrical borders
        pose = pose_org.clone()
        perturb_jumpdof(pose, "JUMPHFfold1", 3, dof_set[0] + 0.1)
        perturb_jumpdof(pose, "JUMPHFfold1", 6, dof_set[1] + 0.1)
        perturb_jumpdof(pose, "JUMPHFfold111", 1, dof_set[3] + 0.1)
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
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        mover = cb.construct_rigidbody_mover(pose)
        for i in range(500):
            mover.apply(pose)
            cb.all_dofs_within_bounds(pose)
            if i % 100 == 0:
                pmm.apply(pose)

        # TEST4, the mover constructed cant go over the bounds
        dofspecification = {"JUMPHFfold1": {
            "z": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5},
            "angle_z": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5},
        },
            "JUMPHFfold111": {
                "x": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5},
            },
            "JUMPHFfold1111": {
                "angle_x": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5},
                "angle_y": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5},
                "angle_z": {"limit_movement": True, "min": -0.1, "max": 0.1, "param1": 5}
            }
        }
        pose = pose_org.clone()
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        mover = cb.construct_rigidbody_mover(pose)
        for i in range(500):
            mover.apply(pose)
            cb.all_dofs_within_bounds(pose)
            if i % 100 == 0:
                pmm.apply(pose)


        # TEST3, assertions are when we try to construct on a pose that have gone over bounds
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
        pose = pose_org.clone()
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        try:
            pose = pose_org.clone()
            perturb_jumpdof(pose, "JUMPHFfold1", 3, -6)
            cb.construct_rigidbody_mover(pose)
        except AssertionError:
            assert True
        else:
            assert False
        try:
            pose = pose_org.clone()
            perturb_jumpdof(pose, "JUMPHFfold1", 6, 6)
            cb.construct_rigidbody_mover(pose)
        except AssertionError:
            assert True
        else:
            assert False
        try:
            pose = pose_org.clone()
            perturb_jumpdof(pose, "JUMPHFfold111", 1, -7)
            cb.construct_rigidbody_mover(pose)
        except AssertionError:
            assert True
        else:
            assert False
        pose = pose_org.clone()
        cb.construct_rigidbody_mover(pose)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["min"], - 5)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["max"], 5)
        assert math.isclose( cb.current_limits["JUMPHFfold1"]["angle_z"]["min"], -5)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["angle_z"]["max"], 5)
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["min"], - 5)
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["max"], 5)
        # print(cb.current_limits)

        # TEST2 the max/min bounds set are as we expect when according to the cubic limits when we move the pose and construct a new rigidbodymover
        pose = pose_org.clone()
        dofspecification = {"JUMPHFfold1": {
            "z": {"limit_movement": True, "min": -500, "max": 500},
            "angle_z": {"limit_movement": True, "min": -500, "max": 500},
        },
            "JUMPHFfold111": {
                "x": {"limit_movement": True, "min": -500, "max": 500},
            },
            "JUMPHFfold1111": {
                "angle_x": {"limit_movement": True, "min": -5, "max": 5},
                "angle_y": {"limit_movement": True, "min": -5, "max": 5},
                "angle_z": {"limit_movement": True, "min": -5, "max": 5}
            }
        }
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        cb.construct_rigidbody_mover(pose)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["min"], dof_set[0])
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["max"], 500)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["angle_z"]["min"], dof_set[1])
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["angle_z"]["max"], dof_set[2])
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["min"], dof_set[3])
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["max"], 500)

        # TEST 1
        # the max/min bounds set are as we expect when we move the pose and construct a new rigidbodymover
        pose = pose_org.clone()
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
        cb = CubicBoundary(symdef=symdef, pose_at_initial_position=pose, dofspecification=dofspecification)
        cb.construct_rigidbody_mover(pose)
        #pmm.apply(pose)
        #pmm.keep_history(True)
        # print(cb.current_limits)
        perturb_jumpdof(pose, "JUMPHFfold1", 3, -4)
        perturb_jumpdof(pose, "JUMPHFfold1", 6, 3)
        perturb_jumpdof(pose, "JUMPHFfold111", 1, -2)
        cb.construct_rigidbody_mover(pose)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["min"], - 5 + 4)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["z"]["max"], 5 + 4)
        assert math.isclose( cb.current_limits["JUMPHFfold1"]["angle_z"]["min"], -5 - 3)
        assert math.isclose(cb.current_limits["JUMPHFfold1"]["angle_z"]["max"], 5 - 3)
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["min"], - 5 + 2)
        assert math.isclose(cb.current_limits["JUMPHFfold111"]["x"]["max"], 5 + 2)
        # print(cb.current_limits)




  # def get_current_angle_z(pose_clone):
    #     return pose_clone.constraint_set().get_all_constraints()[2].angle(
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold1_z"))).atom(2).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold1_z"))).atom(1).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold1_z"))).atom(2).xyz())
    #
    # def get_current_com_angle_x(pose_clone):
    #     return pose_clone.constraint_set().get_all_constraints()[4].angle(
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_x"))).atom(3).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_x"))).atom(1).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_x"))).atom(3).xyz())
    #
    # def get_current_com_angle_y(pose_clone):
    #     return pose_clone.constraint_set().get_all_constraints()[5].angle(
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_y"))).atom(2).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_y"))).atom(1).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_y"))).atom(2).xyz())
    #
    # def get_current_com_angle_z(pose_clone):
    #     return pose_clone.constraint_set().get_all_constraints()[6].angle(
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_z"))).atom(2).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_z"))).atom(1).xyz(),
    #         pose_clone.residue(pose_clone.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, f"JUMPHFfold111_z"))).atom(2).xyz())
    #

# def calculate_bound_penalty(x, sd_, ub_, lb_, rswitch_=0.5):
#     if x > ub_:
#         delta = x - ub_
#     elif lb_ <= x:
#             delta = 0
#     elif x < lb_:
#         delta = lb_ - x
#     else:
#         delta = 0
#     delta/=sd_
#     if x > ub_ + rswitch_*sd_:
#         return 2 * rswitch_ * delta - rswitch_ * rswitch_
#     else:
#         return delta * delta

# def create_dihedral_constraint(pose):
#     func = cb.create_bounded_func(-5, 5, 1)
#     return DihedralConstraint(
#         AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))),
#         AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))),
#         AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))),
#         AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))),
#         func)
#
# def calculate_angle_z(pose):
#     from pyrosetta.rosetta.numeric import dihedral_radians
#     import numpy as np
#     b = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     c = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     a = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     d = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     return dihedral_radians(b,c,a,d) * 180 / math.pi

# def calculate_angle_z2(pose):
#     from pyrosetta.rosetta.numeric import dihedral_radians
#     import numpy as np
#     a = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     b = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     c = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     d = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     print(np.array(a), np.array(b), np.array(c), np.array(d))
#     return dihedral_radians(a, b, c, d) * 180 / math.pi
#
# def calculate_all_combos(pose):
#
# def calculate_all_combos(pose):
#     from pyrosetta.rosetta.numeric import dihedral_radians
#     import numpy as np
#     import itertools
#     a = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     b = pose.residue(pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     c = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     d = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     #
#
#
#
#
#     # e = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     # f = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     # g = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(1).xyz()
#     # h = pose.residue(pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold1_z"))).atom(2).xyz()
#     for comb in itertools.permutations(((a, "a"), (b, "b"), (c, "c"), (d, "d")), 4):
#         print([i[1] for i in comb], "=", dihedral_radians(*[i[0] for i in comb]) * 180 / math.pi)


# for i, x, sd in zip((a, b, c, d, e, f), (position_info), (sd, sd, sd * 1 / (180 / math.pi),
# sd * 1 / (180 / math.pi), sd * 1 / (180 / math.pi), sd * 1 / (180 / math.pi))):

# total_penalty = 0
# all_current_pos = get_position_info(pose_clone, dictionary=True)
# for jump_name, jumpdof_params in dofspecification.items():
#     for dof_name, dof_params in jumpdof_params.items():
#         if dofspecification[jump_name][dof_name].get("limit_movement", False):
#             x = all_current_pos[jump_name][dof_name]
#             in_rad = True if "angle" in dof_name else False
#             if in_rad:
#                 x *= math.pi / 180
#             lb_, ub_ = cb.get_boundary(jump_name, dof_name,  in_rad=in_rad) #cb.boundaries[jump_name][dof_name]["min"]
#             # ub_ = cb.boundaries[jump_name][dof_name]["max"]
#             sd_ = sd #dofspecification[jump_name][dof_name]["sd"]
#             total_penalty += calculate_bound_penalty(x, sd_, ub_, lb_)
