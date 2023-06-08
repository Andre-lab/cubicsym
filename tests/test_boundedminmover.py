#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the BoundedMinMover class
@Author: Mads Jeppesen
@Date: 10/10/22
"""

def test_boundedminmover():
    from cubicsym.actors.boundedminmover import BoundedMinMover
    from cubicsym.actors.cubicboundary import CubicBoundary
    from cubicsym.dofspec import DofSpec
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from simpletestlib.test import setup_test
    import math
    from symmetryhandler.reference_kinematics import perturb_jumpdof_str_str
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, mute=True)
    sfxn = ScoreFunctionFactory.create_score_function("ref2015")
    dofspec = DofSpec(pose)
    dofspec.set_symmetrical_bounds([5]*dofspec.dofsize)
    cb = CubicBoundary(symdef, pose, dof_spec=dofspec)
    bmm = BoundedMinMover(cb, sfxn)
    # try within the bounce. We want the minization to pass and then the pose to change its dofpositions
    pose_mod = pose.clone()
    perturb_jumpdof_str_str(pose_mod, "JUMPHFfold1", "z", 1)
    pose_mod_in = pose_mod.clone()
    bmm.apply(pose_mod_in)
    assert bmm.passed
    assert all(not math.isclose(i, j) for i, j in zip(cb.dof_spec.get_positions_as_list(pose_mod_in),
                                                      cb.dof_spec.get_positions_as_list(pose_mod)))
    # try outside of bounce. We want the minization to NOT pass and then the pose to have identical dofpositions
    pose_mod = pose.clone()
    perturb_jumpdof_str_str(pose_mod, "JUMPHFfold1", "z", -10)
    pose_mod_in = pose_mod.clone()
    bmm.apply(pose_mod_in)
    assert not bmm.passed
    assert all(math.isclose(i, j) for i, j in zip(cb.dof_spec.get_positions_as_list(pose_mod_in),
                                                  cb.dof_spec.get_positions_as_list(pose_mod)))
