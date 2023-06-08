#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference based kinematics associated functions and variables
@Author: Mads Jeppesen
@Date: 9/29/22
"""
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
from cubicsym.cubicsetup import CubicSetup
import random

default_HF_dofs = {
    "JUMPHFfold1": {"z": {"param1": 0.5}},
    "JUMPHFfold1_z": {"angle_z": {"param1": 0.5}},
    "JUMPHFfold111": {"x": {"param1": 0.5}},
    "JUMPHFfold111_x": {"angle_x": {"param1": 0.5}},
    "JUMPHFfold111_y": {"angle_y": {"param1": 0.5}},
    "JUMPHFfold111_z": {"angle_z": {"param1": 0.5}},
}
default_HF_dofs_with_cubic_limits = {
    "JUMPHFfold1": {"z": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold1_z": {"angle_z": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold111": {"x": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold111_x": {"angle_x": {"param1": 0.5}},
    "JUMPHFfold111_y": {"angle_y": {"param1": 0.5}},
    "JUMPHFfold111_z": {"angle_z": {"param1": 0.5}},
}

def get_dofspecification_for_pose(pose):
    """Returns a dofspecification taking into account the symmetry of the pose"""
    jid = CubicSetup.get_jumpidentifier_from_pose(pose)
    return {
        f"JUMP{jid}fold1": {"z": {}},
        f"JUMP{jid}fold1_z": {"angle_z": {}},
        f"JUMP{jid}fold111": {"x": {}},
        f"JUMP{jid}fold111_x": {"angle_x": {}},
        f"JUMP{jid}fold111_y": {"angle_y": {}},
        f"JUMP{jid}fold111_z": {"angle_z": {}},
    }

def translate_away(pose, z=50, x=25):
    """Translate the cubic symmetrical pose away from the center."""
    jid = CubicSetup.get_jumpidentifier_from_pose(pose)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold1", 3, z)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111", 1, x)

def randomize_all_dofs_positive_trans(pose, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=180, fold111_z=180, return_vals=False):
    """Randomizes all the dofs in the given range of +/- the passed value if the dof is a rotation else 0-value. It picks a value uniformly."""
    a = random.uniform(0, fold1)
    b = random.uniform(-fold1_z, fold1_z)
    c = random.uniform(0, fold111)
    d = random.uniform(-fold111_x, fold111_x)
    e = random.uniform(-fold111_y, fold111_y)
    f = random.uniform(-fold111_z, fold111_z)
    jid = CubicSetup.get_jumpidentifier_from_pose(pose)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold1", 3, a)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold1_z", 6, b)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111", 1, c)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_x", 4, d)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_y", 5, e)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_z", 6, f)
    if return_vals:
        return a,b,c,d,e,f

def randomize_all_dofs(pose, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=180, fold111_z=180, return_vals=False):
    """Randomizes all the dofs in the given range of +/- the passed value. It picks a value uniformly."""
    a = random.uniform(-fold1, fold1)
    b = random.uniform(-fold1_z, fold1_z)
    c = random.uniform(-fold111, fold111)
    d = random.uniform(-fold111_x, fold111_x)
    e = random.uniform(-fold111_y, fold111_y)
    f = random.uniform(-fold111_z, fold111_z)
    jid = CubicSetup.get_jumpidentifier_from_pose(pose)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold1", 3, a)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold1_z", 6, b)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111", 1, c)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_x", 4, d)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_y", 5, e)
    perturb_jumpdof_str_int(pose, f"JUMP{jid}fold111_z", 6, f)
    if return_vals:
        return a,b,c,d,e,f
