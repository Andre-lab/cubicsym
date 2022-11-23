#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference based kinematics associated functions and variables
@Author: Mads Jeppesen
@Date: 9/29/22
"""
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int
import random

default_dofs = {
    "JUMPHFfold1": {"z": {"param1": 0.5}},
    "JUMPHFfold1_z": {"angle_z": {"param1": 0.5}},
    "JUMPHFfold111": {"x": {"param1": 0.5}},
    "JUMPHFfold111_x": {"angle_x": {"param1": 0.5}},
    "JUMPHFfold111_y": {"angle_y": {"param1": 0.5}},
    "JUMPHFfold111_z": {"angle_z": {"param1": 0.5}},
}
default_dofs_with_cubic_limits = {
    "JUMPHFfold1": {"z": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold1_z": {"angle_z": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold111": {"x": {"param1": 0.5, "limit_movement": True}},
    "JUMPHFfold111_x": {"angle_x": {"param1": 0.5}},
    "JUMPHFfold111_y": {"angle_y": {"param1": 0.5}},
    "JUMPHFfold111_z": {"angle_z": {"param1": 0.5}},
}

def randomize_all_dofs(pose, fold1=50, fold1_z=180, fold111=50, fold111_x=180, fold111_y=180, fold111_z=180, return_vals=False):
    """Randomizes all the dofs in the given range of +/- the passed value. It picks a value uniformly."""
    a = random.uniform(-fold1, fold1)
    b = random.uniform(-fold1_z, fold1_z)
    c = random.uniform(-fold111, fold111)
    d = random.uniform(-fold111_x, fold111_x)
    e = random.uniform(-fold111_y, fold111_y)
    f = random.uniform(-fold111_z, fold111_z)
    perturb_jumpdof_str_int(pose, "JUMPHFfold1", 3, a)
    perturb_jumpdof_str_int(pose, "JUMPHFfold1_z", 6, b)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111", 1, c)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_x", 4, d)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_y", 5, e)
    perturb_jumpdof_str_int(pose, "JUMPHFfold111_z", 6, f)
    if return_vals:
        return a,b,c,d,e,f
