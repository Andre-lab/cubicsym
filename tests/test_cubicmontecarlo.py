#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen 
@Date: 2/16/23 
"""


def test_load_many_cubicmontecarlo():
    from cubicsym.cubicmontecarlo import CubicMonteCarlo
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.dofspec import DofSpec
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.setup import setup_test
    from cubicsym.utilities import add_id_to_pose_w_base
    mcs = []
    pose, pmm, cmd, symdef = setup_test(name="T", file="2CC9", mute=True, return_symmetry_file=True, symmetrize=True)
    cubicsetup = CubicSetup(symdef=symdef)
    ccs = CloudContactScore(pose, cubicsetup)
    cmc = CubicMonteCarlo(scorefunction=ccs, dofspec=DofSpec(pose))
    for i in range(300):
        mcs.append()
