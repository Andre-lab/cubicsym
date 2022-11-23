#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[Documentation]
@Author: Mads Jeppesen
@Date: 9/26/22
"""

def test_cubic_dofs():
    from cubicsym.cubicdofs import CubicDofs
    from simpletestlib.test import setup_test
    from symmetryhandler.kinematics import default_dofs
    pose = setup_test(name="I", file="1STM", pymol=False)
    cd = CubicDofs(pose, default_dofs)
    assert cd.doforder_str == [('JUMPHFfold1', 'z'),
     ('JUMPHFfold1_z', 'angle_z'),
     ('JUMPHFfold111', 'x'),
     ('JUMPHFfold111_x', 'angle_x'),
     ('JUMPHFfold111_y', 'angle_y'),
     ('JUMPHFfold111_z', 'angle_z')]
    assert cd.get_positions_as_dict(pose) == {'JUMPHFfold1': {'z': 63.62856226628869},
     'JUMPHFfold1_z': {'angle_z': 0.0},
     'JUMPHFfold111': {'x': 28.269317582416896},
     'JUMPHFfold111_x': {'angle_x': 0.0},
     'JUMPHFfold111_y': {'angle_y': 0.0},
     'JUMPHFfold111_z': {'angle_z': 0.0}}