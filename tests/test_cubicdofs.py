#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the CubicDof class
@Author: Mads Jeppesen
@Date: 9/26/22
"""

def test_cubic_dofs():
    from cubicsym.cubicdofs import CubicDofs
    from simpletestlib.test import setup_test
    pose = setup_test(name="I", file="1STM", pymol=False)
    cd = CubicDofs(pose)
    assert cd.__doforder_str == {'HF': [('JUMPHFfold1', 3),
                                        ('JUMPHFfold1_z', 6),
                                        ('JUMPHFfold111', 1),
                                        ('JUMPHFfold111_x', 4),
                                        ('JUMPHFfold111_y', 5),
                                        ('JUMPHFfold111_z', 6)],

    '3F': [('JUMP31fold1', 3),
  ('JUMP31fold1_z', 6),
  ('JUMP31fold111', 1),
  ('JUMP31fold111_x', 4),
  ('JUMP31fold111_y', 5),
  ('JUMP31fold111_z', 6)],

                                 '2F': [('JUMP21fold1', 3),
  ('JUMP21fold1_z', 6),
  ('JUMP21fold111', 1),
  ('JUMP21fold111_x', 4),
  ('JUMP21fold111_y', 5),
  ('JUMP21fold111_z', 6)]}

    assert cd.get_positions_as_dict(pose) == {'JUMPHFfold1': {3: 63.628574325542075},
         'JUMPHFfold1_z': {6: 0.0},
         'JUMPHFfold111': {1: 28.26931756062931},
         'JUMPHFfold111_x': {4: 0.0},
         'JUMPHFfold111_y': {5: 0.0},
         'JUMPHFfold111_z': {6: 0.0}}