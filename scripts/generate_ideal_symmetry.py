#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates ideal symmetry from a given symmetry file
@Author: Mads Jeppesen
@Date: 12/13/22
"""

from cubicsym.cubicsetup import CubicSetup
from symmetryhandler.mathfunctions import vector_angle, vector_projection_on_subspace, rotation_matrix
import math
import scipy
import numpy as np








# cs.get_downstream_connections("JUMP2fold1_z")
# # use the same angle to rotate all vrts that amount instead
# # VRTS to not transform:
# nottotrans = ['VRTglobal', 'VRTHFfold1_z_tref', 'VRTHFfold','VRTHFfold1','VRTHFfold1_z_rref',
#               'VRT3fold1_z_tref', 'VRT3fold','VRT3fold1','VRT3fold1_z_rref',
#               'VRT2fold1_z_tref', 'VRT2fold','VRT2fold1','VRT2fold1_z_rref'],
# R = rotation_matrix([0,0,1], -angle_to_use)
# for vrt in cs._vrts:
#     vrt.rotate(R)
# cs.reset_all_dofs()
# cs.anchor = "COM"
# # cs.visualize(ip="10.8.0.6")
# cs.output("../tests/outputs/test.symm")
#
# cs = CubicSetup()
# cs.read_from_file("../tests/outputs/test.symm")
# cs.apply_dofs()
# cs.visualize(ip="10.8.0.6")


