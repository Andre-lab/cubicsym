#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HypotenuseMover class
@Author: Mads Jeppesen
@Date: 11/29/21
"""
import numpy as np
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_names, sym_dof_jump_num
from symmetryhandler.mathfunctions import rotation_matrix, vector_angle, vector_projection
from symmetryhandler.reference_kinematics import get_x_translation, get_z_translation
from enum import Enum
from pyrosetta.rosetta.numeric import xyzVector_double_t

class DOFS(Enum):
    X_DOF = 1
    Y_DOF = 2
    Z_DOF = 3
    X_ANGLE_DOF = 4
    Y_ANGLE_DOF = 5
    Z_ANGLE_DOF = 6

class HypotenuseMover:
    """Searches the hypotenuse c and the a and b independently of a triangle."""

    def __init__(self, pose):
        self.jumpids = list(dict(pose.conformation().Symmetry_Info().get_dofs().items()).keys())
        self.symdofs = list(dict(pose.conformation().Symmetry_Info().get_dofs().items()).values())
        self.jumpnames = list(sym_dof_names(pose))
        self.a, self.b, self.c = self.get_vectors()

    def get_vectors(self):
        """The vectors are defined below in a capsid.

        A: Global center
        B: VRT5fold1 (5fold center)
        C: VRT5fold1111 (Ca closest to the COM of the subunit)

               b ->
             B#####C
          ^  #####   ^
          |  ###    /
          a  ##    c
             A
        """
        # vector from the global center to the 5fold center
        a = np.array([0, 0, self.symdofs[self.jumpnames.index("JUMPHFfold1")].range1_lower(3)])
        # vector from the 5fold center to the CA closest to COM of the subunit (as defined by the symmetry script)
        b = np.array([self.symdofs[self.jumpnames.index("JUMPHFfold111")].range1_lower(1), 0, 0])
        # vector from the CA closest to COM to the global center (basically the hypotenuse of the triangle we now have)
        c =  b + a
        return a, b, c

    def set_jump(self, pose, jump_name, new_trans):
        """Sets the translation with value trans at dof with jump_name in the pose"""
        jumpid = self.jumpids[self.jumpnames.index(jump_name)]
        flexible_jump = pose.jump(jumpid)
        flexible_jump.set_translation(xyzVector_double_t(*new_trans))
        pose.set_jump(jumpid, flexible_jump)

    def update_a(self, pose):
        jump_id = sym_dof_jump_num(pose, "JUMPHFfold1")
        flexible_jump = pose.jump(jump_id)
        self.a = np.array(get_z_translation(flexible_jump))

    def update_b(self, pose):
        jump_id = sym_dof_jump_num(pose, "JUMPHFfold111")
        flexible_jump = pose.jump(jump_id)
        self.b = np.array(get_x_translation(flexible_jump))

    def update_c(self, pose):
        self.update_a(pose)
        self.update_b(pose)
        self.c = self.a + self.b

    def add_along_vector(self, v, trans):
        return (v / np.linalg.norm(v)) * trans

    def get_a_size(self):
        return np.linalg.norm(self.a)

    def get_b_size(self):
        return np.linalg.norm(self.b)

    def get_c_size(self):
        return np.linalg.norm(self.c)

    def mul_c(self, pose, trans):
        """Moves a along the axis (*trans)"""
        self.update_c(pose)
        self.set_c(pose, self.c * trans)

    def add_a(self, pose, trans):
        """Moves a along the axis (+trans)"""
        self.update_a(pose)
        self.set_a(pose, self.a + self.add_along_vector(self.a, trans))

    def add_b(self, pose, trans):
        """Moves b along the axis (+trans)"""
        self.update_b(pose)
        self.set_b(pose, self.b + self.add_along_vector(self.b, trans))

    def add_c(self, pose, trans):
        """Moves c along the axis (+trans)"""
        self.update_c(pose)
        self.set_c(pose, self.c + self.add_along_vector(self.c, trans))

    def set_a(self, pose, a_new):
        """Sets a exactly"""
        self.a = a_new
        self.set_jump(pose, "JUMPHFfold1", a_new)

    def set_b(self, pose, b_new):
        """Sets b exactly"""
        self.b = b_new
        self.set_jump(pose, "JUMPHFfold111", b_new)

    def set_c(self, pose, c_new):
        """Sets c exactly"""
        self.c = c_new
        self.set_a(pose, vector_projection(c_new, self.a))
        self.set_b(pose,  vector_projection(c_new, self.b))

    # fixme. I think this assumes the self.c vector is perpindicular to the triangular face but it shouldnt be since it is
    #  connected to the com of the subunit. Instead use the 5-fold centers
    def triangular_rotation_matrix(self):
        """Creates a rotation matrix that rotates the triangular face onto the x-y plane"""
        # find the angle to rotate by. This is the rotation angle that will put the triangular face onto the xy-plance
        angle = 90 - vector_angle(self.a, self.c)
        # find the rotation point. This is a vector from the global center to the CA closest to COM
        rotation_point = self.c
        # create the rotation matrix that will put the triangular face onto the xy-plance
        rot = rotation_matrix(np.cross(self.b, self.a), angle)
        return rotation_point, rot
