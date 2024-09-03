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
    """Searches the hypotenuse c and the a and b independently of a triangle

    The vectors are defined below in a capsid.

    A: Global center
    B: 5-fold center
    C: Ca closest to the COM of the subunit

           b ->
         B#####C
      ^  #####   ^
      |  ###    /
      a  ##    c
         A
    """

    def __init__(self, pose):
        self.jumpids = list(dict(pose.conformation().Symmetry_Info().get_dofs().items()).keys())
        self.symdofs = list(dict(pose.conformation().Symmetry_Info().get_dofs().items()).values())
        self.jumpnames = list(sym_dof_names(pose))
        self.a, self.b, self.c = self.get_a(pose), self.get_b(pose), self.get_c(pose)

    # # fixme. I think this assumes the self.c vector is perpindicular to the triangular face but it shouldnt be since it is
    # #  connected to the com of the subunit. Instead use the 5-fold centers
    # def triangular_rotation_matrix(self):
    #     """Creates a rotation matrix that rotates the triangular face onto the x-y plane"""
    #     # find the angle to rotate by. This is the rotation angle that will put the triangular face onto the xy-plane
    #     angle = 90 - vector_angle(self.a, self.c)
    #     # find the rotation point. This is a vector from the global center to the CA closest to COM
    #     rotation_point = self.c
    #     # create the rotation matrix that will put the triangular face onto the xy-plane
    #     rot = rotation_matrix(np.cross(self.b, self.a), angle)
    #     return rotation_point, rot

    ############
    # Getters  #
    ############

    def get_a(self, pose):
        """Gets a."""
        return self.__get_translation_from_jumpname(pose, "JUMPHFfold1")

    def get_b(self, pose):
        """Gets b"""
        return self.__get_translation_from_jumpname(pose, "JUMPHFfold111")

    def get_c(self, pose):
        """Gets c"""
        return self.get_a(pose) + self.get_b(pose)

    def get_a_size(self):
        return np.linalg.norm(self.a)

    def get_b_size(self):
        return np.linalg.norm(self.b)

    def get_c_size(self):
        return np.linalg.norm(self.c)

    ############
    # Setters  #
    ############

    def set_a_to_exact(self, pose, a_new):
        """Sets a exactly"""
        self.a = a_new
        self.__set_jump(pose, "JUMPHFfold1", a_new)

    def set_b_to_exact(self, pose, b_new):
        """Sets b exactly"""
        self.b = b_new
        self.__set_jump(pose, "JUMPHFfold111", b_new)

    def set_c_to_exact(self, pose, c_new):
        """Sets c exactly"""
        self.c = c_new
        self.set_a_to_exact(pose, vector_projection(c_new, self.a))
        self.set_b_to_exact(pose, vector_projection(c_new, self.b))

    def update_a_from_pose(self, pose):
        jump_id = sym_dof_jump_num(pose, "JUMPHFfold1")
        flexible_jump = pose.jump(jump_id)
        self.a = np.array(flexible_jump.get_translation())

    def update_b_from_pose(self, pose):
        jump_id = sym_dof_jump_num(pose, "JUMPHFfold111")
        flexible_jump = pose.jump(jump_id)
        self.b = np.array(flexible_jump.get_translation())

    def update_c_from_pose(self, pose):
        self.update_a_from_pose(pose)
        self.update_b_from_pose(pose)
        self.c = self.a + self.b

    ##############
    # Operators  #
    ##############

    def mul_c(self, pose, value):
        """Moves a along the axis (*value)"""
        self.update_c_from_pose(pose)
        self.set_c_to_exact(pose, self.c * value)

    def add_a(self, pose, value):
        """Moves a along the axis (+value)"""
        self.update_a_from_pose(pose)
        self.set_a_to_exact(pose, self.a + self.__add_along_vector(self.a, value))

    def add_b(self, pose, value):
        """Moves b along the axis (+value)"""
        self.update_b_from_pose(pose)
        self.set_b_to_exact(pose, self.b + self.__add_along_vector(self.b, value))

    def add_c(self, pose, value):
        """Moves c along the axis (+value)"""
        self.update_c_from_pose(pose)
        self.set_c_to_exact(pose, self.c + self.__add_along_vector(self.c, value))

    ##########################################
    # Deep into the code here - GO BACK !!!  #
    ##########################################

    def __get_translation_from_jumpname(self, pose, jump_name):
        """Sets the translation with value trans at dof with jump_name in the pose"""
        jumpid = self.jumpids[self.jumpnames.index(jump_name)]
        return np.array(pose.jump(jumpid).get_translation())

    def __add_along_vector(self, v, trans):
        return (v / np.linalg.norm(v)) * trans

    def __set_jump(self, pose, jump_name, new_trans):
        """Sets the translation with value trans at dof with jump_name in the pose"""
        jumpid = self.jumpids[self.jumpnames.index(jump_name)]
        flexible_jump = pose.jump(jumpid)
        flexible_jump.set_translation(xyzVector_double_t(*new_trans))
        pose.set_jump(jumpid, flexible_jump)


