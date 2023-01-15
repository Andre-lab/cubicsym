#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 7/14/22
"""
import numpy as np
from cubicsym.cubicsetup import CubicSetup
from symmetryhandler.mathfunctions import vector_projection_on_subspace, vector_angle, rotate_right_multiply, rotation_matrix_from_vector_to_vector, rotate
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from symmetryhandler.reference_kinematics import set_jumpdof_str_int
import math
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from pyrosetta.rosetta.core.kinematics import Stub, Jump
from cubicsym.utilities import get_chain_map as util_get_chain_map

class SymDefSwapper:

    def __init__(self, pose, symmetry_file, visualier=None):
        # store information we need in order to transfer
        assert is_symmetric(pose)
        # todo: get_anchor_resi is/can be static
        self.foldHF_setup = CubicSetup(symmetry_file)
        self.is_righthanded = self.foldHF_setup.calculate_if_rightanded()
        self.symmetry_type = self.foldHF_setup.cubic_symmetry_from_setup()
        # self.symmdata = self.init_setup.get_symmdata()
        # self.virutal_coords = {k:v for k,v in self.symmdata.get_virtual_coordinates().items()}
        self.monomer_sizes = pose.size()
        if self.symmetry_type == "I":
            self.fold3_setup = self.foldHF_setup.create_I_3fold_based_symmetry()
            self.fold3_setup.apply_dofs()
            self.fold2_setup = self.foldHF_setup.create_I_2fold_based_symmetry()
            self.fold2_setup.apply_dofs()
        elif self.symmetry_type == "O":
            self.fold3_setup = self.foldHF_setup.create_O_3fold_based_symmetry()
            self.fold3_setup.apply_dofs()
            self.fold2_setup = self.foldHF_setup.create_O_2fold_based_symmetry()
            self.fold2_setup.apply_dofs()
        else:
            assert self.symmetry_type == "T"
            self.fold3_setup = self.foldHF_setup.create_T_3fold_based_symmetry()
            self.fold3_setup.apply_dofs()
            self.fold2_setup = self.foldHF_setup.create_T_2fold_based_symmetry()
            self.fold2_setup.apply_dofs()
        # self.anchor_resi = self.init_setup.get_anchor_residue(pose)
        # self.anchor_resi = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())
        # APPLY THIS AFTERWARDS OR ELSE THE SYMMETRY WILL BE MESSED UP
        self.foldHF_setup.apply_dofs()

        # we also need the angle the initial angle between the 5 and 3-fold
        self.init_x_vectors()
        self.fold2_plane = self.get_2fold_plane()
        self.fold3_plane = self.get_3fold_plane()
        self.foldHF_plane = self.get_HFfold_plane()
        self.visualizer = visualier

    def get_chain_map(self):
        """Returns a list of the mapping between each chain (in Rosetta numbering) for I/O: HF-, 3- and 2-fold or for T: HF- and 2-fold."""
        util_get_chain_map(self.symmetry_type, self.is_righthanded)

    def init_x_vectors(self):
        self.init_HFfold_x_vec = self.get_HFfold_x_vec_from_HFfold()
        self.init_3fold_x_vec = self.get_3fold_x_vec_from_HFfold()
        self.init_2fold_x_vec = self.get_2fold_x_vec_from_HFfold()

    def set_rotation_quick(self, pose1, pose2, jumpstr1, jumpstr2, jumptoset):
        """Sets the jump of jumptoset based on the downstream stubs of jumpstr1 and jumpstr2. The jump is set on pose1."""
        stub1 = Stub(pose1.conformation().upstream_jump_stub(sym_dof_jump_num(pose1, jumpstr1))) # "JUMP31fold1111"
        stub2 = Stub(pose2.conformation().upstream_jump_stub(sym_dof_jump_num(pose2, jumpstr2))) # "JUMPHFfold1111"
        pose1.set_jump(sym_dof_jump_num(pose1, jumptoset), Jump(stub1, stub2))

    def get_HFfold_plane(self):
        """Get the plane of the 5-fold axis spann5d by 2 vectors as rows in a matrix 2x3 matrix."""
        return np.array([[1,0,0], [0,1,0]])

    def get_2fold_plane(self):
        """Get the plane of the 2-fold axis spanned by 2 vectors as rows in a matrix 2x3 matrix."""
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        R = rotation_matrix_from_vector_to_vector(- self.fold2_setup.get_vrt("VRT21fold1").vrt_z, [0, 0, 1])
        x = rotate_right_multiply(x, R)
        y = rotate_right_multiply(y, R)
        assert np.isclose(np.cross(x, y), self.get_2fold_center_from_3fold() / np.linalg.norm(self.get_2fold_center_from_3fold())).all()
        return np.vstack((x, y))

    # todo: could be done better in the future
    def get_3fold_plane(self):
        """Get the plane of the 3-fold axis spanned by 2 vectors as rows in a matrix 2x3 matrix."""
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        R = rotation_matrix_from_vector_to_vector([0, 0, 1], - self.fold3_setup.get_vrt("VRT31fold1").vrt_z)
        x = rotate(x, R)
        y = rotate(y, R)
        assert np.isclose(np.cross(x, y), self.get_3fold_center_from_HFfold() / np.linalg.norm(self.get_3fold_center_from_HFfold())).all()
        return np.vstack((x, y))

    # def create_3fold_pose_from_THFfold(self, pose4, transfer=True):
    #     """Creates a symmetric pose based on the 3-fold axis."""
    #     pose3 = self.foldHF_setup.make_asymmetric_pose(pose4, dont_reset=["JUMPHFfold111_subunit"])
    #     # As we reuse this function the subunit dofs might change and we want to keep them fixed
    #     self.fold3_setup.reset_jumpdofs("JUMP31fold111_subunit")
    #     self.fold3_setup.make_symmetric_pose(pose3)
    #     if transfer:
    #         self.transfer_THFto3(pose4, pose3)
    #     return pose3
    #
    # def create_3fold_pose_from_4fold(self, pose4, transfer=True):
    #     """Creates a symmetric pose based on the 3-fold axis."""
    #     pose3 = self.foldHF_setup.make_asymmetric_pose(pose4, dont_reset=["JUMPHFfold111_subunit"])
    #     # As we reuse this function the subunit dofs might change and we want to keep them fixed
    #     self.fold3_setup.reset_jumpdofs("JUMP31fold111_subunit")
    #     self.fold3_setup.make_symmetric_pose(pose3)
    #     if transfer:
    #         self.transfer_O4to3(pose4, pose3)
    #     return pose3

    def create_3fold_pose_from_HFfold(self, poseHF, transfer=True):
        """Creates a symmetric pose based on the 3-fold axis."""
        pose3 = self.foldHF_setup.make_asymmetric_pose(poseHF, dont_reset=["JUMPHFfold111_subunit"])
        # As we reuse this function the subunit dofs might change and we want to keep them fixed
        # self.fold3_setup.reset_jumpdofs("JUMP31fold111_subunit")
        self.fold3_setup.make_symmetric_pose(pose3)
        if transfer:
            if self.symmetry_type == "I":
                self.transfer_IHFto3(poseHF, pose3)
            elif self.symmetry_type == "O":
                self.__transfer_OHFto3(poseHF, pose3)
            elif self.symmetry_type == "T":
                self.__transfer_THFto3(poseHF, pose3)
        return pose3

    def create_2fold_pose_from_HFfold(self, pose5, transfer=True):
        """Creates a symmetric pose based on the 2-fold axis."""
        pose2 = self.foldHF_setup.make_asymmetric_pose(pose5, dont_reset=["JUMPHFfold111_subunit"])
        # As we reuse this function the subunit dofs might change and we want to keep them fixed
        # self.fold2_setup.reset_jumpdofs("JUMP21fold111_subunit")
        self.fold2_setup.make_symmetric_pose(pose2)
        if transfer:
            self.transfer_HFto2(pose5, pose2)
        return pose2

    def get_HFfold_x_vec_from_HFfold(self):
        a = np.array(self.foldHF_setup.get_vrt("VRTHFfold11").vrt_orig)
        b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111").vrt_orig)
        return b - a

    def get_3fold_x_vec_from_4fold(self):
        a = self.get_3fold_center_from_HFfold()
        b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
        return b - a

    def get_3fold_x_vec_from_HFfold(self):
        a = self.get_3fold_center_from_HFfold()
        b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
        return b - a

    def get_2fold_x_vec_from_3fold(self):
        a = self.get_2fold_center_from_3fold()
        b = np.array(self.fold3_setup.get_vrt("VRT31fold111_z").vrt_orig)
        return b - a

    def get_THFfold_x_vec_from_2fold(self):
        a = self.get_THF_center_from_2fold()
        b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        return b - a

    def get_4fold_x_vec_from_2fold(self):
        a = self.get_4fold_center_from_2fold()
        b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        return b - a

    def get_5fold_x_vec_from_2fold(self):
        a = self.get_5fold_center_from_2fold()
        b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        return b - a

    def get_2fold_x_vec_from_HFfold(self):
        a = self.get_2fold_center_from_HFfold()
        b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
        return b - a

    # fixme these functions can just work on the chain names can be made to be
    def get_3fold_center_from_HFfold(self):
        # todo calculation in rosetta and finally make array
        a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
        b = np.array(self.foldHF_setup.get_vrt("VRT2fold121_z").vrt_orig)
        c = np.array(self.foldHF_setup.get_vrt("VRT3fold111_z").vrt_orig)
        return (a + b + c) / 3

    # def get_3fold_center_from_4fold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     b = np.array(self.foldHF_setup.get_vrt("VRT2fold121_z").vrt_orig)
    #     c = np.array(self.foldHF_setup.get_vrt("VRT3fold111_z").vrt_orig)
    #     return (a + b + c) / 3

    def get_2fold_center_from_3fold(self):
        # todo calculation in rosetta and finally make array
        a = np.array(self.fold3_setup.get_vrt("VRT31fold111_z").vrt_orig)
        if self.symmetry_type in ("I"):
            if self.is_righthanded: # 1stm
                b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
            else: # 6S44
                b = np.array(self.fold3_setup.get_vrt("VRT35fold131_z").vrt_orig)
        elif self.symmetry_type == "O":
            if self.is_righthanded: # 1AEW
                b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
            else: # 1PY3
                b = np.array(self.fold3_setup.get_vrt("VRT33fold131_z").vrt_orig)
            # raise NotImplementedError
        elif self.symmetry_type == "T":
            if self.is_righthanded: # 1HOS
                b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
            else: # 1MOG
                b = np.array(self.fold3_setup.get_vrt("VRT33fold131_z").vrt_orig)
            # raise NotImplementedError
        return (a + b) / 2

    def get_THF_center_from_2fold(self):
        # todo calculation in rosetta and finally make array
        a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        b = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
        c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
        return (a + b + c) / 3

    def get_4fold_center_from_2fold(self):
        # todo calculation in rosetta and finally make array
        a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        b = np.array(self.fold2_setup.get_vrt("VRT24fold111_z").vrt_orig)
        c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
        d = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
        return (a + b + c + d) / 4

    def get_5fold_center_from_2fold(self):
        # todo calculation in rosetta and finally make array
        a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
        b = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
        c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
        d = np.array(self.fold2_setup.get_vrt("VRT24fold111_z").vrt_orig)
        e = np.array(self.fold2_setup.get_vrt("VRT25fold111_z").vrt_orig)
        return (a + b + c + d + e) / 5

    def get_2fold_center_from_HFfold(self):
        a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
        b = np.array(self.foldHF_setup.get_vrt("VRT2fold111_z").vrt_orig)
        return (a + b) / 2

    def get_3_fold_angle_z_from_4fold(self):
        current_3fold_x_vec_proj = vector_projection_on_subspace(self.get_3fold_x_vec_from_4fold(), *self.fold3_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_3fold_x_vec, current_3fold_x_vec_proj), self.fold3_center), 0, abs_tol=0.1) else -1
        return dir * vector_angle(current_3fold_x_vec_proj, self.init_3fold_x_vec)

    def get_3_fold_angle_z_from_5fold(self):
        current_3fold_x_vec_proj = vector_projection_on_subspace(self.get_3fold_x_vec_from_HFfold(), *self.fold3_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_3fold_x_vec, current_3fold_x_vec_proj), self.fold3_center), 0, abs_tol=0.1) else -1
        return dir * vector_angle(current_3fold_x_vec_proj, self.init_3fold_x_vec)

    def get_2_fold_angle_z_from_3fold(self):
        current_2fold_x_vec_proj = vector_projection_on_subspace(self.get_2fold_x_vec_from_3fold(), *self.fold2_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_2fold_x_vec, current_2fold_x_vec_proj), self.fold2_center), 0, abs_tol=0.1) else -1
        return dir * vector_angle(current_2fold_x_vec_proj, self.init_2fold_x_vec)

    def get_5_fold_angle_z_from_2fold(self):
        current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_5fold_x_vec_from_2fold(), *self.foldHF_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.fold5_center), 0, abs_tol=0.1) else -1
        return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)


    def get_4_fold_angle_z_from_2fold(self):
        current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_4fold_x_vec_from_2fold(), *self.foldHF_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.fold4_center), 0,
                                abs_tol=0.1) else -1
        return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)

    def get_THF_fold_angle_z_from_2fold(self):
        current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_THFfold_x_vec_from_2fold(), *self.foldHF_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.foldTHF_center), 0,
                                abs_tol=0.1) else -1
        return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)

    def get_2_fold_angle_z_from_HFfold(self):
        current_2fold_x_vec_proj = vector_projection_on_subspace(self.get_2fold_x_vec_from_HFfold(), *self.fold2_plane)
        # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
        dir = 1 if math.isclose(vector_angle(np.cross(self.init_2fold_x_vec, current_2fold_x_vec_proj), self.fold2_center), 0, abs_tol=0.1) else -1
        return dir * vector_angle(current_2fold_x_vec_proj, self.init_2fold_x_vec)

    def transfer_2toHF(self, pose2, poseHF):
        if self.symmetry_type == "I":
            self.__transfer_2toIHF(pose2, poseHF)
        elif self.symmetry_type == "O":
            self.__transfer_2toOHF(pose2, poseHF)
        elif self.symmetry_type == "T":
            self.__transfer_2toTHF(pose2, poseHF)

    def transfer_HFto3(self, poseHF, pose3):
        if self.symmetry_type == "I":
            self.transfer_IHFto3(poseHF, pose3)
        elif self.symmetry_type == "O":
            self.__transfer_OHFto3(poseHF, pose3)
        elif self.symmetry_type == "T":
            self.__transfer_THFto3(poseHF, pose3)

    def __transfer_THFto3(self, pose4, pose3):
        self.foldHF_setup.update_dofs_from_pose(pose4, apply_dofs=True)
        self.fold3_center = self.get_3fold_center_from_HFfold()
        set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
        set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold()) # TODO to 4
        set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
        self.set_rotation_quick(pose3, pose4, "JUMP31fold111_z", "JUMPHFfold111_subunit", "JUMP31fold111_z")
        #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")

    def __transfer_OHFto3(self, pose4, pose3):
        self.foldHF_setup.update_dofs_from_pose(pose4, apply_dofs=True)
        self.fold3_center = self.get_3fold_center_from_HFfold()
        set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
        set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold()) # TODO to 4
        set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
        self.set_rotation_quick(pose3, pose4, "JUMP31fold111_z", "JUMPHFfold111_subunit", "JUMP31fold111_z")
        #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")

    def transfer_IHFto3(self, pose5, pose3):
        self.foldHF_setup.update_dofs_from_pose(pose5, apply_dofs=True)
        self.fold3_center = self.get_3fold_center_from_HFfold()
        set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
        set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold())
        set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
        self.set_rotation_quick(pose3, pose5, "JUMP31fold111_z", "JUMPHFfold111_subunit", "JUMP31fold111_z")
        #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")

    def transfer_3to2(self, pose3, pose2):
        self.fold3_setup.update_dofs_from_pose(pose3, apply_dofs=True)
        self.fold2_center = self.get_2fold_center_from_3fold()
        set_jumpdof_str_int(pose2, "JUMP21fold1", 3, np.linalg.norm(self.get_2fold_center_from_3fold()))
        set_jumpdof_str_int(pose2, "JUMP21fold1_z", 6, self.get_2_fold_angle_z_from_3fold())
        set_jumpdof_str_int(pose2, "JUMP21fold111", 1, np.linalg.norm(self.get_2fold_x_vec_from_3fold()))
        self.set_rotation_quick(pose2, pose3, "JUMP21fold111_z", "JUMP31fold111_subunit", "JUMP21fold111_z")

    def __transfer_2toTHF(self, pose2, poseTHF):
        self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
        self.foldTHF_center = self.get_THF_center_from_2fold()
        set_jumpdof_str_int(poseTHF, "JUMPHFfold1", 3, np.linalg.norm(self.get_THF_center_from_2fold()))
        set_jumpdof_str_int(poseTHF, "JUMPHFfold1_z", 6, self.get_THF_fold_angle_z_from_2fold())
        set_jumpdof_str_int(poseTHF, "JUMPHFfold111", 1, np.linalg.norm(self.get_THFfold_x_vec_from_2fold()))
        self.set_rotation_quick(poseTHF, pose2, "JUMPHFfold111_z", "JUMP21fold111_subunit", "JUMPHFfold111_z")

    def __transfer_2toOHF(self, pose2, pose4):
        self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
        self.fold4_center = self.get_4fold_center_from_2fold()
        set_jumpdof_str_int(pose4, "JUMPHFfold1", 3, np.linalg.norm(self.get_4fold_center_from_2fold()))
        set_jumpdof_str_int(pose4, "JUMPHFfold1_z", 6, self.get_4_fold_angle_z_from_2fold())
        set_jumpdof_str_int(pose4, "JUMPHFfold111", 1, np.linalg.norm(self.get_4fold_x_vec_from_2fold()))
        self.set_rotation_quick(pose4, pose2, "JUMPHFfold111_z", "JUMP21fold111_subunit", "JUMPHFfold111_z")

    def __transfer_2toIHF(self, pose2, pose5):
        self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
        self.fold5_center = self.get_5fold_center_from_2fold()
        set_jumpdof_str_int(pose5, "JUMPHFfold1", 3, np.linalg.norm(self.get_5fold_center_from_2fold()))
        set_jumpdof_str_int(pose5, "JUMPHFfold1_z", 6, self.get_5_fold_angle_z_from_2fold())
        set_jumpdof_str_int(pose5, "JUMPHFfold111", 1, np.linalg.norm(self.get_5fold_x_vec_from_2fold()))
        self.set_rotation_quick(pose5, pose2, "JUMPHFfold111_z", "JUMP21fold111_subunit", "JUMPHFfold111_z")

    def transfer_HFto2(self, pose5, pose2):
        self.foldHF_setup.update_dofs_from_pose(pose5, apply_dofs=True)
        self.fold2_center = self.get_2fold_center_from_HFfold()
        set_jumpdof_str_int(pose2, "JUMP21fold1", 3, np.linalg.norm(self.get_2fold_center_from_HFfold()))
        set_jumpdof_str_int(pose2, "JUMP21fold1_z", 6, self.get_2_fold_angle_z_from_HFfold())
        set_jumpdof_str_int(pose2, "JUMP21fold111", 1, np.linalg.norm(self.get_2fold_x_vec_from_HFfold()))
        self.set_rotation_quick(pose2, pose5, "JUMP21fold111_z", "JUMPHFfold111_subunit", "JUMP21fold111_z")
