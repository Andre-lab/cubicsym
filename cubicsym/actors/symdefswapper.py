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
from symmetryhandler.reference_kinematics import set_jumpdof_str_int, perturb_jumpdof_str_int
import math
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from pyrosetta.rosetta.core.kinematics import Stub, Jump
from cubicsym.utilities import get_chain_map as util_get_chain_map
from pyrosetta.rosetta.numeric import xyzVector_double_t
from symmetryhandler.mathfunctions import vector_projection
from pyrosetta.rosetta.core.scoring import CA_rmsd
from pyrosetta.rosetta.std import map_unsigned_long_unsigned_long  # core::Size, core::Size
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from symmetryhandler.mathfunctions import rotation_matrix
from pyrosetta.rosetta.numeric import xyzVector_double_t
from symmetryhandler.reference_kinematics import get_dofs


class SymDefSwapper:
    """Swaps symmetry between HF-fold, 3F-fold and 2F-fold based symmetries of a cubic symmetrical pose."""

    def __init__(self, pose, symmetry_file, visualizer=None):
        # store information we need in order to transfer
        assert is_symmetric(pose)
        self.foldHF_setup = CubicSetup(symmetry_file)
        self.is_righthanded = self.foldHF_setup.calculate_if_rightanded()
        self.symmetry_type = self.foldHF_setup.cubic_symmetry()
        self.monomer_sizes = pose.size()
        if self.symmetry_type == "I":
            self.fold3_setup = self.foldHF_setup.create_I_3fold_based_symmetry()
            self.fold2_setup = self.foldHF_setup.create_I_2fold_based_symmetry()
        elif self.symmetry_type == "O":
            self.fold3_setup = self.foldHF_setup.create_O_3fold_based_symmetry()
            self.fold2_setup = self.foldHF_setup.create_O_2fold_based_symmetry()
        elif self.symmetry_type == "T":
            self.fold3_setup = self.foldHF_setup.create_T_3fold_based_symmetry()
            self.fold2_setup = self.foldHF_setup.create_T_2fold_based_symmetry()
        else:
            raise ValueError(f"Symmetry type: {self.symmetry_type}, is not accepted")
        self.foldHF_setup.apply_dofs()
        self.fold3_setup.apply_dofs()
        self.fold2_setup.apply_dofs()
        self.visualizer = visualizer
        self.sanity_check(pose)

    def sanity_check(self, poseHF):
        assert CubicSetup.get_base_from_pose(poseHF) == "HF"
        pose3 = self.create_3fold_pose_from_HFfold(poseHF)
        pose2 = self.create_2fold_pose_from_HFfold(poseHF)
        self.foldHF_setup.vrts_overlap_with_pose(poseHF)
        self.fold3_setup.vrts_overlap_with_pose(pose3)
        self.fold2_setup.vrts_overlap_with_pose(pose2)

    # def get_chain_map(self):
    #     """Returns a list of the mapping between each chain (in Rosetta numbering) for I/O: HF-, 3- and 2-fold or for T: HF- and 2-fold."""
    #     util_get_chain_map(self.symmetry_type, self.is_righthanded)
    #
    # def init_x_vectors(self):
    #     self.init_HFfold_x_vec = self.get_HFfold_x_vec_from_HFfold()
    #     self.init_3fold_x_vec = self.get_3fold_x_vec_from_HFfold()
    #     self.init_2fold_x_vec = self.get_2fold_x_vec_from_HFfold()

        # if sum(abs(trans)) > 0.1:
        #     print("puhadadadada")
        #     trans_vector = xyzVector_double_t(0)
        #     trans_vector.assign(0,0,0)
        #     jump.set_translation(trans_vector)

    #         self.set_rotation_quick(poseB, poseA, f"JUMP{jiB}fold111_sds", f"JUMP{jiA}fold111_subunit", f"JUMP{jiB}fold111_sds", setupA)

    def set_rotation_quick_new(self, poseB, poseA, jiB, jiA, setupA, setupB, flip_x=False, flip_y=False, flip_z=False):
        """
        Transfer the rotation of poseA onto poseB. We want to make the subunit anchor stub overlap in space. That means the
        downstream stub of JUMP{jiB}fold111_subunit should overlap with the JUMP{jiA}fold111_subunit stub
        """
        # Is it upstream or downstream???
        # stub1 = Stub(poseB.conformation().upstream_jump_stub(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"))) # should be VRT{jiB}fold111_sds, but is VRT31fold111_z
        # assert setupB.get_map_pose_resi_to_vrt(poseB)[poseB.fold_tree().upstream_jump_residue(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"))] == f"VRT{jiB}fold111_sds"
        # stub2 = Stub(poseA.conformation().upstream_jump_stub(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit"))) # VRT{jiA}fold111_subunit, but is VRTHFfold111_sds
        # assert setupA.get_map_pose_resi_to_vrt(poseA)[poseA.fold_tree().upstream_jump_residue(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit"))] == f"VRT{jiA}fold111_subunit"
        stub1 = Stub(poseB.conformation().downstream_jump_stub(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"))) # should be VRT{jiB}fold111_sds, but is VRT31fold111_z
        assert setupB.get_map_pose_resi_to_vrt(poseB)[poseB.fold_tree().downstream_jump_residue(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"))] == f"VRT{jiB}fold111_sds"
        stub2 = Stub(poseA.conformation().downstream_jump_stub(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit"))) # VRT{jiA}fold111_subunit, but is VRTHFfold111_sds
        assert poseA.fold_tree().downstream_jump_residue(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit")) == setupA.get_anchor_residue(poseA)

        jump = Jump(stub1, stub2)
        try:
            assert np.isclose(np.array(jump.get_translation()), [0, 0, 0], atol=1e-2).all(), "The stubs should overlap at this point"
        except AssertionError:
            raise AssertionError
        ###### test
        # conversion
        if flip_x or flip_y or flip_z:
            if flip_x:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_x, 180))
            elif flip_y:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_y, 180))
            elif flip_z:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_z, 180))
            vec = xyzVector_double_t(0)
            mros = xyzMatrix_double_t(0)
            for i in [0, 1, 2]:
                vec.assign(rotation[i][0], rotation[i][1], rotation[i][2])
                mros.row(i + 1, vec)
            jump.set_rotation(mros)
        ######
        #poseB.set_jump(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"), jump) # _z and _sds
        poseB.set_jump(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_subunit"), jump) # _sds and subunit

        print("poseA", get_dofs(poseA))
        print("poseB", get_dofs(poseB))

        try:
            # THERES A BUG FOR SOME REASON THAT IT JUST RETURNS 0 EVEN THOUGH IT IS NOT IN THIS CASE!!!?????????????????????
            m = map_unsigned_long_unsigned_long()
            for ri in range(1, poseA.chain_end(1) + 1):
                m[ri] = ri
            rmsd = CA_rmsd(poseA, poseB, m)
            print(rmsd)
            assert math.isclose(rmsd, 0, abs_tol=1e-3)
            assert np.isclose(np.array(poseA.residue(1).atom("CA").xyz()), np.array(poseB.residue(1).atom("CA").xyz()), atol=1e-3).all()
        except AssertionError:
            print("RAISED ASSERTION!!!!!")
            from simpletestlib.test import setup_pymol;
            pmm = setup_pymol(reinitialize=False);
            poseA.pdb_info().name("A");
            poseB.pdb_info().name("B");
            pmm.apply(poseA);
            pmm.apply(poseB)
            raise AssertionError
        ...

    def set_rotation_quick_old(self, poseB, poseA, jumpstrB, jumpstrA, jumptoset, setupA, flip_x=False, flip_y=False, flip_z=False):
        """Sets the jump of jumptoset based on the downstream stubs of jumpstr1 and jumpstr2. The jump is set on pose1."""
        stub1 = Stub(poseB.conformation().upstream_jump_stub(sym_dof_jump_num(poseB, jumpstrB))) # "JUMP31fold1111"
        stub2 = Stub(poseA.conformation().upstream_jump_stub(sym_dof_jump_num(poseA, jumpstrA))) # "JUMPHFfold1111"
        jump = Jump(stub1, stub2)
        try:
            assert np.isclose(np.array(jump.get_translation()), [0, 0, 0], atol=1e-2).all(), "The stubs should overlap at this point"
        except AssertionError:
            raise AssertionError
        ###### test
        # conversion
        if flip_x or flip_y or flip_z:
            if flip_x:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_x, 180))
            elif flip_y:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_y, 180))
            elif flip_z:
                rotation = np.dot(np.array(jump.get_rotation()), rotation_matrix(setupA.get_vrt(jumpstrA.replace("JUMP", "VRT").replace("subunit", "sds"))._vrt_z, 180))
            vec = xyzVector_double_t(0)
            mros = xyzMatrix_double_t(0)
            for i in [0, 1, 2]:
                vec.assign(rotation[i][0], rotation[i][1], rotation[i][2])
                mros.row(i + 1, vec)
            jump.set_rotation(mros)
        ######
        poseB.set_jump(sym_dof_jump_num(poseB, jumptoset), jump)
    #
    # def get_HFfold_plane(self):
    #     """Get the plane of the 5-fold axis spann5d by 2 vectors as rows in a matrix 2x3 matrix."""
    #     return np.array([[1, 0, 0], [0, 1, 0]])
    #
    # def get_2fold_plane(self):
    #     """Get the plane of the 2-fold axis spanned by 2 vectors as rows in a matrix 2x3 matrix."""
    #     x = np.array([1, 0, 0])
    #     y = np.array([0, 1, 0])
    #     R = rotation_matrix_from_vector_to_vector(- self.fold2_setup.get_vrt("VRT21fold1").vrt_z, [0, 0, 1])
    #     x = rotate_right_multiply(x, R)
    #     y = rotate_right_multiply(y, R)
    #     assert np.isclose(np.cross(x, y), self.get_2fold_center_from_3fold() / np.linalg.norm(self.get_2fold_center_from_3fold())).all()
    #     return np.vstack((x, y))
    #
    # # todo: could be done better in the future
    # def get_3fold_plane(self):
    #     """Get the plane of the 3-fold axis spanned by 2 vectors as rows in a matrix 2x3 matrix."""
    #     x = np.array([1,0,0])
    #     y = np.array([0,1,0])
    #     R = rotation_matrix_from_vector_to_vector([0, 0, 1], - self.fold3_setup.get_vrt("VRT31fold1").vrt_z)
    #     x = rotate(x, R)
    #     y = rotate(y, R)
    #     assert np.isclose(np.cross(x, y), self.get_3fold_center_from_HFfold() / np.linalg.norm(self.get_3fold_center_from_HFfold())).all()
    #     return np.vstack((x, y))

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
        pose3 = self.foldHF_setup.make_asymmetric_pose(poseHF)
        self.fold3_setup.make_symmetric_pose(pose3)
        #### DEBUG
        self.fold3_setup.update_dofs_from_pose(pose3, apply_dofs=True)
        # This is moved out of the NEW function
        self.foldHF_setup.update_dofs_from_pose(poseHF, apply_dofs=True)
        try:
            assert np.isclose(np.array(pose3.residue(self.fold3_setup.get_anchor_residue(pose3)).atom("CA").xyz()), self.fold3_setup.get_vrt("VRT31fold111_sds").vrt_orig, atol=1e-2).all()
        except AssertionError:
            raise AssertionError
        try:
            assert np.isclose(np.array(poseHF.residue(self.foldHF_setup.get_anchor_residue(poseHF)).atom("CA").xyz()), self.foldHF_setup.get_vrt("VRTHFfold111_sds").vrt_orig, atol=1e-2).all()
        except AssertionError:
            raise AssertionError
        #####
        if transfer:
            self.transfer_poseA2B(poseHF, pose3)
            # if self.symmetry_type == "I":
            #     self.transfer_IHFto3(poseHF, pose3)
            # elif self.symmetry_type == "O":
            #     self.__transfer_OHFto3(poseHF, pose3)
            # elif self.symmetry_type == "T":
            #     self.transfer_poseA2B(poseHF, pose3)
            #     # self.__transfer_THFto3_NEW(poseHF, pose3)
            #     # self.__transfer_THFto3(poseHF, pose3)
        return pose3

    def create_2fold_pose_from_HFfold(self, poseHF, transfer=True):
        """Creates a symmetric pose based on the 2-fold axis."""
        pose2 = self.foldHF_setup.make_asymmetric_pose(poseHF)
        # As we reuse this function the subunit dofs might change and we want to keep them fixed
        # self.fold2_setup.reset_jumpdofs("JUMP21fold111_subunit")
        self.fold2_setup.make_symmetric_pose(pose2)
        #### DEBUG
        self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
        # This is moved out of the NEW function
        self.foldHF_setup.update_dofs_from_pose(poseHF, apply_dofs=True)
        try:
            assert np.isclose(np.array(pose2.residue(self.fold2_setup.get_anchor_residue(pose2)).atom("CA").xyz()), self.fold2_setup.get_vrt("VRT21fold111_sds").vrt_orig, atol=1e-2).all()
        except AssertionError:
            raise AssertionError
        try:
            assert np.isclose(np.array(poseHF.residue(self.foldHF_setup.get_anchor_residue(poseHF)).atom("CA").xyz()), self.foldHF_setup.get_vrt("VRTHFfold111_sds").vrt_orig, atol=1e-2).all()
        except AssertionError:
            raise AssertionError
        #####
        if transfer:
            self.transfer_poseA2B(poseHF, pose2)
            # self.transfer_HFto2(pose5, pose2)
        return pose2
    #
    # def get_HFfold_x_vec_from_HFfold(self):
    #     a = np.array(self.foldHF_setup.get_vrt("VRTHFfold11").vrt_orig)
    #     b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111").vrt_orig)
    #     return b - a
    #
    # def get_3fold_x_vec_from_4fold(self):
    #     a = self.get_3fold_center_from_HFfold()
    #     b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     return b - a
    #
    # def get_3fold_x_vec_from_HFfold(self):
    #     a = self.get_3fold_center_from_HFfold()
    #     b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     return b - a
    #
    # def get_2fold_x_vec_from_3fold(self):
    #     a = self.get_2fold_center_from_3fold()
    #     b = np.array(self.fold3_setup.get_vrt("VRT31fold111_z").vrt_orig)
    #     return b - a
    #
    # def get_THFfold_x_vec_from_2fold(self):
    #     a = self.get_THF_center_from_2fold()
    #     b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     return b - a
    #
    # def get_4fold_x_vec_from_2fold(self):
    #     a = self.get_4fold_center_from_2fold()
    #     b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     return b - a
    #
    # def get_5fold_x_vec_from_2fold(self):
    #     a = self.get_5fold_center_from_2fold()
    #     b = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     return b - a
    #
    # def get_2fold_x_vec_from_HFfold(self):
    #     a = self.get_2fold_center_from_HFfold()
    #     b = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     return b - a
    #
    # # fixme these functions can just work on the chain names can be made to be
    # def get_3fold_center_from_HFfold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     b = np.array(self.foldHF_setup.get_vrt("VRT2fold121_z").vrt_orig)
    #     c = np.array(self.foldHF_setup.get_vrt("VRT3fold111_z").vrt_orig)
    #     return (a + b + c) / 3
    #
    # # def get_3fold_center_from_4fold(self):
    # #     # todo calculation in rosetta and finally make array
    # #     a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    # #     b = np.array(self.foldHF_setup.get_vrt("VRT2fold121_z").vrt_orig)
    # #     c = np.array(self.foldHF_setup.get_vrt("VRT3fold111_z").vrt_orig)
    # #     return (a + b + c) / 3
    #
    # def get_2fold_center_from_3fold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.fold3_setup.get_vrt("VRT31fold111_z").vrt_orig)
    #     if self.symmetry_type in ("I"):
    #         if self.is_righthanded: # 1stm
    #             b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
    #         else: # 6S44
    #             b = np.array(self.fold3_setup.get_vrt("VRT35fold131_z").vrt_orig)
    #     elif self.symmetry_type == "O":
    #         if self.is_righthanded: # 1AEW
    #             b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
    #         else: # 1PY3
    #             b = np.array(self.fold3_setup.get_vrt("VRT33fold131_z").vrt_orig)
    #         # raise NotImplementedError
    #     elif self.symmetry_type == "T":
    #         if self.is_righthanded: # 1HOS
    #             b = np.array(self.fold3_setup.get_vrt("VRT32fold121_z").vrt_orig)
    #         else: # 1MOG
    #             b = np.array(self.fold3_setup.get_vrt("VRT33fold131_z").vrt_orig)
    #         # raise NotImplementedError
    #     return (a + b) / 2
    #
    # def get_THF_center_from_2fold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     b = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
    #     c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
    #     return (a + b + c) / 3
    #
    # def get_4fold_center_from_2fold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     b = np.array(self.fold2_setup.get_vrt("VRT24fold111_z").vrt_orig)
    #     c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
    #     d = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
    #     return (a + b + c + d) / 4
    #
    # def get_5fold_center_from_2fold(self):
    #     # todo calculation in rosetta and finally make array
    #     a = np.array(self.fold2_setup.get_vrt("VRT21fold111_z").vrt_orig)
    #     b = np.array(self.fold2_setup.get_vrt("VRT22fold111_z").vrt_orig)
    #     c = np.array(self.fold2_setup.get_vrt("VRT23fold111_z").vrt_orig)
    #     d = np.array(self.fold2_setup.get_vrt("VRT24fold111_z").vrt_orig)
    #     e = np.array(self.fold2_setup.get_vrt("VRT25fold111_z").vrt_orig)
    #     return (a + b + c + d + e) / 5
    #
    # def get_2fold_center_from_HFfold(self):
    #     a = np.array(self.foldHF_setup.get_vrt("VRTHFfold111_z").vrt_orig)
    #     b = np.array(self.foldHF_setup.get_vrt("VRT2fold111_z").vrt_orig)
    #     return (a + b) / 2
    #
    # def get_3_fold_angle_z_from_4fold(self):
    #     current_3fold_x_vec_proj = vector_projection_on_subspace(self.get_3fold_x_vec_from_4fold(), *self.fold3_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_3fold_x_vec, current_3fold_x_vec_proj), self.fold3_center), 0, abs_tol=0.1) else -1
    #     return dir * vector_angle(current_3fold_x_vec_proj, self.init_3fold_x_vec)
    #
    # def get_3_fold_angle_z_from_5fold(self):
    #     current_3fold_x_vec_proj = vector_projection_on_subspace(self.get_3fold_x_vec_from_HFfold(), *self.fold3_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_3fold_x_vec, current_3fold_x_vec_proj), self.fold3_center), 0, abs_tol=0.1) else -1
    #     return dir * vector_angle(current_3fold_x_vec_proj, self.init_3fold_x_vec)
    #
    # def get_2_fold_angle_z_from_3fold(self):
    #     current_2fold_x_vec_proj = vector_projection_on_subspace(self.get_2fold_x_vec_from_3fold(), *self.fold2_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_2fold_x_vec, current_2fold_x_vec_proj), self.fold2_center), 0, abs_tol=0.1) else -1
    #     return dir * vector_angle(current_2fold_x_vec_proj, self.init_2fold_x_vec)
    #
    # def get_5_fold_angle_z_from_2fold(self):
    #     current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_5fold_x_vec_from_2fold(), *self.foldHF_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.fold5_center), 0, abs_tol=0.1) else -1
    #     return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)
    #
    #
    # def get_4_fold_angle_z_from_2fold(self):
    #     current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_4fold_x_vec_from_2fold(), *self.foldHF_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.fold4_center), 0,
    #                             abs_tol=0.1) else -1
    #     return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)
    #
    # def get_THF_fold_angle_z_from_2fold(self):
    #     current_5fold_x_vec_proj = vector_projection_on_subspace(self.get_THFfold_x_vec_from_2fold(), *self.foldHF_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_HFfold_x_vec, current_5fold_x_vec_proj), self.foldTHF_center), 0,
    #                             abs_tol=0.1) else -1
    #     return dir * vector_angle(current_5fold_x_vec_proj, self.init_HFfold_x_vec)
    #
    # def get_2_fold_angle_z_from_HFfold(self):
    #     current_2fold_x_vec_proj = vector_projection_on_subspace(self.get_2fold_x_vec_from_HFfold(), *self.fold2_plane)
    #     # if the cross product points in the same direction as the 3-fold center then, because of the opposite direction in
    #     dir = 1 if math.isclose(vector_angle(np.cross(self.init_2fold_x_vec, current_2fold_x_vec_proj), self.fold2_center), 0, abs_tol=0.1) else -1
    #     return dir * vector_angle(current_2fold_x_vec_proj, self.init_2fold_x_vec)

    # def transfer_2toHF(self, pose2, poseHF):
    #     if self.symmetry_type == "I":
    #         self.__transfer_2toIHF(pose2, poseHF)
    #     elif self.symmetry_type == "O":
    #         self.__transfer_2toOHF(pose2, poseHF)
    #     elif self.symmetry_type == "T":
    #         self.__transfer_2toTHF(pose2, poseHF)
    #
    # def transfer_HFto3(self, poseHF, pose3):
    #     if self.symmetry_type == "I":
    #         self.transfer_IHFto3(poseHF, pose3)
    #     elif self.symmetry_type == "O":
    #         self.__transfer_OHFto3(poseHF, pose3)
    #     elif self.symmetry_type == "T":
    #         #self.__transfer_THFto3(poseHF, pose3)
    #         # self.__transfer_THFto3_NEW(poseHF, pose3)
    #         self.transfer_poseA2B(poseHF, pose3)
    #
    # def __transfer_THFto3(self, poseHF, pose3):
    #     self.foldHF_setup.update_dofs_from_pose(poseHF, apply_dofs=True)
    #     self.fold3_center = self.get_3fold_center_from_HFfold()
    #     set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
    #     set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold()) # TODO to 4
    #     set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
    #     self.set_rotation_quick(pose3, poseHF, "JUMP31fold111_sds", "JUMPHFfold111_subunit", "JUMP31fold111_sds")
    #     #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")

    def get_jid(self, pose):
        base = CubicSetup.get_base_from_pose(pose)
        if base == "HF":
            setup = self.foldHF_setup
            jumpid = "HF"
        elif base == "3F":
            setup = self.fold3_setup
            jumpid = "31"
        elif base == "2F":
            setup = self.fold2_setup
            jumpid = "21"
        else:
            raise ValueError(base + " is not accepted.")
        return setup, jumpid

    # todo:
    #  write out the exact logic in text
    #  make this general for I, O, T and 3 and 2
    #  To speed up you can use the vrts in the pose directly
    #  make a debug flag so asserts are called all the time
    def transfer_poseA2B(self, poseA, poseB):
        """Transfers the translation and rotation of each subunit of poseA onto poseB where poseA and poseB have different symmetry setups."""
        # get variables depending on the poses
        setupA, jiA = self.get_jid(poseA) # poseHF
        setupB, jiB = self.get_jid(poseB) # pose3
        assert jiA != jiB, "Does it make sense to use the same setup?? I dont think so!"

        # First we need to align VRT31fold111 onto VRTHFfold111 and secondly we need to take care of the rotation
        # To first align the 2 VRT's we use 3 dofs (a z translation, a x translation and an angle_z rotation).
        # There are 2 cases to consider one where VRTHFfold111 is above the x, y plane of VRT31fold1 and one where it is below.
        # If it is located below the plane we need to do a negative z translation, else we do a positive one.


        # We first check if the point is in the above or below the plane
        point_to_get_to = setupA.get_vrt(f"VRT{jiA}fold111")._vrt_orig
        x = setupB.get_vrt(f"VRT{jiB}fold11")._vrt_x
        y = setupB.get_vrt(f"VRT{jiB}fold11")._vrt_y
        z = - setupB.get_vrt(f"VRT{jiB}fold11")._vrt_z # minus because of rosettas convention
        plane_to_vec = point_to_get_to - vector_projection_on_subspace(point_to_get_to, x, y)
        # IF plane_to_vec is in the same direction as -z we dont need to change direction of the VRT31fold11 vector, else we do
        proj = vector_projection(plane_to_vec, z)
        cross_angle = vector_angle(proj, z)
        dir = 1 if cross_angle < 90 else - 1
        z_vec = vector_projection(point_to_get_to, dir * z)
        # ---
        z = dir * np.linalg.norm(z_vec)
        x_vec = setupA.get_vrt(f"VRT{jiA}fold111")._vrt_orig - z_vec
        x = np.linalg.norm(x_vec)
        # update the current translations in the setup
        setupB.set_dof(f"JUMP{jiB}fold1", "z", "translation", z)
        setupB.set_dof(f"JUMP{jiB}fold111", "x", "translation", x)
        setupB.apply_dofs()
        # ---
        # to get angle_z we need to consider the angle between the current x translation we have and where the point is right now (x_vec)
        current_x_vec = setupB.get_vrt(f"VRT{jiB}fold111")._vrt_orig - setupB.get_vrt(f"VRT{jiB}fold11")._vrt_orig
        cross_angle = vector_angle(np.cross(current_x_vec, x_vec), z_vec)
        dir = - 1 * dir if cross_angle > 90 else 1 * dir
        angle_z = dir * vector_angle(current_x_vec, x_vec)
        # todo, not nescesarray vvv
        setupB.set_dof(f"JUMP{jiB}fold1_z", "z", "rotation", setupB._dofs[f"JUMP{jiB}fold1_z"][0][2] + angle_z)
        setupB.apply_dofs()
        try:
            assert np.isclose(setupB.get_vrt(f"VRT{jiB}fold111")._vrt_orig, setupA.get_vrt(f"VRT{jiA}fold111")._vrt_orig, atol=1e-3).all()
        except AssertionError:
            raise AssertionError
        # todo, not nescesarray ^^^
        # set the dofs
        set_jumpdof_str_int(poseB, f"JUMP{jiB}fold1", 3, z)
        perturb_jumpdof_str_int(poseB, f"JUMP{jiB}fold1_z", 6, angle_z)
        set_jumpdof_str_int(poseB, f"JUMP{jiB}fold111", 1, x)
        #self.set_rotation_quick(poseB, poseA, f"JUMP{jiB}fold111_sds", f"JUMP{jiA}fold111_subunit", f"JUMP{jiB}fold111_sds", setupA)
        self.set_rotation_quick_new(poseB, poseA, jiB, jiA, setupA, setupB)

        # assert that main chain overlaps!
        print("z", z_vec, z)
        print("x", x_vec, x)
        from symmetryhandler.reference_kinematics import get_dofs


    # def __transfer_THFto3_NEW(self, poseHF, pose3):
    #     # First we need to align VRT31fold111 onto VRTHFfold111 and secondly we need to take care of the rotation
    #     # To first align the 2 VRT's we use 3 dofs (a z translation, a x translation and an angle_z rotation).
    #     # There are 2 cases to consider one where VRTHFfold111 is above the x, y plane of VRT31fold1 and one where it is below.
    #     # If it is located below the plane we need to do a negative z translation, else we do a positive one.
    #
    #     # We first check if the point is in the above or below the plane
    #     point_to_get_to = self.foldHF_setup.get_vrt(f"VRTHFfold111")._vrt_orig
    #     x = self.fold3_setup.get_vrt(f"VRT31fold11")._vrt_x
    #     y = self.fold3_setup.get_vrt(f"VRT31fold11")._vrt_y
    #     z = - self.fold3_setup.get_vrt(f"VRT31fold11")._vrt_z # minus because of rosettas convention
    #     plane_to_vec = point_to_get_to - vector_projection_on_subspace(point_to_get_to, x, y)
    #     # IF plane_to_vec is in the same direction as -z we dont need to change direction of the VRT31fold11 vector, else we do
    #     proj = vector_projection(plane_to_vec, z)
    #     cross_angle = vector_angle(proj, z)
    #     dir = 1 if cross_angle < 90 else - 1
    #     z_vec = vector_projection(point_to_get_to, dir * z)
    #     # ---
    #     z = dir * np.linalg.norm(z_vec)
    #     print(z_vec, z)
    #     x_vec = self.foldHF_setup.get_vrt(f"VRTHFfold111")._vrt_orig - z_vec
    #     x = np.linalg.norm(x_vec)
    #     # update the current translations in the setup
    #     self.fold3_setup.set_dof("JUMP31fold1", "z", "translation", z)
    #     self.fold3_setup.set_dof("JUMP31fold111", "x", "translation", x)
    #     self.fold3_setup.apply_dofs()
    #     # ---
    #     # to get angle_z we need to consider the angle between the current x translation we have and where the point is right now (x_vec)
    #     current_x_vec = self.fold3_setup.get_vrt(f"VRT31fold111")._vrt_orig - self.fold3_setup.get_vrt(f"VRT31fold11")._vrt_orig
    #     cross_angle = vector_angle(np.cross(current_x_vec, x_vec), z_vec)
    #     dir = - 1 * dir if cross_angle > 90 else 1 * dir
    #     angle_z = dir * vector_angle(current_x_vec, x_vec)
    #     # todo, not nescesarray vvv
    #     self.fold3_setup.set_dof("JUMP31fold1_z", "z", "rotation", self.fold3_setup._dofs["JUMP31fold1_z"][0][2] + angle_z)
    #     self.fold3_setup.apply_dofs()
    #     try:
    #         assert np.isclose(self.fold3_setup.get_vrt(f"VRT31fold111")._vrt_orig, self.foldHF_setup.get_vrt(f"VRTHFfold111")._vrt_orig).all()
    #     except AssertionError:
    #         raise AssertionError
    #     # todo, not nescesarray ^^^
    #     # set the dofs
    #     set_jumpdof_str_int(pose3, "JUMP31fold1", 3, z)
    #     perturb_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, angle_z)
    #     set_jumpdof_str_int(pose3, "JUMP31fold111", 1, x)
    #     self.set_rotation_quick(pose3, poseHF, "JUMP31fold111_sds", "JUMPHFfold111_subunit", "JUMP31fold111_sds")
    #
    # def __transfer_OHFto3(self, pose4, pose3):
    #     self.foldHF_setup.update_dofs_from_pose(pose4, apply_dofs=True)
    #     self.fold3_center = self.get_3fold_center_from_HFfold()
    #     set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
    #     set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold()) # TODO to 4
    #     set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
    #     self.set_rotation_quick(pose3, pose4, "JUMP31fold111_sds", "JUMPHFfold111_subunit", "JUMP31fold111_sds")
    #     #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")
    #
    # def transfer_IHFto3(self, pose5, pose3):
    #     self.foldHF_setup.update_dofs_from_pose(pose5, apply_dofs=True)
    #     self.fold3_center = self.get_3fold_center_from_HFfold()
    #     set_jumpdof_str_int(pose3, "JUMP31fold1", 3, np.linalg.norm(self.get_3fold_center_from_HFfold()))
    #     set_jumpdof_str_int(pose3, "JUMP31fold1_z", 6, self.get_3_fold_angle_z_from_5fold())
    #     set_jumpdof_str_int(pose3, "JUMP31fold111", 1, np.linalg.norm(self.get_3fold_x_vec_from_HFfold()))
    #     self.set_rotation_quick(pose3, pose5, "JUMP31fold111_sds", "JUMPHFfold111_subunit", "JUMP31fold111_sds")
    #     #self.set_rotation_quick(pose3, pose5, "JUMP31fold1111", "JUMPHFfold1111_subunit", "JUMP31fold1111")
    #
    # def transfer_3to2(self, pose3, pose2):
    #     self.fold3_setup.update_dofs_from_pose(pose3, apply_dofs=True)
    #     self.fold2_center = self.get_2fold_center_from_3fold()
    #     set_jumpdof_str_int(pose2, "JUMP21fold1", 3, np.linalg.norm(self.get_2fold_center_from_3fold()))
    #     set_jumpdof_str_int(pose2, "JUMP21fold1_z", 6, self.get_2_fold_angle_z_from_3fold())
    #     set_jumpdof_str_int(pose2, "JUMP21fold111", 1, np.linalg.norm(self.get_2fold_x_vec_from_3fold()))
    #     self.set_rotation_quick(pose2, pose3, "JUMP21fold111_sds", "JUMP31fold111_subunit", "JUMP21fold111_sds")
    #
    # def __transfer_2toTHF(self, pose2, poseTHF):
    #     self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
    #     self.foldTHF_center = self.get_THF_center_from_2fold()
    #     set_jumpdof_str_int(poseTHF, "JUMPHFfold1", 3, np.linalg.norm(self.get_THF_center_from_2fold()))
    #     set_jumpdof_str_int(poseTHF, "JUMPHFfold1_z", 6, self.get_THF_fold_angle_z_from_2fold())
    #     set_jumpdof_str_int(poseTHF, "JUMPHFfold111", 1, np.linalg.norm(self.get_THFfold_x_vec_from_2fold()))
    #     self.set_rotation_quick(poseTHF, pose2, "JUMPHFfold111_sds", "JUMP21fold111_subunit", "JUMPHFfold111_sds")
    #
    # def __transfer_2toOHF(self, pose2, pose4):
    #     self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
    #     self.fold4_center = self.get_4fold_center_from_2fold()
    #     set_jumpdof_str_int(pose4, "JUMPHFfold1", 3, np.linalg.norm(self.get_4fold_center_from_2fold()))
    #     set_jumpdof_str_int(pose4, "JUMPHFfold1_z", 6, self.get_4_fold_angle_z_from_2fold())
    #     set_jumpdof_str_int(pose4, "JUMPHFfold111", 1, np.linalg.norm(self.get_4fold_x_vec_from_2fold()))
    #     self.set_rotation_quick(pose4, pose2, "JUMPHFfold111_sds", "JUMP21fold111_subunit", "JUMPHFfold111_sds")
    #
    # def __transfer_2toIHF(self, pose2, pose5):
    #     self.fold2_setup.update_dofs_from_pose(pose2, apply_dofs=True)
    #     self.fold5_center = self.get_5fold_center_from_2fold()
    #     set_jumpdof_str_int(pose5, "JUMPHFfold1", 3, np.linalg.norm(self.get_5fold_center_from_2fold()))
    #     set_jumpdof_str_int(pose5, "JUMPHFfold1_z", 6, self.get_5_fold_angle_z_from_2fold())
    #     set_jumpdof_str_int(pose5, "JUMPHFfold111", 1, np.linalg.norm(self.get_5fold_x_vec_from_2fold()))
    #     self.set_rotation_quick(pose5, pose2, "JUMPHFfold111_sds", "JUMP21fold111_subunit", "JUMPHFfold111_sds")
    #
    # def transfer_HFto2(self, pose5, pose2):
    #     self.foldHF_setup.update_dofs_from_pose(pose5, apply_dofs=True)
    #     self.fold2_center = self.get_2fold_center_from_HFfold()
    #     set_jumpdof_str_int(pose2, "JUMP21fold1", 3, np.linalg.norm(self.get_2fold_center_from_HFfold()))
    #     set_jumpdof_str_int(pose2, "JUMP21fold1_z", 6, self.get_2_fold_angle_z_from_HFfold())
    #     set_jumpdof_str_int(pose2, "JUMP21fold111", 1, np.linalg.norm(self.get_2fold_x_vec_from_HFfold()))
    #     self.set_rotation_quick(pose2, pose5, "JUMP21fold111_sds", "JUMPHFfold111_subunit", "JUMP21fold111_sds")
