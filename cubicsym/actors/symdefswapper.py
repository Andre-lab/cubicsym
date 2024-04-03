#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 7/14/22
"""
import numpy as np
from cubicsym.cubicsetup import CubicSetup
from symmetryhandler.mathfunctions import vector_projection_on_subspace, vector_angle
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from symmetryhandler.reference_kinematics import set_jumpdof_str_int, perturb_jumpdof_str_int
import math
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from pyrosetta.rosetta.core.kinematics import Stub, Jump
from symmetryhandler.mathfunctions import vector_projection
from pyrosetta.rosetta.core.scoring import CA_rmsd
from pyrosetta.rosetta.std import map_unsigned_long_unsigned_long
from cubicsym.utilities import pose_cas_are_identical, get_chain_map

class SymDefSwapper:
    """Swaps symmetry between HF-fold, 3F-fold and 2F-fold based symmetries of a cubic symmetrical pose."""

    def __init__(self, pose_X, symdef, visualizer=None, debug_mode=False, check_anchorage_when_debugging=False):
        """Initialize a SymDefSwapper object

        :param pose: Pose_X to read symmetry information from
        :param pose: Symdef file read symmetry information from. Must be the symdef file used to construct pose_X.
        :param visualizer: A visualizer object.
        :param debug_mode: Run in debug mode.
        """
        assert is_symmetric(pose_X)
        if isinstance(symdef, CubicSetup):
            cs = symdef
        else:
            cs = CubicSetup(symdef)
        self.monomer_sizes = pose_X.size()
        self.symmetry_type = cs.cubic_symmetry()
        if cs.is_normalized():
            self.generate_normalized_symmetries(cs)
        else:
            assert cs.is_hf_based(), "For unnormalized symmetry only HF based symmetry is allowed."
            self.generate_symmetry_setups_from_HF(cs, pose_X)
        self.foldHF_setup.apply_dofs()
        self.fold3F_setup.apply_dofs()
        self.fold2F_setup.apply_dofs()
        self.visualizer = visualizer
        self.debug_mode = debug_mode
        self.check_anchorage_when_debugging = check_anchorage_when_debugging
        self.sanity_check(pose_X)

    def generate_normalized_symmetries(self, pose_X):
        """Generates normalized HF, 3F and 2F CubicSetups."""
        self.foldHF_setup = CubicSetup()
        self.foldHF_setup.load_norm_symdef(self.symmetry_type, "HF")
        self.fold3F_setup = CubicSetup()
        self.fold3F_setup.load_norm_symdef(self.symmetry_type, "3F")
        self.fold2F_setup = CubicSetup()
        self.fold2F_setup.load_norm_symdef(self.symmetry_type, "2F")

    def generate_symmetry_setups_from_HF(self, cs, pose):
        """Generates HF, 3F and 2F CubicSetups."""
        self.foldHF_setup = cs
        if self.symmetry_type == "I":
            self.fold3F_setup = self.foldHF_setup.create_I_3fold_based_symmetry()
            self.fold2F_setup = self.foldHF_setup.create_I_2fold_based_symmetry()
            if self.foldHF_setup.has_extra_chains(pose):
                self.fold3F_setup = self.fold3F_setup.add_extra_chains()
                self.fold2F_setup = self.fold2F_setup.add_extra_chains()
        elif self.symmetry_type == "O":
            self.fold3F_setup = self.foldHF_setup.create_O_3fold_based_symmetry()
            self.fold2F_setup = self.foldHF_setup.create_O_2fold_based_symmetry()
        elif self.symmetry_type == "T":
            self.fold3F_setup = self.foldHF_setup.create_T_3fold_based_symmetry()
            self.fold2F_setup = self.foldHF_setup.create_T_2fold_based_symmetry()
        else:
            raise ValueError(f"Symmetry type: {self.symmetry_type}, is not accepted")

    def sanity_check(self, pose_X):
        """Asserts that
        1. All the internal VRTs in the CubicSetups overlaps with the ones in the pose.
        2. Assert that pose_HF, pose_3F, pose_2 overlap with eachother."""
        pose_HF, pose_3F, pose_2F = self.create_remaing_folds(pose_X)
        # 1)
        self.foldHF_setup.vrts_overlap_with_pose(pose_HF, update_and_apply_dofs=True)
        self.fold3F_setup.vrts_overlap_with_pose(pose_3F, update_and_apply_dofs=True)
        self.fold2F_setup.vrts_overlap_with_pose(pose_2F, update_and_apply_dofs=True)
        chain_map = get_chain_map(CubicSetup.cubic_symmetry_from_pose(pose_HF), self.foldHF_setup.righthanded)
        # 2)
        assert pose_cas_are_identical(pose_HF, pose_3F, pose_2F, map_chains=chain_map, atol=1e-3)

    def create_remaing_folds(self, pose_X):
        """Creates the remaing folds from pose_X where pose_X can be have a base of HF, 3F and 2F.
        For instance if pose_X is HF-based, it will output pose_HF, pose_3F, pose_2F where the latter 2 are constucted from pose_X."""
        base = CubicSetup.get_base_from_pose(pose_X)
        if base == "HF":
            pose_HF = pose_X
            pose_3F = self.create_3fold_pose(pose_X)
            pose_2F = self.create_2fold_pose(pose_X)
        elif base == "3F":
            pose_HF = self.create_hffold_pose(pose_X)
            pose_3F = pose_X
            pose_2F = self.create_2fold_pose(pose_X)
        elif base == "2F":
            pose_HF = self.create_hffold_pose(pose_X)
            pose_3F = self.create_3fold_pose(pose_X)
            pose_2F = pose_X
        else:
            raise ValueError(f"Symmetry of type {base} is not accepted.")
        return pose_HF, pose_3F, pose_2F

    def set_rotation_quick(self, poseB, poseA, jiB, jiA, setupA, setupB):
        """Transfers the rotation of poseA onto poseB.

        Algorithm description
        ----------------------
        The goal is to make the anchor stubs of PoseA and poseB overlap in space. That means the
        downstream stub of the 'JUMP{jiB}fold111_subunit' jump should overlap with the stub of the downstream 'JUMP{jiA}fold111_subunit' jump.
        This is done creating a Jump that copies what the jump between VRT{jiB}fold111_sds and the poseA anchor Stub and inserts into
        poseB's 'JUMP{jiB}fold111_subunit' jump.
        """
        stub1 = Stub(poseB.conformation().downstream_jump_stub(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds")))
        if self.debug_mode:
            assert setupB.get_map_pose_resi_to_vrt(poseB)[poseB.fold_tree().downstream_jump_residue(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_sds"))] == f"VRT{jiB}fold111_sds"
        stub2 = Stub(poseA.conformation().downstream_jump_stub(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit")))
        if self.debug_mode:
            assert poseA.fold_tree().downstream_jump_residue(sym_dof_jump_num(poseA, f"JUMP{jiA}fold111_subunit")) == setupA.get_anchor_residue(poseA)
        jump = Jump(stub1, stub2)
        if self.debug_mode:
            assert np.isclose(np.array(jump.get_translation()), [0, 0, 0], atol=1e-2).all(), "The stubs should overlap at this point"
        poseB.set_jump(sym_dof_jump_num(poseB, f"JUMP{jiB}fold111_subunit"), jump) # _sds and subunit
        if self.debug_mode:
            self.check_main_chain_overlap(poseA, poseB)

    def check_main_chain_overlap(self, poseA, poseB):
        """Checks if the first chain of poseA and poseB overlaps.
        Theres a bug, in Rosetta I believe, where for some reason the RMSD just returns 0.0 even though it is not in the case
        in some instances. Therefore this funcitons also test the overlap of the first CA atom of poseA and poseB."""
        m = map_unsigned_long_unsigned_long()
        for ri in range(1, poseA.chain_end(1) + 1):
            m[ri] = ri
        rmsd = CA_rmsd(poseA, poseB, m)
        assert math.isclose(rmsd, 0, abs_tol=1e-3)
        assert np.isclose(np.array(poseA.residue(1).atom("CA").xyz()), np.array(poseB.residue(1).atom("CA").xyz()), atol=1e-3).all()

    def create_hffold_pose(self, pose, check_anchor_is_zero=True):
        """Creates a HF-based symmetric pose from input pose"""
        poseHF = CubicSetup.make_asymmetric_pose(pose, check_anchor_is_zero= self.check_anchorage_when_debugging )
        self.foldHF_setup.make_symmetric_pose(poseHF)
        self.transfer_poseA2B(pose, poseHF)
        return poseHF

    def create_3fold_pose(self, pose, check_anchor_is_zero=True):
        """Creates a 3F-based symmetric pose from input pose"""
        pose3 = CubicSetup.make_asymmetric_pose(pose, check_anchor_is_zero= self.check_anchorage_when_debugging )
        self.fold3F_setup.make_symmetric_pose(pose3)
        self.transfer_poseA2B(pose, pose3)
        return pose3

    def create_2fold_pose(self, pose, check_anchor_is_zero=True):
        """Creates a 2F-based symmetric pose from input pose"""
        pose2 = CubicSetup.make_asymmetric_pose(pose, check_anchor_is_zero= self.check_anchorage_when_debugging )
        self.fold2F_setup.make_symmetric_pose(pose2)
        self.transfer_poseA2B(pose, pose2)
        return pose2

    def get_setup_and_jumpid(self, pose):
        """Returns the CubicSetup that corresponds to the pose as well as the 2-letter jumpidentifier (ex. HF) that identifies a jump."""
        base = CubicSetup.get_base_from_pose(pose)
        if base == "HF":
            setup = self.foldHF_setup
            jumpid = "HF"
        elif base == "3F":
            setup = self.fold3F_setup
            jumpid = "31"
        elif base == "2F":
            setup = self.fold2F_setup
            jumpid = "21"
        else:
            raise ValueError(base + " is not accepted.")
        return setup, jumpid

    def transfer_poseA2B(self, poseA, poseB):
        """Transfers the translation and rotation of each subunit of poseA onto poseB where poseA and poseB
        have different symmetry setups.

        Algorithm description
        ----------------------
        2 things need to happen in order to transfer poseA to poseB
          1. Translating poseB onto poseA (= Align VRT{B}fold111 onto VRT{A}fold111)
          2. Rotate poseB subunit onto poseA (= Align the symmetry anchor stub of poseB onto the poseA anchor residue stub)
        See 1) and 2) in the comments below
        """

        # Get variables depending on the poses and their respective CubicSetups
        setupA, jiA = self.get_setup_and_jumpid(poseA)
        setupB, jiB = self.get_setup_and_jumpid(poseB)
        assert jiA != jiB, "Does it make sense to use the same setup?? I dont think so!"

        # update SetupA, the setup we are trying to get the the information from
        setupA.update_dofs_from_pose(poseA, apply_dofs=True)
        setupB.update_dofs_from_pose(poseB, apply_dofs=True)

        if self.debug_mode:
            assert np.isclose(np.array(poseB.residue(setupB.get_anchor_residue(poseB)).atom("CA").xyz()), setupB.get_vrt(f"VRT{jiB}fold111_sds").vrt_orig, atol=1e-2).all()
            assert np.isclose(np.array(poseA.residue(setupA.get_anchor_residue(poseA)).atom("CA").xyz()), setupA.get_vrt(f"VRT{jiA}fold111_sds").vrt_orig, atol=1e-2).all()

        # 1)
        # To first align the 2 VRT's we use 3 dofs (a z translation, a x translation and an angle_z rotation).
        # There are 2 cases to consider one where VRT{jiA}fold111 is above the x, y plane of VRT{jiB}fold1 and one where it is below.
        # If it is located below the plane we need to do a negative z translation, else we do a positive one.

        # We first check if the point is in the above or below the plane
        point_to_get_to = setupA.get_vrt(f"VRT{jiA}fold111")._vrt_orig
        x = setupB.get_vrt(f"VRT{jiB}fold11")._vrt_x
        y = setupB.get_vrt(f"VRT{jiB}fold11")._vrt_y
        z = - setupB.get_vrt(f"VRT{jiB}fold11")._vrt_z # minus because of rosettas convention
        plane_to_vec = point_to_get_to - vector_projection_on_subspace(point_to_get_to, x, y)
        proj = vector_projection(plane_to_vec, z)
        cross_angle = vector_angle(proj, z)
        # + = the same, - = the opposite
        dir = 1 if cross_angle < 90 else - 1
        # Then we create the z translation.This is done by projecting onto the point_to_get_t taking the dir variable into account
        z_vec = vector_projection(point_to_get_to, dir * z)
        z = dir * np.linalg.norm(z_vec)
        # Then we get the x translation. This is the distance from the projection to the point_to_get_to.
        x_vec = point_to_get_to - z_vec
        x = np.linalg.norm(x_vec)
        # We update SetupB to the current x and z translations in the setup. This is need in order to get angle_z
        # to get angle_z we need to consider the angle between the current x translation we have and where the point is right now (x_vec)
        setupB.set_dof(f"JUMP{jiB}fold1", "z", "translation", z)
        setupB.set_dof(f"JUMP{jiB}fold111", "x", "translation", x)
        setupB.apply_dofs()
        current_x_vec = setupB.get_vrt(f"VRT{jiB}fold111")._vrt_orig - setupB.get_vrt(f"VRT{jiB}fold11")._vrt_orig
        cross_angle = vector_angle(np.cross(current_x_vec, x_vec), z_vec)
        dir = - 1 * dir if cross_angle > 90 else 1 * dir
        angle_z = dir * vector_angle(current_x_vec, x_vec)
        if self.debug_mode:
            setupB.set_dof(f"JUMP{jiB}fold1_z", "z", "rotation", setupB._dofs[f"JUMP{jiB}fold1_z"][0][2] + angle_z)
            setupB.apply_dofs()
            assert np.isclose(setupB.get_vrt(f"VRT{jiB}fold111")._vrt_orig, setupA.get_vrt(f"VRT{jiA}fold111")._vrt_orig, atol=1e-3).all()
        # now we finally set the dofs in the actual pose
        set_jumpdof_str_int(poseB, f"JUMP{jiB}fold1", 3, z)
        perturb_jumpdof_str_int(poseB, f"JUMP{jiB}fold1_z", 6, angle_z)
        set_jumpdof_str_int(poseB, f"JUMP{jiB}fold111", 1, x)

        # 2) see the set_rotation_quick() documentation
        self.set_rotation_quick(poseB, poseA, jiB, jiA, setupA, setupB)