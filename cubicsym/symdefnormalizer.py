#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SymdefNormalizer class
@Author: Mads Jeppesen
@Date: 12/13/22
"""
from cubicsym.cubicsetup import CubicSetup
from symmetryhandler.mathfunctions import vector_angle, vector_projection_on_subspace, rotation_matrix
import math
import scipy
import numpy as np
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_str
from math import isclose
from copy import deepcopy
from symmetryhandler.reference_kinematics import get_jumpdof_str_str, get_dofs
from symmetryhandler.mathfunctions import vector_projection, vector_angle
from cubicsym.actors.symdefswapper import SymDefSwapper
from io import StringIO
from pyrosetta.rosetta.core.kinematics import Stub, Jump
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from cubicsym.cubicmontecarlo import CubicMonteCarlo
from cubicsym.cubicdofs import CubicDofs
from cubicsym.setupaligner import SetupAligner

class SymdefNormalizer:

    def __init__(self):
        pass

    def get_triangle_vector_angle(self, cs):
        """The angle between the vector leading from the HF fold vrt to the main subunit vrt and a vector drawn from the bottom of a triangle
        trough the top consisting of the vrts of the 3-fold subunits."""
        triangle_bottom_midpoint = cs.get_vrt("VRT2fold121").vrt_orig + (
                    cs.get_vrt("VRT3fold111")._vrt_orig - cs.get_vrt("VRT2fold121")._vrt_orig) / 2  # good
        triangle_top_point = cs.get_vrt("VRTHFfold111").vrt_orig
        vector_through_triangle = triangle_top_point - triangle_bottom_midpoint
        vector_through_triangle_projected = vector_projection_on_subspace(vector_through_triangle, [1, 0, 0], [0, 1, 0])
        vector_to_triangle_top_point = cs.get_vrt("VRTHFfold111")._vrt_orig - cs.get_vrt("VRTHFfold11")._vrt_orig
        return vector_angle(vector_through_triangle_projected, - vector_to_triangle_top_point)

    def get_5fold_angle_to_z(self, cs):
        v1 = cs.get_vrt("VRT2fold111")._vrt_orig - cs.get_vrt("VRT2fold11")._vrt_orig
        v2 = cs.get_vrt("VRT2fold121")._vrt_orig - cs.get_vrt("VRT2fold11")._vrt_orig
        v1n, v2n = scipy.linalg.orth(np.array([v1, v2]).T).T
        z_on_5fold = vector_projection_on_subspace([0, 0, 1], v1n, v2n)
        return vector_angle(v2, z_on_5fold)

    def create_normalized_structure(self, cs, pose):
        cs.make_symmetric_pose(pose)
        pose.pdb_info().name("normalized_pose")
        return pose

    def recreate_original_structure(self, pose, cs_norm, cs_org, rotation_angle, rotate_z=False):
        assert not is_symmetric(pose), "pose should not be symmetric"
        if cs_norm.is_hf_based():
            cs_norm.make_symmetric_pose(pose)
            z, angle_z, x = self.get_hf_fold_info(cs_org, pose)
            cs_norm.set_dof("JUMPHFfold111", "x", "translation", x)
            cs_norm.set_dof("JUMPHFfold1_z", "z", "rotation", -angle_z)
            perturb_jumpdof_str_str(pose, "JUMPHFfold1_z", "angle_z", -rotation_angle)
        elif cs_norm.is_3f_based():
            cs_norm.make_symmetric_pose(pose)
            perturb_jumpdof_str_str(pose, "JUMP3fold1_z", "angle_z", -rotation_angle)
        elif cs_norm.is_2f_based():
            cs_norm.make_symmetric_pose(pose)
            perturb_jumpdof_str_str(pose, "JUMP2fold1_z", "angle_z", -rotation_angle)
        if rotate_z:
            pose.rotate(rotation_matrix([0, 0, 1], rotation_angle))
            pose.pdb_info().name("original_structure_rot")
        else:
            pose.pdb_info().name("original_structure")
        return pose

    def can_reconstruct(self, pose_unsymmetrized, pose_symmetrized, pose_crystal, cs, angle_z, z, x, pmm=None):
        if cs.is_hf_based():
            cs.set_dof("JUMPHFfold1", "z", "translation", z)
            cs.set_dof("JUMPHFfold111", "x", "translation", x)
            cs.set_dof("JUMPHFfold1_z", "z", "rotation", -angle_z)
        elif cs.is_3f_based():
            cs.set_dof("JUMP31fold1", "z", "translation", z)
            cs.set_dof("JUMP31fold111", "x", "translation", x)
            cs.set_dof("JUMP31fold1_z", "z", "rotation", -angle_z)
        elif cs.is_2f_based():
            cs.set_dof("JUMP21fold1", "z", "translation", z)
            cs.set_dof("JUMP21fold111", "x", "translation", x)
            cs.set_dof("JUMP21fold1_z", "z", "rotation", -angle_z)
        pose_norm_symmetrized = pose_unsymmetrized.clone()
        cs.make_symmetric_pose(pose_norm_symmetrized)
        # perturb_jumpdof_str_str(pose_norm_symmetrized, "JUMPHFfold1_z", "angle_z", -angle_z)
        crystal_rmsd = cs.CA_rmsd_hf_map(pose_norm_symmetrized, pose_crystal)
        symmetric_rmsd = cs.CA_rmsd_hf_map(pose_norm_symmetrized, pose_symmetrized)
        print("crystal rmsd from norm", crystal_rmsd, "symmetric_rmsd from norm", symmetric_rmsd)
        assert isclose(crystal_rmsd, 0, abs_tol=1e-3)
        assert isclose(symmetric_rmsd, 0, abs_tol=1e-3)

    def normalized_setup(self, cs_norm, out, trans_base=10):
        cs_norm_reset = deepcopy(cs_norm)
        cs_norm_reset.set_dof("JUMPHFfold1", "z", "translation", trans_base)
        cs_norm_reset.set_dof("JUMPHFfold111", "x", "translation", trans_base)

    def get_dof_info(self, cs, pose):
        """Get 3 fold values of z, angle_z and x from the old CubicSetup"""
        pose = pose.clone()
        cs.make_symmetric_pose(pose)
        if cs.is_hf_based():
            z = get_jumpdof_str_str(pose, "JUMPHFfold1", "z")
            angle_z = get_jumpdof_str_str(pose, "JUMPHFfold1_z", "angle_z")
            x = get_jumpdof_str_str(pose, "JUMPHFfold111", "x")
        elif cs.is_3f_based():
            z = get_jumpdof_str_str(pose, "JUMP31fold1", "z")
            angle_z = get_jumpdof_str_str(pose, "JUMP31fold1_z", "angle_z")
            x = get_jumpdof_str_str(pose, "JUMP31fold111", "x")
        elif cs.is_2f_based():
            z = get_jumpdof_str_str(pose, "JUMP21fold1", "z")
            angle_z = get_jumpdof_str_str(pose, "JUMP21fold1_z", "angle_z")
            x = get_jumpdof_str_str(pose, "JUMP21fold111", "x")
        return z, angle_z, x

    def get_hf_fold_info(self, cs, pose_unsymmetrized):
        """Get 3 fold values of z, angle_z and x from the old CubicSetup"""
        pose_hf_original_symmetrized = pose_unsymmetrized.clone()
        cs.make_symmetric_pose(pose_hf_original_symmetrized)
        z = get_jumpdof_str_str(pose_hf_original_symmetrized, "JUMPHFfold1", "z")
        angle_z = get_jumpdof_str_str(pose_hf_original_symmetrized, "JUMPHFfold1_z", "angle_z")
        x = get_jumpdof_str_str(pose_hf_original_symmetrized, "JUMPHFfold111", "x")
        return z, angle_z, x

    def get_3_fold_info(self, cs, pose_unsymmetrized):
        """Get 3 fold values of z, angle_z and x from the old CubicSetup"""
        cs_3fold = cs.create_T_3fold_based_symmetry()
        pose_fold3_symmetrized = pose_unsymmetrized.clone()
        cs_3fold.make_symmetric_pose(pose_fold3_symmetrized)
        z = get_jumpdof_str_str(pose_fold3_symmetrized, "JUMP31fold1", "z")
        angle_z = get_jumpdof_str_str(pose_fold3_symmetrized, "JUMP31fold1_z", "angle_z")
        x = get_jumpdof_str_str(pose_fold3_symmetrized, "JUMP31fold111", "x")
        return z, angle_z, x

    def get_2_fold_info(sefl, cs, pose_unsymmetrizedm):
        """Get 3 fold values of z, angle_z and x from the old CubicSetup"""
        cs_2fold = cs.create_T_2fold_based_symmetry()
        pose_fold2_symmetrized = pose_unsymmetrized.clone()
        cs_2fold.make_symmetric_pose(pose_fold2_symmetrized)
        z = get_jumpdof_str_str(pose_fold2_symmetrized, "JUMP21fold1", "z")
        angle_z = get_jumpdof_str_str(pose_fold2_symmetrized, "JUMP21fold1_z", "angle_z")
        x = get_jumpdof_str_str(pose_fold2_symmetrized, "JUMP21fold111", "x")
        return z, angle_z, x

    # def only_rotate_around_global_z(self, cs, hf_rot_angle):
    #     R_global_z_z = rotation_matrix([0, 0, 1], -hf_rot_angle)
    #     # for HF
    #     R_their_z = rotation_matrix(-cs.get_vrt("VRTHFfold1_z")._vrt_z, hf_rot_angle)
    #     for vrt in cs.get_downstream_connections("JUMPHFfold1_z"):
    #         if vrt != "SUBUNIT":
    #             cs.get_vrt(vrt).rotate(R_global_z_z)
    #     # for 3
    #     R_their_z = rotation_matrix(-cs.get_vrt("VRT3fold1_z")._vrt_z, hf_rot_angle)
    #     for vrt in cs.get_downstream_connections("JUMP3fold1_z_tref"):
    #         if vrt != "SUBUNIT":
    #             cs.get_vrt(vrt).rotate(R_global_z_z)
    #     # for 2
    #     R_their_z = rotation_matrix(-cs.get_vrt("VRT2fold1_z")._vrt_z, hf_rot_angle)
    #     for vrt in cs.get_downstream_connections("JUMP2fold1_z_tref"):
    #         if vrt != "SUBUNIT":
    #             cs.get_vrt(vrt).rotate(R_global_z_z)

    def apply(self, pose_in, symdef, final_z_trans:float = None, final_x_trans:float = None):
        """Creates a CubicSetup that is normalized which means that:
        1. The 3-fold vrts z vectors all point into the same point in space (triangle_vector_angle == 0 degrees).
        2. The x translation is along the global x-axis as previously. (But because of 1, the whole structure has to move about the
           global z-axis.)
        Because of condition 2, 2 poses with the exact same dofs will not overlap in space unless moved around the global z-axis.
        Returns the normalized CubicSetup and the angle that the structure rotated about along its HF-folds and the global z-axis
        """
        # gather information from the symdef file
        cs_in = CubicSetup()
        cs_in.read_from_file(symdef)
        if not cs_in.calculate_if_rightanded():
            raise NotImplementedError("Not yet tested on lefthanded symmetries!")
        cs_in.apply_dofs()
        hf_rot_angle = (self.get_5fold_angle_to_z(cs_in) - cs_in.hf_rotation_angle_per_subunit() / 2)
        # information gathered, now we modify the initial symdef
        cs_hf_norm = CubicSetup()
        cs_hf_norm.read_from_file(symdef)
        # 1. rotate around their z
        # 2. rotate around global z
        R_global_z_z = rotation_matrix([0, 0, 1], -hf_rot_angle)
        # for HF
        R_their_z = rotation_matrix(-cs_hf_norm.get_vrt("VRTHFfold1_z")._vrt_z, hf_rot_angle)
        for vrt in cs_hf_norm.get_downstream_connections("JUMPHFfold1_z"):
            if vrt != "SUBUNIT":
                cs_hf_norm.get_vrt(vrt).rotate(R_their_z)
                cs_hf_norm.get_vrt(vrt).rotate(R_global_z_z)
        # for 3
        R_their_z = rotation_matrix(-cs_hf_norm.get_vrt("VRT3fold1_z")._vrt_z, hf_rot_angle)
        for vrt in cs_hf_norm.get_downstream_connections("JUMP3fold1_z_tref"):
            if vrt != "SUBUNIT":
                cs_hf_norm.get_vrt(vrt).rotate(R_their_z)
                cs_hf_norm.get_vrt(vrt).rotate(R_global_z_z)
        # for 2
        R_their_z = rotation_matrix(-cs_hf_norm.get_vrt("VRT2fold1_z")._vrt_z, hf_rot_angle)
        for vrt in cs_hf_norm.get_downstream_connections("JUMP2fold1_z_tref"):
            if vrt != "SUBUNIT":
                cs_hf_norm.get_vrt(vrt).rotate(R_their_z)
                cs_hf_norm.get_vrt(vrt).rotate(R_global_z_z)
        cs_hf_norm.anchor = "COM"
        # check if normalized correctly
        self.check_if_normalized(cs_hf_norm)
        # make the 3-fold and 2-fold based normalized cubicsetups
        pose = pose_in.clone()
        cs_hf_norm.make_symmetric_pose(pose)
        sds = SymDefSwapper(pose, StringIO(cs_hf_norm.make_symmetry_definition()))
        cs_3_norm, cs_2_norm = sds.fold3_setup, sds.fold2_setup
        for norm, base in zip((cs_hf_norm, cs_3_norm, cs_2_norm), ("HF", "31", "21")):
            # apply the default translation if set
            if final_z_trans is not None:
                norm.set_dof(f"JUMP{base}fold1", "z", "translation", final_z_trans)
            if final_x_trans is not None:
                norm.set_dof(f"JUMP{base}fold111", "x", "translation", final_x_trans)
        return cs_hf_norm, cs_3_norm, cs_2_norm, hf_rot_angle

    def check_if_normalized(self, cs_hf_norm):
        cs = deepcopy(cs_hf_norm)
        cs.apply_dofs()
        assert(math.isclose(self.get_triangle_vector_angle(cs), 0, abs_tol=1e-3)), "File is not normalized correctly!"
