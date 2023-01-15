#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SetupAligner class and RMSDscore class
@Author: Mads Jeppesen
@Date: 12/28/22
"""
from pyrosetta import Pose
from pyrosetta.rosetta.core.scoring import CA_rmsd
from cubicsym.cubicsetup import CubicSetup
from cubicsym.cubicmontecarlo import CubicMonteCarlo
from cubicsym.cubicdofs import CubicDofs
from pyrosetta.rosetta.protocols.rigid import RigidBodyDofAdaptiveMover
from pyrosetta import Pose
from copy import deepcopy
import numpy as np
from symmetryhandler.mathfunctions import vector_projection, vector_angle, rotation_matrix
from symmetryhandler.reference_kinematics import set_jumpdof_str_str
import math

class RMSDscore:

    def __init__(self, cs: CubicSetup, reference: Pose, map_to_hf=False, multiplier=100, same_handedness=True):
        self.cs = cs
        self.reference = reference
        self.map_to_hf = map_to_hf
        self.multiplier = multiplier
        self.same_handedness = same_handedness

    def get_rmsd(self, pose):
        return self.score(pose) / self.multiplier

    def score(self, pose):
        if self.map_to_hf:
            return self.cs.CA_rmsd_hf_map(pose, self.reference, same_handedness=self.same_handedness) * self.multiplier
        else:
            return CA_rmsd(pose, self.reference) * self.multiplier

class SetupAligner:

    def __init__(self, al_from: CubicSetup, al_to: CubicSetup, monomeric_pose: Pose=None, overlap_tol=1e-1, use_hf_chain_mapping=False, same_handedness=True,
                 x_start= None, z_start=None, angle_z_start=None, behavior="global", al_to_monomeric=None, al_from_monomeric=None,
                 assert_overlap_tol=True):
        assert behavior in ("global", "align_then_rotate", "fold_rotation")
        self.al_from = deepcopy(al_from)
        self.al_to = deepcopy(al_to)
        # assert self.has_same_base(al_from, al_to), "al_from and al_to must have the same base: HF, 3F or 2F."
        # assert al_to.is_hf_based()
        if not al_to_monomeric is None:
            self.pose_to = al_to_monomeric.clone()
        else:
            self.pose_to = monomeric_pose.clone()
        if not al_from_monomeric is None:
            self.pose_from = al_from_monomeric.clone()
        else:
            self.pose_from = monomeric_pose.clone()
        self.overlap_tol = overlap_tol
        self.final_rmsd = None
        self.use_hf_chain_mapping = use_hf_chain_mapping
        self.same_handedness = same_handedness
        self.x_start = x_start
        self.z_start = z_start
        self.angle_z_start = angle_z_start
        self.behavior = behavior
        self.assert_overlap_tol = assert_overlap_tol
        # if we are comparing 2 CubicSetups that have been rotated around the global z axis we have to rotate al_to CubicSetup,
        # assuming the al_from has been rotated around the global z axis already. This is important when comparing a normalized CubicSetup
        # to one that is not.
        # if hf_rotation_angle is not None:
        #     self.rotate_around_global_z(self.al_to, hf_rotation_angle)

    def has_same_base(self, cs1, cs2):
        """returns if cs1 and cs2 have the same base"""
        return cs1.get_base() == cs2.get_base()

    def get_extra_options(self, dof_params):
        """HACK: You cannot parse individual keyword parameters to these c++ objects, so you
        have specify them all and then change them before parsing them as below"""
        if dof_params:  # check for not empty
            default = {"step_type": "gauss", "param1": 0.5, "param2": 0.0, "min_pertubation": 0.01,
                       "limit_movement": False, "max": 0, "min": 0}
            default.update(dof_params)
            return default.values()
        else:
            return []

    def rotate_around_global_z(self, cs, hf_rot_angle):
        raise NotImplementedError("Is not implemented correctly!")
        # testing i was using:
        # sa3.rotate_around_global_z(sa3.al_from, hf_rot_angle)
        # pose_from = pose_original_unsymmetrized.clone()
        # sa3.al_from._unapplied_vrts = None
        # sa3.al_from.make_symmetric_pose(pose_from)
        R_global_z_z = rotation_matrix([0, 0, 1], -hf_rot_angle)
        #t_ref_jumps = (jump for jump in cs._jumps.keys() if "z_tref" in jump)
        t_ref_jumps = (jump for jump in cs._jumps.keys() if "fold1" in jump[-5:])
        for jump in t_ref_jumps:
            for vrt in cs.get_downstream_connections(jump):
                if vrt != "SUBUNIT":
                    cs.get_vrt(vrt).rotate(R_global_z_z)

    def make_symmetrical_pose_and_check_overlap(self):
        """Makes symmetrical poses and assert that the vrt orig postions overlap."""
        self.al_from.make_symmetric_pose(self.pose_from)
        self.al_to.make_symmetric_pose(self.pose_to)
        self.anchor_from = self.al_from.get_anchor_residue(self.pose_from)
        self.anchor_to = self.al_from.get_anchor_residue(self.pose_to)
        assert self.anchor_from == self.anchor_to, "al_to and al_from must have the same anchor residues."
        assert all(np.isclose(np.array(self.pose_from.residue(self.anchor_from).atom("CA").xyz()),
                             np.array(self.pose_to.residue(self.anchor_to).atom("CA").xyz()), atol=1e-3)), "CA of the residues must overlap at this point."
        self.pose_from.pdb_info().name("from")
        self.pose_to.pdb_info().name("to")

    def apply(self):
        """Applies the alignment."""
        if self.behavior == "align_then_rotate":
            self.align_trans()
            self.make_symmetrical_pose_and_check_overlap()
            self.align_rot()
        else:
            self.al_from.make_symmetric_pose(self.pose_from)
            self.al_to.make_symmetric_pose(self.pose_to)
            self.anchor_from = self.al_from.get_anchor_residue(self.pose_from)
            self.anchor_to = self.al_from.get_anchor_residue(self.pose_to)
            if self.behavior == "fold_rotation":
                self.global_search()
            else:
                self.global_search()

    def fold_rotation(self):
        if self.al_from.is_hf_based():
            dofs = {
                "JUMPHFfold1_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold1_z", "angle_z", self.angle_z_start)
        elif self.al_from.is_3f_based():
            dofs = {
                "JUMP31fold1_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold1_z", "angle_z", self.angle_z_start)
        elif self.al_from.is_2f_based():
            dofs = {
                "JUMP21fold1_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold1_z", "angle_z", self.angle_z_start)
        rbmover = RigidBodyDofAdaptiveMover("rotations_only")
        for jump_name, jumpdof_params in dofs.items():
            for dof_name, dof_params in jumpdof_params.items():
                rbmover.add_jump(self.pose_from, jump_name,
                                 {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y", "z": "z"}[
                                     dof_name],
                                 *self.get_extra_options(dof_params))
        self.rmsd_score = RMSDscore(self.al_from, self.pose_to, map_to_hf=self.use_hf_chain_mapping,
                                    same_handedness=self.same_handedness)
        cmc = CubicMonteCarlo(self.rmsd_score, CubicDofs(self.pose_from, dofs))
        cmc.reset(self.pose_from)
        print("init", cmc.lowest_score)

        n = 0
        for it in range(10000 * 5):
            rbmover.apply(self.pose_from)
            cmc.apply(self.pose_from)
            if it % 500 == 0:
                print("done 500")
                print(cmc.lowest_score)
                if self.rmsd_score.get_rmsd(self.pose_from) <= self.overlap_tol:
                    break
                else:
                    n += 0.001
                    for jump_name, jumpdof_params in dofs.items():
                        for dof_name, dof_params in jumpdof_params.items():
                            new_param1 = 0.1 - n
                            rbmover.set_param1(new_param1, jump_name,
                                               {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y",
                                                "z": "z"}[
                                                   dof_name])
                    print("new param1", new_param1)

        cmc.recover_lowest_scored_pose(self.pose_from)
        self.al_from.update_dofs_from_pose(self.pose_from)
        self.final_rmsd = self.rmsd_score.get_rmsd(self.pose_from)
        if self.assert_overlap_tol:
            assert self.final_rmsd <= self.overlap_tol

    def global_search(self):
        if self.al_from.is_hf_based():
            dofs = {
                "JUMPHFfold1": {"z": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMPHFfold1_z": {"angle_z": {"param1": 0.1}},
                "JUMPHFfold111": {"x": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMPHFfold111_x": {"angle_x": {"param1": 0.1}},
                "JUMPHFfold111_y": {"angle_y": {"param1": 0.1}},
                "JUMPHFfold111_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMPHFfold1_z", "angle_z", self.angle_z_start)
        elif self.al_from.is_3f_based():
            dofs = {
                "JUMP31fold1": {"z": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMP31fold1_z": {"angle_z": {"param1": 0.1}},
                "JUMP31fold111": {"x": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMP31fold111_x": {"angle_x": {"param1": 0.1}},
                "JUMP31fold111_y": {"angle_y": {"param1": 0.1}},
                "JUMP31fold111_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP31fold1_z", "angle_z", self.angle_z_start)
        elif self.al_from.is_2f_based():
            dofs = {
                "JUMP21fold1": {"z": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMP21fold1_z": {"angle_z": {"param1": 0.1}},
                "JUMP21fold111": {"x": {"param1": 0.1}},#, "limit_movement": True, "min":-5, "max": 5}},
                "JUMP21fold111_x": {"angle_x": {"param1": 0.1}},
                "JUMP21fold111_y": {"angle_y": {"param1": 0.1}},
                "JUMP21fold111_z": {"angle_z": {"param1": 0.1}},
            }
            if self.z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold1", "z", self.z_start)
            if self.x_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold111", "x", self.x_start)
            if self.angle_z_start is not None:
                set_jumpdof_str_str(self.pose_from, "JUMP21fold1_z", "angle_z", self.angle_z_start)
        rbmover = RigidBodyDofAdaptiveMover("rotations_only")
        for jump_name, jumpdof_params in dofs.items():
            for dof_name, dof_params in jumpdof_params.items():
                rbmover.add_jump(self.pose_from, jump_name,
                                 {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y", "z": "z"}[dof_name],
                                 *self.get_extra_options(dof_params))
        self.rmsd_score = RMSDscore(self.al_from, self.pose_to, map_to_hf=self.use_hf_chain_mapping, same_handedness=self.same_handedness)
        cmc = CubicMonteCarlo(self.rmsd_score, CubicDofs(self.pose_from, dofs))
        cmc.reset(self.pose_from)
        print("init", cmc.lowest_score)

        n = 0
        self.steps = 0
        for it in range(10000):
            rbmover.apply(self.pose_from)
            cmc.apply(self.pose_from)
            if it % 500 == 0:
                print("done 500")
                print(cmc.lowest_score)
                if self.rmsd_score.get_rmsd(self.pose_from) <= self.overlap_tol:
                    break
                else:
                    n += 0.001
                    for jump_name, jumpdof_params in dofs.items():
                        for dof_name, dof_params in jumpdof_params.items():
                            new_param1 = 0.1 - n
                            rbmover.set_param1(new_param1, jump_name,
                                               {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y", "z": "z"}[
                                                   dof_name])
                    print("new param1", new_param1)
            self.steps = it
        cmc.recover_lowest_scored_pose(self.pose_from)
        self.al_from.update_dofs_from_pose(self.pose_from)
        self.final_rmsd = self.rmsd_score.get_rmsd(self.pose_from)
        if self.assert_overlap_tol:
            assert self.final_rmsd <= self.overlap_tol

    def align_trans(self):
        tf = "HF" # asumming its always hf
        assert self.al_to.is_hf_based()
        if self.al_from.is_hf_based():
            ff = "HF"
        elif self.al_from.is_3f_based():
            ff = "31"
        elif self.al_from.is_2f_based():
            ff = "21"

        self.al_to.apply_dofs()

        # To get the z translation we need to project onto the al_to z axis onto
        z = np.linalg.norm(vector_projection(self.al_to.get_vrt(f"VRT{tf}fold111")._vrt_orig, self.al_from.get_vrt(f"VRT{ff}fold11")._vrt_z))
        self.al_from.set_dof(f"JUMP{ff}fold1", "z", "translation", z)

        # to get the x translation
        self.al_from.apply_dofs()
        x = np.linalg.norm(self.al_to.get_vrt(f"VRT{tf}fold111")._vrt_orig - self.al_from.get_vrt(f"VRT{ff}fold11")._vrt_orig)
        self.al_from.set_dof(f"JUMP{ff}fold111", "x", "translation", x)

        # angle_z
        angle = vector_angle(self.al_from.get_vrt(f"VRT{ff}fold111")._vrt_orig - self.al_from.get_vrt(f"VRT{ff}fold11")._vrt_orig,
                             self.al_to.get_vrt(f"VRT{tf}fold111")._vrt_orig - self.al_to.get_vrt(f"VRT{tf}fold11")._vrt_orig)
        self.al_from.set_dof("JUMP{ff}fold1_z", "z", "rotation", angle)


    def align_rot(self):
        if self.al_from.is_hf_based():
            dofs = {
                "JUMPHFfold111_x": {"angle_x": {"param1": 0.1}},
                "JUMPHFfold111_y": {"angle_y": {"param1": 0.1}},
                "JUMPHFfold111_z": {"angle_z": {"param1": 0.1}},
            }
        elif self.al_from.is_3f_based():
            dofs = {
                "JUMP31fold111_x": {"angle_x": {"param1": 0.1}},
                "JUMP31fold111_y": {"angle_y": {"param1": 0.1}},
                "JUMP31fold111_z": {"angle_z": {"param1": 0.1}},
            }
        elif self.al_from.is_2f_based():
            dofs = {
                "JUMP21fold111_x": {"angle_x": {"param1": 0.1}},
                "JUMP21fold111_y": {"angle_y": {"param1": 0.1}},
                "JUMP21fold111_z": {"angle_z": {"param1": 0.1}},
            }
        rbmover = RigidBodyDofAdaptiveMover("rotations_only")
        for jump_name, jumpdof_params in dofs.items():
            for dof_name, dof_params in jumpdof_params.items():
                rbmover.add_jump(self.pose_from, jump_name,
                                 {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y", "z": "z"}[dof_name],
                                 *self.get_extra_options(dof_params))
        self.rmsd_score = RMSDscore(self.al_from, self.pose_to, map_to_hf=self.use_hf_chain_mapping, same_handedness=self.same_handedness)
        cmc = CubicMonteCarlo(self.rmsd_score, CubicDofs(self.pose_from, dofs))
        cmc.reset(self.pose_from)
        print("init", cmc.lowest_score)

        n = 0
        for it in range(10000 * 1):
            rbmover.apply(self.pose_from)
            cmc.apply(self.pose_from)
            if it % 500 == 0:
                print("done 500")
                print(cmc.lowest_score)
                if self.rmsd_score.get_rmsd(self.pose_from) <= self.overlap_tol:
                    break
                else:
                    n += 0.001
                    for jump_name, jumpdof_params in dofs.items():
                        for dof_name, dof_params in jumpdof_params.items():
                            new_param1 = 0.1 - n
                            rbmover.set_param1(new_param1, jump_name,
                                               {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y",
                                                "z": "z"}[
                                                   dof_name])
                    print("new param1", new_param1)

        cmc.recover_lowest_scored_pose(self.pose_from)
        self.al_from.update_dofs_from_pose(self.pose_from)
        self.final_rmsd = self.rmsd_score.get_rmsd(self.pose_from)
        assert self.final_rmsd <= self.overlap_tol
