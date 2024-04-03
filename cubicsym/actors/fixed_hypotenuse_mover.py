from symmetryhandler.reference_kinematics import perturb_jumpdof_int_int, dof_str_to_int, get_jumpdof_int_int, perturb_jumpdof_str_str, get_jumpdof_str_str, set_jumpdof_str_str
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
import numpy as np
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from symmetryhandler.mathfunctions import vector_angle, vector_projection, vector_projection_on_subspace
import random
import math


class FixedHypotenuseMover:

    def __init__(self, pose_at_initial_position):
        self.zx_combined_vector, self.slope = self.get_fixed_zx_combined(pose_at_initial_position)

    def get_fixed_zx_combined(self, pose_at_initial_position):
        z_start = pose_at_initial_position.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose_at_initial_position, f"JUMPHFfold1"))
        z_start = np.array(pose_at_initial_position.residue(z_start).atom(1).xyz())
        z_end = pose_at_initial_position.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose_at_initial_position, f"JUMPHFfold1"))
        z_end = np.array(pose_at_initial_position.residue(z_end).atom(1).xyz())
        z = np.array(z_start) - np.array(z_end)

        x_start = pose_at_initial_position.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose_at_initial_position, f"JUMPHFfold111"))
        x_start = np.array(pose_at_initial_position.residue(x_start).atom(1).xyz())
        x_end = pose_at_initial_position.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose_at_initial_position, f"JUMPHFfold111"))
        x_end = np.array(pose_at_initial_position.residue(x_end).atom(1).xyz())
        x = np.array(x_start) - np.array(x_end)

        zx_combined = z + x
        zx_combined_norm = zx_combined / np.linalg.norm(zx_combined)
        slope = get_jumpdof_str_str(pose_at_initial_position, "JUMPHFfold1", "z") / get_jumpdof_str_str(pose_at_initial_position, "JUMPHFfold111", "x")
        # assert math.isclose(slope, self.__get_z_projection_magnitude(zx_combined_norm) / self.__get_x_projection_magnitude(zx_combined_norm))
        return zx_combined_norm, slope

    def __get_z_projection_magnitude(self, zx_combined_vec):
        z0 = zx_combined_vec[2]
        # assert np.all(np.isclose(vector_projection_on_subspace(zx_combined_vec, [0, 0, 1]), [0, 0, z0], atol=0.001))
        return z0

    def __get_x_projection_magnitude(self, zx_combined_vec):
        x0_vec = vector_projection_on_subspace(zx_combined_vec, [1, 0, 0], [0, 1, 0])
        x0 = np.linalg.norm(x0_vec)
        if zx_combined_vec[2] < 0:
            x0 *= -1
        return x0

    def set_x_combined(self, pose, value):
        z0 = get_jumpdof_str_str(pose,  "JUMPHFfold1", "z")
        x0 = z0 / self.slope
        x_to_set = x0 + value
        set_jumpdof_str_str(pose, "JUMPHFfold111", "x", x_to_set)
        # assert math.isclose(self.get_x_combined_value(pose), value)

    def set_zx_combined(self, pose, value):
        zx_combination_vec_new = self.zx_combined_vector * value
        # assert math.isclose(np.linalg.norm(zx_combination_vec_new), abs(value))
        z0_new = self.__get_z_projection_magnitude(zx_combination_vec_new)
        x0_new = self.__get_x_projection_magnitude(zx_combination_vec_new)
        # z0 and x0 is where the hypotenuse should be
        current_combined_x = self.get_x_combined_value(pose)
        set_jumpdof_str_str(pose, "JUMPHFfold1", "z", z0_new)
        set_jumpdof_str_str(pose, "JUMPHFfold111", "x", x0_new + current_combined_x)
        # assert math.isclose(self.get_zx_combined_value(pose), value)

    def perturb_zx_combined(self, pose, value):
        zx_combination_vec_perturbed = self.zx_combined_vector * value
        # assert math.isclose(np.linalg.norm(zx_combination_vec_perturbed), abs(value))
        z0 = self.__get_z_projection_magnitude(zx_combination_vec_perturbed)
        x0 = self.__get_x_projection_magnitude(zx_combination_vec_perturbed)
        # assert math.isclose(abs(value), np.linalg.norm([x0, 0, z0]))
        perturb_jumpdof_str_str(pose, "JUMPHFfold1", "z", z0)
        perturb_jumpdof_str_str(pose, "JUMPHFfold111", "x", x0)

    def perturb_x_combined(self, pose, value):
        perturb_jumpdof_str_str(pose, "JUMPHFfold111", "x", value)

    def get_zx_combined_value(self, pose):
        z0 = get_jumpdof_str_str(pose,  "JUMPHFfold1", "z")
        x0 = z0 / self.slope
        zx_combined = [x0, 0, z0]
        norm = np.linalg.norm(zx_combined)
        if z0 < 0:
            norm *= - 1
        return norm

    def get_x_combined_value(self, pose):
        # given the z value what should the x value be
        z0 = get_jumpdof_str_str(pose,  "JUMPHFfold1", "z")
        x0 = z0 / self.slope
        x_in_pose = get_jumpdof_str_str(pose,  "JUMPHFfold111", "x")
        x = x_in_pose - x0
        return x

    # def get_zx_combined_vector(self, pose):
    #     # from z you can calculate where x is
    #     # from z you can calculate where x is
    #     z0 = get_jumpdof_str_str(pose,  "JUMPHFfold1", "z")
    #     x0 = z0 / self.slope
    #     zx_combination = np.array([x0, 0, z0])
    #     return zx_combination
