#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicDofs class.
@Author: Mads Jeppesen
@Date: 9/26/22
"""
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
from symmetryhandler.reference_kinematics import dof_int_to_str, dof_str_to_int, get_jumpdof_int_int, get_jumpdof_str_str, set_jumpdof_int_int
import numpy as np

class CubicDofs:
    """Simple class that returns the positional info of a cubic symmetrical pose. Has special methods for returning these positions
    in the right order. Also stores additional information about the dofs"""

    def __init__(self, pose, dof_specification):
        """Initialize object

        :param pose: Pose object.
        :param dof_specification (dict): The order in the dofspecificatino is the order used when returning the dofs values.
        as values.
        """
        self.doforder_str = self.__convert_dofspec_to_doforder(dof_specification)
        self.allowed_jumpdofs = self.get_allowed_dofs(pose)
        self.check_doforder_is_correct()
        self.doforder_int = self.__get_dof_order_as_ints(pose)

    def get_allowed_dofs(self, pose):
        """returns the allowed dofs in the pose."""
        allowed_jumpdofs = []
        symdofs = pose.conformation().Symmetry_Info().get_dofs()
        for jump_id, symdof in symdofs.items():
            for dof in range(1, 7):
                if symdof.allow_dof(dof):
                    allowed_jumpdofs.append((jump_num_sym_dof(pose, jump_id), dof_int_to_str[dof]))
        return allowed_jumpdofs

    def transfer_dofs_to_pose(self, pose, *positions):
        """Transfer positions to pose. Asummes that the positions are in the right order."""
        assert len(positions) == len(self.doforder_int)
        for pos, (jump, dof) in zip(positions, self.doforder_int):
            set_jumpdof_int_int(pose, jump, dof, pos)

    def check_doforder_is_correct(self):
        """Checks that all jumps and dofs can be found in the pose and the layot of the dict is correct"""
        for jump, dof in self.doforder_str:
            assert (jump, dof) in self.allowed_jumpdofs, f"{jump} and {dof} is not movable in the pose. Check that the symmetry file is correct"

    def __convert_dofspec_to_doforder(self, dofspec):
        jumpdofs = []
        for jumpname, dofparams in dofspec.items():
            for dofname in dofparams.keys():
                jumpdofs.append((jumpname, dofname))
        return jumpdofs

    def get_positions_as_ndarray(self, pose):
        """Get the current positions in the pose as a ndarray"""
        return np.array(self.get_positions_as_list(pose))

    def get_positions_as_list(self, pose):
        """Get the current positions in the pose as a list"""
        dofs = []
        for jump, dof in self.doforder_int:
            dofs.append(get_jumpdof_int_int(pose, jump, dof))
        return dofs

    def get_jumps_only_as_int(self):
        return [j for j, d in self.doforder_int]

    def get_dofs_only_as_int(self):
        return [d for j, d in self.doforder_int]

    def get_positions_as_dict(self, pose):
        """Get the current positions in the pose as a dictionary"""
        dofs = {}
        for jump, dof in self.doforder_str:
            if not jump in dofs:
                dofs[jump] = {}
            dofs[jump][dof] = get_jumpdof_str_str(pose, jump, dof)
        return dofs

    def __get_dof_order_as_ints(self, pose):
        """Map jump strings to jump ints and dof str to dof ints."""
        doforder_int = []
        for jump, dof in self.doforder_str:
            doforder_int.append((sym_dof_jump_num(pose, jump), dof_str_to_int[dof]))
        return doforder_int


