#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 1/27/23
"""
from cubicsym.cubicsetup import CubicSetup
from copy import deepcopy
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
from symmetryhandler.reference_kinematics import dof_int_to_str, dof_str_to_int, get_jumpdof_int_int, get_jumpdof_str_str, set_jumpdof_int_int
import numpy as np

class DofSpec:
    """A container for a dof specification with several options to manipulate those.

    Details:
    ---------------------
    A dof specification species the degrees of freedom (dofs) of a symmetrical pose (symdofs) in a dictionary format. It designates which
    jumps and dofs (x, y, z, angle_x, angle_y, angle_z) for those jumps are allowed to move as well as advanced options such as:
        - limits of movement in a range (set with limit_movement, max and min)
        - amount of pertubation this dof should have during a docking protocol (set with param1)

    An example is:
        {"JUMPHFfold1": {"z": {"param1": 0.5, "limit_movement": True}},
        "JUMPHFfold1_z": {"angle_z": {"param1": 0.5, "limit_movement": True}},
        "JUMPHFfold111": {"x": {"param1": 0.5, "limit_movement": True}},
        "JUMPHFfold111_x": {"angle_x": {"param1": 0.5}},
        "JUMPHFfold111_y": {"angle_y": {"param1": 0.5}},
        "JUMPHFfold111_z": {"angle_z": {"param1": 0.5}}}

    A dof_specification must be defined in the following order (doforder) for cubic symmetry (where {} corresponds to the jumpidentifier
    which is differnent for differnet depending on the cubic symmetry):
        1. key=JUMP{}fold1, {key=z: {options}}
        2. key=JUMP{}fold1_z, {key=angle_z: {options}}
        3. key=JUMP{}fold111, {key=x: {options}}
        4. key=JUMP{}fold111_x, {key=angle_x: {options}}
        5. key=JUMP{}fold111_y, {key=angle_y: {options}}
        6. key=JUMP{}fold111_z, {key=angle_z: {options}}

    It has methods to quickly retrieve parts of these symdofs as strings (jump and dof names) or ints (jump and dof ids) and transfer dofs
    from one pose to another. It is used for internal use in the CubicSym library and is important for the following classes:
        - CubicBoundary
        - CubicMonteCarlo
    """

    def __init__(self, pose, dof_specification: dict = None):
        """Initialize a DofSpecification object.

        :param pose: Initialize a default dof_specification from a pose. With this option you CANNOT set advanced options
        :param dof_specification: If defined, initialize a dof_specification from this. With this option you CAN set advanced options.
            else a default dof_specification will be created without advanced options.
        """
        if dof_specification is not None:
            self.dof_spec = dof_specification
        else:
            self.dof_spec = self.__get_dofspecification_for_pose(pose)
        self.doforder_str = self.__convert_dofspec_to_doforder()
        self.doforder_int = self.__get_dof_order_as_ints(pose)
        self.__dof_spec_is_in_correct_order()
        self.__check_all_symdofs_are_movable(pose)
        self.dofsize = len(self.doforder_str)
        self.jump_str = [j for j, _ in self.doforder_str]
        self.dof_str = [d for _, d in self.doforder_str]
        self.jump_int = [j for j, _ in self.doforder_int]
        self.dof_int = [d for _, d in self.doforder_int]

    # Setter functions that changes the internal dof_spec
    # ------------------------------------------- #

    def set_symmetrical_bounds(self, bounds):
        """Set the bounds for all dofs in a symmetrical fashion"""
        assert len(bounds) == self.dofsize
        for bound, (jump, dof) in zip(bounds, self.doforder_str):
            self.dof_spec[jump][dof] = {"limit_movement": True, "min": -float(bound), "max": float(bound)}

    # other functions
    # ------------------------------------------- #

    def get_translational_dofs_str(self):
        return [(j, d) for j, d in self.doforder_str if not "angle" in d]

    def get_translational_dofs_int(self):
        return [(ji, di) for (_, ds), (ji, di) in zip(self.doforder_str, self.doforder_int) if not "angle" in ds]

    def transfer_dofs_to_pose(self, pose, *positions):
        """Transfer positions to pose. Asummes that the positions are in the right order."""
        assert len(positions) == len(self.doforder_int)
        for pos, (jump, dof) in zip(positions, self.doforder_int):
            set_jumpdof_int_int(pose, jump, dof, pos)

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

    def get_uniform_dof_specification(self, lb, ub):
        """returns the dof_specification but modify it so all dofs are limited to a uniform range."""
        new_dof_spec = deepcopy(self.dof_spec)
        for jump, dof in self.doforder_str:
            new_dof_spec[jump][dof] = {"limit_movement": True, "min": lb, "max": ub}
        return new_dof_spec

    def __get_jid(self, pose):
        return CubicSetup.get_jumpidentifier_from_pose(pose)

    def __get_allowed_dofs(self, pose):
        """returns the allowed dofs in the pose."""
        allowed_jumpdofs = []
        symdofs = pose.conformation().Symmetry_Info().get_dofs()
        for jump_id, symdof in symdofs.items():
            for dof in range(1, 7):
                if symdof.allow_dof(dof):
                    allowed_jumpdofs.append((jump_num_sym_dof(pose, jump_id), dof_int_to_str[dof]))
        return allowed_jumpdofs

    def __get_dof_order_as_ints(self, pose):
        """Map jump strings to jump ints and dof str to dof ints."""
        doforder_int = []
        for jump, dof in self.doforder_str:
            doforder_int.append((sym_dof_jump_num(pose, jump), dof_str_to_int[dof]))
        return doforder_int

    def __check_all_symdofs_are_movable(self, pose):
        """Checks that all jumps and dofs can be found in the pose and the layot of the dict is correct"""
        allowed_jumpdofs = self.__get_allowed_dofs(pose)
        for jump, dof in self.doforder_str:
            assert (jump, dof) in allowed_jumpdofs, f"{jump} and {dof} is not movable in the pose. Check that the symmetry file is correct"

    def __convert_dofspec_to_doforder(self):
        jumpdofs = []
        for jumpname, dofparams in self.dof_spec.items():
            for dofname in dofparams.keys():
                jumpdofs.append((jumpname, dofname))
        return jumpdofs

    def __dof_spec_is_in_correct_order(self):
        """Asserts the dof_specification is in the right order"""
        try:
            for n, (k, v) in enumerate(self.dof_spec.items()):
                if n == 0:
                    assert k[6:] == "fold1"
                elif n == 1:
                    assert k[6:] == "fold1_z"
                elif n == 2:
                    assert k[6:] == "fold111"
                elif n == 3:
                    assert k[6:] == "fold111_x"
                elif n == 4:
                    assert k[6:] == "fold111_y"
                elif n == 5:
                    assert k[6:] == "fold111_z"
        except AssertionError:
            raise ValueError("Please specify the dof_specification in the right order.")

    def __get_dofspecification_for_pose(self, pose):
        """Returns a dof_specification taking into account the symmetry of the pose"""
        jid = self.__get_jid(pose)
        return {
            f"JUMP{jid}fold1": {"z": {}},
            f"JUMP{jid}fold1_z": {"angle_z": {}},
            f"JUMP{jid}fold111": {"x": {}},
            f"JUMP{jid}fold111_x": {"angle_x": {}},
            f"JUMP{jid}fold111_y": {"angle_y": {}},
            f"JUMP{jid}fold111_z": {"angle_z": {}},
        }
