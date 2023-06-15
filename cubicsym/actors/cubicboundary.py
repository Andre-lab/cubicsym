#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
class for CubicBoundary
@Author: Mads Jeppesen
@Date: 9/1/22
"""
from copy import deepcopy
import math
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.constraints import AngleConstraint, AtomPairConstraint, BoundFunc, DihedralConstraint
from pyrosetta.rosetta.protocols.rigid import RigidBodyDofAdaptiveMover
from pyrosetta.rosetta.core.scoring.func import SquareWellFunc
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num
from pyrosetta.rosetta.numeric import dihedral_radians
from pyrosetta.rosetta.core.scoring import ScoreTypeManager
from symmetryhandler.mathfunctions import vector_angle
from symmetryhandler.reference_kinematics import set_jumpdof_str_str, get_jumpdof_str_str
from cubicsym.kinematics import default_HF_dofs
from cubicsym.cubicsetup import CubicSetup
from cubicsym.dofspec import DofSpec
import numpy as np
import random
import warnings
from cubicsym.actors.symdefswapper import SymDefSwapper

class CubicBoundary:
    """Constructs a symmetrical rigid body mover that respects cubic boundaries as well as attaching constraints that respects them
    onto a pose."""

    # fixme: fix c++ code so you dont have to convert
    hack_map = {"angle_x": "x_angle", "angle_y": "y_angle", "angle_z": "z_angle", "x": "x", "y": "y", "z": "z"}

    default_params = {"step_type": "gauss", "param1": 0.5, "param2": 0.0, "min_pertubation": 0.01,
               "limit_movement": False, "max": 0, "min": 0}

    def __init__(self, symdef, pose_at_initial_position, dof_spec: DofSpec, buffer=0.01, sd=1, well_depth=1e9):
        """

        :param buffer: Used when checking if the pose is inside the bounds or when perturbing the pose back into bounds if it is outisde.
            When checking for out of bounds the pose is allowed to be on the boundary +/- buffer. When pertubing the pose is
            put at the closest boundary border +/- buffer.
        """
        self.symdef = symdef
        self.cubicsetup = CubicSetup(symdef)
        self.vrt_map = self.cubicsetup.get_map_vrt_to_pose_resi(pose_at_initial_position)
        self.resi_map = {v: k for k, v in self.vrt_map.items()}
        self.symmetry_multiplier = self.cubicsetup.cubic_energy_multiplier_from_pose(pose_at_initial_position)
        self.set_jump_names()
        self.buffer = buffer
        self.dof_spec = dof_spec
        self.set_boundary(pose_at_initial_position)
        self.current_limits = {k:{kk:{"min": None, "max": None} for kk in v} for k, v in self.dof_spec.dof_spec.items()}
        assert sd != 0
        assert well_depth != 0
        self.sd = sd
        self.well_depth = well_depth

    def set_jump_names(self):
        jid = self.cubicsetup.get_jumpidentifier()
        self.z = f"JUMP{jid}fold1"
        self.x = f"JUMP{jid}fold111"
        self.z_upstream = f"JUMP{jid}fold"
        self.x_upstream = f"JUMP{jid}fold11"
        self.angle_z = f"JUMP{jid}fold1_z"
        self.com_angle_x = f"JUMP{jid}fold111_x"
        self.com_angle_y = f"JUMP{jid}fold111_y"
        self.com_angle_z = f"JUMP{jid}fold111_z"

    @staticmethod
    def calculate_bound_penalty(x, sd_, lb_, ub_, rswitch_=0.5):
        """The weighted (=1.0) score from BoundFunc."""
        if x > ub_:
            delta = x - ub_
        elif lb_ <= x:
            delta = 0
        elif x < lb_:
            delta = lb_ - x
        else:
            delta = 0
        delta /= sd_
        if x > ub_ + rswitch_ * sd_:
            return 2 * rswitch_ * delta - rswitch_ * rswitch_
        else:
            return delta * delta


    @staticmethod
    def calculate_well_penalty(x, x0_, well_depth_):
        """The weighted (=1.0) score from SquareWellFunc."""
        if x < x0_:
            return well_depth_
        return 0.0

    def constrains_are_set_in_score(self, sfxn):
        answer = True
        answer *= sfxn.get_weight(ScoreTypeManager().score_type_from_name("atom_pair_constraint")) > 0
        answer *= sfxn.get_weight(ScoreTypeManager().score_type_from_name("angle_constraint")) > 0
        answer *= sfxn.get_weight(ScoreTypeManager().score_type_from_name("dihedral_constraint")) > 0
        return answer

    def turn_on_constraint_for_score(self, sfxn):
        sfxn.set_weight(ScoreTypeManager().score_type_from_name("atom_pair_constraint"), 1.0)
        sfxn.set_weight(ScoreTypeManager().score_type_from_name("angle_constraint"), 1.0)
        sfxn.set_weight(ScoreTypeManager().score_type_from_name("dihedral_constraint"), 1.0)

    def calculate_z_boundary_angle_penalty(self, pose):
        return self.calculate_well_penalty(self.get_z_boundary_angle(pose), self.convert_to_rad(90), self.well_depth) * self.symmetry_multiplier

    def calculate_x_boundary_angle_penalty(self, pose):
        return self.calculate_well_penalty(self.get_x_boundary_angle(pose), self.convert_to_rad(90), self.well_depth) * self.symmetry_multiplier

    def calculate_z_distance_penalty(self, pose):
        jid = CubicSetup.get_jumpidentifier_from_pose(pose)
        return self.calculate_bound_penalty(abs(self.get_z_distance(pose)), self.sd, *self.get_boundary(self.z, "z"))

    def calculate_x_distance_penalty(self, pose):
        jid = CubicSetup.get_jumpidentifier_from_pose(pose)
        return self.calculate_bound_penalty(abs(self.get_x_distance(pose)), self.sd, *self.get_boundary(self.x, "x"))

    def calculate_angle_z_penalty(self, pose):
        return self.calculate_bound_penalty(self.get_angle_z(pose), self.sd,
                                                      *self.get_boundary(self.angle_z, "angle_z", in_rad=True))

    def calculate_com_angle_x_penalty(self, pose):
        return self.calculate_bound_penalty(self.get_com_angle_x(pose), self.sd,
                                     *self.get_boundary(self.com_angle_x, "angle_x", in_rad=True))

    def calculate_com_angle_y_penalty(self, pose):
        return self.calculate_bound_penalty(self.get_com_angle_y(pose), self.sd,
                                            *self.get_boundary(self.com_angle_y, "angle_y", in_rad=True))

    def calculate_com_angle_z_penalty(self, pose):
        return self.calculate_bound_penalty(self.get_com_angle_z(pose), self.sd,
                                            *self.get_boundary(self.com_angle_z, "angle_z", in_rad=True))

    def get_contribution_from_each_score(self, pose):
        d = {"z_boundary_angle": self.calculate_z_boundary_angle_penalty(pose),
            "x_boundary_angle": self.calculate_x_boundary_angle_penalty(pose),
            "z_distance": self.calculate_z_distance_penalty(pose),
            "x_distance": self.calculate_x_distance_penalty(pose),
            "angle_z": self.calculate_angle_z_penalty(pose),
            "com_angle_x": self.calculate_com_angle_x_penalty(pose),
            "com_angle_y": self.calculate_com_angle_y_penalty(pose),
            "com_angle_z": self.calculate_com_angle_z_penalty(pose)}
        return d

    def get_score(self, pose):
        """Returns the bound penalty score of the pose without calling the score function machinery."""
        total_penalty = 0
        # penalties from going below zero in z or x (squarewell penalty)
        total_penalty += self.calculate_z_boundary_angle_penalty(pose)
        total_penalty += self.calculate_x_boundary_angle_penalty(pose)
        # penalties from going out of bounds (boundfunc penalty)
        total_penalty += self.calculate_z_distance_penalty(pose)
        total_penalty += self.calculate_x_distance_penalty(pose)
        total_penalty += self.calculate_angle_z_penalty(pose)
        total_penalty += self.calculate_com_angle_x_penalty(pose)
        total_penalty += self.calculate_com_angle_y_penalty(pose)
        total_penalty += self.calculate_com_angle_z_penalty(pose)
        return total_penalty

    # -- distance/angle getters --#

    def get_z_distance(self, pose):
        a1, a2 = self.get_z_atomids(pose)
        return pose.residue(a1.rsd()).atom(a1.atomno()).xyz().distance(pose.residue(a2.rsd()).atom(a2.atomno()).xyz())

    def get_x_distance(self, pose):
        a1, a2 = self.get_x_atomids(pose)
        return pose.residue(a1.rsd()).atom(a1.atomno()).xyz().distance(pose.residue(a2.rsd()).atom(a2.atomno()).xyz())

    def get_z_boundary_angle(self, pose, degrees = False):
        p1, p2, p3 = ( np.array(pose.residue(a.rsd()).atom(a.atomno()).xyz()) for a in self.get_z_boundary_atom_ids(pose) )
        return vector_angle(p1 - p2, p3 - p2, degrees)

    def get_x_boundary_angle(self, pose, degrees=False):
        p1, p2, p3 = ( np.array(pose.residue(a.rsd()).atom(a.atomno()).xyz()) for a in self.get_x_boundary_atom_ids(pose) )
        return vector_angle(p1 - p2, p3 - p2, degrees)

    def get_angle_z(self, pose, degrees=False):
        angle = dihedral_radians(*[pose.residue(a.rsd()).atom(a.atomno()).xyz() for a in self.get_angle_z_atomids(pose)])
        if degrees:
            return self.convert_to_degrees(angle)
        return angle

    def get_com_angle_x(self, pose, degrees=False):
        angle = dihedral_radians(*[pose.residue(a.rsd()).atom(a.atomno()).xyz() for a in self.get_com_angle_x_atomids(pose)])
        if degrees:
            return self.convert_to_degrees(angle)
        return angle

    def get_com_angle_y(self, pose, degrees=False):
        angle = dihedral_radians(*[pose.residue(a.rsd()).atom(a.atomno()).xyz() for a in self.get_com_angle_y_atomids(pose)])
        if degrees:
            return self.convert_to_degrees(angle)
        return angle

    def get_com_angle_z(self, pose, degrees=False):
        angle = dihedral_radians(*[pose.residue(a.rsd()).atom(a.atomno()).xyz() for a in self.get_com_angle_z_atomids(pose)])
        if degrees:
            return self.convert_to_degrees(angle)
        return angle

    #-- AtomID getters -- #

    def get_angle_z_atomids(self, pose):
        return AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.angle_z))), \
               AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.angle_z))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.angle_z))), \
               AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.angle_z)))

    def get_com_angle_x_atomids(self, pose):
        return AtomID(3, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_x))), \
               AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_x))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_x))), \
               AtomID(3, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_x)))

    def get_com_angle_y_atomids(self, pose):
        return AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_y))), \
               AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_y))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_y))), \
               AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_y)))

    def get_com_angle_z_atomids(self, pose):
        return AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_z))), \
               AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_z))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_z))), \
               AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.com_angle_z)))

    def get_z_atomids(self, pose):
        return AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.z))), \
        AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.z)))

    def get_x_atomids(self, pose):
        return AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.x))), \
               AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.x)))

    def get_z_boundary_atom_ids(self, pose):
        return AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.z))), \
        AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.z))), \
        AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.z_upstream)))

    def get_x_boundary_atom_ids(self, pose):
        return AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, self.x))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.x))), \
               AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, self.x_upstream)))

    #-- Constrain setters --#

    def add_z_constrain_below_0(self, pose, func):
        pose.add_constraint(AngleConstraint(*self.get_z_boundary_atom_ids(pose), func))

    def add_x_constrain_below_0(self, pose, func):
        pose.add_constraint(AngleConstraint(*self.get_x_boundary_atom_ids(pose), func))

    def add_z_constraints(self, pose, func):
        pose.add_constraint(AtomPairConstraint(*self.get_z_atomids(pose), func))

    def add_x_constraints(self, pose, func):
        pose.add_constraint(AtomPairConstraint(*self.get_x_atomids(pose), func))

    def add_angle_z_constraints(self, pose, func):
        pose.add_constraint(DihedralConstraint(*self.get_angle_z_atomids(pose), func))

    def add_com_angle_x_constraints(self, pose, func):
        pose.add_constraint(DihedralConstraint(*self.get_com_angle_x_atomids(pose), func))

    def add_com_angle_y_constraints(self, pose, func):
        pose.add_constraint(DihedralConstraint(*self.get_com_angle_y_atomids(pose), func))

    def add_com_angle_z_constraints(self, pose, func):
        pose.add_constraint(DihedralConstraint(*self.get_com_angle_z_atomids(pose), func))

    def create_bounded_func(self, lb, ub, type="idk_if_this_is_important"):
        """Creates a BoundedFunc object to be used in a constraint class object.

        https://www.rosettacommons.org/docs/latest/rosetta_basics/file_types/constraint-file
        rswitch=0.5 because: "Note: Setting rswitch to anything other than 0.5 will (from link)". This is set when calling the
        constructor as we do in this function.

        If x above:((x - ub)**2) / sd
        If x below:((x - lb)**2) / sd
        Then if x is large (for idk what reason):

        if ( x > ub_ + rswitch_*sd_ ) {
            return 2 * rswitch_ * delta - rswitch_ * rswitch_;
        }
        """
        return BoundFunc(lb, ub, self.sd, type)

    def create_well_func(self, x0):
        """Creates a SquareWellFunc object to be used in a contraint class object

        if x < x0_:
            return well_depth_
        else:
            return 0.0
        """
        return SquareWellFunc(x0, self.well_depth)

    def set_constraints(self, pose):
        """Sets cubic symmetrical constraints for the pose. sd is the standard deviation and a smaller number means a larger penalty."""
        self.current_positions = self.dof_spec.get_positions_as_dict(pose)
        jid = CubicSetup.get_jumpidentifier_from_pose(pose)
        self.add_z_constraints(pose, self.create_bounded_func(*self.get_boundary(self.z, "z")))
        self.add_x_constraints(pose, self.create_bounded_func(*self.get_boundary(self.x, "x")))
        self.add_angle_z_constraints(pose, self.create_bounded_func(*self.get_boundary(self.angle_z, "angle_z", in_rad=True)))
        self.add_com_angle_x_constraints(pose, self.create_bounded_func(*self.get_boundary(self.com_angle_x, "angle_x", in_rad=True)))
        self.add_com_angle_y_constraints(pose, self.create_bounded_func(*self.get_boundary(self.com_angle_y, "angle_y", in_rad=True)))
        self.add_com_angle_z_constraints(pose, self.create_bounded_func(*self.get_boundary(self.com_angle_z, "angle_z", in_rad=True)))

        # -------------------
        # There used to be constrains on the translational dofs of z and x but this unfortunately resulted in an occasional
        # RuntimeError: "0-length bonds in AngleConstraint" when the z or x dofs were exactly zero, which could happen during minimization.
        # Therefore these are uncommented and hence removed. I left the old text below intact in case someone would read it.
        # -------------------
        # the system can go out of bounce in 3 ways:
        # 1. z is below 0
        # 2. x is below 0
        # 3. angle_z cross the cn -/+ barrier
        # Option 3 is already protected by the constraint in angle_z alone. However, the distance for x and z can drop below 0 and
        # then go into a range with 0 penalty inside -min and -max. This is because the distance will always be positive in the
        # constraint calculation. Therefore, we add yet another constrain to the z/x distance that penalizes a drop below 0.
        # this is done by angle constraints:
        # if it is = 180 it is good, if 0 it is bad, we set the threshold at 90. We don't want any penalty as long as the
        # z/x distance is above zero and that is accomplished by setting the angle to -1 which is an impossible value to get as the
        # angles are always taken as positive.
        # self.add_z_constrain_below_0(pose, self.create_well_func(self.convert_to_rad(90)))
        # self.add_x_constrain_below_0(pose, self.create_well_func(self.convert_to_rad(90)))

    def convert_to_degrees(self, rad):
        return rad * 180 / math.pi

    def convert_to_rad(self, degrees):
        return degrees * math.pi / 180

    def get_boundary(self, jumpname: str, dofname:str, in_rad=False, add_buffer=False):
        lower, upper = self.boundaries[jumpname][dofname]["min"], self.boundaries[jumpname][dofname]["max"]
        if add_buffer:
            lower, upper = lower + self.buffer, upper - self.buffer
        if in_rad:
            return self.convert_to_rad(lower), self.convert_to_rad(upper)
        else:
            return lower, upper

    def set_boundary(self, pose_at_initial_position):
        self.boundaries = {}
        all_current_pos = self.dof_spec.get_positions_as_dict(pose_at_initial_position)
        for jump_name, jumpdof_params in self.dof_spec.dof_spec.items():
            self.boundaries[jump_name] = {}
            for dof_name, dof_params in jumpdof_params.items():
                self.boundaries[jump_name][dof_name] = {"min": None, "max": None}
                if self.dof_spec.dof_spec[jump_name][dof_name].get("limit_movement"):
                    init_min = self.dof_spec.dof_spec[jump_name][dof_name].get("min")
                    init_max = self.dof_spec.dof_spec[jump_name][dof_name].get("max")
                    assert init_min is not None and init_max is not None, f"limit_movement is set to True for jump: {jump_name} and dof: {dof_name} but min and max are not set"
                    current_pos = all_current_pos[jump_name][dof_name]
                    min_bound = current_pos + init_min
                    max_bound = current_pos + init_max
                    if self.z == jump_name:
                        if dof_name == "z":
                            assert current_pos >= 0 - self.buffer, f"Jump: {self.z} and dof: z is outside its cubic bounds!"
                            # if the minimum goes below zero we have to modify the current minimum
                            if current_pos + init_min < 0:
                                min_bound = 0
                        elif dof_name == "angle_z":
                            a_min, a_max = self.cubicsetup.angle_z_distance(pose_at_initial_position)
                            if init_min < a_min:
                                min_bound = current_pos + a_min
                            if init_max > a_max:
                                max_bound = current_pos + a_max
                    elif self.x == jump_name:
                        if dof_name == "x":
                            assert current_pos >= 0 - self.buffer, f"Jump: {self.x} and dof: x has exceeded its cubic bounds!"
                            # if the minimum goes below zero we have to modify the current minimum
                            if current_pos + init_min < 0:
                                min_bound = 0
                    self.boundaries[jump_name][dof_name]["min"] = min_bound
                    self.boundaries[jump_name][dof_name]["max"] = max_bound

    def put_inside_bounds(self, pose, randomize=True, raise_warning=False):
        """Put the pose inside the specified bounds if it is outside its bounds.
        If randomize == True it will put the pose at random position inside the bounds,
        else it will put it at the nearest boundary +/- self.buffer."""
        for jump_name, jumpdof_params in self.dof_spec.dof_spec.items():
            for dof_name, _ in jumpdof_params.items():
                if self.dof_spec.dof_spec[jump_name][dof_name].get("limit_movement", False):
                    if not self.dof_within_bounds(pose, jump_name, dof_name, raise_warning=raise_warning):
                        if randomize:
                            val_to_set = random.uniform(*self.get_boundary(jump_name, dof_name, add_buffer=True))
                            set_jumpdof_str_str(pose, jump_name, dof_name, val_to_set)
                        else:
                            current_val = get_jumpdof_str_str(pose, jump_name, dof_name)
                            min_bound, max_bound = self.get_boundary(jump_name, dof_name)
                            if current_val <= min_bound:
                                set_jumpdof_str_str(pose, jump_name, dof_name, min_bound + self.buffer)
                            if current_val >= max_bound:
                                set_jumpdof_str_str(pose, jump_name, dof_name, max_bound - self.buffer)

    def construct_rigidbody_mover(self, pose, rb_name="") -> RigidBodyDofAdaptiveMover:
        """Construct rigidbodymover."""
        rb_mover = RigidBodyDofAdaptiveMover(rb_name)
        self.current_positions = self.dof_spec.get_positions_as_dict(pose)
        for jump_name, jumpdof_params in self.dof_spec.dof_spec.items():
            for dof_name, dof_params in jumpdof_params.items():
                # print(*self.get_extra_options(pose, jump_name, dof_name, dof_params))
                extra_options = self.get_extra_options(pose, jump_name, dof_name, dof_params)
                # print(jump_name, dof_name, extra_options)
                rb_mover.add_jump(pose, jump_name, self.hack_map[dof_name], *extra_options)
        return rb_mover

    def get_extra_options(self, pose, jump_name, dof_name, dof_params):
        """HACK: You cannot parse individual keyword parameters to these c++ objects, so you
        have specify them all and then change them before parsing them as below"""
        if dof_params:  # check for not empty
            default_params = deepcopy(self.default_params)
            default_params.update(dof_params)
            self.set_limits(pose, jump_name, dof_name, default_params)
            return default_params.values()
        else:
            return []

    def dof_within_bounds(self, pose, jump: str, dof: str, raise_assertion=False, raise_warning=False):
        """Checks if a single dof of the pose are within its bounds."""
        current_pos = get_jumpdof_str_str(pose, jump, dof)
        min_bound = self.boundaries[jump][dof]["min"]
        max_bound = self.boundaries[jump][dof]["max"]
        if min_bound - self.buffer > current_pos or max_bound  + self.buffer < current_pos:
            comment = f"Pose is not within bounds because {jump}:{dof} is {current_pos} but should lie within {min_bound} and {max_bound}"
            if raise_assertion:
                raise AssertionError(comment)
            elif raise_warning:
                warnings.warn(comment)
            else:
                return False
        return True

    def all_dofs_within_bounds(self, pose, raise_assertion=False, raise_warning=False):
        """Checks if all the dofs of the pose are within bounds."""
        for jump_name, jumpdof_params in self.dof_spec.dof_spec.items():
            for dof_name, dof_params in jumpdof_params.items():
                if self.dof_spec.dof_spec[jump_name][dof_name].get("limit_movement", False):
                    if not self.dof_within_bounds(pose, jump_name, dof_name):
                        if raise_assertion:
                            raise AssertionError(f"JUMP {jump_name} with DOF {dof_name} is not within bounds")
                        elif raise_warning:
                            warnings.warn(f"JUMP {jump_name} with DOF {dof_name} is not within bounds")
                        return False
        return True

    def get_min_max(self, jump_name, dof_name):
        """Get the minimum and maximum pertubation for the jump/dof according"""
        current_pos = self.current_positions[jump_name][dof_name]
        min_bound = self.boundaries[jump_name][dof_name]["min"]
        max_bound = self.boundaries[jump_name][dof_name]["max"]
        min_peturbation = min_bound - current_pos
        max_peturbation = max_bound - current_pos
        return min_peturbation, max_peturbation

    def set_limits(self, pose, jump_name, dof_name, dof_params):
        if self.dof_spec.dof_spec[jump_name][dof_name].get("limit_movement", False):
            cmin, cmax = self.get_min_max(jump_name, dof_name)
            dof_params["min"] = cmin
            dof_params["max"] = cmax
            self.current_limits[jump_name][dof_name]["min"] = cmin
            self.current_limits[jump_name][dof_name]["max"] = cmax

# def add_com_angle_y_constraints(self, pose, func):
#     pose.add_constraint(
#         DihedralConstraint(AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_y"))),
#             AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_y"))),
#             AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_y"))),
#             AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_y"))),
#             func))
#
# def add_com_angle_z_constraints(self, pose, func):
#     pose.add_constraint(
#         DihedralConstraint(AtomID(2, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_z"))),
#             AtomID(1, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_z"))),
#             AtomID(1, pose.fold_tree().upstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_z"))),
#             AtomID(2, pose.fold_tree().downstream_jump_residue(sym_dof_jump_num(pose, "JUMPHFfold111_z"))),
#             func))
