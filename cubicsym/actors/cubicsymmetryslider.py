#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicSymmetrySlider class
@Author: Mads Jeppesen
@Date: 6/7/22
"""
import time
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
import random
from cloudcontactscore.cloudcontactscore import CloudContactScore
from shapedesign.src.utilities.pose import pose_cas_are_identical
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, get_jumpdof_str_int, get_dofs, dof_str_to_int, set_jumpdof_str_int
from shapedesign.src.utilities.pose import CA_rmsd_without_alignment
from shapedesign.src.visualization.visualizer import get_visualizer_with_score_attached
from shapedesign.src.utilities.score import create_sfxn_from_terms
from cubicsym.actors.symdefswapper import SymDefSwapper
import math
from cubicsym.cubicsetup import CubicSetup
from cubicsym.utilities import get_chain_map, reduce_chain_map_to_indices
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
import numpy as np
from cloudcontactscore.cloudcontactscorecontainer import CloudContactScoreContainer
from cubicsym.utilities import copy_pose_id_to_its_bases, add_id_to_pose_w_base
import warnings

class SymmetrySlider:
    """A symmetric slider using CloudContactScore (CCS)"""
    def __init__(self, pose, acceptable_connections=2, trans_mag=3, max_slide_attempts = 1000,
                 visualizer=None, native=None, freeze_x_at=0, freeze_y_at=0, freeze_z_at=0, css_kwargs=None):
        self.ccs = CloudContactScore(pose, **css_kwargs) if css_kwargs else CloudContactScore(pose, None)
        self.x_params,  self.y_params, self.z_params = None, None, None
        self.store_trans_params(pose) # FIXME: this is not going to work if they are spread over multiple jumps
        self.trans_mag = trans_mag
        self.max_slide_attemps = max_slide_attempts
        self.visualizer = get_visualizer_with_score_attached(visualizer, self.ccs)
        self.acceptable_connections = acceptable_connections
        self.native = native
        self.freeze_x_at = freeze_x_at
        self.freeze_y_at = freeze_y_at
        self.freeze_z_at = freeze_z_at
        self.freeze_x = False
        self.freeze_y = False
        self.freeze_z = False

    def x(self, pose, slide_dir):
        if not self.freeze_x:
            perturb_jumpdof_str_int(pose, *self.x_params, value = slide_dir * self.trans_mag)
            if get_jumpdof_str_int(pose, *self.x_params) <= self.freeze_x_at:
                perturb_jumpdof_str_int(pose, *self.z_params, value = (-slide_dir) * self.trans_mag)
                self.freeze_x = True

    def y(self, pose, slide_dir):
        if not self.freeze_y:
            perturb_jumpdof_str_int(pose, *self.y_params, value=slide_dir * self.trans_mag)
            if get_jumpdof_str_int(pose, *self.y_params) <= self.freeze_y_at:
                perturb_jumpdof_str_int(pose, *self.y_params, value=(-slide_dir) * self.trans_mag)
                self.freeze_y = True

    def z(self, pose, slide_dir):
        if not self.freeze_z:
            perturb_jumpdof_str_int(pose, *self.z_params, value = slide_dir * self.trans_mag)
            if get_jumpdof_str_int(pose, *self.z_params) <= self.freeze_z_at:
                perturb_jumpdof_str_int(pose, *self.z_params, value = (-slide_dir) * self.trans_mag)
                self.freeze_z = True

    def store_trans_params(self, pose):
        trans_dofs = []
        for jumpname, jumpparams in get_dofs(pose).items():
            for dofname in jumpparams.keys():
                if dofname == "x":
                    self.x_params = (jumpname, dof_str_to_int[dofname])
                elif dofname == "y":
                    self.y_params = (jumpname, dof_str_to_int[dofname])
                elif dofname == "z":
                    self.z_params = (jumpname, dof_str_to_int[dofname])
        return trans_dofs

    def compute_clashes_and_connections(self, pose):
        self.ccs.internal_update(pose)
        return self.compute_clashes_only(), self.compute_acceptable_hf_interaction()

    def compute_clashes_only(self):
        return self.ccs.compute_clashes()

    def compute_acceptable_hf_interaction(self):
        self.hf_clashes = self.ccs.compute_hf_clashes()
        return sum([v > 0 for k, v in self.hf_clashes.items()])

    def get_moveset(self):
        moveset = []
        if not self.freeze_x and self.x_params:
            moveset.append(self.x)
        if not self.freeze_y and self.y_params:
            moveset.append(self.y)
        if not self.freeze_z and self.z_params:
            moveset.append(self.z)
        random.shuffle(moveset)
        return moveset

    def apply(self, pose, report=False):
        if report:
            t = time.time()
            start_rmsd = CA_rmsd_without_alignment(pose, self.native)
        n = 0
        finish = FalsE
        while not finish and n != self.max_slide_attemps:
            n += 1
            # get current clashes and connections
            self.ccs.internal_update(pose)
            current_clashes, current_connections = self.compute_clashes_and_connections(pose)
            slide_dir = -1 # into the cubic center
            # pick a random move from the moveset and check that it is allowed. If not, pick another moveset until a move has been allowed
            # or all moves have been used. If not are allowed then a move is forced.
            moveset = self.get_moveset()
            if not moveset: # all moves are frozen!
                break
            clashes = []
            for move in moveset:
                move(pose, slide_dir)
                move_clashes, move_connections = self.compute_clashes_and_connections(pose)
                clashes.append(move_clashes)
                # if we have all the connections we want -> Stop
                if move_connections >= self.acceptable_connections:
                    finish = True
                    break
                # if we lost a connection, go back and try to pick another move
                elif move_connections < current_connections:
                    move(pose, - slide_dir)
                    continue
                # if we introduced a clash, go back and try to pick another move
                if move_clashes > current_clashes:
                    move(pose, - slide_dir)
                    continue
                # move was allowed but the protocol is not finished yet because we do not have all the interactions yet
                break
            # all moves have been used and not all connections have been made and they all introduce clashes, then force the move that
            # produces the lowest amount of clashes
            if len(moveset) == len(clashes):
                move = moveset[clashes.index(min(clashes))]
                move(pose, slide_dir)
                self.ccs.internal_update(pose)
                move_connections = self.compute_acceptable_hf_interaction()
                if move_connections == self.acceptable_connections:
                    finish = True
            if self.visualizer:
                self.visualizer.send_pose(pose)
        print(f"Connected {' and '.join([str(k) for k, v in self.hf_clashes.items() if v > 0]) }")
        if report:
            data = get_dofs(pose)
            data["total_time(s)"] = time.time() - t
            data["start_rmsd"] = start_rmsd
            data["end_rmsd"] = CA_rmsd_without_alignment(pose, self.native)
            data["slide_attempts"] = n
            data["connections"] = '+'.join([str(k) for k, v in self.hf_clashes.items() if v > 0])
            return data

# todo: This should inherit from SymmmetrySlider
class CubicSymmetrySlider:
    """A slider or cubic symmetries"""

    def __init__(self, pose, symmetry_file, ccsc: CloudContactScoreContainer, visualizer=None, native=None, trans_mag=0.3, pymolmover=None, max_slide_attempts=100,
                 cubicboundary=None, set_within_bounds_first=False, debug_mode=False):
        assert is_symmetric(pose)
        self.trans_mag = trans_mag
        self.ccsc = ccsc
        # todo: dont use this:
        #self.fa_rep_score = create_sfxn_from_terms(("fa_rep",), (1.0,))
        self.max_slide_attemps = max_slide_attempts
        self.visualizer = visualizer
        self.pymolmover = pymolmover # for Evodock else can delete
        if self.pymolmover:
            self.visualizer = True
        self.native = native
        self.move_x = False
        self.move_z = False
        self.sds = SymDefSwapper(pose, symmetry_file, debug_mode=False) # debug_mode)
        self.cubicboundary = cubicboundary
        self.set_within_bounds_first = set_within_bounds_first
        self.cubicsetup = CubicSetup(symmetry_file)
        self.debug_mode = debug_mode
        self.hf_hit = None
        self.f3_hit = None
        self.f2_hit = None
        self.hf_slide_moves = None
        self.f3_slide_moves = None
        self.f2_slide_moves = None

    def visualize(self, pose):
        """Visualize the pose."""
        if self.visualizer:
            if self.pymolmover:
                self.pymolmover.apply(pose)
            else:
                self.visualizer.send_pose(pose)

    def n_clashes(self, pose):
        return self.ccsc.ccs.number_of_clashes(pose)

    # TODO: within bounds can be used here but there needs to be an equivalent version for 2-fold, 3-fold and 4-fold
    def slide_trial(self, pose, slide_dir, max_slide_attempts, trans_mag) -> (bool, int):
        """Slides the pose along its fold towards the center. Returns a tuple of of size 2:
            index 0 = False if the folds never hit each other, True otherwise.
            index 1 = The number of slide_moves used."""
        self.ccsc.set_ccs_and_cmc(pose)
        moved = False
        init_n_clashes = self.n_clashes(pose)
        slide_moves = 0
        foldid = CubicSetup.get_jumpidentifier_from_pose(pose)
        self.visualize(pose)
        while self.n_clashes(pose) == init_n_clashes or self.is_z_below_0(pose, foldid) or self.is_x_below_0(pose, foldid):
            moved = True
            self.slide_z(pose, slide_dir, trans_mag, foldid)
            self.visualize(pose)
            slide_moves += 1
            if slide_moves >= max_slide_attempts:
                self.slide_z(pose, slide_dir * -1 * slide_moves, trans_mag, foldid) # go back to the previous location when the function was called
                return False, slide_moves
        if moved:
            self.slide_z(pose, slide_dir * -1, trans_mag, foldid)  # go back to the previous location, just before the pose was moved the last time
            self.visualize(pose)
        return True, slide_moves

    def slide_z(self, pose, slide_dir, trans_mag, foldid):
        """Slide in the z direction."""
        perturb_jumpdof_str_int(pose, f"JUMP{foldid}fold1", 3, value=slide_dir * trans_mag)

    def slide_x(self, pose, slide_dir, trans_mag, foldid):
        """Slide in the x direction."""
        perturb_jumpdof_str_int(pose, f"JUMP{foldid}fold111", 1, value=slide_dir * trans_mag)

    def __debug_apply(self, *poses, atol=0.5):
        """Debugs apply"""
        chain_map = get_chain_map(self.cubicsetup.cubic_symmetry(), self.cubicsetup.righthanded)
        chain_map = reduce_chain_map_to_indices(chain_map, *poses)
        try:
            assert pose_cas_are_identical(*poses, map_chains=chain_map, atol=atol)
        except AssertionError:
            if self.pymolmover:
                for pose in poses:
                    pose.pdb_info().name(f"pose_{CubicSetup.get_base_from_pose(pose)}")
                    self.pymolmover.apply(pose)
            raise AssertionError

        if self.pymolmover:
            for pose in poses:
                pose.pdb_info().name(f"pose_{CubicSetup.get_base_from_pose(pose)}")
                self.pymolmover.apply(pose)

    def is_z_below_0(self, pose, foldid):
        """Checks if z is below zero"""
        return get_jumpdof_str_int(pose, f"JUMP{foldid}fold1", 3) < 0

    def is_x_below_0(self, pose, foldid):
        """Checks if z is below zero"""
        return get_jumpdof_str_int(pose, f"JUMP{foldid}fold111", 1) < 0

    def set_above_z_0(self, pose):
        """If the z translation is below 0 this will set it to +5"""
        fold_id = CubicSetup.get_jumpidentifier_from_pose(pose)
        if get_jumpdof_str_int(pose, f"JUMP{fold_id}fold1", 3) < 0:
            set_jumpdof_str_int(pose, f"JUMP{fold_id}fold1", 3, 5)

    # fixme: also remember to save the id, or have an id flag
    def apply(self, pose_X, score_buffer=2, atol=1):
        """Applies local sliding"""

        # Set inside bounds if it is not
        if self.set_within_bounds_first:
            self.cubicboundary.put_inside_bounds(pose_X, randomize=True)

        # if z is below 0, we need to put it above it
        self.set_above_z_0(pose_X)

        # if for some reason (could potentially happen if the different folds/subunits dont hit eachother)
        # this slider puts the structure into bounce again by assigning the original pose to the input pose (pose_X),
        # therefor we save the original structure here.
        pose_X_org = pose_X.clone()

        # create the other folds based on pose_X and then transfer the id of pose_X to them
        pose_HF, pose_3F, pose_2F = self.sds.create_remaing_folds(pose_X)
        copy_pose_id_to_its_bases(pose_X, pose_HF, pose_3F, pose_2F)

        if self.debug_mode:
            self.__debug_apply(pose_HF, pose_3F, pose_2F)

        # Slide the HF fold
        self.hf_hit, self.hf_slide_moves = self.slide_trial(pose_HF, -1, self.max_slide_attemps, self.trans_mag)
        self.sds.transfer_poseA2B(pose_HF, pose_3F)

        if self.debug_mode:
            self.__debug_apply(pose_HF, pose_3F)
            self.ccsc.ccs.pose_atoms_and_cloud_atoms_overlap(pose_HF)

        # Slide the 3F fold
        self.f3_hit, self.f3_slide_moves = self.slide_trial(pose_3F, -1, self.max_slide_attemps, self.trans_mag)
        self.sds.transfer_poseA2B(pose_3F, pose_2F)

        if self.debug_mode:
            self.__debug_apply(pose_3F, pose_2F)
            self.ccsc.ccs.pose_atoms_and_cloud_atoms_overlap(pose_3F)

        # Slide the 2F fold
        self.f2_hit, self.f2_slide_moves = self.slide_trial(pose_2F, -1, self.max_slide_attemps, self.trans_mag)
        self.sds.transfer_poseA2B(pose_2F, pose_HF)

        if self.debug_mode:
            self.__debug_apply(pose_2F, pose_HF)

        # create pose_X again (unnecessary if the pose is already HF):
        if not self.cubicsetup.is_hf_based():
            self.sds.transfer_poseA2B(pose_HF, pose_X)
        else:
            assert pose_HF is pose_X, "pose_HF and pose_X should reference the same object"

        if not self.cubicboundary.all_dofs_within_bounds(pose_X):
            pose_X.assign(pose_X_org)

    def get_last_hit_status(self):
        return {"HF": {"hit": self.hf_hit, "moves": self.hf_slide_moves},
                "3F": {"hit": self.f3_hit, "moves": self.f3_slide_moves},
                "2F": {"hit": self.f2_hit, "moves": self.f2_slide_moves}}


class InitCubicSymmetrySlider(CubicSymmetrySlider):
    """Initial Cubic symmetrical slider. Slides away until no contacts are felt between the individual folds, and thereafter slides onto
    until clashes are felt between the folds"""

    def __init__(self, pose, symmetry_file, ccsc: CloudContactScoreContainer, visualizer=None, pymolmover=None):
        """Initialize a InitCubicSymmetrySlider object."""
        super().__init__(pose, symmetry_file, ccsc, visualizer, pymolmover=pymolmover)

    def slide_away(self, pose, foldid):
        """Slide away until no clashes are felt between either of the folds"""
        # slide away until the energy is as if no fold are touching each other
        no_fold_clashes = self.get_clashes_when_sliding_folds_away(pose, foldid)
        slide_attempt = 1
        while self.ccsc.ccs.number_of_clashes(pose) > no_fold_clashes and not slide_attempt == self.max_slide_attemps:
            self.slide_z(pose, slide_dir=1, trans_mag=10, foldid=foldid)
            self.visualize(pose)
            slide_attempt += 1
        return slide_attempt == self.max_slide_attemps

    def slide_onto(self, pose, foldid):
        """Slide onto until clashes are felt between either of the folds"""
        current_clashes = self.ccsc.ccs.number_of_clashes(pose)
        slide_attempt = 1
        while self.ccsc.ccs.number_of_clashes(pose) == current_clashes and not slide_attempt == self.max_slide_attemps:
            self.slide_z(pose, slide_dir=-1, trans_mag=self.trans_mag, foldid=foldid)
            self.visualize(pose)
            slide_attempt += 1
        return slide_attempt == self.max_slide_attemps

    def get_clashes_when_sliding_folds_away(self, pose, foldid):
        self.slide_z(pose, slide_dir=1, trans_mag=2000, foldid=foldid)
        clashes = self.ccsc.ccs.number_of_clashes(pose)
        self.slide_z(pose, slide_dir=-1, trans_mag=2000, foldid=foldid)
        return clashes

    def apply(self, pose):
        """Apply initial sliding."""
        pose_before = pose.clone()
        foldid = CubicSetup.get_jumpidentifier_from_pose(pose)
        self.ccsc.set_ccs_and_cmc(pose)
        max_slides_away_hit = self.slide_away(pose, foldid)
        if max_slides_away_hit:
            warnings.warn(f"Max slide away attempts hit {max_slides_away_hit}. You might want to increase the max_slide_attempts" 
                          f" or change the bounds.")
        max_slides_onto_hit = self.slide_onto(pose, foldid)
        if max_slides_onto_hit:
            warnings.warn(f"Max slide onto attempts hit {max_slides_onto_hit}. You might want to increase the max_slide_attempts"
                          f" or change the bounds. The pose will be reverted to the dofs before the slide was applied.")
            pose.assign(pose_before)
        return max_slides_onto_hit
