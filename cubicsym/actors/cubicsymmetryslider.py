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
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_int, get_jumpdof_str_int, get_dofs, dof_str_to_int
from cubicsym.actors.hypotenusemover import HypotenuseMover
from shapedesign.src.utilities.pose import CA_rmsd_without_alignment
from shapedesign.src.visualization.visualizer import get_visualizer_with_score_attached
from shapedesign.src.utilities.score import create_sfxn_from_terms
from cubicsym.actors.symdefswapper import SymDefSwapper
import math

class SymmetrySlider:
    """A symmetric slider using CloudContactScore (CCS)"""
    def __init__(self, pose, acceptable_connections=2, trans_mag=3, max_slide_attempts = 1000,
                 visualizer=None, native=None, freeze_x_at=0, freeze_y_at=0, freeze_z_at=0, css_kwargs=None):
        self.ccs = CloudContactScore(pose, **css_kwargs) if css_kwargs else CloudContactScore(pose)
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

# FIXME: This should inherit from SymmmetrySlider
class CubicSymmetrySlider:
    """A slider or cubic symmetries"""

    def __init__(self, pose, symmetry_file, visualizer=None, native=None, trans_mag=0.3, pymolmover=None, max_slide_attempts=100,
                 cubicboundary=None, set_within_bounds_first=False):
        assert is_symmetric(pose)
        self.trans_mag = trans_mag
        self.store_params(pose)
        self.fa_rep_score = create_sfxn_from_terms(("fa_rep",), (1.0,))
        self.max_slide_attemps = max_slide_attempts
        self.visualizer = visualizer
        ####### for Evodock else can delete
        self.pymolmover = pymolmover
        if self.pymolmover:
            self.visualizer = True
        ####### for Evodock else can delete
        self.native = native
        self.move_x = False
        self.move_z = False
        self.symdefswapper = SymDefSwapper(pose, symmetry_file)
        self.cubicboundary = cubicboundary
        self.set_within_bounds_first = set_within_bounds_first

    def get_chain_map(self):
        return self.symdefswapper.get_chain_map()

    def store_params(self, pose):
        trans_dofs = []
        for jumpname, jumpparams in get_dofs(pose).items():
            for dofname in jumpparams.keys():
                if jumpname == "JUMPHFfold111" and dofname == "x":
                    self.x_params = (jumpname, dof_str_to_int[dofname])
                elif jumpname in ("JUMPHFfold1", "JUMP31fold1", "JUMP21fold1") and dofname == "z":
                    self.zHF_params = (jumpname, dof_str_to_int[dofname])
                    self.z3_params = ("JUMP31fold1", dof_str_to_int[dofname])
                    self.z2_params = ("JUMP21fold1", dof_str_to_int[dofname])
                elif jumpname == "JUMPHFfold1" and dofname == "angle_z":
                    self.z_rot_params = (jumpname, dof_str_to_int[dofname])
        return trans_dofs

    def visualize(self, pose):
        if self.visualizer:
            if self.pymolmover:
                self.pymolmover.apply(pose)
            else:
                self.visualizer.send_pose(pose)

    def local_trial_move(self, pose, f, slide_dir, max_slide_attempts, trans_mag):
        moved = False
        init_score = self.fa_rep_score(pose)
        slide_move = 0
        # TODO: within bounds can be used here but there needs to be an equivalent version for 2-fold, 3-fold and 4-fold
        while self.fa_rep_score(pose) <= init_score: # and self.is_within_bounds(pose):
            f(pose, slide_dir, trans_mag)
            moved = True
            self.visualize(pose)
            slide_move += 1
            if slide_move >= max_slide_attempts:
                f(pose, slide_dir * -1 * slide_move, trans_mag)  # go back to the previous location
                return
        if moved:
            f(pose, slide_dir * -1, trans_mag) # go back to the previous location
            self.visualize(pose)

    def slide_HFfold(self, poseHF, slide_dir, trans_mag):
        perturb_jumpdof_str_int(poseHF, *self.zHF_params, value =slide_dir * trans_mag)

    def slide_3fold(self, pose3, slide_dir, trans_mag):
        perturb_jumpdof_str_int(pose3, *self.z3_params, value = slide_dir * trans_mag)

    def slide_2fold(self, pose2, slide_dir, trans_mag):
        perturb_jumpdof_str_int(pose2, *self.z2_params, value = slide_dir * trans_mag)

    def get_max_dif(self, pose5, pose3, pose2):
        s = (self.fa_rep_score.score(pose5),
             self.fa_rep_score.score(pose3),
             self.fa_rep_score.score(pose2),
             )
        return max(s) - min(s)

    def apply(self, poseHF, score_buffer=2, debug=False, atol=1):
        """Applies local sliding"""
        if self.set_within_bounds_first:
            self.cubicboundary.put_inside_bounds(poseHF, randomize=True)
        pose_org = poseHF.clone()
        # print(f"DEBUG IS {debug}")
        pose3 = self.symdefswapper.create_3fold_pose_from_HFfold(poseHF)
        pose2 = self.symdefswapper.create_2fold_pose_from_HFfold(poseHF)
        if debug:
            try:
                assert pose_cas_are_identical(poseHF, pose3, pose2, map_chains=self.get_chain_map(), atol=atol)
            except AssertionError:
                raise AssertionError
            poseHF.pdb_info().name("poseHF")
            pose3.pdb_info().name("pose3")
            pose2.pdb_info().name("pose2")
        if debug:
            if self.pymolmover:
                # self.pymolmover.keep_history(True)
                self.pymolmover.apply(poseHF)
                self.pymolmover.apply(pose3)
                self.pymolmover.apply(pose2)
        # todo: randomize the order and base choice on the interface energies
        self.local_trial_move(poseHF, self.slide_HFfold, -1, self.max_slide_attemps, self.trans_mag)
        self.symdefswapper.transfer_HFto3(poseHF, pose3)
        if debug:
            try:
                assert pose_cas_are_identical(poseHF, pose3, map_chains=[(i[0], i[1]) for i in self.get_chain_map()], atol=atol)
            except AssertionError:
                raise AssertionError
        self.local_trial_move(pose3, self.slide_3fold, -1, self.max_slide_attemps, self.trans_mag)
        self.symdefswapper.transfer_3to2(pose3, pose2)
        if debug:
            assert pose_cas_are_identical(pose3, pose2, map_chains=[(i[1], i[2]) for i in self.get_chain_map()], atol=atol)
        self.local_trial_move(pose2, self.slide_2fold, -1, self.max_slide_attemps, self.trans_mag)
        self.symdefswapper.transfer_2toHF(pose2, poseHF)
        if debug:
            assert pose_cas_are_identical(pose2, poseHF, map_chains=[(i[2], i[0]) for i in self.get_chain_map()], atol=atol)
        # todo: when implementing is during local_trial_slide this is not needed anymore
        if self.cubicboundary and not self.cubicboundary.all_dofs_within_bounds(poseHF):
            poseHF.assign(pose_org)

class CubicGlobalSymmetrySlider(CubicSymmetrySlider):

    def __init__(self, pose, symmetry_file, visualizer=None, pymolmover=None, normalize_trans=(2000, 1000), slide_x_away=False):
        super().__init__(pose, symmetry_file, visualizer, pymolmover=pymolmover)
        self.normalize_trans = normalize_trans
        self.slide_x_away = slide_x_away

    def slide_away(self, pose):
        fa_rep_null = self.get_null_fa_rep(pose)
        current_score = self.fa_rep_score(pose)
        # slide away until the energy is as if no chain are touching eachother
        while current_score > fa_rep_null or not math.isclose(current_score, fa_rep_null, abs_tol=1e-1):
            self.slide_HFfold(pose, 1, 10)
            if self.slide_x_away:
                self.slide_x(pose, 1, 5)
            self.visualize(pose)
            current_score = self.fa_rep_score(pose)

    def slide_onto(self, pose):
        # first move x
        moved = False
        init_score = self.fa_rep_score(pose)
        while self.fa_rep_score(pose) <= init_score:
            self.slide_x(pose, -1, 0.3)
            self.visualize(pose)
            moved = True
        if moved:
            self.slide_x(pose, 1, 0.3)
            self.visualize(pose)
        # then move HF
        moved = False
        init_score = self.fa_rep_score(pose)
        while self.fa_rep_score(pose) <= init_score:
            self.slide_HFfold(pose, -1, 0.3)
            self.visualize(pose)
            moved = True
        if moved:
            self.slide_HFfold(pose, 1, 0.3)
            self.visualize(pose)

    def get_null_fa_rep(self, pose):
        """Get the energy of the pose when all chains dont touch each other."""
        self.slide_HFfold(pose, 1, trans_mag=self.normalize_trans[0])
        if self.slide_x_away:
            self.slide_x(pose, 1, trans_mag=self.normalize_trans[1])
        self.visualize(pose)
        fa_rep_null = self.fa_rep_score.score(pose)
        self.slide_HFfold(pose, -1, trans_mag=self.normalize_trans[0])
        self.visualize(pose)
        if self.slide_x_away:
            self.slide_x(pose, -1, trans_mag=self.normalize_trans[1])
        self.visualize(pose)
        return fa_rep_null

    def slide_x(self, pose, slide_dir, trans_mag):
        perturb_jumpdof_str_int(pose, *self.x_params, value = slide_dir * trans_mag)

    def apply(self, pose):
        self.slide_away(pose)
        self.slide_onto(pose)