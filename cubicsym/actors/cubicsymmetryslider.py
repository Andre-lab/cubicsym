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
    """A symmetric slider using CloudContactScore (CCS) with a focus on Cubic Symmetries."""

    def __init__(self, pose, symmetry_file, global_slide=False, use_freeze=True, visualizer=None, native=None, freeze_x_at=0, freeze_z_at=0,
                 freeze_zx_at=0, trans_mag=0.25, pymolmover=None, max_slide_attempts=100, cubicboundary=None, set_within_bounds_first=False):
        assert is_symmetric(pose)
        self.trans_mag = trans_mag
        self.store_params(pose)
        if global_slide:
            self.ccs = CloudContactScore(pose)
        else:
            self.fa_rep_score = create_sfxn_from_terms(("fa_rep",), (1.0,))
        self.global_slide = global_slide
        self.max_slide_attemps = max_slide_attempts
        self.visualizer = visualizer
        ####### for Evodock else can delete
        self.pymolmover = pymolmover
        if self.pymolmover:
            self.visualizer = True
        ####### for Evodock else can delete
        self.acceptable_connections = 2
        self.native = native
        self.freeze_x_at = freeze_x_at
        self.freeze_z_at = freeze_z_at
        self.freeze_zx_at = freeze_zx_at
        self.freeze_x = False
        self.freeze_z = False
        self.freeze_zx = False
        self.use_freeze = use_freeze
        self.symdefswapper = SymDefSwapper(pose, symmetry_file)
        self.cubicboundary = cubicboundary
        self.set_within_bounds_first = set_within_bounds_first

    def get_chain_map(self):
        return self.symdefswapper.get_chain_map()

    def x(self, pose, slide_dir):
        if not self.freeze_x:
            perturb_jumpdof_str_int(pose, *self.x_params, value = slide_dir * self.trans_mag)
            if self.use_freeze and get_jumpdof_str_int(pose, *self.x_params) <= self.freeze_x_at:
                perturb_jumpdof_str_int(pose, *self.z5_params, value =(-slide_dir) * self.trans_mag)
                self.freeze_x = True

    def z(self, pose, slide_dir):
        if not self.freeze_z:
            perturb_jumpdof_str_int(pose, *self.z5_params, value =slide_dir * self.trans_mag)
            if self.use_freeze and get_jumpdof_str_int(pose, *self.z5_params) <= self.freeze_z_at:
                perturb_jumpdof_str_int(pose, *self.z5_params, value =(-slide_dir) * self.trans_mag)
                self.freeze_z = True

    def zx(self, pose, slide_dir):
        if not self.freeze_zx:
            self.hypotenusemover.add_c(pose, slide_dir * self.trans_mag)
            if self.use_freeze and self.hypotenusemover.get_c_size() <= self.freeze_zx_at:
                self.hypotenusemover.add_c(pose, (-slide_dir) * self.trans_mag)
                self.freeze_zx = True

    def store_params(self, pose):
        trans_dofs = []
        for jumpname, jumpparams in get_dofs(pose).items():
            for dofname in jumpparams.keys():
                if jumpname == "JUMPHFfold111" and dofname == "x":
                    self.x_params = (jumpname, dof_str_to_int[dofname])
                elif jumpname in ("JUMPHFfold1", "JUMP31fold1", "JUMP21fold1") and dofname == "z":
                    self.z5_params = (jumpname, dof_str_to_int[dofname])
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

    def compute_clashes_and_connections(self, pose):
        self.ccs.internal_update(pose)
        return self.ccs.compute_clashes(), self.compute_acceptable_hf_interaction()

    def compute_acceptable_hf_interaction(self):
        self.hf_clashes = self.ccs.compute_hf_clashes()
        return sum([v > 0 for k, v in self.hf_clashes.items()])

    def init(self, pose):
        self.hypotenusemover = HypotenuseMover(pose)

    def get_moveset(self):
        moveset = []
        if not self.freeze_x:
            moveset.append(self.x)
        if not self.freeze_z:
            moveset.append(self.z)
        if not self.freeze_zx:
            moveset.append(self.zx)
        random.shuffle(moveset)
        return moveset

    def apply(self, pose, report=False, debug=False):
        if self.global_slide:
            self.apply_global_slide(pose, report)
        else:
            self.apply_local_slide(pose, debug=debug)

    def local_trial_move(self, pose, f, slide_dir):
        moved = False
        init_score = self.fa_rep_score(pose)
        slide_move = 0
        # TODO: within bounds can be used here but there needs to be an equivalent version for 2-fold, 3-fold and 4-fold
        while self.fa_rep_score(pose) <= init_score: # and self.is_within_bounds(pose):
            f(pose, slide_dir)
            moved = True
            self.visualize(pose)
            slide_move += 1
            if slide_move >= self.max_slide_attemps:
                f(pose, slide_dir * -1 * slide_move)  # go back to the previous location
                return
        if moved:
            f(pose, slide_dir * -1) # go back to the previous location
            self.visualize(pose)

    def slide_5fold(self, pose5, slide_dir):
        perturb_jumpdof_str_int(pose5, *self.z5_params, value = slide_dir * self.trans_mag)

    def slide_3fold(self, pose3, slide_dir):
        perturb_jumpdof_str_int(pose3, *self.z3_params, value = slide_dir * self.trans_mag)

    def slide_2fold(self, pose2, slide_dir):
        perturb_jumpdof_str_int(pose2, *self.z2_params, value = slide_dir * self.trans_mag)

    def get_max_dif(self, pose5, pose3, pose2):
        s = (self.fa_rep_score.score(pose5),
             self.fa_rep_score.score(pose3),
             self.fa_rep_score.score(pose2),
             )
        return max(s) - min(s)

    def apply_local_slide(self, poseHF, score_buffer=2, debug=False, atol=1):
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
        self.local_trial_move(poseHF, self.slide_5fold, -1)
        self.symdefswapper.transfer_HFto3(poseHF, pose3)
        if debug:
            try:
                assert pose_cas_are_identical(poseHF, pose3, map_chains=[(i[0], i[1]) for i in self.get_chain_map()], atol=atol)
            except AssertionError:
                raise AssertionError
        self.local_trial_move(pose3, self.slide_3fold, -1)
        self.symdefswapper.transfer_3to2(pose3, pose2)
        if debug:
            assert pose_cas_are_identical(pose3, pose2, map_chains=[(i[1], i[2]) for i in self.get_chain_map()], atol=atol)
        self.local_trial_move(pose2, self.slide_2fold, -1)
        self.symdefswapper.transfer_2toHF(pose2, poseHF)
        if debug:
            assert pose_cas_are_identical(pose2, poseHF, map_chains=[(i[2], i[0]) for i in self.get_chain_map()], atol=atol)
        # todo: when implementing is during local_trial_slide this is not needed anymore
        if self.cubicboundary and not self.cubicboundary.all_dofs_within_bounds(poseHF):
            poseHF.assign(pose_org)

    # TODO one idea could be to only check the clashes between any other connections than the current fold you move.
    # FIXME: Does not include twofold and threefold moves
    def apply_global_slide(self, pose, report=False):
        if report:
            t = time.time()
            start_rmsd = CA_rmsd_without_alignment(pose, self.native)
        self.init(pose)
        n = 0
        finish = False
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

    # def apply(self, pose):
    #     # see if we can change any of the dofs without changes the clashes
    #     current_clashes = self.compute_clashes_only(pose)
    #     self.compute_acceptable_hf_interaction(pose)
    #     for slide_attempt in range(self.max_slide_attemps):
    #         changed_clash_state = []
    #         for jump, dof in self.trans_dofs:
    #             changed_clash_state.append(self.trial_trans_with_clashes(pose, current_clashes, jump, dof))
    #         if all(changed_clash_state):
    #             break
    #         if self.visualizer:
    #             self.visualizer.send_pose(pose)
     #
    #

    #   def extract_symmetric_info(self, pose, symmetry_file):
    #         self.symmetrysetup = SymmetrySetup()
    #         self.symmetrysetup.read_from_file(symmetry_file)
    #         self.symmetrysetup.update_dofs_from_pose(pose)
    #         self.symmetrysetup.apply_dofs()
    #         self.anchor_resi = self.symmetrysetup.get_anchor_residue(pose)
    #         self.setup_threefold_zx_projection(pose)
    #         self.setup_twofold_zx_projection(pose)
    #
    #     def setup_threefold_zx_projection(self, pose):
    #         # get VRT2fold1211 and VRT3fold1111 and VRTHFfold1111
    #         a = self.symmetrysetup.get_vrt("VRTHFfold1111").vrt_orig
    #         b = self.symmetrysetup.get_vrt("VRT2fold1211").vrt_orig
    #         c = self.symmetrysetup.get_vrt("VRT3fold1111").vrt_orig
    #         # pseudoatom threefold, pos=[40.75134243,  4.85462394, 53.72139532]
    #         threefold_center = (a + b + c) / 3
    #         self.threefold_center = threefold_center / np.linalg.norm(threefold_center)
    #
    #     def setup_twofold_zx_projection(self, pose):
    #         # get VRT2fold1111 and VRT3fold1111
    #         a = self.symmetrysetup.get_vrt("VRTHFfold1111").vrt_orig
    #         b = self.symmetrysetup.get_vrt("VRT2fold1111").vrt_orig
    #         self.twofold_axis = (a + b) / 2
    #         self.twofold_axis_norm = self.twofold_axis / np.linalg.norm(self.twofold_axis)
    #         self.twofold_x_vector = self.twofold_axis_norm[:2] + [0]
    #
    #     def x(self, pose, slide_dir):
    #         if not self.freeze_x:
    #             perturb_jumpdof(pose, *self.x_params, value = slide_dir * self.trans_mag)
    #             if self.use_freeze and get_jumpdof(pose, *self.x_params) <= self.freeze_x_at:
    #                 perturb_jumpdof(pose, *self.z_params, value = (-slide_dir) * self.trans_mag)
    #                 self.freeze_x = True
    #
    #     def z(self, pose, slide_dir):
    #         if not self.freeze_z:
    #             perturb_jumpdof(pose, *self.z_params, value = slide_dir * self.trans_mag)
    #             if self.use_freeze and get_jumpdof(pose, *self.z_params) <= self.freeze_z_at:
    #                 perturb_jumpdof(pose, *self.z_params, value = (-slide_dir) * self.trans_mag)
    #                 self.freeze_z = True
    #
    #     def zx(self, pose, slide_dir):
    #         if not self.freeze_zx:
    #             self.hypotenusemover.add_c(pose, slide_dir * self.trans_mag)
    #             if self.use_freeze and self.hypotenusemover.get_c_size() <= self.freeze_zx_at:
    #                 self.hypotenusemover.add_c(pose, (-slide_dir) * self.trans_mag)
    #                 self.freeze_zx = True
    #
    #     def get_norm_z_and_x_vector(self, pose):
    #         x = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())
    #         x_norm = x / np.linalg.norm(x)
    #         z_norm = np.array([0, 0, 1])
    #         return x_norm, z_norm
    #
    #     def twofold(self, pose, slide_dir):
    #         self.visualizer.send_pose(pose, "start")
    #         # rotate to twofold
    #         x_pos = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())[:2] + [0]
    #         angle = vector_angle(x_pos, self.twofold_x_vector)
    #         perturb_jumpdof(pose, *self.z_rot_params, value=(-1 * angle))
    #         self.visualizer.send_pose(pose, "rotate_to_2fold")
    #         # extend x to the center point
    #         angle = vector_angle(self.twofold_axis, [0, 0, 1])
    #         current_z_len = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())[2]
    #         cross_point = math.tan(math.radians(angle)) * current_z_len
    #         x_to_add = (cross_point - np.linalg.norm(x_pos)) * slide_dir
    #         perturb_jumpdof(pose, *self.x_params, value=x_to_add)
    #         self.visualizer.send_pose(pose, "extended_to_center_point")
    #         # do zx (hypotenuse move)
    #         self.zx(pose, slide_dir)
    #         self.visualizer.send_pose(pose, "zx_move")
    #         # extend x to the center point back again
    #         perturb_jumpdof(pose, *self.x_params, value=-x_to_add)
    #         self.visualizer.send_pose(pose, "extended_back")
    #         # # rotate back
    #         perturb_jumpdof(pose, *self.z_rot_params, value=(1 * angle))
    #         self.visualizer.send_pose(pose, "rotate_back")
    #         # ...
    #
    #
    #
    #         # test:
    #         # x_anc, _, z_anc = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())
    #         # x, y, z = self.trans_mag * self.twofold_axis * slide_dir
    #         # set circle width
    #         # circle_width_diff = (np.linalg.norm([x, y])) * slide_dir
    #         # perturb_jumpdof(pose, *self.x_params, value=circle_width_diff)
    #         # # set circle height
    #         # circle_height_diff = z * slide_dir
    #         # perturb_jumpdof(pose, *self.z_params, value=circle_height_diff)
    #         # # set circle rotation
    #         # # test: now this should be equal
    #         # x_anc_new, _, z_anc_new = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())
    #         # assert x_anc + np.linalg.norm([x,y]) == x_anc_new
    #         # assert z_anc + z == z_anc_new
    #
    #         # # should only ve
    #         # current_x = np.array(pose.residue(self.anchor_resi).atom("CA").xyz())
    #         # # current_x_norm = np.linalg.norm(current_x)
    #         # current_x_norm = np.array(pose.jump(sym_dof_jump_num(pose, "JUMPHFfold111")).get_translation())[0]
    #         # # current_z_norm = np.array(pose.jump(sym_dof_jump_num(pose, "JUMP5fold1")).get_translation())
    #         # new_x = [x, y] * slide_dir
    #         # new_x_projected = vector_projection(new_x, current_x[:2])
    #         # x_diff = np.linalg.norm(new_x_projected) * slide_dir
    #         # z_diff = z * slide_dir
    #         # # wrong: should only be the x part
    #         # # change_in_x = np.linalg.norm(vector_projection(x, current_x))
    #         # perturb_jumpdof(pose, *self.x_params, value=x_diff)
    #         # perturb_jumpdof(pose, *self.z_params, value=z_diff)
    #         # get new unit sphere
    #         # unit_circle_size = np.linalg.norm((current_x_norm - x_diff) / 2)
    #         # y = y / unit_circle_size # y in unit sphere if the unit sphere had r=1
    #         # # angle = math.acos(math.radians(y))
    #         # angle = math.acos(y)
    #         # perturb_jumpdof(pose, *self.z_rot_params, value=(-1 * angle))
    #
    #         # change = self.trans_mag * self.twofold_axis * slide_dir
    #         # x_norm, z_norm = self.get_norm_z_and_x_vector(pose)
    #         # change_proj = vector_projection_on_subspace(change, x_norm, z_norm)
    #         # # now project the x part onto x and z_norm onto z
    #         # change_in_x = np.linalg.norm(vector_projection(change_proj, x_norm))
    #         # change_in_z = np.linalg.norm(vector_projection(change_proj, z_norm))
    #         # perturb_jumpdof(pose, *self.x_params, value=slide_dir * change_in_x)
    #         # perturb_jumpdof(pose, *self.z_params, value=slide_dir * change_in_z)
