#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for SymmetryMapper
@Author: Mads Jeppesen
@Date: 11/18/22
"""
import subprocess
from itertools import combinations
from pyrosetta import pose_from_file, Pose, init
import pandas as pd
from cubicsym.alphafold.matcher import Matcher
import uuid
import numpy as np
from symmetryhandler.symmetrysetup import SymmetrySetup
from symmetryhandler.mathfunctions import rotation_matrix, vector_angle, vector_projection_on_subspace, rotation_matrix_from_vector_to_vector, vector_projection
from symmetryhandler.reference_kinematics import x_rotation_matrix
from io import StringIO
from pathlib import Path
import tempfile
from cubicsym.cubicsetup import CubicSetup
from symmetryhandler.symmetrysetup import SymmetrySetup
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from pyrosetta.rosetta.core.conformation.symmetry import residue_center_of_mass
from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit

class SymmetryMapper:
    """Finds the chain combination that produces the best overlap with predicted AlphaFold structure,
    when using the options the -a -i in make_symdef_file.pl

    The documention and options for make_symmdef_file.pl is here:
    https://www.rosettacommons.org/docs/latest/application_documentation/utilities/make-symmdef-file
    """

    def __init__(self, rosetta_path="/home/mads/Rosetta_release"):
        """Initialize class.

        :param rosetta_path: Path to the Rosetta folder.
        :param flip_from_first_apply: Sets the flip direction based on the first structure that is parsed to the
        """
        self.rosetta_path = rosetta_path
        self.make_symdef_file_path = f"{self.rosetta_path}/main/source/src/apps/public/symmetry/make_symmdef_file.pl"
        self.tmp_file_dir = tempfile.gettempdir()
        self.matcher = Matcher()

    def get_combinations(self, pose):
        """Get combinations of the 2 pairs of chains"""
        int_combos = list(combinations([c for c in range(1, pose.num_chains() + 1)], 2))
        str_combos = [tuple([pose.pdb_info().chain(pose.chain_begin(c)) for c in cc]) for cc in int_combos]
        return int_combos, str_combos

    def get_chain_strings(self, pose, chain_matches):
        return [(pose.pdb_info().chain(pose.chain_begin(a)), pose.pdb_info().chain(pose.chain_begin(b))) for (a, b) in chain_matches]

    def _get_x(self, out):
        for line in out.split("\n"):
            l = line.split()
            if len(l) >= 2 and l[0] == "set_dof" and l[1] == "JUMP0_to_com":
                x = l[2].replace("x(", "").replace(")", "")
                return float(x)
        else:
            raise ValueError("X WAS NOT FOUND!")

    def get_center(self, pose):
        """Get the center of the pose"""
        xyz = np.zeros(3)
        n = 0
        # if use_nterm:
        #     assert is_symmetric(pose), "To use use_cterm=True the pose needs to be symmetrical"
        #     for chain in range(1, pose.num_chains()): # DONT use the last chain as it is based on VRTs
        #         if pose.residue(pose.chain_begin(chain)).is_protein():
        #             xyz += np.array(pose.residue(pose.chain_begin(chain)).atom("CA").xyz())
        #             n += 1
        #         else:
        #             raise ValueError("The pose needs to have a C-alpha atom at its N-termini")
        #
        #     center = xyz / n
        # else:
        for ri in range(1, pose.size() + 1):
            if pose.residue(ri).is_protein():
                xyz += np.array(pose.residue(ri).atom("CA").xyz())
                n += 1
        center = xyz / n
        return center

    def align_pose_along_z(self, pose, current_z_axis, onto_z_axis):
        self.center_pose_based_on_ca(pose)
        R = rotation_matrix_from_vector_to_vector(current_z_axis, onto_z_axis)
        # axis = np.cross(current_z_axis, onto_z_axi)
        # angle = vector_angle(, z_axis)
        # R = rotation_matrix(axis, angle)
        pose.rotate(R)

    def align_to_x_axis(self, pose, anchor_resi, x_axis, y_axis, z_axis):
        anchor_ca_xyz = np.array(pose.residue(anchor_resi).atom("CA").xyz())
        x_proj = vector_projection_on_subspace(anchor_ca_xyz, x_axis, y_axis, atol=1e-4)
        angle = - vector_angle(x_proj, x_axis)
        if vector_angle(np.cross(x_proj, x_axis), z_axis) < 90:
            angle = - angle
        R = rotation_matrix(z_axis, angle)
        pose.rotate(R)

    def do_a_180_around_axis(self, pose, axis):
        """Rotates the pose 180 degrees around the axis."""
        pose.rotate(rotation_matrix(axis, 180))

    def center_pose_based_on_ca(self, pose):
        """Centers the pose while only considering the CA atoms."""
        pose_center = self.get_center(pose)
        pose.translate(-pose_center)

    def center_pose_based_on_anchor(self, pose, anchor_xyz):
        """Centers the pose so that the anchor CA atom is at origo."""
        pose.translate(-anchor_xyz)

    def is_point_symmetric(self, pose, cn, ca_rmsd_threshold=2):
        """Checks if the pose is point symmetric."""
        pose = pose.clone()
        pose.center()
        self.align_pose_along_z(pose, [0, 0, 1])
        chains = [Pose(pose, pose.chain_begin(i), pose.chain_end(i)) for i in range(1, cn + 1)]
        # map first chain onto all the others
        first = chains[0]
        angle = 360 / cn
        total_success = 0
        for chain in chains[1:]:
            for mul in range(1, cn):
                R = rotation_matrix([0,0,1], angle*mul)
                temp = first.clone()
                temp.rotate(R)
                ca_rmsd = self.matcher.CA_rmsd_no_super(temp, chain, range(1, temp.size() + 1), range(1, chain.size() + 1))
                if ca_rmsd < ca_rmsd_threshold:
                    total_success += 1
        if total_success == cn - 1:
            return True
        return False

    def process_data(self, poses_to_use, model_outdir):
        for outdir in ("up", "down"):
            out_data = {"model_name": [], "JUMPHFfold111": []}
            outdir = Path(model_outdir.joinpath(outdir))
            outdir.mkdir(parents=True, exist_ok=True)
            outdir_input = outdir.joinpath("input")
            outdir_input.mkdir(parents=True, exist_ok=True)
            outdir_symm = outdir.joinpath("symm")
            outdir_symm.mkdir(parents=True, exist_ok=True)
            for v in poses_to_use:
                pose, model, out, anchor_resi = v
                # dump the symmetrized version of the multimerto disk
                pose.dump_pdb(f"{outdir_symm}/{Path(model).stem}_symm.pdb")
                # dump the input file to disk
                pose_main = Pose(pose, pose.chain_begin(1), pose.chain_end(1))
                anchor_xyz = np.array(pose_main.residue(anchor_resi).atom("CA").xyz())
                pose_main.translate(-anchor_xyz)
                # dump the input
                out_name = f"{outdir_input}/{Path(model).stem}_input.pdb"
                pose_main.dump_pdb(out_name)
                # store data
                x = self._get_x(out)
                out_data["model_name"] = model
                # fixme this should be dependent on the particular symmetry you are using
                out_data["JUMPHFfold111"] = x
        return out_data

    def run(self, model, cn, symmetry, chains_allowed=None, pymolmover=None, T3F=False):
        """model can either be path the models directly or the poses themselves."""
        init("-initialize_rigid_body_dofs")
        # todo: it should handle failures as well
        # todo: it should handle up side down cases as well
        if isinstance(model, Path):
            pose_asym = pose_from_file(str(model))
        elif isinstance(model, Pose):
            pose_asym = model
        else:
            raise ValueError("model must be of either type: Pose or Path")
        if chains_allowed is None:
            chains_allowed = [pose_asym.pdb_info().chain(pose_asym.chain_begin(i)) for i in range(1, pose_asym.num_chains() + 1)]
        # assert pose.pdb_info().chain(1) == "A", "This checks that make_symdef_file always produced the first chain as A. " \
        #                                         "We use the first chain below as the main. but the original chain could be " \
        #                                         "from B or C etc."

        # find combinations
        combo_info = self.find_combos(pose_asym, cn, main_chains_allowed=chains_allowed, check_for_point_symmetry=False) # fixme: set to True, now it is for speedup

        # check for succes
        # FIXME
        exit("YOU NEED TO FIX the return statement here. The below thing will produce an error.")
        if not combo_info.get("success"):
            print(f"{pose_asym.pdb_info().name()} is not {cn}-fold symmetric and will not be used.")
        else:
            # extract the best combination and rerun the make_symdef_file.sh script with that
            main_str, other_str = combo_info["best_combo"]
            tmp_id = f"{self.tmp_file_dir}/{uuid.uuid4()}"
            tmp_file = f"{tmp_id}.pdb"
            pose_asym.dump_pdb(tmp_file)  # write pose to tmp_file
            sp = subprocess.Popen([self.make_symdef_file_path, "-m", "NCS", "-p", tmp_file, "-a", main_str, "-i", other_str],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = sp.communicate()
            out, err = out.decode(), err.decode()

            # system_center = list(map(float, err.split("\n")[4].split()[-3:]))

            # construct the best combination pose from the outout of the make_symdef_file.sh script
            ss_cn = SymmetrySetup(file=StringIO(out))
            pose_sym = pose_from_file(f"{tmp_id}_INPUT.pdb")
            ss_cn.make_symmetric_pose(pose_sym)

            # align the and gets it cubic symmetry setup
            cs, x_axis, y_axis, z_axis = self.align_pose_along_fold(pose_sym, str(cn), symmetry, ss_cn, T3F)

            # center chain A on anchor resi
            input_pose = Pose()
            extract_asymmetric_unit(pose_sym, input_pose, False)
            input_pose.pdb_info().set_chains("A")
            anchor_xyz = np.array(input_pose.residue(residue_center_of_mass(input_pose.conformation(), 1, input_pose.chain_end(1))).atom("CA").xyz())
            self.center_pose_based_on_anchor(input_pose, anchor_xyz)
            # now create one that is flipped around z
            input_pose_flip = input_pose.clone()
            self.do_a_180_around_axis(input_pose_flip, x_axis)
            input_pose, input_pose_flip = self.correct_flip(input_pose, input_pose_flip, x_axis, y_axis, z_axis)
            input_pose_asym = input_pose.clone()
            input_pose_flip_asym = input_pose_flip.clone()
            cs.make_symmetric_pose(input_pose)
            cs.make_symmetric_pose(input_pose_flip)

            return cs, input_pose, input_pose_flip, input_pose_asym, input_pose_flip_asym, self.get_x_trans(anchor_xyz, z_axis), main_str

    def correct_flip(self, input_pose, input_pose_flip, x_axis, y_axis, z_axis):
        """Makes sure the poses labelled as 'flipped' stays consistent. A pose is labelled as 'flipped' if its N-termini CA atom is
        located above the fold-plane (defined by the vectors spanned by x_axis and y_axis)."""
        n_term_ca_xyz = np.array(input_pose.residue(1).atom("CA").xyz())
        vec = n_term_ca_xyz - vector_projection_on_subspace(n_term_ca_xyz, x_axis, y_axis)
        # if vec is not in the same direction as the z_axis it is 'flipped' aka the CA atom is below the plane
        angle = vector_angle(vec, z_axis)
        if angle > 90.0: # swap variables:
            return input_pose_flip, input_pose
        else:
            return input_pose, input_pose_flip


    def get_x_trans(self, anchor_xyz, z_axis):
        """Get the x translation for the Cn symmetrical structure."""
        return np.linalg.norm(vector_projection(anchor_xyz, z_axis) - anchor_xyz)
        # make_symmdef_file does not give the correct x_distance
        # for dof, doftype, val in ss_cn._dofs["JUMP0_to_com"]:
        #     if dof == "x" and doftype == "translation":
        #         return val

    def align_pose_along_fold(self, pose: Pose, cn:str, symmetry:str, ss_cn: SymmetrySetup, T3F):
        # get the symmetry axis vrt
        vrt0 = ss_cn.get_vrt("VRT0")._vrt_orig
        current_z_axis = ss_cn.get_vrt("VRT0")._vrt_z
        if symmetry == "I":
            if cn == "5":
                fold, vrt_id = "HF", "HF"
            elif cn == "3":
                fold, vrt_id = "3F", "31"
            elif cn == "2":
                fold, vrt_id = "2F", "21"
        elif symmetry == "O":
            if cn == "4":
                fold, vrt_id = "HF", "HF"
            elif cn == "3":
                fold, vrt_id = "3F", "31"
            elif cn == "2":
                fold, vrt_id = "2F", "21"
        elif symmetry == "T":
            if cn == "3":
                if T3F:
                    fold, vrt_id = "3F", "31"
                else:
                    fold, vrt_id = "HF", "HF"
            elif cn == "2":
                fold, vrt_id = "2F", "21"
        return self._alignment_subroutine(pose, symmetry, current_z_axis, fold, vrt_id)

    def _alignment_subroutine(self, pose, symmetry, current_z_axis, fold, vrt_id):
        cs = CubicSetup()
        cs.load_norm_symdef(symmetry, fold)
        onto_z_axis = - cs.get_vrt(f"VRT{vrt_id}fold")._vrt_z
        self.align_pose_along_z(pose, current_z_axis, onto_z_axis)
        onto_x_axis = - cs.get_vrt(f"VRT{vrt_id}fold")._vrt_x
        onto_y_axis = cs.get_vrt(f"VRT{vrt_id}fold")._vrt_y
        anchor_resi = residue_center_of_mass(pose.conformation(), 1, pose.chain_end(1))
        self.align_to_x_axis(pose, anchor_resi, onto_x_axis, onto_y_axis, onto_z_axis)
        return cs, onto_x_axis, onto_y_axis, onto_z_axis

    def get_combos(self, pose, main_chains_allowed):
        # extract the unique chain combinations
        int_combos, str_combos = self.get_combinations(pose)
        # get both directions so we can have main be both index 0 and 1
        int_combos += [tuple(reversed(combo)) for combo in int_combos]
        str_combos += [tuple(reversed(combo)) for combo in str_combos]
        # remove combines in which the main (which is index 0) is not in main_chains_allowed
        int_combos = [int_combo for int_combo, str_combo in zip(int_combos, str_combos) if str_combo[0] in main_chains_allowed]
        str_combos = [str_combo for str_combo in str_combos if str_combo[0] in main_chains_allowed]
        return int_combos, str_combos

    def find_combos(self, pose, cn, return_df=False, main_chains_allowed=None, check_for_point_symmetry=False):
        """Finds the best chain combination (see class documentation) for a given cn symmetry. There's an option to return a
        pandas DateFrame containing all information gathered throughout the selection."""
        tmp_file = f"{self.tmp_file_dir}/{uuid.uuid4()}.pdb"
        pose.dump_pdb(tmp_file)  # write pose to tmp_file
        combo_info = {"combo": [], "success": [], "tmscore": [], "total_rmsd": [], "chain_matches_int": [],
                      "chain_matches_str": [], "main_rmsd": [], "combo_int":[]}
        int_combos, str_combos = self.get_combos(pose, main_chains_allowed)
        for int_combo, str_combo in zip(int_combos, str_combos):
            combo_info["combo"].append(str_combo)
            combo_info["combo_int"].append(int_combo)
            main, other = str_combo
            sp = subprocess.Popen([self.make_symdef_file_path, "-m", "NCS", "-p", tmp_file, "-a", main, "-i", other],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = sp.communicate()
            out, err = out.decode(), err.decode()
            err = err.split("\n")
            axis_found = err[2].split(" ")[1]
            if axis_found == f"{cn}-fold" and (self.is_point_symmetric(pose, int(cn)) if check_for_point_symmetry else True):
                combo_info["success"].append(True)
                combo_info["main_rmsd"].append(float(err[1].split("RMS=")[1][:-1]))
                out_model = tmp_file.replace(".pdb", f"_model_{''.join(str_combo)}.pdb")
                symmetrized_pose = pose_from_file(out_model)
                chain_matches, alignment, tmscore, rmsd = self.matcher.apply(pose, symmetrized_pose, move_poses=False)
                combo_info["chain_matches_int"].append(chain_matches)
                combo_info["chain_matches_str"].append(self.get_chain_strings(pose, chain_matches))
                combo_info[f"tmscore"].append(tmscore)
                combo_info[f"total_rmsd"].append(rmsd)
                pose.pdb_info().name("pose")
                symmetrized_pose.pdb_info().name("symmetrized")
            else:
                combo_info["success"].append(False)
                # add nans to all key-value pairs
                for k in combo_info.keys():
                    if k not in ("success", "combo", "combo_int"):
                        combo_info[k].append(np.nan)
        df = pd.DataFrame(combo_info).sort_values("total_rmsd")
        if return_df:
            return df
        else:
            returns = {"success": False, "best_combo": np.nan, "tmscore": np.nan, "total_rmsd": np.nan, "main_rmsd": np.nan}
            if any(df["success"].values):
                returns["success"] = True
                returns["best_combo"] = df["combo"].values[0]
                returns["best_combo_int"] = df["combo_int"].values[0]
                returns["tmscore"] = df["tmscore"].values[0]
                returns["main_rmsd"] = df["main_rmsd"].values[0]
                returns["total_rmsd"] = df["total_rmsd"].values[0]
                return returns
            else:
                return returns
