#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for SymmetryMapper
@Author: Mads Jeppesen
@Date: 11/18/22
"""
import subprocess
from itertools import combinations
from shapedesign.src.utilities.alignment import tmalign
from pyrosetta import pose_from_file
import numpy as np
import pandas as pd
from cubicsym.alphafold.matcher import Matcher
import uuid
import numpy as np

class SymmetryMapper:
    """Finds the chain combination that produces the best overlap with predicted AlphaFold structure,
    when using the options the -a -i in make_symdef_file.pl

    The documention and options for make_symmdef_file.pl is here:
    https://www.rosettacommons.org/docs/latest/application_documentation/utilities/make-symmdef-file
    """

    def __init__(self, rosetta_path="/home/mads/Rosetta_release", tmp_file_dir="/tmp"):
        """Initialize class.

        :param rosetta_path: Path to the Rosetta folder.
        :param tmp_file_dir: Path to a directory where temporary files can be stored.
        """
        self.rosetta_path = rosetta_path
        self.make_symdef_file_path = f"{self.rosetta_path}/main/source/src/apps/public/symmetry/make_symmdef_file.pl"
        self.tmp_file_dir = tmp_file_dir
        self.matcher = Matcher()

    def get_combinations(self, pose):
        """Get combinations of the 2 pairs of chains"""
        int_combos = list(combinations([c for c in range(1, pose.num_chains() + 1)], 2))
        str_combos = [tuple([pose.pdb_info().chain(pose.chain_begin(c)) for c in cc]) for cc in int_combos]
        return int_combos, str_combos

    def get_chain_strings(self, pose, chain_matches):
        return [(pose.pdb_info().chain(pose.chain_begin(a)), pose.pdb_info().chain(pose.chain_begin(b))) for (a, b) in chain_matches]

    def apply(self, pose, cn, return_df=False):
        """Finds the best chain combination (see class documentation) for a given cn symmetry. There's an option to return a
        pandas DateFrame containing all information gathered throughout the selection."""
        tmp_file = f"{self.tmp_file_dir}/{uuid.uuid4()}.pdb"
        pose.dump_pdb(tmp_file)  # write pose to tmp_file
        combo_info = {"combo": [], "success": [], "tmscore": [], "total_rmsd": [], "chain_matches_int": [],
                      "chain_matches_str": [], "main_rmsd": []}
        int_combos, str_combos = self.get_combinations(pose)
        for int_combo, str_combo in zip(int_combos, str_combos):
            combo_info["combo"].append(str_combo)
            main, other = str_combo
            sp = subprocess.Popen([self.make_symdef_file_path, "-m", "NCS", "-p", tmp_file, "-a", main, "-i", other],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = sp.communicate()
            out, err = out.decode(), err.decode()
            err = err.split("\n")
            axis_found = err[2].split(" ")[1]
            if axis_found == f"{cn}-fold":
                combo_info["success"].append(True)
                combo_info["main_rmsd"].append(float(err[1].split("RMS=")[1][:-1]))
                out_model = tmp_file.replace(".pdb", f"_model_{''.join(str_combo)}.pdb")
                symmetrized_pose = pose_from_file(out_model)
                # initial alignment without specifying chains. Tmscore will not be good since it is based on the matches the same chains
                _ = tmalign(pose, symmetrized_pose)
                # now we align again, now specifying the correct chain mapping
                chain_matches, tmscore, rmsd = self.matcher.apply(pose, symmetrized_pose, move_poses=False)
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
                    if k not in ("success", "combo"):
                        combo_info[k].append(np.nan)
        df = pd.DataFrame(combo_info).sort_values("total_rmsd")
        if return_df:
            return df
        else:
            returns = {"success": False, "best_combo": np.nan, "tmscore": np.nan, "total_rmsd": np.nan, "main_rmsd": np.nan}
            if any(df["success"].values):
                returns["success"] = True
                returns["best_combo"] = df["combo"].values[0]
                returns["tmscore"] = df["tmscore"].values[0]
                returns["main_rmsd"] = df["main_rmsd"].values[0]
                returns["total_rmsd"] = df["total_rmsd"].values[0]
                return returns
            else:
                return returns



# RULES for Cn symmetries:
# only use 1 main chain and 1 other chain.

# make_symmdef_file.pl -m NCS -a A -i B -p hffold.pdb
# Running in mode NCS.
# Aligning A and B wth RMS=9.2519998564275e-07.
# Found 4-fold (4) axis to B : 1.56125112837913e-16 -1.77809156287623e-17 1
# translation error = 3.41461447127311e-14
# system center	 -1.06581410364015e-14 -3.5527136788005e-15 40.2777117647059
# Found a total of 4 monomers in the symmetric complex.
# Placing 5 virtual residues.
#  Adding interface '0'
#  Adding interface '1'
#  Adding interface '2'
#  Adding interface '3'
# symmetry_name hffold__4
# E = 4*VRT0_base + 4*(VRT0_base:VRT1_base) + 2*(VRT0_base:VRT2_base)
# anchor_residue COM
# virtual_coordinates_start
# xyz VRT0  -0.9946675,0.1031333,0.0000000  -0.1031333,-0.9946675,-0.0000000  -0.0000000,-0.0000000,40.2777118
# xyz VRT0_base  -0.9946675,0.1031333,0.0000000  -0.1031333,-0.9946675,-0.0000000  30.2560529,-3.1371353,40.2777118
# xyz VRT1  0.1031333,0.9946675,0.0000000  -0.9946675,0.1031333,0.0000000  -0.0000000,-0.0000000,40.2777118
# xyz VRT1_base  0.1031333,0.9946675,0.0000000  -0.9946675,0.1031333,0.0000000  -3.1371353,-30.2560529,40.2777118
# xyz VRT2  0.9946675,-0.1031333,0.0000000  0.1031333,0.9946675,0.0000000  -0.0000000,-0.0000000,40.2777118
# xyz VRT2_base  0.9946675,-0.1031333,0.0000000  0.1031333,0.9946675,0.0000000  -30.2560529,3.1371353,40.2777118
# xyz VRT3  -0.1031333,-0.9946675,0.0000000  0.9946675,-0.1031333,-0.0000000  -0.0000000,-0.0000000,40.2777118
# xyz VRT3_base  -0.1031333,-0.9946675,0.0000000  0.9946675,-0.1031333,-0.0000000  3.1371353,30.2560529,40.2777118
# xyz VRT  0.0000000,-1.0000000,0.0000000  1.0000000,0.0000000,-0.0000000  -0.0000000,-1.0000000,40.2777118
# virtual_coordinates_stop
# connect_virtual JUMP0_to_com VRT0 VRT0_base
# connect_virtual JUMP0_to_subunit VRT0_base SUBUNIT
# connect_virtual JUMP1_to_com VRT1 VRT1_base
# connect_virtual JUMP1_to_subunit VRT1_base SUBUNIT
# connect_virtual JUMP2_to_com VRT2 VRT2_base
# connect_virtual JUMP2_to_subunit VRT2_base SUBUNIT
# connect_virtual JUMP3_to_com VRT3 VRT3_base
# connect_virtual JUMP3_to_subunit VRT3_base SUBUNIT
# connect_virtual JUMP0 VRT VRT0
# connect_virtual JUMP1 VRT0 VRT1
# connect_virtual JUMP2 VRT0 VRT2
# connect_virtual JUMP3 VRT0 VRT3
# set_dof JUMP0_to_com x(30.4182569755874)
# set_dof JUMP0_to_subunit angle_x angle_y angle_z
# set_jump_group JUMPGROUP2 JUMP0_to_com JUMP1_to_com JUMP2_to_com JUMP3_to_com
# set_jump_group JUMPGROUP3 JUMP0_to_subunit JUMP1_to_subunit JUMP2_to_subunit JUMP3_to_subunit
# Writing interface 0 as chain A
# Writing interface 1 as chain B
# Writing interface 2 as chain C
# Writing interface 3 as chain D
