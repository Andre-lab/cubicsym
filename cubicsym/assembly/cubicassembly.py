#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicSymmetricAssembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import numpy as np
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
import time
import math
import difflib
import os
import random
import string
import itertools
from io import StringIO
from cubicsym.assembly.assembly import Assembly
from cubicsym.mathfunctions import shortest_path, criteria_check, angle, distance, vector
from symmetryhandler.mathfunctions import vector_angle, vector_projection, rotate, vector_projection_on_subspace, rotation_matrix, rotate_right_multiply
from symmetryhandler.coordinateframe import CoordinateFrame
from symmetryhandler.symmetrysetup import SymmetrySetup
from cubicsym.cubicsetup import CubicSetup
from cubicsym.exceptions import ToHighGeometry, ToHighRMSD, ValueToHigh, NoSymmetryDetected
from pyrosetta.rosetta.std import ostringstream
from pyrosetta import init, pose_from_file, Pose
from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model
from pathlib import Path
import tempfile

class CubicSymmetricAssembly(Assembly):
    """Cubic symmetrical assembly of either I, O or T symmetry. Has additional methods from a regular assembly related to symmetry."""

    def __init__(self, mmcif_file=None, mmcif_symmetry: str = "1", force_symmetry=None, start_rmsd_diff=0.5, start_angles_diff=2.0, id_="1",
                 rmsd_diff_increment=1, angle_diff_increment=2, total_increments=5, rosetta_units=None, ignore_chains=None,
                 use_full=False, model_together=False):
        """Initialization of an instance from an mmCIF file and either a specified symmetry or assembly id.

        :param mmcif_file: mmCIF file to create the assembly from.
        :param mmcif_symmetry: Symmetry of the assembly. Either I, O, T or the exact assembly id. If I, O or T is given, it
        will construct the assembly from the first assembly matching the requested symmetry type.
        :param force_symmetry: Instead of auto detecting the symmetry this will force a specific symmetry from a given assembly id. For instance,
        if you want the assembly id of 1 to be I symmetry. Parse force_symmetry="I" and mmcif_symmetry="1".
        :param start_rmsd_diff: The starting allowed RMSD difference when creating symmetry. If the assembly is not perfectly symmetric,
        the RMSD difference is > 0.0. Higher RMSD can arise, for example, from the chains being different, from different
        conformations of the backbone or from the chains can be positioned differently.
        :param rmsd_diff_increment: If no symmetry is found the script will reattempt to find symmetry but now with an allowed RMSD of
        start_rmsd_diff + rmsd_diff_increment. The script will continue with higher and higher RMSD a total of total_increments times.
        :param total_increments: The total amount of increments to try for both rmsd_diff and angle_diff.
        :param start_angles_diff: Same argument as for the start_rmsd_diff although this concerns the angles of the folds with respect
        to the geomtric center of each subunit.
        :param angle_diff_increment: Same argument as for the start_rmsd_diff, the next increment will be:
        start_angle_diff + angle_diff_increment
        :param id:
        """
        assert mmcif_symmetry in ("I", "O", "T") or mmcif_symmetry.isdigit(), \
            "symmetry definition is wrong. Has to be either I, O, T or a number exclusively."
        super().__init__(mmcif_file, mmcif_symmetry if mmcif_symmetry.isdigit() else "1", id_, rosetta_units, ignore_chains, use_full, model_together)
        self.intrinsic_perfect_symmetry = None
        self.idealized_symmetry = None
        self.start_rmsd_diff = start_rmsd_diff
        self.start_angles_diff = start_angles_diff
        self.rmsd_diff_increment = rmsd_diff_increment
        self.angles_diff_increment = angle_diff_increment
        self.total_increments = total_increments
        self.lowest_5fold_rmsd = None
        self.highest_5fold_accepted_rmsd = None
        self.lowest_4fold_rmsd = None
        self.highest_4fold_accepted_rmsd = None
        self.lowest_3fold_rmsd = None
        self.highest_3fold_accepted_rmsd = None
        self.symmetry = None
        if mmcif_file:
            self.center()
            if force_symmetry:
                self.force_symmetry(mmcif_file, mmcif_symmetry, force_symmetry)
            else:
                self.find_symmetry(mmcif_file, mmcif_symmetry)

    @classmethod
    def _extract_symmetric_info(cls, pose, setup, n_chains, input_file=None):
        new_pose = Pose()
        extract_asymmetric_unit(pose, new_pose, False)
        # create a Model from the symmetrized pose master subunit
        # fsuffix = Path(input_file).suffix
        # assert fsuffix in (".cif", ".pdb"), f"File has to have either extension '.cif' of '.pdb' not {fsuffix}"
        # if fsuffix == ".cif":
        #     p = MMCIFParser()
        # else:
        p = PDBParser(PERMISSIVE=1)
        buffer = ostringstream()
        new_pose.dump_pdb(buffer)
        if input_file is not None and isinstance(input_file, str):
            structure = p.get_structure(input_file, StringIO(buffer.str()))
        elif input_file is None:
            with tempfile.TemporaryFile() as asym_pose_file:
                apose = setup.make_asymmetric_pose(pose)
                apose.dump_pdb(str(asym_pose_file))
                structure = p.get_structure(asym_pose_file, StringIO(buffer.str()))
        else:
            raise ValueError
        # construct an emtpy CubicSymmetricAssembly class
        cass = cls()
        cass.symmetry = cass.determine_cubic_symmetry_from_setup(setup)
        hf = cass.get_highest_fold()
        angle = cass._get_angle_of_highest_fold()
        chain_ids = cls._get_rosetta_chain_ordering()
        # create a master model
        count = itertools.count(start=1)
        master = Model(f"{next(count)}")
        for n, chain in enumerate(structure.get_chains()):
            if n >= n_chains:
                break
            chain.id = next(chain_ids) # should be the same but we need to use them up anyways
            master.add(chain)
        cass.add(master)
        return cass, hf, angle, master, count, chain_ids

    @classmethod
    def from_pose_input(cls, pose, setup, half_capsid_only=False):
        """Initialize an instance from a pose and a CubicSetup. If no symdef file is given,
        assume that the symdef information is stored in a SYMMETRY line in the input file."""
        # create and symmetrize a pose.
        assert is_symmetric(pose)
        n_chains = pose.num_chains()
        if not setup.is_hf_based():
            from cubicsym.actors.symdefswapper import SymDefSwapper
            sds = SymDefSwapper(pose, setup)
            setup = sds.foldHF_setup
            pose = sds.create_hffold_pose(pose)
        cass, hf, angle, master, count, chain_ids = cls._extract_symmetric_info(pose, setup, n_chains)
        z1high = -setup.get_vrt("VRTHFfold")._vrt_z
        z2high = -setup.get_vrt("VRT2fold")._vrt_z
        z3high = -setup.get_vrt("VRT3fold")._vrt_z
        #     z1high = -setup.get_vrt("VRT31fold")._vrt_z
        #     z2high = -setup.get_vrt("VRT32fold")._vrt_z
        #     z3high = -setup.get_vrt("VRT3fold")._vrt_z
        # elif setup.is_3f_based():
        #     z1high = -setup.get_vrt("VRT21fold")._vrt_z
        #     z2high = -setup.get_vrt("VRT32fold")._vrt_z
        #     z3high = -setup.get_vrt("VRT3fold")._vrt_z
        # 1. Make high fold around master
        for i in range(1, hf):
            new_subunit = Model(f"{next(count)}")
            for chain in master.get_chains():
                new_chain = chain.copy()
                new_chain.transform(rotation_matrix(z1high, angle * i), [0, 0, 0])
                new_chain.id = next(chain_ids)
                new_subunit.add(new_chain)
            cass.add(new_subunit)
        # 2. Make the 2-fold subunit and rotate it around its two-fold axis
        new_subunit = Model(f"{next(count)}")
        for chain in master.get_chains():
            new_chain = chain.copy()
            new_chain.transform(rotation_matrix( z1high + z2high, 180), [0, 0, 0])
            new_chain.id = next(chain_ids)
            new_subunit.add(new_chain)
        cass.add(new_subunit)
        # 3. Make the surrounding 5-folds of the 2-fold subunit first and the rest of them (#hf)
        for i in range(1, hf):
            new_subunit = Model(f"{next(count)}")
            for chain in cass.get_subunits(ids=f"{hf+1}")[0].get_chains():
                new_chain = chain.copy()
                new_chain.transform(rotation_matrix(z2high, angle * i), [0, 0, 0])
                new_chain.id = next(chain_ids)
                new_subunit.add(new_chain)
            cass.add(new_subunit)
        for i in range(1, hf):
            for subunit in cass.get_subunits(ids = tuple(map(str, range(hf+1, hf*2+1)))):
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_matrix(z1high, angle * i), [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
        # For T we are done but for I and O we need a little more
        vec1 = z2high + ((z3high - z2high) / 2.0)  # a vector from midlle of the 2-fold/hf-fold axis to the 3-fold/hf-fold axis
        if cass.symmetry == "O":
            rotation_for_O = rotation_matrix(vec1, 180.0)
            for subunit in cass.get_subunits(ids=("1", "2", "3", "4"))[:]:
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_for_O, [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
        elif cass.symmetry == "I":
            if half_capsid_only:
                return cass
            # before the loops we create vectors that are important for rotating half the capsid
            vec2 = np.dot(z1high, rotation_matrix(np.cross(vec1, z1high), vector_angle(z1high, vec1) * 2))  # points to another 5-fold axis just below the 3/2-fold/hf-fold-axis
            vec3 = vec2 + ((z2high - vec2) / 2.0)  # points bewteen the 3-fold axis and the above hf-fold axis
            rotation_for_I = rotation_matrix(vec3, 180.0)
            for subunit in cass.get_subunits()[:]:
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_for_I, [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
            # need a twofold rotation
        return cass

    @classmethod
    def from_rosetta_input(cls, input_file, symdef_file=None):
        """Initialize an instance from a Rosetta input file and optionally a symdef file. If no symdef file is given,
        assume that the symdef information is stored in a SYMMETRY line in the input file."""
        # create and symmetrize a pose.
        # TODO: The only reason for this is that we want the master subunit in its correct position. We could also
        #  use setup._dofs to do that if we want the code to be pyrosetta free and shorter code.
        init("-initialize_rigid_body_dofs true -pdb_comments")
        pose = pose_from_file(input_file)
        setup = CubicSetup()
        n_chains = pose.num_chains()
        if not symdef_file:
            setup.read_from_pose_comment(pose)
        else:
            setup.read_from_file(symdef_file)
        setup.make_symmetric_pose(pose)
        # from the setup construct the vectors that point to the high-fold centers
        # fixme: if not HF-based you can get it from other axis combined - do that!
        if not setup.is_hf_based():
            from cubicsym.actors.symdefswapper import SymDefSwapper
            sds = SymDefSwapper(pose, setup)
            setup = sds.foldHF_setup
            pose = sds.create_hffold_pose(pose)
        cass, hf, angle, master, count, chain_ids = cls._extract_symmetric_info(pose, setup, n_chains, input_file)
        z1high = -setup.get_vrt("VRTHFfold")._vrt_z
        z2high = -setup.get_vrt("VRT2fold")._vrt_z
        z3high = -setup.get_vrt("VRT3fold")._vrt_z
        #     z1high = -setup.get_vrt("VRT31fold")._vrt_z
        #     z2high = -setup.get_vrt("VRT32fold")._vrt_z
        #     z3high = -setup.get_vrt("VRT3fold")._vrt_z
        # elif setup.is_3f_based():
        #     z1high = -setup.get_vrt("VRT21fold")._vrt_z
        #     z2high = -setup.get_vrt("VRT32fold")._vrt_z
        #     z3high = -setup.get_vrt("VRT3fold")._vrt_z
        # 1. Make high fold around master
        for i in range(1, hf):
            new_subunit = Model(f"{next(count)}")
            for chain in master.get_chains():
                new_chain = chain.copy()
                new_chain.transform(rotation_matrix(z1high, angle * i), [0, 0, 0])
                new_chain.id = next(chain_ids)
                new_subunit.add(new_chain)
            cass.add(new_subunit)
        # 2. Make the 2-fold subunit and rotate it around its two-fold axis
        new_subunit = Model(f"{next(count)}")
        for chain in master.get_chains():
            new_chain = chain.copy()
            new_chain.transform(rotation_matrix( z1high + z2high, 180), [0, 0, 0])
            new_chain.id = next(chain_ids)
            new_subunit.add(new_chain)
        cass.add(new_subunit)
        # 3. Make the surrounding 5-folds of the 2-fold subunit first and the rest of them (#hf)
        for i in range(1, hf):
            new_subunit = Model(f"{next(count)}")
            for chain in cass.get_subunits(ids=f"{hf+1}")[0].get_chains():
                new_chain = chain.copy()
                new_chain.transform(rotation_matrix(z2high, angle * i), [0, 0, 0])
                new_chain.id = next(chain_ids)
                new_subunit.add(new_chain)
            cass.add(new_subunit)
        for i in range(1, hf):
            for subunit in cass.get_subunits(ids = tuple(map(str, range(hf+1, hf*2+1)))):
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_matrix(z1high, angle * i), [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
        # For T we are done but for I and O we need a little more
        vec1 = z2high + ((z3high - z2high) / 2.0)  # a vector from midlle of the 2-fold/hf-fold axis to the 3-fold/hf-fold axis
        if cass.symmetry == "O":
            rotation_for_O = rotation_matrix(vec1, 180.0)
            for subunit in cass.get_subunits(ids=("1", "2", "3", "4"))[:]:
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_for_O, [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
        elif cass.symmetry == "I":
            # before the loops we create vectors that are important for rotating half the capsid
            vec2 = np.dot(z1high, rotation_matrix(np.cross(vec1, z1high), vector_angle(z1high, vec1) * 2))  # points to another 5-fold axis just below the 3/2-fold/hf-fold-axis
            vec3 = vec2 + ((z2high - vec2) / 2.0)  # points bewteen the 3-fold axis and the above hf-fold axis
            rotation_for_I = rotation_matrix(vec3, 180.0)
            for subunit in cass.get_subunits()[:]:
                new_subunit = Model(f"{next(count)}")
                for chain in subunit.get_chains():
                    new_chain = chain.copy()
                    new_chain.transform(rotation_for_I, [0, 0, 0])
                    new_chain.id = next(chain_ids)
                    new_subunit.add(new_chain)
                cass.add(new_subunit)
            # need a twofold rotation
        return cass

    @staticmethod
    def determine_cubic_symmetry_from_setup(setup: SymmetrySetup):
        """Determine the cubic symmetry from a SymmetrySetup object."""
        if "60" in setup.energies:
            return "I"
        elif "24" in setup.energies:
            return "O"
        elif "12" in setup.energies:
            return "T"
        else:
            raise ValueError("Symmetry is not cubic!")

    # TODO OR DELETE
    # @classmethod
    # def from_rosetta_repr(cls, file, symdef_file):
    #     """Initialize an instance from the Rosetta representation file.
    #     The algorithm is as follows:
    #      1. Generate the 2 fold from the chain A
    #      2. generate 5-fold (five subunits) from master subunit
    #      3. generate 5-fold (five sununits) from the subunit that is part of the 2-fold axis with the master subunit.
    #         and rotate the latter five subunit 5-fold  72*5 degrees. Now we have half a capsid.
    #      4. rotate the half capsid 180 degrees at specific point along the middle.
    #
    #     :param file: #TODO
    #     :param symmetry_file: #TODO
    #     :return:"""
    #     init("-initialize_rigid_body_dofs true -pdb_comments")
    #     # pose = pose_from_file(input_file)
    #     # if not symdef_file:
    #     #     symdef_file = pose.data().get_ptr(CacheableDataType.STRING_MAP).map()["SYMMETRY"].replace("|", "\n")
    #     #     s = SymmData()
    #     #     s.read_symmetry_data_from_stream(istringstream(symdef_file))
    #     #     setupmover = SetupForSymmetryMover(s)
    #     #     symdef_file = StringIO(symdef_file)
    #     # else:
    #     #     setupmover = SetupForSymmetryMover(symdef_file)
    #     # setupmover.apply(pose)
    #     # buffer = ostringstream()
    #     # pose.dump_pdb(buffer)
    #     # input_file = StringIO(buffer.str())
    #     # Read symmetry from file
    #     setup = SymmetrySetup()
    #     setup.read_from_file(symdef_file)
    #     # construct an emtpy CubicSymmetricAssembly class
    #     cass = cls()
    #     cass.symmetry = cls.determine_cubic_symmetry_from_setup(setup)
    #     # Read rosetta pdb file from file
    #     fsuffix = Path(file).suffix
    #     assert fsuffix in (".cif", ".pdb"), f"File has to have either extension '.cif' of '.pdb' not {fsuffix}"
    #     if fsuffix == ".cif":
    #         p = MMCIFParser()
    #     else:
    #         p = PDBParser(PERMISSIVE=1)
    #     structure = p.get_structure(file, file)
    #     # Add the chains present in the rosetta representation
    #     chain_ids = cls._get_rosetta_chain_ordering()
    #     n_chains_in_subunit = len(list(structure.get_chains())) // 9
    #     all_subunit_chains = [[]]
    #     for chain in enumerate(structure.get_chains()):
    #         all_subunit_chains[-1].append(chain)
    #         if len(all_subunit_chains[-1]) == n_chains_in_subunit:
    #             all_subunit_chains.append([])
    #     for n, subunit_chains in enumerate(all_subunit_chains, 1):
    #         subunit = Model(f"{n}")
    #         for chain in subunit_chains:
    #             chain.id = next(chain_ids) # should be the same but we need to use them up anyways
    #             subunit.add(chain)
    #         cass.add(subunit)
    #     # Variable that will contain all chains of the assembly
    #     chains = []
    #
    #     # The 3 5-fold axes availble for an icosahedral structure in the symmetry file
    #     # minus because rosetta is awesome and have turned the coordinate systems arounD
    #     z1high = -setup.get_vrt_name("VRTHFfold")._vrt_z
    #     z2high = -setup.get_vrt_name("VRT2fold")._vrt_z
    #     z3high = -setup.get_vrt_name("VRT3fold")._vrt_z
    #
    #     for chain_letter in list(ascii_uppercase)[0:n_chains // 9]:
    #
    #         # for main in ["A"]
    #         # Construct the master structure
    #         master = structure[0][chain_letter]
    #
    #         ### 1
    #         # copy the twofold and rotate it around its two-fold axis
    #         two_fold = cass.get_subunits("1").copy()
    #         mid_z15_z25 = z1high + z2high
    #         two_fold.transform(rotation_matrix(mid_z15_z25, 180), [0, 0, 0])
    #
    #         ### 2
    #         # make 5 fold around master
    #         chains.append(master)
    #         for i in range(1, 5):
    #             master_5fold = master.copy()
    #             master_5fold.transform(rotation_matrix(z1high, 72 * i), [0, 0, 0])
    #             chains.append(master_5fold)
    #
    #         ### 3
    #         # make the surrounding 5-folds
    #         surrounding_5_folds = []
    #         surrounding_5_folds.append(two_fold)
    #         # make the first five fold
    #         for i in range(1, 5):
    #             two_fold_5fold = two_fold.copy()
    #             two_fold_5fold.transform(rotation_matrix(z2high, 72 * i), [0, 0, 0])
    #             surrounding_5_folds.append(two_fold_5fold)
    #         # make the rest (4 of them)
    #         for i in range(1, 5):
    #             for j in range(5):
    #                 extra_5_fold = surrounding_5_folds[j].copy()
    #                 extra_5_fold.transform(rotation_matrix(z1high, 72 * i), [0, 0, 0])
    #                 surrounding_5_folds.append(extra_5_fold)
    #         for chain in surrounding_5_folds:
    #             chains.append(chain)
    #
    #         ### 4
    #         # make the rest of the capsid
    #         # before the loops we create vectors that are important for rotating half the capsid
    #         vec1 = z2high + ((z3high - z2high) / 2.0)  # a vector from midlle of the 2-fold/5-fold axis to the 3-fold/5-fold axis
    #         vec2 = np.dot(z1high, rotation_matrix(np.cross(vec1, z1high), vector_angle(z1high,
    #                                                                              vec1) * 2))  # points to another 5-fold axis just below the 3/2-fold/5-fold-axis
    #         vec3 = vec2 + ((z2high - vec2) / 2.0)  # points bewteen the 3-fold axis and the above 5-fold axis
    #         rotation_for_half_capsid = rotation_matrix(vec3, 180.0)
    #         for chain in chains[:]:
    #             new_chain = chain.copy()
    #             new_chain.transform(rotation_for_half_capsid, [0, 0, 0])
    #             chains.append(new_chain)
    #
    #         # Add all chains to the assembly
    #         for n, chain in enumerate(chains, 1):
    #             subunit = Bio.PDB.Model.Model(str(n))
    #             chain.id = next(chain_ids)  # A
    #             subunit.add(chain)
    #             self.add(subunit)
    #     return self

    def force_symmetry(self, file, mmcif_symmetry, force_symmetry):
        self.symmetry = force_symmetry
        self.from_mmcif(file)

    def find_symmetry(self, file, symmetry):
        """Attempts to retrive the symmetry in either of 2 ways:
            1. If symmetry is a number, will retrieve whatever symmetry that assembly id has.
            2. If symmetry is either I, O or I, will iterate through the available assemblies until the corresponding symmetry is found.
        If No symmetry is found or the symmetry requested as I, O or I is not found it will return an exception."""
        self.symmetry = self._detect_cubic_symmetry()
        # if I, O or T has been parsed then attempt to retrieve from other assemblies if the current one does not match the
        # requested symmetry
        additional_symms = []
        if not symmetry.isdigit():
            n = 1
            while self.symmetry != symmetry and n != len(self.allowed_ids):
                # try another id and check if the symmetry is correct
                self.assembly_id = self.allowed_ids[n]
                self.from_mmcif(file)
                n += 1
                self.symmetry = self._detect_cubic_symmetry()
                if self.symmetry == symmetry:
                    break
                elif self.symmetry:
                    additional_symms.append(self.symmetry)
        if not self.symmetry:
            raise NoSymmetryDetected
        elif additional_symms:
            raise Exception(f"{file} does not contain {','.join(additional_symms)} but does contain {self.symmetry}")
        else:
            print(f"Correct symmetry: {self.symmetry} found from assembly id: {self.assembly_id}")

    def output_rosetta_symmetry(self, symmetry_name=None, input_name=None, master_to_use="1", outformat="cif", idealize=True,
                                foldmap:dict=None) -> set:
        """Sets up a symmetric representation of a cubic assembly for use in Rosetta.

        :param symmetry_name: The name given to the symmetry file (Input to Rosetta)
        :param input_name: The name given to the input file (Input to Rosetta)
        :param master_to_use: The master subunit id to use. Use the ids specified in the assembly.
        :param outformat: output format for the input file.
        :param foldmap: If parsed the HF, 3-fold and 2 fold will be decided based on those values.
        :return:
        """
        if symmetry_name == None:
            symmetry_name = self.id + ".symm"
        if input_name == None:
            input_name = self.id + ".symm"
        start_time = time.time()
        setup, master, selected_ids = self.setup_symmetry(symmetry_name, master_to_use, idealize=idealize, foldmap=foldmap)
        print("Writing the output structure that is used in Rosetta (" + input_name + "." + outformat + ") to disk. Use this one as input for Rosetta!")
        self.output_subunit_as_specified(input_name, master, format=outformat)
        print("Writing the symmetry file (" + symmetry_name + ") to disk.")
        setup.output(symmetry_name, headers=[f"righthanded={setup.calculate_if_rightanded()}"])
        print("Symmetry set up in: " + str(round(time.time() - start_time, 1)) + "s")
        return selected_ids

    def get_symmetry(self):
        """Retrieves the symmetry of the cubic assembly. If it hasn't been calculated then it is calculated."""
        if self.symmetry == None:
            self.center()
            self.symmetry = self._detect_cubic_symmetry()
        return self.symmetry

    def show_symmetry(self, apply_dofs=True, mark_jumps=True):
        """TODO"""
        name = "tmp_symmetry.symm"
        tmp = f"/tmp/{name}"
        setup, _ = self.setup_symmetry(name)
        setup.print_visualization(tmp, apply_dofs, mark_jumps)
        self.cmd.run(self.server_root_path + tmp)
        os.remove(tmp)

    def get_highest_fold(self):
        """Get the highest symmetrical fold for the symmetry."""
        if self.get_symmetry() == "I":
            return 5
        elif self.get_symmetry() == "O":
            return 4
        else: # self.get_symmetry() == "T":
            return 3

    def get_closest_and_furthest_2_fold(self, master_id, master_2_folds):
        """Mark which of the 2-folds are closest and furthest away from the master subunit."""
        twofold_1_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[0][1])
        twofold_2_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[1][1])
        if twofold_1_distance < twofold_2_distance:
            twofold_closest_id = master_2_folds[0][1]
            twofold_furthest_id = master_2_folds[1][1]
        else:
            twofold_closest_id = master_2_folds[1][1]
            twofold_furthest_id = master_2_folds[0][1]
        return twofold_closest_id, twofold_furthest_id

    def remap_chain_labels(self, master_high_fold, master_3_fold, closest_2fold_id, furthest_2fold_id, second_high_fold, third_high_fold):
        """Creates an ordering that maps the order of ids to the chain naming order that will occur when Rosetta names the chains
        after applying symmetry."""
        # see map: {subunit.id: chain.id for subunit in self.get_subunits() for chain in subunit.get_chains()}
        ids_ordering = master_high_fold[:]
        # which high fold is in the two_fold_furthest
        if furthest_2fold_id in second_high_fold:
            for id_ in second_high_fold:
                if id_ in master_3_fold:
                    ids_ordering += [id_, furthest_2fold_id]
                    break
        else:
            for id_ in third_high_fold:
                if id_ in master_3_fold:
                    ids_ordering += [id_, furthest_2fold_id]
                    break
        ids_ordering += [closest_2fold_id]
        # get the 3 fold that is not already in the ids_ordering
        ids_ordering += [id_ for id_ in master_3_fold if id_ not in ids_ordering]
        chain_ordering = self._get_rosetta_chain_ordering()
        random_labels = []
        for id_ in ids_ordering:
            for subunit in self.get_subunits(ids_ordering):
                if subunit.id == id_:
                    # first scramble all ids of the chains else the following error can occur:
                    # Cannot change id from `j` to `n`. The id `and` is already used for a sibling of this entity.
                    for chain in subunit.get_chains():
                        chain.id = "".join(random.choices(string.ascii_lowercase, k=10))
                    # now change each structure
                    for chain in subunit.get_chains():
                        chain.id = next(chain_ordering)
                    break
        # now the rest of the chains have to be named according to the rest of the chain labels
        for subunit in [subunit for subunit in self.get_subunits() if subunit.id not in ids_ordering]:
            for chain in subunit.get_chains():
                chain.id = "".join(random.choices(string.ascii_lowercase, k=10))
            for chain in subunit.get_chains():
                chain.id = next(chain_ordering)

    def has_perfect_symmetry(self,  z1, z2, z3, rtol=1e-9):
        """Checks if the assembly has perfect symmetry by assessing the 3 vectors pointing towards the closest 3 highest fold centers.
        rtol=1e-4 is chosen because this is the accuracy stored in a PDB or MMCIF file."""
        angle = self.facet_angle()
        try:
            # All angles between the highest-fold centers are the correct 'angle'
            assert np.isclose(vector_angle(z1, z2), angle, rtol)
            assert np.isclose(vector_angle(z1, z3), angle, rtol)
            assert np.isclose(vector_angle(z2, z3), angle, rtol)
            # The length between the highest-fold centers are equal
            assert np.isclose(distance(z1, z2), distance(z1, z3), rtol)
            assert np.isclose(distance(z1, z2), distance(z2, z3), rtol)
            # The norm of the vector pointing towards the highest-fold centers are equal.
            # It doesnt matter in the end since the length are set by z1
            # in the final rosetta output file. But it is nice to see.
            assert np.isclose(np.linalg.norm(z1), np.linalg.norm(z2))
            assert np.isclose(np.linalg.norm(z1), np.linalg.norm(z3))
        except AssertionError:
            return False
        else:
            return True

    def facet_angle(self):
        """The angle between the facets of the cube."""
        if self.get_symmetry() == "I":
            return math.degrees(math.acos(1/math.sqrt(5)))
        elif self.get_symmetry() == "O":
            return 90
        else: # self.get_symmetry() == "T":
            return math.degrees(math.acos(-1/3))

    # def idealize_vectors_better(self, z1, z2, z3, rtol=1e-6):
    #     """Idealize the vectors z2 and z3 relative to z1 to achieve perfect I, O or T symmetry."""
    #     angle = self.facet_angle()
    #     # assert that z1 is at the global center
    #     global_x, global_y, global_z, global_center = self._get_global_center()
    #     assert np.isclose(z1[0], 0, rtol=rtol) and np.isclose(z1[1], 0, rtol=rtol), "z1 is not at global center"
    #     # Step 1:
    #     # Set z2 and z3 to be 'angle' away from each other
    #     angle_diff = vector_angle(z2, z3) - angle
    #     # rotate each vector diff/2 away from eachother
    #     # z2 X z3 => z1 direction = left handed, else right handed (True if right handed).
    #     # if it is right handed z2 is to the left and z3 to the right
    #     if self._right_handed_vectors(z2, z3, z1):
    #         z2_new = rotate(z2, rotation_matrix(np.cross(z2, z3), - angle_diff / 2))
    #         z3_new = rotate(z3, rotation_matrix(np.cross(z2, z3), angle_diff / 2))
    #     else:
    #         z2_new = rotate(z2, rotation_matrix(np.cross(z2, z3), angle_diff / 2))
    #         z3_new = rotate(z3, rotation_matrix(np.cross(z2, z3), - angle_diff / 2))
    #     # Step 2:
    #     # Rotate z2 and z3 so that that the are perpindicular to z1
    #     z2_z3_center = (z2_new + z3_new) / 2
    #     z3_z2_to_z1_angle_diff = vector_angle(z2_new - z3_new, z1) - 90
    #     if self._right_handed_vectors(z2_new, z3_new, z1):
    #         pass
    #     else:
    #         pass
    #
    #
    #
    #
    #     assert np.isclose(vector_angle(z1, z2_new), angle, rtol=rtol)
    #     angle_diff = vector_angle(z1, z3) - angle
    #     z3_new = rotate(z3, rotation_matrix(np.cross(z1, z3), angle_diff))
    #     assert np.isclose(vector_angle(z1, z3_new), angle, rtol=rtol)
    #     # Step 2: Set z2 and z3 to be 'angle' away from each other while maintaining their respective 'angle' to z1
    #     # 2a: project onto z1 plane (same as the vectors spanned by global_x and global_y since we assert z1 == along z axis earlier) and
    #     # then get the angle between them
    #     z2_new_proj = vector_projection_on_subspace(z2_new, global_x, global_y)
    #     z3_new_proj = vector_projection_on_subspace(z3_new, global_x, global_y)
    #     projected_angle = self._get_angle_of_highest_fold()
    #     angle_diff = vector_angle(z2_new_proj, z3_new_proj) - projected_angle
    #     # 2b: figure out which way to rotate z2 and z3
    #     if self._right_handed_vectors(z2, z3, z1):
    #         # If True the rotation axis points in the opposite direct of z1
    #         # then rotate z2 in the positive direction and z3 in the negative direction if the angle_diff > 0, else do the opposite
    #         if angle_diff > 0:
    #             z2_rotdir, z3_rotdir = 1, -1
    #         else:
    #             z2_rotdir, z3_rotdir = -1, 1
    #     else:
    #         # If False the rotation axis points in the same direct of z1
    #         # then rotate z2 in negative direction and z3 in the positive direction if the angle_diff > 0, else do the opposite
    #         if angle_diff > 0:
    #             z2_rotdir, z3_rotdir = -1, 1
    #         else:
    #             z2_rotdir, z3_rotdir = 1, -1
    #     # 2c: rotate z2 and z3 with angle / 2 each
    #     z2_final = rotate(z2_new, rotation_matrix(z1, (angle_diff / 2) * z2_rotdir))
    #     z3_final = rotate(z3_new, rotation_matrix(z1, (angle_diff / 2) * z3_rotdir))
    #     assert self.has_perfect_symmetry(z1, z2_final, z3_final), "Symmetry was not idealized appropiately"
    #     return z2_final, z3_final

    def idealize_vectors(self, z1, z2, z3, rtol=1e-9):
        """Idealize the vectors z2 and z3 relative to z1 to achieve perfect I, O or T symmetry."""
        angle = self.facet_angle()
        global_x, global_y, global_z, global_center = self._get_global_center()
        assert np.isclose(z1[0], 0, rtol=rtol) and np.isclose(z1[1], 0, rtol=rtol)
        # Step 1: Set z2 and z3 to be 'angle' away from z1
        angle_diff = vector_angle(z1, z2) - angle
        z2_new = rotate(z2, rotation_matrix(np.cross(z1, z2), - angle_diff))
        assert np.isclose(vector_angle(z1, z2_new), angle, rtol=rtol)
        angle_diff = vector_angle(z1, z3) - angle
        z3_new = rotate(z3, rotation_matrix(np.cross(z1, z3), - angle_diff))
        assert np.isclose(vector_angle(z1, z3_new), angle, rtol=rtol)
        # Step 2: Set z2 and z3 to be 'angle' away from each other while maintaining their respective 'angle' to z1
        # 2a: project onto z1 plane (same as the vectors spanned by global_x and global_y since we assert z1 == along z axis earlier) and
        # then get the angle between them
        z2_new_proj = vector_projection_on_subspace(z2_new, global_x, global_y)
        z3_new_proj = vector_projection_on_subspace(z3_new, global_x, global_y)
        projected_angle = self._get_angle_of_highest_fold()
        angle_diff = vector_angle(z2_new_proj, z3_new_proj) - projected_angle
        # 2b: figure out which way to rotate z2 and z3
        # if True, z2 is to the right else left
        cross = np.cross(z2_new_proj, z3_new_proj)
        if vector_angle(cross, z1) > 90:
            # If True the rotation axis points in the opposite direct of z1
            # then rotate z3 in the positive direction and z2 in the negative direction if the angle_diff > 0, else do the opposite
            if angle_diff > 0:
                z2_rotdir, z3_rotdir = -1, 1
            else:
                z2_rotdir, z3_rotdir = 1, -1
        else:
            # If False the rotation axis points in the same direct of z1
            # then rotate z3 in negative direction and z2 in the positive direction if the angle_diff > 0, else do the opposite
            if angle_diff > 0:
                z2_rotdir, z3_rotdir = 1, -1
            else:
                z2_rotdir, z3_rotdir = -1, 1
        # 2c: rotate z2 and z3 with angle / 2 each
        z2_final = rotate(z2_new, rotation_matrix(z1, (abs(angle_diff) / 2) * z2_rotdir))
        z3_final = rotate(z3_new, rotation_matrix(z1, (abs(angle_diff) / 2) * z3_rotdir))
        assert self.has_perfect_symmetry(z1, z2_final, z3_final), "Symmetry was not idealized appropiately"
        print("z1-z2 angle_diff:", vector_angle(z1, z2_final) - angle)
        print("z1-z3 angle_diff:", vector_angle(z1, z3_final) - angle)
        print("z2-z3 angle_diff:", vector_angle(z2_final, z3_final) - angle)
        print("z1, z2, z3 lengths:", np.linalg.norm(z1), np.linalg.norm(z2_final), np.linalg.norm(z3_final))
        return z2_final, z3_final

    def _get_angle_of_highest_fold(self):
        """The angle between the subunits in the highest order fold of the cubic symmetry of the assembly."""
        return int(360 / self.get_highest_fold())

    # fixme: rewrite this for
    def find_other_independent_highfolds(self, highest_fold, master_highfold, master_3_fold):
        """Finds high-folds that contain unique subunits"""
        attempts = 0
        rmsd_inital, angle_initial = self.start_rmsd_diff, self.start_angles_diff
        while attempts < self.total_increments:
            second_high_fold = self.find_X_folds(highest_fold, master_3_fold[1])[0]
            third_high_fold = self.find_X_folds(highest_fold, master_3_fold[2])[0]
            # detect that they are all unique
            all_subunits = master_highfold + second_high_fold + third_high_fold
            if len(all_subunits) == len(np.unique(all_subunits)):
                self.start_rmsd_diff, self.start_angles_diff = rmsd_inital, angle_initial
                return second_high_fold, third_high_fold
            else:
                self.start_rmsd_diff += self.rmsd_diff_increment
                self.start_angles_diff += self.angles_diff_increment
                attempts += 1
        rmsd_inital, angle_initial = self.start_rmsd_diff, self.start_angles_diff
        raise ValueError("Could not find independent high-folds next to the master high-fold")

    def predict_best_z(self, rosetta_input, setup, fudge_factor=1):

        # create rotation matrix that aligns everything in the x-plane
        a = setup.get_vrt("VRTHFfold").vrt_z
        b = setup.get_vrt("VRT3fold").vrt_z
        c = setup.get_vrt("VRT2fold").vrt_z
        facet_center = (a + b + c) / 3
        angle = vector_angle(a, facet_center)
        axis = np.cross(a, facet_center)
        rot = rotation_matrix(axis, angle)
        # now get the largest x-displacement
        xpos = []
        for atom in rosetta_input.copy().get_atoms(): # we modify the coordinates below so we have to copy
            atom.transform(rot, [0, 0, 0])
            x_coord = atom.get_coord()[0]
            xpos.append(x_coord)
        max_facet_displacement = abs(max(xpos) - min(xpos))
        return max_facet_displacement * fudge_factor
        # now lets imagine that the max_facet_displacemnet is the distance to any triangular corner to its triangular center
        # of an equilateral triangle.
        # from that we can calculate
        # now translate the back to the z-displacement


        # crossprodcut of the 3 5-fold vectors
        # triangular_axis = np.cross(a - b, a - c)
        # # now we need to rotate the crossproduct to the facet center
        # # fixme: have this as a function in mathsfuncs
        # angle = vector_angle(triangular_axis, facet_center)
        # axis = np.cross(triangular_axis, facet_center)
        # triangular_axis = rotate(triangular_axis, rotation_matrix(axis, angle))
        # triangular_axis






        # plane defined by 5 fold vectors
        # from shapedesign.source.movers.hypotenusemover import HypotenuseMover
        # from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
        # # Should apply on the asymmetric unit!
        # pose = Pose()
        # buf = ostringstream()
        # pose.dump_pdb(buf)
        # pose_from_pdbstring(pose, buf.str())
        #
        # fixed_asym = Pose()
        # rotating_asym = Pose()
        # extract_asymmetric_unit(fixed, fixed_asym, False)
        # extract_asymmetric_unit(rotating, rotating_asym, False)
        # max_x_disp = []
        # for pose, pose_asym in zip((fixed, rotating), (fixed_asym, rotating_asym)):
        #     # find the largest x axis displacement
        #     xpos = []
        #     hm = HypotenuseMover(pose, scorefxn=None)
        #     rotation_point, rot = hm.triangular_rotation_matrix()
        #     for resi in range(1, pose_asym.size() + 1):
        #         if pose.residue(resi).is_protein():
        #             if CA_only == True:
        #                 xyz = np.array(pose.residue(resi).xyz("CA"))
        #                 xyz_centered = xyz - rotation_point
        #                 rotated_xyz = rotate(xyz_centered, rot)
        #                 xpos.append(rotated_xyz[0])
        #             else:
        #                 raise NotImplementedError
        #     max_x_disp.append(abs(max(xpos) - min(xpos)))
        # # 0 is fixed, 1 is rotating
        # ratio = max_x_disp[1] / max_x_disp[0]
        # # hm should contain rotating now (since this was the last)
        # hm.mul_c(pose, ratio)

    def create_from_5fold(self, file, rosetta_input_out, symdef_out, outformat="cif"):
        #1 first symmetrize the 5fold
        ca = Assembly(pdb=file)
        ca.center() # important?
        #1a: rotate around the geometric center
        ca.align_to_z_axis()
        ca.align_subunit_to_x_axis(subunit="1")
        #2 create starting input
        rosetta_input = ca.create_centered_subunit("1")
        #3 symmetrize it. Probably should use som AF metrics to pick the best one for instance
        setup = SymmetrySetup()
        # FIXME: THIS DOES NOT HAVE DEFAULT SYMMETRY. THE Z ROTATION IS SET FROM 1STM
        #  either construct one from scratch or have a value for z_angle in the symmetry files
        #  I think the latter would be the best then the boundary box can always be set from the center and when
        #  dealing with existing z_angles then subtract those! To make the z_rot default, it would be nice to have all
        #  the 3-fold vrts pointing towards the 3-fold center.
        setup.read_from_file(str(Path(__file__).parent.joinpath("../data/5fold.symm")))
        setup.reset_all_dofs()
        setup.set_dof("JUMPHFfold1", "z", "translation", self.predict_best_z(rosetta_input, setup))
        # fixme: this has to be native to the symmetry file
        #######
        z1high, z2high, z3high = - setup.get_vrt("VRTHFfold1")._vrt_z, -setup.get_vrt("VRT2fold1")._vrt_z, -setup.get_vrt("VRT3fold1")._vrt_z
        master_id = "1"
        z1high_to_hf_center = vector((z1high + z2high + z3high) / 3, z1high)
        master_com_new_pos = ca.get_subunit_CA_closest_to_GC(master_id)
        z1high_to_ca_com_on_pose = vector(master_com_new_pos, z1high)
        z1high_norm = z1high / np.linalg.norm(z1high)
        z2high_norm = z2high / np.linalg.norm(z2high)
        z3high_norm = z3high / np.linalg.norm(z3high)
        z1high_onto_plane = vector_projection_on_subspace(z1high_to_ca_com_on_pose, z1high_norm - z2high_norm, z1high_norm - z3high_norm)
        z_angle = vector_angle(z1high_to_hf_center, z1high_onto_plane)  # skal bruge X-prodcut til at find ud af om - eller + angle
        #######
        x_dist = distance(ca.get_geometric_center("1"), ca.get_geometric_center())
        setup.set_dof("JUMPHFfold111", "x", "translation", x_dist)
        setup.set_dof("JUMPHFfold1", "z", "rotation", z_angle)
        self.output_subunit_as_specified(rosetta_input_out, rosetta_input, format=outformat)
        setup.output(symdef_out)

    def create_from_4fold(self, file, rosetta_input_out, symdef_out, outformat="cif"):
        #1 first symmetrize the 5fold
        ca = Assembly(pdb=file)
        ca.center() # important?
        #1a: rotate around the geometric center
        ca.align_to_z_axis()
        ca.align_subunit_to_x_axis(subunit="1")
        #2 create starting input
        rosetta_input = ca.create_centered_subunit("1")
        #3 symmetrize it. Probably should use som AF metrics to pick the best one for instance
        setup = SymmetrySetup()
        # FIXME: THIS DOES NOT HAVE DEFAULT SYMMETRY. THE Z ROTATION IS SET FROM 1STM
        #  either construct one from scratch or have a value for z_angle in the symmetry files
        #  I think the latter would be the best then the boundary box can always be set from the center and when
        #  dealing with existing z_angles then subtract those! To make the z_rot default, it would be nice to have all
        #  the 3-fold vrts pointing towards the 3-fold center.
        setup.read_from_file(str(Path(__file__).parent.joinpath("../data/4fold.symm")))
        setup.reset_all_dofs()
        setup.set_dof("JUMPHFfold1", "z", "translation", self.predict_best_z(rosetta_input, setup))
        # fixme: this has to be native to the symmetry file
        #######
        z1high, z2high, z3high = - setup.get_vrt("VRTHFfold1")._vrt_z, -setup.get_vrt("VRT2fold1")._vrt_z, -setup.get_vrt("VRT3fold1")._vrt_z
        master_id = "1"
        z1high_to_hf_center = vector((z1high + z2high + z3high) / 3, z1high)
        master_com_new_pos = ca.get_subunit_CA_closest_to_GC(master_id)
        z1high_to_ca_com_on_pose = vector(master_com_new_pos, z1high)
        z1high_norm = z1high / np.linalg.norm(z1high)
        z2high_norm = z2high / np.linalg.norm(z2high)
        z3high_norm = z3high / np.linalg.norm(z3high)
        z1high_onto_plane = vector_projection_on_subspace(z1high_to_ca_com_on_pose, z1high_norm - z2high_norm, z1high_norm - z3high_norm)
        z_angle = vector_angle(z1high_to_hf_center, z1high_onto_plane)  # skal bruge X-prodcut til at find ud af om - eller + angle
        #######
        x_dist = distance(ca.get_geometric_center("1"), ca.get_geometric_center())
        setup.set_dof("JUMPHFfold111", "x", "translation", x_dist)
        setup.set_dof("JUMPHFfold1", "z", "rotation", z_angle)
        self.output_subunit_as_specified(rosetta_input_out, rosetta_input, format=outformat)
        setup.output(symdef_out)

    def create_from_3fold(self, file, rosetta_input_out, symdef_out, outformat="cif"):
        # 1 first symmetrize the 5fold
        ca = Assembly(pdb=file)
        ca.center()  # important?
        # 1a: rotate around the geometric center
        ca.align_to_z_axis()
        ca.align_subunit_to_x_axis(subunit="1")
        # 2 create starting input
        rosetta_input = ca.create_centered_subunit("1")
        # 3 symmetrize it. Probably should use som AF metrics to pick the best one for instance
        setup = SymmetrySetup()
        # FIXME: THIS DOES NOT HAVE DEFAULT SYMMETRY. THE Z ROTATION IS SET FROM 1STM
        #  either construct one from scratch or have a value for z_angle in the symmetry files
        #  I think the latter would be the best then the boundary box can always be set from the center and when
        #  dealing with existing z_angles then subtract those! To make the z_rot default, it would be nice to have all
        #  the 3-fold vrts pointing towards the 3-fold center.
        setup.read_from_file(str(Path(__file__).parent.joinpath("../data/3fold.symm")))
        setup.reset_all_dofs()
        setup.set_dof("JUMPHFfold1", "z", "translation", self.predict_best_z(rosetta_input, setup, fudge_factor=0.7))
        # fixme: this has to be native to the symmetry file
        #######
        z1high, z2high, z3high = - setup.get_vrt("VRTHFfold1")._vrt_z, -setup.get_vrt(
            "VRT2fold1")._vrt_z, -setup.get_vrt("VRT3fold1")._vrt_z
        master_id = "1"
        z1high_to_hf_center = vector((z1high + z2high + z3high) / 3, z1high)
        master_com_new_pos = ca.get_subunit_CA_closest_to_GC(master_id)
        z1high_to_ca_com_on_pose = vector(master_com_new_pos, z1high)
        z1high_norm = z1high / np.linalg.norm(z1high)
        z2high_norm = z2high / np.linalg.norm(z2high)
        z3high_norm = z3high / np.linalg.norm(z3high)
        z1high_onto_plane = vector_projection_on_subspace(z1high_to_ca_com_on_pose, z1high_norm - z2high_norm,
                                                          z1high_norm - z3high_norm)
        z_angle = vector_angle(z1high_to_hf_center, z1high_onto_plane)  # skal bruge X-prodcut til at find ud af om - eller + angle
        #######
        x_dist = distance(ca.get_geometric_center("1"), ca.get_geometric_center())
        setup.set_dof("JUMPHFfold111", "x", "translation", x_dist)
        setup.set_dof("JUMPHFfold1", "z", "rotation", z_angle)
        self.output_subunit_as_specified(rosetta_input_out, rosetta_input, format=outformat)
        setup.output(symdef_out)

    def _attach_subunits_to_foldmap(self, foldmap):
        """Attaches the subunit objects to the foldmap instead of their str ids."""
        foldmap_subunits = {}
        for key, val in foldmap.items():
            foldmap_subunits[key] = self.get_subunits(val)
        return foldmap_subunits

    def add_along_vector(self, origo, vector, mul=1, dir=-1):
        return origo + (dir * vector / np.linalg.norm(vector)) * mul

    # def add_1_along_z(self, origo, z):
    #     return origo + (-1 * z / np.linalg.norm(z))

    def setup_symmetry(self, symmetry_name, master_id="1", subunits_in_other_highfold=2, idealize=True, foldmap:dict=None):
        """Sets up a SymmetrySetup object for either Icosahedral (I), Octahedral (O) or Tetrahedral (T) symmetry.

        The symmetrical setup consists of the 1 master subunit and its interacting subunits from the 3 closest highest possible folds.
        'high' is a substitute for the highest symmetrical fold here and in the code. So the following is given: I: high=5, O: high=4 and T: high=3.
        For I symmetry, for instance, the symmetrical setup will consist of the 3 closets 5-folds. The master subunit will be a part of 1 of
        these 5-folds and all chains present in the 5-fold will be present in the symmetrical setup. A variable number of subunits will
        be present from the 2 other 5-folds which can be set by the 'subunits_in_other_highfold' variable which by default is 2.

        :param symmetry_name: Name given to the SymmetrySeutp object
        :param master_id: subunit id to use as the master subunit
        :param subunits_in_other_highfold: Subunits to use in the other high-folds
        :param idealize: To idealize the geometry of the setup in order to obtain (numerically) perfect symmetry. If the asymmetric
        unit of the crystal structure is not perfectly symmetric, the resulting symmetry setup will not perfectly symmetric either. When
        idealize is on it will change the symmetry to be perfect at the expense of the resulting output not looking exactly like the
        biologial assembly.
        :return: SymmetrySetup, Bio.PDB.Structure.Structure object corresponding to the chosen master subunit.
        """
        setup = CubicSetup(symmetry_name=symmetry_name)
        self.center()
        highest_fold = self.get_highest_fold()
        highest_angle = self._get_angle_of_highest_fold()
        # todo: put this earlier and assert that the foldmap makes sense in a function
        if foldmap:
            foldmap_s = self._attach_subunits_to_foldmap(foldmap)
        if foldmap.get("hf1", False):
            master_high_fold = foldmap.get("hf1")
        else:
            master_high_fold = self.find_X_folds(highest_fold, id=master_id)[0]
        if foldmap.get("3", False):
            master_3_fold = foldmap.get("3")
        else:
            if highest_fold == 3:
                master_3_fold = self.find_X_folds(3, id=master_id, closest=2)[1]
            else:
                master_3_fold = self.find_X_folds(3, id=master_id, )[0]
        # the 2 other highest-folds that are part of the 3-fold
        if foldmap.get("hf2", False) and foldmap.get("hf3", False):
            second_high_fold, third_high_fold = foldmap.get("hf2"), foldmap.get("hf3")
        else:
            second_high_fold, third_high_fold = self.find_other_independent_highfolds(highest_fold, master_high_fold, master_3_fold)
        # get the 2 closest 2-folds that are part of the other second or third highest fold
        # closest=10 is arbitary, but we want to make sure we can go through enough
        if foldmap.get("21", False) and foldmap.get("22", False):
            master_2_folds = [foldmap_s.get("21"), foldmap_s.get("22")]
            self._set_alignment_geometric_center(ids=[s.id for ss in master_2_folds for s in ss])
            master_2_folds.sort(key=lambda ss: shortest_path(*[s.xtra["alignment_geometric_center"] for s in ss]))
            master_2_folds = [[s.id for s in ss] for ss in master_2_folds]
        else:
            master_2_folds = self.find_X_folds(2, id=master_id, closest=10, must_contain=second_high_fold + third_high_fold, minimum_folds=2)
        # get the coordinates for the global center
        global_x, global_y, global_z, global_center = self._get_global_center()
        # Add a global coordinate frame to the setup
        global_coordinate_frame = CoordinateFrame("VRTglobal", global_x, global_y, global_z, global_center)
        setup.add_vrt(global_coordinate_frame)
        # Rotate the assembly so that it is aligned with the z-axis
        current_z_axis = vector(self.get_geometric_center(master_high_fold), global_center)
        self.rotate_about_axis(np.cross(global_z, current_z_axis), vector_angle(global_z, current_z_axis))
        # Rotate the assembly so that the geometric center (Rosetta style: so it is the CA closest to the real geometric center)
        # of the master subunit is aligned to the global x-axis. Side note: Rosetta uses COM with the geometric center synonymously.
        master_com = self.get_subunit_CA_closest_to_GC(master_id)
        z1high = vector_projection(master_com, global_z)
        x_vector = vector(master_com, z1high)
        x_rotation = vector_angle(x_vector, global_x)
        self.rotate_about_axis(np.cross(global_x, x_vector), x_rotation)
        # Because of Rosettas convention with coordinate frames being the other way around (z = -z and x = -x) we have to make
        # a coordinate frame that is reversed in the z and x axis.
        reversed_global_z = - np.array([0, 0, 1])
        reversed_global_x = - np.array([1, 0, 0])

        # Now we add all the vrts and connect them through VRTglobal
        # ---- Main highest fold ----
        # First make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # for highest fold == 5: VRTglobal->VRT5fold->VRT5fold1 -> this then connects to all of the subunits
        name = f"VRTHFfold"
        jump = f"JUMPHFfold"
        si = ("1_z_tref", "", "1", "1_z_rref", "1_z", "1_x_tref", "", "1", "1_x_rref", "1_x", "1_y_rref", "1_y", "1_z_rref", "1_z")
        for i in range(5): #
            x, y, z, origo = reversed_global_x, global_y, reversed_global_z, global_center
            if "1_z_tref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z, dir=1)))
            elif "1_z_rref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z)))
            else:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, origo))
            if si[i] == "1_z_tref":
                setup.add_jump(jump + si[i], "VRTglobal", name + si[i])
            else:
                setup.add_jump(jump + si[i], name + si[i-1], name + si[i])
        to_go_through = [highest_angle * k for k in range(0, highest_fold)]
        for j, angle in enumerate(to_go_through, 1):
            rt = rotation_matrix(z1high, angle)
            sj = f"1{j}"
            for i in range(5, len(si)):
                suffix = sj + si[i]
                suffixp = si[i - 1] if i == 5 else sj + si[i - 1]
                x, y, z, origo = rotate_right_multiply(reversed_global_x, rt), rotate_right_multiply(global_y, rt), rotate_right_multiply(reversed_global_z, rt), global_center
                if "1_x_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x)))
                elif "1_y_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, y)))
                elif "1_z_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, z)))
                elif "1_x_tref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x, dir=1)))
                else:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, origo))
                setup.add_jump(jump + suffix, name + suffixp, name + suffix)
            # add an sds jump and and vrt just before the SUBUNIT
            sds_name = name + f"{sj}1" + "_sds"
            setup.add_vrt(CoordinateFrame(sds_name, x, y, z, origo))  # gets the last
            setup.add_jump(jump + f"{sj}1" + "_sds", name + suffix, sds_name)
            # add jump
            setup.add_jump(jump + f"{sj}1" + "_subunit", sds_name, "SUBUNIT")
        # ---------------------

        closest_2fold_id, furthest_2fold_id = self.get_closest_and_furthest_2_fold(master_id, master_2_folds)

        # Now make sure that the 2-fold that is closest and the 3-fold are coming from opposite highest-folds
        # Projected vector to the highest-fold geometric center was calculated above (z1high). Now calculate the non-projected ones to
        # the two highest-folds next to the master one
        # the z2high contains the closest 2-fold, and z3high the 3-fold.
        if closest_2fold_id in second_high_fold:
            z2high = vector(self.get_geometric_center(second_high_fold), global_center)
            z3high = vector(self.get_geometric_center(third_high_fold), global_center)
        else:
            z2high = vector(self.get_geometric_center(third_high_fold), global_center)
            z3high = vector(self.get_geometric_center(second_high_fold), global_center)
        # We normalize them to be the same length as z1high. z1high is based on the projection of the master subunit geometric center
        # into the global z-axis. So there ought to be some difference between z2high/z3high to z1high since these are based on
        # the geometric center of the whole high-fold. What is important is that they point in the correct orientation since the actual
        # length in the final rosetta symmetry file is set by the norm of z1high. But for the sake of understanding and also to make sure
        # we actually have perfect symmetry (if idealize is on) we need to check that these are the same length, hence the below normalization.
        z2high *= np.linalg.norm(z1high) / np.linalg.norm(z2high)
        z3high *= np.linalg.norm(z1high) / np.linalg.norm(z3high)

        # Mark if the symmetry is perfect or not and idealize (make it perfect) if idealize == True
        self.intrinsic_perfect_symmetry = self.has_perfect_symmetry(z1high, z2high, z3high)
        self.idealized_symmetry = idealize
        if idealize:
            z2high, z3high = self.idealize_vectors(z1high, z2high, z3high)

        # change the chain labels to match the Rosetta output
        self.remap_chain_labels(master_high_fold, master_3_fold, closest_2fold_id, furthest_2fold_id, second_high_fold, third_high_fold)

        # ---- Main 3-fold's high-fold ----
        axis = z3high
        # we have the axis, now we need to rotate the the coordinate system from the global axis to the one relative to the 3-fold 5-fold
        rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
        # first make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # VRTglobal->VRT3fold->VRT3fold1 -> this then connects to all of the subunits
        suffix = ""
        name = "VRT3fold"
        jump = "JUMP3fold"
        for i in range(5):
            x, y, z = rotate_right_multiply(reversed_global_x, rt_from_global_z), rotate_right_multiply(global_y, rt_from_global_z), rotate_right_multiply(reversed_global_z, rt_from_global_z)
            origo = global_center
            if "1_z_tref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z, dir=1)))
            elif "1_z_rref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z)))
            else:
                setup.add_vrt(CoordinateFrame(name + si[i], x,y, z, origo))
            if si[i] == "1_z_tref":
                setup.add_jump(jump + si[i], "VRTglobal", name + si[i]) # VRTglobal->VRT3fold
            else:
                setup.add_jump(jump + si[i], name + si[i-1], name + si[i]) # VRT3fold->VRT3fold1
        start = 180 - highest_angle
        to_go_through = [start + highest_angle * k for k in range(0, subunits_in_other_highfold)]
        for j, angle in enumerate(to_go_through, 1):
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the opposite direction. This is why we check for the y coordinate in the condition below. See for instance 1stm vs 4bcu.
            if z3high[1] < 0:
                angle *= -1
            rt_around_axis = rotation_matrix(axis, angle)  # check angle / positive or negative
            rt = np.dot(rt_from_global_z, rt_around_axis)  # check that this is okay
            sj = f"1{j}"
            for i in range(5, len(si)):
                suffix = sj + si[i]
                suffixp = si[i - 1] if i == 5 else sj + si[i - 1]
                x, y, z, origo = rotate_right_multiply(reversed_global_x, rt), rotate_right_multiply(global_y, rt), rotate_right_multiply(reversed_global_z, rt), global_center
                if "1_x_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x)))
                elif "1_y_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, y)))
                elif "1_z_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, z)))
                elif "1_x_tref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x, dir=1)))
                else:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, origo))
                setup.add_jump(jump + suffix, name + suffixp, name + suffix)
            # add an sds jump and and vrt just before the SUBUNIT
            sds_name = name + f"{sj}1" + "_sds"
            setup.add_vrt(CoordinateFrame(sds_name, x, y, z, origo))  # gets the last
            setup.add_jump(jump + f"{sj}1" + "_sds", name + suffix, sds_name)
            # add the subunit jump
            setup.add_jump(jump + f"{sj}1" + "_subunit", sds_name, "SUBUNIT")
        # ------------------------------

        # ---- Main 2-fold's 5-fold ----
        axis = z2high
        # rotate the the coordinate system from the global axis to the one relative to the 2-fold 5-fold
        rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
        # first make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # VRTglobal->VRT2fold->VRT2fold1 -> this then connects to all of the subunits
        suffix = ""
        name = "VRT2fold"
        jump = "JUMP2fold"
        for i in range(5):
            x, y, z = rotate_right_multiply(reversed_global_x, rt_from_global_z), rotate_right_multiply(global_y, rt_from_global_z), rotate_right_multiply(reversed_global_z, rt_from_global_z)
            origo = global_center
            if "1_z_tref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z, dir=1)))
            elif "1_z_rref" == si[i]:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, self.add_along_vector(origo, z)))
            else:
                setup.add_vrt(CoordinateFrame(name + si[i], x, y, z, origo))
            if si[i] == "1_z_tref":
                setup.add_jump(jump + si[i], "VRTglobal", name + si[i]) # VRTglobal->VRT2fold
            else:
                setup.add_jump(jump + si[i], name + si[i-1], name + si[i]) # VRT2fold->VRT2fold1
        start = 180
        to_go_through = [start + highest_angle * k for k in range(0, subunits_in_other_highfold)]
        for j, angle in enumerate(to_go_through, 1):
            rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
            if z3high[1] < 0:
                angle *= -1
            rt_around_axis = rotation_matrix(axis, angle)  # check angle / positive or negative
            rt = np.dot(rt_from_global_z, rt_around_axis)  # check that this is okay
            sj = f"1{j}"
            for i in range(5, len(si)):
                suffix = sj + si[i]
                suffixp = si[i - 1] if i == 5 else sj + si[i - 1]
                x, y, z, origo = rotate_right_multiply(reversed_global_x, rt), rotate_right_multiply(global_y, rt), rotate_right_multiply(reversed_global_z, rt), global_center
                if "1_x_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x)))
                elif "1_y_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, y)))
                elif "1_z_rref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, z)))
                elif "1_x_tref" == si[i]:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, self.add_along_vector(origo, x, dir=1)))
                else:
                    setup.add_vrt(CoordinateFrame(name + suffix, x, y, z, origo))
                setup.add_jump(jump + suffix, name + suffixp, name + suffix)
            # add an sds jump and and vrt just before the SUBUNIT
            sds_name = name + f"{sj}1" + "_sds"
            setup.add_vrt(CoordinateFrame(sds_name, x, y, z, origo)) # gets the last
            setup.add_jump(jump +  f"{sj}1" + "_sds", name + suffix, sds_name)
            # add the subunit jump
            setup.add_jump(jump +  f"{sj}1" + "_subunit", sds_name, "SUBUNIT")
        # ------------------------------

        # The 6 degrees of freedom
        setup.add_dof(f"JUMPHFfold1", 'z', "translation", np.linalg.norm(z1high))
        setup.add_dof(f"JUMPHFfold1_z", 'z', "rotation", 0)
        setup.add_dof(f"JUMPHFfold111", 'x', "translation", np.linalg.norm(vector(master_com, z1high)))
        setup.add_dof(f"JUMPHFfold111_x", 'x', "rotation", 0)
        setup.add_dof(f"JUMPHFfold111_y", 'y', "rotation", 0)
        setup.add_dof(f"JUMPHFfold111_z", 'z', "rotation", 0)
        # setup.add_dof(f"JUMPHFfold111_subunit", 'x', "rotation", 0)
        # setup.add_dof(f"JUMPHFfold111_subunit", 'y', "rotation", 0)
        # setup.add_dof(f"JUMPHFfold111_subunit", 'z', "rotation", 0)
        setup.add_jumpgroup("JUMPGROUP1", f"JUMPHFfold1", "JUMP3fold1", "JUMP2fold1")
        setup.add_jumpgroup("JUMPGROUP2", f"JUMPHFfold1_z", "JUMP3fold1_z", "JUMP2fold1_z")

        setup.add_jumpgroup("JUMPGROUP3", *[f"JUMPHFfold1{i}1" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}1" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}1" for i in range(1, subunits_in_other_highfold + 1)])

        for n, d in zip(("4", "5", "6"), ("x", "y", "z")):
            setup.add_jumpgroup(f"JUMPGROUP{n}", *[f"JUMPHFfold1{i}1_{d}" for i in range(1, highest_fold + 1)],
                                *[f"JUMP3fold1{i}1_{d}" for i in range(1, subunits_in_other_highfold + 1)],
                                *[f"JUMP2fold1{i}1_{d}" for i in range(1, subunits_in_other_highfold + 1)])

        setup.add_jumpgroup("JUMPGROUP7", *[f"JUMPHFfold1{i}1_sds" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}1_sds" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}1_sds" for i in range(1, subunits_in_other_highfold + 1)])

        setup.add_jumpgroup("JUMPGROUP8", *[f"JUMPHFfold1{i}1_subunit" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}1_subunit" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}1_subunit" for i in range(1, subunits_in_other_highfold + 1)])

        # If T, skip adding bonus to the extra-3-fold subunit since it is the same as the other one
        if self.get_symmetry() == "T":
            setup.energies = "{}*VRTHFfold111_sds + " \
                             "{}*(VRTHFfold111_sds:VRTHFfold121_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT3fold111_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT2fold111_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT3fold121_sds)".format(*self.get_energies())
        else:
            setup.energies = "{}*VRTHFfold111_sds + " \
                             "{}*(VRTHFfold111_sds:VRTHFfold121_sds) + " \
                             "{}*(VRTHFfold111_sds:VRTHFfold131_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT3fold111_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT2fold111_sds) + " \
                             "{}*(VRTHFfold111_sds:VRT3fold121_sds)".format(*self.get_energies())

        # finally, in order to be z_angle(0) to be the same for all structures, we rotate the following vrts along their axis
        # "VRTHFfold", "VRT3fold", "VRT2fold".
        # but first we need to find the angle to rotate
        # center of hf center
        # a = setup.get_vrt_name("VRTHFfold").vrt_z
        # b = setup.get_vrt_name("VRT3fold").vrt_z
        # c = setup.get_vrt_name("VRT2fold").vrt_z
        # FINE -----
        # z1high_to_hf_center = vector((z1high+z2high+z3high) / 3, z1high)
        # master_com_new_pos = self.get_subunit_CA_closest_to_GC(master_id)
        # z1high_to_ca_com_on_pose = vector(master_com_new_pos, z1high)
        # z1high_norm = z1high / np.linalg.norm(z1high)
        # z2high_norm = z2high / np.linalg.norm(z2high)
        # z3high_norm = z3high / np.linalg.norm(z3high)
        # z1high_onto_plane = vector_projection_on_subspace(z1high_to_ca_com_on_pose, z1high_norm - z2high_norm, z1high_norm - z3high_norm)
        # angle = vector_angle(z1high_to_hf_center, z1high_onto_plane) # skal bruge X-prodcut til at find ud af om - eller + angle
        # # FINE -----
        # setup.set_dof("JUMPHFfold1", "z", "rotation", angle) # vender i -z retning, men går den anden vej
        # # pseudoatom z1high, pos=[ 0.        ,  0.        , 63.62856227]
        # # pseudoatom z2high, pos=[ 49.67583251, -27.77029643,  28.4555133 ]
        # # pseudoatom z3high, pos=[41.76177361, 38.66302065, 28.45556247]
        # # pseudoatom z1high_to_ca_com_on_pose, pos=[ 3.34401906, 28.07083634,  0.        ]
        # # pseudoatom master_com, pos=[ 3.34401906, 28.07083634, 63.62856227]
        # # pseudoatom z1high_onto_plane, pos=[ 4.62113777, 14.38674186, -4.80552251]
        # # for vrtname in ("VRTHFfold", "VRT3fold", "VRT2fold"):
        # for vrt in setup._vrts:
        #     if vrt.name == "VRTglobal":
        #         axis = z1high
        #     if "VRTHF" in vrt.name:
        #         axis = z1high
        #     elif "VRT2" in vrt.name:
        #         axis = z2high
        #     elif "VRT3" in vrt.name:
        #         axis = z3high
        #     else:
        #         raise Exception
        #     vrt.rotate(rotation_matrix(axis, -angle))
        # todo: there seems be a difference between rosetta COM and my COM so in the case of 6u20 for instance - you'll not get the correct symmetry if recenter is not turned of
        # anchor residues
        setup.anchor = "COM"
        # # checks that is should be recentered
        # setup.recenter = True
        adjust = vector(global_center, self.get_subunit_CA_closest_to_GC(master_id))
        master = self.get_subunit_with_id(master_id).copy()
        master.transform(np.identity(3), adjust)
        # self.rotate_about_axis(global_z, - angle)

        selected_ids = self._get_selected_ids(master_high_fold, master_3_fold, furthest_2fold_id, closest_2fold_id,  second_high_fold, third_high_fold,
                                z3high, z2high, z1high)

        return setup, master, selected_ids

    def _get_selected_ids(self, master_high_fold, master_3_fold, furthest_2fold_id, closest_2fold_id,  second_high_fold, third_high_fold,
                          z3high, z2high, z1high):
        """Returns the ids that are selected from the assembly. Returns them in such an order that the chain numbers matches exactly in
        Rosetta symmetry file."""
        if closest_2fold_id in second_high_fold:
            closest_3fold_id = [i for i in second_high_fold if i in master_3_fold]
            furtest_3fold_id = [i for i in third_high_fold if i in master_3_fold]
        else:
            closest_3fold_id = [i for i in third_high_fold if i in master_3_fold]
            furtest_3fold_id = [i for i in second_high_fold if i in master_3_fold]
        return master_high_fold + furtest_3fold_id + [furthest_2fold_id] + [closest_2fold_id] + closest_3fold_id

    def get_angle_to_I_facet(self, z1high, z2high, z3high, master_id):
        z1high_to_hf_center = vector((z1high + z2high + z3high) / 3, z1high)
        master_com_new_pos = self.get_subunit_CA_closest_to_GC(master_id)
        z1high_to_ca_com_on_pose = vector(master_com_new_pos, z1high)
        z1high_norm = z1high / np.linalg.norm(z1high)
        z2high_norm = z2high / np.linalg.norm(z2high)
        z3high_norm = z3high / np.linalg.norm(z3high)
        z1high_onto_plane = vector_projection_on_subspace(z1high_to_ca_com_on_pose, z1high_norm - z2high_norm, z1high_norm - z3high_norm)
        return vector_angle(z1high_to_hf_center, z1high_onto_plane)  # skal bruge X-prodcut til at find ud af om - eller + angle

    def get_energies(self):
        # X times the subunit itself
        # Y1 times the highest fold itself
        # Y2 times the highest fold itself (for I + O symmetries)
        # Z times the 3-fold
        # W1 times the 2-fold (closest)
        # W2 times the 2-fold (furthest)
        if self.get_symmetry() == "I":
            return ("60", "60", "60", "60", "30", "30")
        elif self.get_symmetry() == "O":
            return ("24", "24", "24", "24", "12", "12")
        else: # self.get_symmetry() == "T":
            return ("12", "12", "12", "6", "6")

    def _find_X_folds_subroutine(self, func, id_, closest, must_contain, minimum_folds):
        """Runs a specific find_x_fold_subunits function for a number of attempts with varying parameters of the angles and RMSD."""
        attempts = 0
        rmsd_max, angle_max = self.start_rmsd_diff, self.start_angles_diff
        while attempts < self.total_increments:
            try: # try to detect I, O or T symmetry
                subunits = func(rmsd_max, angle_max, id_, closest, must_contain, minimum_folds)
            except ValueToHigh:
                pass
            else:
                if subunits:
                    return subunits
            rmsd_max += self.rmsd_diff_increment
            angle_max += self.angles_diff_increment
            attempts += 1
        return []

    # todo: The underlying functions can be written generally in the future with a recursive functions.
    def find_X_folds(self, x, id="1", closest=1, must_contain:list=None, minimum_folds=None):
        """Returns the closet x-fold subunits to a subunit with the id label.
        :param x: The fold (2-5 are supported!).
        :param id: The id of the subunit to find the fold of.
        :param closest: return the number of closest subunits folds to that subunit.
        :return: subunits of 1 ore more folds closest to the subunit.
        """
        assert(x > 1 and x < 6), "only 2-5 folds are supported"
        if x == 2:
            return self._find_X_folds_subroutine(self._find_2_fold_subunits, id, closest, must_contain, minimum_folds)
        if x == 3:
            return self._find_X_folds_subroutine(self._find_3_fold_subunits, id, closest, must_contain, minimum_folds)
        if x == 4:
            return self._find_X_folds_subroutine(self._find_4_fold_subunits, id, closest, must_contain, minimum_folds)
        if x == 5:
            return self._find_X_folds_subroutine(self._find_5_fold_subunits, id, closest, must_contain, minimum_folds)

    def _detect_cubic_symmetry(self):
        """Detects for the following symmetry: Icosahedral (I) or Octahedral (O) or Tetrahedral (T).
        :returns str type of either 'I', 'O' or 'T' if symmetry is found else None."""
        folds, stypes, sizes = (5, 4, 3), ("I", "O", "T"), (60, 24, 12)
        for fold, stype, size in zip(folds, stypes, sizes):
            # detect if the size if correct for the symmetry
            if len(self) == size:
                # detect if we can find a symmetry fold intrinsic to the symmetry
                xfolds = self.find_X_folds(fold)
                if xfolds and len(xfolds) > 0:
                    print(f"{stype} symmetry detected")
                    return stype
        print(f"No symmetry dectected")

    def get_adequate_subunits(self, subunits, must_contain=None, minimum_folds=None):
        """return subunits that if set have ids in must_contain and if set in total consists of atleast minimum_folds."""
        if must_contain:
            subunits = [ss for ss in subunits if any(s in must_contain for s in ss)]
        if minimum_folds and not len(subunits) >= minimum_folds:
            return []
        else:
            return subunits

    def _find_2_fold_subunits(self, rmsd_max, angle_max, id="1", closest=1, must_contain=None, minimum_folds=None):
        """Finds the other subunits belonging to the 2 fold axis that the subunit with the id "name" is in.

         Algorithm:
         Checks that the 180 degrees rotation around the center of the 2 subunits overlaps them. If so you have
         a two_fold!

        TODO:
        :param rmsd_max:
        :param angle_max:
        :param id:
        :param closest:
        :return:
        """

        # set the alginment geometric center
        self._set_alignment_geometric_center()

        # find subunits to search
        subunits_to_search = [int(subunit.id) for subunit in self.get_models() if subunit.id != id]

        # container for the x folds
        twofolds = []

        # special case if x = 2:
        if not "center" in self.xtra:
            self._set_center()
        subunit = self[id]
        for c in subunits_to_search:
            subunit_reference = self[f"{c}"]
            center_between_subunits = ((subunit_reference.xtra["alignment_geometric_center"] - subunit.xtra["alignment_geometric_center"]) / 2) + subunit.xtra["alignment_geometric_center"]
            center_to_center_between_subunits = center_between_subunits - self.xtra["center"]
            rt = rotation_matrix(center_to_center_between_subunits, 180)

            # extract the largest overlapping sequence
            subunit_seq = [res.get_resname() for res in subunit.get_residues()]
            subunit_reference_seq = [res.get_resname() for res in subunit_reference.get_residues()]
            sm = difflib.SequenceMatcher(None, subunit_seq, subunit_reference_seq)
            pos_a, pos_b, size = sm.find_longest_match(0, len(subunit_seq), 0, len(subunit_reference_seq))
            subunit_res = [res for res in list(subunit.get_residues())[pos_a:pos_a+size]]
            subunit_reference_res = [res for res in list(subunit_reference.get_residues())[pos_b:pos_b+size]]
            # do a transform and check if they overlap
            n = 0
            distances = np.zeros(3, dtype=np.float64)
            for res, res_reference in zip(subunit_res, subunit_reference_res):
                for atom, atom_reference in zip(res, res_reference):
                    if atom.get_fullname() == "CA":
                        atom_reference = atom_reference.copy()
                        atom_reference.transform(rt, [0,0,0])
                        # if not criteria_check(0, distance(atom.get_coord(),atom_reference.get_coord()), diff=1):
                        distances += distance(atom.coord, atom_reference.coord)
                        n += 1
            rmsd = sum(distances) / n
            # largest RMSD value to be used is.
            if criteria_check(0, rmsd, rmsd_max):
                twofolds.append([subunit, subunit_reference])

        if len(twofolds) > 0:
            twofolds = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in twofolds})
            twofolds.sort(key=lambda subunits: shortest_path(*[subunit.xtra["alignment_geometric_center"] for subunit in subunits]))
            subunits = [[subunit.id for subunit in subunits] for subunits in twofolds[:closest]]
            return self.get_adequate_subunits(subunits, must_contain, minimum_folds)
        else:
            raise ToHighRMSD("None of the 2-folds found are valid 2-folds.")

    def angle_check(self, tobe, p1, p2, p3, angle_max):
        return criteria_check(tobe, angle(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"],
                                        p3.xtra["alignment_geometric_center"]), diff=angle_max)

    def distance_check(self, p1A, p2A, p1B, p2B, distance_max):
        return criteria_check(distance(p1A.xtra["alignment_geometric_center"], p2A.xtra["alignment_geometric_center"]),
                              distance(p1B.xtra["alignment_geometric_center"], p2B.xtra["alignment_geometric_center"]), diff=distance_max)

    def shortest_path_in_subunits(self, subunits, *positions):
        return shortest_path(*(subunits[i].xtra["alignment_geometric_center"] for i in positions))

    def _find_3_fold_subunits(self, rmsd_max, angle_max, id="1", closest=1, must_contain=None, minimum_folds=None):
        """Finds the other subunits belonging to the three fold axis that the subunit with the id "name" is in.

        Algorithm:
        Similar to the algorithm used to find the 5-fold symmetry units (see below). It instead use the degrees of
        a triangle instead of a pentamer.

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 3-fold symmetry subunits the subunit belong to.
        :return
        """
        self._set_alignment_geometric_center()
        # the subunit with the given id (retrieve p1)
        p1 = None
        for subunit in self.get_models():
            if subunit.id == id:
                p1 = subunit
                break
        if p1 == None:
            print(id, "does not exist!")
            return []
        # list of subunits excluding subunit with the id given
        # todo: should be fast if just references
        threefold = []
        subunits_to_search = [subunit for subunit in self.get_models() if subunit.id != id]
        # Find the other 2 protein_list.txt making up the three-fold symmetry axis
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p2 != p3:
                    # p1,p2,p3 angle should be equal to 108 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(60, p1, p2, p3, angle_max) and self.distance_check(p1, p2, p2, p3, angle_max):
                        threefold.append((p1, p2, p3))
        if len(threefold) != 0:
            # threefold = list({tuple(sorted(subunits,key=lambda subunit: subunit.id)) for subunits in threefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            threefold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(3)))
            threefolds = []
            for subunits in threefold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(3, rmsd_max, *subunits):
                        threefolds.append([subunit.id for subunit in subunits])
                    if len(threefolds) == closest:
                        return self.get_adequate_subunits(threefolds, must_contain, minimum_folds)
            if len(threefolds) > 0:
                return self.get_adequate_subunits(threefolds, must_contain, minimum_folds)
            else:
                raise ToHighRMSD("None of the 3-folds found are valid 3-folds.")
        else:
            raise ToHighGeometry("A 3-fold axis was not found")

    def _right_handed_vectors(self, v1, v2, axis):
        """Returns true if the point v1 going to v2 relative to the axis is right-handed. It is left-handed if the cross product v1 X v2
        points in the same direction as the axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        cross = np.cross(np.array(v1 - axis), np.array(v2 - axis))
        return vector_angle(cross, axis) > 90 # True -> It is right-handed

    def is_right_handed(self, subunits):
        """Calcualtes the handedness of the sunbunits order, and returns True if it right-handed and False otherwise"""
        all_gc = self.get_geometric_center([s.id for s in subunits])
        s0_gc = self.get_geometric_center(subunits[0].id)
        s1_gc = self.get_geometric_center(subunits[1].id)
        # if the cross product points opposite to the all_gc vector then the subunits are right-handed The way that I set up the
        # symmetric system the angle should be 180 for right-handed and 0 for left-handed (aka pointing in the same direction) but I allow for
        # some leeway (symmetrical inaccuracies + floating point number) so I just say < / > 90 inside the function
        return self._right_handed_vectors(s0_gc, s1_gc, all_gc)

    def _find_4_fold_subunits(self, rmsd_max, angle_max, id="1", closest=1, must_contain=None, minimum_folds=None):
        """Finds the other subunits belonging to the four fold axis that the subunit with the id "name" is in.

        Algorithm:
        Similar to the algorithm used to find the 5-fold symmetry units (see below). It instead use the degrees of
        a square instead of a pentamer.

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 4-fold symmetry subunits the subunit belong to.
        :return
        """
        self._set_alignment_geometric_center()
        # the subunit with the given id (retrieve p1)
        p1 = None
        for subunit in self.get_models():
            if subunit.id == id:
                p1 = subunit
                break
        if p1 == None:
            print(id, "does not exist!")
            return []
        # list of subunits excluding subunit with the id given
        subunits_to_search = [subunit for subunit in self.get_models() if
                              subunit.id != id]  # todo: should be fast if just references
        fourfold = []
        # Find the other 3 proteins making up the fourfold symmetry axis (p2, p3 and p4)
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p3.id != p2.id:
                    # p1,p2,p3 angle should be equal to 90 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(90, p1, p2, p3, angle_max) and self.distance_check(p1, p2, p2, p3, angle_max):
                        for p4 in subunits_to_search:
                            if p4 != p2 and p4 != p3:
                                # p1,p3,p4 angle should be equal to 45
                                # the distance between p2 and p3 should be the same as p3 and p4
                                if self.angle_check(45, p1, p3, p4, angle_max) and self.distance_check(p2, p3, p3, p4, angle_max):
                                    fourfold.append((p1, p2, p3, p4))

        if len(fourfold) != 0:
            # fourfold = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in fourfold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            fourfold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(4)))
            fourfolds = []
            for subunits in fourfold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(4, rmsd_max, *subunits):
                        fourfolds.append([subunit.id for subunit in subunits])
                    if len(fourfolds) == closest:
                        return self.get_adequate_subunits(fourfolds, must_contain, minimum_folds)
            if len(fourfolds) > 0:
                return self.get_adequate_subunits(fourfolds, must_contain, minimum_folds)
            else:
                raise ToHighRMSD("None of the 4-folds found are valid 4-folds.")
        else:
            raise ToHighGeometry("A 4-fold axis was not found.")

    def _find_5_fold_subunits(self, rmsd_max, angle_max, id="1", closest=1, must_contain=None,  minimum_folds=None):
        """Finds the other subunits belonging to the five fold axis that the subunit with the name "name" is in.

        Algorithm:
        It does so by first finding all the geometric centers of each subunit in the assembly and then sorting them
        according to the distance from the specified protein. Then it will find the closest two subunits having a total
        angle of 108 degrees between them. This corresponds to the first angle in a pentagon (5-fold-symmetry).
        Then it will find then one having 72 between the start subunit and the last subunit in the previous found subunit
        and the one having 36 degrees between the first and the next found subunit, corresponding to the other
        two angles in a pentagon. It exhaustively searches all found pentagons and determines the pentagon with the shortest
        path between the 5 points to be the fivefold axis that is returned.

        For future reference for a better approach: see https://www.sciencedirect.com/science/article/pii/S1047847718301126

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 5-fold symmetry subunits the subunit belong to.
        :param int closest: the closest 5-fold subunits to return
        :return

        """
        self._set_alignment_geometric_center()
        # the subunit with the given id (retrieve p1)
        p1 = None
        for subunit in self.get_models():
            if subunit.id == id:
                p1 = subunit
                break
        if p1 == None:
            print(id, "does not exist!")
            return []
        # list of subunits excluding subunit with the id given
        subunits_to_search = [subunit for subunit in self.get_models() if
                              subunit.id != id]  # todo: should be fast if just references
        fivefold = []
        # Find the other 4 protein_list.txt making up the five fold symmetry axis (p2, p3, p4 and p5)
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p3.id != p2.id:
                    # p1,p2,p3 angle should be equal to 108 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(108.0, p1, p2, p3, angle_max) and self.distance_check(p1, p2, p2, p3, angle_max):
                        for p4 in subunits_to_search:
                            if p4 != p2 and p4 != p3:
                                # p1,p3,p4 angle should be equal to 72
                                # the distance between p2 and p3 should be the same as p3 and p4
                                if self.angle_check(72, p1, p3, p4, angle_max) and self.distance_check(p2, p3, p3, p4, angle_max):
                                    for p5 in subunits_to_search:
                                        if p5 != p2 and p5 != p3 and p5 != p4:
                                            # p1,p4,p5 angle should be equal to 36
                                            # the distance between p3 and p4 should be the same as p4 and p5
                                            if self.angle_check(36, p1, p4, p5, angle_max) and self.distance_check(p3, p4, p4, p5, angle_max):
                                                fivefold.append((p1, p2, p3, p4, p5))
        if len(fivefold) != 0:
            # fivefold = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in fivefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            fivefold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(5)))
            fivefolds = []
            for subunits in fivefold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(5, rmsd_max, *subunits):
                        fivefolds.append([subunit.id for subunit in subunits])
                    if len(fivefolds) == closest:
                        return self.get_adequate_subunits(fivefolds, must_contain, minimum_folds)
            if len(fivefolds) > 0:
                return self.get_adequate_subunits(fivefolds, must_contain, minimum_folds)
            else:
                raise ToHighRMSD("None of the 5-folds found are valid 5-folds.")
        else:
            raise ToHighGeometry("A 5-fold axis was not found.")

    def is_correct_x_fold(self, fold, rmsd_max, *args):
        """Checks if the fold is a correct 3, 4 or 5-fold.

        It does this by superimposing 1 subunit on top of the other using 3, 4 or 5 fold rotations around the folds
        geometric center of mass. A RMSD criteria is checked to conclude if the superposition is good enough to
        validate it as a 3, 4 or 5 fold.
        :param rmsd_max:
        """
        accepted_rmsds = []
        # create rotation matrices for 3, 4 or 5 fold rotations
        center = self.get_geometric_center([x.id for x in args])
        if fold == 3:
            angles = [120, 120 * 2]
            # angles += [- angle for angle in angles]
        elif fold == 4:
            angles = [90, 90 * 2, 90*3]
        elif fold == 5:
            angles = [72, 72 * 2, 72 * 3, 72 * 4]
            # angles += [- angle for angle in angles]
        else:
            raise NotImplementedError("only works for a 3, 4 and 5 folds")
        rots = []
        for angle in angles:
            rots.append(rotation_matrix(center, angle))
        s1 = args[0]
        s1_residues = list(s1.get_residues())
        sxs = args[1:] # + args[1:]
        correct_fold = []
        # check if the subunits come on top of each other when using the rotation
        for rotation in rots:
            for sx in sxs:
                distances = 0
                n = 0
                # Use the alignment residues to iterate through here:
                sx_residues = list(sx.get_residues())
                for s1_pos, sx_pos in zip(s1.xtra["alignment"], sx.xtra["alignment"]):
                    s1_resi = s1_residues[s1_pos]
                    sx_resi = sx_residues[sx_pos]
                    assert sx_resi.get_resname() == s1_resi.get_resname(), "RMSD calculation cannot be between different amino acids!"
                    for atom, atom_reference in zip(s1_resi.get_atoms(), sx_resi.get_atoms()):
                        if atom.get_fullname() == "CA":
                            atom_new = atom_reference.copy()
                            atom_new.transform(rotation, [0, 0, 0])
                            # if not criteria_check(0, distance(atom.get_coord(),atom_reference.get_coord()), diff=1):
                            distances += distance(atom.coord, atom_new.coord)
                            n += 1
                # debug
                # print(f"pseudoatom atom, pos={list(atom.coord)}")
                # print(f"pseudoatom atom_reference, pos={list(atom_reference.coord)}")
                # print(f"pseudoatom atom_new, pos={list(atom_new.coord)}")
                # print(f"sx: {sx}")
                # print(f"rot: {rotation}")
                # print(f"rot: {angle}")
                # debug
                rmsd = math.sqrt(distances / n)
                if fold == 3:
                    if self.lowest_3fold_rmsd:
                        self.lowest_3fold_rmsd = min(self.lowest_3fold_rmsd, rmsd)
                    else:
                        self.lowest_3fold_rmsd = rmsd
                elif fold == 4:
                    if self.lowest_4fold_rmsd:
                        self.lowest_4fold_rmsd = min(self.lowest_4fold_rmsd, rmsd)
                    else:
                        self.lowest_4fold_rmsd = rmsd
                elif fold == 5:
                    if self.lowest_5fold_rmsd:
                        self.lowest_5fold_rmsd = min(self.lowest_5fold_rmsd, rmsd)
                    else:
                        self.lowest_5fold_rmsd = rmsd

                # largest RMSD value to be used is.
                if criteria_check(0, rmsd, diff=rmsd_max):
                    correct_fold.append(True)
                    accepted_rmsds.append(rmsd)
                else:
                    correct_fold.append(False)

            if fold == 3:
                if sum(correct_fold) >= 2:
                    if self.highest_3fold_accepted_rmsd:
                        self.highest_3fold_accepted_rmsd = max(self.highest_3fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_3fold_accepted_rmsd = max(accepted_rmsds)
                    return True
            if fold == 4:
                if sum(correct_fold) >= 3:
                    if self.highest_4fold_accepted_rmsd:
                        self.highest_4fold_accepted_rmsd = max(self.highest_4fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_4fold_accepted_rmsd = max(accepted_rmsds)
                    return True
            elif fold == 5:
                if sum(correct_fold) >= 4:
                    if self.highest_5fold_accepted_rmsd:
                        self.highest_5fold_accepted_rmsd = max(self.highest_5fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_5fold_accepted_rmsd = max(accepted_rmsds)
                    return True
        return False