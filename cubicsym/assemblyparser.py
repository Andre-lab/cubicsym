#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AssemblyParser class
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import Bio
import time
import xmlrpc.client as xmlrpclib
import numpy as np
from io import StringIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from string import ascii_lowercase, ascii_uppercase
from cubicsym.mathfunctions import rotation_matrix, vector_angle
from cubicsym.cubicassembly import CubicSymmetricAssembly
from cubicsym.assembly import Assembly
from cubicsym.exceptions import NoSymmetryDetected
from symmetryhandler.symmetryhandler import SymmetrySetup
# pyrosetta -> to use from_asymmetric_output
from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.pose.datacache import CacheableDataType
from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
from pyrosetta.rosetta.std import istringstream
from pyrosetta.rosetta.core.conformation.symmetry import SymmData
from pyrosetta.rosetta.std import ostringstream
from pyrosetta import init
from pathlib import Path

class AssemblyParser:
    """Parses different formats to an assembly."""
    def __init__(self):
        self.parser = MMCIFParser()

    def from_pose(cls, pose, name=""):
        return CubicSymmetricAssembly(name, )

    def create_symmetric_pose_from_asymmetric_output(self, file: str, return_symmetry_file=False):
        init("-initialize_rigid_body_dofs true -pdb_comments")
        pose = pose_from_file(file)
        symmetry_file = pose.data().get_ptr(CacheableDataType.STRING_MAP).map()["SYMMETRY"].replace("|", "\n")
        s = SymmData()
        s.read_symmetry_data_from_stream(istringstream(symmetry_file))
        setup = SetupForSymmetryMover(s)
        setup.apply(pose)
        if return_symmetry_file:
            return pose, symmetry_file
        else:
            return pose

    def rosetta_representation_from_asymmetric_output(self, file:str, symmetry_file=None):
        init("-initialize_rigid_body_dofs true -pdb_comments")
        pose = pose_from_file(file)
        if symmetry_file:
            setup = SetupForSymmetryMover(symmetry_file)
            setup.apply(pose)
            return pose
        else:
            return self.create_symmetric_pose_from_asymmetric_output(file)

    # TODO: can be independent on from_symmetric_output_and_symmetry_file and therefor faster as well.
    def capsid_from_asymmetric_output(self, file:str):
        """Creates a capsid from an asymmetric output from the shapedocking protocol."""
        pose, symmetry_file = self.create_symmetric_pose_from_asymmetric_output(file, return_symmetry_file=True)
        buffer = ostringstream()
        pose.dump_pdb(buffer)
        return self.from_symmetric_output_pdb_and_symmetry_file(StringIO(buffer.str()), StringIO(symmetry_file), assembly_name=file)

    def from_symmetric_output_pdb_and_symmetry_file(cls, file, symmetry_file, assembly_name=None):
        """
        The algorithm is as follows:
         1. Generate the 2 fold from the chain A
         2. generate 5-fold (five subunits) from master subunit
         3. generate 5-fold (five sununits) from the subunit that is part of the 2-fold axis with the master subunit.
            and rotate the latter five subunit 5-fold  72*5 degrees. Now we have half a capsid.
         4. rotate the half capsid 180 degrees at specific point along the middle.
        # todo: other than icosahedral structures in the future
        # todo: Have to rewrite for multichain systems.
        :param file:
        :param symmetry_file:
        :return:
        """


        # read symmetry from file
        setup = SymmetrySetup()
        setup.read_from_file(symmetry_file)

        # read rosetta pdb file from file
        fsuffix = Path(file).suffix
        assert fsuffix in (".cif", ".pdb"), f"File has to have either extension '.cif' of '.pdb' not {fsuffix}"
        if fsuffix == ".cif":
            p = MMCIFParser()
        else:
            p = PDBParser(PERMISSIVE=1)
        structure = p.get_structure(file, file)
        global_z = setup.get_vrt_name("VRTglobal")._vrt_z

        # Variable that will contain all chains of the assembly
        chains = []

        # chain name and subunit number specifiers
        chain_ids = list(ascii_uppercase) + list(ascii_lowercase) + [str(i) for i in range(0, 5000)]

        # this is to get the chains_ids to match the output from Rosetta. Only works for chain a 60-mer!:
        for letter1, letter2 in zip(["J", "F", "G", "d", "Z", "a"], ["I", "J", "K", "F", "G", "H"]):
            pos1 = chain_ids.index(letter1)
            pos2 = chain_ids.index(letter2)
            chain_ids[pos1] = letter2
            chain_ids[pos2] = letter1
        chain_ids = iter(chain_ids)

        # The 3 5-fold axes availble for an icosahedral structure in the symmetry file
        # minus because rosetta is awesome and have turned the coordinate systems arounD
        z15 = -setup.get_vrt_name("VRTHFfold")._vrt_z
        z25 = -setup.get_vrt_name("VRT2fold")._vrt_z
        z35 = -setup.get_vrt_name("VRT3fold")._vrt_z

        # construct assembly
        assembly = Assembly()

        # count the chains in the structure
        n_chains = len(list(structure.get_chains())) # number_of_chains(pdb)

        for chain_letter in list(ascii_uppercase)[0:n_chains // 9]:

            # for main in ["A"]
            # Construct the master structure
            master = structure[0][chain_letter]

            ### 1
            # copy the twofold and rotate it around its two-fold axis
            two_fold = master.copy()
            mid_z15_z25 = z15 + z25
            two_fold.transform(rotation_matrix(mid_z15_z25, 180), [0, 0, 0])

            ### 2
            # make 5 fold around master
            chains.append(master)
            for i in range(1,5):
                master_5fold = master.copy()
                master_5fold.transform(rotation_matrix(z15, 72 * i), [0, 0, 0])
                chains.append(master_5fold)

            ### 3
            # make the surrounding 5-folds
            surrounding_5_folds = []
            surrounding_5_folds.append(two_fold)
            # make the first five fold
            for i in range(1, 5):
                two_fold_5fold = two_fold.copy()
                two_fold_5fold.transform(rotation_matrix(z25, 72*i), [0, 0, 0])
                surrounding_5_folds.append(two_fold_5fold)
            # make the rest (4 of them)
            for i in range(1, 5):
                for j in range(5):
                    extra_5_fold = surrounding_5_folds[j].copy()
                    extra_5_fold.transform(rotation_matrix(z15, 72 * i), [0, 0, 0])
                    surrounding_5_folds.append(extra_5_fold)
            for chain in surrounding_5_folds:
                chains.append(chain)

            ### 4
            # make the rest of the capsid
            # before the loops we create vectors that are important for rotating half the capsid
            vec1 = z25 + ((z35 - z25) / 2.0)  # a vector from midlle of the 2-fold/5-fold axis to the 3-fold/5-fold axis
            vec2 = np.dot(z15, rotation_matrix(np.cross(vec1, z15), vector_angle(z15,vec1) * 2))  # points to another 5-fold axis just below the 3/2-fold/5-fold-axis
            vec3 = vec2 + ((z25 - vec2) / 2.0)  # points bewteen the 3-fold axis and the above 5-fold axis
            rotation_for_half_capsid = rotation_matrix(vec3, 180.0)
            for chain in chains[:]:
                new_chain = chain.copy()
                new_chain.transform(rotation_for_half_capsid, [0,0,0])
                chains.append(new_chain)

            # Add all chains to the assembly
            for n, chain in enumerate(chains, 1):
                subunit = Bio.PDB.Model.Model(str(n))
                chain.id = next(chain_ids)  # A
                subunit.add(chain)
                assembly.add(subunit)
        return assembly

    def from_cif_url(cls, file):
        pass

    def from_full_assembly(self, file):
        start_time = time.time()
        parser = MMCIFParser()
        structure_name = file.split("/")[-1].split(".")[0]
        structure = parser.get_structure(structure_name, file)
        mmcif_dict = MMCIF2Dict(file)

        if len(structure.get_list()) % 60 == 0:
            print("Structure is icosahedral")
            assembly = CubicSymmetricAssembly(structure_name + "_assembly", )
            # chains_in_subunit = len(unique_chain_names) // 60
            # print("Subunits consists of", chains_in_subunit)
            # for subunit_number, chains in enumerate([[chain for chain in list(structure.get_chains())[i:i+i*chains_in_subunit]] for i in range(0, 60*chains_in_subunit, chains_in_subunit)],1):
            #     subunit = Bio.PDB.Model.Model(f"s{subunit_number}")
            #     for chain in chains:
            #         subunit.add(chain)
            #     assembly.add(subunit)
            return assembly
        else:
            pass

    def _get_symmetry_operations(self, pick, mmcif_dict):
        """

        :param pick:
        :param mmcif_dict:
        :return:
        """

        # fetch symmetry operations numbers to generate the picked assembly
        symmetry_operations = mmcif_dict["_pdbx_struct_assembly_gen.oper_expression"][pick]
        if "(" in symmetry_operations:
            symmetry_operations = symmetry_operations[1:-1].split(",")
        else:
            symmetry_operations = symmetry_operations.split(",")
        for ele in symmetry_operations[:]:
            if ele.find("-") != -1:
                symmetry_operations.remove(ele)
                start = ele.split("-")[0]
                end = ele.split("-")[1]
                for value in range(int(start), int(end) + 1):
                    symmetry_operations.append(str(value))
        symmetry_operations.sort(key=int)

        # fetch the rotation and translation matrices/vectors of the symmetry operations
        rm_list = []
        tv_list = []
        for operation_number in symmetry_operations:
            index = mmcif_dict["_pdbx_struct_oper_list.id"].index(operation_number)
            rm = [[float(mmcif_dict["_pdbx_struct_oper_list.matrix[1][1]"][index]),
                   float(mmcif_dict["_pdbx_struct_oper_list.matrix[1][2]"][index]),
                   float(mmcif_dict["_pdbx_struct_oper_list.matrix[1][3]"][index])]]
            rm += [[float(mmcif_dict["_pdbx_struct_oper_list.matrix[2][1]"][index]),
                    float(mmcif_dict["_pdbx_struct_oper_list.matrix[2][2]"][index]),
                    float(mmcif_dict["_pdbx_struct_oper_list.matrix[2][3]"][index])]]
            rm += [[float(mmcif_dict["_pdbx_struct_oper_list.matrix[3][1]"][index]),
                    float(mmcif_dict["_pdbx_struct_oper_list.matrix[3][2]"][index]),
                    float(mmcif_dict["_pdbx_struct_oper_list.matrix[3][3]"][index])]]
            tv = [float(mmcif_dict["_pdbx_struct_oper_list.vector[1]"][index]),
                  float(mmcif_dict["_pdbx_struct_oper_list.vector[2]"][index]),
                  float(mmcif_dict["_pdbx_struct_oper_list.vector[3]"][index])]
            rm_list.append(rm)
            tv_list.append(tv)

        return rm_list, tv_list

    def _reconstruct(self, mmcif_dict, model, canonical=True):
        """Reconstructs the model.
        Modifies:
         - Chain names so that the label_asym_id are used instead of the author ones
         - Removes all non-protein AA (option available to only pick canonical).
         - Renumbers the author_seq_id

        :param mmcif_dict: mmcif dicitionary from the cif file
        :param model:
        :param canonical: To only choose canonical or not. fx. FML of GPL will be allowed if canonical=False
        :return:
        """

        # now we have to deconstruct the model and change the chains into something we want
        atom_name = mmcif_dict["_atom_site.type_symbol"]
        atom_residue_name =mmcif_dict["_atom_site.label_comp_id"]
        # if use_author_chains:
        #     atom_chain_name = mmcif_dict["_atom_site.auth_asym_id"]
        # else:
        atom_chain_name = mmcif_dict["_atom_site.label_asym_id"]
        # ---the following part makes sure we renumber the chain!
        atom_residue_seq_id = mmcif_dict["_atom_site.label_seq_id"]
        old_id = 0
        new_id = 0
        new_atom_residue_seq_id = []
        for id in atom_residue_seq_id:
            try:
                # detects new chain
                if int(id) < int(old_id):
                    new_id = 0
                # detects that the id has changed
                if id != old_id:
                    new_id += 1
                old_id = id
                new_atom_residue_seq_id.append(new_id)
                # detects if id is a number - then just put it in
            except:
                new_atom_residue_seq_id.append(id)
        # ---
        # atom_residue_seq_id = [res for res in range(1, len(set(setmmcif_dict["_atom_site.label_seq_id"])) + 1)]
        atom_coords = [[float(mmcif_dict["_atom_site.Cartn_x"][i]), float(mmcif_dict["_atom_site.Cartn_y"][i]),
                        float(mmcif_dict["_atom_site.Cartn_z"][i])] for i in range(len(atom_name))]
        atom_bfactor = mmcif_dict["_atom_site.B_iso_or_equiv"]
        atom_occupancy = mmcif_dict["_atom_site.occupancy"]
        # altloc = " "
        atom_fullname = mmcif_dict["_atom_site.label_atom_id"]
        new_model = Bio.PDB.Model.Model(model.id) # only place to use it the model
        for name, residue_name, residue_seq_id, chain_name, coords, bfactor, occupancy, fullname in \
                zip(atom_name, atom_residue_name, new_atom_residue_seq_id, atom_chain_name,
                    atom_coords, atom_bfactor, atom_occupancy, atom_fullname):
            # skip if atom is not part of an amino acid
            if is_aa(residue_name, standard=canonical):
                # add chain if it does not exist in the new_model
                if not new_model.has_id(chain_name):
                    new_chain = Bio.PDB.Chain.Chain(chain_name)
                    new_model.add(new_chain)
                if not new_chain.has_id(int(residue_seq_id)):
                    new_res = Bio.PDB.Residue.Residue((' ', int(residue_seq_id), ' '), residue_name, '')
                    new_chain.add(new_res)
                new_atom = Bio.PDB.Atom.Atom(fullname, coords, float(bfactor), float(occupancy), ' ', fullname, None, element=name)
                # new_atom.id = fullname
                # if it not possible then that means that the residue has alternative conformation (present in model as atom.altloc) - but we are not iterating thorugh it here!
                try:
                    new_res.add(new_atom)
                except Bio.PDB.PDBExceptions.PDBConstructionException:
                   pass


        return new_model

    def cubic_assembly_from_cif(self, file, symmetry):
        """Constructs a cubic assembly from an mmcif file.

        :param file: File to construct from.
        :param symmetry: Symmetry type to use. Either I, O or T."""
        start_time = time.time()
        cubicassembly = CubicSymmetricAssembly(file, symmetry)
        print("Created the assembly in: " + str(round(time.time() - start_time, 1)) + "s")
        return cubicassembly