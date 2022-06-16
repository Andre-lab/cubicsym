#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import warnings

import Bio
import re
import os
import random
import xmlrpc.client as xmlrpclib
import numpy as np
import time
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import PDBIO
from Bio import SeqIO
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MafftCommandline
from cubicsym.mathfunctions import rotation_matrix, distance
from string import ascii_lowercase, ascii_uppercase
from Bio.PDB.Polypeptide import is_aa, three_to_one
from Bio.PDB.Structure import Structure
from Bio.PDB.vectors import Vector
from cubicsym.mathfunctions import rotation_matrix, shortest_path, criteria_check, angle, \
    distance, vector, vector_angle, vector_projection, rotate, vector_projection_on_subspace


class Assembly(Structure):
    """A class build on top of the Structure class in BioPython"""

    def __init__(self, mmcif=None, assembly_id=None, id_="1", pdb=None):
        super().__init__(id_)
        self.cmd = xmlrpclib.ServerProxy('http://localhost:9123')
        self.server_root_path = ""
        self.assembly_id = assembly_id
        if mmcif and assembly_id:
            self.from_mmcif(mmcif)
        if pdb and not assembly_id:
            self.from_pdb(pdb)

    # FIXME: Only works for single chain subunits
    def from_pdb(self, file):
        chain_ids = iter(list(ascii_uppercase) + list(ascii_lowercase) + [str(i) for i in range(0, 10000)])
        structure = PDBParser(PERMISSIVE=1).get_structure(file, file)
        for subunit_number, chain in enumerate(structure.get_chains(), 1):
            subunit = Bio.PDB.Model.Model(f"{subunit_number}")
            new_chain = chain.copy()
            # new_chain.transform(r,t) - problem is that it right multiplies!
            new_chain.id = next(chain_ids)
            subunit.add(new_chain)
            self.add(subunit)
        # now we check if the assembly is valid and return if it is:
        self._create_sequence_alignment_map()

    # TODO: THE USE AUTHOR CHAINS CAN BE AUTOMATIC BY COMPARING THE AUTHOR CHAINS TO LABELS CHAINS TO THE ASYM CHAINS SPECIFIED IN struct_asym lines
    def from_mmcif(self, file):
        """Constructs an assembly from a mmcif file."""
        for subunit in self.get_subunits():
            self.detach_child(subunit.id)
        structure_name = Path(file).stem
        structure = MMCIFParser().get_structure(structure_name, file)
        model = structure[0] # only work with the first model (handles NMR in this case)
        mmcif_dict = MMCIF2Dict(file)
        self.allowed_ids = mmcif_dict["_pdbx_struct_assembly.id"]
        try:
            assin = self.allowed_ids.index(self.assembly_id)
        except ValueError as e:
            msg = f"The assembly id {assin} is not found in {file}. The available assembly ids are {', '.join(self.allowed_ids)}"
            raise ValueError(msg) from e
        else:
            start_time = time.time()
            assembly_details = mmcif_dict["_pdbx_struct_assembly.details"][assin]
            print("Generating assembly: id=" + self.assembly_id + " type='" + assembly_details + "'")
            # reconstruct the model so the chain names are easier to handle
            model = self._reconstruct(mmcif_dict, model)
            # fetch chains to apply apply_symmetry_to generate the assembly
            chain_to_apply_symmetry_to = [chain for chain in mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"][assin].split(",") if
                                          model.has_id(chain)]
            # retrieve symmetry operations
            rotations, translations = self._get_symmetry_operations(assin, mmcif_dict)
            # create a subunit list outer: [] is the subunits, inner: [] is the chains of the different subunits.
            asymmetric_chains = {}
            subunits = []
            for asym_chain, number in list(zip(mmcif_dict["_struct_asym.id"], mmcif_dict["_struct_asym.entity_id"])):
                if not model.has_id(asym_chain):
                    continue
                if not number in asymmetric_chains:
                    asymmetric_chains[number] = []
                asymmetric_chains[number].append(asym_chain)
            asymmetric_chains = sorted([chains for number, chains in asymmetric_chains.items()], key=len)
            # Four cases cases:
            # 1: [(1, ["A", "B", "C"])] - subunit with 1 chain where all other chains are asymmetric partners of that one. example: 1stm
            # 2: [(1, ["A", "B"]), 2: ["C", "D"])] subunit with many chains but each division have equal length
            # 3: [(1, ["A"]), 2: ["C", "D"])] subunit with many chains but each division does NOT have equal length: example: 3j17
            # 4: [(1, ["A", "B", "C"])] - similar to #1 but you want to include all the asymmetrical chains in the subunit: example: 1m1c
            # if 1 or 2:
            if all([len(chains) == len(asymmetric_chains[0]) for chains in asymmetric_chains]) and len(rotations) * len(
                    asymmetric_chains[0]) == 60:
                # 1
                if len(asymmetric_chains) == 1:
                    subunits = [[chain] for chain in asymmetric_chains[0]]
                # 2
                else:
                    subunits = [list(chains) for chains in zip(*asymmetric_chains)]
            # if 3 or 4
            else:
                subunits = [[chain for chains in asymmetric_chains for chain in chains]]
            # chain name and subunit number specifiers
            chain_ids = iter(list(ascii_uppercase) + list(ascii_lowercase) + [str(i) for i in range(0, 10000)])

            subunit_number = 0
            if len(rotations) == 0 and len(translations) == 0:
                # just make an identity rotation and translation
                rotations.append(np.identity(3))
                translations.append(np.zeros(3))
                # print("file contains full assembly")
                # assert (1 == 2)  # just fail here!
            for symmetry_operation_number, (r, t) in enumerate(zip(rotations, translations), 1):
                print("applying symmetry operation " + str(symmetry_operation_number) + "/" + str(len(rotations)))
                for chains in subunits:
                    subunit_number += 1
                    subunit = Bio.PDB.Model.Model(f"{subunit_number}")
                    for chain in chains:
                        for model_chain in model.get_chains():
                            if model_chain.id == chain:
                                new_chain = model_chain.copy()
                                # new_chain.transform(r,t) - problem is that it right multiplies!
                                for atom in new_chain.get_atoms():
                                    atom.coord = np.dot(r, atom.coord) + t
                                new_chain.id = next(chain_ids)
                                subunit.add(new_chain)
                    self.add(subunit)
                # now we check if the assembly is valid and return if it is:
            self._create_sequence_alignment_map()
            print("Created the assembly in: " + str(round(time.time() - start_time, 1)) + "s")

    def set_server_proxy(self, string: str):
        """Sets the xmlrpclib ServerProxy with string"""
        self.cmd = xmlrpclib.ServerProxy(string)

    def set_server_root_path(self, string: str):
        """Sets the root any path when using server functionality. Usefull if running PyMOL on another server and you mount it locally."""
        self.server_root_path = string

    def center(self):
        """Centers the assembly around the global center [0,0,0]."""
        self._clear_xtra()
        self._set_center()
        self.transform(np.identity(3), - self.xtra["center"])
        self.xtra["center"] = np.array([0,0,0], dtype=np.float32)

    def rotate_about_axis(self, axis, angle):
        """Rotates the assembly around the axis with an the angle."""
        self.transform(rotation_matrix(axis, angle), np.array([0,0,0]))
        self._clear_xtra()

    def make_sub_assemby(self, ids, name=""):
        """Makes a subassembly from the current assembly containing the ids and with an output name."""
        subunits = self.get_subunits(ids)
        assembly = Assembly(name)
        for subunit in subunits:
            assembly.add(subunit.copy())
        return assembly

    def show(self, name="assembly", ids=None, map_subunit_ids_to_chains=False):
        """shows the assembly or subunits with ids, if specified, in pymol."""
        tmp = f"/tmp/{name}.cif"
        self.output(tmp, ids, map_subunit_ids_to_chains=map_subunit_ids_to_chains)
        self.cmd.load(self.server_root_path + tmp)
        # time.sleep(0.1)
        os.remove(tmp)

    def show_center_of_mass(self, name="com", ids=None):
        """shows the center of mass of the subunits with ids in pymol."""
        tmp = f"/tmp/{name}.pdb"
        self.output_center_of_mass(tmp, ids)
        self.cmd.load(self.server_root_path + tmp)
        os.remove(tmp)

    def show_geometric_center(self, name="gc",  ids=None):
        """shows the geometric center of the subunits with ids in pymol."""
        tmp = f"/tmp/{name}.pdb"
        self.output_geometric_center(self.server_root_path + tmp, ids)
        self.cmd.load(tmp)
        os.remove(tmp)

    def show_alignment_geometric_center(self, name="all_gc",  ids=None):
        """shows the alignment geometric center of the subunits with ids in pymol."""
        tmp = f"/tmp/{name}.pdb"
        self.output_alignment_geometric_center(self.server_root_path + tmp, ids)
        self.cmd.load(tmp)
        os.remove(tmp)

    @staticmethod
    def output_subunit_as_specified(filename, subunit, format="cif"):
        """TODO"""
        if format == "cif":
            io = MMCIFIO()
            io.set_structure(subunit)
            io.save(filename)
            # rosetta is awesome so you have to add _citation.title in the end ...
            file = open(filename, "a+")
            file.write("_citation.title rosetta_wants_me_to_do_this..")
            file.close()
        else:
            io = PDBIO()
            io.set_structure(subunit)
            io.save(filename)

    def map_subunit_id_onto_chains(self):
        for subunit in self.get_subunits():
            for chain in subunit.get_chains():
                chain.id = subunit.id

    def output(self, filename, ids=None, format="cif", same=True, map_subunit_ids_to_chains=False):
        """Outputs the assembly to file.

        :param filename: The name of the file.
        :param ids: ids to use.
        :param format: format of the output [pdb, cif].
        :param same: have the same serial num or not. If True, PyMOL will by default set cartoon on all subunits.
        :param map_subunit_ids_to_chains: Will label the chains of each subunit as the subunit id. Useful for debugging.
        :return: None
        """
        if map_subunit_ids_to_chains:
            self.map_subunit_id_onto_chains()
        if format == "cif":
            io = MMCIFIO()
            if ids==None:
                if same == True:
                    for subunit in self.get_subunits():
                        subunit.serial_num = "1"
                    io.set_structure(self)
                    io.save(filename)
                    for subunit in self.get_subunits():
                        subunit.serial_num = subunit.id
                else:
                    io.set_structure(self)
                    io.save(filename)
            else:
                io.set_structure(self.make_sub_assemby(ids, filename))
                io.save(filename)
            # rosetta is awesome so you have to add _citation.title in the end ...
            file = open(filename, "a+")
            file.write("_citation.title rosetta_wants_me_to_do_this..")
            file.close()
        elif format == "pdb":
            io = PDBIO()
            if ids==None:
                if same == True:
                    for subunit in self.get_subunits():
                        subunit.serial_num = "1"
                    io.set_structure(self)
                    io.save(filename)
                    for subunit in self.get_subunits():
                        subunit.serial_num = subunit.id
                else:
                    io.set_structure(self)
                    io.save(filename)
            else:
                io.set_structure(self.make_sub_assemby(ids, filename))
                io.save(filename)

    def output_center_of_mass(self, filename, ids=None):
        """Prints a pdb file containing a single atom corresponding to the center of mass.
        The atom number is equal to the suffix of the protein name. The name must be "s<prefix>"""
        file = open(filename, "w")
        self._set_center_of_mass(ids)
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            center_of_mass = subunit.xtra["center_of_mass"]
            string = "{:<6}{:>5d}  {:<3}{:>4} {}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}        " \
                     "{:>4}{}".format("HETATM", int(subunit.id), subunit.id, "COM", "P",
                                      int(subunit.id), center_of_mass[0], center_of_mass[1],
                                      center_of_mass[2], 1.00, 0.00,
                                      "X", "")
            file.write(string + "\n")
        file.close()

    def output_geometric_center(self, filename, ids=None):
        """Prints a pdb file containing a single atom corresponding to the geometric center.
        The atom number is equal to the suffix of the protein name. The name must be "s<prefix>"""
        file = open(filename, "w")
        self._set_geometric_center(ids)
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            center_of_mass = subunit.xtra["geometric_center"]
            string = "{:<6}{:>5d}  {:<3}{:>4} {}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}        " \
                     "{:>4}{}".format("HETATM", int(subunit.id), subunit.id, "gc", "P",
                                      int(subunit.id), center_of_mass[0], center_of_mass[1],
                                      center_of_mass[2], 1.00, 0.00,
                                      "X", "")
            file.write(string + "\n")
        file.close()

    def output_alignment_geometric_center(self, filename, ids=None):
        """Prints a pdb file containing a single atom corresponding to the aligned geometric center.
        The atom number is equal to the suffix of the protein name. The name must be "s<prefix>"""
        file = open(filename, "w")
        self._set_geometric_center(ids)
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            center_of_mass = subunit.xtra["alignment_geometric_center"]
            string = "{:<6}{:>5d}  {:<3}{:>4} {}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}        " \
                     "{:>4}{}".format("HETATM", int(subunit.id), subunit.id, "gc", "P",
                                      int(subunit.id), center_of_mass[0], center_of_mass[1],
                                      center_of_mass[2], 1.00, 0.00,
                                      "X", "")
            file.write(string + "\n")
        file.close()

    def get_n_chains(self):
        """Gets the number of chains in 1 subunit."""
        return len(self["1"].get_list())

    def get_n_residues(self):
        """Gets the number of residues in 1 subunit"""
        return sum([len(chain.get_list()) for chain in self["1"].get_chains()])

    def get_size(self):
        """Gets the size of the assembly (= COM of the assembly to the COM 1 subunit)"""
        self._set_geometric_center("1")
        self._set_geometric_center()
        return round(distance(self.xtra["geometric_center"], self["1"].xtra["geometric_center"]), 2)

    def get_geometric_center(self, ids=None):
        """Returns the geometric center of either the total assembly of the subunits if specied through ids."""
        subunits = self.get_subunits(ids)
        self._set_geometric_center(ids)
        if ids==None:
            return self.xtra["geometric_center"]
        else:
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(subunits, 1):
                geometric_center_unorm += subunit.xtra["geometric_center"]
            geometric_center = geometric_center_unorm / total
            return geometric_center

    def get_alignment_geometric_center(self, ids=None):
        """Returns the alignment geometric center of either the total assembly of the subunits if specied."""
        subunits = self.get_subunits(ids)
        self._set_alignment_geometric_center(ids)
        if ids==None:
            return self.xtra["alignment_geometric_center"]
        else:
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(subunits, 1):
                geometric_center_unorm += subunit.xtra["alignment_geometric_center"]
            geometric_center = geometric_center_unorm / total
            return geometric_center

    def get_center_of_mass(self, ids=None):
        """Returns the center of mass of either the total assembly of the subunits if specied."""
        subunits = self.get_subunits(ids)
        self._set_center_of_mass(ids)
        if ids == None:
            return self.xtra["center_of_mass"]
        else:
            center_of_mass_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(subunits, 1):
                center_of_mass_unorm += subunit.xtra["center_of_mass"]
            center_of_mass = center_of_mass_unorm / total
            return center_of_mass

    def get_subunits(self, ids=None):
        """Returns the subunits with the given ids.

        :param ids: The ids of the subunits
        :return: Bio.PDB.Model.Model
        """
        if ids == None:
            return self.get_list()
        else:
            if isinstance(ids, str):
                ids = [ids]
            return [subunit for subunit in self.get_models() if subunit.id in ids]

    def get_subunit_with_id(self, id):
        """Returns the subunit with the given id.

        :param id: The id of the subunit
        :return: Bio.PDB.Model.Model
        """
        for subunit in self.get_models():
            if subunit.id == id:
                return subunit

    def residue_center_of_mass(self, id_="1"):
        """Replica of the Rosetta function of the same name. Used when """
        center = Vector(0, 0, 0)
        atoms = [a for a in self.get_subunit_with_id(id_).get_atoms() if a.name == "CA"]
        for atom in atoms:
            center += atom.get_coord()
        center /= len(atoms)
        # calling return_nearest_residue in Rosetta
        closest_coord = None
        min_dist = np.inf
        for atom in atoms:
            norm = (center - atom.get_coord()).norm()
            if norm < min_dist:
                min_dist = norm
                closest_coord = atom.get_coord()
        return closest_coord

    def get_subunit_CA_closest_to_COM(self, id="1"):
        """Gets the CA atom that is closest to the COM

        :param id: The id of the subunit
        :return: Bio.PDB.Atom.Atom.get_coords()
        """
        self._set_center_of_mass(id)
        subunit = self.get_subunit_with_id(id)
        shortest_distance = None
        shortest_coordinate = None
        for atom in subunit.get_atoms():
            if not atom.get_fullname().replace(" ", "") == "CA":
                continue
            new_distance = np.linalg.norm(atom.get_coord() - subunit.xtra["center_of_mass"])
            if not shortest_distance:
                shortest_distance = new_distance
                continue
            if new_distance < shortest_distance:
                shortest_distance = new_distance
                shortest_coordinate = atom.get_coord()
        return shortest_coordinate

    def get_subunit_CA_closest_to_GC(self, id="1"):
        """Gets the CA atom that is closest to the GC

        :param id: The id of the subunit
        :return: Bio.PDB.Atom.Atom.get_coords()
        """
        self._set_geometric_center(id)
        subunit = self.get_subunit_with_id(id)
        shortest_distance = None
        shortest_coordinate = None
        for atom in subunit.get_atoms():
            if not atom.get_fullname().replace(" ", "") == "CA":
                continue
            new_distance = np.linalg.norm(atom.get_coord() - subunit.xtra["geometric_center"])
            if not shortest_distance:
                shortest_distance = new_distance
                continue
            if new_distance < shortest_distance:
                shortest_distance = new_distance
                shortest_coordinate = atom.get_coord()
        return shortest_coordinate

    def find_minimum_distance_between_subunits(self, id, id_of_ref):
        """TODO"""
        min_distance = 999999
        subunit = self.get_subunit_with_id(id)
        subunit_ref = self.get_subunit_with_id(id_of_ref)
        for atom in subunit.get_atoms():
            for atom_ref in subunit_ref.get_atoms():
                min_distance = min(distance(atom.get_coord(), atom_ref.get_coord()), min_distance)
        return min_distance

    def find_minimum_atom_type_distance_between_subunits(self, id, id_of_ref, atom_type="CB"):
        """TODO"""
        min_distance = 999999
        subunit = self.get_subunit_with_id(id)
        subunit_ref = self.get_subunit_with_id(id_of_ref)
        subunit_atoms = [atom for atom in subunit.get_atoms() if atom.fullname == atom_type]
        subunit_ref_atoms = [atom for atom in subunit_ref.get_atoms() if atom.fullname == atom_type]
        for atom in subunit_atoms:
            for atom_ref in subunit_ref_atoms:
                min_distance = min(distance(atom.get_coord(), atom_ref.get_coord()), min_distance)
        return min_distance

    def _set_center(self):
        """Calculates the center of the assembly and store it in xtra["center"]"""
        current_center = np.array([0, 0, 0], dtype=np.float32)
        for n, atom in enumerate(self.get_atoms(), 1):
            current_center += atom.get_coord()
        current_center /= n
        self.xtra["center"] = current_center

    def _clear_xtra(self):
        """Will clear all geometrical data (center_of_mass for instance) for each subunit. Are called when when these
        become invalid. This could be when center() or rotate_about_axis is called."""
        xtra_to_save = ("alignment")
        for subunit in self.get_models():
            subunit.xtra = {key:val for key, val  in subunit.xtra.items() if key == xtra_to_save}

    def _set_center_of_mass(self, ids=None):
        """Sets the center of mass of the assembly."""
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            if "center_of_mass" in subunit.xtra:
                continue
            center_of_mass_unorm = np.array([0, 0, 0], dtype=np.float64)
            total_mass = 0
            for atom in subunit.get_atoms():
                center_of_mass_unorm += atom.get_coord() * atom.mass
                total_mass += atom.mass
            center_of_mass = center_of_mass_unorm / total_mass
            subunit.xtra["center_of_mass"] = center_of_mass
        # if no subunits are specified also set the assembly center_of_mass
        if ids==None:
            center_of_mass_unorm = np.array([0, 0, 0], dtype=np.float64)
            total_mass = 0
            for total, subunit in enumerate(self.get_models(), 1):
                center_of_mass_unorm += subunit.xtra["center_of_mass"]
                # the total mass is recalculated!
                for atom in subunit.get_atoms():
                    center_of_mass_unorm += atom.get_coord() * atom.mass
                    total_mass += atom.mass
            center_of_mass = center_of_mass_unorm / total_mass
            self.xtra["center_of_mass"] = center_of_mass

    def get_ca_atoms_for_subunit(self, subunit):
        """Creates a list containing only CA atoms of the subunit."""
        return [a for a in subunit.get_atoms() if a.name == "CA"]

    def _set_geometric_center(self, ids=None):
        """Sets the geometric center of the assembly."""
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            if "geometric_center" in subunit.xtra:
                continue
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            ca_atoms = self.get_ca_atoms_for_subunit(subunit)
            for atom in ca_atoms:
                geometric_center_unorm += atom.get_coord()
            geometric_center = geometric_center_unorm / len(ca_atoms)
            subunit.xtra["geometric_center"] = geometric_center
        # if no subunits are specified also set the assembly geometric_center
        # tis works because all geometric centers are set above because self.get_subunits(None) returns all subunits.
        if ids == None:
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(self.get_models(), 1):
                geometric_center_unorm += subunit.xtra["geometric_center"]
                geometric_center = geometric_center_unorm / total
            self.xtra["geometric_center"] = geometric_center

    def _set_alignment_geometric_center(self, ids=None):
        """Sets the geometric center of the assembly but uses only the correspoding residue posistions given
        in subunit.xtra["alignment"]."""
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            if "alignment_geometric_center" in subunit.xtra:
                continue
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            total = 0
            ### WHAT IS DIFFERENT! ###
            residues = list(subunit.get_residues())
            try:
                for position in subunit.xtra["alignment"]:
                    resi = residues[position]
                ##########################
                    for atom in resi.get_atoms():
                        geometric_center_unorm += atom.get_coord()
                        total += 1
                geometric_center = geometric_center_unorm / total
                ### WHAT IS DIFFERENT! ###
                subunit.xtra["alignment_geometric_center"] = geometric_center
                ##########################
            except:
                pass
        # if no subunits are specified also set the assembly geometric_center
        # this works because all geometric centers are set above because self.get_subunits(None) returns all subunits.
        if ids == None:
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(self.get_models(), 1):
                ### WHAT IS DIFFERENT! ###
                geometric_center_unorm += subunit.xtra["alignment_geometric_center"]
                ##########################
                geometric_center = geometric_center_unorm / total
            ### WHAT IS DIFFERENT! ###
            self.xtra["alignment_geometric_center"] = geometric_center
            ##########################

    def _write_fasta(self, out, wrap=80):
        with open(out, "w") as f:
            for subunit in self.get_subunits():
                seq = "".join([three_to_one(i.get_resname()) for i in subunit.get_residues() if is_aa(i, standard=True)])
                f.write(f">{subunit.id}\n")
                for seq_line in [seq[i:i + wrap] for i in range(0, len(seq), wrap)]:
                    f.write(f"{seq_line}\n")

    def _create_sequence_alignment_map(self):


        # # todo: theres is something weird about the way i have build self,
        # #   such that more direct methods dont work. I only get one chain when
        # #   I should get multiple for ppb.build_peptides(subunit) or another method,
        # #   seqrecord = PdbIO.AtomIterator("1stm_assembly", self), so I am doing this manually now
        # ppb = ppb=PPBuilder()
        # seqs = []
        # for subunit in self.get_subunits():
        #     seq = ppb.build_peptides(subunit, aa_only=0)
        #     seqs.append(SeqRecord(seq[0].get_sequence(), id=subunit.id, description=""))
        # # todo: remeber to delete this file again in tmå
        # in_file = f"/tmp/{''.join([str(random.randint(0,9)) for i in range(10)])}.fasta"
        # SeqIO.write(seqs, in_file, "fasta")


        # 1. get the sequence for all chains and create a fasta file:
        in_file = f"/tmp/{''.join([str(random.randint(0,9)) for i in range(10)])}.fasta"
        self._write_fasta(in_file)
        mafft_cline = MafftCommandline(input=in_file, clustalout=True)  # , thread=1)
        stdout, stderr = mafft_cline()
        new = stdout.split("\n")
        new.pop(0)  # remove header
        new = [i for i in new if i != ""]  # remove all empty elements
        mafft_alignment = {}
        stars = ""
        for i in new:
            if i[0] != " ":
                i = re.split('\s+', i)
                idn, string = int(i[0]), i[1]
                if idn in mafft_alignment:
                    mafft_alignment[idn] += string
                else:
                    mafft_alignment[idn] = string
            else:
                # so there's 16 letters from start to sequence, including the id number:
                # 1               IVPFIRSLLMPTTGPASIPDDTLEKHTLRSETSTYNLTVGDTGSGLIVFFPGFPGSIVGA
                # 60              HYTLQSNGNYKFDQMLLTAQNLPASYNYCRLVSRSLTVRSSTLPGGVYALNGTINAVTFQ
                #                 ***********************************************************
                stars += i[16:]

        counter = {i: 0 for i in range(1, 61)}
        alignment_map = {i: [] for i in range(1, 61)}

        # WE ARE USING ZERO INDEXING!
        for n, star in enumerate(stars):
            if star == " ": # no match
                # increase counters for which there are lettes, and if '-' dont.
                for i in mafft_alignment.keys():
                    letter = mafft_alignment[i][counter[i]]
                    if letter != "-":
                        counter[i] += 1
            # if "*" that means that that particular sequence residue matches across all chains
            # and we therefore want to put that into the alignment map
            elif star == "*":
                for i in alignment_map.keys():
                    alignment_map[i].append(counter[i])
                    # Now that we have processed that letter increase all counters by 1
                    counter[i] += 1

        assert all([len(alignment_map[1]) == len(i) for i in alignment_map.values()])

        self.alignment_map = alignment_map
        self.mafft_alignment = mafft_alignment
        self.mafft_stars = stars

        # now put that into the all the subunits
        for n, subunit in enumerate(self.get_subunits(), 1):
            subunit.xtra["alignment"] = alignment_map[n]

    def _reconstruct(self, mmcif_dict, model, canonical=False):
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

    def _get_symmetry_operations(self, pick, mmcif_dict):
        """

        :param pick:
        :param mmcif_dict:
        :return:
        """
        # fetch symmetry operations numbers to generate the picked assembly
        symmetry_operations = mmcif_dict["_pdbx_struct_assembly_gen.oper_expression"][pick]
        if ")(" in symmetry_operations:
            warnings.warn(f"Code does not yet support combinations of operations such as {symmetry_operations}")
            return [], []
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
        # What is the benifit of sorting them? This can give rise to value errors if the names are not digits!!
        # try:
        #     symmetry_operations.sort(key=int)
        # except ValueError: # operations consists of non-numbers. Just keep it as it is.
        #     ...

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

    def _get_global_center(self):
        """Defines the global center and its coordinate frame."""
        global_z = np.array([0, 0, 1])
        global_y = np.array([0, 1, 0])
        global_x = np.array([1, 0, 0])
        global_center = np.array([0, 0, 0])
        return global_x, global_y, global_z, global_center

    def align_to_z_axis(self, ids=None):
        """Aligns the assembly to the z-axis. Can choose certain ids to use."""
        _, _, global_z, global_center = self._get_global_center()
        current_axis = vector(self.get_geometric_center(ids=ids), global_center)
        self.rotate_about_axis(np.cross(global_z, current_axis), vector_angle(global_z, current_axis))

    def align_subunit_to_x_axis(self, subunit):
        """Aligns the assembly so that the subunit CA atom closest to the geometric center is along the x-axis."""
        global_x, _, global_z, global_center = self._get_global_center()
        subunit_com = self.get_subunit_CA_closest_to_GC(subunit)
        z1high = vector_projection(subunit_com, global_z)
        x_vector = vector(subunit_com, z1high)
        x_rotation = vector_angle(x_vector, global_x)
        self.rotate_about_axis(np.cross(global_x, x_vector), x_rotation)

    def create_centered_subunit(self, subunit_id):
        """Centers the subunit so that the CA atom closest to the geometric center is located at the global center."""
        _, _, _, global_center = self._get_global_center()
        adjust = vector(global_center, self.get_subunit_CA_closest_to_GC(subunit_id))
        master = self.get_subunit_with_id(subunit_id).copy()
        master.transform(np.identity(3), adjust)
        return master
