#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""

import Bio
import re
import os
import random
import xmlrpc.client as xmlrpclib
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import PDBIO
from Bio import SeqIO
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MafftCommandline
from .mathfunctions import rotation_matrix, distance

class Assembly(Bio.PDB.Structure.Structure):
    """A class build on top of BioPython. Bio.PDB.Structure.Structure with additional methods."""
    def __init__(self, id):
        super().__init__(id)
        self.pymol = xmlrpclib.ServerProxy('http://localhost:9123')

# public

    def set_server_proxy(self, string):
        """

        :param str:
        :return:
        """
        self.pymol = xmlrpclib.ServerProxy(string)

    def center(self):
        """Centers the assembly around the global center [0,0,0]
        """
        self._clear_xtra()
        self._set_center()
        self.transform(np.identity(3), - self.xtra["center"])
        self.xtra["center"] = np.array([0,0,0], dtype=np.float32)

    def rotate_about_axis(self, axis, angle):
        """

        :param axis:
        :param deegres:
        :return: None
        """
        self.transform(rotation_matrix(axis, angle), np.array([0,0,0]))
        self._clear_xtra()

    def make_sub_assemby(self, ids, name=""):
        """

        :param name:
        :param ids:
        :return:
        """
        subunits = [subunit for subunit in self.get_models() if subunit.id in ids]
        assembly = Assembly(name)
        for subunit in subunits:
            assembly.add(subunit.copy())
        return assembly

    def show(self, name="assembly", ids=None):
        """
        shows the assembly or subunits with ids, if specified, in pymol.

        :param ids:
        :return:
        """
        tmp = f"/tmp/{name}.cif"
        self.output(tmp, ids)
        self.pymol.load(tmp)
        # time.sleep(0.1)
        os.remove(tmp)

    def show_center_of_mass(self, name="com", ids=None):
        """
        shows the center of mass of the subunits with ids in pymol

        :param ids:
        :return:
        """
        tmp = f"/tmp/{name}.pdb"
        self.output_center_of_mass(tmp, ids)
        self.pymol.load(tmp)
        os.remove(tmp)

    def show_geometric_center(self, name="gc",  ids=None):
        """
        shows the geometric center of the subunits with ids in pymol

        :param ids:
        :return:
        """
        tmp = f"/tmp/{name}.pdb"
        self.output_geometric_center(tmp, ids)
        self.pymol.load(tmp)
        os.remove(tmp)

    def show_alignment_geometric_center(self, name="all_gc",  ids=None):
        """shows the alignment geometric center of the subunits with ids in pymol

        :param ids:
        :return:
        """
        tmp = f"/tmp/{name}.pdb"
        self.output_alignment_geometric_center(tmp, ids)
        self.pymol.load(tmp)
        os.remove(tmp)

    def output_subunit_as_specified(self, filename, subunit, format="cif"):
        """

        :param filename:
        :param format:
        :param ids:
        :return:
        """
        if format == "cif":
            io = MMCIFIO()
            io.set_structure(subunit)
            io.save(filename)
            # rosetta is awesome so you have to add _citation.title in the end ...
            file = open(filename, "a+")
            file.write("_citation.title rosetta_wants_me_to_do_this..")
            file.close()

    def output(self, filename, ids=None, format="cif", same=True):
        """

        :param filename:
        :param ids:
        :param format:
        :param same: have the same serial num or not. If True, PyMOL will by default set cartoon on all subunits
        :return:
        """
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

        The atom number is equal to the suffix of the protein name.
        The name must be "s<prefix>".

        :param str filename: name of the pdb file.

        """
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
        """Prints a pdb file containing a single atom corresponding to the center of mass.

        The atom number is equal to the suffix of the protein name.
        The name must be "s<prefix>".

        :param str filename: name of the pdb file.

        """
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
        """Prints a pdb file containing a single atom corresponding to the center of mass.

        The atom number is equal to the suffix of the protein name.
        The name must be "s<prefix>".

        :param str filename: name of the pdb file.

        """
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
        self._set_center_of_mass("1")
        self._set_center_of_mass()
        return round(distance(self.xtra["center_of_mass"], self["1"].xtra["center_of_mass"]), 2)

    def get_geometric_center(self, ids=None):
        """
        Returns the geometric center of either the total assembly of the subunits if specied.

        :param subunits:
        :return:
        """
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
        """Returns the alignment geometric center of either the total assembly of the subunits if specied.

        :param subunits:
        :return:
        """
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
        """
        Returns the center of mass of either the total assembly of the subunits if specied.

        :param subunits:
        :return:
        """
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
        """
        Returns the subunits with the given ids.

        :param id: The id of the subunits
        :return: Bio.PDB.Model.Model
        """
        if ids == None:
            return self.get_list()
        else:
            return [subunit for subunit in self.get_models() if subunit.id in ids]

    def get_subunit_with_id(self, id):
        """
        Returns the subunit with the given id.

        :param id: The id of the subunit
        :return: Bio.PDB.Model.Model
        """
        for subunit in self.get_models():
            if subunit.id == id:
                return subunit

    def get_subunit_CA_closest_to_COM(self, id="1"):
        """
        Gets the CA atom that is closest to the COM

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

    def find_minimum_distance_between_subunits(self, id, id_of_ref):
        """

        :param id:
        :param id_of_ref:
        :return:
        """
        min_distance = 999999
        subunit = self.get_subunit_with_id(id)
        subunit_ref = self.get_subunit_with_id(id_of_ref)
        for atom in subunit.get_atoms():
            for atom_ref in subunit_ref.get_atoms():
                min_distance = min(distance(atom.get_coord(), atom_ref.get_coord()), min_distance)
        return min_distance

    def find_minimum_atom_type_distance_between_subunits(self, id, id_of_ref, atom_type="CB"):
        """

        :param id:
        :param id_of_ref:
        :return:
        """
        min_distance = 999999
        subunit = self.get_subunit_with_id(id)
        subunit_ref = self.get_subunit_with_id(id_of_ref)
        subunit_atoms = [atom for atom in subunit.get_atoms() if atom.fullname == atom_type]
        subunit_ref_atoms = [atom for atom in subunit_ref.get_atoms() if atom.fullname == atom_type]
        for atom in subunit_atoms:
            for atom_ref in subunit_ref_atoms:
                min_distance = min(distance(atom.get_coord(), atom_ref.get_coord()), min_distance)
        return min_distance

# private

    def _set_center(self):
        """Calculates the center of the assembly and store it in xtra["center"]
        """
        current_center = np.array([0, 0, 0], dtype=np.float32)
        for n, atom in enumerate(self.get_atoms(), 1):
            current_center += atom.get_coord()
        current_center /= n
        self.xtra["center"] = current_center

    def _clear_xtra(self):
        """Will clear all geometrical data (center_of_mass for instance) for each subunit. Are called when when these
        become invalid. This could be when center() or rotate_about_axis is called.
        """
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
            for total, subunit in enumerate(self.get_models(), 1):
                center_of_mass_unorm += subunit.xtra["center_of_mass"]
            center_of_mass = center_of_mass_unorm / total_mass
            self.xtra["center_of_mass"] = center_of_mass

    def _set_geometric_center(self, ids=None):
        """Sets the geometric center of the assembly."""
        subunits = self.get_subunits(ids)
        for subunit in subunits:
            if "geometric_center" in subunit.xtra:
                continue
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            total = 0
            for atom in subunit.get_atoms():
                geometric_center_unorm += atom.get_coord()
                total += 1
            geometric_center = geometric_center_unorm / total
            subunit.xtra["geometric_center"] = geometric_center
        # if no subunits are specified also set the assembly geometric_center
        # this works because all geometric centers are set above because self.get_subunits(None) returns all subunits.
        if ids == None:
            geometric_center_unorm = np.array([0, 0, 0], dtype=np.float64)
            for total, subunit in enumerate(self.get_models(), 1):
                geometric_center_unorm += subunit.xtra["geometric_center"]
                geometric_center = geometric_center_unorm / total
            self.xtra["geometric_center"] = geometric_center

    def _set_alignment_geometric_center(self, ids=None):
        """Sets the geometric center of the assembly but it uses only the correspoding residue posistions given
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

    def _create_sequence_alignment_map(self):

        # 1. get the sequence for all chains and create a fasta file:

        # todo: theres is something weird about the way i have build self,
        #   such that more direct methods dont work. I only get one chain when
        #   I should get multiple for ppb.build_peptides(subunit) or another method,
        #   seqrecord = PdbIO.AtomIterator("1stm_assembly", self), so I am doing this manually now
        ppb = ppb=PPBuilder()
        seqs = []
        for subunit in self.get_subunits():
            seqs.append(SeqRecord( ppb.build_peptides(subunit)[0].get_sequence(), id=subunit.id, description=""))
        # todo: remeber to delete this file again in tmå
        in_file = f"/tmp/{''.join([str(random.randint(0,9)) for i in range(10)])}.fasta"
        SeqIO.write(seqs, in_file, "fasta")
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
