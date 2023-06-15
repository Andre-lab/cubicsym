#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 12/5/22
"""
import random
import re
from Bio.Align.Applications import MafftCommandline
from scipy.spatial.distance import cdist
from typing import Iterable
from pyrosetta.rosetta.protocols.hybridization import TMalign, partial_align
from pyrosetta.rosetta.std import list_unsigned_long_t
from pyrosetta.rosetta.core.id import AtomID_Map_AtomID
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.pose import initialize_atomid_map
from pyrosetta import Vector1

class AtomSelector:

    def __init__(self, include_names=None, exclude_names=None, include_hydrogens=False):
        """Selects which atoms are allowed.

        :param include_hydrogens: to include hydrogens or not. Will always override include_names
        :param include_names: atom names to include.WWill always override exclude_names.
        :param exlude_names: atom names to exclude.
        """
        self.include_names = include_names if include_names else []
        self.exclude_names = exclude_names if exclude_names else []
        self.include_hydrogens = include_hydrogens

    def is_ok(self, residue, atom_index):
        # check if atom is hydrogen and if it is not allowed (self.include_hydrogen) then return False
        if not self.include_hydrogens and residue.atom_is_hydrogen(atom_index):
            return False
        # check if residue is in the include_names and return True if so
        if self.include_names and residue.atom_type(atom_index).name() in self.include_names:
            return True
        if self.exclude_names and residue.atom_type(atom_index).name() in self.exclude_names:
            return False
        # if include_names is defined, then we only want the ones in the include list
        if self.include_names:
            return False
        else:
            return True

def contactmap_with_mapped_resi(pose, distance_cutoff=10.0, include_names=None, exclude_names=None, include_hydrogens=False,
                                return_raw_distances=False):
    """Creates a contactmap of the pose and also returns a dict mapping the residue to the contactmap indcices.
    the residue map consist of 4 keys.
        1. pdb_resi: this the pdb information regarding the residue, that is the chain id and resi id.
        2. start: the starting atom number in the contactmap
        3. stop: the end atom number in the contactmap.
        4. atom_names: the names of the atoms mapped to the atom numbers."""
    atomsel = AtomSelector(include_names, exclude_names, include_hydrogens)
    atom_array = []
    atoms = 0
    resi_map = {}
    for i in range(1, pose.size() + 1):
        ri = pose.residue(i)
        ri0 = i - 1
        resi_map[ri0] = {"pdb_resi": pose.pdb_info().pose2pdb(i), "start": atoms, "stop": None, "atom_names": []}
        for ai in [i for i in range(1, ri.natoms() + 1) if atomsel.is_ok(ri, i)]:
            atom_array.append(list(ri.atom(ai).xyz()))
            resi_map[ri0]["atom_names"].append(ri.atom_type(ai).name())
            atoms += 1
        resi_map[ri0]["stop"] = atoms - 1
    assert atom_array, "NO ATOMS FOUND! Change the atom selection preferences. Remember in Rosetta the atoms might have different names. " \
                       "For instance 'CA' is called 'CAbb' in Rosetta."
    distances = cdist(np.array(atom_array), np.array(atom_array))
    if return_raw_distances:
        return distances, resi_map
    else:
        return distances <= distance_cutoff, resi_map

def convert_to_cpp_list(l):
    cpp_list = list_unsigned_long_t()
    for i in l:
        cpp_list.push_back(i)
    return cpp_list

def tmalign(query_pose, ref_pose, query_pose_residue_list: Iterable = None, ref_pose_residue_list: Iterable = None):
    """Use Rosetta implementation of TMalign to align query_pose (query pose) onto the ref_pose (reference pose).
    Optionally use the pose residue numbers of query_pose_residue_list and ref_pose_residue_list to do the alignment
    Remember that if you use the residue lists: TMALIGN WILL MOVE ONLY THE RESIDUES SPECIFIED SO YOUR query pose WILL NOT LOOK THE SAME
    as the original afterwards!"""
    # this follows the logic of test.protocols.hybridization.TMalign.cxxtest.hh in Rosetta
    tm_align = TMalign()
    query_pose_residue_list = convert_to_cpp_list(query_pose_residue_list if query_pose_residue_list else range(1, query_pose.size() + 1))
    ref_pose_residue_list = convert_to_cpp_list(ref_pose_residue_list if ref_pose_residue_list else range(1, ref_pose.size() + 1))
    tm_align.apply(query_pose, ref_pose, query_pose_residue_list, ref_pose_residue_list)
    # atom_map (vector< vector >) contains the number of atoms per resiude
    atom_map = AtomID_Map_AtomID()
    initialize_atomid_map( atom_map, query_pose, AtomID.BOGUS_ATOM_ID() )
    n_mapped_residues=0
    tm_align.alignment2AtomMap(query_pose, ref_pose, query_pose_residue_list, ref_pose_residue_list, n_mapped_residues, atom_map)
    normalize_length = min(query_pose.size(), ref_pose.size())
    TMscore = tm_align.TMscore(normalize_length)
    aln_cutoffs = Vector1([2, 1.5, 1.0, 0.5])
    min_coverage = 0.2
    # this does the actual alignment
    partial_align(query_pose,ref_pose, atom_map, query_pose_residue_list, True, aln_cutoffs, min_coverage)
    return TMscore


def write_fasta_on_chain_set(in_file, pose1, pose2, chain_set1, chain_set2):
    with open(in_file, "w") as f:
        # first write pose 1 / chain_set2
        for chain1 in chain_set1:
            f.write(f">1_{chain1}\n")
            f.write(pose1.chain_sequence(chain1) + "\n")
        for chain2 in chain_set2:
            f.write(f">2_{chain2}\n")
            f.write(pose2.chain_sequence(chain2) + "\n")


def sequence_alignment_on_chain_set(pose1, pose2, chain_set1, chain_set2):
    # 1. get the sequence for all chains and create a fasta file:
    in_file = f"/tmp/{''.join([str(random.randint(0, 9)) for i in range(10)])}.fasta"
    write_fasta_on_chain_set(in_file, pose1, pose2, chain_set1, chain_set2)
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
            idn, string = i[0], i[1]
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
    ids = ["1_" + str(chain) for chain in chain_set1] + ["2_" + str(chain) for chain in chain_set2]
    counter = {i: 0 for i in ids}
    alignment_map = {i: [] for i in ids}

    for star in stars:
        if star == " ":  # no match
            # increase counters for which there are letters, and if '-' dont.
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

    # since the sequences were from different chains, to get the correct residue number we have to multiply them by pose.chain_begin(chain)
    # this will automatically also push this to the correct 1-indexing for pose
    correct_alignment_map = {}
    for chain in chain_set1:
        idx = "1_" + str(chain)
        correct_alignment_map[idx] = [i + pose1.chain_begin(chain) for i in alignment_map[idx]]
    for chain in chain_set2:
        idx = "2_" + str(chain)
        correct_alignment_map[idx] = [i + pose2.chain_begin(chain) for i in alignment_map[idx]]

    # assert all lengths are the same
    assert all([len(correct_alignment_map[ids[0]]) == len(i) for i in correct_alignment_map.values()])
    # assert all residues are the same
    unique_sequences = set()
    for chain in chain_set1:
        idx = "1_" + str(chain)
        seq = "".join([pose1.residue(i).name1() for i in correct_alignment_map[idx]])
        unique_sequences.add(seq)
    for chain in chain_set2:
        idx = "2_" + str(chain)
        seq = "".join([pose2.residue(i).name1() for i in correct_alignment_map[idx]])
        unique_sequences.add(seq)
    assert len(unique_sequences) == 1, f"The sequence alignment does not consists of of identical chains (pose name = {pose1.pdb_info().name()})"

    return correct_alignment_map