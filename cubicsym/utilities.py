#!/usr/bin/env python3
# coding=utf-8
"""
Utility functions
@Author: Mads Jeppesen
@Date: 4/6/22
"""
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from itertools import combinations
from symmetryhandler.reference_kinematics import get_dofs



def cut_all_but_chains(pose, *chains):
    """Cuts all chain in the pose except for the chains given."""
    # collect all residues not in the chains of the pose and find the min, max residue number
    resi_chain_map = {}
    for resi in range(1, pose.size() + 1):
        chain = pose.pdb_info().chain(resi)
        if not chain in chains:
            if chain not in resi_chain_map:
                resi_chain_map[chain] = []
            resi_chain_map[chain].append(resi)
    ranges = []
    for k, v in resi_chain_map.items():
        ranges.append({"min":min(v), "max":max(v)})
    # now delete from the top and down using the min and max residue number
    for v in sorted(ranges, key=lambda x: x["max"], reverse=True):
        pose.delete_residue_range_slow(v["min"], v["max"])
    return pose

def number_of_chains(file, pdb=True, cif=False):
    """Returns the number of chains in the file."""
    structure_name = file.split("/")[-1].split(".")[0]
    parser = None
    if pdb == True:
        parser = PDBParser()
    if cif == True:
        parser = MMCIFParser()
    structure = parser.get_structure(structure_name, file)
    return len(list(structure.get_chains()))

def number_of_residues(file, pdb=True, cif=False):
    """Returns the number of residues in a file."""
    structure_name = file.split("/")[-1].split(".")[0]
    parser = None
    if pdb == True:
        parser = PDBParser()
    if cif == True:
        parser = MMCIFParser()
    structure = parser.get_structure(structure_name, file)
    return len(list(structure.get_residues()))

def mpi_starmap(func, comm, *inputs, files=None):
    """Substitute for mpi4py starmap. Works similarly to MPIPoolExecutor().starmap.
    Can also read parameters from several files, supplied as an iterable of files.

    :param func: Function to be called across cores.
    :param comm: The mpi communicato
    :param inputs: Inputs to be parsed to the function. Inputs should be contained in a list, even if it is only 1 element.
    :param files: LIST of files to read input parameters from. This will be prepended first to the argument list.
    :return: A list of what 'func' returns.
    """
    assert all(type(inp) == list for inp in inputs), "all inputs must be contained in a list"
    # first check if the files parameter is passed and then prepend them to inputs
    if files:
        inputs = list(inputs)
        for file in files:
            parameter = []
            for line in open(file, "r"):
                line = line.rstrip("\n")
                if line:
                    parameter.append(line)
            inputs = [parameter] + inputs
        inputs = tuple(inputs) # just to put it back to what is was. It is not important!

    # all inputs should be the same len. If there are 3 different sets of lens then fail. If there are 2 sets,
    # then make sure the one set is only len = 1. Then make it the len of the longest
    len_set = set([len(inp) for inp in inputs])
    assert len(len_set) < 3, "You cannot have 3 different lengths of inputs"
    if len(len_set) == 2:
        temp_inputs = []
        highest = max(len_set)
        assert min(len_set) == 1, "You cannot have 2 lengths of inputs where one of them are not 1, and the other the same length."
        for input in inputs:
            if len(input) == 1:
                temp_inputs.append([input[0] for _ in range(highest)])
            else:
                temp_inputs.append(input)
        inputs = temp_inputs
    # divide the inputs among the cores
    if comm.Get_rank() == 0:
        data = list(zip(*inputs))
        k, m = divmod(len(data), comm.Get_size())
        data = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(comm.Get_size())]
    else:
        data = None
    data = comm.scatter(data, root=0)
    results = [func(*d) for d in data]
    return results

def write_string_to_bytearray(string):
    """Turns a string into a bytearray."""
    b = bytearray()
    b.extend(map(ord, string))
    return b

def divide_to_scatter(data, comm):
    """Divides inputs to the amount of cores present."""
    k, m = divmod(len(data), comm.Get_size())
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(comm.Get_size())]

def get_all_ca_atoms_slow(pose):
    """Gets all the backbane CA atoms. Slow because batch_get_xyz() is probably faster."""
    xyz = []
    for resi in range(1, pose.size() +1):
        if pose.residue(resi).is_protein():
            xyz.append(pose.residue(resi).xyz("CA"))
    return np.array(xyz)

def get_all_ca_atoms_slow_in_range(pose, start, end):
    """Gets all the backbane CA atoms. Slow because batch_get_xyz() is probably faster."""
    xyz = []
    for resi in range(start, end):
        if pose.residue(resi).is_protein():
            xyz.append(pose.residue(resi).xyz("CA"))
    return np.array(xyz)

def pose_cas_are_identical(*poses, atol=1e-3, map_chains:list=None):
    """Checks if the pairwise distance of the pose CA's are below a threshold and returns True if so."""
    if map_chains:
        for chains in map_chains:
            xyzs = []
            for chain, pose in zip(chains, poses):
                xyzs.append([get_all_ca_atoms_slow_in_range(pose, pose.chain_begin(chain), pose.chain_end(chain) + 1)])
            # pairwise comparison
            if not all([np.allclose(xyzs[i], xyzs[j], atol=atol) for i, j in list(combinations(range(len(poses)), 2))]):
                return False
        return True
    else:
        xyzs = [get_all_ca_atoms_slow(p) for p in poses]
        return all([np.allclose(xyzs[i], xyzs[j], atol=atol) for i, j in list(combinations(range(len(poses)), 2))])

def get_chain_map(symmetry_type, is_righthanded):
    """Returns a list of the mapping between each chain (in Rosetta numbering) between a HF-, 3- and 2-fold based setup.
    The first index is the HF-fold, the second 3-fold and third the 2-fold"""
    if symmetry_type == "I":
        if is_righthanded:
            return [(1, 1, 1), (2, 4, 7), (3, 8, 6), (4, 9, 5), (5, 6, 3), (6, 2, 4), (7, 7, 8), (8, 5, 2), (9, 3, 9)] # 1stm
        else:
            return [(1, 1, 1), (2, 4, 3), (3, 8, 5), (4, 9, 6), (5, 6, 7), (6, 3, 4), (7, 5, 8), (8, 7, 2), (9, 2, 9)] # 6s44
    elif symmetry_type == "O":
        if is_righthanded:
            return [(1, 1, 1), (2, 4, 6), (3, 8, 5), (4, 6, 3), (5, 2, 4), (6, 7, 7), (7, 5, 2), (8, 3, 8)] # 1AEW
        else:
            return [(1, 1, 1), (2, 4, 3), (3, 8, 5), (4, 6, 6), (5, 3, 4), (6, 5, 7), (7, 7, 2), (8, 2, 8)] # 1PY3
    else:
        if is_righthanded:
            return [(1, 1, 1), (2, 4, 5), (3, 6, 3), (4, 2, 4), (5, 7, 6), (6, 5, 2), (7, 3, 7)] # 1H0S
        else:
            return [(1, 1, 1), (2, 4, 3), (3, 6, 5), (4, 3, 4), (5, 5, 6), (6, 7, 2), (7, 2, 7)] # 1MOG

# def map_hf_right_to_left(symmetry_type):
#     if symmetry_type == "I":
#         return [1, 2, 3, 4, 5, 9, 8, 7, 6]
#         raise NotImplementedError("CHECK THIS IS CORRECT!!! you should probably just need some angle_z movement")
#     elif symmetry_type == "O":
#         raise NotImplementedError("CHECK THIS IS CORRECT!!! you should probably just need some angle_z movement")
#         return #[1, 2, 3, 4, 8, 7, 6, 5]
#     else:
#         return #[1, 2, 3, 7, 6, 5, 4]


def map_hf_right_to_left_hf(symmetry_type):
    if symmetry_type == "I":
        return [1, 2, 3, 4, 5, 9, 8, 7, 6]
    elif symmetry_type == "O":
        return [1, 2, 3, 4, 8, 7, 6, 5]
    else:
        return [1, 2, 3, 7, 6, 5, 4]

def map_3f_right_to_left_hf(symmetry_type):
    if symmetry_type == "I":
        return [1, 4, 8, 9, 6, 3, 5, 7, 2]
    elif symmetry_type == "O":
        return [1, 4, 8, 6, 3, 5, 7, 2]
    else:
        return [1, 4, 6, 3, 5, 7, 2]

def map_2f_right_to_left_hf(symmetry_type):
    if symmetry_type == "I":
        raise NotImplementedError("NO CHAIN MAP EXIST FOR THIS YET SORRY!")
    elif symmetry_type == "O":
        raise NotImplementedError("NO CHAIN MAP EXIST FOR THIS YET SORRY!")
    else:
        return [1, 4, 7, 3, 6, 2, 5]

def get_jumpidentifier(pose) -> str:
    return list(get_dofs(pose).keys())[0].split("JUMP")[-1].split("fold")[0]


