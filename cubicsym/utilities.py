#!/usr/bin/env python3
# coding=utf-8
"""
Utility functions
@Author: Mads Jeppesen
@Date: 4/6/22
"""
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser



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