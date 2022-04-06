#!/usr/bin/env python3
# coding=utf-8
"""
Utility functions
@Author: Mads Jeppesen
@Date: 4/6/22
"""
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

def number_of_chains(file, pdb=True, cif=False):
    """Returns the number of of chains in the file."""
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