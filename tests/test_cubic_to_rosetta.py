#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of cubic_to_rosetta.py
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import argparse
import os
from unittest import mock

import pandas as pd

from cubicsym.paths import I, O, T, IP, EXCLUDE_LIST
from pathlib import Path

def test_quality_assurance():
    from scripts.cubic_to_rosetta import quality_assurance
    from pyrosetta import init
    rosetta_repr = "inputs/1stm_repr.pdb"
    ico_repr = "inputs/1stm_ico.cif"
    input_file = "inputs/1stm.cif"
    symm_file = "inputs/1stm.symm"

    quality_assurance(rosetta_repr, ico_repr, input_file, symm_file, "1stm", 1)

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(I.joinpath("1STM.cif"))],
                symmetry="I",
                overwrite=True,
                rosetta_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv']
            ))
def test_cubic_to_rosetta_I(mock_args):
    from scripts.cubic_to_rosetta import main
    main()

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures= [str(O.joinpath("1AEW.cif"))], #[str(O.joinpath("3LVX.cif"))],
                symmetry="O",
                overwrite=True,
                rosetta_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv']
            ))
def test_cubic_to_rosetta_O(mock_args):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # import pydevd_pycharm
    # port_mapping = [64243, 64241]
    # pydevd_pycharm.settrace(IP, port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    from scripts.cubic_to_rosetta import main
    main()

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(T.joinpath("6JDD.cif"))],
                symmetry="T",
                overwrite=True,
                rosetta_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv']
            ))
def test_cubic_to_rosetta_T(mock_args):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    from mpi4py import MPI
    # import pydevd_pycharm
    # port_mapping = [64243, 64241]
    # pydevd_pycharm.settrace(IP, port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    from scripts.cubic_to_rosetta import main
    main()

  # # output names
  #   parser.add_argument('--symmetry_visualization_names', help="Names given to icosahedral files.", default='<prefix>_symmetry_visualization.py', nargs="+", type=str)

def create_test_list(symmetry):
    working = 0
    df = pd.read_csv("outputs/I/full_report.csv")
    structure = df[(df["working"] == False) & (df["rosetta_seqfault"] == False)]["structures"].values
    # sort them according to their size
    return sorted([str(p) for p in structure], key=os.path.getsize)[working:]

# uncomment @pytest.mark.mpi if you dont want to run with mpi
# mpirun command is:
# conda activate shapedesign && mpirun -n 2 python -m pytest -s -k test_construct_mode tests/test_cubic_to_rosetta.py --with-mpi
# @pytest.mark.mpi
output_stems = [p.stem for p in Path("outputs").glob("*")]
@mock.patch('argparse.ArgumentParser.parse_args',
    return_value=argparse.Namespace(
    structures=create_test_list(symmetry="I"),
    symmetry="I",
    overwrite=True,
    rosetta_repr_on=True,
    full_repr_on=True,
    symmetry_visualization_on=None,
    report_on=True,
    symdef_outpath="outputs",
    input_outpath="outputs",
    rosetta_repr_outpath="outputs",
    full_repr_outpath="outputs",
    symmetry_visualization_outpath="outputs",
    report_outpath="outputs",
    symdef_names='<prefix>.symm',
    input_names='<prefix>.cif',
    rosetta_repr_names='<prefix>_rosetta.pdb',
    full_repr_names='<prefix>_full.cif',
    symmetry_visualization_names='<prefix>_symmetry_visualization.py',
    quality_assurance=True,
    ))
def test_cubic_to_rosetta_I_multiple(mock_args):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # import pydevd_pycharm
    # port_mapping = [64243, 64241]
    # pydevd_pycharm.settrace(IP, port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    from scripts.cubic_to_rosetta import main
    main()
