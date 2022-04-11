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
from cubicsym.paths import I, O, T, IP

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
                rosetta_repr_on=True,
                ico_on=True,
                symmetry_visualization_on=None,
                report_on=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                ico_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names='<prefix>.symm',
                input_names='<prefix>.cif',
                rosetta_repr_names='<prefix>_repr.pdb',
                ico_names='<prefix>_ico.cif',
                symmetry_visualization_names='<prefix>_symmetry_visualization.py',
                quality_assurance=True,
            ))
def test_cubic_to_rosetta_I(mock_args):
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
                structures= [str(O.joinpath("1AEW.cif"))], #[str(O.joinpath("3LVX.cif"))],
                symmetry="O",
                overwrite=True,
                rosetta_repr_on=True,
                ico_on=True,
                symmetry_visualization_on=True,
                report_on=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                ico_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names='<prefix>.symm',
                input_names='<prefix>.cif',
                rosetta_repr_names='<prefix>_repr.pdb',
                ico_names='<prefix>_ico.cif',
                symmetry_visualization_names='<prefix>_symmetry_visualization.py',
                quality_assurance=True,
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
                rosetta_repr_on=True,
                ico_on=True,
                symmetry_visualization_on=True,
                report_on=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                ico_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names='<prefix>.symm',
                input_names='<prefix>.cif',
                rosetta_repr_names='<prefix>_repr.pdb',
                ico_names='<prefix>_ico.cif',
                symmetry_visualization_names='<prefix>_symmetry_visualization.py',
                quality_assurance=True,
            ))
def test_cubic_to_rosetta_T(mock_args):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # import pydevd_pycharm
    # port_mapping = [64243, 64241]
    # pydevd_pycharm.settrace(IP, port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    from scripts.cubic_to_rosetta import main
    main()

  # # output names
  #   parser.add_argument('--symmetry_visualization_names', help="Names given to icosahedral files.", default='<prefix>_symmetry_visualization.py', nargs="+", type=str)

# uncomment @pytest.mark.mpi if you dont want to run with mpi
# mpirun command is:
# conda activate shapedesign && mpirun -n 2 python -m pytest -s -k test_construct_mode tests/test_cubic_to_rosetta.py --with-mpi
# @pytest.mark.mpi
@mock.patch('argparse.ArgumentParser.parse_args',
    return_value=argparse.Namespace(
    structures=sorted([str(p) for p in I.glob("*") if p.stem[0] != "."], key=os.path.getsize), # sorted by file size
    overwrite=False,
    rosetta_repr_on=True,
    ico_on=True,
    symmetry_visualization_on=None,
    report_on=True,
    symdef_outpath="outputs",
    input_outpath="outputs",
    rosetta_repr_outpath="outputs",
    ico_outpath="outputs",
    symmetry_visualization_outpath="outputs",
    report_outpath="outputs",
    symdef_names='<prefix>.symm',
    input_names='<prefix>.cif',
    rosetta_repr_names='<prefix>_repr.pdb',
    ico_names='<prefix>_ico.cif',
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
