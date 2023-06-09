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
from cubicsym.private_paths import I, O, T, IP
from pathlib import Path

def test_quality_assurance():
    from scripts.cubic_to_rosetta import quality_assurance
    from pyrosetta import init
    rosetta_repr = "inputs/1stm_repr.pdb"
    ico_repr = "inputs/1stm_ico.cif"
    input_file = "inputs/1stm.cif"
    symm_file = "inputs/1stm.symm"
    quality_assurance(rosetta_repr, ico_repr, input_file, symm_file, "1stm", 1)

# @mock.patch('argparse.ArgumentParser.parse_args',
#             return_value=argparse.Namespace(
#                 structures=[str(T.joinpath("4AMU.cif"))],
#                 symmetry="T",
#                 overwrite=True,
#                 rosetta_repr=True,
#                 crystal_repr=True,
#                 full_repr=True,
#                 symmetry_visualization=None,
#                 report=True,
#                 symdef_outpath="outputs",
#                 input_outpath="outputs",
#                 rosetta_repr_outpath="outputs",
#                 crystal_repr_outpath="outputs",
#                 full_repr_outpath="outputs",
#                 symmetry_visualization_outpath="outputs",
#                 report_outpath="outputs",
#                 symdef_names=['<prefix>.symm'],
#                 input_names=['<prefix>.cif'],
#                 rosetta_repr_names=['<prefix>_rosetta.pdb'],
#                 crystal_repr_names=['<prefix>_crystal.pdb'],
#                 full_repr_names=['<prefix>_full.cif'],
#                 symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
#                 quality_assurance=True,
#                 idealize=True,
#                 report_names=['<prefix>.csv']
#             ))
# def test_cubic_to_rosetta_T_asym(mock_args):
#     from mpi4py import MPI
#     size = MPI.COMM_WORLD.Get_size()
#     rank = MPI.COMM_WORLD.Get_rank()
#     from mpi4py import MPI
#     # import pydevd_pycharm
#     # port_mapping = [64243, 64241]
#     # pydevd_pycharm.settrace(IP, port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
#     from scripts.cubic_to_rosetta import main
#     main()

def test_crystl_repr():
    from pyrosetta import pose_from_file, init
    from scripts.cubic_to_rosetta import main
    from shapedesign.src.utilities.alignment import tmalign
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    from math import isclose
    init("-mute all")
    sym_files = {"I": ["6S44", "1STM"],
                 "O": ["1AEW", "1P3Y"],
                 "T": ["1MOG", "1H0S"],
                 }
    for k, vv in sym_files.items():
        if k == "I":
            loc = I
        elif k == "O":
            loc = O
        else:
            loc = T
        for v in vv:
            @mock.patch('argparse.ArgumentParser.parse_args',
                        return_value=argparse.Namespace(
                            structures=[str(loc.joinpath(f"{v}.cif"))],
                            symmetry=k,
                            overwrite=True,
                            rosetta_repr=True,
                            crystal_repr=True,
                            full_repr=True,
                            symmetry_visualization=None,
                            report=True,
                            symdef_outpath="outputs",
                            input_outpath="outputs",
                            rosetta_repr_outpath="outputs",
                            crystal_repr_outpath="outputs",
                            full_repr_outpath="outputs",
                            symmetry_visualization_outpath="outputs",
                            report_outpath="outputs",
                            symdef_names=['<prefix>.symm'],
                            input_names=['<prefix>.cif'],
                            rosetta_repr_names=['<prefix>_rosetta.pdb'],
                            crystal_repr_names=['<prefix>_crystal.pdb'],
                            full_repr_names=['<prefix>_full.cif'],
                            symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                            quality_assurance=True,
                            idealize=True,
                            report_names=['<prefix>.csv'],
                            ignore_chains=None,
                            main_id="1"
                        ))
            def test(mock_args):
                main()
            test()
            # align them and assert that chain wise the exactly overlap!
            # # open up the rosetta repr
            # rosetta_repr = pose_from_file(str(loc.parent.parent.joinpath(f"idealized/rosetta_repr/native/{v}_rosetta.pdb")))
            # rosetta_repr.pdb_info().name("rosetta")
            # # open up the output file
            # crystal_repr = pose_from_file(f"outputs/{v}_crystal.pdb")
            # crystal_repr.pdb_info().name("crystal")
            # from simpletestlib.test import setup_test
            # pose, pmm, cmd = setup_test(name=k, file=v, mute=True, return_symmetry_file=False, symmetrize=True)
            # pmm.keep_history(True)
            # pmm.apply(crystal_repr)
            # pmm.apply(rosetta_repr)
            # # assert the chain order is the same
            # for a, b, c in [(crystal_repr.pdb_info().chain(a), rosetta_repr.pdb_info().chain(b), pose.pdb_info().chain(c)) for a, b, c in
            #  zip([i * pose.chain_end(1) for i in range(1, pose.num_chains())], [i * pose.chain_end(1) for i in range(1, pose.num_chains())], [i * pose.chain_end(1) for i in range(1, pose.num_chains())])]:
            #     # print(a,b,c)
            #     assert a == b and b == c, f"The chains {a}, {b}, {c} are not identical "
            #
            # tmscore = tmalign(rosetta_repr, crystal_repr)
            # rmsd = CA_rmsd(crystal_repr, rosetta_repr)
            # print(tmscore, rmsd)
            # assert isclose(tmscore, 1, abs_tol=1e-3)
            # assert isclose(rmsd, 0, abs_tol=1e-3)

            # you can use a map to match onto

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(I.joinpath("6ZLO.cif"))],
                symmetry="I",
                overwrite=True,
                rosetta_repr=True,
                crystal_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                crystal_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                crystal_repr_names=['<prefix>_crystal.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv'],
                ignore_chains=None,
                main_id="1"
            ))
def test_cubic_to_rosetta_I(mock_args):
    from scripts.cubic_to_rosetta import main
    main()

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(I.joinpath("1T0T.cif"))],
                symmetry="I",
                overwrite=True,
                rosetta_repr=True,
                crystal_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=False,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                crystal_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                crystal_repr_names=['<prefix>_crystal.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv'],
                ignore_chains=None,
                main_id="1",
                hf1=None,
                hf2=None,
                hf3=None,
                f3=["1", "14", "29"],
                f21=None,
                f22=None,
                output_generated_structure=False,
            ))
def test_cubic_to_rosetta_I_foldmap(mock_args):
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
    from scripts.cubic_to_rosetta import main
    main()

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(T.joinpath("1H0S.cif"))],
                symmetry="T",
                overwrite=True,
                rosetta_repr=True,
                crystal_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                crystal_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                crystal_repr_names=['<prefix>_crystal.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv'],
                ignore_chains=None,
                main_id="1",
                hf1=None,
                hf2=None,
                hf3=None,
                f3=None,
                f21=None,
                f22=None,
                output_generated_structure=False,
            ))
def test_cubic_to_rosetta_T(mock_args):
    from scripts.cubic_to_rosetta import main
    main()

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(
                structures=[str(T.joinpath("3LEO.cif"))],
                symmetry="T",
                overwrite=True,
                rosetta_repr=True,
                crystal_repr=True,
                full_repr=True,
                symmetry_visualization=None,
                report=True,
                symdef_outpath="outputs",
                input_outpath="outputs",
                rosetta_repr_outpath="outputs",
                crystal_repr_outpath="outputs",
                full_repr_outpath="outputs",
                symmetry_visualization_outpath="outputs",
                report_outpath="outputs",
                symdef_names=['<prefix>.symm'],
                input_names=['<prefix>.cif'],
                rosetta_repr_names=['<prefix>_rosetta.pdb'],
                crystal_repr_names=['<prefix>_crystal.pdb'],
                full_repr_names=['<prefix>_full.cif'],
                symmetry_visualization_names=['<prefix>_symmetry_visualization.py'],
                quality_assurance=True,
                idealize=True,
                report_names=['<prefix>.csv'],
                ignore_chains=None,
                main_id="1",
                hf1=None,
                hf2=None,
                hf3=None,
                f3=["1", "6", "11"],
                f21=None,
                f22=None,
                output_generated_structure=False,
            ))
def test_cubic_to_rosetta_T_foldmap(mock_args):
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
    structures=None, #todo add here and make it work i cant run other test without it so therefore uncommented create_test_list(symmetry="I"),
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
