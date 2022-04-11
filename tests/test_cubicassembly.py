#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the CubicAssembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""

def test_init():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    ca = CubicSymmetricAssembly()
    assert True

def create_symmetrical_pose(input_name, symmetry_name, repr_name):
    from pyrosetta import init, pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    init("-symmetry:initialize_rigid_body_dofs true "
         "-detect_disulf false "
         "-out:file:output_pose_energies_table false")
    pose = pose_from_file(str(input_name))
    symmetrize = SetupForSymmetryMover(str(symmetry_name))
    symmetrize.apply(pose)
    pose.dump_pdb(str(repr_name))

def test_setup_symmetry_I():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    from cubicsym.assemblyparser import AssemblyParser
    parser = AssemblyParser()
    ca = parser.cubic_assembly_from_cif("inputs/1STM.cif")
    input_name = "outputs/1stm.cif"
    symmetry_name = "outputs/1stm.symm"
    repr_name = "outputs/1stm_repr.pdb"
    ico_name = "outputs/1stm_ico.cif"
    ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", rmsd_diff=0.5, angles_diff=2.0)
    create_symmetrical_pose(input_name, symmetry_name, repr_name)
    ca.output(ico_name)
    # ca = parser.from_symmetric_output_pdb_and_symmetry_file("outputs/1stm.cif", "outputs/1stm.symm")
    #
    assert True


