#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the CubicAssembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""


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
    from symmetryhandler.symmetryhandler import SymmetrySetup
    ca = CubicSymmetricAssembly("inputs/1STM.cif", "I")
    input_name = "outputs/1stm.cif"
    symmetry_name = "outputs/1stm.symm"
    repr_name = "outputs/1stm_repr.pdb"
    ico_name = "outputs/1stm_ico.cif"
    ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=False)
    create_symmetrical_pose(input_name, symmetry_name, repr_name)
    ca.output(ico_name)
    setup = SymmetrySetup()
    setup.read_from_file(symmetry_name)
    setup.print_visualization("outputs/1stm_symm.py")


def test_setup_symmetry_I_idealized():
    # For Icosahedral: 1stm == perfect symmetry, 1jh5 == Not
    # For Octahedral: 1aew == perfect symmetry, XXXX ==j Not
    # For tetrahedral: 6JDD == perfect symmetry, XXXX == Not
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    # from scripts.cubic_to_rosetta import output_symmetry_visualization_script
    #for name, perfect, symm in zip(("1YZV","1PVV", "6JDD", "1AEW", "1JH5", "1STM"), (True, False, True, False, True), ("O", "T")):
    for name, perfect, symm in zip(("5CVZ",), (True, False, True, False, True), ("I", "I")):
        ca = CubicSymmetricAssembly(f"inputs/{name}.cif", symm)
        input_name = f"outputs/{name}.pdb"
        symmetry_name = f"outputs/{name}.symm"
        repr_name = f"outputs/{name}_repr.pdb"
        ico_name = f"outputs/{name}_ico.cif"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat="pdb")
        create_symmetrical_pose(input_name, symmetry_name, repr_name)
        ca.output(ico_name)
        # output_symmetry_visualization_script(symmetry_name, f"{name}_symm.py", "outputs", True)
        # assert ca.intrinsic_perfect_symmetry == perfect, f"{name} should {'NOT' if not perfect else ''} have intrinsic perfect symmetry"
        assert ca.idealized_symmetry
        # generate the idealized structures
        from cubicsym.assemblyparser import AssemblyParser
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_from_rosetta_input():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    for name in ("6JDD", "1AEW", "1STM"):
        input_name = f"outputs/{name}.cif"
        symmetry_name = f"outputs/{name}.symm"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_from_5fold():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    csa = CubicSymmetricAssembly()
    inputf = "inputs/1STM_AFM5.pdb"
    outputf = "outputs/1STM_AFM5_sym.cif"
    symdeff = "outputs/1STM_AFM5_sym.symm"
    cubicf = "outputs/1STM_AFM5_I.cif"
    csa.create_from_5fold(inputf, outputf, symdeff)
    ca_ideal = CubicSymmetricAssembly.from_rosetta_input(outputf, symdeff)
    ca_ideal.output(cubicf)


def test_nonexisting():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    # from scripts.cubic_to_rosetta import output_symmetry_visualization_script
    #for name, perfect, symm in zip(("1YZV","1PVV", "6JDD", "1AEW", "1JH5", "1STM"), (True, False, True, False, True), ("O", "T")):
    for name, perfect, symm in zip(("5CVZ",), (True, False, True, False, True), ("I", "I")):
        ca = CubicSymmetricAssembly(f"inputs/{name}.cif", symm)
        input_name = f"outputs/{name}.pdb"
        symmetry_name = f"outputs/{name}.symm"
        repr_name = f"outputs/{name}_repr.pdb"
        ico_name = f"outputs/{name}_ico.cif"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat="pdb")
        create_symmetrical_pose(input_name, symmetry_name, repr_name)
        ca.output(ico_name)
        # output_symmetry_visualization_script(symmetry_name, f"{name}_symm.py", "outputs", True)
        # assert ca.intrinsic_perfect_symmetry == perfect, f"{name} should {'NOT' if not perfect else ''} have intrinsic perfect symmetry"
        assert ca.idealized_symmetry
        # generate the idealized structures
        from cubicsym.assemblyparser import AssemblyParser
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_foldmap():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    from shapedesign.settings import SYMMETRICAL
    name, symmfolder, foldmap, symmetry = "7M2V", "I", {"hf1": ["1", "2", "3", "4", "11"], "hf2": ["5", "16", "14", "13", "10"], "hf3": ["17", "6", "20", "18", "19"],
              "3": ["10", "19", "1"], "21": ["1", "18"], "22": ["1", "13"]}, "I"
    force_symmetry, rosetta_asym_unit = None, None
    ca = CubicSymmetricAssembly(SYMMETRICAL.joinpath(f"{symmfolder}/unrelaxed/native/{name}.cif"), mmcif_symmetry=symmetry, force_symmetry=force_symmetry, rosetta_asymmetric_units=rosetta_asym_unit)
    input_name = f"outputs/{name}.pdb"
    symmetry_name = f"outputs/{name}.symm"
    repr_name = f"outputs/{name}_repr.pdb"
    ico_name = f"outputs/{name}_ico.cif"
    ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
    ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat="pdb",
                               foldmap=foldmap)
    create_symmetrical_pose(input_name, symmetry_name, repr_name)
    ca.output(ico_name)
    assert ca.idealized_symmetry
    ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
    ca_ideal.output(ico_name_ideal)

def test_afoncubicI():
    from cubicsym.cubicassembly import CubicSymmetricAssembly
    from shapedesign.settings import SYMMETRICAL
    # from scripts.cubic_to_rosetta import output_symmetry_visualization_script
    # to make it fit i have:
    # for 7MV2 I need to specifiy the
    for name, symmfolder, foldmap, symmetry in zip(("6RPO", "6JJA", "6S44", "6ZLO", "7NO0"), ("I", "I", "I", "I", "I"), (None, None, None, None, None), ("I", "I", "I", "I", "I")):
        force_symmetry, rosetta_asym_unit = None, None
        ca = CubicSymmetricAssembly(SYMMETRICAL.joinpath(f"{symmfolder}/unrelaxed/native/{name}.cif"), mmcif_symmetry=symmetry, force_symmetry=force_symmetry, rosetta_asymmetric_units=rosetta_asym_unit)
        input_name = f"outputs/{name}.pdb"
        symmetry_name = f"outputs/{name}.symm"
        repr_name = f"outputs/{name}_repr.pdb"
        ico_name = f"outputs/{name}_ico.cif"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat="pdb",
                                   foldmap=foldmap)
        create_symmetrical_pose(input_name, symmetry_name, repr_name)
        ca.output(ico_name)
        # output_symmetry_visualization_script(symmetry_name, f"{name}_symm.py", "outputs", True)
        # assert ca.intrinsic_perfect_symmetry == perfect, f"{name} should {'NOT' if not perfect else ''} have intrinsic perfect symmetry"
        assert ca.idealized_symmetry
        # generate the idealized structures
        from cubicsym.assemblyparser import AssemblyParser
        # ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        # ca_ideal.output(ico_name_ideal)
