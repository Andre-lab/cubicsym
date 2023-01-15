#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the CubicAssembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""

def test_assembly():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from cubicsym.assembly.assembly import Assembly
    for name, sym in zip(("3RCO", "6LPE", "4AM5",), ("T", "O", "O", )):
        path = f"/home/shared/databases/SYMMETRICAL/{sym}/unrelaxed/native"
        ca = Assembly(f"{path}/{name}.cif", assembly_id="2")
        ca.output(f"outputs/{name}_test.cif", map_subunit_ids_to_chains=True)
        break

def test_cubicassembly():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from cubicsym.assembly.assembly import Assembly
    for name, sym in zip(("4AM5", "6LPE", "3FVB", "3RCO", "2FKA", "6LPE", "4AM5", ), ("O", "O", "O", "T", "T", "O", "O",  )):
        # "5CY5" har Pseudo Stoichiometry:
        path = f"/home/shared/databases/SYMMETRICAL/{sym}/unrelaxed/native"
        ca = CubicSymmetricAssembly(f"{path}/{name}.cif", sym, ignore_chains=["B"] if name == "2FKA" else None)
        ca.output(f"outputs/{name}_test.cif", map_subunit_ids_to_chains=True)
        # test in future
        input_name = f"outputs/{name}.pdb"
        symmetry_name = f"outputs/{name}.symm"
        repr_name = f"outputs/{name}_repr.pdb"
        ico_name = f"outputs/{name}_ico.cif"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat="pdb")
        create_symmetrical_pose(input_name, symmetry_name, repr_name)
        ca.output(ico_name)
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def create_symmetrical_pose(input_name, symmetry_name, repr_name, mute=True, outformat="cif"):
    from pyrosetta import init, pose_from_file
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    init("-symmetry:initialize_rigid_body_dofs true "
         "-detect_disulf false "
         "-out:file:output_pose_energies_table false "
         f"{'-mute all' if mute else ''}")
    pose = pose_from_file(str(input_name))
    symmetrize = SetupForSymmetryMover(str(symmetry_name))
    symmetrize.apply(pose)
    if outformat == "cif":
        pose.dump_cif(str(repr_name))
    else: # pdb
        pose.dump_pdb(str(repr_name))

def test_setup_symmetry_I():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
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
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
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
        from cubicsym.assembly.assemblyparser import AssemblyParser
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_setup_symmetry_T_idealized():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from cubicsym.assembly.assemblyparser import AssemblyParser
    name = "3RCO"
    path = "/home/shared/databases/SYMMETRICAL/T/unrelaxed/native"
    ca = CubicSymmetricAssembly(f"{path}/{name}.cif", "T")
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
    ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
    ca_ideal.output(ico_name_ideal)

def test_setup_symmetry_O_idealized():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from cubicsym.assembly.assemblyparser import AssemblyParser
    name = "4AM5"
    path = "/home/shared/databases/SYMMETRICAL/O/unrelaxed/native"
    ca = CubicSymmetricAssembly(f"{path}/{name}.cif", "O")
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
    ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
    ca_ideal.output(ico_name_ideal)



def test_from_rosetta_input():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    for name in ("6JDD", "1AEW", "1STM"):
        input_name = f"outputs/{name}.cif"
        symmetry_name = f"outputs/{name}.symm"
        ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_from_5fold():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    csa = CubicSymmetricAssembly()
    inputf = "inputs/1STM_AFM5.pdb"
    outputf = "outputs/1STM_AFM5_sym.cif"
    symdeff = "outputs/1STM_AFM5_sym.symm"
    cubicf = "outputs/1STM_AFM5_I.cif"
    csa.create_from_5fold(inputf, outputf, symdeff)
    ca_ideal = CubicSymmetricAssembly.from_rosetta_input(outputf, symdeff)
    ca_ideal.output(cubicf)

def test_nonexisting():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
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
        from cubicsym.assembly.assemblyparser import AssemblyParser
        ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        ca_ideal.output(ico_name_ideal)

def test_foldmap():
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from shapedesign.settings import SYMMETRICAL
    name, symmfolder, foldmap, symmetry = "7M2V", "I", {"hf1": ["1", "2", "3", "4", "11"], "hf2": ["5", "16", "14", "13", "10"], "hf3": ["17", "6", "20", "18", "19"],
              "3": ["10", "19", "1"], "21": ["1", "18"], "22": ["1", "13"]}, "I"
    force_symmetry, rosetta_asym_unit = None, None
    ca = CubicSymmetricAssembly(SYMMETRICAL.joinpath(f"{symmfolder}/unrelaxed/native/{name}.cif"), mmcif_symmetry=symmetry, force_symmetry=force_symmetry, rosetta_units=rosetta_asym_unit)
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
    from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
    from shapedesign.settings import SYMMETRICAL
    # from scripts.cubic_to_rosetta import output_symmetry_visualization_script
    # to make it fit i have:
    # for 7MV2 I need to specifiy the
    ext = "pdb"
    for name, symmfolder, foldmap, symmetry in zip(("1STM", "6S44",  "6RPO", "6JJA",  "6ZLO", "7NO0"), ("I", "I", "I", "I", "I", "I"), (None, None, None, None, None, None), ("I", "I", "I", "I", "I", "I")):
        force_symmetry, rosetta_asym_unit = None, None
        ca = CubicSymmetricAssembly(SYMMETRICAL.joinpath(f"{symmfolder}/unrelaxed/native/{name}.cif"), mmcif_symmetry=symmetry, force_symmetry=force_symmetry, rosetta_units=rosetta_asym_unit)
        input_name = f"outputs/{name}.{ext}"
        symmetry_name = f"outputs/{name}.symm"
        repr_name = f"outputs/{name}_repr.{ext}"
        ico_name = f"outputs/{name}_ico.cif"
        # ico_name_ideal = f"outputs/{name}_ico_ideal.cif"
        ca.output_rosetta_symmetry(symmetry_name=symmetry_name, input_name=input_name, master_to_use="1", idealize=True, outformat=ext,
                                   foldmap=foldmap)
        create_symmetrical_pose(input_name, symmetry_name, repr_name, outformat=ext)
        ca.output(ico_name)
        # output_symmetry_visualization_script(symmetry_name, f"{name}_symm.py", "outputs", True)
        # assert ca.intrinsic_perfect_symmetry == perfect, f"{name} should {'NOT' if not perfect else ''} have intrinsic perfect symmetry"
        assert ca.idealized_symmetry
        # generate the idealized structures
        from cubicsym.assembly.assemblyparser import AssemblyParser
        # ca_ideal = CubicSymmetricAssembly.from_rosetta_input(input_name, symmetry_name)
        # ca_ideal.output(ico_name_ideal)
