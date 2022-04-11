#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cubic_to_rosetta.py script
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import importlib
import traceback
import requests
import numpy as np
import gzip
from Bio.PDB import Superimposer
from cubicsym.assemblyparser import AssemblyParser
from cubicsym.exceptions import ToHighGeometry, ToHighRMSD
from pathlib import Path
from cubicsym.utilities import mpi_starmap, write_string_to_bytearray
from mpi4py import MPI
from pyrosetta import pose_from_file, init
from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def quality_assurance(rosetta_repr, ico_repr, input_file, symm_file, pdbid, assembly_id, output_alignment=False):
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    from io import StringIO
    from Bio.PDB import Superimposer, MMCIFIO
    from Bio.PDB.vectors import Vector
    from pyrosetta.rosetta.core.conformation.symmetry import residue_center_of_mass

    # Create the ico and rosetta repr structures
    pdb_parser, cif_parser = PDBParser(), MMCIFParser()
    print(f"constructing the rosetta_repr structure from {rosetta_repr}")
    rosetta_repr_struct = pdb_parser.get_structure(pdbid + "_rosetta", rosetta_repr)
    print(f"constructing the ico_repr structure from {ico_repr}")
    ico_repr_struct = cif_parser.get_structure(pdbid + "_ico", ico_repr)

    # TEST 1: Assert we have the capsid at the center and in the same time assure that it is symemtrical. If symmetrical all vectors should
    # add up to zero as well. Theres bound to be some inaccuracies that come from:
    #   1. The cif file stores 3 decimals but get_vector() stores 2.
    #   2. The original RCSB asymmetric unit can give rise to inaccuracies in the vectors z15, z25 and z35 (vectors to the fivefolds), so
    #      these are not placed exactly symmetrical
    allowed = 3
    tot = Vector(0,0,0)
    for v in [a.get_vector() for a in ico_repr_struct.get_atoms()]:
        tot += v
    assert all(abs(i) < allowed for i in tot)

    # TEST 2: Assert that the master protein lies along the x-axis with the Rosetta residue COM CA atom on this axis.
    init("-symmetry:initialize_rigid_body_dofs true -detect_disulf false -out:file:output_pose_energies_table false")
    pose = pose_from_file(input_file)
    # resi = residue_center_of_mass(pose.conformation())
    resi = residue_center_of_mass(pose.conformation(), 1, pose.conformation().size())
    # [a for a in rosetta_repr_struct.get_atoms() if a.name == "CA" and a.parent.parent.id == "A"]
    SetupForSymmetryMover(symm_file).apply(pose)
    assert np.isclose(pose.residue(resi).atom("CA").xyz()[1], 0, atol=1e-3), f"{rosetta_repr} does not have COM along the x-axis"

    # TEST 3: Check that the rmsd of the same chains between the rosetta_repr and ico_repr structure is small
    # Allow a distance difference of 0.01 per ca atom
    allowed = 0.01
    rosetta_repr_chains = [chain.id for chain in rosetta_repr_struct.get_chains()]
    rosetta_repr_atoms = {c.id: [a for a in c.get_atoms() if a.name == "CA"] for c in rosetta_repr_struct.get_chains()}
    ico_repr_atoms = {c.id: [a for a in c.get_atoms() if a.name == "CA"] for c in ico_repr_struct.get_chains() if c.id in rosetta_repr_chains}
    for chain in rosetta_repr_atoms:
        # check same length
        str_id = f"chain {chain} of rosetta_repr ({rosetta_repr}) and ico_repr ({ico_repr})"
        assert len(rosetta_repr_atoms[chain]) == len(ico_repr_atoms[chain]), f"{str_id} have unequal length"
        diff = 0
        for atom_repr, atom_ico in zip(rosetta_repr_atoms[chain], ico_repr_atoms[chain]):
            diff += (atom_repr.get_vector() - atom_ico.get_vector()).norm()
        diff /= len(rosetta_repr_atoms[chain])
        assert diff <= allowed, f"The allowed distance differential {allowed} was crossed with {diff} for {str_id}"

    # TEST 4: The capsid is identical to the assembly deposited in the pdb
    url = "https://files.rcsb.org/pub/pdb/data/biounit/{}/all/"
    connect_timeout, read_timeout = 5, 10
    print(f"Retrieving assembly {assembly_id} for {pdbid}.")
    try: # first try pdb
        filename = f"{pdbid}.pdb{assembly_id}.gz".lower()
        response = requests.get(url.format("PDB") + f"/{filename}")
        if response.status_code == 200:
            r = gzip.decompress(response.content).decode()
            rcsb_struct = pdb_parser.get_structure(pdbid, StringIO(r))
        else: # try cif if pdb is not found
            filename = f"{pdbid}-assembly{assembly_id}.cif.gz"
            response = requests.get(url.format("mmCIF") + f"/{filename}")
            r = gzip.decompress(response.content).decode()
            rcsb_struct = cif_parser.get_structure(pdbid, StringIO(r))
            if not response.status_code == 200:
                response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise SystemExit(e.strerror + f"\nCheck if the website is reachable...")
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    assert rcsb_struct, "rcsb file was not created."
    # align to rosetta_repr to the biological assembly
    rcsb_struct_atoms = [a for a in rcsb_struct.get_atoms() if a.name == "CA"]
    # rosetta_repr_atoms = [a for a in rosetta_repr_struct.get_atoms() if a.name == "CA"]
    ico_repr_atoms = [a for a in ico_repr_struct.get_atoms() if a.name == "CA"]
    # alignment
    super_imposer = Superimposer()
    super_imposer.set_atoms(rcsb_struct_atoms, ico_repr_atoms)
    super_imposer.apply(ico_repr_struct.get_atoms())
    # output_the_aligned_structure + the rcsb_structrure
    if output_alignment:
        parent = Path(ico_repr).parent
        io = MMCIFIO()
        io.set_structure(ico_repr_struct)
        io.save(str(parent.joinpath(Path(ico_repr).stem + "_aligned.cif")))
        io.set_structure(rcsb_struct)
        io.save(str(parent.joinpath(pdbid + "_aligned_reference.cif")))


    # Ultimate test:
    # 1. Assert that you get the output (pdb and symm) and no errors occurs.
    # 2. Output the capsid and the rosetta_repr (write code for that) with a flag. Get the biological assembly (with the same number)
    #    that was used to generate the assembly and then check that the alignment of the capsid and the rosetta_repr is good!
    # 3. Make sure the capsid is pointing in the correct direction and is centered. So measure the center of the capsid -> should be [0,0,0] and
    # measure the x-direction!

def output_rosetta_repr(symmetry_out_name, input_out_name, rosetta_repr_name, rosetta_repr_outpath, overwrite):
    """Outputs the Rosetta representation pdb."""
    # check first if we can do it.
    rosetta_out_name = Path(rosetta_repr_outpath).joinpath(rosetta_repr_name)
    if rosetta_out_name.exists() and not overwrite:
        print(f"Skips making {rosetta_out_name} because this file already exist. "
              f"Pass --overwrite to overwrite files.")
        return
    if not importlib.util.find_spec("pyrosetta"):
        print(f"To ouput the Rosetta representation of {rosetta_out_name}, pyrosetta needs to be installed.")
        return

    init("-symmetry:initialize_rigid_body_dofs true "
         "-detect_disulf false "
         "-out:file:output_pose_energies_table false")
    pose = pose_from_file(str(input_out_name))
    symmetrize = SetupForSymmetryMover(str(symmetry_out_name))
    symmetrize.apply(pose)
    pose.dump_pdb(str(rosetta_out_name))

def output_symmetry_visualization_script(symdef_out_name, symmetry_visualization_name, symmetry_visualization_outpath, overwrite):
    """Outputs the symmetry_visualization script to be run in PyMOL."""
    # check first that we can do it
    script_out_name = Path(symmetry_visualization_outpath).joinpath(symmetry_visualization_name)
    if script_out_name.exists() and not overwrite:
        print(f"Skips making {script_out_name} because this file already exist. "
              f"Pass --overwrite to overwrite files.")
        return
    if not importlib.util.find_spec("symmetryhandler"):
        print(f"To ouput the Rosetta representation of {script_out_name} symmetryhandler needs to be installed.")
        return

    # now do it
    from symmetryhandler.symmetryhandler import SymmetrySetup
    setup = SymmetrySetup()
    setup.read_from_file(str(symdef_out_name))
    setup.print_visualization(str(script_out_name))

def make_capsid_symmetry(structure, symmetry, overwrite, symdef_name, symdef_outpath, input_name, input_outpath,
                         rosetta_repr_on, rosetta_repr_name, rosetta_repr_outpath, ico_on, ico_name, ico_outpath,
                         symmetry_visualization_on, symmetry_visualization_name, symmetry_visualization_outpath, quality_assurance_on):
    """Makes a capid symdef and Rosetta input file and optionally the icosahedral structure, the Rosetta representation
    structure or a symmetry visualization script."""

    print(f"Constructs symmetry for {structure} ")

    # preprocess names
    stem = Path(structure).stem
    symdef_name = symdef_name if symdef_name != "<prefix>.symm" else f"{stem}.symm"
    input_name = input_name if input_name != "<prefix>.cif" else f"{stem}.cif"
    rosetta_repr_name = rosetta_repr_name if rosetta_repr_name != "<prefix>_repr.pdb" else f"{stem}_repr.pdb"
    ico_name = ico_name if ico_name != "<prefix>_ico.cif" else f"{stem}_ico.cif"
    symmetry_visualization_name = symmetry_visualization_name if symmetry_visualization_name != "<prefix>_symmetry_visualization.py" else f"{stem}_symmetry_visualization.py"

    succeded = True
    try:
        parser = AssemblyParser()
        ico = parser.cubic_assembly_from_cif(structure, symmetry)
        try:
            # setup symmetry and output the symdef file as well as the output file.
            symdef_out_name = Path(symdef_outpath).joinpath(symdef_name)
            input_out_name = Path(input_outpath).joinpath(input_name)
            if (symdef_out_name.exists() and input_out_name.exists()) and not overwrite:
                print(f"Skips making {symdef_out_name} and {input_out_name} because these files already exist. "
                      f"Pass --overwrite to overwrite files.")
            else:
                ico.output_rosetta_symmetry(str(symdef_out_name), str(input_out_name))
            # output the rosetta representation if set
            if rosetta_repr_on:
                output_rosetta_repr(symdef_out_name, input_out_name, rosetta_repr_name, rosetta_repr_outpath, overwrite)
            # output the icosahedral structure if set.
            ico_name_temp = Path(ico_outpath).joinpath(ico_name)
            if ico_on:
                if ico_name_temp.exists() and not overwrite:
                    print(f"Skips making {ico_name_temp} because this file already exist. "
                          f"Pass --overwrite to overwrite files.")
                else:
                    ico.output(str(ico_name_temp))
            if symmetry_visualization_on:
                output_symmetry_visualization_script(symdef_out_name, symmetry_visualization_name, symmetry_visualization_outpath, overwrite)
        except ToHighRMSD:
            succeded = False
        except ToHighGeometry:
            # TODO: Deal with this in the future, now it just fails.
            traceback.print_exc()
            print(f"Failed with structure: {input_name}")
            comm.Abort()
            return
    except:
        traceback.print_exc()
        print(f"Failed with structure: {input_name}")
        comm.Abort()

    # prefixed set:
    full_input_outpath = str(Path(input_outpath).joinpath(input_name))# + ".cif"))
    full_symdef_outpath = str(Path(symdef_outpath).joinpath(symdef_name))# + ".symm"))
    full_rosetta_repr_outpath = str(Path(rosetta_repr_outpath).joinpath(rosetta_repr_name))# + ".pdb"))
    full_ico_outpath = str(Path(ico_outpath).joinpath(ico_name))# + ".cif"))
    full_symmetry_visualization_outpath = str(Path(symmetry_visualization_outpath).joinpath(symmetry_visualization_name))

    if quality_assurance_on:
        print("Running quality assurance checks")
        quality_assurance(full_rosetta_repr_outpath, full_ico_outpath, full_input_outpath, full_symdef_outpath, pdbid=stem, assembly_id=1)

    return [succeded, Path(input_name).stem, full_input_outpath, full_symdef_outpath, full_rosetta_repr_outpath, full_ico_outpath, full_symmetry_visualization_outpath, ico]

def submain(structures, symmetry, overwrite, symdef_names, symdef_outpath, input_names, input_outpath,
            rosetta_repr_on, rosetta_repr_names, rosetta_repr_outpath,
            ico_on, ico_names, ico_outpath,
            symmetry_visualization_on, symmetry_visualization_names, symmetry_visualization_outpath,
            report_on, report_output_path, quality_assurance):

    # process the names here -> make them lists if they arent!
    if type(symdef_names) != list:
        symdef_names = [symdef_names]
    if type(input_names) != list:
        input_names = [input_names]
    if type(rosetta_repr_names) != list:
        rosetta_repr_names = [rosetta_repr_names]
    if type(ico_names) != list:
        ico_names = [ico_names]
    if type(symmetry_visualization_names) != list:
        symmetry_visualization_names = [symmetry_visualization_names]

    arguments = (
        structures, [symmetry], [overwrite],
        symdef_names, [symdef_outpath],
        input_names, [input_outpath],
        [rosetta_repr_on], rosetta_repr_names, [rosetta_repr_outpath],
        [ico_on], ico_names, [ico_outpath],
        [symmetry_visualization_on], symmetry_visualization_names, [symmetry_visualization_outpath], [quality_assurance])
        #[report_on], [report_output_path])

    results = mpi_starmap(make_capsid_symmetry, comm, *arguments)

    # report stuff
    if report_on:
        report_output_path = str(Path(report_output_path).joinpath("metadata.csv"))
        if comm.Get_rank() == 0:
            print("Making a report")
            if overwrite:
                Path(report_output_path).open("w")
            if overwrite or not Path(report_output_path).exists():
                with Path(report_output_path).open("w") as f:
                    header = ",".join(["stem", "succeded","chains", "residues", "size(Å)",
                              "lowest_3fold_rmsd", "highest_3fold_accepted_rmsd", "lowest_5fold_rmsd", "highest_5fold_accepted_rmsd",
                              "input_paths", "symmdef_paths", "rosetta_repr_paths", "ico_paths", "symmetry_visualization_paths"])
                    f.write(header + "\n")
            comm.Barrier()
        else:
            comm.Barrier()
        amode = MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND
        fh = MPI.File.Open(comm, str(report_output_path), amode)
        for succeded, stem, input_path, symddef_path, rosetta_repr_path, ico_path, symmetry_visualization_path, ico in results:
            chains = ico.get_n_chains()
            residues = ico.get_n_residues()
            size = ico.get_size()
            lowest_3fold_rmsd = round(ico.lowest_3fold_rmsd, 5) if ico.lowest_3fold_rmsd else None
            highest_3fold_accepted_rmsd = round(ico.highest_3fold_accepted_rmsd, 5) if ico.highest_3fold_accepted_rmsd else None
            lowest_5fold_rmsd = round(ico.lowest_5fold_rmsd, 5) if ico.lowest_5fold_rmsd else None
            highest_5fold_accepted_rmsd = round(ico.highest_5fold_accepted_rmsd, 5) if ico.highest_5fold_accepted_rmsd else None
            line = map(str, [stem, succeded, chains, residues, size, lowest_3fold_rmsd, highest_3fold_accepted_rmsd,
                    lowest_5fold_rmsd, highest_5fold_accepted_rmsd, input_path, symddef_path, rosetta_repr_path,
                    ico_path, symmetry_visualization_path])
            fh.Write_shared(write_string_to_bytearray(",".join(line) + "\n"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="From a mmcif file containing cubic symmetry information (I/O/T) this script makes a cubic symdef file, "
                                                 "Rosetta input file, and if set, the full cubic representation and the Rosetta "
                                                 "representaion from the protein in the mmcif file(s)")
    # Input structures
    parser.add_argument('--structures', help="mmcif files to read.", nargs="+", type=str)
    parser.add_argument('--symmetry', help="Symmetry to generate. Either use 'I', 'O', 'T' or the assembly id number to generate the symmetry from. "
                                           "If 'I', 'O' or 'T' is used the script will iterate through each available assembly, check its symmetry,"
                                           "and return the first instance of the assembly with the corresponding symmetry. If a number is used instead the script will "
                                           "attempt to generate whatever cubic symmetrical structure it corresponds to (if possible).", type=str)
    # overwrite
    parser.add_argument('--overwrite', help="To overwrite the files (and if set, the report), or not", action="store_true")
    # on/off for output
    parser.add_argument('--rosetta_repr_on', help="Output the structure that is represented with the symmetryfile.", action="store_true")
    parser.add_argument('--ico_on', help="Output the corresponding cubic structure.", action="store_true")
    parser.add_argument('--symmetry_visualization_on', help="ouputs a symmetry visualization script that can be used in pymol.", action="store_true")
    parser.add_argument('--report_on', help="Output a report file that reports symmetry information", action="store_true")
    # output paths
    parser.add_argument('--symdef_outpath', help="Path to the directory of where to output the symdef files.", default=".", type=str)
    parser.add_argument('--input_outpath', help="Path to the directory of where to output the Rosetta input pdb files.", default=".", type=str)
    parser.add_argument('--rosetta_repr_outpath', help="Path to the directory of where to output files set with '--rosetta_repr_on' ", default=".", type=str)
    parser.add_argument('--ico_outpath', help="Path to the directory of where to output files set with '--ico_on'", default=".", type=str)
    parser.add_argument('--symmetry_visualization_outpath', help="Path to the directory of where to output files set with '--symmetry_visualization_on'", default=".", type=str)
    parser.add_argument('--report_outpath', help="Path to the directory of where to output the report.", default=".", type=str)
    # output names
    parser.add_argument('--symdef_names', help="Names given to the symmetry files.", default='<prefix>.symm', nargs="+", type=str)
    parser.add_argument('--input_names', help="Name given to input files.", default='<prefix>.cif', nargs="+", type=str)
    parser.add_argument('--rosetta_repr_names', help="Names given to icosahedral files.", default='<prefix>_repr.pdb', nargs="+", type=str)
    parser.add_argument('--ico_names', help="Names given to icosahedral files.", default='<prefix>_ico.cif', nargs="+", type=str)
    parser.add_argument('--symmetry_visualization_names', help="Names given to icosahedral files.", default='<prefix>_symmetry_visualization.py', nargs="+", type=str)
    # quality assurance
    parser.add_argument('--quality_assurance', help="Will run a quality assurance check to see that the outputs of the script is in "
                                                    "accordance with the actual cubic structure present in the RCSB along with other things."
                                                    " To run this, internet access is needed and the stem of the filename (like 1stm in 1stm.cif)"
                                                    " needs to present in the RCSB.", action="store_true")
    # parse the arguments.
    args = parser.parse_args()

    submain(args.structures, args.symmetry, args.overwrite,
            args.symdef_names, args.symdef_outpath,
            args.input_names, args.input_outpath,
            args.rosetta_repr_on, args.rosetta_repr_names, args.rosetta_repr_outpath,
            args.ico_on, args.ico_names, args.ico_outpath,
            args.symmetry_visualization_on, args.symmetry_visualization_names, args.symmetry_visualization_outpath,
            args.report_on, args.report_outpath, args.quality_assurance)

if __name__ == '__main__':
    main()