#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cubic_to_rosetta.py script
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import importlib
import requests
import numpy as np
import gzip
from cubicsym.assembly.cubicassembly import CubicSymmetricAssembly
from cubicsym.exceptions import ToHighGeometry, ToHighRMSD
from pathlib import Path
from cubicsym.utilities import mpi_starmap, write_string_to_bytearray
from mpi4py import MPI
from pyrosetta import pose_from_file, init
from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
from symmetryhandler.symmetrysetup import SymmetrySetup
import pandas as pd

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def quality_assurance(rosetta_repr, full_repr, input_file, symm_file, pdbid, assembly_id, idealize, output_alignment=False):
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    from io import StringIO
    from Bio.PDB import Superimposer, MMCIFIO
    from Bio.PDB.vectors import Vector
    from pyrosetta.rosetta.core.conformation.symmetry import residue_center_of_mass

    # Create the full and rosetta repr structures
    pdb_parser, cif_parser = PDBParser(), MMCIFParser()
    print(f"constructing the rosetta_repr structure from {rosetta_repr}")
    rosetta_repr_struct = pdb_parser.get_structure(pdbid + "_rosetta", rosetta_repr)
    print(f"constructing the full_repr structure from {full_repr}")
    full_repr_struct = cif_parser.get_structure(pdbid + "_full", full_repr)

    # TEST 1: Assert we have the capsid at the center and in the same time assure that it is symmetrical. If symmetrical all vectors should
    # add up to zero as well. There's bound to be some inaccuracies that come from:
    #   1. The cif file stores 3 decimals but get_vector() stores 2.
    #   2. The original RCSB asymmetric unit can give rise to inaccuracies in the vectors z15, z25 and z35 (vectors to the fivefolds), so
    #      these are not placed exactly symmetrical
    allowed = 1
    tot = Vector(0,0,0)
    atoms = [a.get_vector() for a in full_repr_struct.get_atoms() if a.name == "CA"]
    for v in atoms:
        tot += v
    tot = tot / len(atoms)
    assert all(abs(i) < allowed for i in tot / len(atoms)), f"The full representation is not centered, the center is: {tot}"

    # TEST 2: Assert that the master protein lies along the x-axis with the Rosetta residue COM CA atom on this axis.
    init("-symmetry:initialize_rigid_body_dofs true -detect_disulf false -out:file:output_pose_energies_table false")
    pose = pose_from_file(input_file)
    # resi = residue_center_of_mass(pose.conformation())
    resi = residue_center_of_mass(pose.conformation(), 1, pose.conformation().size())
    # [a for a in rosetta_repr_struct.get_atoms() if a.name == "CA" and a.parent.parent.id == "A"]
    SetupForSymmetryMover(symm_file).apply(pose)
    assert np.isclose(pose.residue(resi).atom("CA").xyz()[1], 0, atol=1e-3), f"{rosetta_repr} does not have COM along the x-axis"

    # TEST 3: Check that the rmsd of the same chains between the rosetta_repr and full_repr structure is small
    # Allow a distance difference of 0.01 per ca atom
    allowed = 1
    rosetta_repr_chains = [chain.id for chain in rosetta_repr_struct.get_chains()]
    rosetta_repr_atoms = {c.id: [a for a in c.get_atoms() if a.name == "CA"] for c in rosetta_repr_struct.get_chains()}
    full_repr_atoms = {c.id: [a for a in c.get_atoms() if a.name == "CA"] for c in full_repr_struct.get_chains() if c.id in rosetta_repr_chains}
    for chain in rosetta_repr_atoms:
        # check same length
        str_id = f"chain {chain} of rosetta_repr ({rosetta_repr}) and full_repr ({full_repr})"
        assert len(rosetta_repr_atoms[chain]) == len(full_repr_atoms[chain]), f"{str_id} have unequal length"
        diff = 0
        for atom_repr, atom_full in zip(rosetta_repr_atoms[chain], full_repr_atoms[chain]):
            diff += (atom_repr.get_vector() - atom_full.get_vector()).norm()
        diff /= len(rosetta_repr_atoms[chain])
        assert diff <= allowed, f"The allowed distance differential {allowed} was crossed with {diff} for {str_id}"

    # TEST 4: The capsid is identical to the assembly deposited in the pdb. This should not be done if the symmetry was idealized
    # since there in that case could be discrepancies
    if not idealize:
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
        full_repr_atoms = [a for a in full_repr_struct.get_atoms() if a.name == "CA"]
        # alignment
        super_imposer = Superimposer()
        super_imposer.set_atoms(rcsb_struct_atoms, full_repr_atoms)
        super_imposer.apply(full_repr_struct.get_atoms())
        # output_the_aligned_structure + the rcsb_structrure
        if output_alignment:
            parent = Path(full_repr).parent
            io = MMCIFIO()
            io.set_structure(full_repr_struct)
            io.save(str(parent.joinpath(Path(full_repr).stem + "_aligned.cif")))
            io.set_structure(rcsb_struct)
            io.save(str(parent.joinpath(pdbid + "_aligned_reference.cif")))

def make_cubic_symmetry(structures, symmetry, overwrite, symdef_names, symdef_outpath, input_names, input_outpath,
                        rosetta_repr, rosetta_repr_names, rosetta_repr_outpath, crystal_repr, crystal_repr_names, crystal_repr_outpath,
                        full_repr, full_repr_names, full_repr_outpath,
                        symmetry_visualization, symmetry_visualization_names, symmetry_visualization_outpath, quality_assurance_on,
                        idealize, report, report_outpath, report_names):
    """Makes a cubic symdef and Rosetta input file and optionally the full cubic structure, the Rosetta representation
    structure and a symmetry visualization script."""
    for structure, symdef_name, input_name, rosetta_repr_name, crystal_repr_name, full_repr_name, symvis_name, report_name in zip(structures, symdef_names, input_names,
                                                                        rosetta_repr_names, crystal_repr_names, full_repr_names, symmetry_visualization_names,
                                                                                                               report_names):
        print(f"Constructs symmetry for {structure} ")
        results = {"succeded": False,
                   "failure_reason": "",
                   "file": structure,
                   "symdef_name": Path(symdef_outpath).joinpath(symdef_name),
                   "input_name": Path(input_outpath).joinpath(input_name),
                   "rosetta_repr_name":  Path(rosetta_repr_outpath).joinpath(rosetta_repr_name),
                   "crystal_repr_name":  Path(crystal_repr_outpath).joinpath(crystal_repr_name),
                   "full_repr_name": Path(full_repr_outpath).joinpath(full_repr_name),
                   "symvis_name": Path(symmetry_visualization_outpath).joinpath(symvis_name),
                   "quality_assurance_checked": quality_assurance_on,
                   "quality_error": "",
                   "quality_ok": False,
                   "idealized": idealize,
                   "chains": None,
                   "residues": None,
                   "size": None}
        try:
            full = CubicSymmetricAssembly(mmcif_file=structure, mmcif_symmetry=symmetry)
            # Make the symmetry file and input file
            if (results.get("symdef_name").exists() and results.get("input_name").exists()) and not overwrite:
                print(f"Skips making {results.get('symdef_name')} and {results.get('input_name')} because these files already exists. Pass --overwrite to overwrite files.")
            else:
                selected_ids = full.output_rosetta_symmetry(str(results.get("symdef_name")), str(results.get("input_name")), idealize=idealize)
            # Make the Rosetta repr file
            if crystal_repr:
                if results.get("crystal_repr_name").exists() and not overwrite:
                    print(f"Skips making {results.get('crystal_repr_name')} because this file already exist. Pass --overwrite to overwrite files.")
                else:
                    full.output(filename=str(results.get("crystal_repr_name")), ids=selected_ids, format="pdb", map_chains_to_ids_in_order=True)
            if rosetta_repr:
                if results.get("rosetta_repr_name").exists() and not overwrite:
                    print(f"Skips making {results.get('rosetta_repr_name')} because this file already exist. Pass --overwrite to overwrite files.")
                else:
                    init("-symmetry:initialize_rigid_body_dofs true -detect_disulf false -out:file:output_pose_energies_table false")
                    pose = pose_from_file(str(results.get("input_name")))
                    symmetrize = SetupForSymmetryMover(str(results.get("symdef_name")))
                    symmetrize.apply(pose)
                    pose.dump_pdb(str(results.get("rosetta_repr_name")))
            # Make the full repr file
            if full_repr:
                if results.get("full_repr_name").exists() and not overwrite:
                    print(f"Skips making {results.get('full_repr_name')} because this file already exist. Pass --overwrite to overwrite files.")
                else:
                    full.output(str(results.get("full_repr_name")))
            # Make the visualization script
            if symmetry_visualization:
                if results.get("symvis_name").exists() and not overwrite:
                    print(
                        f"Skips making {results.get('symvis_name')} because this file already exist. Pass --overwrite to overwrite files.")
                else:
                    setup = SymmetrySetup()
                    setup.read_from_file(str(results.get("symdef_name")))
                    setup.print_visualization(str(results.get("symvis_name")))
            results["succeded"] = True
        except ToHighRMSD as e:
            results["failure_reason"] = e.message
        except ToHighGeometry as e:
            results["failure_reason"] = e.message
        except Exception as e:
            results["failure_reason"] = e
        if results.get("succeded"):
            if quality_assurance_on:
                print("Running quality assurance checks")
                pdbid = Path(results.get("input_name")).stem
                try:
                    quality_assurance(str(results.get("rosetta_repr_name")), str(results.get("full_repr_name")), str(results.get("input_name")),
                                      str(results.get("symdef_name")), pdbid=pdbid, assembly_id=1, idealize=idealize)
                    results["quality_ok"] = True
                except AssertionError as e:
                    results["quality_error"] = e
            results["chains"] = full.get_n_chains()
            results["residues"] = full.get_n_residues()
            results["size"] = full.get_size()
        if report:
            pd.DataFrame({k: [v] for k, v in results.items()}).to_csv(Path(report_outpath).joinpath(report_name), index=False)
        else:
            print(f"Structure {'succeded' if results.get('succeded') else 'failed'} with the following results:")
            for k, v in results.items():
                print(f"{k}:", "v")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="From a mmcif file containing cubic symmetry information (I/O/T) this script makes a cubic symdef file, "
                                                 "Rosetta input file, and if set, the full cubic representation and the Rosetta "
                                                 "representaion from the protein in the mmcif file(s)")
    # input structures
    parser.add_argument('--structures', help="mmcif files to read.", nargs="+", type=str)
    parser.add_argument('--symmetry', help="Symmetry to generate. Either use 'I', 'O', 'T' or the assembly id number to generate the symmetry from. "
                                           "If 'I', 'O' or 'T' is used the script will iterate through each available assembly, check its symmetry,"
                                           "and return the first instance of the assembly with the corresponding symmetry. If a number is used instead the script will "
                                           "attempt to generate whatever cubic symmetrical structure it corresponds to (if possible).", type=str)
    # overwrite
    parser.add_argument('--overwrite', help="To overwrite the files (and if set, the report), or not", action="store_true")
    # on/off for output
    parser.add_argument('--rosetta_repr', help="Output the structure that is represented with the symmetryfile.", default=False, type=bool)
    parser.add_argument('--crystal_repr', help="Output the crystal structure containing only the chains present in the symmetric pose."
                                               "This can be used for RMSD calcuations.", default=False, type=bool)
    parser.add_argument('--full_repr', help="Output the corresponding cubic structure.", default=False, type=bool)
    parser.add_argument('--symmetry_visualization', help="ouputs a symmetry visualization script that can be used in pymol.",  default=False, type=bool)
    parser.add_argument('--report', help="Output a report file that reports symmetry information", default=False, type=bool)
    # output paths
    parser.add_argument('--symdef_outpath', help="Path to the directory of where to output the symdef files.", default=".", type=str)
    parser.add_argument('--input_outpath', help="Path to the directory of where to output the Rosetta input pdb files.", default=".", type=str)
    parser.add_argument('--rosetta_repr_outpath', help="Path to the directory of where to output files set with '--rosetta_repr' ", default=".", type=str)
    parser.add_argument('--crystal_repr_outpath', help="Path to the directory of where to output files set with '--crystal_repr' ", default=".", type=str)
    parser.add_argument('--full_repr_outpath', help="Path to the directory of where to output files set with '--full_repr'", default=".", type=str)
    parser.add_argument('--symmetry_visualization_outpath', help="Path to the directory of where to output files set with '--symmetry_visualization'", default=".", type=str)
    parser.add_argument('--report_outpath', help="Path to the directory of where to output the report.", default=".", type=str)
    # output names
    parser.add_argument('--symdef_names', help="Names given to the symmetry files.", default='<prefix>.symm', nargs="+", type=str)
    parser.add_argument('--input_names', help="Name given to input files.", default='<prefix>.cif', nargs="+", type=str)
    parser.add_argument('--rosetta_repr_names', help="Names given to Rosetta representation files.", default='<prefix>_rosetta.pdb', nargs="+", type=str)
    parser.add_argument('--crystal_repr_names', help="Names given to crystal representation files.", default='<prefix>_crystal.pdb', nargs="+", type=str)
    parser.add_argument('--full_repr_names', help="Names given to full representation files.", default='<prefix>_full.cif', nargs="+", type=str)
    parser.add_argument('--symmetry_visualization_names', help="Names given to symmetry visualization files.", default='<prefix>_symmetry_visualization.py', nargs="+", type=str)
    parser.add_argument('--report_names', help="Names given to report files.", default='<prefix>.csv', nargs="+", type=str)
    # other options
    parser.add_argument('--quality_assurance', help="Will run a quality assurance check to see that the outputs of the script is in "
                                                    "accordance with the actual cubic structure present in the RCSB along with other things."
                                                    " To run this, internet access is needed and the stem of the filename (like 1stm in 1stm.cif)"
                                                    " needs to present in the RCSB.", default=False, type=bool)
    parser.add_argument('--idealize', help="To idealize the symmetry", default=True, type=bool)
    # parse the arguments.
    args = parser.parse_args()

    # default name handling
    if rank == 0:
        args.symdef_names = [n if n != "<prefix>.symm" else f"{Path(s).stem}.symm" for n, s in zip([args.symdef_names] * len(args.structures) if isinstance(args.symdef_names, str) else args.symdef_names, args.structures)]
        args.input_names = [n if n != "<prefix>.cif" else f"{Path(s).stem}.cif" for n, s in zip([args.input_names] * len(args.structures) if isinstance(args.input_names, str) else args.input_names, args.structures)]
        args.rosetta_repr_names = [n if n != "<prefix>_rosetta.pdb" else f"{Path(s).stem}_rosetta.pdb" for n, s in zip([args.rosetta_repr_names] * len(args.structures) if isinstance(args.rosetta_repr_names, str) else args.rosetta_repr_names, args.structures)]
        args.crystal_repr_names = [n if n != "<prefix>_crystal.pdb" else f"{Path(s).stem}_crystal.pdb" for n, s in zip([args.crystal_repr_names] * len(args.structures) if isinstance(args.crystal_repr_names, str) else args.crystal_repr_names, args.structures)]
        args.full_repr_names = [n if n != "<prefix>_full.cif" else f"{Path(s).stem}_full.cif" for n, s in zip([args.full_repr_names] * len(args.structures) if isinstance(args.full_repr_names, str) else args.full_repr_names, args.structures)]
        args.symmetry_visualization_names = [n if n != '<prefix>_symmetry_visualization.py' else f"{Path(s).stem}_symmetry_visualization.py" for n, s in zip([args.symmetry_visualization_names] * len(args.structures) if isinstance(args.symmetry_visualization_names, str) else args.symmetry_visualization_names, args.structures)]
        args.report_names = [n if n != "<prefix>.csv" else f"{Path(s).stem}.csv" for n, s in zip([args.report_names] * len(args.structures) if isinstance(args.report_names, str) else args.report_names, args.structures)]

    # divide the options
    if rank == 0:
        args.structures = np.array_split(args.structures, size)
        args.symdef_names = np.array_split(args.symdef_names, size)
        args.input_names = np.array_split(args.input_names, size)
        args.rosetta_repr_names = np.array_split(args.rosetta_repr_names, size)
        args.crystal_repr_names = np.array_split(args.crystal_repr_names, size)
        args.full_repr_names = np.array_split(args.full_repr_names, size)
        args.symmetry_visualization_names = np.array_split(args.symmetry_visualization_names, size)
        args.report_names = np.array_split(args.report_names, size)
    args.structures = comm.scatter(args.structures, root=0)
    args.symdef_names = comm.scatter(args.symdef_names, root=0)
    args.input_names = comm.scatter(args.input_names, root=0)
    args.rosetta_repr_names = comm.scatter(args.rosetta_repr_names, root=0)
    args.crystal_repr_names = comm.scatter(args.crystal_repr_names, root=0)
    args.full_repr_names = comm.scatter(args.full_repr_names, root=0)
    args.symmetry_visualization_names = comm.scatter(args.symmetry_visualization_names, root=0)
    args.report_names = comm.scatter(args.report_names, root=0)

    make_cubic_symmetry(args.structures, args.symmetry, args.overwrite, args.symdef_names, args.symdef_outpath, args.input_names, args.input_outpath,
                        args.rosetta_repr, args.rosetta_repr_names, args.rosetta_repr_outpath, args.crystal_repr, args.crystal_repr_names, args.crystal_repr_outpath,
                        args.full_repr, args.full_repr_names, args.full_repr_outpath,
                        args.symmetry_visualization, args.symmetry_visualization_names, args.symmetry_visualization_outpath, args.quality_assurance,
                        args.idealize, args.report, args.report_outpath, args.report_names)

if __name__ == '__main__':
    main()