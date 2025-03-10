#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicSetut class
@Author: Mads Jeppesen
@Date: 9/21/22
"""
from symmetryhandler.symmetrysetup import SymmetrySetup
from symmetryhandler.mathfunctions import rotate
import numpy as np
import copy
from io import StringIO
import math
from symmetryhandler.mathfunctions import rotation_matrix, vector_angle, vector_projection_on_subspace
import textwrap
from cubicsym.utilities import cut_all_but_chains
from pyrosetta.rosetta.core.scoring import CA_rmsd, rms_at_all_corresponding_atoms, superimpose_pose, rms_at_corresponding_atoms_no_super, calpha_superimpose_pose
from pyrosetta.rosetta.std import map_unsigned_long_unsigned_long # core::Size, core::Size
from cubicsym.utilities import get_chain_map as util_get_chain_map
from cubicsym.utilities import map_hf_right_to_left_hf, map_3f_right_to_left_hf, map_2f_right_to_left_hf, get_base_from_pose
import yaml
from pyrosetta import pose_from_file
from pathlib import Path
from cubicsym.paths import DATA
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
from cubicsym.alignment import sequence_alignment_on_chain_set
from pyrosetta.rosetta.protocols.scoring import Interface
from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID
from pyrosetta import AtomID
from symmetryhandler.reference_kinematics import perturb_jumpdof_str_str
import tempfile

class CubicSetup(SymmetrySetup):

    def __init__(self, symdef=None, pose=None, symmetry_name=None):
        super().__init__(symdef, pose, symmetry_name)
        self.chain_map_str2int = {k: v for k, v in zip(tuple("ABCDEFGHI"), tuple(range(1, 10)))}
        self.extract_headers()

    @classmethod
    def from_symmetry_setup(cls, ss):
        file = StringIO(ss.make_symmetry_definition())
        return cls(file)

    def sds_overlaps_with_anchor(self, pose, update_and_apply_dofs=True, atol=1e-1):
        if update_and_apply_dofs:
            self.update_dofs_from_pose(pose, apply_dofs=True)
        anchor_pos = np.array(pose.residue(self.get_anchor_residue(pose)).atom("CA").xyz())
        jid = self.get_jumpidentifier_from_pose(pose)
        sds_pos = np.array(pose.residue(self.get_map_vrt_to_pose_resi(pose)[f"VRT{jid}fold111_sds"]).atom(1).xyz())
        try:
            assert np.isclose(anchor_pos, sds_pos, atol).all()
        except AssertionError:
            raise AssertionError(f"The anchor residue coordinates is {anchor_pos}, but should be be overlayed with VRT{jid}fold111_sds at {sds_pos}")

    @staticmethod
    def has_extra_chains(pose):
        return pose.num_chains() == 16 + 1

    @staticmethod
    def get_norm_symdef_path(symmetry, fold):
        return str(Path(DATA).joinpath(f"{symmetry}/{symmetry}_{fold}_norm.symm"))

    def load_norm_symdef(self, symmetry, fold):
        self.read_from_file(self.get_norm_symdef_path(symmetry, fold))

    def is_normalized(self):
        """Checks if the symmetry setup is normalized"""
        return bool(eval(self.headers.get("normalized", "False")))

    def read_from_file(self, filename, check_for_reference_symmetry=True):
        super().read_from_file(filename, check_for_reference_symmetry)
        self.extract_headers()

    def construct_chain_match_to_hf(self, same_handedness=True):
        assert isinstance(self.righthanded, bool), "#righthanded=True/False must be given in the symmetry file."
        symmetry = self.cubic_symmetry()

        if self.is_hf_based():
            if same_handedness:
                return [cm[0] for cm in util_get_chain_map(symmetry, self.righthanded)]
            else:
                return map_hf_right_to_left_hf(symmetry)
        elif self.is_3f_based():
            if same_handedness:
                return [cm[1] for cm in util_get_chain_map(symmetry, self.righthanded)]
            else:
                return map_3f_right_to_left_hf(symmetry)
        elif self.is_2f_based():
            if same_handedness:
                return [cm[2] for cm in util_get_chain_map(symmetry, self.righthanded)]
            else:
                return map_2f_right_to_left_hf(symmetry)

        # pose_mapping =
        # if not same_handedness:
        #     if not self.righthanded:
        #         raise NotImplementedError
        #     pose_new_mapping = map_hf_right_to_left(symmetry)
        #     pose_mapping = list(np.array(pose_mapping)[np.array(pose_new_mapping) - 1])
        # return pose_mapping

    def interface_residues(self, pose, main_chain=1):
        """Calculates which residues are in the interface of the main_chain. This is done by detecting all CA atoms
        of all other chains that are within 10Å to any CA atom of the main chain."""
        main_resis = list(range(pose.chain_begin(main_chain), pose.chain_end(main_chain) + 1))
        other_resis = [ri for ri in range(1, pose.size()) if ri < pose.chain_begin(main_chain) or ri > pose.chain_end(main_chain)]
        main_resis_interface, other_resis_interface = set(), set()
        for main_resi in main_resis:
            main_ca_xyz = pose.residue(main_resi).atom("CA").xyz()
            for other_resi in other_resis:
                other_ca_xyz = pose.residue(other_resi).atom("CA").xyz()
                if main_ca_xyz.distance(other_ca_xyz) <= 10:
                    main_resis_interface.add(main_resi)
                    other_resis_interface.add(other_resi)
        return main_resis_interface.union(other_resis_interface)

    def construct_atom_map_any2hf(self, pose, pose_ref, same_handedness=True, interface=False, predicate="ca", use_map=None) -> map_core_id_AtomID_core_id_AtomID:
        """Constructs a map that maps the atoms of a pose with any fold type (HF, 3F, 2F) and any handedness,
        to a HF-based pose_ref with any handedness. The predicate determines which atoms are selected. ca: all CA atoms
        are selected. heavey_atoms: all heavy atoms are selected."""
        assert predicate in ("ca", "heavy_atoms")
        resi_map = self.construct_residue_map_any2hf(pose, pose_ref, same_handedness=same_handedness, interface=interface, use_map=use_map)
        atommap = map_core_id_AtomID_core_id_AtomID()
        for ri_pose, ri_pose_ref in resi_map.items():
            if predicate == "heavy_atoms":
                for ai in range(1, pose.residue(ri_pose).natoms() + 1):
                    assert pose.residue(ri_pose).atom_type(ai).name() == pose_ref.residue(ri_pose_ref).atom_type(ai).name()
                    if pose.residue(ri_pose).atom_type(ai).is_heavyatom():
                        atommap[AtomID(ai, ri_pose)] = AtomID(ai, ri_pose_ref)
            elif predicate == "ca":
                atommap[AtomID(2, ri_pose)] = AtomID(2, ri_pose_ref)
        return atommap

    def construct_residue_map_any2hf(self, pose, pose_ref, same_handedness=True, interface=False, use_map=None) -> map_unsigned_long_unsigned_long:
        """Constructs a map that maps the residues of a pose with any fold type (HF, 3F, 2F) and any handedness,
        to a HF-based pose_ref with any handedness"""
        # 1. map  chains to eachother
        if use_map is not None:
            pose_mapping = use_map
            assert len(use_map) in (7, 8, 9) # must have enough chains
        else:
            pose_mapping = self.construct_chain_match_to_hf(same_handedness=same_handedness)
        # 2. map residues to eachtother taking into account the sequence alignment
        if is_symmetric(pose_ref):
            ref_mapping = [c for c in range(1, pose_ref.num_chains())] # this will not take the VRT residues with it
        else:
            ref_mapping = [c for c in range(1, pose_ref.num_chains() + 1)]
        if use_map is not None:
            ref_mapping = [c_ref for c_ref, c in zip(ref_mapping, pose_mapping) if c is not None]
            pose_mapping = [c for c in pose_mapping if c is not None]
        alignment = sequence_alignment_on_chain_set(pose, pose_ref, pose_mapping, ref_mapping) # not +1 because we dont want VRT
        resis = [i for ii in [alignment[k] for k in [f"1_{m}" for m in pose_mapping]] for i in ii]
        resis_ref = [i for ii in [alignment[k] for k in [f"2_{m}" for m in ref_mapping]] for i in ii]
        if interface:
            assert not is_symmetric(pose_ref)
            interface_resi = self.interface_residues(pose_ref)
            # reduce
            resis_new, resis_ref_new = [], []
            for resi, resi_ref in zip(resis, resis_ref):
                if resi_ref in interface_resi:
                    resis_new.append(resi)
                    resis_ref_new.append(resi_ref)
            resis, resis_ref = resis_new, resis_ref_new

            # # For now we just do asymmetric only. It will most likely NOT work for symmetry but haven't tried.
            # # In Interface::calculate() it will call symmetric_protein_calculate if the protein is symmetric, so
            # # perhaps it will work?
            # assert not is_symmetric(pose_ref)
            # # We want to do the interface calculation as in protocols::docking::calc_Irmsd to stay consistent with
            # # previous approaches to calculating this.
            # # when looking at the Interface::protein_calculate() function it seems like the pose needs to be scored first
            # # in order to get the Interface class to work as it uses pose.energies().energy_graph(). previously I couldn't
            # # get 4DCL to work before I did this
            # from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
            # ScoreFunctionFactory.create_score_function("ref2015").score(pose_ref)
            # resis_new, resis_ref_new = [], []
            # # the interface has to be calculated across all jumps
            # for jump in range(1, pose_ref.num_jump() + 1):
            #     interface = Interface(jump)
            #     interface.distance(10.0)  # 10.0 as in Rosetta Code
            #     interface.calculate(pose_ref)
            #     for resi, resi_ref in zip(resis, resis_ref):
            #         if interface.is_interface(resi_ref):
            #             resis_new.append(resi)
            #             resis_ref_new.append(resi_ref)
            #     break
            # resis, resis_ref = resis_new, resis_ref_new
            # assert len(resis) > 0, "No interface residues are found!"
            # find interface in pose_re
        # old way without sequence alignment
        # resis = [i for ii in [range(pose.chain_begin(c), pose.chain_end(c) + 1) for c in pose_mapping] for i in ii]
        # resis_ref = list(range(1, pose_ref.size() + 1))
        m = map_unsigned_long_unsigned_long()
        for resi, resi_ref in zip(resis, resis_ref):
            m[resi] = resi_ref
        return m

    # def construct_chain_map_handedness(self, pose, pose_ref):
    #     # todo: map both T and O with right handed and left handed - use the tests in cubicsetup todo this!
    #     assert isinstance(self.righthanded, bool), "#righthanded=True/False must be given in the symmetry file."
    #     chain_map = util_get_chain_map(self.cubic_symmetry_from_setup(), self.righthanded)
    #     m = map_unsigned_long_unsigned_long()
    #     if self.is_hf_based():
    #         index = 0
    #     elif self.is_3f_based():
    #         index = 1
    #     elif self.is_2f_based():
    #         index = 2
    #     resis = [i for ii in [range(pose.chain_begin(c), pose.chain_end(c) + 1) for c in [cm[index] for cm in chain_map]] for i in ii]
    #     resis_ref = list(range(1, pose_ref.size() + 1))
    #     for resi, resi_ref in zip(resis, resis_ref):
    #         m[resi] = resi_ref
    #     return m

    # fixme: Delete as it is unused
    # def calpha_superimpose_pose_hf_map(self, pose, pose_ref, same_handedness=True):
    #     """Aligns a cubic symmetrical pose onto another pose_ref taking the correct chain mapping into account. It removes the symetry
    #     and creates a new asymmetrical pose with reordered chains so that it matches the ordering in pose_ref. It returns the aligned pose."""
    #     # lets reconstruct the chains
    #     pose_chains = self.construct_chain_match_to_hf(same_handedness=same_handedness)
    #     pose.dump_pdb("/tmp/pose_from.pdb")
    #     pose_ref.dump_pdb("/tmp/pose_to.pdb")
    #     pose_from = pose_from_file("/tmp/pose_from.pdb")
    #     pose_to = pose_from_file("/tmp/pose_to.pdb")
    #     split_chains = list(pose_from.split_by_chain())  # [np.array(pose_chains) - 1]
    #     chains_in_new_order = [split_chains[i - 1] for i in pose_chains]
    #     pose_from_new = chains_in_new_order[0]
    #     for n, chain in enumerate(chains_in_new_order[1:], 1):
    #         pose_from_new.append_pose_by_jump(chain, pose_from_new.chain_end(n))
    #     calpha_superimpose_pose(pose_from_new, pose_to)
    #     return pose_from_new

    def get_register_shift_angle(self):
        sym = self.cubic_symmetry()
        base = self.get_base()
        if base == "HF":
            if sym == "I":
                return 72, 5
            elif sym == "O":
                return 90, 4
            elif sym == "T":
                return 120, 3
        elif base == "3F":
            return 120, 3
        elif base == "2F":
            return 180, 2

    def rmsd_hf_map(self, pose, pose_ref, same_handedness=True, interface=False, predicate="ca", register_shift=True, use_map=None):
        """Calculates the CA RMSD bewteen pose and pose_ref where the change in chain numbering is taking into account between the
        HF-, 3- and 2-fold based CubicSetup. The pose CubicSetup must be the one belonging to pose and can be either 2-, 3- or HF-fold.
        The CubicSetup of pose_ref must be HF based. The pose_ref can either be the same handedness (left/right) as the pose (which is the
        default) or it can be opposite. If it is opposite, parse same_handedness=False.

        :param pose: pose to calculate RMSD with
        :param pose_ref: pose to calculate RMSD against
        :param same_handedness: Do both pose and pose_ref have the same handedness? Are they both righthanded or both lefthanded?
        :param interface: Use only interface atoms for the RMSD calculation.
        :param predicate: Which atom types to use for the RMSD calculation. Now only 'ca' (CA atoms) or 'heavy_atoms' (heavy atoms)
            are supported.
        :param register_shift: Calculate the RMSD at all equivalent symmetrial positions by rotating around the base fold.
        """
        assert predicate in ("ca", "heavy_atoms")
        atom_map = self.construct_atom_map_any2hf(pose, pose_ref, same_handedness=same_handedness, interface=interface, predicate=predicate, use_map=use_map)
        return self.rmsd_hf_map_with_atom_map(pose, pose_ref, atom_map, register_shift)

    def rmsd_hf_map_with_atom_map(self, pose, pose_ref, atom_map, register_shift=True):
        """Same as CA_rmsd_hf_map but uses a precalculated atom_map (constructed with the construct_atom_map_any2hf
        function) for faster RMSD calculation. See rmsd_hf_map for more information on the options."""
        # if align_map is not None:
            # 1. Align based on the align_map
            # this function is not deterministic and gives crap sometimes! In the description of the function
            # it even says you shouldt superimpose asym with sym pose. I can't find a function in Rosetta
            # that can align a symmetric pose onto a non symmetric pose and therefore I will not do this.
            # superimpose_pose(pose, pose_ref, align_map)
            # 2. Calculate rmsd with no super
            # return rms_at_corresponding_atoms_no_super(pose, pose_ref, rmsd_map)
        # else:
            # Superimpose and calculate rmsd in one go. The atoms that are used for the RMSD calculation are also used
            # for the alignment.
        if register_shift:
            pose_temp = pose.clone()
            angle, rots = self.get_register_shift_angle()
            jump = f"JUMP{self.get_jumpidentifier()}fold1_z"
            rmsds = []
            rmsds.append(rms_at_all_corresponding_atoms(pose_temp, pose_ref, atom_map))
            for n in range(1, rots):
                perturb_jumpdof_str_str(pose_temp, jump, "angle_z", angle)
                rmsds.append(rms_at_all_corresponding_atoms(pose_temp, pose_ref, atom_map))
            return min(rmsds)
        else:
            return rms_at_all_corresponding_atoms(pose, pose_ref, atom_map)

    @staticmethod
    def get_jumpidentifier_from_base(base) -> str:
        """Returns the identifier for the jump names. The movable jump names are given as JUMP<IDENTIFIER>fold<VRTTYPE>."""
        if base == "HF":
            return "HF"
        elif base == "3F":
            return "31"
        elif base == "2F":
            return "21"

    def get_jumpidentifier(self) -> str:
        """Returns the identifier for the jump names. The movable jump names are given as JUMP<IDENTIFIER>fold<VRTTYPE>."""
        if self.is_hf_based():
            return "HF"
        elif self.is_3f_based():
            return "31"
        elif self.is_2f_based():
            return "21"

    @staticmethod
    def get_jumpidentifier_from_pose(pose):
        """Returns the identifier for the jump names from the pose. The movable jump names are given as JUMP<IDENTIFIER>fold<VRTTYPE>."""
        base = CubicSetup.get_base_from_pose(pose)
        if base == "HF":
            return "HF"
        elif base == "3F":
            return "31"
        elif base == "2F":
            return "21"

    @staticmethod
    def get_fold_jumpidentifier_from_pose(pose):
        """Returns the identifier for the jump names from the pose. The movable jump names are given as JUMP<IDENTIFIER>fold<VRTTYPE>."""
        base = CubicSetup.get_base_from_pose(pose)
        if base == "HF":
            return "HF", "3", "2"
        elif base == "3F":
            raise NotImplementedError
            return "31"
        elif base == "2F":
            raise NotImplementedError
            return "21"

    @staticmethod
    def get_base_from_pose(pose):
        """Returns the cubicsetup type as a str from a pose."""
        return get_base_from_pose(pose)

    def get_base(self) -> str:
        """Returns the cubicsetup type as a str"""
        if self.is_hf_based():
            return "HF"
        elif self.is_3f_based():
            return "3F"
        elif self.is_2f_based():
            return "2F"

    def is_hf_based(self):
        return "JUMPHFfold" in self.jumps.keys()

    def is_3f_based(self):
        return "JUMP31fold" in self.jumps.keys()

    def is_2f_based(self):
        return "JUMP21fold" in self.jumps.keys()

    def get_HF_chains(self, pose):
        """Get the HF chains of the pose only."""
        return cut_all_but_chains(pose.clone(), *self.get_HF_chain_ids())

    def get_3fold_chains(self, pose):
        """Get the 3 fold chains of the pose only."""
        return cut_all_but_chains(pose.clone(), *self.get_3fold_chain_ids())

    def get_2fold_chains(self, pose):
        """Get the HF chains of the pose only. This returns both of the 2 folds. The first one is closest and the
        second one is the furthest."""
        closest = cut_all_but_chains(pose.clone(), *self.get_2fold_chain_ids()[0])
        furthest = cut_all_but_chains(pose.clone(), *self.get_2fold_chain_ids()[1])
        return closest, furthest

    @staticmethod
    def get_pose_with_chains(pose, *chains):
        """Get a new instance of a pose with the chains only."""
        return cut_all_but_chains(pose.clone(), *chains)

    def __get_chains_ids(self, ids, rosetta_number):
        if rosetta_number:
            return tuple([self.chain_map_str2int[i] for i in ids])
        return ids

    def get_HF_chain_ids(self, rosetta_number=False):
        """Get the HF fold chains names either as a str (default) or Rosetta number."""
        if "I" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("ABCDE"), rosetta_number)
        if "O" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("ABCD"), rosetta_number)
        if "T" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("ABC"), rosetta_number)

    def get_3fold_chain_ids(self, rosetta_number=False):
        """Get the 3 fold chains names either as a str (default) or Rosetta number.."""
        if "I" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("AIF"), rosetta_number)
        if "O" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("AEH"), rosetta_number)
        if "T" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("ADG"), rosetta_number)

    def get_2fold_chain_ids(self, rosetta_number=False):
        """Get the 2 fold chains names either as a str (default) or Rosetta number. This returns both of the 2 folds.
        The first one is closest and the second one is the furthest."""

        if "I" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("AH"), rosetta_number), self.__get_chains_ids(tuple("AG"), rosetta_number)
        if "O" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("AG"), rosetta_number), self.__get_chains_ids(tuple("AF"), rosetta_number)
        if "T" == self.cubic_symmetry():
            return self.__get_chains_ids(tuple("AF"), rosetta_number), self.__get_chains_ids(tuple("AE"), rosetta_number)

    def cubic_energy_multiplier_from_pose(self, pose) -> int:
       symmetry_type = self.cubic_symmetry_from_pose(pose)
       if "I" == symmetry_type:
           return 60
       if "O" == symmetry_type:
           return 24
       if "T" == symmetry_type:
           return 12
       else:
           raise ValueError("Symmetry is not cubic!")

    def is_cubic(self):
        if any((e in self.energies for e in ("60*", "24*", "12*"))):
            if self.is_hf_based() or self.is_3f_based() or self.is_2f_based():
                return True
        return False

    def cubic_symmetry(self):
        """Determine the cubic symmetry from a SymmetrySetup object."""
        if "60*" in self.energies:
            return "I"
        elif "24*" in self.energies:
            return "O"
        elif "12*" in self.energies:
            return "T"
        else:
            raise ValueError("Symmetry is not cubic!")

    def hf_rotation_angle_per_subunit(self):
        """Determine the cubic symmetry from a SymmetrySetup object."""
        if "60" in self.energies:
            return 72
        elif "24" in self.energies:
            return 90
        elif "12" in self.energies:
            return 120
        else:
            raise ValueError("Symmetry is not cubic!")

    # fixme: dangerous if you have more than 1 subunit as the monomer
    @staticmethod
    def cubic_symmetry_from_pose(pose):
        nsubs = pose.conformation().Symmetry_Info().subunits()
        if nsubs == 16:
            return "I"
        elif nsubs == 9:
            return "I"
        elif nsubs == 8:
            return "O"
        elif nsubs == 7:
            return "T"
        else:
            raise ValueError("Symmetry is not cubic!")

    # def get_cubic_limits(self, pose, init_pos=None, angle_mod=None):
    #     position_info = get_cubic_dofs(pose)
    #     min_, max_ = self.icosahedral_angle_z(pose)
    #     total_angle = abs(min_) + max_
    #     if self.cubic_symmetry_type(pose) == "I":
    #         assert math.isclose(total_angle, 72.0,
    #                             rel_tol=1e-2), f"The icosahedral 5-fold angle should approximately 72 degrees not {total_angle} !"
    #     if init_pos and angle_mod:
    #         pass
    #     limit_info = {}
    #     limit_info["JUMPHFfold1"] = {"z": {"min": - position_info.get("JUMPHFfold1")["z"], "max": 1000}}
    #     limit_info["JUMPHFfold1_z"] = {"z_angle": {k: v for k, v in zip(["min", "max"], [min_, max_])}}
    #     limit_info["JUMPHFfold111"] = {"x": {"min": - position_info.get("JUMPHFfold111")["x"], "max": 1000}}
    #     return limit_info

    def angle_z_distance(self, pose):
        symm_type = self.cubic_symmetry_from_pose(pose)
        if symm_type == "I":
            min_, max_ = self.icosahedral_angle_z(pose)
            total_angle = abs(min_) + max_
            assert math.isclose(total_angle, 72.0,
                                rel_tol=1e-2), f"The icosahedral 5-fold angle should approximately 72 degrees not {total_angle} !"
            return min_, max_
        elif symm_type == "O":
            raise NotImplementedError
        elif symm_type == "T":
            raise NotImplementedError

    def icosahedral_angle_z(self, pose, visualize=False, cmd=None):
        """Gets the rotation angles from the master subunit com to the two 2-fold symmetry axes.

        When looking at the icosahedral structure in PyMOL the neative angle_z rotation is to the left and the positive
        to the right.
        """
        symmetry_setup = copy.deepcopy(self)
        symmetry_setup.update_dofs_from_pose(pose)
        symmetry_setup.apply_dofs()

        # main 5-fold vectors
        v_5fold_center_to_5fold_master_com = symmetry_setup.get_vrt("VRTHFfold111_z").vrt_orig - symmetry_setup.get_vrt(
            "VRTHFfold1").vrt_orig
        # v_5fold_center_to_5fold_slave2_com = symmetry_setup.get_vrt_name("VRTHFfold1211").vrt_orig - symmetry_setup.get_vrt_name("VRTHFfold1").vrt_orig
        # v_5fold_center_to_5fold_slave5_com = symmetry_setup.get_vrt_name("VRTHFfold1511").vrt_orig - symmetry_setup.get_vrt_name("VRTHFfold1").vrt_orig

        # other 5-fold vectors
        v_5fold_center_to_2fold_center = symmetry_setup.get_vrt("VRT2fold1").vrt_orig - symmetry_setup.get_vrt("VRTHFfold1").vrt_orig
        # v_2fold_center_to_3fold_center = symmetry_setup.get_vrt_name("VRT3fold1").vrt_orig - symmetry_setup.get_vrt_name("VRT2fold1").vrt_orig
        v_5fold_center_to_3fold_center = symmetry_setup.get_vrt("VRT3fold1").vrt_orig - symmetry_setup.get_vrt("VRTHFfold1").vrt_orig

        # project these onto subspace
        # NOTE: vector_projection_on_subspace assumes orthonormal vectors in subspace!!!
        #  Since the capsid is oriented in the z direction then x-y spane a plane slicing through it. We can use that.
        v_5fold_center_to_2fold_center_projected = vector_projection_on_subspace(v_5fold_center_to_2fold_center,
                                                                                 np.array([1, 0, 0]),
                                                                                 np.array([0, 1, 0]))
        v_5fold_center_to_3fold_center_projected = vector_projection_on_subspace(v_5fold_center_to_3fold_center,
                                                                                 np.array([1, 0, 0]),
                                                                                 np.array([0, 1, 0]))

        angle_to_nearest_2fold = vector_angle(v_5fold_center_to_5fold_master_com, v_5fold_center_to_2fold_center_projected)
        angle_to_furthest_2fold = vector_angle(v_5fold_center_to_5fold_master_com, v_5fold_center_to_3fold_center_projected)

        if visualize:
            cmd.do(f"pseudoatom v_5fold_center_to_5fold_master_com, pos={list(v_5fold_center_to_5fold_master_com)}")
            # cmd.do(f"v_5fold_center_to_5fold_master_com {symmetry_setup.get_vrt_name('VRTHFfold1').vrt_orig}, {symmetry_setup.get_vrt_name("VRTHFfold1111").vrt_orig})
            # cmd.do(f"pseudoatom v_5fold_center_to_5fold_slave2_com, pos={list(v_5fold_center_to_5fold_slave2_com)}")
            # cmd.do(f"pseudoatom v_5fold_center_to_5fold_slave5_com, pos={list(v_5fold_center_to_5fold_slave5_com)}")
            cmd.do(f"pseudoatom v_5fold_center_to_2fold_center, pos={list(v_5fold_center_to_2fold_center)}")
            cmd.do(f"pseudoatom v_5fold_center_to_3fold_center, pos={list(v_5fold_center_to_3fold_center)}")
            cmd.do(f"pseudoatom v_5fold_center_to_2fold_center_projected, pos={list(v_5fold_center_to_2fold_center_projected)}")
            cmd.do(f"pseudoatom v_5fold_center_to_3fold_center_projected, pos={list(v_5fold_center_to_3fold_center_projected)}")

        # todo: record this before hand
        # Now, the two 2 2-folds can be either right or left ot the master subunuit. To determine if they are right of left we can take the
        # cross product of one of them and sew how it aligns with the global z-axis. If it is -z (global axis), the 2-fold is to the left,
        # or the negative direction, while it is vica versa for the other. We arbitrarily pick the two-fold that is closest. This is
        # the one connected by VRT2fold1 to calculate the cross product from.
        z_value = np.cross(v_5fold_center_to_5fold_master_com, v_5fold_center_to_2fold_center_projected)[2]
        if z_value < 0:  # the nearest twofold is to the left / in the negative angle_z rotation direction. [1stm case]
            max_negative_angle = -angle_to_nearest_2fold
            max_positive_angle = angle_to_furthest_2fold
        else:  # the nearest twofold is to the right / in the positive angle_z rotation direction. [4v4m case]
            max_negative_angle = -angle_to_furthest_2fold
            max_positive_angle = angle_to_nearest_2fold

        return max_negative_angle, max_positive_angle

        # TODO: If needed create an universal n-fold function

    def create_independent_4fold_symmetries(self, pose):
        """Creates independent symmetries for the 4-fold."""
        # chain 1-2 and chain 1-3
        symmetry_setup = copy.deepcopy(self)
        symmetry_setup.apply_dofs()
        symmetry_setup.update_dofs_from_pose(pose)

        # what dofs are available in the old file

        chain1_2 = CubicSetup()
        chain1_2.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name chain1_2 
              E = 4*VRT000111 + 4*(VRT000111:VRT000222)
              anchor_residue COM 
              recenter
              virtual_coordinates_start
              xyz VRTglobal 1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT0001 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT0002 -0.000000,-1.000000,0.000000 -1.000000,0.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT00011 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT00022 -0.000000,-1.000000,0.000000 -1.000000,0.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT000111 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT000222 -0.000000,-1.000000,0.000000 -1.000000,0.000000,0.000000 0.000000,0.000000,0.000000
              virtual_coordinates_stop
              connect_virtual JUMPG1 VRTglobal VRT0001
              connect_virtual JUMPG2 VRTglobal VRT0002
              connect_virtual JUMP1 VRT0001 VRT00011
              connect_virtual JUMP2 VRT0002 VRT00022
              connect_virtual JUMP11 VRT00011 VRT000111 
              connect_virtual JUMP22 VRT00022 VRT000222 
              connect_virtual JUMP111 VRT000111 SUBUNIT
              connect_virtual JUMP222 VRT000222 SUBUNIT
              set_dof JUMP1 x({symmetry_setup._dofs['JUMP1'][0][2]}) 
              set_dof JUMP11 angle_x({symmetry_setup._dofs['JUMP11'][0][2]}) angle_y({symmetry_setup._dofs['JUMP11'][1][2]}) angle_z({symmetry_setup._dofs['JUMP11'][2][2]})
              set_dof JUMP111 angle_x({symmetry_setup._dofs['JUMP111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP111'][2][2]})
              set_jump_group MODIFIED_BASEJUMP1 JUMP1 JUMP2
              set_jump_group MODIFIED_BASEJUMP2 JUMP11 JUMP22
              set_jump_group MODIFIED_BASEJUMP3 JUMP111 JUMP222
              """)))

        chain1_3 = CubicSetup()
        chain1_3.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name chain1_3 
              E = 4*VRT000111 + 2*(VRT000111:VRT000333)
              anchor_residue COM 
              recenter
              virtual_coordinates_start
              xyz VRTglobal 1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT0001 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT0003 1.000000,-0.000000,0.000000 -0.000000,-1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT00011 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT00033 1.000000,-0.000000,0.000000 -0.000000,-1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT000111 -1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,0.000000
              xyz VRT000333 1.000000,-0.000000,0.000000 -0.000000,-1.000000,0.000000 0.000000,0.000000,0.000000
              virtual_coordinates_stop
              connect_virtual JUMPG1 VRTglobal VRT0001
              connect_virtual JUMPG3 VRTglobal VRT0003
              connect_virtual JUMP1 VRT0001 VRT00011
              connect_virtual JUMP3 VRT0003 VRT00033
              connect_virtual JUMP11 VRT00011 VRT000111 
              connect_virtual JUMP33 VRT00033 VRT000333 
              connect_virtual JUMP111 VRT000111 SUBUNIT
              connect_virtual JUMP333 VRT000333 SUBUNIT
              set_dof JUMP1 x({symmetry_setup._dofs['JUMP1'][0][2]}) 
              set_dof JUMP11 angle_x({symmetry_setup._dofs['JUMP11'][0][2]}) angle_y({symmetry_setup._dofs['JUMP11'][1][2]}) angle_z({symmetry_setup._dofs['JUMP11'][2][2]})
              set_dof JUMP111 angle_x({symmetry_setup._dofs['JUMP111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP111'][2][2]})
              set_jump_group MODIFIED_BASEJUMP1 JUMP1 JUMP3
              set_jump_group MODIFIED_BASEJUMP2 JUMP11 JUMP33
              set_jump_group MODIFIED_BASEJUMP3 JUMP111 JUMP333
              """)))

        return chain1_2, chain1_3


    def create_independent_icosahedral_symmetries(self, pose):
        """Creates independent symmetries for the icosahedral 5-fold, 3-fold and two 2-folds."""
        symmetry_setup = copy.deepcopy(self)
        symmetry_setup.apply_dofs()
        symmetry_setup.update_dofs_from_pose(pose)

        fold5 = CubicSetup()
        fold5.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name 5fold
                      E = 5*VRTHFfold111_sds + 5*(VRTHFfold111_sds:VRTHFfold121_sds) + 5*(VRTHFfold111_sds:VRTHFfold131_sds)
                      anchor_residue COM
                      virtual_coordinates_start
            {self.get_vrt("VRTglobal")}
            {self.get_vrt("VRTHFfold1_z_tref")}
            {self.get_vrt("VRTHFfold")}
            {self.get_vrt("VRTHFfold1")}
            {self.get_vrt("VRTHFfold1_z_rref")}
            {self.get_vrt("VRTHFfold1_z")}
            {self.get_vrt("VRTHFfold111_x_tref")}
            {self.get_vrt("VRTHFfold11")}
            {self.get_vrt("VRTHFfold111")}
            {self.get_vrt("VRTHFfold111_x_rref")}
            {self.get_vrt("VRTHFfold111_x")}
            {self.get_vrt("VRTHFfold111_y_rref")}
            {self.get_vrt("VRTHFfold111_y")}
            {self.get_vrt("VRTHFfold111_z_rref")}
            {self.get_vrt("VRTHFfold111_z")}
            {self.get_vrt("VRTHFfold111_sds")}
            {self.get_vrt("VRTHFfold121_x_tref")}
            {self.get_vrt("VRTHFfold12")}
            {self.get_vrt("VRTHFfold121")}
            {self.get_vrt("VRTHFfold121_x_rref")}
            {self.get_vrt("VRTHFfold121_x")}
            {self.get_vrt("VRTHFfold121_y_rref")}
            {self.get_vrt("VRTHFfold121_y")}
            {self.get_vrt("VRTHFfold121_z_rref")}
            {self.get_vrt("VRTHFfold121_z")}
            {self.get_vrt("VRTHFfold121_sds")}
            {self.get_vrt("VRTHFfold131_x_tref")}
            {self.get_vrt("VRTHFfold13")}
            {self.get_vrt("VRTHFfold131")}
            {self.get_vrt("VRTHFfold131_x_rref")}
            {self.get_vrt("VRTHFfold131_x")}
            {self.get_vrt("VRTHFfold131_y_rref")}
            {self.get_vrt("VRTHFfold131_y")}
            {self.get_vrt("VRTHFfold131_z_rref")}
            {self.get_vrt("VRTHFfold131_z")}
            {self.get_vrt("VRTHFfold131_sds")}
            {self.get_vrt("VRTHFfold141_x_tref")}
            {self.get_vrt("VRTHFfold14")}
            {self.get_vrt("VRTHFfold141")}
            {self.get_vrt("VRTHFfold141_x_rref")}
            {self.get_vrt("VRTHFfold141_x")}
            {self.get_vrt("VRTHFfold141_y_rref")}
            {self.get_vrt("VRTHFfold141_y")}
            {self.get_vrt("VRTHFfold141_z_rref")}
            {self.get_vrt("VRTHFfold141_z")}
            {self.get_vrt("VRTHFfold141_sds")}
            {self.get_vrt("VRTHFfold151_x_tref")}
            {self.get_vrt("VRTHFfold15")}
            {self.get_vrt("VRTHFfold151")}
            {self.get_vrt("VRTHFfold151_x_rref")}
            {self.get_vrt("VRTHFfold151_x")}
            {self.get_vrt("VRTHFfold151_y_rref")}
            {self.get_vrt("VRTHFfold151_y")}
            {self.get_vrt("VRTHFfold151_z_rref")}
            {self.get_vrt("VRTHFfold151_z")}
            {self.get_vrt("VRTHFfold151_sds")}
            virtual_coordinates_stop  
            connect_virtual JUMPHFfold1_z_tref VRTglobal VRTHFfold1_z_tref
            connect_virtual JUMPHFfold VRTHFfold1_z_tref VRTHFfold
            connect_virtual JUMPHFfold1 VRTHFfold VRTHFfold1
            connect_virtual JUMPHFfold1_z_rref VRTHFfold1 VRTHFfold1_z_rref
            connect_virtual JUMPHFfold1_z VRTHFfold1_z_rref VRTHFfold1_z
            connect_virtual JUMPHFfold111_x_tref VRTHFfold1_z VRTHFfold111_x_tref
            connect_virtual JUMPHFfold11 VRTHFfold111_x_tref VRTHFfold11
            connect_virtual JUMPHFfold111 VRTHFfold11 VRTHFfold111
            connect_virtual JUMPHFfold111_x_rref VRTHFfold111 VRTHFfold111_x_rref
            connect_virtual JUMPHFfold111_x VRTHFfold111_x_rref VRTHFfold111_x
            connect_virtual JUMPHFfold111_y_rref VRTHFfold111_x VRTHFfold111_y_rref
            connect_virtual JUMPHFfold111_y VRTHFfold111_y_rref VRTHFfold111_y
            connect_virtual JUMPHFfold111_z_rref VRTHFfold111_y VRTHFfold111_z_rref
            connect_virtual JUMPHFfold111_z VRTHFfold111_z_rref VRTHFfold111_z
            connect_virtual JUMPHFfold111_sds VRTHFfold111_z VRTHFfold111_sds
            connect_virtual JUMPHFfold111_subunit VRTHFfold111_sds SUBUNIT
            connect_virtual JUMPHFfold121_x_tref VRTHFfold1_z VRTHFfold121_x_tref
            connect_virtual JUMPHFfold12 VRTHFfold121_x_tref VRTHFfold12
            connect_virtual JUMPHFfold121 VRTHFfold12 VRTHFfold121
            connect_virtual JUMPHFfold121_x_rref VRTHFfold121 VRTHFfold121_x_rref
            connect_virtual JUMPHFfold121_x VRTHFfold121_x_rref VRTHFfold121_x
            connect_virtual JUMPHFfold121_y_rref VRTHFfold121_x VRTHFfold121_y_rref
            connect_virtual JUMPHFfold121_y VRTHFfold121_y_rref VRTHFfold121_y
            connect_virtual JUMPHFfold121_z_rref VRTHFfold121_y VRTHFfold121_z_rref
            connect_virtual JUMPHFfold121_z VRTHFfold121_z_rref VRTHFfold121_z
            connect_virtual JUMPHFfold121_sds VRTHFfold121_z VRTHFfold121_sds
            connect_virtual JUMPHFfold121_subunit VRTHFfold121_sds SUBUNIT
            connect_virtual JUMPHFfold131_x_tref VRTHFfold1_z VRTHFfold131_x_tref
            connect_virtual JUMPHFfold13 VRTHFfold131_x_tref VRTHFfold13
            connect_virtual JUMPHFfold131 VRTHFfold13 VRTHFfold131
            connect_virtual JUMPHFfold131_x_rref VRTHFfold131 VRTHFfold131_x_rref
            connect_virtual JUMPHFfold131_x VRTHFfold131_x_rref VRTHFfold131_x
            connect_virtual JUMPHFfold131_y_rref VRTHFfold131_x VRTHFfold131_y_rref
            connect_virtual JUMPHFfold131_y VRTHFfold131_y_rref VRTHFfold131_y
            connect_virtual JUMPHFfold131_z_rref VRTHFfold131_y VRTHFfold131_z_rref
            connect_virtual JUMPHFfold131_z VRTHFfold131_z_rref VRTHFfold131_z
            connect_virtual JUMPHFfold131_sds VRTHFfold131_z VRTHFfold131_sds
            connect_virtual JUMPHFfold131_subunit VRTHFfold131_sds SUBUNIT
            connect_virtual JUMPHFfold141_x_tref VRTHFfold1_z VRTHFfold141_x_tref
            connect_virtual JUMPHFfold14 VRTHFfold141_x_tref VRTHFfold14
            connect_virtual JUMPHFfold141 VRTHFfold14 VRTHFfold141
            connect_virtual JUMPHFfold141_x_rref VRTHFfold141 VRTHFfold141_x_rref
            connect_virtual JUMPHFfold141_x VRTHFfold141_x_rref VRTHFfold141_x
            connect_virtual JUMPHFfold141_y_rref VRTHFfold141_x VRTHFfold141_y_rref
            connect_virtual JUMPHFfold141_y VRTHFfold141_y_rref VRTHFfold141_y
            connect_virtual JUMPHFfold141_z_rref VRTHFfold141_y VRTHFfold141_z_rref
            connect_virtual JUMPHFfold141_z VRTHFfold141_z_rref VRTHFfold141_z
            connect_virtual JUMPHFfold141_sds VRTHFfold141_z VRTHFfold141_sds
            connect_virtual JUMPHFfold141_subunit VRTHFfold141_sds SUBUNIT
            connect_virtual JUMPHFfold151_x_tref VRTHFfold1_z VRTHFfold151_x_tref
            connect_virtual JUMPHFfold15 VRTHFfold151_x_tref VRTHFfold15
            connect_virtual JUMPHFfold151 VRTHFfold15 VRTHFfold151
            connect_virtual JUMPHFfold151_x_rref VRTHFfold151 VRTHFfold151_x_rref
            connect_virtual JUMPHFfold151_x VRTHFfold151_x_rref VRTHFfold151_x
            connect_virtual JUMPHFfold151_y_rref VRTHFfold151_x VRTHFfold151_y_rref
            connect_virtual JUMPHFfold151_y VRTHFfold151_y_rref VRTHFfold151_y
            connect_virtual JUMPHFfold151_z_rref VRTHFfold151_y VRTHFfold151_z_rref
            connect_virtual JUMPHFfold151_z VRTHFfold151_z_rref VRTHFfold151_z
            connect_virtual JUMPHFfold151_sds VRTHFfold151_z VRTHFfold151_sds
            connect_virtual JUMPHFfold151_subunit VRTHFfold151_sds SUBUNIT 
            set_dof JUMPHFfold1 z({symmetry_setup.dofs['JUMPHFfold1'][0][2]}) 
            set_dof JUMPHFfold1_z angle_z({symmetry_setup.dofs['JUMPHFfold1_z'][0][2]})
            set_dof JUMPHFfold111 x({symmetry_setup.dofs['JUMPHFfold111'][0][2]})
            set_dof JUMPHFfold111_x angle_x({symmetry_setup.dofs['JUMPHFfold111_x'][0][2]})
            set_dof JUMPHFfold111_y angle_y({symmetry_setup.dofs['JUMPHFfold111_y'][0][2]})
            set_dof JUMPHFfold111_z angle_z({symmetry_setup.dofs['JUMPHFfold111_z'][0][2]})
            set_jump_group JUMPGROUP1 JUMPHFfold111 JUMPHFfold121 JUMPHFfold131 JUMPHFfold141 JUMPHFfold151 
            set_jump_group JUMPGROUP2 JUMPHFfold111_x JUMPHFfold121_x JUMPHFfold131_x JUMPHFfold141_x JUMPHFfold151_x 
            set_jump_group JUMPGROUP3 JUMPHFfold111_y JUMPHFfold121_y JUMPHFfold131_y JUMPHFfold141_y JUMPHFfold151_y 
            set_jump_group JUMPGROUP4 JUMPHFfold111_z JUMPHFfold121_z JUMPHFfold131_z JUMPHFfold141_z JUMPHFfold151_z 
            set_jump_group JUMPGROUP5 JUMPHFfold111_sds JUMPHFfold121_sds JUMPHFfold131_sds JUMPHFfold141_sds JUMPHFfold151_sds 
            set_jump_group JUMPGROUP6 JUMPHFfold111_subunit JUMPHFfold121_subunit JUMPHFfold131_subunit JUMPHFfold141_subunit JUMPHFfold151_subunit 
          """)))

        # TODO: change the symmetry so that depending on if it is 4v4m or 1stm different symmetries have to be used

        fold3 = CubicSetup()
        fold3.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name 3fold
            E = 3*VRTHFfold111_sds + 3*(VRTHFfold111_sds:VRT3fold111_sds)
            anchor_residue COM
            virtual_coordinates_start
            {self.get_vrt("VRTglobal")}
            {self.get_vrt("VRTHFfold1_z_tref")}
            {self.get_vrt("VRTHFfold")}
            {self.get_vrt("VRTHFfold1")}
            {self.get_vrt("VRTHFfold1_z_rref")}
            {self.get_vrt("VRTHFfold1_z")}
            {self.get_vrt("VRTHFfold111_x_tref")}
            {self.get_vrt("VRTHFfold11")}
            {self.get_vrt("VRTHFfold111")}
            {self.get_vrt("VRTHFfold111_x_rref")}
            {self.get_vrt("VRTHFfold111_x")}
            {self.get_vrt("VRTHFfold111_y_rref")}
            {self.get_vrt("VRTHFfold111_y")}
            {self.get_vrt("VRTHFfold111_z_rref")}
            {self.get_vrt("VRTHFfold111_z")}
            {self.get_vrt("VRTHFfold111_sds")}
            {self.get_vrt("VRT3fold1_z_tref")}
            {self.get_vrt("VRT3fold")}
            {self.get_vrt("VRT3fold1")} 
            {self.get_vrt("VRT3fold1_z_rref")} 
            {self.get_vrt("VRT3fold1_z")} 
            {self.get_vrt("VRT3fold111_x_tref")} 
            {self.get_vrt("VRT3fold11")} 
            {self.get_vrt("VRT3fold111")} 
            {self.get_vrt("VRT3fold111_x_rref")} 
            {self.get_vrt("VRT3fold111_x")} 
            {self.get_vrt("VRT3fold111_y_rref")} 
            {self.get_vrt("VRT3fold111_y")} 
            {self.get_vrt("VRT3fold111_z_rref")}
            {self.get_vrt("VRT3fold111_z")} 
            {self.get_vrt("VRT3fold111_sds")} 
            {self.get_vrt("VRT2fold1_z_tref")} 
            {self.get_vrt("VRT2fold")} 
            {self.get_vrt("VRT2fold1")} 
            {self.get_vrt("VRT2fold121_x_tref")} 
            {self.get_vrt("VRT2fold1_z_rref")} 
            {self.get_vrt("VRT2fold1_z")} 
            {self.get_vrt("VRT2fold12")} 
            {self.get_vrt("VRT2fold121")} 
            {self.get_vrt("VRT2fold121_x_rref")} 
            {self.get_vrt("VRT2fold121_x")} 
            {self.get_vrt("VRT2fold121_y_rref")} 
            {self.get_vrt("VRT2fold121_y")} 
            {self.get_vrt("VRT2fold121_z_rref")} 
            {self.get_vrt("VRT2fold121_z")} 
            {self.get_vrt("VRT2fold121_sds")} 
            virtual_coordinates_stop  
            connect_virtual JUMPHFfold1_z_tref VRTglobal VRTHFfold1_z_tref
            connect_virtual JUMPHFfold VRTHFfold1_z_tref VRTHFfold
            connect_virtual JUMPHFfold1 VRTHFfold VRTHFfold1
            connect_virtual JUMPHFfold1_z_rref VRTHFfold1 VRTHFfold1_z_rref
            connect_virtual JUMPHFfold1_z VRTHFfold1_z_rref VRTHFfold1_z
            connect_virtual JUMPHFfold111_x_tref VRTHFfold1_z VRTHFfold111_x_tref
            connect_virtual JUMPHFfold11 VRTHFfold111_x_tref VRTHFfold11
            connect_virtual JUMPHFfold111 VRTHFfold11 VRTHFfold111
            connect_virtual JUMPHFfold111_x_rref VRTHFfold111 VRTHFfold111_x_rref
            connect_virtual JUMPHFfold111_x VRTHFfold111_x_rref VRTHFfold111_x
            connect_virtual JUMPHFfold111_y_rref VRTHFfold111_x VRTHFfold111_y_rref
            connect_virtual JUMPHFfold111_y VRTHFfold111_y_rref VRTHFfold111_y
            connect_virtual JUMPHFfold111_z_rref VRTHFfold111_y VRTHFfold111_z_rref
            connect_virtual JUMPHFfold111_z VRTHFfold111_z_rref VRTHFfold111_z
            connect_virtual JUMPHFfold111_sds VRTHFfold111_z VRTHFfold111_sds
            connect_virtual JUMPHFfold111_subunit VRTHFfold111_sds SUBUNIT
            connect_virtual JUMP3fold1_z_tref VRTglobal VRT3fold1_z_tref
            connect_virtual JUMP3fold VRT3fold1_z_tref VRT3fold
            connect_virtual JUMP3fold1 VRT3fold VRT3fold1
            connect_virtual JUMP3fold1_z_rref VRT3fold1 VRT3fold1_z_rref
            connect_virtual JUMP3fold1_z VRT3fold1_z_rref VRT3fold1_z
            connect_virtual JUMP3fold111_x_tref VRT3fold1_z VRT3fold111_x_tref
            connect_virtual JUMP3fold11 VRT3fold111_x_tref VRT3fold11
            connect_virtual JUMP3fold111 VRT3fold11 VRT3fold111
            connect_virtual JUMP3fold111_x_rref VRT3fold111 VRT3fold111_x_rref
            connect_virtual JUMP3fold111_x VRT3fold111_x_rref VRT3fold111_x
            connect_virtual JUMP3fold111_y_rref VRT3fold111_x VRT3fold111_y_rref
            connect_virtual JUMP3fold111_y VRT3fold111_y_rref VRT3fold111_y
            connect_virtual JUMP3fold111_z_rref VRT3fold111_y VRT3fold111_z_rref
            connect_virtual JUMP3fold111_z VRT3fold111_z_rref VRT3fold111_z
            connect_virtual JUMP3fold111_sds VRT3fold111_z VRT3fold111_sds
            connect_virtual JUMP3fold111_subunit VRT3fold111_sds SUBUNIT
            connect_virtual JUMP2fold1_z_tref VRTglobal VRT2fold1_z_tref
            connect_virtual JUMP2fold VRT2fold1_z_tref VRT2fold
            connect_virtual JUMP2fold1 VRT2fold VRT2fold1
            connect_virtual JUMP2fold1_z_rref VRT2fold1 VRT2fold1_z_rref
            connect_virtual JUMP2fold1_z VRT2fold1_z_rref VRT2fold1_z
            connect_virtual JUMP2fold121_x_tref VRT2fold1_z VRT2fold121_x_tref
            connect_virtual JUMP2fold12 VRT2fold121_x_tref VRT2fold12
            connect_virtual JUMP2fold121 VRT2fold12 VRT2fold121
            connect_virtual JUMP2fold121_x_rref VRT2fold121 VRT2fold121_x_rref
            connect_virtual JUMP2fold121_x VRT2fold121_x_rref VRT2fold121_x
            connect_virtual JUMP2fold121_y_rref VRT2fold121_x VRT2fold121_y_rref
            connect_virtual JUMP2fold121_y VRT2fold121_y_rref VRT2fold121_y
            connect_virtual JUMP2fold121_z_rref VRT2fold121_y VRT2fold121_z_rref
            connect_virtual JUMP2fold121_z VRT2fold121_z_rref VRT2fold121_z
            connect_virtual JUMP2fold121_sds VRT2fold121_z VRT2fold121_sds
            connect_virtual JUMP2fold121_subunit VRT2fold121_sds SUBUNIT
            set_dof JUMPHFfold1 z({symmetry_setup.dofs['JUMPHFfold1'][0][2]}) 
            set_dof JUMPHFfold1_z angle_z({symmetry_setup.dofs['JUMPHFfold1_z'][0][2]})
            set_dof JUMPHFfold111 x({symmetry_setup.dofs['JUMPHFfold111'][0][2]})
            set_dof JUMPHFfold111_x angle_x({symmetry_setup.dofs['JUMPHFfold111_x'][0][2]})
            set_dof JUMPHFfold111_y angle_y({symmetry_setup.dofs['JUMPHFfold111_y'][0][2]})
            set_dof JUMPHFfold111_z angle_z({symmetry_setup.dofs['JUMPHFfold111_z'][0][2]})
            set_jump_group JUMPGROUP1 JUMPHFfold1 JUMP3fold1 JUMP2fold1
            set_jump_group JUMPGROUP2 JUMPHFfold1_z JUMP3fold1_z JUMP2fold1_z
            set_jump_group JUMPGROUP3 JUMPHFfold111 JUMP3fold111 JUMP2fold121
            set_jump_group JUMPGROUP4 JUMPHFfold111_x JUMP3fold111_x JUMP2fold121_x
            set_jump_group JUMPGROUP5 JUMPHFfold111_y JUMP3fold111_y JUMP2fold121_y
            set_jump_group JUMPGROUP6 JUMPHFfold111_z JUMP3fold111_z JUMP2fold121_z
            set_jump_group JUMPGROUP7 JUMPHFfold111_sds JUMP3fold111_sds JUMP2fold121_sds
            set_jump_group JUMPGROUP8 JUMPHFfold111_subunit JUMP3fold111_subunit JUMP2fold121_subunit
            """)))

        fold2_1 = CubicSetup()
        fold2_1.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name 2fold_1
          E = 2*VRTHFfold111_sds + 1*(VRTHFfold111_sds:VRT2fold111_sds)
          anchor_residue COM
          virtual_coordinates_start
          {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRTHFfold1_z_tref")}
          {self.get_vrt("VRTHFfold")}
          {self.get_vrt("VRTHFfold1")}
          {self.get_vrt("VRTHFfold1_z_rref")}
          {self.get_vrt("VRTHFfold1_z")}
          {self.get_vrt("VRTHFfold111_x_tref")}
          {self.get_vrt("VRTHFfold11")}
          {self.get_vrt("VRTHFfold111")}
          {self.get_vrt("VRTHFfold111_x_rref")}
          {self.get_vrt("VRTHFfold111_x")}
          {self.get_vrt("VRTHFfold111_y_rref")}
          {self.get_vrt("VRTHFfold111_y")}
          {self.get_vrt("VRTHFfold111_z_rref")}
          {self.get_vrt("VRTHFfold111_z")}
          {self.get_vrt("VRTHFfold111_sds")}
          {self.get_vrt("VRT2fold1_z_tref")}
          {self.get_vrt("VRT2fold")}
          {self.get_vrt("VRT2fold1")}
          {self.get_vrt("VRT2fold1_z_rref")}
          {self.get_vrt("VRT2fold1_z")}
          {self.get_vrt("VRT2fold111_x_tref")}
          {self.get_vrt("VRT2fold11")}
          {self.get_vrt("VRT2fold111")}
          {self.get_vrt("VRT2fold111_x_rref")}
          {self.get_vrt("VRT2fold111_x")}
          {self.get_vrt("VRT2fold111_y_rref")}
          {self.get_vrt("VRT2fold111_y")}
          {self.get_vrt("VRT2fold111_z_rref")}
          {self.get_vrt("VRT2fold111_z")}
          {self.get_vrt("VRT2fold111_sds")}
          connect_virtual JUMPHFfold1_z_tref VRTglobal VRTHFfold1_z_tref
          connect_virtual JUMPHFfold VRTHFfold1_z_tref VRTHFfold
          connect_virtual JUMPHFfold1 VRTHFfold VRTHFfold1
        connect_virtual JUMPHFfold1_z_rref VRTHFfold1 VRTHFfold1_z_rref
        connect_virtual JUMPHFfold1_z VRTHFfold1_z_rref VRTHFfold1_z
        connect_virtual JUMPHFfold111_x_tref VRTHFfold1_z VRTHFfold111_x_tref
        connect_virtual JUMPHFfold11 VRTHFfold111_x_tref VRTHFfold11
        connect_virtual JUMPHFfold111 VRTHFfold11 VRTHFfold111
        connect_virtual JUMPHFfold111_x_rref VRTHFfold111 VRTHFfold111_x_rref
        connect_virtual JUMPHFfold111_x VRTHFfold111_x_rref VRTHFfold111_x
        connect_virtual JUMPHFfold111_y_rref VRTHFfold111_x VRTHFfold111_y_rref
        connect_virtual JUMPHFfold111_y VRTHFfold111_y_rref VRTHFfold111_y
        connect_virtual JUMPHFfold111_z_rref VRTHFfold111_y VRTHFfold111_z_rref
        connect_virtual JUMPHFfold111_z VRTHFfold111_z_rref VRTHFfold111_z
        connect_virtual JUMPHFfold111_sds VRTHFfold111_z VRTHFfold111_sds
        connect_virtual JUMPHFfold111_subunit VRTHFfold111_sds SUBUNIT
        connect_virtual JUMP2fold1_z_tref VRTglobal VRT2fold1_z_tref
        connect_virtual JUMP2fold VRT2fold1_z_tref VRT2fold
        connect_virtual JUMP2fold1 VRT2fold VRT2fold1
        connect_virtual JUMP2fold1_z_rref VRT2fold1 VRT2fold1_z_rref
        connect_virtual JUMP2fold1_z VRT2fold1_z_rref VRT2fold1_z
        connect_virtual JUMP2fold111_x_tref VRT2fold1_z VRT2fold111_x_tref
        connect_virtual JUMP2fold11 VRT2fold111_x_tref VRT2fold11
        connect_virtual JUMP2fold111 VRT2fold11 VRT2fold111
        connect_virtual JUMP2fold111_x_rref VRT2fold111 VRT2fold111_x_rref
        connect_virtual JUMP2fold111_x VRT2fold111_x_rref VRT2fold111_x
        connect_virtual JUMP2fold111_y_rref VRT2fold111_x VRT2fold111_y_rref
        connect_virtual JUMP2fold111_y VRT2fold111_y_rref VRT2fold111_y
        connect_virtual JUMP2fold111_z_rref VRT2fold111_y VRT2fold111_z_rref
        connect_virtual JUMP2fold111_z VRT2fold111_z_rref VRT2fold111_z
        connect_virtual JUMP2fold111_sds VRT2fold111_z VRT2fold111_sds
        connect_virtual JUMP2fold111_subunit VRT2fold111_sds SUBUNIT
        set_dof JUMPHFfold1 z({symmetry_setup.dofs['JUMPHFfold1'][0][2]}) 
        set_dof JUMPHFfold1_z angle_z({symmetry_setup.dofs['JUMPHFfold1_z'][0][2]})
        set_dof JUMPHFfold111 x({symmetry_setup.dofs['JUMPHFfold111'][0][2]})
        set_dof JUMPHFfold111_x angle_x({symmetry_setup.dofs['JUMPHFfold111_x'][0][2]})
        set_dof JUMPHFfold111_y angle_y({symmetry_setup.dofs['JUMPHFfold111_y'][0][2]})
        set_dof JUMPHFfold111_z angle_z({symmetry_setup.dofs['JUMPHFfold111_z'][0][2]})
        set_jump_group JUMPGROUP1 JUMPHFfold1 JUMP2fold1
        set_jump_group JUMPGROUP2 JUMPHFfold1_z JUMP2fold1_z
        set_jump_group JUMPGROUP3 JUMPHFfold111 JUMP2fold111
        set_jump_group JUMPGROUP4 JUMPHFfold111_x JUMP2fold111_x
        set_jump_group JUMPGROUP5 JUMPHFfold111_y JUMP2fold111_y
        set_jump_group JUMPGROUP6 JUMPHFfold111_z JUMP2fold111_z
        set_jump_group JUMPGROUP7 JUMPHFfold111_sds JUMP2fold111_sds
        set_jump_group JUMPGROUP8 JUMPHFfold111_subunit JUMP2fold111_subunit""")))

        fold2_2 = CubicSetup()
        fold2_2.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name fold2_2 
          E = 2*VRTHFfold111_sds + 1*(VRTHFfold111_sds:VRT3fold121_sds)
          anchor_residue COM
          virtual_coordinates_start
            {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRTHFfold1_z_tref")}
          {self.get_vrt("VRTHFfold")}
          {self.get_vrt("VRTHFfold1")}
          {self.get_vrt("VRTHFfold1_z_rref")}
          {self.get_vrt("VRTHFfold1_z")}
          {self.get_vrt("VRTHFfold111_x_tref")}
          {self.get_vrt("VRTHFfold11")}
          {self.get_vrt("VRTHFfold111")}
          {self.get_vrt("VRTHFfold111_x_rref")}
          {self.get_vrt("VRTHFfold111_x")}
          {self.get_vrt("VRTHFfold111_y_rref")}
          {self.get_vrt("VRTHFfold111_y")}
          {self.get_vrt("VRTHFfold111_z_rref")}
          {self.get_vrt("VRTHFfold111_z")}
          {self.get_vrt("VRTHFfold111_sds")}
          {self.get_vrt("VRT3fold1_z_tref")}
          {self.get_vrt("VRT3fold")}
          {self.get_vrt("VRT3fold1")}
          {self.get_vrt("VRT3fold1_z_rref")}
          {self.get_vrt("VRT3fold1_z")}
          {self.get_vrt("VRT3fold121_x_tref")}
          {self.get_vrt("VRT3fold12")}
          {self.get_vrt("VRT3fold121")}
          {self.get_vrt("VRT3fold121_x_rref")}
          {self.get_vrt("VRT3fold121_x")}
          {self.get_vrt("VRT3fold121_y_rref")}
          {self.get_vrt("VRT3fold121_y")}
          {self.get_vrt("VRT3fold121_z_rref")}
          {self.get_vrt("VRT3fold121_z")}
          {self.get_vrt("VRT3fold121_sds")}
          connect_virtual JUMPHFfold1_z_tref VRTglobal VRTHFfold1_z_tref
          connect_virtual JUMPHFfold VRTHFfold1_z_tref VRTHFfold
          connect_virtual JUMPHFfold1 VRTHFfold VRTHFfold1
        connect_virtual JUMPHFfold1_z_rref VRTHFfold1 VRTHFfold1_z_rref
        connect_virtual JUMPHFfold1_z VRTHFfold1_z_rref VRTHFfold1_z
        connect_virtual JUMPHFfold111_x_tref VRTHFfold1_z VRTHFfold111_x_tref
        connect_virtual JUMPHFfold11 VRTHFfold111_x_tref VRTHFfold11
        connect_virtual JUMPHFfold111 VRTHFfold11 VRTHFfold111
        connect_virtual JUMPHFfold111_x_rref VRTHFfold111 VRTHFfold111_x_rref
        connect_virtual JUMPHFfold111_x VRTHFfold111_x_rref VRTHFfold111_x
        connect_virtual JUMPHFfold111_y_rref VRTHFfold111_x VRTHFfold111_y_rref
        connect_virtual JUMPHFfold111_y VRTHFfold111_y_rref VRTHFfold111_y
        connect_virtual JUMPHFfold111_z_rref VRTHFfold111_y VRTHFfold111_z_rref
        connect_virtual JUMPHFfold111_z VRTHFfold111_z_rref VRTHFfold111_z
        connect_virtual JUMPHFfold111_sds VRTHFfold111_z VRTHFfold111_sds
        connect_virtual JUMPHFfold111_subunit VRTHFfold111_sds SUBUNIT
        connect_virtual JUMP3fold1_z_tref VRTglobal VRT3fold1_z_tref
        connect_virtual JUMP3fold VRT3fold1_z_tref VRT3fold
        connect_virtual JUMP3fold1 VRT3fold VRT3fold1
        connect_virtual JUMP3fold1_z_rref VRT3fold1 VRT3fold1_z_rref
        connect_virtual JUMP3fold1_z VRT3fold1_z_rref VRT3fold1_z
        connect_virtual JUMP3fold111_x_tref VRT3fold1_z VRT3fold121_x_tref
        connect_virtual JUMP3fold12 VRT3fold121_x_tref VRT3fold12
        connect_virtual JUMP3fold121 VRT3fold12 VRT3fold121
        connect_virtual JUMP3fold121_x_rref VRT3fold121 VRT3fold121_x_rref
        connect_virtual JUMP3fold121_x VRT3fold121_x_rref VRT3fold121_x
        connect_virtual JUMP3fold121_y_rref VRT3fold121_x VRT3fold121_y_rref
        connect_virtual JUMP3fold121_y VRT3fold121_y_rref VRT3fold121_y
        connect_virtual JUMP3fold121_z_rref VRT3fold121_y VRT3fold121_z_rref
        connect_virtual JUMP3fold121_z VRT3fold121_z_rref VRT3fold121_z
        connect_virtual JUMP3fold121_sds VRT3fold121_z VRT3fold121_sds
        connect_virtual JUMP3fold121_subunit VRT3fold121_sds SUBUNIT
        set_dof JUMPHFfold1 z({symmetry_setup.dofs['JUMPHFfold1'][0][2]}) 
        set_dof JUMPHFfold1_z angle_z({symmetry_setup.dofs['JUMPHFfold1_z'][0][2]})
        set_dof JUMPHFfold111 x({symmetry_setup.dofs['JUMPHFfold111'][0][2]})
        set_dof JUMPHFfold111_x angle_x({symmetry_setup.dofs['JUMPHFfold111_x'][0][2]})
        set_dof JUMPHFfold111_y angle_y({symmetry_setup.dofs['JUMPHFfold111_y'][0][2]})
        set_dof JUMPHFfold111_z angle_z({symmetry_setup.dofs['JUMPHFfold111_z'][0][2]})
        set_jump_group JUMPGROUP1 JUMPHFfold1 JUMP3fold1
        set_jump_group JUMPGROUP2 JUMPHFfold1_z JUMP3fold1_z
        set_jump_group JUMPGROUP3 JUMPHFfold111 JUMP3fold121
        set_jump_group JUMPGROUP4 JUMPHFfold111_x JUMP3fold121_x
        set_jump_group JUMPGROUP5 JUMPHFfold111_y JUMP3fold121_y
        set_jump_group JUMPGROUP6 JUMPHFfold111_z JUMP3fold121_z
        set_jump_group JUMPGROUP7 JUMPHFfold111_sds JUMP3fold121_sds
        set_jump_group JUMPGROUP8 JUMPHFfold111_subunit JUMP3fold121_subunit
        """)))

        return fold5, fold3, fold2_1, fold2_2

    def get_5fold_center_(self):
        raise ValueError
        return self.get_vrt("VRTHFfold111").vrt_orig

    def get_3fold_plane_from_HFfold(self):
        a = self.get_vrt("VRTHFfold111_z").vrt_orig
        b = self.get_vrt("VRT2fold121_z").vrt_orig
        c = self.get_vrt("VRT3fold111_z").vrt_orig
        return a, b, c

    def get_3fold_center_from_HFfold(self):
        """Returns the center of the 3-fold"""
        try: # for icosahedral symmetry
            a, b, c = self.get_3fold_plane_from_HFfold()
            return (a + b + c) / 3
        except ValueError:
            raise NotImplementedError("Only works for icosahedral symmetry")

    def get_2fold_center_from_HFfold(self):
        """Returns the center of the 3-fold"""
        try: # for icosahedral symmetry
            a = self.get_vrt("VRTHFfold111_z").vrt_orig
            b = self.get_vrt("VRT2fold111_z").vrt_orig
            return (a + b) / 2
        except ValueError:
            raise NotImplementedError("Only works for icosahedral symmetry")

    def calculate_if_rightanded(self):
        """Returns true if the point fold3_axis going to fold2_axis relative to the foldHF_axis is right-handed. It is left-handed if the cross product fold3_axis X fold2_axis
        points in the same direction as the foldF_axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        if self.is_hf_based():
            foldHF_axis = -self.get_vrt("VRTHFfold1").vrt_z
            fold3_axis = -self.get_vrt("VRT3fold1").vrt_z
            fold2_axis = -self.get_vrt("VRT2fold1").vrt_z
            return self._right_handed_vectors(fold3_axis, fold2_axis, foldHF_axis)
        elif self.is_3f_based():
            raise NotImplementedError("This does not work for O 3-fold based symmetry for instance as the left and right hands are identical"
                                      " and you would have to use previous information of the HF fold from which it was based.")

    @staticmethod
    def _create_final_ref_dofs(ss_f, ss_t, ss_f_nb1, ss_f_nb2, ss_t_nb1, ss_t_nb2, R=None, f="rotate",
                               suffix="", make_sds=True):  # ss, nb1:str, nb2:str, suffix:str= ""):
        assert make_sds == True, "We need sds VRT's in the SymDefSwapper class"
        if not R is None:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x_rref{suffix}",
                                       f"VRT{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x{suffix}").__getattribute__(f)(R,
                                                                                                                                      True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y_rref{suffix}",
                                       f"VRT{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y{suffix}").__getattribute__(f)(R,
                                                                                                                                      True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z_rref{suffix}",
                                       f"VRT{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}").__getattribute__(f)(R,
                                                                                                                                      True))
            # add sds vrt
            if make_sds:
                ss_t.add_vrt(
                    ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_sds{suffix}").__getattribute__(f)(R,
                                                                                                                                          True))
        else:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}"))
            # add sds vrt
            if make_sds:
                ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_sds{suffix}"))
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_x{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_x{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_y{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_y{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}",
                      f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}")
        # add sds jump
        if make_sds:
            ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_sds{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}",
                          f"VRT{ss_t_nb1}fold{ss_t_nb2}_sds{suffix}")
            ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_subunit{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_sds{suffix}", "SUBUNIT")
        else:
            ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_subunit{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}", "SUBUNIT")

    def straightinator(self, cs):
        """Makes the last VRT's that control the COM rotation are set identical to their parent VRT '=straighten'. This is
        important when we flip the subunits during EvoDOCK"""
        for vrts in [cs.get_downstream_connections(j)[:-2] for j in cs.jumpgroups["JUMPGROUP3"]]:
            for n, vrt in enumerate(vrts):
                if n == 0:
                    vrt_to_copy = copy.deepcopy(cs.get_unapplied_vrt(vrt))
                else:
                    vrt_to_replace = cs.get_unapplied_vrt(vrt)
                    vrt_to_replace.vrt_orig = vrt_to_copy.vrt_orig
                    vrt_to_replace.vrt_x = vrt_to_copy.vrt_x
                    vrt_to_replace.vrt_y = vrt_to_copy.vrt_y
                    vrt_to_replace.vrt_z = vrt_to_copy.vrt_z
                    if "x_rref" in vrt_to_replace.name:
                        vrt_to_replace.vrt_orig = self.add_along_vector(vrt_to_replace.vrt_orig, vrt_to_replace.vrt_x)
                    elif "y_rref" in vrt_to_replace.name:
                        vrt_to_replace.vrt_orig = self.add_along_vector(vrt_to_replace.vrt_orig, vrt_to_replace.vrt_y)
                    elif "z_rref" in vrt_to_replace.name:
                        vrt_to_replace.vrt_orig = self.add_along_vector(vrt_to_replace.vrt_orig, vrt_to_replace.vrt_z)
                    cs.init_vrts[cs.init_vrts.index(vrt_to_replace)] = vrt_to_replace


    @staticmethod
    def _create_base_dofs(ss_f, ss_t, ss_f_nb1, ss_t_nb1, base_vrt, R=None, f="rotate", suffix=""):
        """Based on VRTXfold create: VRTXfold1_z_tref -> VRTXfold (created before) -> VRTXfold1 -> VRTXfold1_z_rref -> VRTXfold1_z """
        ss_t.add_vrt(base_vrt)
        if not R is None:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_tref{suffix}", move_origo=True, axis="z",
                                       dir=1).__getattribute__(f)(R, True))
            # ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_rref{suffix}", move_origo=True,
                                       axis="z").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z{suffix}").__getattribute__(f)(R, True))
        else:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_tref{suffix}", move_origo=True, axis="z", dir=1))
            # ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_rref{suffix}", move_origo=True, axis="z"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z{suffix}"))
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1_z_tref{suffix}", f"VRTglobal{suffix}", f"VRT{ss_t_nb1}fold1_z_tref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_tref{suffix}", f"VRT{ss_t_nb1}fold{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1{suffix}", f"VRT{ss_t_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1_z_rref{suffix}", f"VRT{ss_t_nb1}fold1{suffix}", f"VRT{ss_t_nb1}fold1_z_rref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1_z{suffix}", f"VRT{ss_t_nb1}fold1_z_rref{suffix}", f"VRT{ss_t_nb1}fold1_z{suffix}")

    @staticmethod
    def _create_chain_connection(ss_f, ss_t, ss_f_nb1, ss_f_nb2, ss_t_nb1, ss_t_nb2, R=None, f="rotate", suffix=""):
        """From VRTXfold1_z create: VRTXfold1_z -> VRTXfold1_x_tref -> VRTXfold1Y -> VRTXfold1Y1"""
        if not R is None:
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}1_x_tref{suffix}", move_origo=True,
                              axis="x", dir=1).__getattribute__(f)(R, True))
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}1{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}{suffix}").__getattribute__(f)(R,
                                                                                                                                     True))
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}1{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}1{suffix}").__getattribute__(f)(R,
                                                                                                                                      True))
        else:
            ss_t.add_vrt(
                ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}1_x_tref{suffix}", move_origo=True,
                              axis="x", dir=1))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold1{ss_f_nb2}1{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}1{suffix}"))
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1{ss_t_nb2}1_x_tref{suffix}", f"VRT{ss_t_nb1}fold1_z{suffix}",
                      f"VRT{ss_t_nb1}fold1{ss_t_nb2}1_x_tref{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1{ss_t_nb2}{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}1_x_tref{suffix}",
                      f"VRT{ss_t_nb1}fold1{ss_t_nb2}{suffix}")
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold1{ss_t_nb2}1{suffix}", f"VRT{ss_t_nb1}fold1{ss_t_nb2}{suffix}",
                      f"VRT{ss_t_nb1}fold1{ss_t_nb2}1{suffix}")



    def create_O_3fold_based_symmetry(self, suffix="", make_sds=True, straighten_COM=True):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        ss3.headers = self.headers
        ss3.righthanded = self.righthanded
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        # colors are as given in HF symmetry
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss3.energies = " + ".join((
           f"24*VRT31fold111_{last}{suffix}", # green
           f"24*(VRT31fold111_{last}{suffix}:VRT32fold111_{last}{suffix})", # ligth blue (4fold closest)
           f"24*(VRT31fold111_{last}{suffix}:VRT34fold111_{last}{suffix})", # pink (4fold furthest)
           f"24*(VRT31fold111_{last}{suffix}:VRT31fold121_{last}{suffix})", # brown (other 3fold)
           f"12*(VRT31fold111_{last}{suffix}:VRT32fold121_{last}{suffix})", # dark blue (2fold closest)
           f"12*(VRT31fold111_{last}{suffix}:VRT33fold131_{last}{suffix})")) # white

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss3.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        #######################
        # Create first 3 fold #
        #######################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 3-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt31fold = ss3.copy_vrt(f"VRTglobal{suffix}", f"VRT31fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt31fold.vrt_y, 180)
        vrt31fold.rotate_right_multiply(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt31fold.vrt_z, 180)
        vrt31fold.rotate_right_multiply(R)
        # 3) then rotate to the 3-fold
        center3 = setup_applied_dofs.get_3fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center3, - vrt31fold.vrt_z), vector_angle(center3, - vrt31fold.vrt_z))
        vrt31fold.rotate_right_multiply(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center3  # needs to be rotated onto the 3fold plane
        rot_angle = vector_angle(anchor_resi_vec, - vrt31fold.vrt_x)
        # find out which way to rotate
        if self._right_handed_vectors(anchor_resi_vec, vrt31fold.vrt_x, center3):
            R = rotation_matrix(center3, rot_angle)
        else:
            R = rotation_matrix(center3, - rot_angle)
        vrt31fold.rotate_right_multiply(R)
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRTglobal", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT31fold1{suffix}"))
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="31", base_vrt=vrt31fold)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_tref{suffix}", move_origo=True, axis="z", dir=1))
        # ss3.add_jump(f"JUMP31fold_z_tref{suffix}", f"VRTglobal{suffix}", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRT31fold1_z_tref{suffix}", f"VRT31fold{suffix}")
        # ss3.add_jump(f"JUMP31fold1{suffix}", f"VRT31fold{suffix}", f"VRT31fold1{suffix}")

        # ---- chain 1 ----
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}", move_origo=True, axis="z"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z{suffix}"))
        # ss3.add_jump(f"JUMP31fold_z_rref{suffix}", f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}")
        # ss3.add_jump(f"JUMP31fold_z{suffix}", f"VRT31fold1_z_rref{suffix}", f"VRT31fold1_z{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="", ss_t_nb1="31", ss_t_nb2="1", suffix=suffix)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold111_x_tref{suffix}", move_origo=True, axis="x", dir=1))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold11{suffix}"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # ss3.add_jump(f"JUMP31fold111_x_tref{suffix}", f"VRT31fold1_z{suffix}", f"VRT31fold111_x_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold11{suffix}", f"VRT31fold111_x_tref{suffix}", f"VRT31fold11{suffix}")
        # ss3.add_jump(f"JUMP31fold111{suffix}", f"VRT31fold11{suffix}", f"VRT31fold111{suffix}")
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        ##############################
        # Create surrounding 3 folds #
        ##############################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 90)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT32fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="32", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT32fold1{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold{suffix}", f"VRTglobal{suffix}", f"VRT32fold{suffix}")
        # ss3.add_jump(f"JUMP32fold1{suffix}", f"VRT32fold{suffix}", f"VRT32fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="32", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT32fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="32", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold11{suffix}", f"VRT32fold1{suffix}", f"VRT32fold11{suffix}")
        # ss3.add_jump(f"JUMP32fold111{suffix}", f"VRT32fold11{suffix}", f"VRT32fold111{suffix}")
        # ss3.add_jump(f"JUMP32fold1111{suffix}", f"VRT32fold111{suffix}", f"VRT32fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="2", ss_t_nb1="32", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="121", ss_t_nb1="32", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold12{suffix}", f"VRT32fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold121{suffix}", f"VRT32fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1211{suffix}", f"VRT32fold1211{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold12{suffix}", f"VRT32fold1{suffix}", f"VRT32fold12{suffix}")
        # ss3.add_jump(f"JUMP32fold121{suffix}", f"VRT32fold12{suffix}", f"VRT32fold121{suffix}")
        # ss3.add_jump(f"JUMP32fold1211{suffix}", f"VRT32fold121{suffix}", f"VRT32fold1211{suffix}")
        # ss3.add_jump(f"JUMP32fold1111_subunit{suffix}", f"VRT32fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP32fold1211_subunit{suffix}", f"VRT32fold1211{suffix}", "SUBUNIT")

        # -- - 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], - 90)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="33", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt35fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT35fold{suffix}").rotate_right_multiply(R, True)
        # vrt35fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT35fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt35fold)
        # ss3.add_vrt(vrt35fold1)
        # ss3.add_jump(f"JUMP35fold{suffix}", f"VRTglobal{suffix}", f"VRT35fold{suffix}")
        # ss3.add_jump(f"JUMP35fold1{suffix}", f"VRT35fold{suffix}", f"VRT35fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="33", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="33", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT35fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT35fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT35fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold11{suffix}", f"VRT35fold1{suffix}", f"VRT35fold11{suffix}")
        # ss3.add_jump(f"JUMP35fold111{suffix}", f"VRT35fold11{suffix}", f"VRT35fold111{suffix}")
        # ss3.add_jump(f"JUMP35fold1111{suffix}", f"VRT35fold111{suffix}", f"VRT35fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="3", ss_t_nb1="33", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="131", ss_t_nb1="33", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold13{suffix}", f"VRT35fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold131{suffix}", f"VRT35fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1311{suffix}", f"VRT35fold1311{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold13{suffix}", f"VRT35fold1{suffix}", f"VRT35fold13{suffix}")
        # ss3.add_jump(f"JUMP35fold131{suffix}", f"VRT35fold13{suffix}", f"VRT35fold131{suffix}")
        # ss3.add_jump(f"JUMP35fold1311{suffix}", f"VRT35fold131{suffix}", f"VRT35fold1311{suffix}")
        # ss3.add_jump(f"JUMP35fold1111_subunit{suffix}", f"VRT35fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP35fold1311_subunit{suffix}", f"VRT35fold1311{suffix}", "SUBUNIT")

        #######################################################
        # Create last 2 3-folds that are part of the fourfold #
        #######################################################
        # -- 144 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 180)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="34", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt33fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        # vrt33fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT33fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt33fold)
        # ss3.add_vrt(vrt33fold1)
        # ss3.add_jump(f"JUMP33fold{suffix}", f"VRTglobal{suffix}", f"VRT33fold{suffix}")
        # ss3.add_jump(f"JUMP33fold1{suffix}", f"VRT33fold{suffix}", f"VRT33fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="34", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="34", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT33fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT33fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT33fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP33fold11{suffix}", f"VRT33fold1{suffix}", f"VRT33fold11{suffix}")
        # ss3.add_jump(f"JUMP33fold111{suffix}", f"VRT33fold11{suffix}", f"VRT33fold111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111{suffix}", f"VRT33fold111{suffix}", f"VRT33fold1111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111_subunit{suffix}", f"VRT33fold1111{suffix}", "SUBUNIT")

        # # -- - 144 rotation --
        # # ---- create base ----
        # R = rotation_matrix([0, 0, 1], - 72 * 2)
        # base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="34", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # # vrt34fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # # vrt34fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT34fold1{suffix}").rotate_right_multiply(R, True)
        # # ss3.add_vrt(vrt34fold)
        # # ss3.add_vrt(vrt34fold1)
        # # ss3.add_jump(f"JUMP34fold{suffix}", f"VRTglobal{suffix}", f"VRT34fold{suffix}")
        # # ss3.add_jump(f"JUMP34fold1{suffix}", f"VRT34fold{suffix}", f"VRT34fold1{suffix}")
        # # ---- chain 1 ----
        # ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="34", ss_t_nb2="1", R=R, f="rotate_right_multiply",
        #                              suffix=suffix)
        # ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="34", ss_t_nb2="111", R=R, f="rotate_right_multiply",
        #                            suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT34fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT34fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT34fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP34fold11{suffix}", f"VRT34fold1{suffix}", f"VRT34fold11{suffix}")
        # ss3.add_jump(f"JUMP34fold111{suffix}", f"VRT34fold11{suffix}", f"VRT34fold111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111{suffix}", f"VRT34fold111{suffix}", f"VRT34fold1111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111_subunit{suffix}", f"VRT34fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss3.add_dof(f"JUMP31fold1{suffix}", 'z', "translation", np.linalg.norm(center3))
        ss3.add_dof(f"JUMP31fold1_z{suffix}", 'z', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center3))
        ss3.add_dof(f"JUMP31fold111_x{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_y{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_z{suffix}", 'z', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
        ss3.add_jumpgroup("JUMPGROUP1", f"JUMP31fold1{suffix}", f"JUMP32fold1{suffix}",  f"JUMP33fold1{suffix}",
                          f"JUMP34fold1{suffix}")
        ss3.add_jumpgroup("JUMPGROUP2", f"JUMP31fold1_z{suffix}", f"JUMP32fold1_z{suffix}",
                          f"JUMP33fold1_z{suffix}", f"JUMP34fold1_z{suffix}")
        ss3.add_jumpgroup("JUMPGROUP3", f"JUMP31fold111{suffix}", f"JUMP31fold121{suffix}", f"JUMP31fold131{suffix}",
                          f"JUMP32fold111{suffix}", f"JUMP32fold121{suffix}",
                          f"JUMP33fold111{suffix}", f"JUMP33fold131{suffix}", f"JUMP34fold111{suffix}")
        ss3.add_jumpgroup("JUMPGROUP4", f"JUMP31fold111_x{suffix}", f"JUMP31fold121_x{suffix}", f"JUMP31fold131_x{suffix}",
                          f"JUMP32fold111_x{suffix}", f"JUMP32fold121_x{suffix}",
                          f"JUMP33fold111_x{suffix}", f"JUMP33fold131_x{suffix}", f"JUMP34fold111_x{suffix}")
        ss3.add_jumpgroup("JUMPGROUP5", f"JUMP31fold111_y{suffix}", f"JUMP31fold121_y{suffix}", f"JUMP31fold131_y{suffix}",
                          f"JUMP32fold111_y{suffix}", f"JUMP32fold121_y{suffix}",
                          f"JUMP33fold111_y{suffix}", f"JUMP33fold131_y{suffix}", f"JUMP34fold111_y{suffix}")
        ss3.add_jumpgroup("JUMPGROUP6", f"JUMP31fold111_z{suffix}", f"JUMP31fold121_z{suffix}", f"JUMP31fold131_z{suffix}",
                          f"JUMP32fold111_z{suffix}", f"JUMP32fold121_z{suffix}",
                          f"JUMP33fold111_z{suffix}", f"JUMP33fold131_z{suffix}", f"JUMP34fold111_z{suffix}")
        ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_sds{suffix}", f"JUMP31fold121_sds{suffix}",
                          f"JUMP31fold131_sds{suffix}", f"JUMP32fold111_sds{suffix}", f"JUMP32fold121_sds{suffix}",
                          f"JUMP33fold111_sds{suffix}", f"JUMP33fold131_sds{suffix}", f"JUMP34fold111_sds{suffix}")
        ss3.add_jumpgroup("JUMPGROUP8", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP33fold111_subunit{suffix}", f"JUMP33fold131_subunit{suffix}", f"JUMP34fold111_subunit{suffix}")
        ss3._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss3)
        return ss3

    #         # If T, skip adding bonus to the extra-3-fold subunit since it is the same as the other one
    #         if self.get_symmetry() == "T":
    #            Perfect for 1MOG but not so for 4CIY
    #             setup.energies = "12*VRTHFfold111_z + " \ green
    #                              "12*(VRTHFfold111_z:VRTHFfold121_z) + " \ light blue
    #                              "12*(VRTHFfold111_z:VRT3fold111_z) + " \ yellow
    #                              "6*(VRTHFfold111_z:VRT2fold111_z) + " \ white
    #                              "6*(VRTHFfold111_z:VRT3fold121_z)".format(*self.get_energies()) \ brown
    #         else:
    #             setup.energies = "24*VRTHFfold111_z + " \ green
    #                              "24*(VRTHFfold111_z:VRTHFfold121_z) + " \ light blue
    #                              "24*(VRTHFfold111_z:VRTHFfold131_z) + " \ pink
    #                              "24*(VRTHFfold111_z:VRT3fold111_z) + " \ brown
    #                              "12*(VRTHFfold111_z:VRT2fold111_z) + " \ dark blue
    #                              "12*(VRTHFfold111_z:VRT3fold121_z)".format(*self.get_energies()) \ white

    #     if self.get_symmetry() == "I":
    #             return ("60", "60", "60", "60", "30", "30")
    #         elif self.get_symmetry() == "O":
    #             return ("24", "24", "24", "24", "12", "12")
    #         else: # self.get_symmetry() == "T":
    #             return ("12", "12", "12", "6", "6")

    def create_O_2fold_based_symmetry(self, suffix='', make_sds=True, straighten_COM=True):
        """Creates a 2-fold based symmetry file from a HF-based (CURRENTLY ONLY 5-fold) one."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        ss2.headers = self.headers
        ss2.righthanded = self.righthanded

        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss2.energies = " + ".join((
           f"24*VRT21fold111_{last}{suffix}", # green
           f"24*(VRT21fold111_{last}{suffix}:VRT24fold111_{last}{suffix})", # ligth blue (4fold closest)
           f"24*(VRT21fold111_{last}{suffix}:VRT23fold111_{last}{suffix})", # pink (4fold furthest)
           f"24*(VRT21fold111_{last}{suffix}:VRT22fold121_{last}{suffix})", # brown (other 3fold)
           f"12*(VRT21fold111_{last}{suffix}:VRT21fold121_{last}{suffix})", # dark blue (2fold closest)
           f"12*(VRT21fold111_{last}{suffix}:VRT25fold111_{last}{suffix})")) # white

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss2.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        ############################
        # Create first full 2 fold #
        ############################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 2-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt21fold = ss2.copy_vrt(f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt21fold.vrt_y, 180)
        vrt21fold.rotate(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt21fold.vrt_z, 180)
        vrt21fold.rotate(R)
        # 3) then rotate to the 2-fold
        center2 = setup_applied_dofs.get_2fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center2, -vrt21fold.vrt_z), -vector_angle(center2, -vrt21fold.vrt_z))
        vrt21fold.rotate(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2  # needs to be rotated onto the 2fold plane
        #         R = rotation_matrix(center3, -vector_angle(anchor_resi_vec, - vrt31fold.vrt_x))
        rot_angle = vector_angle(anchor_resi_vec, -vrt21fold.vrt_x)
        if self._right_handed_vectors(anchor_resi_vec, vrt21fold.vrt_x, center2):
            R = rotation_matrix(center2, -rot_angle)
        else:
            R = rotation_matrix(center2, rot_angle)
        is_righthanded = self.calculate_if_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(vrt21fold)
        # ss2.add_jump(f"JUMP21fold{suffix}", f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT21fold1{suffix}"))
        # ss2.add_jump(f"JUMP21fold1{suffix}", f"VRT21fold{suffix}", f"VRT21fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT21fold11{suffix}"))
        # # VRT21fold111 and VRT21fold1111 we want to make exactly like VRTHFfold1111 because we want to be able to
        # # transfer the rotations of that directly between a fivefold setup to a threefold setup
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold111", f"VRT21fold111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold1111", f"VRT21fold1111{suffix}"))
        # ###
        # ss2.add_jump(f"JUMP21fold11{suffix}", f"VRT21fold1{suffix}", f"VRT21fold11{suffix}")
        # ss2.add_jump(f"JUMP21fold111{suffix}", f"VRT21fold11{suffix}", f"VRT21fold111{suffix}")
        # ss2.add_jump(f"JUMP21fold1111{suffix}", f"VRT21fold111{suffix}", f"VRT21fold1111{suffix}")
        # ---- chain 2 ----

        R = rotation_matrix(center2, 180)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="21", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT21fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT21fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT21fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP21fold12{suffix}", f"VRT21fold1{suffix}", f"VRT21fold12{suffix}")
        # ss2.add_jump(f"JUMP21fold121{suffix}", f"VRT21fold12{suffix}", f"VRT21fold121{suffix}")
        # ss2.add_jump(f"JUMP21fold1211{suffix}", f"VRT21fold121{suffix}", f"VRT21fold1211{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP21fold1111_subunit{suffix}", f"VRT21fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP21fold1211_subunit{suffix}", f"VRT21fold1211{suffix}", "SUBUNIT")

        ########################
        # Create second 2 fold #
        ########################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 90 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="22", base_vrt=vrt_base, R=R)

        # vrt22fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        # vrt22fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT22fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt22fold)
        # ss2.add_vrt(vrt22fold1)
        # ss2.add_jump(f"JUMP22fold{suffix}", f"VRTglobal{suffix}", f"VRT22fold{suffix}")
        # ss2.add_jump(f"JUMP22fold1{suffix}", f"VRT22fold{suffix}", f"VRT22fold1{suffix}")
        # ---- chain 1 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="22", ss_t_nb2="1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold12{suffix}", f"VRT22fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold121{suffix}", f"VRT22fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1211{suffix}", f"VRT22fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold12{suffix}", f"VRT22fold1{suffix}", f"VRT22fold12{suffix}")
        # ss2.add_jump(f"JUMP22fold121{suffix}", f"VRT22fold12{suffix}", f"VRT22fold121{suffix}")
        # ss2.add_jump(f"JUMP22fold1211{suffix}", f"VRT22fold121{suffix}", f"VRT22fold1211{suffix}")
        # ss2.add_jump(f"JUMP22fold1111_subunit{suffix}", f"VRT22fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP22fold1211_subunit{suffix}", f"VRT22fold1211{suffix}", "SUBUNIT")

        ############################################
        # Create rest of the 2 folds in the 5-fold #
        ############################################
        for i in range(2, 4):
            R = rotation_matrix([0, 0, 1], 90 * i * (1 if is_righthanded else -1))
            # ---- create base ----
            n = str(i + 1)
            vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"2{n}", base_vrt=vrt_base, R=R)
            ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"", ss_t_nb1=f"2{n}", ss_t_nb2=f"1", R=R, suffix=suffix)
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)

            # vrt2nfold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            # vrt2nfold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT2{n}fold1{suffix}").rotate(R, True)
            # ss2.add_vrt(vrt2nfold)
            # ss2.add_vrt(vrt2nfold1)
            # ss2.add_jump(f"JUMP2{n}fold{suffix}", f"VRTglobal{suffix}", f"VRT2{n}fold{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1{suffix}", f"VRT2{n}fold{suffix}", f"VRT2{n}fold1{suffix}")
            # # ---- chain 1 ----
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT2{n}fold11{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT2{n}fold111{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT2{n}fold1111{suffix}").rotate(R, True))
            # ss2.add_jump(f"JUMP2{n}fold11{suffix}", f"VRT2{n}fold1{suffix}", f"VRT2{n}fold11{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold111{suffix}", f"VRT2{n}fold11{suffix}", f"VRT2{n}fold111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111{suffix}", f"VRT2{n}fold111{suffix}", f"VRT2{n}fold1111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111_subunit{suffix}", f"VRT2{n}fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(self.find_center_between_vtrs(setup_applied_dofs, "VRTHFfold111_z", "VRT3fold121_z"), 180)
        # ---- create base ----
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT25fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"25", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1=f"25", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"25", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)

        # vrt26fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT26fold{suffix}").rotate(R, True)
        # vrt26fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT26fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt26fold)
        # ss2.add_vrt(vrt26fold1)
        # ss2.add_jump(f"JUMP26fold{suffix}", f"VRTglobal{suffix}", f"VRT26fold{suffix}")
        # ss2.add_jump(f"JUMP26fold1{suffix}", f"VRT26fold{suffix}", f"VRT26fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT26fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT26fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT26fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP26fold11{suffix}", f"VRT26fold1{suffix}", f"VRT26fold11{suffix}")
        # ss2.add_jump(f"JUMP26fold111{suffix}", f"VRT26fold11{suffix}", f"VRT26fold111{suffix}")
        # ss2.add_jump(f"JUMP26fold1111{suffix}", f"VRT26fold111{suffix}", f"VRT26fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP26fold1111_subunit{suffix}", f"VRT26fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(-self.get_vrt("VRT3fold1")._vrt_z, 90 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT26fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="22", ss_t_nb1=f"26", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="22", ss_f_nb2="", ss_t_nb1=f"26", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"26", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)
        # # ---- create base ----
        # vrt27fold = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT27fold{suffix}").rotate(R, True)
        # vrt27fold1 = ss2.copy_vrt(f"VRT22fold1{suffix}", f"VRT27fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt27fold)
        # ss2.add_vrt(vrt27fold1)
        # ss2.add_jump(f"JUMP27fold{suffix}", f"VRTglobal{suffix}", f"VRT27fold{suffix}")
        # ss2.add_jump(f"JUMP27fold1{suffix}", f"VRT27fold{suffix}", f"VRT27fold1{suffix}")
        # # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold11{suffix}", f"VRT27fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold111{suffix}", f"VRT27fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold1111{suffix}", f"VRT27fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP27fold11{suffix}", f"VRT27fold1{suffix}", f"VRT27fold11{suffix}")
        # ss2.add_jump(f"JUMP27fold111{suffix}", f"VRT27fold11{suffix}", f"VRT27fold111{suffix}")
        # ss2.add_jump(f"JUMP27fold1111{suffix}", f"VRT27fold111{suffix}", f"VRT27fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP27fold1111_subunit{suffix}", f"VRT27fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss2.add_dof(f"JUMP21fold1{suffix}", 'z', "translation", np.linalg.norm(center2))
        ss2.add_dof(f"JUMP21fold1_z{suffix}", 'z', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2))
        ss2.add_dof(f"JUMP21fold111_x{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_y{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_z{suffix}", 'z', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
        ss2.add_jumpgroup("JUMPGROUP1", f"JUMP21fold1{suffix}", f"JUMP22fold1{suffix}", f"JUMP23fold1{suffix}", f"JUMP24fold1{suffix}",
                           f"JUMP25fold1{suffix}", f"JUMP26fold1{suffix}")
        ss2.add_jumpgroup("JUMPGROUP2", f"JUMP21fold1_z{suffix}", f"JUMP22fold1_z{suffix}", f"JUMP23fold1_z{suffix}",
                          f"JUMP24fold1_z{suffix}",  f"JUMP25fold1_z{suffix}", f"JUMP26fold1_z{suffix}")
        ss2.add_jumpgroup("JUMPGROUP3", f"JUMP21fold111{suffix}", f"JUMP21fold121{suffix}", f"JUMP22fold111{suffix}",
                          f"JUMP22fold121{suffix}", f"JUMP23fold111{suffix}", f"JUMP24fold111{suffix}",
                          f"JUMP25fold111{suffix}", f"JUMP26fold111{suffix}")
        ss2.add_jumpgroup("JUMPGROUP4", f"JUMP21fold111_x{suffix}", f"JUMP21fold121_x{suffix}", f"JUMP22fold111_x{suffix}",
                          f"JUMP22fold121_x{suffix}", f"JUMP23fold111_x{suffix}", f"JUMP24fold111_x{suffix}",
                           f"JUMP25fold111_x{suffix}", f"JUMP26fold111_x{suffix}")
        ss2.add_jumpgroup("JUMPGROUP5", f"JUMP21fold111_y{suffix}", f"JUMP21fold121_y{suffix}", f"JUMP22fold111_y{suffix}",
                          f"JUMP22fold121_y{suffix}", f"JUMP23fold111_y{suffix}", f"JUMP24fold111_y{suffix}",
                           f"JUMP25fold111_y{suffix}", f"JUMP26fold111_y{suffix}")
        ss2.add_jumpgroup("JUMPGROUP6", f"JUMP21fold111_z{suffix}", f"JUMP21fold121_z{suffix}", f"JUMP22fold111_z{suffix}",
                          f"JUMP22fold121_z{suffix}", f"JUMP23fold111_z{suffix}", f"JUMP24fold111_z{suffix}",
                           f"JUMP25fold111_z{suffix}", f"JUMP26fold111_z{suffix}")
        ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_sds{suffix}", f"JUMP21fold121_sds{suffix}",
                          f"JUMP22fold111_sds{suffix}", f"JUMP22fold121_sds{suffix}", f"JUMP23fold111_sds{suffix}",
                          f"JUMP24fold111_sds{suffix}", f"JUMP25fold111_sds{suffix}",
                          f"JUMP26fold111_sds{suffix}")
        ss2.add_jumpgroup("JUMPGROUP8", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}", f"JUMP25fold111_subunit{suffix}",
                          f"JUMP26fold111_subunit{suffix}")
        ss2._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss2)
        return ss2

    def create_T_3fold_based_symmetry(self, suffix="", make_sds=True, straighten_COM=True):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        ss3.headers = self.headers
        ss3.righthanded = self.righthanded
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        # colors are as given in HF symmetry
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss3.energies = " + ".join((
           f"12*VRT31fold111_{last}{suffix}", # green
           f"12*(VRT31fold111_{last}{suffix}:VRT31fold121_{last}{suffix})", # dark blue (other 3fold)
           f"12*(VRT31fold111_{last}{suffix}:VRT32fold111_{last}{suffix})", # light blue (3fold)
           f"6*(VRT31fold111_{last}{suffix}:VRT32fold121_{last}{suffix})", # brown (2-fold closest)
           f"6*(VRT31fold111_{last}{suffix}:VRT33fold131_{last}{suffix})")) # white (2-fold furthest)

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss3.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        #######################
        # Create first 3 fold #
        #######################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 3-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt31fold = ss3.copy_vrt(f"VRTglobal{suffix}", f"VRT31fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt31fold.vrt_y, 180)
        vrt31fold.rotate_right_multiply(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt31fold.vrt_z, 180)
        vrt31fold.rotate_right_multiply(R)
        # 3) then rotate to the 3-fold
        center3 = setup_applied_dofs.get_3fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center3, - vrt31fold.vrt_z), vector_angle(center3, - vrt31fold.vrt_z))
        vrt31fold.rotate_right_multiply(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt(f"VRTHFfold111_z{suffix}").vrt_orig - center3  # needs to be rotated onto the 3fold plane
        rot_angle = vector_angle(anchor_resi_vec, - vrt31fold.vrt_x)
        # find out which way to rotate
        if self._right_handed_vectors(anchor_resi_vec, vrt31fold.vrt_x, center3):
            R = rotation_matrix(center3, rot_angle)
        else:
            R = rotation_matrix(center3, - rot_angle)
        vrt31fold.rotate_right_multiply(R)
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRTglobal", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT31fold1{suffix}"))
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="31", base_vrt=vrt31fold, suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_tref{suffix}", move_origo=True, axis="z", dir=1))
        # ss3.add_jump(f"JUMP31fold_z_tref{suffix}", f"VRTglobal{suffix}", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRT31fold1_z_tref{suffix}", f"VRT31fold{suffix}")
        # ss3.add_jump(f"JUMP31fold1{suffix}", f"VRT31fold{suffix}", f"VRT31fold1{suffix}")

        # ---- chain 1 ----
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}", move_origo=True, axis="z"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z{suffix}"))
        # ss3.add_jump(f"JUMP31fold_z_rref{suffix}", f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}")
        # ss3.add_jump(f"JUMP31fold_z{suffix}", f"VRT31fold1_z_rref{suffix}", f"VRT31fold1_z{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="", ss_t_nb1="31", ss_t_nb2="1", suffix=suffix)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold111_x_tref{suffix}", move_origo=True, axis="x", dir=1))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold11{suffix}"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # ss3.add_jump(f"JUMP31fold111_x_tref{suffix}", f"VRT31fold1_z{suffix}", f"VRT31fold111_x_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold11{suffix}", f"VRT31fold111_x_tref{suffix}", f"VRT31fold11{suffix}")
        # ss3.add_jump(f"JUMP31fold111{suffix}", f"VRT31fold11{suffix}", f"VRT31fold111{suffix}")
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        ##############################
        # Create surrounding 3 folds #
        ##############################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 120)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT32fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="32", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT32fold1{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold{suffix}", f"VRTglobal{suffix}", f"VRT32fold{suffix}")
        # ss3.add_jump(f"JUMP32fold1{suffix}", f"VRT32fold{suffix}", f"VRT32fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="32", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT32fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="32", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold11{suffix}", f"VRT32fold1{suffix}", f"VRT32fold11{suffix}")
        # ss3.add_jump(f"JUMP32fold111{suffix}", f"VRT32fold11{suffix}", f"VRT32fold111{suffix}")
        # ss3.add_jump(f"JUMP32fold1111{suffix}", f"VRT32fold111{suffix}", f"VRT32fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="2", ss_t_nb1="32", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="121", ss_t_nb1="32", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold12{suffix}", f"VRT32fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold121{suffix}", f"VRT32fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1211{suffix}", f"VRT32fold1211{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold12{suffix}", f"VRT32fold1{suffix}", f"VRT32fold12{suffix}")
        # ss3.add_jump(f"JUMP32fold121{suffix}", f"VRT32fold12{suffix}", f"VRT32fold121{suffix}")
        # ss3.add_jump(f"JUMP32fold1211{suffix}", f"VRT32fold121{suffix}", f"VRT32fold1211{suffix}")
        # ss3.add_jump(f"JUMP32fold1111_subunit{suffix}", f"VRT32fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP32fold1211_subunit{suffix}", f"VRT32fold1211{suffix}", "SUBUNIT")

        # -- - 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], - 120)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="33", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt35fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT35fold{suffix}").rotate_right_multiply(R, True)
        # vrt35fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT35fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt35fold)
        # ss3.add_vrt(vrt35fold1)
        # ss3.add_jump(f"JUMP35fold{suffix}", f"VRTglobal{suffix}", f"VRT35fold{suffix}")
        # ss3.add_jump(f"JUMP35fold1{suffix}", f"VRT35fold{suffix}", f"VRT35fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="33", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="33", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT35fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT35fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT35fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold11{suffix}", f"VRT35fold1{suffix}", f"VRT35fold11{suffix}")
        # ss3.add_jump(f"JUMP35fold111{suffix}", f"VRT35fold11{suffix}", f"VRT35fold111{suffix}")
        # ss3.add_jump(f"JUMP35fold1111{suffix}", f"VRT35fold111{suffix}", f"VRT35fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="3", ss_t_nb1="33", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="131", ss_t_nb1="33", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold13{suffix}", f"VRT35fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold131{suffix}", f"VRT35fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1311{suffix}", f"VRT35fold1311{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold13{suffix}", f"VRT35fold1{suffix}", f"VRT35fold13{suffix}")
        # ss3.add_jump(f"JUMP35fold131{suffix}", f"VRT35fold13{suffix}", f"VRT35fold131{suffix}")
        # ss3.add_jump(f"JUMP35fold1311{suffix}", f"VRT35fold131{suffix}", f"VRT35fold1311{suffix}")
        # ss3.add_jump(f"JUMP35fold1111_subunit{suffix}", f"VRT35fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP35fold1311_subunit{suffix}", f"VRT35fold1311{suffix}", "SUBUNIT")

        # #######################################################
        # # Create last 2 3-folds that are part of the fourfold #
        # #######################################################
        # # -- 144 rotation --
        # # ---- create base ----
        # R = rotation_matrix([0, 0, 1], 180)
        # base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="34", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # # vrt33fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        # # vrt33fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT33fold1{suffix}").rotate_right_multiply(R, True)
        # # ss3.add_vrt(vrt33fold)
        # # ss3.add_vrt(vrt33fold1)
        # # ss3.add_jump(f"JUMP33fold{suffix}", f"VRTglobal{suffix}", f"VRT33fold{suffix}")
        # # ss3.add_jump(f"JUMP33fold1{suffix}", f"VRT33fold{suffix}", f"VRT33fold1{suffix}")
        # # ---- chain 1 ----
        # ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="34", ss_t_nb2="1", R=R, f="rotate_right_multiply",
        #                              suffix=suffix)
        # ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="34", ss_t_nb2="111", R=R, f="rotate_right_multiply",
        #                            suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT33fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT33fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT33fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP33fold11{suffix}", f"VRT33fold1{suffix}", f"VRT33fold11{suffix}")
        # ss3.add_jump(f"JUMP33fold111{suffix}", f"VRT33fold11{suffix}", f"VRT33fold111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111{suffix}", f"VRT33fold111{suffix}", f"VRT33fold1111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111_subunit{suffix}", f"VRT33fold1111{suffix}", "SUBUNIT")

        # # -- - 144 rotation --
        # # ---- create base ----
        # R = rotation_matrix([0, 0, 1], - 72 * 2)
        # base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="34", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # # vrt34fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # # vrt34fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT34fold1{suffix}").rotate_right_multiply(R, True)
        # # ss3.add_vrt(vrt34fold)
        # # ss3.add_vrt(vrt34fold1)
        # # ss3.add_jump(f"JUMP34fold{suffix}", f"VRTglobal{suffix}", f"VRT34fold{suffix}")
        # # ss3.add_jump(f"JUMP34fold1{suffix}", f"VRT34fold{suffix}", f"VRT34fold1{suffix}")
        # # ---- chain 1 ----
        # ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="34", ss_t_nb2="1", R=R, f="rotate_right_multiply",
        #                              suffix=suffix)
        # ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="34", ss_t_nb2="111", R=R, f="rotate_right_multiply",
        #                            suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT34fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT34fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT34fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP34fold11{suffix}", f"VRT34fold1{suffix}", f"VRT34fold11{suffix}")
        # ss3.add_jump(f"JUMP34fold111{suffix}", f"VRT34fold11{suffix}", f"VRT34fold111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111{suffix}", f"VRT34fold111{suffix}", f"VRT34fold1111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111_subunit{suffix}", f"VRT34fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss3.add_dof(f"JUMP31fold1{suffix}", 'z', "translation", np.linalg.norm(center3))
        ss3.add_dof(f"JUMP31fold1_z{suffix}", 'z', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center3))
        ss3.add_dof(f"JUMP31fold111_x{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_y{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_z{suffix}", 'z', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
        ss3.add_jumpgroup("JUMPGROUP1", f"JUMP31fold1{suffix}", f"JUMP32fold1{suffix}",  f"JUMP33fold1{suffix}")
        ss3.add_jumpgroup("JUMPGROUP2", f"JUMP31fold1_z{suffix}", f"JUMP32fold1_z{suffix}",
                          f"JUMP33fold1_z{suffix}")
        ss3.add_jumpgroup("JUMPGROUP3", f"JUMP31fold111{suffix}", f"JUMP31fold121{suffix}", f"JUMP31fold131{suffix}",
                          f"JUMP32fold111{suffix}", f"JUMP32fold121{suffix}",
                          f"JUMP33fold111{suffix}", f"JUMP33fold131{suffix}")
        ss3.add_jumpgroup("JUMPGROUP4", f"JUMP31fold111_x{suffix}", f"JUMP31fold121_x{suffix}", f"JUMP31fold131_x{suffix}",
                          f"JUMP32fold111_x{suffix}", f"JUMP32fold121_x{suffix}",
                          f"JUMP33fold111_x{suffix}", f"JUMP33fold131_x{suffix}")
        ss3.add_jumpgroup("JUMPGROUP5", f"JUMP31fold111_y{suffix}", f"JUMP31fold121_y{suffix}", f"JUMP31fold131_y{suffix}",
                          f"JUMP32fold111_y{suffix}", f"JUMP32fold121_y{suffix}",
                          f"JUMP33fold111_y{suffix}", f"JUMP33fold131_y{suffix}")
        ss3.add_jumpgroup("JUMPGROUP6", f"JUMP31fold111_z{suffix}", f"JUMP31fold121_z{suffix}", f"JUMP31fold131_z{suffix}",
                          f"JUMP32fold111_z{suffix}", f"JUMP32fold121_z{suffix}",
                          f"JUMP33fold111_z{suffix}", f"JUMP33fold131_z{suffix}")
        if make_sds:
            ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_sds{suffix}", f"JUMP31fold121_sds{suffix}",
                              f"JUMP31fold131_sds{suffix}", f"JUMP32fold111_sds{suffix}", f"JUMP32fold121_sds{suffix}",
                              f"JUMP33fold111_sds{suffix}", f"JUMP33fold131_sds{suffix}")
        ss3.add_jumpgroup("JUMPGROUP8", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP33fold111_subunit{suffix}", f"JUMP33fold131_subunit{suffix}")
        ss3._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss3)
        return ss3

    def create_T_2fold_based_symmetry(self, suffix='', make_sds=True, straighten_COM=True):
        """Creates a 2-fold based symmetry file from a Tetrahedral HF-based symmetry setup."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        ss2.headers = self.headers
        ss2.righthanded = self.righthanded

        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss2.energies = f"12*VRT21fold111_{last}{suffix} + " \
                       f"12*(VRT21fold111_{last}{suffix}:VRT23fold111_{last}{suffix}) + " \
                       f"12*(VRT21fold111_{last}{suffix}:VRT22fold121_{last}{suffix}) + " \
                       f"6*(VRT21fold111_{last}{suffix}:VRT21fold121_{last}{suffix}) + " \
                       f"6*(VRT21fold111_{last}{suffix}:VRT24fold111_{last}{suffix})"

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss2.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        ############################
        # Create first full 2 fold #
        ############################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 2-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt21fold = ss2.copy_vrt(f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt21fold.vrt_y, 180)
        vrt21fold.rotate(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt21fold.vrt_z, 180)
        vrt21fold.rotate(R)
        # 3) then rotate to the 2-fold
        center2 = setup_applied_dofs.get_2fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center2, -vrt21fold.vrt_z), -vector_angle(center2, -vrt21fold.vrt_z))
        vrt21fold.rotate(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2  # needs to be rotated onto the 2fold plane
        #         R = rotation_matrix(center3, -vector_angle(anchor_resi_vec, - vrt31fold.vrt_x))
        rot_angle = vector_angle(anchor_resi_vec, -vrt21fold.vrt_x)
        if self._right_handed_vectors(anchor_resi_vec, vrt21fold.vrt_x, center2):
            R = rotation_matrix(center2, -rot_angle)
        else:
            R = rotation_matrix(center2, rot_angle)
        is_righthanded = self.calculate_if_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(vrt21fold)
        # ss2.add_jump(f"JUMP21fold{suffix}", f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT21fold1{suffix}"))
        # ss2.add_jump(f"JUMP21fold1{suffix}", f"VRT21fold{suffix}", f"VRT21fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT21fold11{suffix}"))
        # # VRT21fold111 and VRT21fold1111 we want to make exactly like VRTHFfold1111 because we want to be able to
        # # transfer the rotations of that directly between a fivefold setup to a threefold setup
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold111", f"VRT21fold111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold1111", f"VRT21fold1111{suffix}"))
        # ###
        # ss2.add_jump(f"JUMP21fold11{suffix}", f"VRT21fold1{suffix}", f"VRT21fold11{suffix}")
        # ss2.add_jump(f"JUMP21fold111{suffix}", f"VRT21fold11{suffix}", f"VRT21fold111{suffix}")
        # ss2.add_jump(f"JUMP21fold1111{suffix}", f"VRT21fold111{suffix}", f"VRT21fold1111{suffix}")
        # ---- chain 2 ----

        R = rotation_matrix(center2, 180)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="21", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT21fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT21fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT21fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP21fold12{suffix}", f"VRT21fold1{suffix}", f"VRT21fold12{suffix}")
        # ss2.add_jump(f"JUMP21fold121{suffix}", f"VRT21fold12{suffix}", f"VRT21fold121{suffix}")
        # ss2.add_jump(f"JUMP21fold1211{suffix}", f"VRT21fold121{suffix}", f"VRT21fold1211{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP21fold1111_subunit{suffix}", f"VRT21fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP21fold1211_subunit{suffix}", f"VRT21fold1211{suffix}", "SUBUNIT")

        ########################
        # Create second 2 fold #
        ########################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 120 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="22", base_vrt=vrt_base, R=R)

        # vrt22fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        # vrt22fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT22fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt22fold)
        # ss2.add_vrt(vrt22fold1)
        # ss2.add_jump(f"JUMP22fold{suffix}", f"VRTglobal{suffix}", f"VRT22fold{suffix}")
        # ss2.add_jump(f"JUMP22fold1{suffix}", f"VRT22fold{suffix}", f"VRT22fold1{suffix}")
        # ---- chain 1 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="22", ss_t_nb2="1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold12{suffix}", f"VRT22fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold121{suffix}", f"VRT22fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1211{suffix}", f"VRT22fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold12{suffix}", f"VRT22fold1{suffix}", f"VRT22fold12{suffix}")
        # ss2.add_jump(f"JUMP22fold121{suffix}", f"VRT22fold12{suffix}", f"VRT22fold121{suffix}")
        # ss2.add_jump(f"JUMP22fold1211{suffix}", f"VRT22fold121{suffix}", f"VRT22fold1211{suffix}")
        # ss2.add_jump(f"JUMP22fold1111_subunit{suffix}", f"VRT22fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP22fold1211_subunit{suffix}", f"VRT22fold1211{suffix}", "SUBUNIT")

        ############################################
        # Create rest of the 2 folds in the 5-fold #
        ############################################
        for i in range(2, 3):
            R = rotation_matrix([0, 0, 1], 120 * i * (1 if is_righthanded else -1))
            # ---- create base ----
            n = str(i + 1)
            vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"2{n}", base_vrt=vrt_base, R=R)
            ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"", ss_t_nb1=f"2{n}", ss_t_nb2=f"1", R=R, suffix=suffix)
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)

            # vrt2nfold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            # vrt2nfold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT2{n}fold1{suffix}").rotate(R, True)
            # ss2.add_vrt(vrt2nfold)
            # ss2.add_vrt(vrt2nfold1)
            # ss2.add_jump(f"JUMP2{n}fold{suffix}", f"VRTglobal{suffix}", f"VRT2{n}fold{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1{suffix}", f"VRT2{n}fold{suffix}", f"VRT2{n}fold1{suffix}")
            # # ---- chain 1 ----
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT2{n}fold11{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT2{n}fold111{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT2{n}fold1111{suffix}").rotate(R, True))
            # ss2.add_jump(f"JUMP2{n}fold11{suffix}", f"VRT2{n}fold1{suffix}", f"VRT2{n}fold11{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold111{suffix}", f"VRT2{n}fold11{suffix}", f"VRT2{n}fold111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111{suffix}", f"VRT2{n}fold111{suffix}", f"VRT2{n}fold1111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111_subunit{suffix}", f"VRT2{n}fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(self.find_center_between_vtrs(setup_applied_dofs, "VRTHFfold111_z", "VRT3fold121_z"), 180)
        # ---- create base ----
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT24fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"24", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1=f"24", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"24", ss_t_nb2=f"111", R=R, suffix=suffix,  make_sds=make_sds)

        # vrt26fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT26fold{suffix}").rotate(R, True)
        # vrt26fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT26fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt26fold)
        # ss2.add_vrt(vrt26fold1)
        # ss2.add_jump(f"JUMP26fold{suffix}", f"VRTglobal{suffix}", f"VRT26fold{suffix}")
        # ss2.add_jump(f"JUMP26fold1{suffix}", f"VRT26fold{suffix}", f"VRT26fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT26fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT26fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT26fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP26fold11{suffix}", f"VRT26fold1{suffix}", f"VRT26fold11{suffix}")
        # ss2.add_jump(f"JUMP26fold111{suffix}", f"VRT26fold11{suffix}", f"VRT26fold111{suffix}")
        # ss2.add_jump(f"JUMP26fold1111{suffix}", f"VRT26fold111{suffix}", f"VRT26fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP26fold1111_subunit{suffix}", f"VRT26fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(-self.get_vrt("VRT3fold1")._vrt_z, 120 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT25fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="22", ss_t_nb1=f"25", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="22", ss_f_nb2="", ss_t_nb1=f"25", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"25", ss_t_nb2=f"111", R=R, suffix=suffix,  make_sds=make_sds)
        # # ---- create base ----
        # vrt27fold = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT27fold{suffix}").rotate(R, True)
        # vrt27fold1 = ss2.copy_vrt(f"VRT22fold1{suffix}", f"VRT27fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt27fold)
        # ss2.add_vrt(vrt27fold1)
        # ss2.add_jump(f"JUMP27fold{suffix}", f"VRTglobal{suffix}", f"VRT27fold{suffix}")
        # ss2.add_jump(f"JUMP27fold1{suffix}", f"VRT27fold{suffix}", f"VRT27fold1{suffix}")
        # # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold11{suffix}", f"VRT27fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold111{suffix}", f"VRT27fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold1111{suffix}", f"VRT27fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP27fold11{suffix}", f"VRT27fold1{suffix}", f"VRT27fold11{suffix}")
        # ss2.add_jump(f"JUMP27fold111{suffix}", f"VRT27fold11{suffix}", f"VRT27fold111{suffix}")
        # ss2.add_jump(f"JUMP27fold1111{suffix}", f"VRT27fold111{suffix}", f"VRT27fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP27fold1111_subunit{suffix}", f"VRT27fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss2.add_dof(f"JUMP21fold1{suffix}", 'z', "translation", np.linalg.norm(center2))
        ss2.add_dof(f"JUMP21fold1_z{suffix}", 'z', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2))
        ss2.add_dof(f"JUMP21fold111_x{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_y{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_z{suffix}", 'z', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
        ss2.add_jumpgroup("JUMPGROUP1", f"JUMP21fold1{suffix}", f"JUMP22fold1{suffix}", f"JUMP23fold1{suffix}",
                          f"JUMP24fold1{suffix}", f"JUMP25fold1{suffix}")
        ss2.add_jumpgroup("JUMPGROUP2", f"JUMP21fold1_z{suffix}", f"JUMP22fold1_z{suffix}", f"JUMP23fold1_z{suffix}",
                           f"JUMP24fold1_z{suffix}", f"JUMP25fold1_z{suffix}")
        ss2.add_jumpgroup("JUMPGROUP3", f"JUMP21fold111{suffix}", f"JUMP21fold121{suffix}", f"JUMP22fold111{suffix}",
                          f"JUMP22fold121{suffix}", f"JUMP23fold111{suffix}",
                          f"JUMP24fold111{suffix}", f"JUMP25fold111{suffix}")
        ss2.add_jumpgroup("JUMPGROUP4", f"JUMP21fold111_x{suffix}", f"JUMP21fold121_x{suffix}", f"JUMP22fold111_x{suffix}",
                          f"JUMP22fold121_x{suffix}", f"JUMP23fold111_x{suffix}",
                          f"JUMP24fold111_x{suffix}", f"JUMP25fold111_x{suffix}")
        ss2.add_jumpgroup("JUMPGROUP5", f"JUMP21fold111_y{suffix}", f"JUMP21fold121_y{suffix}", f"JUMP22fold111_y{suffix}",
                          f"JUMP22fold121_y{suffix}", f"JUMP23fold111_y{suffix}",
                          f"JUMP24fold111_y{suffix}", f"JUMP25fold111_y{suffix}")
        ss2.add_jumpgroup("JUMPGROUP6", f"JUMP21fold111_z{suffix}", f"JUMP21fold121_z{suffix}", f"JUMP22fold111_z{suffix}",
                          f"JUMP22fold121_z{suffix}", f"JUMP23fold111_z{suffix}",
                          f"JUMP24fold111_z{suffix}", f"JUMP25fold111_z{suffix}")
        if make_sds:
            ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_sds{suffix}", f"JUMP21fold121_sds{suffix}",
                              f"JUMP22fold111_sds{suffix}", f"JUMP22fold121_sds{suffix}", f"JUMP23fold111_sds{suffix}",
                              f"JUMP24fold111_sds{suffix}",
                              f"JUMP25fold111_sds{suffix}")
        ss2.add_jumpgroup("JUMPGROUP8", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}",
                          f"JUMP25fold111_subunit{suffix}")
        ss2._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss2)
        return ss2

    # ASSUMING IT IS 5FOLD - well it can only be that so because O doesnt have 3fold and T already is
    def create_I_3fold_based_symmetry(self, suffix='', make_sds=True, straighten_COM=True):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        ss3.headers = self.headers
        ss3.righthanded = self.righthanded

        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss3.energies = f"60*VRT31fold111_{last}{suffix} + " \
                       f"60*(VRT31fold111_{last}{suffix}:VRT32fold111_{last}{suffix}) + 60*(VRT31fold111_{last}{suffix}:VRT33fold111_{last}{suffix}) + " \
                       f"60*(VRT31fold111_{last}{suffix}:VRT31fold121_{last}{suffix}) + " \
                       f"30*(VRT31fold111_{last}{suffix}:VRT32fold121_{last}{suffix}) + 30*(VRT31fold111_{last}{suffix}:VRT35fold131_{last}{suffix})"

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss3.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        #######################
        # Create first 3 fold #
        #######################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 3-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt31fold = ss3.copy_vrt(f"VRTglobal{suffix}", f"VRT31fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt31fold.vrt_y, 180)
        vrt31fold.rotate_right_multiply(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt31fold.vrt_z, 180)
        vrt31fold.rotate_right_multiply(R)
        # 3) then rotate to the 3-fold
        center3 = setup_applied_dofs.get_3fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center3, - vrt31fold.vrt_z), vector_angle(center3, - vrt31fold.vrt_z))
        vrt31fold.rotate_right_multiply(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center3  # needs to be rotated onto the 3fold plane
        rot_angle = vector_angle(anchor_resi_vec, - vrt31fold.vrt_x)
        # find out which way to rotate
        if self._right_handed_vectors(anchor_resi_vec, vrt31fold.vrt_x, center3):
            R = rotation_matrix(center3, rot_angle)
        else:
            R = rotation_matrix(center3, - rot_angle)
        vrt31fold.rotate_right_multiply(R)
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRTglobal", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT31fold1{suffix}"))
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="31", base_vrt=vrt31fold)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_tref{suffix}", move_origo=True, axis="z", dir=1))
        # ss3.add_jump(f"JUMP31fold_z_tref{suffix}", f"VRTglobal{suffix}", f"VRT31fold1_z_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold{suffix}", f"VRT31fold1_z_tref{suffix}", f"VRT31fold{suffix}")
        # ss3.add_jump(f"JUMP31fold1{suffix}", f"VRT31fold{suffix}", f"VRT31fold1{suffix}")

        # ---- chain 1 ----
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}", move_origo=True, axis="z"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1_z{suffix}"))
        # ss3.add_jump(f"JUMP31fold_z_rref{suffix}", f"VRT31fold1{suffix}", f"VRT31fold1_z_rref{suffix}")
        # ss3.add_jump(f"JUMP31fold_z{suffix}", f"VRT31fold1_z_rref{suffix}", f"VRT31fold1_z{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="", ss_t_nb1="31", ss_t_nb2="1", suffix=suffix)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold111_x_tref{suffix}", move_origo=True, axis="x", dir=1))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold11{suffix}"))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # ss3.add_jump(f"JUMP31fold111_x_tref{suffix}", f"VRT31fold1_z{suffix}", f"VRT31fold111_x_tref{suffix}")
        # ss3.add_jump(f"JUMP31fold11{suffix}", f"VRT31fold111_x_tref{suffix}", f"VRT31fold11{suffix}")
        # ss3.add_jump(f"JUMP31fold111{suffix}", f"VRT31fold11{suffix}", f"VRT31fold111{suffix}")
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        ##############################
        # Create surrounding 3 folds #
        ##############################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 72)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT32fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="32", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT32fold1{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold{suffix}", f"VRTglobal{suffix}", f"VRT32fold{suffix}")
        # ss3.add_jump(f"JUMP32fold1{suffix}", f"VRT32fold{suffix}", f"VRT32fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="32", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT32fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="32", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT32fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold11{suffix}", f"VRT32fold1{suffix}", f"VRT32fold11{suffix}")
        # ss3.add_jump(f"JUMP32fold111{suffix}", f"VRT32fold11{suffix}", f"VRT32fold111{suffix}")
        # ss3.add_jump(f"JUMP32fold1111{suffix}", f"VRT32fold111{suffix}", f"VRT32fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="2", ss_t_nb1="32", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="121", ss_t_nb1="32", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)

        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold12{suffix}", f"VRT32fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold121{suffix}", f"VRT32fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1211{suffix}", f"VRT32fold1211{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP32fold12{suffix}", f"VRT32fold1{suffix}", f"VRT32fold12{suffix}")
        # ss3.add_jump(f"JUMP32fold121{suffix}", f"VRT32fold12{suffix}", f"VRT32fold121{suffix}")
        # ss3.add_jump(f"JUMP32fold1211{suffix}", f"VRT32fold121{suffix}", f"VRT32fold1211{suffix}")
        # ss3.add_jump(f"JUMP32fold1111_subunit{suffix}", f"VRT32fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP32fold1211_subunit{suffix}", f"VRT32fold1211{suffix}", "SUBUNIT")

        # -- - 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], - 72)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT35fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="35", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt35fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT35fold{suffix}").rotate_right_multiply(R, True)
        # vrt35fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT35fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt35fold)
        # ss3.add_vrt(vrt35fold1)
        # ss3.add_jump(f"JUMP35fold{suffix}", f"VRTglobal{suffix}", f"VRT35fold{suffix}")
        # ss3.add_jump(f"JUMP35fold1{suffix}", f"VRT35fold{suffix}", f"VRT35fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="35", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="35", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT35fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT35fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT35fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold11{suffix}", f"VRT35fold1{suffix}", f"VRT35fold11{suffix}")
        # ss3.add_jump(f"JUMP35fold111{suffix}", f"VRT35fold11{suffix}", f"VRT35fold111{suffix}")
        # ss3.add_jump(f"JUMP35fold1111{suffix}", f"VRT35fold111{suffix}", f"VRT35fold1111{suffix}")
        # ---- chain 2 ----
        # R = rotation_matrix(center3, 120)
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="3", ss_t_nb1="35", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="131", ss_t_nb1="35", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold13{suffix}", f"VRT35fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold131{suffix}", f"VRT35fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1311{suffix}", f"VRT35fold1311{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP35fold13{suffix}", f"VRT35fold1{suffix}", f"VRT35fold13{suffix}")
        # ss3.add_jump(f"JUMP35fold131{suffix}", f"VRT35fold13{suffix}", f"VRT35fold131{suffix}")
        # ss3.add_jump(f"JUMP35fold1311{suffix}", f"VRT35fold131{suffix}", f"VRT35fold1311{suffix}")
        # ss3.add_jump(f"JUMP35fold1111_subunit{suffix}", f"VRT35fold1111{suffix}", "SUBUNIT")
        # ss3.add_jump(f"JUMP35fold1311_subunit{suffix}", f"VRT35fold1311{suffix}", "SUBUNIT")

        #######################################################
        # Create last 2 3-folds that are part of the fivefold #
        #######################################################
        # -- 144 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 72 * 2)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="33", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt33fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT33fold{suffix}").rotate_right_multiply(R, True)
        # vrt33fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT33fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt33fold)
        # ss3.add_vrt(vrt33fold1)
        # ss3.add_jump(f"JUMP33fold{suffix}", f"VRTglobal{suffix}", f"VRT33fold{suffix}")
        # ss3.add_jump(f"JUMP33fold1{suffix}", f"VRT33fold{suffix}", f"VRT33fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="33", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="33", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT33fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT33fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT33fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP33fold11{suffix}", f"VRT33fold1{suffix}", f"VRT33fold11{suffix}")
        # ss3.add_jump(f"JUMP33fold111{suffix}", f"VRT33fold11{suffix}", f"VRT33fold111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111{suffix}", f"VRT33fold111{suffix}", f"VRT33fold1111{suffix}")
        # ss3.add_jump(f"JUMP33fold1111_subunit{suffix}", f"VRT33fold1111{suffix}", "SUBUNIT")

        # -- - 144 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], - 72 * 2)
        base_vrt = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        ss3._create_base_dofs(ss3, ss3, ss_f_nb1="31", ss_t_nb1="34", base_vrt=base_vrt, R=R, f="rotate_right_multiply", suffix=suffix)
        # vrt34fold = ss3.copy_vrt(f"VRT31fold{suffix}", f"VRT34fold{suffix}").rotate_right_multiply(R, True)
        # vrt34fold1 = ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT34fold1{suffix}").rotate_right_multiply(R, True)
        # ss3.add_vrt(vrt34fold)
        # ss3.add_vrt(vrt34fold1)
        # ss3.add_jump(f"JUMP34fold{suffix}", f"VRTglobal{suffix}", f"VRT34fold{suffix}")
        # ss3.add_jump(f"JUMP34fold1{suffix}", f"VRT34fold{suffix}", f"VRT34fold1{suffix}")
        # ---- chain 1 ----
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="34", ss_t_nb2="1", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="34", ss_t_nb2="111", R=R, f="rotate_right_multiply",
                                   suffix=suffix, make_sds=make_sds)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT34fold11{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT34fold111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1111{suffix}", f"VRT34fold1111{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP34fold11{suffix}", f"VRT34fold1{suffix}", f"VRT34fold11{suffix}")
        # ss3.add_jump(f"JUMP34fold111{suffix}", f"VRT34fold11{suffix}", f"VRT34fold111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111{suffix}", f"VRT34fold111{suffix}", f"VRT34fold1111{suffix}")
        # ss3.add_jump(f"JUMP34fold1111_subunit{suffix}", f"VRT34fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss3.add_dof(f"JUMP31fold1{suffix}", 'z', "translation", np.linalg.norm(center3))
        ss3.add_dof(f"JUMP31fold1_z{suffix}", 'z', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center3))
        ss3.add_dof(f"JUMP31fold111_x{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_y{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_z{suffix}", 'z', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
        ss3.add_jumpgroup("JUMPGROUP1", f"JUMP31fold1{suffix}", f"JUMP32fold1{suffix}", f"JUMP35fold1{suffix}", f"JUMP33fold1{suffix}",
                          f"JUMP34fold1{suffix}")
        ss3.add_jumpgroup("JUMPGROUP2", f"JUMP31fold1_z{suffix}", f"JUMP32fold1_z{suffix}", f"JUMP35fold1_z{suffix}",
                          f"JUMP33fold1_z{suffix}", f"JUMP34fold1_z{suffix}")
        ss3.add_jumpgroup("JUMPGROUP3", f"JUMP31fold111{suffix}", f"JUMP31fold121{suffix}", f"JUMP31fold131{suffix}",
                          f"JUMP32fold111{suffix}", f"JUMP32fold121{suffix}", f"JUMP35fold111{suffix}", f"JUMP35fold131{suffix}",
                          f"JUMP33fold111{suffix}", f"JUMP34fold111{suffix}")
        ss3.add_jumpgroup("JUMPGROUP4", f"JUMP31fold111_x{suffix}", f"JUMP31fold121_x{suffix}", f"JUMP31fold131_x{suffix}",
                          f"JUMP32fold111_x{suffix}", f"JUMP32fold121_x{suffix}", f"JUMP35fold111_x{suffix}", f"JUMP35fold131_x{suffix}",
                          f"JUMP33fold111_x{suffix}", f"JUMP34fold111_x{suffix}")
        ss3.add_jumpgroup("JUMPGROUP5", f"JUMP31fold111_y{suffix}", f"JUMP31fold121_y{suffix}", f"JUMP31fold131_y{suffix}",
                          f"JUMP32fold111_y{suffix}", f"JUMP32fold121_y{suffix}", f"JUMP35fold111_y{suffix}", f"JUMP35fold131_y{suffix}",
                          f"JUMP33fold111_y{suffix}", f"JUMP34fold111_y{suffix}")
        ss3.add_jumpgroup("JUMPGROUP6", f"JUMP31fold111_z{suffix}", f"JUMP31fold121_z{suffix}", f"JUMP31fold131_z{suffix}",
                          f"JUMP32fold111_z{suffix}", f"JUMP32fold121_z{suffix}", f"JUMP35fold111_z{suffix}", f"JUMP35fold131_z{suffix}",
                          f"JUMP33fold111_z{suffix}", f"JUMP34fold111_z{suffix}")
        if make_sds:
            ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_sds{suffix}", f"JUMP31fold121_sds{suffix}",
                              f"JUMP31fold131_sds{suffix}", f"JUMP32fold111_sds{suffix}", f"JUMP32fold121_sds{suffix}",
                              f"JUMP35fold111_sds{suffix}", f"JUMP35fold131_sds{suffix}", f"JUMP33fold111_sds{suffix}",
                              f"JUMP34fold111_sds{suffix}")
        ss3.add_jumpgroup("JUMPGROUP8", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP35fold111_subunit{suffix}", f"JUMP35fold131_subunit{suffix}", f"JUMP33fold111_subunit{suffix}",
                          f"JUMP34fold111_subunit{suffix}")
        ss3._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss3)
        return ss3

    # fixme: delete if not used
    def m(self, string, right=False):
        map_ = {"VRT3fold1211": "VRT2fold1111",
                "VRT3fold1111": "VRT2fold1211",
                "VRT2fold1211": "VRT3fold1111",
                "VRT2fold1111": "VRT3fold1211"}
        if string in map_.keys():
            if right:
                return map_[string]
            else:
                return string

    @staticmethod
    def _right_handed_vectors(v1, v2, axis):
        """Returns true if the point v1 going to v2 relative to the axis is right-handed. It is left-handed if the cross product v1 X v2
        points in the same direction as the axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        cross = np.cross(np.array(v1 - axis), np.array(v2 - axis))
        return vector_angle(cross, axis) > 90  # True -> It is right-handed



    @staticmethod
    def _get_z_from_downstream_jump(pose, jump_name):
        jump_id = sym_dof_jump_num(pose, jump_name)
        vrt = pose.residue(pose.fold_tree().downstream_jump_residue(jump_id))
        # 1 seems to be origo
        # 2 is x
        # 3 is y
        o = np.array(vrt.atom(1).xyz())
        x = np.array(vrt.atom(2).xyz()) - o
        y = np.array(vrt.atom(3).xyz()) - o
        z = np.cross(x, y)
        return z

    @staticmethod
    def calculate_if_righthanded_from_pose(pose):
        base = CubicSetup.get_base_from_pose(pose)
        ji_1, ji_2, ji_3 = CubicSetup.get_fold_jumpidentifier_from_pose(pose)
        jn = "JUMP{}fold1"
        # foldHF_axis = -self.get_vrt("VRTHFfold1").vrt_z
        # fold3_axis = -self.get_vrt("VRT3fold1").vrt_z
        # fold2_axis = -self.get_vrt("VRT2fold1").vrt_z
        if base == "HF":
            ji_1_z = - CubicSetup._get_z_from_downstream_jump(pose, jn.format(ji_1))
            ji_2_z = - CubicSetup._get_z_from_downstream_jump(pose, jn.format(ji_2))
            ji_3_z = - CubicSetup._get_z_from_downstream_jump(pose, jn.format(ji_3))
        elif base == "3F":
            raise NotImplementedError
        elif base == "2F":
            raise NotImplementedError
        return CubicSetup._right_handed_vectors(ji_2_z, ji_3_z, ji_1_z) # 3, 2, HF

    def create_I_2fold_based_symmetry(self, suffix='', make_sds=True, straighten_COM=True):
        """Creates a 2-fold based symmetry file from a HF-based (CURRENTLY ONLY 5-fold) one."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        ss2.headers = self.headers
        ss2.righthanded = self.righthanded
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        if make_sds:
            last = "sds"
        else:
            last = "z"
        ss2.energies = f"60*VRT21fold111_{last}{suffix} + " \
                       f"60*(VRT21fold111_{last}{suffix}:VRT25fold111_{last}{suffix}) + 60*(VRT21fold111_{last}{suffix}:VRT24fold111_{last}{suffix}) + " \
                       f"60*(VRT21fold111_{suffix}:VRT22fold121_{last}{suffix}) + " \
                       f"30*(VRT21fold111_{last}{suffix}:VRT21fold121_{last}{suffix}) + 30*(VRT21fold111_{last}{suffix}:VRT26fold111_{last}{suffix})"

        setup_applied_dofs = copy.deepcopy(self)
        setup_applied_dofs.apply_dofs()

        # -- create global center --
        ss2.add_vrt(self.copy_vrt("VRTglobal", f"VRTglobal{suffix}"))

        ############################
        # Create first full 2 fold #
        ############################

        # ---- create base ----
        # 1+2: because Rosetta features with reverse dof x and z movement we have to:
        # - 1: to accomodate opposite z movement: make 180 degree turn around y axis
        # - 2: to accomodate opposite x movement: make 180 degree turn around z axis
        # 3: we then want it to point towards the 2-fold
        # 4: finally we want to point the x-axis towards the anchor residue (actually - x-axis).
        vrt21fold = ss2.copy_vrt(f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # 1) first rotate around y
        R = rotation_matrix(vrt21fold.vrt_y, 180)
        vrt21fold.rotate(R)
        # 2) then rotate around z
        R = rotation_matrix(vrt21fold.vrt_z, 180)
        vrt21fold.rotate(R)
        # 3) then rotate to the 2-fold
        center2 = setup_applied_dofs.get_2fold_center_from_HFfold()
        R = rotation_matrix(np.cross(center2, -vrt21fold.vrt_z), -vector_angle(center2, -vrt21fold.vrt_z))
        vrt21fold.rotate(R)
        # 4) then rotate towards the anchor residue
        anchor_resi_vec = setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2  # needs to be rotated onto the 2fold plane
        #         R = rotation_matrix(center3, -vector_angle(anchor_resi_vec, - vrt31fold.vrt_x))
        rot_angle = vector_angle(anchor_resi_vec, -vrt21fold.vrt_x)
        if self._right_handed_vectors(anchor_resi_vec, vrt21fold.vrt_x, center2):
            R = rotation_matrix(center2, -rot_angle)
        else:
            R = rotation_matrix(center2, rot_angle)
        is_righthanded = self.calculate_if_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(vrt21fold)
        # ss2.add_jump(f"JUMP21fold{suffix}", f"VRTglobal{suffix}", f"VRT21fold{suffix}")
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT21fold1{suffix}"))
        # ss2.add_jump(f"JUMP21fold1{suffix}", f"VRT21fold{suffix}", f"VRT21fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT21fold11{suffix}"))
        # # VRT21fold111 and VRT21fold1111 we want to make exactly like VRTHFfold1111 because we want to be able to
        # # transfer the rotations of that directly between a fivefold setup to a threefold setup
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold111{suffix}"))
        # # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold1{suffix}", f"VRT31fold1111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold111", f"VRT21fold111{suffix}"))
        # ss2.add_vrt(self.copy_vrt(f"VRTHFfold1111", f"VRT21fold1111{suffix}"))
        # ###
        # ss2.add_jump(f"JUMP21fold11{suffix}", f"VRT21fold1{suffix}", f"VRT21fold11{suffix}")
        # ss2.add_jump(f"JUMP21fold111{suffix}", f"VRT21fold11{suffix}", f"VRT21fold111{suffix}")
        # ss2.add_jump(f"JUMP21fold1111{suffix}", f"VRT21fold111{suffix}", f"VRT21fold1111{suffix}")
        # ---- chain 2 ----

        R = rotation_matrix(center2, 180)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="21", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT21fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT21fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT21fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP21fold12{suffix}", f"VRT21fold1{suffix}", f"VRT21fold12{suffix}")
        # ss2.add_jump(f"JUMP21fold121{suffix}", f"VRT21fold12{suffix}", f"VRT21fold121{suffix}")
        # ss2.add_jump(f"JUMP21fold1211{suffix}", f"VRT21fold121{suffix}", f"VRT21fold1211{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP21fold1111_subunit{suffix}", f"VRT21fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP21fold1211_subunit{suffix}", f"VRT21fold1211{suffix}", "SUBUNIT")

        ########################
        # Create second 2 fold #
        ########################

        # -- 72 rotation --
        # ---- create base ----
        R = rotation_matrix([0, 0, 1], 72 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="22", base_vrt=vrt_base, R=R)

        # vrt22fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT22fold{suffix}").rotate(R, True)
        # vrt22fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT22fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt22fold)
        # ss2.add_vrt(vrt22fold1)
        # ss2.add_jump(f"JUMP22fold{suffix}", f"VRTglobal{suffix}", f"VRT22fold{suffix}")
        # ss2.add_jump(f"JUMP22fold1{suffix}", f"VRT22fold{suffix}", f"VRT22fold1{suffix}")
        # ---- chain 1 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="1", ss_t_nb1="22", ss_t_nb2="1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix, make_sds=make_sds)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold12{suffix}", f"VRT22fold12{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold121{suffix}", f"VRT22fold121{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1211{suffix}", f"VRT22fold1211{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold12{suffix}", f"VRT22fold1{suffix}", f"VRT22fold12{suffix}")
        # ss2.add_jump(f"JUMP22fold121{suffix}", f"VRT22fold12{suffix}", f"VRT22fold121{suffix}")
        # ss2.add_jump(f"JUMP22fold1211{suffix}", f"VRT22fold121{suffix}", f"VRT22fold1211{suffix}")
        # ss2.add_jump(f"JUMP22fold1111_subunit{suffix}", f"VRT22fold1111{suffix}", "SUBUNIT")
        # ss2.add_jump(f"JUMP22fold1211_subunit{suffix}", f"VRT22fold1211{suffix}", "SUBUNIT")

        ############################################
        # Create rest of the 2 folds in the 5-fold #
        ############################################
        for i in range(2, 5):
            R = rotation_matrix([0, 0, 1], 72 * i * (1 if is_righthanded else -1))
            # ---- create base ----
            n = str(i + 1)
            vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"2{n}", base_vrt=vrt_base, R=R)
            ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"", ss_t_nb1=f"2{n}", ss_t_nb2=f"1", R=R, suffix=suffix)
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)

            # vrt2nfold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT2{n}fold{suffix}").rotate(R, True)
            # vrt2nfold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT2{n}fold1{suffix}").rotate(R, True)
            # ss2.add_vrt(vrt2nfold)
            # ss2.add_vrt(vrt2nfold1)
            # ss2.add_jump(f"JUMP2{n}fold{suffix}", f"VRTglobal{suffix}", f"VRT2{n}fold{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1{suffix}", f"VRT2{n}fold{suffix}", f"VRT2{n}fold1{suffix}")
            # # ---- chain 1 ----
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT2{n}fold11{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT2{n}fold111{suffix}").rotate(R, True))
            # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT2{n}fold1111{suffix}").rotate(R, True))
            # ss2.add_jump(f"JUMP2{n}fold11{suffix}", f"VRT2{n}fold1{suffix}", f"VRT2{n}fold11{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold111{suffix}", f"VRT2{n}fold11{suffix}", f"VRT2{n}fold111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111{suffix}", f"VRT2{n}fold111{suffix}", f"VRT2{n}fold1111{suffix}")
            # ss2.add_jump(f"JUMP2{n}fold1111_subunit{suffix}", f"VRT2{n}fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(self.find_center_between_vtrs(setup_applied_dofs, "VRTHFfold111_z", "VRT3fold121_z"), 180)
        # ---- create base ----
        vrt_base = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT26fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1=f"26", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1=f"26", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"26", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)

        # vrt26fold = ss2.copy_vrt(f"VRT21fold{suffix}", f"VRT26fold{suffix}").rotate(R, True)
        # vrt26fold1 = ss2.copy_vrt(f"VRT21fold1{suffix}", f"VRT26fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt26fold)
        # ss2.add_vrt(vrt26fold1)
        # ss2.add_jump(f"JUMP26fold{suffix}", f"VRTglobal{suffix}", f"VRT26fold{suffix}")
        # ss2.add_jump(f"JUMP26fold1{suffix}", f"VRT26fold{suffix}", f"VRT26fold1{suffix}")
        # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT26fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold111{suffix}", f"VRT26fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT26fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP26fold11{suffix}", f"VRT26fold1{suffix}", f"VRT26fold11{suffix}")
        # ss2.add_jump(f"JUMP26fold111{suffix}", f"VRT26fold11{suffix}", f"VRT26fold111{suffix}")
        # ss2.add_jump(f"JUMP26fold1111{suffix}", f"VRT26fold111{suffix}", f"VRT26fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP26fold1111_subunit{suffix}", f"VRT26fold1111{suffix}", "SUBUNIT")

        ##########################################
        # Create the 2 folds in the other 5-fold #
        ##########################################
        R = rotation_matrix(-self.get_vrt("VRT3fold1")._vrt_z, 72 * (1 if is_righthanded else -1))
        vrt_base = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT27fold{suffix}").rotate(R, True)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="22", ss_t_nb1=f"27", base_vrt=vrt_base, R=R)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="22", ss_f_nb2="", ss_t_nb1=f"27", ss_t_nb2=f"1", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"27", ss_t_nb2=f"111", R=R, suffix=suffix, make_sds=make_sds)
        # # ---- create base ----
        # vrt27fold = ss2.copy_vrt(f"VRT22fold{suffix}", f"VRT27fold{suffix}").rotate(R, True)
        # vrt27fold1 = ss2.copy_vrt(f"VRT22fold1{suffix}", f"VRT27fold1{suffix}").rotate(R, True)
        # ss2.add_vrt(vrt27fold)
        # ss2.add_vrt(vrt27fold1)
        # ss2.add_jump(f"JUMP27fold{suffix}", f"VRTglobal{suffix}", f"VRT27fold{suffix}")
        # ss2.add_jump(f"JUMP27fold1{suffix}", f"VRT27fold{suffix}", f"VRT27fold1{suffix}")
        # # ---- chain 1 ----
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold11{suffix}", f"VRT27fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold111{suffix}", f"VRT27fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT22fold1111{suffix}", f"VRT27fold1111{suffix}").rotate(R, True))
        #
        # ss2.add_jump(f"JUMP27fold11{suffix}", f"VRT27fold1{suffix}", f"VRT27fold11{suffix}")
        # ss2.add_jump(f"JUMP27fold111{suffix}", f"VRT27fold11{suffix}", f"VRT27fold111{suffix}")
        # ss2.add_jump(f"JUMP27fold1111{suffix}", f"VRT27fold111{suffix}", f"VRT27fold1111{suffix}")
        # # ---- create the last jumps to the subunits ----
        # ss2.add_jump(f"JUMP27fold1111_subunit{suffix}", f"VRT27fold1111{suffix}", "SUBUNIT")

        ##################################
        # Create the dofs and jumpgroups #
        ##################################
        ss2.add_dof(f"JUMP21fold1{suffix}", 'z', "translation", np.linalg.norm(center2))
        ss2.add_dof(f"JUMP21fold1_z{suffix}", 'z', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111{suffix}", 'x', "translation",
                    np.linalg.norm(setup_applied_dofs.get_vrt("VRTHFfold111_z").vrt_orig - center2))
        ss2.add_dof(f"JUMP21fold111_x{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_y{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_z{suffix}", 'z', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        # ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
        ss2.add_jumpgroup("JUMPGROUP1", f"JUMP21fold1{suffix}", f"JUMP22fold1{suffix}", f"JUMP23fold1{suffix}", f"JUMP24fold1{suffix}",
                          f"JUMP25fold1{suffix}", f"JUMP26fold1{suffix}", f"JUMP27fold1{suffix}")
        ss2.add_jumpgroup("JUMPGROUP2", f"JUMP21fold1_z{suffix}", f"JUMP22fold1_z{suffix}", f"JUMP23fold1_z{suffix}",
                          f"JUMP24fold1_z{suffix}", f"JUMP25fold1_z{suffix}", f"JUMP26fold1_z{suffix}", f"JUMP27fold1_z{suffix}")
        ss2.add_jumpgroup("JUMPGROUP3", f"JUMP21fold111{suffix}", f"JUMP21fold121{suffix}", f"JUMP22fold111{suffix}",
                          f"JUMP22fold121{suffix}", f"JUMP23fold111{suffix}", f"JUMP24fold111{suffix}", f"JUMP25fold111{suffix}",
                          f"JUMP26fold111{suffix}", f"JUMP27fold111{suffix}")
        ss2.add_jumpgroup("JUMPGROUP4", f"JUMP21fold111_x{suffix}", f"JUMP21fold121_x{suffix}", f"JUMP22fold111_x{suffix}",
                          f"JUMP22fold121_x{suffix}", f"JUMP23fold111_x{suffix}", f"JUMP24fold111_x{suffix}",
                          f"JUMP25fold111_x{suffix}", f"JUMP26fold111_x{suffix}", f"JUMP27fold111_x{suffix}")
        ss2.add_jumpgroup("JUMPGROUP5", f"JUMP21fold111_y{suffix}", f"JUMP21fold121_y{suffix}", f"JUMP22fold111_y{suffix}",
                          f"JUMP22fold121_y{suffix}", f"JUMP23fold111_y{suffix}", f"JUMP24fold111_y{suffix}",
                          f"JUMP25fold111_y{suffix}", f"JUMP26fold111_y{suffix}", f"JUMP27fold111_y{suffix}")
        ss2.add_jumpgroup("JUMPGROUP6", f"JUMP21fold111_z{suffix}", f"JUMP21fold121_z{suffix}", f"JUMP22fold111_z{suffix}",
                          f"JUMP22fold121_z{suffix}", f"JUMP23fold111_z{suffix}", f"JUMP24fold111_z{suffix}",
                          f"JUMP25fold111_z{suffix}", f"JUMP26fold111_z{suffix}", f"JUMP27fold111_z{suffix}")
        if make_sds:
            ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_sds{suffix}", f"JUMP21fold121_sds{suffix}",
                              f"JUMP22fold111_sds{suffix}", f"JUMP22fold121_sds{suffix}", f"JUMP23fold111_sds{suffix}",
                              f"JUMP24fold111_sds{suffix}", f"JUMP25fold111_sds{suffix}", f"JUMP26fold111_sds{suffix}",
                              f"JUMP27fold111_sds{suffix}")
        ss2.add_jumpgroup("JUMPGROUP8", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}", f"JUMP25fold111_subunit{suffix}", f"JUMP26fold111_subunit{suffix}",
                          f"JUMP27fold111_subunit{suffix}")
        ss2._set_init_vrts()
        if straighten_COM:
            self.straightinator(ss2)
        return ss2

    def add_extra_chains(self):
        ss = copy.deepcopy(self)
        ss.symmetry_name = self.symmetry_name #+ "_full_chains"
        ss._set_vrts_to_init_vrts()
        if ss.is_hf_based():
            return self._add_hf_extra_chains(ss)
        elif ss.is_3f_based():
            return self._add_3f_extra_chains(ss)
        elif ss.is_2f_based():
            return self._add_2f_extra_chains(ss)

    def _add_2f_extra_chains(self, ss):

        mul = -1 if self.righthanded else 1

        # First we make completly new 2-folds
        ss_t = copy.deepcopy(self)
        ss_t.apply_dofs()
        orig = ss_t.get_vrt(f"VRT27fold1_z").vrt_orig

        # modify the energy term!
        new_3F = " + ".join(f"60*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in
                            zip(["25", "X", "23", "Z", "T", "Y"], ["121", "111", "121", "111", "111", "111"]))
        new_2F = " + ".join(f"30*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in zip(["P"], ["111"]))
        ss.energies += f" + {new_3F} + {new_2F}"

        for r1, r2, r3, id_ in zip((0, 0, 0), (0, 180, 0), (mul*72, mul*72, -1 * mul * 72*2), ("X", "Y", "Z")):
            R = rotation_matrix(orig, r1)
            R2 = rotation_matrix(orig, r2)
            R3 = rotation_matrix([0, 0, 1], r3)

            f = "27"
            # add vrts no
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_tref", f"VRT{id_}fold1_z_tref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold", f"VRT{id_}fold").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1", f"VRT{id_}fold1").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_rref", f"VRT{id_}fold1_z_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z", f"VRT{id_}fold1_z").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))

            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold11", f"VRT{id_}fold11").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_tref", f"VRT{id_}fold111_x_tref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111", f"VRT{id_}fold111").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_rref", f"VRT{id_}fold111_x_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x", f"VRT{id_}fold111_x").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y_rref", f"VRT{id_}fold111_y_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y", f"VRT{id_}fold111_y").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z_rref", f"VRT{id_}fold111_z_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z", f"VRT{id_}fold111_z").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_sds", f"VRT{id_}fold111_sds").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True))

            # add vrts no
            ss.add_jump(f"JUMP{id_}fold1_z_tref", f"VRTglobal", f"VRT{id_}fold1_z_tref")
            ss.add_jump(f"JUMP{id_}fold", f"VRT{id_}fold1_z_tref", f"VRT{id_}fold")
            ss.add_jump(f"JUMP{id_}fold1", f"VRT{id_}fold", f"VRT{id_}fold1")
            ss.add_jump(f"JUMP{id_}fold1_z_rref", f"VRT{id_}fold1", f"VRT{id_}fold1_z_rref")
            ss.add_jump(f"JUMP{id_}fold1_z", f"VRT{id_}fold1_z_rref", f"VRT{id_}fold1_z")
            ss.add_jump(f"JUMP{id_}fold111_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold111_x_tref")
            ss.add_jump(f"JUMP{id_}fold11", f"VRT{id_}fold111_x_tref", f"VRT{id_}fold11")
            ss.add_jump(f"JUMP{id_}fold111", f"VRT{id_}fold11", f"VRT{id_}fold111")
            ss.add_jump(f"JUMP{id_}fold111_x_rref", f"VRT{id_}fold111", f"VRT{id_}fold111_x_rref")
            ss.add_jump(f"JUMP{id_}fold111_x", f"VRT{id_}fold111_x_rref", f"VRT{id_}fold111_x")
            ss.add_jump(f"JUMP{id_}fold111_y_rref", f"VRT{id_}fold111_x", f"VRT{id_}fold111_y_rref")
            ss.add_jump(f"JUMP{id_}fold111_y", f"VRT{id_}fold111_y_rref", f"VRT{id_}fold111_y")
            ss.add_jump(f"JUMP{id_}fold111_z_rref", f"VRT{id_}fold111_y", f"VRT{id_}fold111_z_rref")
            ss.add_jump(f"JUMP{id_}fold111_z", f"VRT{id_}fold111_z_rref", f"VRT{id_}fold111_z")
            ss.add_jump(f"JUMP{id_}fold111_sds", f"VRT{id_}fold111_z", f"VRT{id_}fold111_sds")
            ss.add_jump(f"JUMP{id_}fold111_subunit", f"VRT{id_}fold111_sds", f"SUBUNIT")

            ss.jumpgroups["JUMPGROUP1"].append(f"JUMP{id_}fold1")
            ss.jumpgroups["JUMPGROUP2"].append(f"JUMP{id_}fold1_z")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold111")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold111_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold111_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold111_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold111_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold111_subunit")

        for id_ in ("25", "23"):

            ss_t = copy.deepcopy(self)
            ss_t.apply_dofs()
            orig = ss_t.get_vrt(f"VRT{id_}fold1_z").vrt_orig
            R2 = rotation_matrix(orig, 180)

            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_x_tref", f"VRT{id_}fold121_x_tref").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold11", f"VRT{id_}fold12").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111", f"VRT{id_}fold121").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_x_rref", f"VRT{id_}fold121_x_rref").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_x", f"VRT{id_}fold121_x").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_y_rref", f"VRT{id_}fold121_y_rref").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_y", f"VRT{id_}fold121_y").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_z_rref", f"VRT{id_}fold121_z_rref").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_z", f"VRT{id_}fold121_z").__getattribute__("rotate")(R2, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold111_sds", f"VRT{id_}fold121_sds").__getattribute__("rotate")(R2, True))

            ss.add_jump(f"JUMP{id_}fold121_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold121_x_tref")
            ss.add_jump(f"JUMP{id_}fold12", f"VRT{id_}fold121_x_tref", f"VRT{id_}fold12")
            ss.add_jump(f"JUMP{id_}fold121", f"VRT{id_}fold12", f"VRT{id_}fold121")
            ss.add_jump(f"JUMP{id_}fold121_x_rref", f"VRT{id_}fold121", f"VRT{id_}fold121_x_rref")
            ss.add_jump(f"JUMP{id_}fold121_x", f"VRT{id_}fold121_x_rref", f"VRT{id_}fold121_x")
            ss.add_jump(f"JUMP{id_}fold121_y_rref", f"VRT{id_}fold121_x", f"VRT{id_}fold121_y_rref")
            ss.add_jump(f"JUMP{id_}fold121_y", f"VRT{id_}fold121_y_rref", f"VRT{id_}fold121_y")
            ss.add_jump(f"JUMP{id_}fold121_z_rref", f"VRT{id_}fold121_y", f"VRT{id_}fold121_z_rref")
            ss.add_jump(f"JUMP{id_}fold121_z", f"VRT{id_}fold121_z_rref", f"VRT{id_}fold121_z")
            ss.add_jump(f"JUMP{id_}fold121_sds", f"VRT{id_}fold121_z", f"VRT{id_}fold121_sds")
            ss.add_jump(f"JUMP{id_}fold121_subunit", f"VRT{id_}fold121_sds", f"SUBUNIT")

            # jumpgroups
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold121")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold121_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold121_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold121_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold121_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold121_subunit")


        for f, final_angle, id_ in zip(("22", "25"), (mul*72*2, -1 * mul *72), ("T", "P")):

            ss_t = copy.deepcopy(self)
            ss_t.apply_dofs()
            orig = ss_t.get_vrt(f"VRT{f}fold1_z").vrt_orig

            # The angles to get
            angle = vector_angle([0, 0, 1], orig)
            rotvec = np.cross([0, 0, 1], orig)
            other_fivefold = rotate(orig, rotation_matrix(rotvec, angle ))
            R = rotation_matrix(other_fivefold, final_angle)

            # add vrts n
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_tref", f"VRT{id_}fold1_z_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold", f"VRT{id_}fold").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1", f"VRT{id_}fold1").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_rref", f"VRT{id_}fold1_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z", f"VRT{id_}fold1_z").__getattribute__("rotate")(R, True))

            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold12", f"VRT{id_}fold11").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_tref", f"VRT{id_}fold111_x_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121", f"VRT{id_}fold111").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_rref", f"VRT{id_}fold111_x_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x", f"VRT{id_}fold111_x").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y_rref", f"VRT{id_}fold111_y_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y", f"VRT{id_}fold111_y").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z_rref", f"VRT{id_}fold111_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z", f"VRT{id_}fold111_z").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_sds", f"VRT{id_}fold111_sds").__getattribute__("rotate")(R, True))

            # add vrts no
            ss.add_jump(f"JUMP{id_}fold1_z_tref", f"VRTglobal", f"VRT{id_}fold1_z_tref")
            ss.add_jump(f"JUMP{id_}fold", f"VRT{id_}fold1_z_tref", f"VRT{id_}fold")
            ss.add_jump(f"JUMP{id_}fold1", f"VRT{id_}fold", f"VRT{id_}fold1")
            ss.add_jump(f"JUMP{id_}fold1_z_rref", f"VRT{id_}fold1", f"VRT{id_}fold1_z_rref")
            ss.add_jump(f"JUMP{id_}fold1_z", f"VRT{id_}fold1_z_rref", f"VRT{id_}fold1_z")
            ss.add_jump(f"JUMP{id_}fold111_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold111_x_tref")
            ss.add_jump(f"JUMP{id_}fold11", f"VRT{id_}fold111_x_tref", f"VRT{id_}fold11")
            ss.add_jump(f"JUMP{id_}fold111", f"VRT{id_}fold11", f"VRT{id_}fold111")
            ss.add_jump(f"JUMP{id_}fold111_x_rref", f"VRT{id_}fold111", f"VRT{id_}fold111_x_rref")
            ss.add_jump(f"JUMP{id_}fold111_x", f"VRT{id_}fold111_x_rref", f"VRT{id_}fold111_x")
            ss.add_jump(f"JUMP{id_}fold111_y_rref", f"VRT{id_}fold111_x", f"VRT{id_}fold111_y_rref")
            ss.add_jump(f"JUMP{id_}fold111_y", f"VRT{id_}fold111_y_rref", f"VRT{id_}fold111_y")
            ss.add_jump(f"JUMP{id_}fold111_z_rref", f"VRT{id_}fold111_y", f"VRT{id_}fold111_z_rref")
            ss.add_jump(f"JUMP{id_}fold111_z", f"VRT{id_}fold111_z_rref", f"VRT{id_}fold111_z")
            ss.add_jump(f"JUMP{id_}fold111_sds", f"VRT{id_}fold111_z", f"VRT{id_}fold111_sds")
            ss.add_jump(f"JUMP{id_}fold111_subunit", f"VRT{id_}fold111_sds", f"SUBUNIT")

            ss.jumpgroups["JUMPGROUP1"].append(f"JUMP{id_}fold1")
            ss.jumpgroups["JUMPGROUP2"].append(f"JUMP{id_}fold1_z")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold111")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold111_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold111_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold111_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold111_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold111_subunit")

        ss._set_init_vrts()
        return ss

    def _add_3f_extra_chains(self, ss):

        # First we make completly new 3-folds
        ss_t = copy.deepcopy(self)
        ss_t.apply_dofs()
        orig = ss_t.get_vrt(f"VRT31fold1_z").vrt_orig
        if ss.righthanded:
            angle = -72 * 2
            angle2 = 120
        else:
            angle = 72 * 2
            angle2 = -120

        # modify the energy term!
        # this depends on if it is righthanded or not

        if ss.righthanded:
            new_3F = " + ".join(f"60*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in
                                zip(["L", "T", "H", "K", "Y", "X"], ["111", "111", "111", "111", "111", "111"]))
            new_2F = " + ".join(f"30*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in zip(["Z"], ["111"]))
        else:
            new_3F = " + ".join(f"60*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in
                                zip(["K", "H", "T", "L", "X", "Y"], ["111", "111", "111", "111", "111", "111"]))
            new_2F = " + ".join(f"30*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in zip(["Z"], ["111"]))
        ss.energies += f" + {new_3F} + {new_2F}"

        for (r1, r2, r4), id_ in zip([(180, 120, -72), (180, -120, 72), (180, angle2, angle)], ["X", "Y", "Z"]):

            # The angles to get for
            a = vector_angle([0, 0, 1], orig)
            c = math.degrees(math.pi / 2 - math.atan(1/2))
            final_angle = 180 - a*2 - c
            rotvec = np.cross([0, 0, 1], orig)
            R = rotation_matrix(orig, r1)
            R2 = rotation_matrix(orig, r2)
            R3 = rotation_matrix(rotvec, final_angle)
            R4 = rotation_matrix([0, 0, 1], r4)

            f = "31"

            # add vrts no
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_tref", f"VRT{id_}fold1_z_tref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold", f"VRT{id_}fold").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1", f"VRT{id_}fold1").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_rref", f"VRT{id_}fold1_z_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z", f"VRT{id_}fold1_z").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))

            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold11", f"VRT{id_}fold11").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_tref", f"VRT{id_}fold111_x_tref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111", f"VRT{id_}fold111").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_rref", f"VRT{id_}fold111_x_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x", f"VRT{id_}fold111_x").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y_rref", f"VRT{id_}fold111_y_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y", f"VRT{id_}fold111_y").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z_rref", f"VRT{id_}fold111_z_rref").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z", f"VRT{id_}fold111_z").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_sds", f"VRT{id_}fold111_sds").__getattribute__("rotate")(R, True).__getattribute__("rotate")(R2, True).__getattribute__("rotate")(R3, True).__getattribute__("rotate")(R4, True))

            ss.add_jump(f"JUMP{id_}fold1_z_tref", f"VRTglobal", f"VRT{id_}fold1_z_tref")
            ss.add_jump(f"JUMP{id_}fold", f"VRT{id_}fold1_z_tref", f"VRT{id_}fold")
            ss.add_jump(f"JUMP{id_}fold1", f"VRT{id_}fold", f"VRT{id_}fold1")
            ss.add_jump(f"JUMP{id_}fold1_z_rref", f"VRT{id_}fold1", f"VRT{id_}fold1_z_rref")
            ss.add_jump(f"JUMP{id_}fold1_z", f"VRT{id_}fold1_z_rref", f"VRT{id_}fold1_z")
            ss.add_jump(f"JUMP{id_}fold111_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold111_x_tref")
            ss.add_jump(f"JUMP{id_}fold11", f"VRT{id_}fold111_x_tref", f"VRT{id_}fold11")
            ss.add_jump(f"JUMP{id_}fold111", f"VRT{id_}fold11", f"VRT{id_}fold111")
            ss.add_jump(f"JUMP{id_}fold111_x_rref", f"VRT{id_}fold111", f"VRT{id_}fold111_x_rref")
            ss.add_jump(f"JUMP{id_}fold111_x", f"VRT{id_}fold111_x_rref", f"VRT{id_}fold111_x")
            ss.add_jump(f"JUMP{id_}fold111_y_rref", f"VRT{id_}fold111_x", f"VRT{id_}fold111_y_rref")
            ss.add_jump(f"JUMP{id_}fold111_y", f"VRT{id_}fold111_y_rref", f"VRT{id_}fold111_y")
            ss.add_jump(f"JUMP{id_}fold111_z_rref", f"VRT{id_}fold111_y", f"VRT{id_}fold111_z_rref")
            ss.add_jump(f"JUMP{id_}fold111_z", f"VRT{id_}fold111_z_rref", f"VRT{id_}fold111_z")
            ss.add_jump(f"JUMP{id_}fold111_sds", f"VRT{id_}fold111_z", f"VRT{id_}fold111_sds")
            ss.add_jump(f"JUMP{id_}fold111_subunit", f"VRT{id_}fold111_sds", f"SUBUNIT")

            ss.jumpgroups["JUMPGROUP1"].append(f"JUMP{id_}fold1")
            ss.jumpgroups["JUMPGROUP2"].append(f"JUMP{id_}fold1_z")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold111")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold111_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold111_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold111_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold111_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold111_subunit")

        # Now the rest we can add to existing 3-fodls
        for f, id_, angle in zip(("33", "32", "34", "35"), ("L", "T", "K", "H"), (-120, 120, 120, -120)):

            ss_t = copy.deepcopy(self)
            ss_t.apply_dofs()
            orig = ss_t.get_vrt(f"VRT{f}fold1_z").vrt_orig
            R = rotation_matrix(orig, angle)

            # add vrts no
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_tref", f"VRT{id_}fold1_z_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold", f"VRT{id_}fold").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1", f"VRT{id_}fold1").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_rref", f"VRT{id_}fold1_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z", f"VRT{id_}fold1_z").__getattribute__("rotate")(R, True))

            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold11", f"VRT{id_}fold11").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_tref", f"VRT{id_}fold111_x_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111", f"VRT{id_}fold111").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_rref", f"VRT{id_}fold111_x_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x", f"VRT{id_}fold111_x").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y_rref", f"VRT{id_}fold111_y_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y", f"VRT{id_}fold111_y").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z_rref", f"VRT{id_}fold111_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z", f"VRT{id_}fold111_z").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_sds", f"VRT{id_}fold111_sds").__getattribute__("rotate")(R, True))

            ss.add_jump(f"JUMP{id_}fold1_z_tref", f"VRTglobal", f"VRT{id_}fold1_z_tref")
            ss.add_jump(f"JUMP{id_}fold", f"VRT{id_}fold1_z_tref", f"VRT{id_}fold")
            ss.add_jump(f"JUMP{id_}fold1", f"VRT{id_}fold", f"VRT{id_}fold1")
            ss.add_jump(f"JUMP{id_}fold1_z_rref", f"VRT{id_}fold1", f"VRT{id_}fold1_z_rref")
            ss.add_jump(f"JUMP{id_}fold1_z", f"VRT{id_}fold1_z_rref", f"VRT{id_}fold1_z")
            ss.add_jump(f"JUMP{id_}fold111_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold111_x_tref")
            ss.add_jump(f"JUMP{id_}fold11", f"VRT{id_}fold111_x_tref", f"VRT{id_}fold11")
            ss.add_jump(f"JUMP{id_}fold111", f"VRT{id_}fold11", f"VRT{id_}fold111")
            ss.add_jump(f"JUMP{id_}fold111_x_rref", f"VRT{id_}fold111", f"VRT{id_}fold111_x_rref")
            ss.add_jump(f"JUMP{id_}fold111_x", f"VRT{id_}fold111_x_rref", f"VRT{id_}fold111_x")
            ss.add_jump(f"JUMP{id_}fold111_y_rref", f"VRT{id_}fold111_x", f"VRT{id_}fold111_y_rref")
            ss.add_jump(f"JUMP{id_}fold111_y", f"VRT{id_}fold111_y_rref", f"VRT{id_}fold111_y")
            ss.add_jump(f"JUMP{id_}fold111_z_rref", f"VRT{id_}fold111_y", f"VRT{id_}fold111_z_rref")
            ss.add_jump(f"JUMP{id_}fold111_z", f"VRT{id_}fold111_z_rref", f"VRT{id_}fold111_z")
            ss.add_jump(f"JUMP{id_}fold111_sds", f"VRT{id_}fold111_z", f"VRT{id_}fold111_sds")
            ss.add_jump(f"JUMP{id_}fold111_subunit", f"VRT{id_}fold111_sds", f"SUBUNIT")

            ss.jumpgroups["JUMPGROUP1"].append(f"JUMP{id_}fold1")
            ss.jumpgroups["JUMPGROUP2"].append(f"JUMP{id_}fold1_z")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold111")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold111_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold111_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold111_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold111_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold111_subunit")


        ss._set_init_vrts()
        return ss

    def _add_hf_extra_chains(self, ss):

        # get the roration angles
        if ss.righthanded:
            angles1 = [-72, 72*2]
            angles2 = [-72*2, None]
            angle3 =  72
            angle4 = -72 * 2
        else:
            angles1 = [72, -72*2]
            angles2 = [72*2, None]
            angle3 = -72
            angle4 = 72 * 2

        # modify the energy term!
        new_3F = " + ".join(f"60*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in zip(["X", "X", "Y", "Y", "3", "2"], ["111", "121", "111", "121", "131", "131"]))
        new_2F = " + ".join(f"30*(VRTHFfold111_sds:VRT{i}fold{j}_sds)" for i, j in zip(["X"], ["131"]))
        ss.energies += f" + {new_3F} + {new_2F}"

        # Which fold to use
        f = "2"

        for id_, angle, angle2 in zip(["X", "Y"], angles1, angles2):

            # rotate them
            R = rotation_matrix([0, 0, 1], angle)

            # add vrts no
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_tref",  f"VRT{id_}fold1_z_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold", f"VRT{id_}fold").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1", f"VRT{id_}fold1").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z_rref", f"VRT{id_}fold1_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold1_z", f"VRT{id_}fold1_z").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_tref",  f"VRT{id_}fold111_x_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold11", f"VRT{id_}fold11").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111", f"VRT{id_}fold111").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x_rref",  f"VRT{id_}fold111_x_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_x", f"VRT{id_}fold111_x").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y_rref",f"VRT{id_}fold111_y_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_y", f"VRT{id_}fold111_y").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z_rref", f"VRT{id_}fold111_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_z", f"VRT{id_}fold111_z").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold111_sds", f"VRT{id_}fold111_sds").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_tref", f"VRT{id_}fold121_x_tref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold12", f"VRT{id_}fold12").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121", f"VRT{id_}fold121").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_rref", f"VRT{id_}fold121_x_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x", f"VRT{id_}fold121_x").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y_rref", f"VRT{id_}fold121_y_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y", f"VRT{id_}fold121_y").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z_rref", f"VRT{id_}fold121_z_rref").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z", f"VRT{id_}fold121_z").__getattribute__("rotate")(R, True))
            ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_sds", f"VRT{id_}fold121_sds").__getattribute__("rotate")(R, True))

            # add jumps
            ss.add_jump(f"JUMP{id_}fold1_z_tref", f"VRTglobal", f"VRT{id_}fold1_z_tref")
            ss.add_jump(f"JUMP{id_}fold", f"VRT{id_}fold1_z_tref", f"VRT{id_}fold")
            ss.add_jump(f"JUMP{id_}fold1", f"VRT{id_}fold", f"VRT{id_}fold1")
            ss.add_jump(f"JUMP{id_}fold1_z_rref", f"VRT{id_}fold1", f"VRT{id_}fold1_z_rref")
            ss.add_jump(f"JUMP{id_}fold1_z", f"VRT{id_}fold1_z_rref", f"VRT{id_}fold1_z")
            ss.add_jump(f"JUMP{id_}fold111_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold111_x_tref")
            ss.add_jump(f"JUMP{id_}fold11", f"VRT{id_}fold111_x_tref", f"VRT{id_}fold11")
            ss.add_jump(f"JUMP{id_}fold111", f"VRT{id_}fold11", f"VRT{id_}fold111")
            ss.add_jump(f"JUMP{id_}fold111_x_rref", f"VRT{id_}fold111", f"VRT{id_}fold111_x_rref")
            ss.add_jump(f"JUMP{id_}fold111_x", f"VRT{id_}fold111_x_rref", f"VRT{id_}fold111_x")
            ss.add_jump(f"JUMP{id_}fold111_y_rref", f"VRT{id_}fold111_x", f"VRT{id_}fold111_y_rref")
            ss.add_jump(f"JUMP{id_}fold111_y", f"VRT{id_}fold111_y_rref", f"VRT{id_}fold111_y")
            ss.add_jump(f"JUMP{id_}fold111_z_rref", f"VRT{id_}fold111_y", f"VRT{id_}fold111_z_rref")
            ss.add_jump(f"JUMP{id_}fold111_z", f"VRT{id_}fold111_z_rref", f"VRT{id_}fold111_z")
            ss.add_jump(f"JUMP{id_}fold111_sds", f"VRT{id_}fold111_z", f"VRT{id_}fold111_sds")
            ss.add_jump(f"JUMP{id_}fold111_subunit", f"VRT{id_}fold111_sds", f"SUBUNIT")
            ss.add_jump(f"JUMP{id_}fold121_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold121_x_tref")
            ss.add_jump(f"JUMP{id_}fold12", f"VRT{id_}fold121_x_tref", f"VRT{id_}fold12")
            ss.add_jump(f"JUMP{id_}fold121", f"VRT{id_}fold12", f"VRT{id_}fold121")
            ss.add_jump(f"JUMP{id_}fold121_x_rref", f"VRT{id_}fold121", f"VRT{id_}fold121_x_rref")
            ss.add_jump(f"JUMP{id_}fold121_x", f"VRT{id_}fold121_x_rref", f"VRT{id_}fold121_x")
            ss.add_jump(f"JUMP{id_}fold121_y_rref", f"VRT{id_}fold121_x", f"VRT{id_}fold121_y_rref")
            ss.add_jump(f"JUMP{id_}fold121_y",  f"VRT{id_}fold121_y_rref", f"VRT{id_}fold121_y")
            ss.add_jump(f"JUMP{id_}fold121_z_rref", f"VRT{id_}fold121_y", f"VRT{id_}fold121_z_rref")
            ss.add_jump(f"JUMP{id_}fold121_z", f"VRT{id_}fold121_z_rref", f"VRT{id_}fold121_z")
            ss.add_jump(f"JUMP{id_}fold121_sds", f"VRT{id_}fold121_z", f"VRT{id_}fold121_sds")
            ss.add_jump(f"JUMP{id_}fold121_subunit", f"VRT{id_}fold121_sds", f"SUBUNIT")

            # jumpgroups
            ss.jumpgroups["JUMPGROUP1"].append(f"JUMP{id_}fold1")
            ss.jumpgroups["JUMPGROUP2"].append(f"JUMP{id_}fold1_z")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold111")
            ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold121")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold111_x")
            ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold121_x")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold111_y")
            ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold121_y")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold111_z")
            ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold121_z")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold111_sds")
            ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold121_sds")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold111_subunit")
            ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold121_subunit")

            if angle2 is not None:

                # add one more more in the
                R2 = rotation_matrix(ss.get_vrt(f"VRT{id_}fold111").vrt_z, angle2)
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_x_tref", f"VRT{id_}fold131_x_tref").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold12", f"VRT{id_}fold13").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121", f"VRT{id_}fold131").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_x_rref", f"VRT{id_}fold131_x_rref").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_x", f"VRT{id_}fold131_x").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_y_rref", f"VRT{id_}fold131_y_rref").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_y", f"VRT{id_}fold131_y").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_z_rref", f"VRT{id_}fold131_z_rref").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_z", f"VRT{id_}fold131_z").__getattribute__("rotate")(R2, True))
                ss.add_vrt(ss.copy_vrt(f"VRT{id_}fold121_sds", f"VRT{id_}fold131_sds").__getattribute__("rotate")(R2, True))

                ss.add_jump(f"JUMP{id_}fold131_x_tref", f"VRT{id_}fold1_z", f"VRT{id_}fold131_x_tref")
                ss.add_jump(f"JUMP{id_}fold13", f"VRT{id_}fold131_x_tref", f"VRT{id_}fold13")
                ss.add_jump(f"JUMP{id_}fold131", f"VRT{id_}fold13", f"VRT{id_}fold131")
                ss.add_jump(f"JUMP{id_}fold131_x_rref", f"VRT{id_}fold131", f"VRT{id_}fold131_x_rref")
                ss.add_jump(f"JUMP{id_}fold131_x", f"VRT{id_}fold131_x_rref", f"VRT{id_}fold131_x")
                ss.add_jump(f"JUMP{id_}fold131_y_rref", f"VRT{id_}fold131_x", f"VRT{id_}fold131_y_rref")
                ss.add_jump(f"JUMP{id_}fold131_y", f"VRT{id_}fold131_y_rref", f"VRT{id_}fold131_y")
                ss.add_jump(f"JUMP{id_}fold131_z_rref", f"VRT{id_}fold131_y", f"VRT{id_}fold131_z_rref")
                ss.add_jump(f"JUMP{id_}fold131_z", f"VRT{id_}fold131_z_rref", f"VRT{id_}fold131_z")
                ss.add_jump(f"JUMP{id_}fold131_sds", f"VRT{id_}fold131_z", f"VRT{id_}fold131_sds")
                ss.add_jump(f"JUMP{id_}fold131_subunit", f"VRT{id_}fold131_sds", f"SUBUNIT")

                # jumpgroups
                ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{id_}fold131")
                ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{id_}fold131_x")
                ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{id_}fold131_y")
                ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{id_}fold131_z")
                ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{id_}fold131_sds")
                ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{id_}fold131_subunit")

        # i also want to add a new
        # in the 2_fold

        # add one more more in the
        f = "3"
        R3 = rotation_matrix(ss.get_vrt(f"VRT{f}fold111").vrt_z, angle3)
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_tref", f"VRT{f}fold131_x_tref").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold12", f"VRT{f}fold13").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121", f"VRT{f}fold131").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_rref", f"VRT{f}fold131_x_rref").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x", f"VRT{f}fold131_x").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y_rref", f"VRT{f}fold131_y_rref").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y", f"VRT{f}fold131_y").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z_rref", f"VRT{f}fold131_z_rref").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z", f"VRT{f}fold131_z").__getattribute__("rotate")(R3, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_sds", f"VRT{f}fold131_sds").__getattribute__("rotate")(R3, True))

        ss.add_jump(f"JUMP{f}fold131_x_tref", f"VRT{f}fold1_z", f"VRT{f}fold131_x_tref")
        ss.add_jump(f"JUMP{f}fold13", f"VRT{f}fold131_x_tref", f"VRT{f}fold13")
        ss.add_jump(f"JUMP{f}fold131", f"VRT{f}fold13", f"VRT{f}fold131")
        ss.add_jump(f"JUMP{f}fold131_x_rref", f"VRT{f}fold131", f"VRT{f}fold131_x_rref")
        ss.add_jump(f"JUMP{f}fold131_x", f"VRT{f}fold131_x_rref", f"VRT{f}fold131_x")
        ss.add_jump(f"JUMP{f}fold131_y_rref", f"VRT{f}fold131_x", f"VRT{f}fold131_y_rref")
        ss.add_jump(f"JUMP{f}fold131_y", f"VRT{f}fold131_y_rref", f"VRT{f}fold131_y")
        ss.add_jump(f"JUMP{f}fold131_z_rref", f"VRT{f}fold131_y", f"VRT{f}fold131_z_rref")
        ss.add_jump(f"JUMP{f}fold131_z", f"VRT{f}fold131_z_rref", f"VRT{f}fold131_z")
        ss.add_jump(f"JUMP{f}fold131_sds", f"VRT{f}fold131_z", f"VRT{f}fold131_sds")
        ss.add_jump(f"JUMP{f}fold131_subunit", f"VRT{f}fold131_sds", f"SUBUNIT")

        # jumpgroups
        ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{f}fold131")
        ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{f}fold131_x")
        ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{f}fold131_y")
        ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{f}fold131_z")
        ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{f}fold131_sds")
        ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{f}fold131_subunit")

        # add one more more in the
        f = "2"
        R4 = rotation_matrix(ss.get_vrt(f"VRT{f}fold111").vrt_z, angle4)
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_tref", f"VRT{f}fold131_x_tref").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold12", f"VRT{f}fold13").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121", f"VRT{f}fold131").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x_rref", f"VRT{f}fold131_x_rref").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_x", f"VRT{f}fold131_x").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y_rref", f"VRT{f}fold131_y_rref").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_y", f"VRT{f}fold131_y").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z_rref", f"VRT{f}fold131_z_rref").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_z", f"VRT{f}fold131_z").__getattribute__("rotate")(R4, True))
        ss.add_vrt(ss.copy_vrt(f"VRT{f}fold121_sds", f"VRT{f}fold131_sds").__getattribute__("rotate")(R4, True))

        ss.add_jump(f"JUMP{f}fold131_x_tref", f"VRT{f}fold1_z", f"VRT{f}fold131_x_tref")
        ss.add_jump(f"JUMP{f}fold13", f"VRT{f}fold131_x_tref", f"VRT{f}fold13")
        ss.add_jump(f"JUMP{f}fold131", f"VRT{f}fold13", f"VRT{f}fold131")
        ss.add_jump(f"JUMP{f}fold131_x_rref", f"VRT{f}fold131", f"VRT{f}fold131_x_rref")
        ss.add_jump(f"JUMP{f}fold131_x", f"VRT{f}fold131_x_rref", f"VRT{f}fold131_x")
        ss.add_jump(f"JUMP{f}fold131_y_rref", f"VRT{f}fold131_x", f"VRT{f}fold131_y_rref")
        ss.add_jump(f"JUMP{f}fold131_y", f"VRT{f}fold131_y_rref", f"VRT{f}fold131_y")
        ss.add_jump(f"JUMP{f}fold131_z_rref", f"VRT{f}fold131_y", f"VRT{f}fold131_z_rref")
        ss.add_jump(f"JUMP{f}fold131_z", f"VRT{f}fold131_z_rref", f"VRT{f}fold131_z")
        ss.add_jump(f"JUMP{f}fold131_sds", f"VRT{f}fold131_z", f"VRT{f}fold131_sds")
        ss.add_jump(f"JUMP{f}fold131_subunit", f"VRT{f}fold131_sds", f"SUBUNIT")

        # jumpgroups
        ss.jumpgroups["JUMPGROUP3"].append(f"JUMP{f}fold131")
        ss.jumpgroups["JUMPGROUP4"].append(f"JUMP{f}fold131_x")
        ss.jumpgroups["JUMPGROUP5"].append(f"JUMP{f}fold131_y")
        ss.jumpgroups["JUMPGROUP6"].append(f"JUMP{f}fold131_z")
        ss.jumpgroups["JUMPGROUP7"].append(f"JUMP{f}fold131_sds")
        ss.jumpgroups["JUMPGROUP8"].append(f"JUMP{f}fold131_subunit")

        ss._set_init_vrts()
        return ss
