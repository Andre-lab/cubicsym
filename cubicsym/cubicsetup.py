#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicSetup class
@Author: Mads Jeppesen
@Date: 9/21/22
"""
from symmetryhandler.symmetrysetup import SymmetrySetup
import numpy as np
import copy
from io import StringIO
import math
from scipy.spatial.transform import Rotation as R
from symmetryhandler.mathfunctions import rotation_matrix, vector_angle, vector_projection_on_subspace
from symmetryhandler.kinematics import set_jumpdof_str_str, get_dofs
from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
from pyrosetta import Pose
from pyrosetta.rosetta.core.pose.symmetry import extract_asymmetric_unit
import textwrap
from cubicsym.utilities import cut_all_but_chains

class CubicSetup(SymmetrySetup):

    def __init__(self, file=None, pose=None, symmetry_name=None):
        super().__init__(file, pose, symmetry_name)
        self.symmetry_type = self.cubic_symmetry_from_setup()
        self.righthanded = self.is_rightanded()
        self.chain_map = {k: v for k, v in zip(tuple("ABCDEFGHI"), tuple(range(1, 10)))}

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

    def __get_chains_ids(self, ids, rosetta_number):
        if rosetta_number:
            return tuple([self.chain_map[i] for i in ids])
        return ids

    def get_HF_chain_ids(self, rosetta_number=False):
        """Get the HF fold chains names either as a str (default) or Rosetta number."""
        if "I" == self.symmetry_type:
            return self.__get_chains_ids(tuple("ABCDE"), rosetta_number)
        if "O" == self.symmetry_type:
            return self.__get_chains_ids(tuple("ABCD"), rosetta_number)
        if "T" == self.symmetry_type:
            return self.__get_chains_ids(tuple("ABC"), rosetta_number)

    def get_3fold_chain_ids(self, rosetta_number=False):
        """Get the 3 fold chains names either as a str (default) or Rosetta number.."""
        if "I" == self.symmetry_type:
            return self.__get_chains_ids(tuple("AIF"), rosetta_number)
        if "O" == self.symmetry_type:
            return self.__get_chains_ids(tuple("AEH"), rosetta_number)
        if "T" == self.symmetry_type:
            return self.__get_chains_ids(tuple("ADG"), rosetta_number)

    def get_2fold_chain_ids(self, rosetta_number=False):
        """Get the 2 fold chains names either as a str (default) or Rosetta number. This returns both of the 2 folds.
        The first one is closest and the second one is the furthest."""

        if "I" == self.symmetry_type:
            return self.__get_chains_ids(tuple("AH"), rosetta_number), self.__get_chains_ids(tuple("AG"), rosetta_number)
        if "O" == self.symmetry_type:
            return self.__get_chains_ids(tuple("AG"), rosetta_number), self.__get_chains_ids(tuple("AF"), rosetta_number)
        if "T" == self.symmetry_type:
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

    def cubic_symmetry_from_setup(self):
        """Determine the cubic symmetry from a SymmetrySetup object."""
        if "60" in self.energies:
            return "I"
        elif "24" in self.energies:
            return "O"
        elif "12" in self.energies:
            return "T"
        else:
            raise ValueError("Symmetry is not cubic!")

    # fixme: dangerous if you have more than 1 subunit as the monomer
    @staticmethod
    def cubic_symmetry_from_pose(pose):
        nsubs = pose.conformation().Symmetry_Info().subunits()
        if nsubs == 9:
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
        # v_5fold_center_to_5fold_slave2_com = symmetry_setup.get_vrt_name("VRT5fold1211").vrt_orig - symmetry_setup.get_vrt_name("VRT5fold1").vrt_orig
        # v_5fold_center_to_5fold_slave5_com = symmetry_setup.get_vrt_name("VRT5fold1511").vrt_orig - symmetry_setup.get_vrt_name("VRT5fold1").vrt_orig

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
            # cmd.do(f"v_5fold_center_to_5fold_master_com {symmetry_setup.get_vrt_name('VRT5fold1').vrt_orig}, {symmetry_setup.get_vrt_name("VRT5fold1111").vrt_orig})
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
          E = 60*VRT5fold1111 + 60*(VRT5fold1111:VRT5fold1211) + 60*(VRT5fold1111:VRT5fold1311)
          anchor_residue COM
          virtual_coordinates_start
          {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRT5fold")}
          {self.get_vrt("VRT5fold1")}
          {self.get_vrt("VRT5fold11")}
          {self.get_vrt("VRT5fold111")}
          {self.get_vrt("VRT5fold1111")}
          {self.get_vrt("VRT5fold12")}
          {self.get_vrt("VRT5fold121")}
          {self.get_vrt("VRT5fold1211")}
          {self.get_vrt("VRT5fold13")}
          {self.get_vrt("VRT5fold131")}
          {self.get_vrt("VRT5fold1311")}
          {self.get_vrt("VRT5fold14")}
          {self.get_vrt("VRT5fold141")}
          {self.get_vrt("VRT5fold1411")}
          {self.get_vrt("VRT5fold15")}
          {self.get_vrt("VRT5fold151")}
          {self.get_vrt("VRT5fold1511")}
          virtual_coordinates_stop  
          connect_virtual JUMP5fold VRTglobal VRT5fold
          connect_virtual JUMP5fold1 VRT5fold VRT5fold1
          connect_virtual JUMP5fold11 VRT5fold1 VRT5fold11
          connect_virtual JUMP5fold111 VRT5fold11 VRT5fold111
          connect_virtual JUMP5fold1111 VRT5fold111 VRT5fold1111
          connect_virtual JUMP5fold1111_subunit VRT5fold1111 SUBUNIT
          connect_virtual JUMP5fold12 VRT5fold1 VRT5fold12
          connect_virtual JUMP5fold121 VRT5fold12 VRT5fold121
          connect_virtual JUMP5fold1211 VRT5fold121 VRT5fold1211
          connect_virtual JUMP5fold1211_subunit VRT5fold1211 SUBUNIT
          connect_virtual JUMP5fold13 VRT5fold1 VRT5fold13
          connect_virtual JUMP5fold131 VRT5fold13 VRT5fold131
          connect_virtual JUMP5fold1311 VRT5fold131 VRT5fold1311
          connect_virtual JUMP5fold1311_subunit VRT5fold1311 SUBUNIT
          connect_virtual JUMP5fold14 VRT5fold1 VRT5fold14
          connect_virtual JUMP5fold141 VRT5fold14 VRT5fold141
          connect_virtual JUMP5fold1411 VRT5fold141 VRT5fold1411
          connect_virtual JUMP5fold1411_subunit VRT5fold1411 SUBUNIT
          connect_virtual JUMP5fold15 VRT5fold1 VRT5fold15
          connect_virtual JUMP5fold151 VRT5fold15 VRT5fold151
          connect_virtual JUMP5fold1511 VRT5fold151 VRT5fold1511
          connect_virtual JUMP5fold1511_subunit VRT5fold1511 SUBUNIT
          set_dof JUMP5fold1 z({symmetry_setup._dofs['JUMP5fold1'][0][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1'][1][2]})
          set_dof JUMP5fold111 x({symmetry_setup._dofs['JUMP5fold111'][0][2]})
          set_dof JUMP5fold1111 angle_x({symmetry_setup._dofs['JUMP5fold1111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111'][2][2]})
          set_dof JUMP5fold1111_subunit angle_x({symmetry_setup._dofs['JUMP5fold1111_subunit'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111_subunit'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111_subunit'][2][2]})
          set_jump_group JUMPGROUP1 JUMP5fold1 
          set_jump_group JUMPGROUP2 JUMP5fold111 JUMP5fold121 JUMP5fold131 JUMP5fold141 JUMP5fold151 
          set_jump_group JUMPGROUP3 JUMP5fold1111 JUMP5fold1211 JUMP5fold1311 JUMP5fold1411 JUMP5fold1511 
          set_jump_group JUMPGROUP4 JUMP5fold1111_subunit JUMP5fold1211_subunit JUMP5fold1311_subunit JUMP5fold1411_subunit JUMP5fold1511_subunit 
          """)))

        # TODO: change the symmetry so that depending on if it is 4v4m or 1stm different symmetries have to be used

        fold3 = CubicSetup()
        fold3.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name 3fold
          E = 60*VRT5fold1111 + 60*(VRT5fold1111:VRT3fold1111)
          anchor_residue COM
          virtual_coordinates_start
          {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRT5fold")}
          {self.get_vrt("VRT5fold1")}
          {self.get_vrt("VRT5fold11")}
          {self.get_vrt("VRT5fold111")}
          {self.get_vrt("VRT5fold1111")}
          {self.get_vrt("VRT3fold")}
          {self.get_vrt("VRT3fold1")}
          {self.get_vrt("VRT3fold11")}
          {self.get_vrt("VRT3fold111")}
          {self.get_vrt("VRT3fold1111")}
          {self.get_vrt("VRT2fold")}
          {self.get_vrt("VRT2fold1")}
          {self.get_vrt("VRT2fold12")}
          {self.get_vrt("VRT2fold121")}
          {self.get_vrt("VRT2fold1211")}
          virtual_coordinates_stop
          connect_virtual JUMP5fold VRTglobal VRT5fold
          connect_virtual JUMP5fold1 VRT5fold VRT5fold1
          connect_virtual JUMP5fold11 VRT5fold1 VRT5fold11
          connect_virtual JUMP5fold111 VRT5fold11 VRT5fold111
          connect_virtual JUMP5fold1111 VRT5fold111 VRT5fold1111
          connect_virtual JUMP5fold1111_subunit VRT5fold1111 SUBUNIT
          connect_virtual JUMP3fold VRTglobal VRT3fold
          connect_virtual JUMP3fold1 VRT3fold VRT3fold1
          connect_virtual JUMP3fold11 VRT3fold1 VRT3fold11
          connect_virtual JUMP3fold111 VRT3fold11 VRT3fold111
          connect_virtual JUMP3fold1111 VRT3fold111 VRT3fold1111
          connect_virtual JUMP3fold1111_subunit VRT3fold1111 SUBUNIT
          connect_virtual JUMP2fold VRTglobal VRT2fold
          connect_virtual JUMP2fold1 VRT2fold VRT2fold1
          connect_virtual JUMP2fold12 VRT2fold1 VRT2fold12
          connect_virtual JUMP2fold121 VRT2fold12 VRT2fold121
          connect_virtual JUMP2fold1211 VRT2fold121 VRT2fold1211
          connect_virtual JUMP2fold1211_subunit VRT2fold1211 SUBUNIT
          set_dof JUMP5fold1 z({symmetry_setup._dofs['JUMP5fold1'][0][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1'][1][2]})
          set_dof JUMP5fold111 x({symmetry_setup._dofs['JUMP5fold111'][0][2]})
          set_dof JUMP5fold1111 angle_x({symmetry_setup._dofs['JUMP5fold1111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111'][2][2]})
          set_dof JUMP5fold1111_subunit angle_x({symmetry_setup._dofs['JUMP5fold1111_subunit'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111_subunit'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111_subunit'][2][2]})
          set_jump_group JUMPGROUP1 JUMP5fold1 JUMP3fold1 JUMP2fold1
          set_jump_group JUMPGROUP2 JUMP5fold111 JUMP3fold111  JUMP2fold121
          set_jump_group JUMPGROUP3 JUMP5fold1111 JUMP3fold1111  JUMP2fold1211
          set_jump_group JUMPGROUP4 JUMP5fold1111_subunit JUMP3fold1111_subunit JUMP2fold1211_subunit
          """)))

        fold2_1 = CubicSetup()
        fold2_1.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name 2fold_1
          E = 60*VRT5fold1111 + 30*(VRT5fold1111:VRT2fold1111)
          anchor_residue COM
          virtual_coordinates_start
          {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRT5fold")}
          {self.get_vrt("VRT5fold1")}
          {self.get_vrt("VRT5fold11")}
          {self.get_vrt("VRT5fold111")}
          {self.get_vrt("VRT5fold1111")}
          {self.get_vrt("VRT2fold")}
          {self.get_vrt("VRT2fold1")}
          {self.get_vrt("VRT2fold11")}
          {self.get_vrt("VRT2fold111")}
          {self.get_vrt("VRT2fold1111")}
          virtual_coordinates_stop
          connect_virtual JUMP5fold VRTglobal VRT5fold
          connect_virtual JUMP5fold1 VRT5fold VRT5fold1
          connect_virtual JUMP5fold11 VRT5fold1 VRT5fold11
          connect_virtual JUMP5fold111 VRT5fold11 VRT5fold111
          connect_virtual JUMP5fold1111 VRT5fold111 VRT5fold1111
          connect_virtual JUMP5fold1111_subunit VRT5fold1111 SUBUNIT
          connect_virtual JUMP2fold VRTglobal VRT2fold
          connect_virtual JUMP2fold1 VRT2fold VRT2fold1
          connect_virtual JUMP2fold11 VRT2fold1 VRT2fold11
          connect_virtual JUMP2fold111 VRT2fold11 VRT2fold111
          connect_virtual JUMP2fold1111 VRT2fold111 VRT2fold1111
          connect_virtual JUMP2fold1111_subunit VRT2fold1111 SUBUNIT
          set_dof JUMP5fold1 z({symmetry_setup._dofs['JUMP5fold1'][0][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1'][1][2]})
          set_dof JUMP5fold111 x({symmetry_setup._dofs['JUMP5fold111'][0][2]})
          set_dof JUMP5fold1111 angle_x({symmetry_setup._dofs['JUMP5fold1111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111'][2][2]})
          set_dof JUMP5fold1111_subunit angle_x({symmetry_setup._dofs['JUMP5fold1111_subunit'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111_subunit'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111_subunit'][2][2]})
          set_jump_group JUMPGROUP1 JUMP5fold1 JUMP2fold1
          set_jump_group JUMPGROUP2 JUMP5fold111 JUMP2fold111 
          set_jump_group JUMPGROUP3 JUMP5fold1111 JUMP2fold1111 
          set_jump_group JUMPGROUP4 JUMP5fold1111_subunit JUMP2fold1111_subunit 
          """)))

        fold2_2 = CubicSetup()
        fold2_2.read_from_file(
            StringIO(textwrap.dedent(f"""symmetry_name fold2_2 
          E = 60*VRT5fold1111 + 30*(VRT5fold1111:VRT3fold1211)
          anchor_residue COM
          virtual_coordinates_start
          {self.get_vrt("VRTglobal")}
          {self.get_vrt("VRT5fold")}
          {self.get_vrt("VRT5fold1")}
          {self.get_vrt("VRT5fold11")}
          {self.get_vrt("VRT5fold111")}
          {self.get_vrt("VRT5fold1111")}
          {self.get_vrt("VRT3fold")}
          {self.get_vrt("VRT3fold1")}
          {self.get_vrt("VRT3fold12")}
          {self.get_vrt("VRT3fold121")}
          {self.get_vrt("VRT3fold1211")}
          virtual_coordinates_stop
          connect_virtual JUMP5fold VRTglobal VRT5fold
          connect_virtual JUMP5fold1 VRT5fold VRT5fold1
          connect_virtual JUMP5fold11 VRT5fold1 VRT5fold11
          connect_virtual JUMP5fold111 VRT5fold11 VRT5fold111
          connect_virtual JUMP5fold1111 VRT5fold111 VRT5fold1111
          connect_virtual JUMP5fold1111_subunit VRT5fold1111 SUBUNIT
          connect_virtual JUMP3fold VRTglobal VRT3fold
          connect_virtual JUMP3fold1 VRT3fold VRT3fold1
          connect_virtual JUMP3fold12 VRT3fold1 VRT3fold12
          connect_virtual JUMP3fold121 VRT3fold12 VRT3fold121
          connect_virtual JUMP3fold1211 VRT3fold121 VRT3fold1211
          connect_virtual JUMP3fold1211_subunit VRT3fold1211 SUBUNIT
          set_dof JUMP5fold1 z({symmetry_setup._dofs['JUMP5fold1'][0][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1'][1][2]})
          set_dof JUMP5fold111 x({symmetry_setup._dofs['JUMP5fold111'][0][2]})
          set_dof JUMP5fold1111 angle_x({symmetry_setup._dofs['JUMP5fold1111'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111'][2][2]})
          set_dof JUMP5fold1111_subunit angle_x({symmetry_setup._dofs['JUMP5fold1111_subunit'][0][2]}) angle_y({symmetry_setup._dofs['JUMP5fold1111_subunit'][1][2]}) angle_z({symmetry_setup._dofs['JUMP5fold1111_subunit'][2][2]})
          set_jump_group JUMPGROUP1 JUMP5fold1 JUMP3fold1
          set_jump_group JUMPGROUP2 JUMP5fold111 JUMP3fold121
          set_jump_group JUMPGROUP3 JUMP5fold1111 JUMP3fold1211
          set_jump_group JUMPGROUP4 JUMP5fold1111_subunit JUMP3fold1211_subunit
          """)))

        # setup_3fold = SymmetrySetup("3fold")
        # vrtglobal = symmetry_setup.get_vrt_name("VRTglobal")
        # center_of_3fold = np.array([symmetry_setup.get_vrt_name(vrt).vrt_orig for vrt in ("VRT5fold1111", "VRT3fold1111", "VRT2fold1111")]).sum(axis=1) / 3
        # rotation_to_3fold = rotation_matrix_from_vector_to_vector(vrtglobal.vrt_orig, center_of_3fold)
        # vrt3fold = copy.deepcopy(vrtglobal).rotate(rotation_to_3fold)

        return fold5, fold3, fold2_1, fold2_2

    def get_3fold_center_from_HFfold(self):
        """Returns the center of the 3-fold"""
        try: # for icosahedral symmetry
            a = self.get_vrt("VRTHFfold111_z").vrt_orig
            b = self.get_vrt("VRT2fold121_z").vrt_orig
            c = self.get_vrt("VRT3fold111_z").vrt_orig
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

    # fixme: should handle O and T case
    def is_rightanded(self):
        """Returns true if the point fold3_axis going to fold2_axis relative to the foldHF_axis is right-handed. It is left-handed if the cross product fold3_axis X fold2_axis
        points in the same direction as the foldF_axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        foldHF_axis = -self.get_vrt("VRTHFfold1").vrt_z
        fold3_axis = -self.get_vrt("VRT3fold1").vrt_z
        fold2_axis = -self.get_vrt("VRT2fold1").vrt_z
        return self._right_handed_vectors(fold3_axis, fold2_axis, foldHF_axis)

    @staticmethod
    def _create_final_ref_dofs(ss_f, ss_t, ss_f_nb1, ss_f_nb2, ss_t_nb1, ss_t_nb2, R=None, f="rotate",
                               suffix=""):  # ss, nb1:str, nb2:str, suffix:str= ""):
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
        else:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_x{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_x{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_y{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_y{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z_rref{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z_rref{suffix}"))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{ss_f_nb2}_z{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}"))
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
        ss_t.add_jump(f"JUMP{ss_t_nb1}fold{ss_t_nb2}_subunit{suffix}", f"VRT{ss_t_nb1}fold{ss_t_nb2}_z{suffix}", "SUBUNIT")

    @staticmethod
    def _create_base_dofs(ss_f, ss_t, ss_f_nb1, ss_t_nb1, base_vrt, R=None, f="rotate", suffix=""):
        """Based on VRTXfold create: VRTXfold1_z_tref -> VRTXfold (created before) -> VRTXfold1 -> VRTXfold1_z_rref -> VRTXfold1_z """
        ss_t.add_vrt(base_vrt)
        if not R is None:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold", f"VRT{ss_t_nb1}fold1_z_tref{suffix}", move_origo=True, axis="z",
                                       dir=1).__getattribute__(f)(R, True))
            # ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1{suffix}").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z_rref{suffix}", move_origo=True,
                                       axis="z").__getattribute__(f)(R, True))
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold{suffix}", f"VRT{ss_t_nb1}fold1_z{suffix}").__getattribute__(f)(R, True))
        else:
            ss_t.add_vrt(ss_f.copy_vrt(f"VRT{ss_f_nb1}fold", f"VRT{ss_t_nb1}fold1_z_tref{suffix}", move_origo=True, axis="z", dir=1))
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



    def create_O_3fold_based_symmetry(self, suffix=""):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        # colors are as given in HF symmetry
        ss3.energies = " + ".join((
           f"24*VRT31fold111_z{suffix}", # green
           f"24*(VRT31fold111_z{suffix}:VRT32fold111_z{suffix})", # ligth blue (4fold closest)
           f"24*(VRT31fold111_z{suffix}:VRT34fold111_z{suffix})", # pink (4fold furthest)
           f"24*(VRT31fold111_z{suffix}:VRT31fold121_z{suffix})", # brown (other 3fold)
           f"12*(VRT31fold111_z{suffix}:VRT32fold121_z{suffix})", # dark blue (2fold closest)
           f"12*(VRT31fold111_z{suffix}:VRT33fold131_z{suffix})")) # white

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
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP33fold111_subunit{suffix}", f"JUMP33fold131_subunit{suffix}", f"JUMP34fold111_subunit{suffix}")

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

    def create_O_2fold_based_symmetry(self, suffix=''):
        """Creates a 2-fold based symmetry file from a HF-based (CURRENTLY ONLY 5-fold) one."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        ss2.energies = " + ".join((
           f"24*VRT21fold111_z{suffix}", # green
           f"24*(VRT21fold111_z{suffix}:VRT24fold111_z{suffix})", # ligth blue (4fold closest)
           f"24*(VRT21fold111_z{suffix}:VRT23fold111_z{suffix})", # pink (4fold furthest)
           f"24*(VRT21fold111_z{suffix}:VRT22fold121_z{suffix})", # brown (other 3fold)
           f"12*(VRT21fold111_z{suffix}:VRT21fold121_z{suffix})", # dark blue (2fold closest)
           f"12*(VRT21fold111_z{suffix}:VRT25fold111_z{suffix})")) # white

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
        is_righthanded = self.is_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix)
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
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"25", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"26", ss_t_nb2=f"111", R=R, suffix=suffix)
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
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}", f"JUMP25fold111_subunit{suffix}",
                          f"JUMP26fold111_subunit{suffix}")

        return ss2

    def create_T_3fold_based_symmetry(self, suffix=""):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        # colors are as given in HF symmetry
        ss3.energies = " + ".join((
           f"12*VRT31fold111_z{suffix}", # green
           f"12*(VRT31fold111_z{suffix}:VRT31fold121_z{suffix})", # dark blue (other 3fold)
           f"12*(VRT31fold111_z{suffix}:VRT32fold111_z{suffix})", # light blue (3fold)
           f"6*(VRT31fold111_z{suffix}:VRT32fold121_z{suffix})", # brown (2-fold closest)
           f"6*(VRT31fold111_z{suffix}:VRT33fold131_z{suffix})")) # white (2-fold furthest)

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
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP33fold111_subunit{suffix}", f"JUMP33fold131_subunit{suffix}")

        return ss3

    def create_T_2fold_based_symmetry(self, suffix=''):
        """Creates a 2-fold based symmetry file from a Tetrahedral HF-based symmetry setup."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        ss2.energies = f"12*VRT21fold111_z{suffix} + " \
                       f"12*(VRT21fold111_z{suffix}:VRT23fold111_z{suffix}) + " \
                       f"12*(VRT21fold111_z{suffix}:VRT22fold121_z{suffix}) + " \
                       f"6*(VRT21fold111_z{suffix}:VRT21fold121_z{suffix}) + " \
                       f"6*(VRT21fold111_z{suffix}:VRT24fold111_z{suffix})"

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
        is_righthanded = self.is_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix)
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
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"24", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"25", ss_t_nb2=f"111", R=R, suffix=suffix)
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
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}",
                          f"JUMP25fold111_subunit{suffix}")

        return ss2

    # ASSUMING IT IS 5FOLD - well it can only be that so because O doesnt have 3fold and T already is
    def create_I_3fold_based_symmetry(self, suffix=''):

        # 1: find 3 fold and create a an initial setup based on that. Use the same anchor atom
        # 2: rotate +72 and -72 and include 2 of the chains present in the 5-fold setup
        # 3: rotate all around and include the rest of the fivefold
        ss3 = CubicSetup()
        ss3.reference_symmetric = True
        ss3.symmetry_name = self.symmetry_name + "_3fold_based"
        ss3.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        ss3.energies = f"60*VRT31fold111_z{suffix} + " \
                       f"60*(VRT31fold111_z{suffix}:VRT32fold111_z{suffix}) + 60*(VRT31fold111_z{suffix}:VRT33fold111_z{suffix}) + " \
                       f"60*(VRT31fold111_z{suffix}:VRT31fold121_z{suffix}) + " \
                       f"30*(VRT31fold111_z{suffix}:VRT32fold121_z{suffix}) + 30*(VRT31fold111_z{suffix}:VRT35fold131_z{suffix})"

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
        ss3._create_final_ref_dofs(self, ss3, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="111", suffix=suffix)
        # ---- chain 2 ----
        R = rotation_matrix(center3, 120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold12{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold121{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold12{suffix}", f"VRT31fold1{suffix}", f"VRT31fold12{suffix}")
        # ss3.add_jump(f"JUMP31fold121{suffix}", f"VRT31fold12{suffix}", f"VRT31fold121{suffix}")

        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="2", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="121", R=R, f="rotate_right_multiply",
                                   suffix=suffix)
        # ---- chain 3 ----
        R = rotation_matrix(center3, -120)
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold11{suffix}", f"VRT31fold13{suffix}").rotate_right_multiply(R, True))
        # ss3.add_vrt(ss3.copy_vrt(f"VRT31fold111{suffix}", f"VRT31fold131{suffix}").rotate_right_multiply(R, True))
        # ss3.add_jump(f"JUMP31fold13{suffix}", f"VRT31fold1{suffix}", f"VRT31fold13{suffix}")
        # ss3.add_jump(f"JUMP31fold131{suffix}", f"VRT31fold13{suffix}", f"VRT31fold131{suffix}")
        ss3._create_chain_connection(ss3, ss3, ss_f_nb1="31", ss_f_nb2="1", ss_t_nb1="31", ss_t_nb2="3", R=R, f="rotate_right_multiply",
                                     suffix=suffix)
        ss3._create_final_ref_dofs(ss3, ss3, ss_f_nb1="31", ss_f_nb2="111", ss_t_nb1="31", ss_t_nb2="131", R=R, f="rotate_right_multiply",
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)

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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
                                   suffix=suffix)
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
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'x', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'y', "rotation", 0)
        ss3.add_dof(f"JUMP31fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss3.add_jumpgroup("JUMPGROUP7", f"JUMP31fold111_subunit{suffix}", f"JUMP31fold121_subunit{suffix}",
                          f"JUMP31fold131_subunit{suffix}", f"JUMP32fold111_subunit{suffix}", f"JUMP32fold121_subunit{suffix}",
                          f"JUMP35fold111_subunit{suffix}", f"JUMP35fold131_subunit{suffix}", f"JUMP33fold111_subunit{suffix}",
                          f"JUMP34fold111_subunit{suffix}")

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

    def _right_handed_vectors(self, v1, v2, axis):
        """Returns true if the point v1 going to v2 relative to the axis is right-handed. It is left-handed if the cross product v1 X v2
        points in the same direction as the axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        cross = np.cross(np.array(v1 - axis), np.array(v2 - axis))
        return vector_angle(cross, axis) > 90  # True -> It is right-handed

    # fixme: should handle O and T case
    def is_rightanded(self):
        """Returns true if the point fold3_axis going to fold2_axis relative to the foldHF_axis is right-handed. It is left-handed if the cross product fold3_axis X fold2_axis
        points in the same direction as the foldF_axis and right-handed if it points the opposite way with the cutoff being 180/2 degrees."""
        foldHF_axis = -self.get_vrt("VRTHFfold1").vrt_z
        fold3_axis = -self.get_vrt("VRT3fold1").vrt_z
        fold2_axis = -self.get_vrt("VRT2fold1").vrt_z
        return self._right_handed_vectors(fold3_axis, fold2_axis, foldHF_axis)


    def create_I_2fold_based_symmetry(self, suffix=''):
        """Creates a 2-fold based symmetry file from a HF-based (CURRENTLY ONLY 5-fold) one."""
        ss2 = CubicSetup()
        ss2.reference_symmetric = True
        ss2.symmetry_name = self.symmetry_name + "_2fold_based"
        ss2.anchor = self.anchor
        # 1 subunit
        # 2 fivefolds
        # 1 threefold
        # 2 twofolds
        ss2.energies = f"60*VRT21fold111_z{suffix} + " \
                       f"60*(VRT21fold111_z{suffix}:VRT25fold111_z{suffix}) + 60*(VRT21fold111_z{suffix}:VRT24fold111_z{suffix}) + " \
                       f"60*(VRT21fold111_z{suffix}:VRT22fold121_z{suffix}) + " \
                       f"30*(VRT21fold111_z{suffix}:VRT21fold121_z{suffix}) + 30*(VRT21fold111_z{suffix}:VRT26fold111_z{suffix})"

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
        is_righthanded = self.is_rightanded()
        # if is_righthanded:
        #     R = rotation_matrix(center2, - rot_angle)
        # else:
        #     R = rotation_matrix(center2, rot_angle)
        vrt21fold.rotate(R)
        ss2._create_base_dofs(ss2, ss2, ss_f_nb1="21", ss_t_nb1="21", base_vrt=vrt21fold)
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="", ss_t_nb1="21", ss_t_nb2="1", suffix=suffix)
        ss2._create_final_ref_dofs(self, ss2, ss_f_nb1="HF", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="111", suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="21", ss_t_nb2="121", R=R, suffix=suffix)
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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1="22", ss_t_nb2="111", R=R, suffix=suffix)
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold11{suffix}", f"VRT22fold11{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold111{suffix}").rotate(R, True))
        # ss2.add_vrt(ss2.copy_vrt(f"VRT21fold1111{suffix}", f"VRT22fold1111{suffix}").rotate(R, True))
        # ss2.add_jump(f"JUMP22fold11{suffix}", f"VRT22fold1{suffix}", f"VRT22fold11{suffix}")
        # ss2.add_jump(f"JUMP22fold111{suffix}", f"VRT22fold11{suffix}", f"VRT22fold111{suffix}")
        # ss2.add_jump(f"JUMP22fold1111{suffix}", f"VRT22fold111{suffix}", f"VRT22fold1111{suffix}")
        # ---- chain 2 ----
        ss2._create_chain_connection(ss2, ss2, ss_f_nb1="21", ss_f_nb2="2", ss_t_nb1="22", ss_t_nb2="2", R=R, suffix=suffix)
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="121", ss_t_nb1="22", ss_t_nb2="121", R=R, suffix=suffix)
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
            ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2=f"111", ss_t_nb1=f"2{n}", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="21", ss_f_nb2="111", ss_t_nb1=f"26", ss_t_nb2=f"111", R=R, suffix=suffix)

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
        ss2._create_final_ref_dofs(ss2, ss2, ss_f_nb1="22", ss_f_nb2="111", ss_t_nb1=f"27", ss_t_nb2=f"111", R=R, suffix=suffix)
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
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'x', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'y', "rotation", 0)
        ss2.add_dof(f"JUMP21fold111_subunit{suffix}", 'z', "rotation", 0)
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
        ss2.add_jumpgroup("JUMPGROUP7", f"JUMP21fold111_subunit{suffix}", f"JUMP21fold121_subunit{suffix}",
                          f"JUMP22fold111_subunit{suffix}", f"JUMP22fold121_subunit{suffix}", f"JUMP23fold111_subunit{suffix}",
                          f"JUMP24fold111_subunit{suffix}", f"JUMP25fold111_subunit{suffix}", f"JUMP26fold111_subunit{suffix}",
                          f"JUMP27fold111_subunit{suffix}")

        return ss2
