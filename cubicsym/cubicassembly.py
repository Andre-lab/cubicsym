#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CubicAssembly class
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import numpy as np
import time
import math
from cubicsym.assembly import Assembly
from cubicsym.mathfunctions import rotation_matrix, shortest_path, criteria_check, angle, \
    distance, vector, vector_angle, vector_projection, rotate
from symmetryhandler.symmetryhandler import CoordinateFrame,  SymmetrySetup
import os

class CubicSymmetricAssembly(Assembly):
    """"""
    def __init__(self, id):
        super().__init__(id)

        self.symmetry = None

        self.rmsd_diff = None
        self.lowest_3fold_rmsd = None
        self.highest_3fold_accepted_rmsd = None
        self.lowest_5fold_rmsd = None
        self.highest_5fold_accepted_rmsd = None

        self.angles_diff = None

# public

    def setup_symmetry(self, symmetry_name=None, input_name=None, master_to_use="1", rmsd_diff=0.5, angles_diff=2.0):
        """Sets up a symmetric representation of an assembly for use in Rosetta.

        :param symmetry_name: The name given to the symmetry file (Input to Rosetta)
        :param input_name: The name given to the input file (Input to Rosetta)
        :param master_to_use: The master subunit id to use. Use the ids specified in the assembly.
        :param rmsd_diff: The allowed RMSD difference when creating symmetry. If the assembly is not perfectly symmetric,
        the RMSD difference is > 0.0. Higher RMSD can arise, for example, from the chains being different, for instance different
        conformations of the backbone, or for example, the chains can be positioned differently.
        Examples of structures having those problems are: 4y5z. RMSD is checked when rotating subunits around their
        3-folds and 5-folds and if they are not superimposing with RMSD <= rmsd_diff, then an ToHighRMSD exception will occur.
        :return:
        """

        self.rmsd_diff = rmsd_diff
        self.angles_diff = angles_diff

        if symmetry_name == None:
            name = self.id + ".symm"

        if input_name == None:
            name = self.d + ".symm"

        start_time = time.time()

        # this is important to match proteins with different sequence lengths
        self._create_sequence_alignment_map()

        if self.get_symmetry() ==  "I":
            setup, master = self._setup_icosahedral_symmetry(symmetry_name, input_name, master_to_use)

        print("writing the output pdb structure that is used in Rosetta (" + input_name + ") to disk. Use this one as input for Rosetta!")
        self.output_subunit_as_specified(input_name, master)

        print("writing the symmetry file (" + symmetry_name + ") to disk.")
        setup.output(symmetry_name)

        print("Symmetry set up in: " + str(round(time.time() - start_time, 1)) + "s")

    def get_symmetry(self):
        """

        :return:
        """
        if self.symmetry == None:
            self.center()
            self._detect_cubic_symmetry()
        return self.symmetry

    def show_symmetry(self, apply_dofs=True, mark_jumps=True):
        """

        :param name:
        :return:
        """
        name = "tmp_symmetry.symm"
        tmp = f"/tmp/{name}"
        if self.get_symmetry() == "I":
            setup, _ = self._setup_icosahedral_symmetry(name, "")
        else:
            raise NotImplementedError
        setup.print_visualization(tmp, apply_dofs, mark_jumps)
        self.pymol.run(tmp)
        os.remove(tmp)

    def _setup_icosahedral_symmetry(self, symmetry_name,
                                    input_name,
                                    master_to_use="1",
                                    # main_5_fold_to_use=5,
                                    # other_3_folds_5_folds_to_use=3,
                                    # other_2_folds_5_folds_to_use=3,
                                    #full_symmetry=True
                                    ):
        """

        :param symmetry_name:
        :param input_name:
        :return:
        """

        # Name the master subunit.
        master_id = master_to_use

        # Create symmetry system
        setup = SymmetrySetup(symmetry_name)

        # sets the capsid at the global center
        self.center()

        # extract 2,3,5 fold symmetry units
        master_5_fold = self.find_x_folds(5, id=master_id)[0]
        master_3_fold = self.find_x_folds(3, id=master_id)[0]
        master_2_folds = self.find_x_folds(2, id=master_id, closest=2)
        # the 2 other 5-folds are part of the 3-fold
        second_5_fold = self.find_x_folds(5, master_3_fold[1])[0]
        third_5_fold = self.find_x_folds(5, master_3_fold[2])[0]

        # Define global center and coordinate frame
        global_z = np.array([0, 0, 1])
        global_y = np.array([0, 1, 0])
        global_x = np.array([1, 0, 0])
        global_center = np.array([0, 0, 0])

        # Add a global coordinate frame to the setup
        global_coordinate_frame = CoordinateFrame("VRTglobal", global_x, global_y, global_z, global_center)
        setup.add_vrt(global_coordinate_frame)

        # Rotate the capsid so that it is aligned with the z-axis
        current_z_axis = vector(self.get_geometric_center(master_5_fold), global_center)
        self.rotate_about_axis(np.cross(global_z, current_z_axis), vector_angle(global_z, current_z_axis))

        # Rotate the capsid so that the center of mass (Rosetta style: so it is the CA closest to the real center of mass)
        # of the master subunit is aligned to the global x axis
        master_com = self.get_subunit_CA_closest_to_COM(master_id)
        pz15 = vector_projection(master_com, global_z)
        x_vector = vector(master_com, pz15)
        x_rotation = vector_angle(x_vector, global_x)
        self.rotate_about_axis(np.cross(global_x, x_vector), x_rotation)

        # Because of Rosettas convention with coordinate frames being the other way around (z = -z and x = -x) we have to make
        # a coordinate frame that is reversed in the z and x axis.
        reversed_global_z = - np.array([0, 0, 1])
        reversed_global_x = - np.array([1, 0, 0])

        #### Now we add all the vrts and connect them through VRTglobal

        #### Main 5 fold

        # first make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # VRTglobal->VRT5fold->VRT5fold1 -> this then connects to all of the subunits
        suffix = ""
        name = "VRT5fold"
        jump = "JUMP5fold"
        for i in range(2):
            setup.add_vrt(CoordinateFrame(name + suffix, reversed_global_x, global_y, reversed_global_z, global_center))
            if i == 0:
                setup.add_jump(jump, "VRTglobal", name) # VRTglobal->VRT5fold
            else:
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix) # VRT5fold->VRT5fold1
            suffix += "1"

        main_5_fold_to_use = 5
        to_go_through = [72 * i for i in range(0, main_5_fold_to_use)]
        for i, angle in enumerate(to_go_through, 1):
            rt = rotation_matrix(pz15, angle)
            suffix = "1" + str(i)
            for i in range(3):
                setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt), rotate(global_y, rt),
                                              rotate(reversed_global_z, rt), global_center))
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix)
                suffix += "1"
            setup.add_jump(jump + suffix[:-1] + "_subunit", name + suffix[:-1], "SUBUNIT")

        #### END

        #### figure out which 2-fold is the one that is closest -> this will determine
        # the vectors: z25 and z35 that points to the center of the 2 fivefolds

        twofold_1_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[0][1])
        twofold_2_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[1][1])
        if twofold_1_distance < twofold_2_distance:
            twofold_closest_id = master_2_folds[0][1]
        else:
            twofold_closest_id = master_2_folds[1][1]
        # Now make sure that the 2-fold that is closest and the 3-fold are coming from opposite 5-folds
        # Projected vector to the 5-fold geometric center was calculated above (pz15). Now calculate the non-projected ones to
        # the two 5-folds next to the master one
        # the z25 contains the closest 2-fold, and z35 the 3-fold.
        if twofold_closest_id in second_5_fold:
            z25 = vector(self.get_geometric_center(second_5_fold), global_center)
            z35 = vector(self.get_geometric_center(third_5_fold), global_center)
        else:
            z25 = vector(self.get_geometric_center(third_5_fold), global_center)
            z35 = vector(self.get_geometric_center(second_5_fold), global_center)

        #### The 3-fold 5-fold

        # determine
        axis = z35
        # we have the axis, now we need to rotate the the coordinate system from the global axis to the one relative to the 3-fold 5-fold
        rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))

        # first make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # VRTglobal->VRT3fold->VRT3fold1 -> this then connects to all of the subunits
        suffix = ""
        name = "VRT3fold"
        jump = "JUMP3fold"
        for i in range(2):
            setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt_from_global_z), rotate(global_y, rt_from_global_z),
                                          rotate(reversed_global_z, rt_from_global_z), global_center))
            if i == 0:
                setup.add_jump(jump, "VRTglobal", name) # VRTglobal->VRT3fold
            else:
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix) # VRT3fold->VRT3fold1
            suffix += "1"
        # for i in range(2):
        #     setup.add_vrt(CoordinateFrame(name + suffix, reversed_global_x, global_y, reversed_global_z, global_center))
        #     if i == 0:
        #         setup.add_jump(jump, "VRTglobal", name) # VRTglobal->VRT3fold
        #     else:
        #         setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix) # VRT3fold->VRT3fold1
        #     suffix += "1"

        # if other_3_folds_5_folds_to_use == 1:
        #     start = 180 - 72
        # elif other_3_folds_5_folds_to_use == 3:
        #     start = 180 - 72*2
        # elif other_3_folds_5_folds_to_use == 5:
        #     start = 180 - 72*3
        # else:
        #     # It could still make sense though at 2 or 4 - but it just a non-odd number so which of them should we choose?
        #     # this decision I have not taken yet! so there this error!
        #     raise NotImplementedError
        other_3_folds_5_folds_to_use = 2
        start = 180 - 72

        to_go_through = [start + 72 * i for i in range(0, other_3_folds_5_folds_to_use)]
        for i, angle in enumerate(to_go_through, 1):
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
            if z35[1] < 0:
                angle *= -1
            rt_around_axis = rotation_matrix(axis, angle)  # check angle / positive or negative
            rt = np.dot(rt_from_global_z, rt_around_axis)  # check that this is okay
            suffix = "1" + str(i)
            for i in range(3):
                setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt), rotate(global_y, rt),
                                              rotate(reversed_global_z, rt), global_center))
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix)
                suffix += "1"
            setup.add_jump(jump + suffix[:-1] + "_subunit", name + suffix[:-1], "SUBUNIT")

        #### END

        #### The 2-fold 5-fold

        axis = z25
        # rotate the the coordinate system from the global axis to the one relative to the 2-fold 5-fold
        rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))

        # first make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # VRTglobal->VRT2fold->VRT2fold1 -> this then connects to all of the subunits
        suffix = ""
        name = "VRT2fold"
        jump = "JUMP2fold"
        for i in range(2):
            setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt_from_global_z),
                                          rotate(global_y, rt_from_global_z),
                                          rotate(reversed_global_z, rt_from_global_z), global_center))
            if i == 0:
                setup.add_jump(jump, "VRTglobal", name) # VRTglobal->VRT2fold
            else:
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix) # VRT2fold->VRT2fold1
            suffix += "1"

        # if other_2_folds_5_folds_to_use == 1:
        #     start = 180
        # elif other_2_folds_5_folds_to_use == 3:
        #     start = 180 - 72
        # elif other_2_folds_5_folds_to_use == 5:
        #     start = 180 - 72*2
        # else:
        #     # It could still make sense though at 2 or 4 - but it just a non-odd number so which of them should we choose?
        #     # this decision I have not taken yet! so there this error!
        #     raise NotImplementedError
        other_2_folds_5_folds_to_use = 2
        start = 180

        to_go_through = [start + 72 * i for i in range(0, other_2_folds_5_folds_to_use)]
        for i, angle in enumerate(to_go_through, 1):
            rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
            if z35[1] < 0:
                angle *= -1
            rt_around_axis = rotation_matrix(axis, angle)  # check angle / positive or negative
            rt = np.dot(rt_from_global_z, rt_around_axis)  # check that this is okay
            suffix = "1" + str(i)
            for i in range(3):
                setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt), rotate(global_y, rt),
                                              rotate(reversed_global_z, rt), global_center))
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix)
                suffix += "1"
            setup.add_jump(jump + suffix[:-1] + "_subunit", name + suffix[:-1], "SUBUNIT")



        # # flag if we want an extra 5fold or not in the z35 fold!
        # extra_2_fold = True
        # if extra_2_fold:
        #     axiss, angles, jumps, names = (z35, z25, z35), (180 - 72, 180, 180), ("JUMP3fold", "JUMP2fold", "JUMP2extra"),("VRT3fold", "VRT2fold", "VRT2extra")
        # else:
        #     axiss, angles, jumps, names = (z35, z25), (180 - 72, 180), ("JUMP3fold", "JUMP2fold"), ("VRT3fold", "VRT2fold")
        #
        # # fold35 & fold25 | adjacent subunit to master (1 in the 3-fold rotation axis and 1 in the 2 fold axis)
        # for axis, angle, jump, name in zip(axiss, angles, jumps, names):
        #     rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
        #     # so before coming here we have the master subunit along the positive x-axis.
        #     # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
        #     # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
        #     if z35[1] < 0:
        #         angle *= -1
        #     rt_around_axis = rotation_matrix(axis, angle)  # check angle / positive or negative
        #     rt = np.dot(rt_from_global_z, rt_around_axis)  # check that this is okay
        #     suffix = ""
        #     for i in range(5):
        #         setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt), rotate(global_y, rt),
        #                                       rotate(reversed_global_z, rt), global_center))
        #         if i == 0:
        #             setup.add_jump(jump, "VRTglobal", name)
        #         else:
        #             setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix)
        #         suffix += "1"
        #     setup.add_jump(jump + suffix[:-1] + "_subunit", name + suffix[:-1], "SUBUNIT")

        # The 6 degrees of freedom
        setup.add_dof("JUMP5fold1", 'z', "translation", np.linalg.norm(pz15))
        setup.add_dof("JUMP5fold1", 'z', "rotation", 0)
        setup.add_dof("JUMP5fold111", 'x', "translation", np.linalg.norm(vector(master_com, pz15)))
        setup.add_dof("JUMP5fold1111", 'x', "rotation", 0)
        setup.add_dof("JUMP5fold1111", 'y', "rotation", 0)
        setup.add_dof("JUMP5fold1111", 'z', "rotation", 0)
        setup.add_dof("JUMP5fold1111_subunit", 'x', "rotation", 0)
        setup.add_dof("JUMP5fold1111_subunit", 'y', "rotation", 0)
        setup.add_dof("JUMP5fold1111_subunit", 'z', "rotation", 0)

        # set jumpgroups
        # if extra_2_fold:
        #     setup.add_jumpgroup("JUMPGROUP1", "JUMP5fold1", "JUMP3fold1", "JUMP2fold1", "JUMP2extra1")
        #     setup.add_jumpgroup("JUMPGROUP2", "JUMP5fold111", "JUMP5fold121", "JUMP5fold131", "JUMP3fold111",
        #                         "JUMP2fold111", "JUMP2extra111")
        #     setup.add_jumpgroup("JUMPGROUP3", "JUMP5fold1111", "JUMP5fold1211", "JUMP5fold1311", "JUMP3fold1111",
        #                         "JUMP2fold1111","JUMP2extra1111")
        #     setup.add_jumpgroup("JUMPGROUP4", "JUMP5fold1111_subunit", "JUMP5fold1211_subunit", "JUMP5fold1311_subunit",
        #                         "JUMP3fold1111_subunit", "JUMP2fold1111_subunit", "JUMP2extra1111_subunit")
        # else:
        setup.add_jumpgroup("JUMPGROUP1", "JUMP5fold1", "JUMP3fold1", "JUMP2fold1")

        setup.add_jumpgroup("JUMPGROUP2", *[f"JUMP5fold1{i}1" for i in range(1, main_5_fold_to_use + 1)],
                            *[f"JUMP3fold1{i}1" for i in range(1, other_3_folds_5_folds_to_use + 1)],
                            *[f"JUMP2fold1{i}1" for i in range(1, other_2_folds_5_folds_to_use + 1)])

        setup.add_jumpgroup("JUMPGROUP3", *[f"JUMP5fold1{i}11" for i in range(1, main_5_fold_to_use + 1)],
                            *[f"JUMP3fold1{i}11" for i in range(1, other_3_folds_5_folds_to_use + 1)],
                            *[f"JUMP2fold1{i}11" for i in range(1, other_2_folds_5_folds_to_use + 1)])

        setup.add_jumpgroup("JUMPGROUP4", *[f"JUMP5fold1{i}11_subunit" for i in range(1, main_5_fold_to_use + 1)],
                            *[f"JUMP3fold1{i}11_subunit" for i in range(1, other_3_folds_5_folds_to_use + 1)],
                            *[f"JUMP2fold1{i}11_subunit" for i in range(1, other_2_folds_5_folds_to_use + 1)])

        # set energies
        # if extra_2_fold:
        #     setup.energies = "60*VRT5fold1111 + 60*(VRT5fold1111:VRT5fold1211) + 60*(VRT5fold1111:VRT5fold1311) + 60*(VRT5fold1111:VRT3fold1111) + 30*(VRT5fold1111:VRT2fold1111) + 30*(VRT5fold1111:VRT2extra1111)"
        # else:
        # todo: think if should be _subunit
        # old:
        # setup.energies = "60*VRT5fold1111 + 60*(VRT5fold1111:VRT5fold1211) + 60*(VRT5fold1111:VRT5fold1311) + 60*(VRT5fold1111:VRT3fold1111) + 30*(VRT5fold1111:VRT2fold1111)"
        setup.energies = f"60*VRT5fold1111 + " \
                         f"60*(VRT5fold1111:VRT5fold1211) + " \
                         f"60*(VRT5fold1111:VRT5fold1311) + " \
                         f"60*(VRT5fold1111:VRT3fold1{1}11) + " \
                         f"30*(VRT5fold1111:VRT2fold1{1}11) + " \
                         f"30*(VRT5fold1111:VRT3fold1{2}11)"
        # f"30*(VRT5fold1111:VRT2fold1{1 if other_2_folds_5_folds_to_use == 1 else 2 if other_2_folds_5_folds_to_use == 3 else 3}11)"
                         # f"60*(VRT5fold1111:VRT3fold1{1 if other_3_folds_5_folds_to_use == 1 else 2 if other_3_folds_5_folds_to_use == 3 else 3}11) + " \
                         # f"30*(VRT5fold1111:VRT2fold1{1 if other_2_folds_5_folds_to_use == 1 else 2 if other_2_folds_5_folds_to_use == 3 else 3}11)"

        # if full_symmetry:
        #     if other_2_folds_5_folds_to_use == 3:
        #         setup.energies += " + 30*(VRT5fold1111:VRT2fold1111) + "
        #         # setup.energies += "30*(VRT5fold1111:VRT2fold1131) + " - the extra 3-fold. should not include it
        #     else:
        #         raise NotImplementedError
        #     if other_3_folds_5_folds_to_use == 3:
        #         setup.energies += "30*(VRT5fold1111:VRT3fold1111) + "
        #         setup.energies += "30*(VRT5fold1111:VRT3fold1311)"
        #     else:
        #         raise NotImplementedError

        # todo: there seems be a difference between rosetta COM and my COM so in the case of 6u20 for instance - you'll not get the correct symmetry if recenter is not turned of
        # anchor residues
        setup.anchor = "COM"
        # # checks that is should be recentered
        # setup.recenter = True

        adjust = vector(global_center, self.get_subunit_CA_closest_to_COM(master_id))
        master = self.get_subunit_with_id(master_id).copy()
        master.transform(np.identity(3), adjust)

        return setup, master

    def find_x_folds(self,x, id="1", closest=1):
        """
        Wrapper function. Returns the closet x-fold subunits to subunit with the id.
        The underlying functions can be written generally in the future with a recursive functions.

        Only 2-5 fold symmetries are supported!

        :param x:
        :param id:
        :param closest:
        :return:
        """

        # only 2-5 are supported
        assert(x!=1 and x < 6)

        if x == 2:
            return self._find_2_fold_subunits(id, closest)

        if x == 3:
            return self._find_3_fold_subunits(id, closest)

        if x == 4:
            pass
            #return find_3_fold_subunits(id, closest)

        if x == 5:
            return self._find_5_fold_subunits(id, closest)

# private

    def _detect_cubic_symmetry(self):
        """Detects symmetry."""

        # if a 5-fold symmetry axis is present - then it is icosahedral
        if len(self.find_x_folds(5)) > 0:
            self.symmetry = "I"
            print("I symmetry detected.")
        elif len(self.find_x_folds(4)) > 0:
            self.symmetry = "O"
            print("O symmetry detected.")
        elif len(self.find_x_folds(3)) > 0:
            self.symmetry = "T"
            print("T symmetry detected.")
        else:
            raise Exception("No symmetry found.")

    def _find_2_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the 2 fold axis that the subunit with the id "name" is in.

         Algorithm:
         Checks that the 180 degrees rotation around the center of the 2 subunits overlaps them. If so you have
         a two_fold!

        :param id:
        :param closest:
        :return:
        """

        # set the alginment geometric center
        self._set_alignment_geometric_center()

        # find subunits to search
        subunits_to_search = [int(subunit.id) for subunit in self.get_models() if subunit.id != id]

        # container for the x folds
        twofolds = []

        # special case if x = 2:
        if not "center" in self.xtra:
            self._set_center()
        subunit = self[id]
        for c in subunits_to_search:
            two_fold = True
            subunit_reference = self[f"{c}"]
            center_between_subunits = ((subunit_reference.xtra["alignment_geometric_center"] - subunit.xtra["alignment_geometric_center"]) / 2) + subunit.xtra["alignment_geometric_center"]
            center_to_center_between_subunits = center_between_subunits - self.xtra["center"]
            rt = rotation_matrix(center_to_center_between_subunits, 180)

            # extract the largest overlapping sequence
            subunit_seq = [res.get_resname() for res in subunit.get_residues()]
            subunit_reference_seq = [res.get_resname() for res in subunit_reference.get_residues()]
            sm = difflib.SequenceMatcher(None, subunit_seq, subunit_reference_seq)
            pos_a, pos_b, size = sm.find_longest_match(0, len(subunit_seq), 0, len(subunit_reference_seq))
            subunit_res = [res for res in list(subunit.get_residues())[pos_a:pos_a+size]]
            subunit_reference_res = [res for res in list(subunit_reference.get_residues())[pos_b:pos_b+size]]
            # do a transform and check if they overlap
            n = 0
            distances = np.zeros(3, dtype=np.float64)
            for res, res_reference in zip(subunit_res, subunit_reference_res):
                for atom, atom_reference in zip(res, res_reference):
                    if atom.get_fullname() == "CA":
                        atom_reference = atom_reference.copy()
                        atom_reference.transform(rt, [0,0,0])
                        # if not criteria_check(0, distance(atom.get_coord(),atom_reference.get_coord()), diff=1):
                        distances += distance(atom.coord, atom_reference.coord)
                        n += 1
            rmsd = sum(distances) / n
            # largest RMSD value to be used is.
            if criteria_check(0, rmsd, self.angles_diff):
                twofolds.append([subunit, subunit_reference])

        if len(twofolds) > 0:
            twofolds = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in twofolds})
            twofolds.sort(key=lambda subunits: shortest_path(*[subunit.xtra["alignment_geometric_center"] for subunit in subunits]))
            return [[subunit.id for subunit in subunits] for subunits in twofolds[:closest]]
        else:
            raise Exception("""A 2-fold axis was not found. Did you specify all models appropriately? The criteria for accepted angles might be too low.""")

    def _find_3_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the three fold axis that the subunit with the id "name" is in.

        Algorithm:
        Similar to the algorithm used to find the 5-fold symmetry units (see above). It instead use the degrees of
        a triangle instead of a pentamer.

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 3-fold symmetry subunits the subunit belong to.
        :return SymmetricAssembly: a symmetric protein assembly that consist of 3 subunits with C3 symmetry.
        """

        self._set_alignment_geometric_center()

        threefold = []

        # the subunit with the given id (retrieve p1)
        p1 = None
        for subunit in self.get_models():
            if subunit.id == id:
                p1 = subunit
                break
        if p1 == None:
            print(id, "does not exist!")
            return []

        # list of subunits excluding subunit with the id given
        # todo: should be fast if just references
        subunits_to_search = [subunit for subunit in self.get_models() if subunit.id != id]

        # Find the other 2 protein_list.txt making up the three-fold symmetry axis
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p2 == p3:
                    continue

                # p1,p2,p3 angle should be equal to 108 degrees
                if not criteria_check(60, angle(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"],
                      p3.xtra["alignment_geometric_center"]) , diff=self.angles_diff):
                    continue

                # the distance between p1 and p2 should be the same as p2 and p3
                if not criteria_check(distance(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"]),
                                      distance(p2.xtra["alignment_geometric_center"], p3.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                    continue

                threefold.append((p1, p2, p3))


        if len(threefold) != 0:
            threefold = list({tuple(sorted(subunits,key=lambda subunit: subunit.id)) for subunits in threefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            threefold.sort(key=lambda subunits: shortest_path(subunits[0].xtra["alignment_geometric_center"],
                                                             subunits[1].xtra["alignment_geometric_center"],
                                                             subunits[2].xtra["alignment_geometric_center"]))

            threefolds = []
            for subunits in threefold:
                if self.is_correct_x_fold(3, *subunits):
                    threefolds.append([subunit.id for subunit in subunits])
                if len(threefolds) == closest:
                    return threefolds
            raise ToHighRMSD("None of the 3-folds found are valid 3-folds.")
        else:
            raise ToHighGeometry("A 3-fold axis was not found")


    def is_correct_x_fold(self, fold, *args):
        """Checks if the fold is a correct 3 or 5 fold.

        It does this by superimposing 1 subunit on top of the other using 3 or 5 fold rotations around the folds
        geometric center of mass. A RMSD criteria is checked to conclude if the superposition is good enough to
        validate it as a 3 or 5 fold.
        """
        accepted_rmsds = []
        # create rotation matrices for 3 or 5 fold rotations
        center = self.get_geometric_center([x.id for x in args])
        if fold == 3:
            angles = [120, 120 * 2]
            # angles += [- angle for angle in angles]
        elif fold == 5:
            angles = [72, 72*2, 72*3, 72*4]
            # angles += [- angle for angle in angles]
        else:
            raise NotImplementedError("only works for a 3 and 5 fold as of yet")
        rots = []
        for angle in angles:
            rots.append(rotation_matrix(center, angle))
        s1 = args[0]
        s1_residues = list(s1.get_residues())
        sxs = args[1:] + args[1:]
        correct_fold = []
        # check if the subunits come on top of eachother when using the rotation
        for rotation, angle in zip(rots, angles):
            for sx in sxs:
                distances = 0
                n = 0
                # Use the alignment residues to iterate through here:
                sx_residues = list(sx.get_residues())
                for s1_pos, sx_pos in zip(s1.xtra["alignment"], sx.xtra["alignment"]):
                    s1_resi = s1_residues[s1_pos]
                    sx_resi = sx_residues[sx_pos]
                    assert sx_resi.get_resname() == s1_resi.get_resname(), "RMSD calculation cannot be between different amino acids!"
                    for atom, atom_reference in zip(s1_resi.get_atoms(), sx_resi.get_atoms()):
                        if atom.get_fullname() == "CA":
                            atom_new = atom_reference.copy()
                            atom_new.transform(rotation, [0, 0, 0])
                            # if not criteria_check(0, distance(atom.get_coord(),atom_reference.get_coord()), diff=1):
                            distances += distance(atom.coord, atom_new.coord)
                            n += 1
                # debug
                # print(f"pseudoatom atom, pos={list(atom.coord)}")
                # print(f"pseudoatom atom_reference, pos={list(atom_reference.coord)}")
                # print(f"pseudoatom atom_new, pos={list(atom_new.coord)}")
                # print(f"sx: {sx}")
                # print(f"rot: {rotation}")
                # print(f"rot: {angle}")
                # debug
                rmsd = math.sqrt(distances / n)
                if fold == 3:
                    if self.lowest_3fold_rmsd:
                        self.lowest_3fold_rmsd = min(self.lowest_3fold_rmsd, rmsd)
                    else:
                        self.lowest_3fold_rmsd = rmsd
                elif fold == 5:
                    if self.lowest_5fold_rmsd:
                        self.lowest_5fold_rmsd = min(self.lowest_5fold_rmsd, rmsd)
                    else:
                        self.lowest_5fold_rmsd = rmsd
                else:
                    raise NotImplementedError

                # largest RMSD value to be used is.
                if criteria_check(0, rmsd, diff=self.rmsd_diff):
                    correct_fold.append(True)
                    accepted_rmsds.append(rmsd)
                else:
                    correct_fold.append(False)

            if fold == 3:
                if sum(correct_fold) >= 2:
                    if self.highest_3fold_accepted_rmsd:
                        self.highest_3fold_accepted_rmsd = max(self.highest_3fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_3fold_accepted_rmsd = max(accepted_rmsds)
                    return True
            elif fold == 5:
                if sum(correct_fold) >= 5:
                    if self.highest_5fold_accepted_rmsd:
                        self.highest_5fold_accepted_rmsd = max(self.highest_5fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_5fold_accepted_rmsd = max(accepted_rmsds)
                    return True
        return False


    def _find_5_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the five fold axis that the subunit with the name "name" is in.

        Algorithm:
        It does so by first finding all the geometric centers of each subunit in the assembly and then sorting them
        according to the distance from the specified protein. Then it will find the closest two subunits having a total
        angle of 108 degrees between them. This corresponds to the first angle in a pentagon (5-fold-symmetry).
        Then it will find then one having 72 between the start subunit and the last subunit in the previous found subunit
        and the one having 36 degrees between the first and the next found subunit, corresponding to the other
        two angles in a pentagon. It exhaustively searches all found pentagons and determines the pentagon with the shortest
        path between the 5 points to be the fivefold axis that is returned.

        For future reference for a better approach: see https://www.sciencedirect.com/science/article/pii/S1047847718301126

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 5-fold symmetry subunits the subunit belong to.
        :param int closest: the closest 5-fold subunits to return
        :return SymmetricAssembly: a symmetric protein assembly that consist of 5 subunits with C5 symmetry.

        """
        self._set_alignment_geometric_center()

        fivefold = []

        # the subunit with the given id (retrieve p1)
        p1 = None
        for subunit in self.get_models():
            if subunit.id == id:
                p1 = subunit
                break
        if p1 == None:
            print(id, "does not exist!")
            return []

        # list of subunits excluding subunit with the id given
        subunits_to_search = [subunit for subunit in self.get_models() if
                              subunit.id != id]  # todo: should be fast if just references

        # Find the other 4 protein_list.txt making up the five fold symmetry axis (p2, p3, p4 and p5)
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p3.id == p2.id:
                    continue

                # p1,p2,p3 angle should be equal to 108 degrees
                if not criteria_check(108.0, angle(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"],
                                                   p3.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                    continue

                # the distance between p1 and p2 should be the same as p2 and p3
                if not criteria_check(distance(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"]),
                                      distance(p2.xtra["alignment_geometric_center"], p3.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                    continue

                for p4 in subunits_to_search:
                    if p4 == p2 or p4 == p3:
                        continue

                    # p1,p3,p4 angle should be equal to 72
                    if not criteria_check(72.0, angle(p1.xtra["alignment_geometric_center"], p3.xtra["alignment_geometric_center"],
                                                      p4.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                        continue

                    # the distance between p2 and p3 should be the same as p3 and p4
                    if not criteria_check(distance(p2.xtra["alignment_geometric_center"], p3.xtra["alignment_geometric_center"]),
                                          distance(p3.xtra["alignment_geometric_center"], p4.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                        continue

                    for p5 in subunits_to_search:
                        if p5 == p2 or p5 == p3 or p5 == p4:
                            continue

                        # p1,p4,p5 angle should be equal to 36
                        if not criteria_check(36.0, angle(p1.xtra["alignment_geometric_center"], p4.xtra["alignment_geometric_center"],
                                                          p5.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                            continue

                        # the distance between p3 and p4 should be the same as p4 and p5
                        if not criteria_check(distance(p3.xtra["alignment_geometric_center"], p4.xtra["alignment_geometric_center"]),
                                              distance(p4.xtra["alignment_geometric_center"], p5.xtra["alignment_geometric_center"]), diff=self.angles_diff):
                            continue

                        fivefold.append((p1, p2, p3, p4, p5))


        if len(fivefold) != 0:
            fivefold = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in fivefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            fivefold.sort(key=lambda subunits: shortest_path(subunits[0].xtra["alignment_geometric_center"],
                                                             subunits[1].xtra["alignment_geometric_center"],
                                                             subunits[2].xtra["alignment_geometric_center"],
                                                             subunits[3].xtra["alignment_geometric_center"],
                                                             subunits[4].xtra["alignment_geometric_center"]))
            fivefolds = []
            for subunits in fivefold:
                if self.is_correct_x_fold(5, *subunits):
                    fivefolds.append([subunit.id for subunit in subunits])
                if len(fivefolds) == closest:
                    return fivefolds
            raise ToHighRMSD("None of the 5-folds found are valid 5-folds.")
        else:
            raise ToHighGeometry("A 5-fold axis was not found.")
