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
import difflib
import os
from cubicsym.assembly import Assembly
from cubicsym.mathfunctions import rotation_matrix, shortest_path, criteria_check, angle, \
    distance, vector, vector_angle, vector_projection, rotate
from symmetryhandler.symmetryhandler import CoordinateFrame,  SymmetrySetup
from cubicsym.exceptions import ToHighGeometry, ToHighRMSD, ValueToHigh, NoSymmetryDetected

class CubicSymmetricAssembly(Assembly):
    """Cubic symmetrical assembly of either I, O or T symmetry. Has additional methods from a regular assembly related to symmetry."""

    def __init__(self, file, symmetry: str, rmsd_diff=0.5, angles_diff=2.0, id_="1"):
        """Initialization of the class.

        :param file: mmCIF file to create the assembly from.
        :param symmetry: Symmetry of the assembly. Either I, O, T or the exact assembly id. If I, O or T is given, it
        will construct the assembly from the first assembly matching the requested symmetry type
        :param rmsd_diff: The allowed RMSD difference when creating symmetry. If the assembly is not perfectly symmetric,
        the RMSD difference is > 0.0. Higher RMSD can arise, for example, from the chains being different, for instance different
        conformations of the backbone, or for example, the chains can be positioned differently.
        Examples of structures having those problems are: 4y5z. RMSD is checked when rotating subunits around their
        3-folds and 5-folds and if they are not superimposing with RMSD <= rmsd_diff, then an ToHighRMSD exception will occur.
        :param angles_diff: The allowed angle difference when creating symmetry. Same argument as for rmsd_diff. If angle <= angles_diff,
        a To
        :param id:
        """
        assert symmetry in ("I", "O", "T") or symmetry.isdigit(), \
            "symmetry definition is wrong. Has to be either I, O, T or a number exclusively."
        super().__init__(file, symmetry if symmetry.isdigit() else "1", id_)
        self.rmsd_diff = rmsd_diff
        self.angles_diff = angles_diff
        self.lowest_5fold_rmsd = None
        self.highest_5fold_accepted_rmsd = None
        # FIXME: MAKE SURE THESE ARE USED (4folds)
        self.lowest_4fold_rmsd = None
        self.highest_4fold_accepted_rmsd = None
        self.lowest_3fold_rmsd = None
        self.highest_3fold_accepted_rmsd = None
        self.find_symmetry(file, symmetry)

    def find_symmetry(self, file, symmetry):
        """Attempts to retrive the symmetry in either of 2 ways:
            1. If symmetry is a number, will retrieve whatever symmetry that assembly id has.
            2. If symmetry is either I, O or I, will iterate through the available assemblies until the corresponding symmetry is found.
        If No symmetry is found or the symmetry requested as I, O or I is not found it will return an exception."""
        self.symmetry = self._detect_cubic_symmetry()
        # if I, O or T has been parsed then attempt to retrieve from other assemblies if the current one does not match the
        # requested symmetry
        additional_symms = []
        if not symmetry.isdigit():
            n = 1
            while self.symmetry != symmetry and n != len(self.allowed_ids):
                # try another id and check if the symmetry is correct
                self.assembly_id = self.allowed_ids[n]
                self.from_mmcif(file)
                n += 1
                self.symmetry = self._detect_cubic_symmetry()
                if self.symmetry == symmetry:
                    break
                elif self.symmetry:
                    additional_symms.append(self.symmetry)
        if not self.symmetry:
            raise NoSymmetryDetected
        elif additional_symms:
            raise Exception(f"{file} does not contain {','.join(additional_symms)} but does contain {self.symmetry}")
        else:
            print(f"Correct symmetry: {self.symmetry} found from assembly id: {self.assembly_id}")

    def output_rosetta_symmetry(self, symmetry_name=None, input_name=None, master_to_use="1", outformat="cif"):
        """Sets up a symmetric representation of a cubic assembly for use in Rosetta.

        :param symmetry_name: The name given to the symmetry file (Input to Rosetta)
        :param input_name: The name given to the input file (Input to Rosetta)
        :param master_to_use: The master subunit id to use. Use the ids specified in the assembly.
        :param outformat: output format for the input file.
        :return:
        """
        if symmetry_name == None:
            symmetry_name = self.id + ".symm"
        if input_name == None:
            input_name = self.id + ".symm"
        start_time = time.time()
        # TODO: delete below line because it is calculated earlier!!!
        # self._create_sequence_alignment_map() # this is important to match proteins with different sequence lengths
        setup, master = self.setup_symmetry(symmetry_name, master_to_use)
        # if self.get_symmetry() == "I":
        #     setup, master = self._setup_I_symmetry(symmetry_name, master_to_use)
        # elif self.get_symmetry() == "O":
        #     setup, master = self._setup_O_symmetry(symmetry_name, master_to_use)
        # elif self.get_symmetry() == "T":
        #     raise NotImplementedError
        print("Writing the output pdb structure that is used in Rosetta (" + input_name + ") to disk. Use this one as input for Rosetta!")
        self.output_subunit_as_specified(input_name, master, format=outformat)
        print("Writing the symmetry file (" + symmetry_name + ") to disk.")
        setup.output(symmetry_name)
        print("Symmetry set up in: " + str(round(time.time() - start_time, 1)) + "s")

    def get_symmetry(self):
        """Retrieves the symmetry of the cubic assembly. If it hasn't been calculated then it is calculated."""
        if self.symmetry == None:
            self.center()
            self.symmetry = self._detect_cubic_symmetry()
        return self.symmetry

    def show_symmetry(self, apply_dofs=True, mark_jumps=True):
        """TODO"""
        name = "tmp_symmetry.symm"
        tmp = f"/tmp/{name}"
        if self.get_symmetry() == "I":
            setup, _ = self._setup_I_symmetry(name, "1")
        elif self.get_symmetry() == "O":
            setup, master = self._setup_O_symmetry(name, "1")
        setup.print_visualization(tmp, apply_dofs, mark_jumps)
        self.cmd.run(self.server_root_path + tmp)
        os.remove(tmp)

    def _define_global_center(self):
        """Defines the global center and its coordinate frame."""
        global_z = np.array([0, 0, 1])
        global_y = np.array([0, 1, 0])
        global_x = np.array([1, 0, 0])
        global_center = np.array([0, 0, 0])
        return global_x, global_y, global_z, global_center

    def _get_rosetta_chain_ordering(self):
        """Returns the chain ordering Rosetta used as an iterable including extra numbers. This is the chain ordering Rosetta
        applies according to src.basic.pymol_chains.hh. I have added extra 10000 numbers to increase the iter so
        as to not run out of chain labels."""
        rosetta_chain_ordering = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz!@#$&.<>?]|-_\\~=%"
        extra_chains = "".join(map(str, range(0, 10000)))
        return iter(rosetta_chain_ordering + extra_chains)

    # def setup_symmetry(self, symmetry_name, master_id="1"):
    #     self.setup_symmetry_subroutine(setup, master_id)
    #     # if self.get_symmetry() == "I":
    #     # elif self.get_symmetry() == "O":
    #     #     self.setup_symmetry_subroutine(setup, 4, master_id)
    #     # else: # T
    #     #     pass

    def get_highest_fold(self):
        """Get the highest symmetrical fold for the symmetry."""
        if self.get_symmetry() == "I":
            return 5
        elif self.get_symmetry() == "O":
            return 4
        else: # self.get_symmetry() == "T":
            return 3

    def get_closest_and_furthest_2_fold(self, master_id, master_2_folds):
        """Mark which of the 2-folds are closest and furthest away from the master subunit."""
        twofold_1_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[0][1])
        twofold_2_distance = self.find_minimum_atom_type_distance_between_subunits(master_id, master_2_folds[1][1])
        if twofold_1_distance < twofold_2_distance:
            twofold_closest_id = master_2_folds[0][1]
            twofold_furthest_id = master_2_folds[1][1]
        else:
            twofold_closest_id = master_2_folds[1][1]
            twofold_furthest_id = master_2_folds[0][1]
        return twofold_closest_id, twofold_furthest_id

    def remap_chain_labels(self, master_high_fold, master_3_fold, closest_2fold_id, furthest_2fold_id, second_high_fold, third_high_fold):
        """Creates an ordering that maps the order of ids to the chain naming order that will occur when Rosetta names the chains
        after applying symmetry."""
        # see map: {subunit.id: chain.id for subunit in self.get_subunits() for chain in subunit.get_chains()}
        ids_ordering = master_high_fold[:]
        # which high fold is in the two_fold_furthest
        if furthest_2fold_id in second_high_fold:
            for id_ in second_high_fold:
                if id_ in master_3_fold:
                    ids_ordering += [id_, furthest_2fold_id]
                    break
        else:
            for id_ in third_high_fold:
                if id_ in master_3_fold:
                    ids_ordering += [id_, furthest_2fold_id]
                    break
        ids_ordering += [closest_2fold_id]
        # get the 3 fold that is not already in the ids_ordering
        ids_ordering += [id_ for id_ in master_3_fold if id_ not in ids_ordering]
        chain_ordering = self._get_rosetta_chain_ordering()
        for id_ in ids_ordering:
            for subunit in self.get_subunits(ids_ordering):
                if subunit.id == id_:
                    for chain in subunit.get_chains():
                        chain.id = next(chain_ordering)
                    break
        # now the rest of the chains have to be named according to the rest of the chain labels
        for subunit in [subunit for subunit in self.get_subunits() if subunit.id not in ids_ordering]:
            for chain in subunit.get_chains():
                chain.id = next(chain_ordering)

    def setup_symmetry(self, symmetry_name, master_id="1", subunits_in_other_highfold=2):
        """Sets up a SymmetrySetup object for either Icosahedral (I), Octahedral (O) or Tetrahedral (T) symmetry.

        The symmetrical setup consists of the 1 master subunit and its interacting subunits from the 3 closest highest possible folds.
        'high' is a substitute for the highest symmetrical fold here and in the code. So the following is given: I: high=5, O: high=4 and T: high=3.
        For I symmetry, for instance, the symmetrical setup will consist of the 3 closets 5-folds. The master subunit will be a part of 1 of
        these 5-folds and all chains present in the 5-fold will be present in the symmetrical setup. A variable number of subunits will
        be present from the the 2 other 5-folds which can be set by the 'subunits_in_other_highfold' variable which by default is 2.

        :param symmetry_name: Name given to the SymmetrySeutp object
        :param master_id: subunit id to use as the master subunit
        :param subunits_in_other_highfold: Subunits to use in the other high-folds
        :return: SymmetrySetup, Bio.PDB.Structure.Structure object corresponding to the chosen master subunit.
        """
        setup = SymmetrySetup(symmetry_name)
        self.center()
        highest_fold = self.get_highest_fold()
        highest_angle = int(360 / highest_fold)
        master_high_fold = self.find_X_folds(highest_fold, id=master_id)[0]
        if highest_fold == 3:
            master_3_fold = self.find_X_folds(3, id=master_id, closest=2)[1]
        else:
            master_3_fold = self.find_X_folds(3, id=master_id)[0]
        # the 2 other highest-folds that aer part of the 3-fold
        second_high_fold = self.find_X_folds(highest_fold, master_3_fold[1])[0]
        third_high_fold = self.find_X_folds(highest_fold, master_3_fold[2])[0]
        # get the 2 closest 2-folds that are part of the other second or third highest fold
        # closest=10 is arbitary, but we want to make sure we can go through enough
        master_2_folds = [id2 for id2 in self.find_X_folds(2, id=master_id, closest=10) if id2[1] in second_high_fold + third_high_fold]
        assert len(master_2_folds) == 2, "retrieved too many 2-folds!"
        # get the coordinates for the global center
        global_x, global_y, global_z, global_center = self._define_global_center()
        # Add a global coordinate frame to the setup
        global_coordinate_frame = CoordinateFrame("VRTglobal", global_x, global_y, global_z, global_center)
        setup.add_vrt(global_coordinate_frame)
        # Rotate the assembly so that it is aligned with the z-axis
        current_z_axis = vector(self.get_geometric_center(master_high_fold), global_center)
        self.rotate_about_axis(np.cross(global_z, current_z_axis), vector_angle(global_z, current_z_axis))
        # Rotate the assembly so that the center of mass (Rosetta style: so it is the CA closest to the real center of mass)
        # of the master subunit is aligned to the global x axis
        master_com = self.get_subunit_CA_closest_to_GC(master_id)
        pz1high = vector_projection(master_com, global_z)
        x_vector = vector(master_com, pz1high)
        x_rotation = vector_angle(x_vector, global_x)
        self.rotate_about_axis(np.cross(global_x, x_vector), x_rotation)
        # Because of Rosettas convention with coordinate frames being the other way around (z = -z and x = -x) we have to make
        # a coordinate frame that is reversed in the z and x axis.
        reversed_global_z = - np.array([0, 0, 1])
        reversed_global_x = - np.array([1, 0, 0])

        # Now we add all the vrts and connect them through VRTglobal
        # ---- Main highest fold ----
        # First make the following (not sure why this is needed to be done this way, but when i did this (1 year ago) there prob. was some reason...
        # for highest fold == 5: VRTglobal->VRT5fold->VRT5fold1 -> this then connects to all of the subunits
        suffix = ""
        name = f"VRTHFfold"
        jump = f"JUMPHFfold"
        for i in range(2):
            setup.add_vrt(CoordinateFrame(name + suffix, reversed_global_x, global_y, reversed_global_z, global_center))
            if i == 0:
                setup.add_jump(jump, "VRTglobal", name) # VRTglobal->VRT5fold
            else:
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix) # VRT5fold->VRT5fold1
            suffix += "1"
        to_go_through = [highest_angle * i for i in range(0, highest_fold)]
        for i, angle in enumerate(to_go_through, 1):
            rt = rotation_matrix(pz1high, angle)
            suffix = "1" + str(i)
            for i in range(3):
                setup.add_vrt(CoordinateFrame(name + suffix, rotate(reversed_global_x, rt), rotate(global_y, rt),
                                              rotate(reversed_global_z, rt), global_center))
                setup.add_jump(jump + suffix, name + suffix[:-1], name + suffix)
                suffix += "1"
            setup.add_jump(jump + suffix[:-1] + "_subunit", name + suffix[:-1], "SUBUNIT")
        # ---------------------

        closest_2fold_id,furthest_2fold_id = self.get_closest_and_furthest_2_fold(master_id, master_2_folds)

        # Now make sure that the 2-fold that is closest and the 3-fold are coming from opposite highest-folds
        # Projected vector to the highest-fold geometric center was calculated above (pz1high). Now calculate the non-projected ones to
        # the two highest-folds next to the master one
        # the z2high contains the closest 2-fold, and z3high the 3-fold.
        if closest_2fold_id in second_high_fold:
            z2high = vector(self.get_geometric_center(second_high_fold), global_center)
            z3high = vector(self.get_geometric_center(third_high_fold), global_center)
        else:
            z2high = vector(self.get_geometric_center(third_high_fold), global_center)
            z3high = vector(self.get_geometric_center(second_high_fold), global_center)

        # change the chain labels to match the Rosetta output
        self.remap_chain_labels(master_high_fold, master_3_fold, closest_2fold_id, furthest_2fold_id, second_high_fold, third_high_fold)

        # ---- Main 3-fold's high-fold ----
        axis = z3high
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
        start = 180 - highest_angle
        to_go_through = [start + highest_angle * i for i in range(0, subunits_in_other_highfold)]
        for i, angle in enumerate(to_go_through, 1):
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
            if z3high[1] < 0:
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
        # ------------------------------

        # ---- Main 2-fold's 5-fold ----
        axis = z2high
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
        start = 180
        to_go_through = [start + highest_angle * i for i in range(0, subunits_in_other_highfold)]
        for i, angle in enumerate(to_go_through, 1):
            rt_from_global_z = rotation_matrix(np.cross(axis, global_z), vector_angle(axis, global_z))
            # so before coming here we have the master subunit along the positive x-axis.
            # if the 3-fold axis is left to the x-axis (y coordinate of the adjacent subunit is negative), we have to rotate
            # in the obbosite direction. This is why we check for the y coordinate in the below. See for instance 1stm vs 4bcu.
            if z3high[1] < 0:
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
        # ------------------------------

        # The 6 degrees of freedom
        setup.add_dof(f"JUMPHFfold1", 'z', "translation", np.linalg.norm(pz1high))
        setup.add_dof(f"JUMPHFfold1", 'z', "rotation", 0)
        setup.add_dof(f"JUMPHFfold111", 'x', "translation", np.linalg.norm(vector(master_com, pz1high)))
        setup.add_dof(f"JUMPHFfold1111", 'x', "rotation", 0)
        setup.add_dof(f"JUMPHFfold1111", 'y', "rotation", 0)
        setup.add_dof(f"JUMPHFfold1111", 'z', "rotation", 0)
        setup.add_dof(f"JUMPHFfold1111_subunit", 'x', "rotation", 0)
        setup.add_dof(f"JUMPHFfold1111_subunit", 'y', "rotation", 0)
        setup.add_dof(f"JUMPHFfold1111_subunit", 'z', "rotation", 0)
        setup.add_jumpgroup("JUMPGROUP1", f"JUMPHFfold1", "JUMP3fold1", "JUMP2fold1")

        setup.add_jumpgroup("JUMPGROUP2", *[f"JUMPHFfold1{i}1" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}1" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}1" for i in range(1, subunits_in_other_highfold + 1)])

        setup.add_jumpgroup("JUMPGROUP3", *[f"JUMPHFfold1{i}11" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}11" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}11" for i in range(1, subunits_in_other_highfold + 1)])

        setup.add_jumpgroup("JUMPGROUP4", *[f"JUMPHFfold1{i}11_subunit" for i in range(1, highest_fold + 1)],
                            *[f"JUMP3fold1{i}11_subunit" for i in range(1, subunits_in_other_highfold + 1)],
                            *[f"JUMP2fold1{i}11_subunit" for i in range(1, subunits_in_other_highfold + 1)])

        setup.energies = f"60*VRTHFfold1111 + " \
                         f"60*(VRTHFfold1111:VRTHFfold1211) + " \
                         f"60*(VRTHFfold1111:VRTHFfold1311) + " \
                         f"60*(VRTHFfold1111:VRT3fold1{1}11) + " \
                         f"30*(VRTHFfold1111:VRT2fold1{1}11) + " \
                         f"30*(VRTHFfold1111:VRT3fold1{2}11)"

        # todo: there seems be a difference between rosetta COM and my COM so in the case of 6u20 for instance - you'll not get the correct symmetry if recenter is not turned of
        # anchor residues
        setup.anchor = "COM"
        # # checks that is should be recentered
        # setup.recenter = True
        adjust = vector(global_center, self.get_subunit_CA_closest_to_GC(master_id))
        master = self.get_subunit_with_id(master_id).copy()
        master.transform(np.identity(3), adjust)
        return setup, master

    def find_X_folds(self, x, id="1", closest=1):
        """Wrapper function. Returns the closet x-fold subunits to subunit with the id.
        The underlying functions can be written generally in the future with a recursive functions.

        Only 2-5 fold symmetries are supported!

        TODO:
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
            return self._find_4_fold_subunits(id, closest)

        if x == 5:
            return self._find_5_fold_subunits(id, closest)

    def _detect_cubic_symmetry(self):
        """Detects for the following symmetry: Icosahedral (I) or Octahedral (O) or Tetrahedral (T).
        :returns str type of either 'I', 'O' or 'T' if symmetry is found else None."""
        symmetry = None
        try:
            xfolds = self.find_X_folds(5)
            if xfolds and len(xfolds) > 0:
                symmetry = "I"
        except ValueToHigh:
            pass
        if not symmetry:
            try:
                xfolds = self.find_X_folds(4)
                if xfolds and len(xfolds) > 0:
                    symmetry = "O"
            except ValueToHigh:
                pass
        if not symmetry:
            try:
                xfolds = self.find_X_folds(3)
                if xfolds and len(xfolds) > 0:
                    symmetry = "T"
            except ValueToHigh:
                pass
        if symmetry:
            print(f"{symmetry} symmetry detected.")
        else:
            print(f"{symmetry} symmetry NOT detected.")
        return symmetry

    def _find_2_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the 2 fold axis that the subunit with the id "name" is in.

         Algorithm:
         Checks that the 180 degrees rotation around the center of the 2 subunits overlaps them. If so you have
         a two_fold!

        TODO:
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

    def angle_check(self, tobe, p1, p2, p3):
        return criteria_check(tobe, angle(p1.xtra["alignment_geometric_center"], p2.xtra["alignment_geometric_center"],
                                        p3.xtra["alignment_geometric_center"]), diff=self.angles_diff)

    def distance_check(self, p1A, p2A, p1B, p2B):
        return criteria_check(distance(p1A.xtra["alignment_geometric_center"], p2A.xtra["alignment_geometric_center"]),
                    distance(p1B.xtra["alignment_geometric_center"], p2B.xtra["alignment_geometric_center"]), diff=self.angles_diff)

    def shortest_path_in_subunits(self, subunits, *positions):
        return shortest_path(*(subunits[i].xtra["alignment_geometric_center"] for i in positions))

    def _find_3_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the three fold axis that the subunit with the id "name" is in.

        Algorithm:
        Similar to the algorithm used to find the 5-fold symmetry units (see below). It instead use the degrees of
        a triangle instead of a pentamer.

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 3-fold symmetry subunits the subunit belong to.
        :return
        """
        self._set_alignment_geometric_center()
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
        threefold = []
        subunits_to_search = [subunit for subunit in self.get_models() if subunit.id != id]
        # Find the other 2 protein_list.txt making up the three-fold symmetry axis
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p2 != p3:
                    # p1,p2,p3 angle should be equal to 108 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(60, p1, p2, p3) and self.distance_check(p1, p2, p2, p3):
                        threefold.append((p1, p2, p3))
        if len(threefold) != 0:
            # threefold = list({tuple(sorted(subunits,key=lambda subunit: subunit.id)) for subunits in threefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            threefold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(3)))
            threefolds = []
            for subunits in threefold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(3, *subunits):
                        threefolds.append([subunit.id for subunit in subunits])
                    if len(threefolds) == closest:
                        return threefolds
            raise ToHighRMSD("None of the 3-folds found are valid 3-folds.")
        else:
            raise ToHighGeometry("A 3-fold axis was not found")

    def is_right_handed(self, subunits):
        """Calcualtes the handedness of the sunbunits order, and returns True if it right-handed and False otherwise"""
        all_gc = self.get_geometric_center([s.id for s in subunits])
        s0_gc = self.get_geometric_center(subunits[0].id)
        s1_gc = self.get_geometric_center(subunits[1].id)
        cross = np.cross(np.array(s0_gc - all_gc), np.array(s1_gc - all_gc))
        # if the cross product points opposite to the all_gc vector then the subunits are right-handed The way that I set up the
        # symmetric system 'a' should be 180 for right-handed and 0 for left-handed (aka pointing in the same direction) but I allow for
        # some leeway (symmetrical inaccuracies + floating point number) so I just say < / > 90
        a = vector_angle(cross, all_gc)
        return a > 90 # True -> It is right-handed

    def _find_4_fold_subunits(self, id="1", closest=1):
        """Finds the other subunits belonging to the four fold axis that the subunit with the id "name" is in.

        Algorithm:
        Similar to the algorithm used to find the 5-fold symmetry units (see below). It instead use the degrees of
        a square instead of a pentamer.

        :param str name: the name given to the return SymmetricAssembly return object.
        :param str id: the name of the subunit to find the 4-fold symmetry subunits the subunit belong to.
        :return
        """
        self._set_alignment_geometric_center()
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
        fourfold = []
        # Find the other 3 proteins making up the fourfold symmetry axis (p2, p3 and p4)
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p3.id != p2.id:
                    # p1,p2,p3 angle should be equal to 90 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(90, p1, p2, p3) and self.distance_check(p1, p2, p2, p3):
                        for p4 in subunits_to_search:
                            if p4 != p2 and p4 != p3:
                                # p1,p3,p4 angle should be equal to 45
                                # the distance between p2 and p3 should be the same as p3 and p4
                                if self.angle_check(45, p1, p3, p4) and self.distance_check(p2, p3, p3, p4):
                                    fourfold.append((p1, p2, p3, p4))

        if len(fourfold) != 0:
            # fourfold = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in fourfold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            fourfold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(4)))
            fourfolds = []
            for subunits in fourfold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(4, *subunits):
                        fourfolds.append([subunit.id for subunit in subunits])
                    if len(fourfolds) == closest:
                        return fourfolds
            raise ToHighRMSD("None of the 4-folds found are valid 4-folds.")
        else:
            raise ToHighGeometry("A 4-fold axis was not found.")

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
        :return

        """
        self._set_alignment_geometric_center()
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
        fivefold = []
        # Find the other 4 protein_list.txt making up the five fold symmetry axis (p2, p3, p4 and p5)
        for p2 in subunits_to_search:
            for p3 in subunits_to_search:
                if p3.id != p2.id:
                    # p1,p2,p3 angle should be equal to 108 degrees
                    # the distance between p1 and p2 should be the same as p2 and p3
                    if self.angle_check(108.0, p1, p2, p3) and self.distance_check(p1, p2, p2, p3):
                        for p4 in subunits_to_search:
                            if p4 != p2 and p4 != p3:
                                # p1,p3,p4 angle should be equal to 72
                                # the distance between p2 and p3 should be the same as p3 and p4
                                if self.angle_check(72, p1, p3, p4) and self.distance_check(p2, p3, p3, p4):
                                    for p5 in subunits_to_search:
                                        if p5 != p2 and p5 != p3 and p5 != p4:
                                            # p1,p4,p5 angle should be equal to 36
                                            # the distance between p3 and p4 should be the same as p4 and p5
                                            if self.angle_check(36, p1, p4, p5) and self.distance_check(p3, p4, p4, p5):
                                                fivefold.append((p1, p2, p3, p4, p5))
        if len(fivefold) != 0:
            # fivefold = list({tuple(sorted(subunits, key=lambda subunit: subunit.id)) for subunits in fivefold})
            # find and retrieve the 5fold subunit that is closest to the subunit searched for
            fivefold.sort(key=lambda subunits: self.shortest_path_in_subunits(subunits, *range(5)))
            fivefolds = []
            for subunits in fivefold:
                if self.is_right_handed(subunits):
                    if self.is_correct_x_fold(5, *subunits):
                        fivefolds.append([subunit.id for subunit in subunits])
                    if len(fivefolds) == closest:
                        return fivefolds
            raise ToHighRMSD("None of the 5-folds found are valid 5-folds.")
        else:
            raise ToHighGeometry("A 5-fold axis was not found.")

    def is_correct_x_fold(self, fold, *args):
        """Checks if the fold is a correct 3, 4 or 5-fold.

        It does this by superimposing 1 subunit on top of the other using 3, 4 or 5 fold rotations around the folds
        geometric center of mass. A RMSD criteria is checked to conclude if the superposition is good enough to
        validate it as a 3, 4 or 5 fold.
        """
        accepted_rmsds = []
        # create rotation matrices for 3, 4 or 5 fold rotations
        center = self.get_geometric_center([x.id for x in args])
        if fold == 3:
            angles = [120, 120 * 2]
            # angles += [- angle for angle in angles]
        elif fold == 4:
            angles = [90, 90 * 2, 90*3]
        elif fold == 5:
            angles = [72, 72 * 2, 72 * 3, 72 * 4]
            # angles += [- angle for angle in angles]
        else:
            raise NotImplementedError("only works for a 3, 4 and 5 folds")
        rots = []
        for angle in angles:
            rots.append(rotation_matrix(center, angle))
        s1 = args[0]
        s1_residues = list(s1.get_residues())
        sxs = args[1:] # + args[1:]
        correct_fold = []
        # check if the subunits come on top of each other when using the rotation
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
                elif fold == 4:
                    if self.lowest_4fold_rmsd:
                        self.lowest_4fold_rmsd = min(self.lowest_4fold_rmsd, rmsd)
                    else:
                        self.lowest_4fold_rmsd = rmsd
                elif fold == 5:
                    if self.lowest_5fold_rmsd:
                        self.lowest_5fold_rmsd = min(self.lowest_5fold_rmsd, rmsd)
                    else:
                        self.lowest_5fold_rmsd = rmsd

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
            if fold == 4:
                if sum(correct_fold) >= 3:
                    if self.highest_4fold_accepted_rmsd:
                        self.highest_4fold_accepted_rmsd = max(self.highest_4fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_4fold_accepted_rmsd = max(accepted_rmsds)
                    return True
            elif fold == 5:
                if sum(correct_fold) >= 4:
                    if self.highest_5fold_accepted_rmsd:
                        self.highest_5fold_accepted_rmsd = max(self.highest_5fold_accepted_rmsd, max(accepted_rmsds))
                    else:
                        self.highest_5fold_accepted_rmsd = max(accepted_rmsds)
                    return True
        return False
