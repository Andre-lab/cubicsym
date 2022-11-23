#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 10/10/22
"""
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap

class BoundedMinMover:
    """Returns the pose to the state it was before minimization if the minimization put the pose out of bounce"""

    def __init__(self, cubicboundary, sfxn, min_tolerance = 0.01, min_type = "lbfgs_armijo_nonmonotone", nb_list = True):
        self.cubicboundary = cubicboundary
        self.sfxn = sfxn
        self.min_tolerance = min_tolerance
        self.min_type = min_type
        self.nb_list = nb_list
        self.minmover = self._construct_minmover()
        self.passed = None

    def _construct_movemap(self):
        # Jumps are only minimized
        movemap = MoveMap()
        movemap.set_chi(False)
        movemap.set_bb(False)
        for jumpid in self.cubicboundary.cubicdofs.get_jumps_only_as_int():
            movemap.set_jump(jumpid, True)
        return movemap

    def _construct_minmover(self):
        """Wrapper for the initialization of a MinMover."""
        minmover = MinMover()
        minmover.score_function(self.sfxn)
        minmover.tolerance(self.min_tolerance)
        minmover.min_type(self.min_type)
        minmover.nb_list(self.nb_list)
        minmover.movemap(self._construct_movemap())
        return minmover

    def apply(self, pose):
        self.passed = False
        pose_to_min = pose.clone()
        self.minmover.apply(pose_to_min)
        if self.cubicboundary.all_dofs_within_bounds(pose_to_min):
            self.passed = True
            pose.assign(pose_to_min)