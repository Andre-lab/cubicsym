#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python based class for common protocol based classes in Rosetta such as the TrialMover, JumpOutMover and SequenceMover classes
@Author: Mads Jeppesen
@Date: 10/10/22
"""

class TrialMover:
    """Stand in for the Rosetta c++ TrialMover. Works on python-based movers that do not inherit from the c++ mover class"""

    def __init__(self, mover, mc):
        """Initialize object."""
        self.mover = mover
        self.mc = mc

    def apply(self, pose):
        """Apply"""
        self.mover.apply(pose)
        self.mc.boltzmann(pose)

class JumpOutMover:
    """Stand in for the Rosetta c++ JumpOutMover. Works on python-based movers that do not inherit from the c++ mover class"""

    def __init__(self, first_mover_in, second_mover_in, scorefxn_in, tolerance_in):
        """Initialize object."""
        self.first_mover = first_mover_in
        self.second_mover = second_mover_in
        self.scorefxn = scorefxn_in
        self.tolerance = tolerance_in

    def apply(self, pose):
        """Apply on pose"""
        raise NotImplementedError("this has to be tested and I dont think it will work with pose_total_eneriges")
        initial_score = pose.energies().total_energy()
        self.first_mover.apply(pose)
        self.scorefxn(pose)
        move_score = pose.energies().total_energy()
        if move_score - initial_score < self.tolerance:
            self.second_mover.apply(pose)

class SequenceMover:
    """Stand in for the Rosetta c++ SequenceMover. Works on python-based movers that do not inherit from the c++ mover class.
    It does not have the use_mover_status=true functionality as its counterpart (it is also off by default anyways)."""

    def __init__(self):
        """Initialize object."""
        self.movers = []

    def add_mover(self, mover):
        """Add a mover to the sequence."""
        self.movers.append(mover)

    def apply(self, pose):
        """Apply the sequence of movers on the pose."""
        for mover in self.movers:
            mover.apply(pose)

class CycleMover:
    """Stand in for the Rosetta c++ CycleMover. Works on python-based movers that do not inherit from the c++ mover class"""

    def __init__(self):
        """Initialize object."""
        self.movers = []
        self.next_move = 0

    def add_mover(self, mover):
        """Add a mover to the cycle."""
        self.movers.append(mover)

    def reset_cycle_index(self):
        """Resets the cycle so that it starts from the beginning."""
        self.next_move = 0

    def apply(self, pose):
        """Apply the next mover in the cycle to the pose."""
        self.next_move %= len(self.movers)
        self.movers[self.next_move].apply(pose)
        self.next_move += 1
