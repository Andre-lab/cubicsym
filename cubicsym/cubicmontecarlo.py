#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 10/11/22
"""
import random
import math
from cubicsym.dofspec import DofSpec
from copy import deepcopy

# you should try to immitate this one: protocols/monte_carlo/GenericSimulatedAnnealer.hh
# MonteCarlo from doesnt by default have simulated annealing as far as I can see.
# GenericSimulatedAnnealer has a quite complicated annealing schedule. It has multiple temperatures,
# keeps in check for each filter or score. Look at scale_temperatures to see how it works.
class CubicMonteCarlo:
    """Monte Carlo mover with simulated annealing. Instead of cloning poses on each success it only copies the dofs.
    This is about 10x faster!"""

    def __init__(self, scorefunction, dofspec: DofSpec, pose=None, reset_on_first_apply=False, annealing=False, t_delta=0.001, t_start=0.8):
        self.sfxn = scorefunction
        self.reset_on_first_apply = reset_on_first_apply
        self.dofspec = dofspec
        if pose is not None:
            self.reset(pose)
        self.t_delta = t_delta # how much to decrement the temperature with
        self.t_start = t_start # starting temperature. Same as for SymDockAdaptiveMover
        self.t = self.t_start
        self.annealing = annealing

    def reset(self, pose):
        """Sets the best score and best pose from passed pose."""
        self.lowest_score = self.sfxn.score(pose)
        self.lowest_scored_positions = self.dofspec.get_positions_as_list(pose)
        self.last_accepted_score = self.lowest_score
        self.last_accepted_positions = deepcopy(self.lowest_scored_positions)
        self.accepted = None

    def recover_lowest_scored_pose(self, pose):
        """Will assign the lowest scored positions to the pose."""
        self.dofspec.transfer_dofs_to_pose(pose, *self.lowest_scored_positions)

    def recover_last_accepted_pose(self, pose):
        """Will assign the last scored positions to the pose."""
        self.dofspec.transfer_dofs_to_pose(pose, *self.last_accepted_positions)

    def correct_for_best_score(self):
        """If the last accepted score is better than the globally best one then replace the globally best one."""
        if self.last_accepted_score < self.lowest_score:
            self.lowest_score = self.last_accepted_score
            self.lowest_scored_positions = deepcopy(self.last_accepted_positions)

    def apply(self, pose):
        """Metropolis-Hastings MC with potential simulated annealing."""

        if self.reset_on_first_apply:
            self.reset_on_first_apply = False
            self.reset(pose)
            return

        # calculate the energy difference
        current_score = self.sfxn.score(pose)
        score_diff = current_score - self.last_accepted_score

        # if score is better, accept it (exponential >1 always and boltzmann criteria does not make sense)
        # if not, accept based on the boltzmann criteria and the current temperature
        if score_diff < 0 or random.uniform(0, 1) < math.exp(-score_diff / self.t):
            self.last_accepted_score = current_score
            self.last_accepted_positions = self.dofspec.get_positions_as_list(pose)
            self.correct_for_best_score()
            self.accepted = True
        else:
            self.recover_last_accepted_pose(pose)
            self.accepted = False

        # decrement the temperature if annealing is turned on
        if self.annealing:
            self.t -= self.t_start

        return self.accepted
