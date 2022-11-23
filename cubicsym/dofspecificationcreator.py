#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
class for DofspecificationCreator
@Author: Mads Jeppesen
@Date: 10/10/22
"""

class DofspecificationCreator:

    def __int__(self):
        pass

    @staticmethod
    def create_uniform_dofs(lb, ub):
        """Creates a dofspecification where all dofs are limited to a uniform range."""
        dofspecification = {
            "JUMPHFfold1": {"z": {"limit_movement": True, "min": lb, "max": ub}},
            "JUMPHFfold1_z": {"angle_z": {"limit_movement": True, "min": lb, "max": ub}},
            "JUMPHFfold111": {"x": {"limit_movement": True, "min": lb, "max": ub}},
            "JUMPHFfold111_x": {"angle_x": {"limit_movement": True, "min": lb, "max": ub}},
            "JUMPHFfold111_y": {"angle_y": {"limit_movement": True, "min": lb, "max": ub}},
            "JUMPHFfold111_z": {"angle_z": {"limit_movement": True, "min": lb, "max": ub}},
        }
        return dofspecification