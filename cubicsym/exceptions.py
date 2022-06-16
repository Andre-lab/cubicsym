#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Additional exceptions classes
@Author: Mads Jeppesen
@Date: 4/6/22
"""

class ValueToHigh(BaseException):
    def __init__(self, message):
        super(ValueToHigh, self).__init__()

class ToHighRMSD(ValueToHigh):
    def __init__(self, message):
        extra_message="To high RMSD occured! This can happen if the structure is not near perfectly symmetric."  \
                      "If you still want to model the structure increase the 'rmsd_diff' criteria."
        self.message = message + "\n\n" + extra_message
        print(self.message)

class ToHighGeometry(ValueToHigh):
    def __init__(self, message):
        extra_message="To high angles differences occured! This can happen if the structure is not near perfectly symmetric."  \
                      "If you still want to model the structure increase the 'angle_diff' criteria."
        self.message = message + "\n\n" + extra_message
        print(self.message)

class NoSymmetryDetected(BaseException):
    def __init__(self, message=""):
        extra_message = "No symmetry was detected"
        self.message = message + "\n\n" + extra_message
        print(self.message)

