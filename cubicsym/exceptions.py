#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Additional exceptions classes
@Author: Mads Jeppesen
@Date: 4/6/22
"""

class ToHighRMSD(BaseException):

    def __init__(self, message):
        extra_message="To high RMSD occured! This can happen if the structure is not near perfectly symmetric."  \
                      "If you still want to model the structure increase the 'rmsd_diff' criteria."
        self.message = message + "\n\n" + extra_message
        print(self.message)

class ToHighGeometry(BaseException):

    def __init__(self, message):
        extra_message="To high angles differences occured! This can happen if the structure is not near perfectly symmetric."  \
                      "If you still want to model the structure increase the 'angle_diff' criteria."
        self.message = message + "\n\n" + extra_message
        print(self.message)

