#!/usr/bin/env python3
# coding=utf-8
"""
Math functions
@Author: Mads Jeppesen
@Date: 4/6/22
"""
import math
import numpy as np

def vector(end, start):
    """Creates a vector from a start and end point.

    :param ndarray end: Endpoint of the vector.
    :param ndarray start: Startpoint of the vector.
    :return ndarray: A vector from start to end.

    """
    return end - start


def distance(vec1, vec2):
    """Calculates the distance between vectors.

    :param ndarray vec1: Vector 1.
    :param ndarray vec2: Vector 2.
    :return float: the ditsance between vector 1 and 2.

    """
    return np.linalg.norm(vec1 - vec2)

def angle(p1, p2, p3):
    """Calculates the angle between 3 points.

    :param ndarray p1: Point 1.
    :param ndarray p2: Point 2.
    :param ndarray p3: Point 3.
    :return float: The angle in degrees.

    """
    p1s2 = p1 - p2
    p3s2 = p3 - p2
    cos_angle = np.dot(p1s2, p3s2) / (np.linalg.norm(p1s2) * np.linalg.norm(p3s2))
    return math.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def criteria_check(value1, value2, diff=2.0):
    """Checks that two values are within a give threshold.

    Helper function for the SymmetricAssembly class.
    The threshold is hardcoded. In the future it should be passed as a param.

    NOTE! the PDB id 5j36 needs at least a difference of 0.3. Therefore the value is set this high.
          now is seems 4rft needs at least 0.7! 4zor needs above 1.0!

    :param float value1: The first value.
    :param float value2: The second value.
    :return bool: True if values are within the threshold, False if not.

    """
    difference = value2 - value1
    if difference > - diff and difference < diff:
        return True
    return False

def shortest_path(*positions):
    """Returns the total distance between the positions in cartesian space

    :param: positions to consider.
    :return float: the distance between the positions
    """
    distance = 0
    for i in range(len(positions) - 1):
        distance += np.linalg.norm(positions[i] - positions[i + 1])
    distance += np.linalg.norm(positions[0] - positions[-1])
    return distance
