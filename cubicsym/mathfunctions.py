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


def normalize(vec):
    """Normalize a vector.

    :param ndarray vec: Vector to normalize.
    :return ndarray: The normalized vector.

    """
    norm = np.linalg.norm(vec)
    return vec / norm

def distance(vec1, vec2):
    """Calculates the distance between vectors.

    :param ndarray vec1: Vector 1.
    :param ndarray vec2: Vector 2.
    :return float: the ditsance between vector 1 and 2.

    """
    return np.linalg.norm(vec1 - vec2)

def scalar_projection(vec1, vec2):
    """Scalar projection of a vector (vec1) onto another vector (vec2).

    Ref: Linear Algebra with Applications, international edition, 2014 - p 248.

    :param ndarray vec1: Vector to project.
    :param ndarray vec2: Vector to project onto.
    :return float: The scalar projection.

    """
    inner_product = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec2)
    return inner_product / norm


def vector_projection(vec1, vec2):
    """Vector projection of a vector (vec1) onto another vector (vec2).

    Ref: Linear Algebra with Applications, international edition, 2014 - p 248.

    :param ndarray vec1: Vector to project.
    :param ndarray vec2: Vector to project onto.
    :return ndarray: Projected vector.

    """
    return scalar_projection(vec1, vec2) * normalize(vec2)

# FIXME: Assumes orthonormal vectors in subspace!!!
def vector_projection_on_subspace(vec1, *vectors):
    """Vector projection of a vector (vec1) onto another subspace spanned by *vectors.

    Ref: My imagination.

    The projection is a sum of projections onto all subspace vectors

    :param np.ndarray vec1: Vector to project.
    :param vectors: Vector(s) that spans the subspace.
    :return np.ndarray: Projected vector.

    """
    projection = np.zeros((len(vec1)))
    for vector in vectors:
        projection += scalar_projection(vec1, vector) * normalize(vector)
    return projection


def rotation_matrix(axis, angle):
    """Generates a rotation matrix for a rotation about an arbitrary axis.

    :param ndarray axis: Axis to rotate about. Shape(3,).
    :param float angle: The angle of the rotation in degrees.
    :return ndarray: The rotation matrix. Shape(3,3).

    """
    unit = axis / np.linalg.norm(axis)
    cos_theta = math.cos(math.radians(angle))
    one_minus_cos_theta = 1.0 - cos_theta
    sin_theta = math.sin(math.radians(angle))
    xx = cos_theta + unit[0] * unit[0] * one_minus_cos_theta
    xy = unit[0] * unit[1] * one_minus_cos_theta - unit[2] * sin_theta
    xz = unit[0] * unit[2] * one_minus_cos_theta + unit[1] * sin_theta
    yx = unit[1] * unit[0] * one_minus_cos_theta + unit[2] * sin_theta
    yy = cos_theta + unit[1] * unit[1] * one_minus_cos_theta
    yz = unit[1] * unit[2] * one_minus_cos_theta - unit[0] * sin_theta
    zx = unit[2] * unit[0] * one_minus_cos_theta - unit[1] * sin_theta
    zy = unit[2] * unit[1] * one_minus_cos_theta + unit[0] * sin_theta
    zz = cos_theta + unit[2] * unit[2] * one_minus_cos_theta
    rot = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    return rot


def rotate(vec, R):
    """Rotates a 1D vector.

    :param ndarray vec: Vector to rotate.
    :param ndarray R: Rotation matrix.
    :return: ndarray: Rotated vector.

    """
    return np.dot(vec, R)


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


def vector_angle(vec1, vec2):
    """Calculates the angle between two vectors.

    :param ndarray vec1: The first vector.
    :param ndarray vec2: The second vector.
    :return float: The angle in degrees.

    """
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
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
