#!/usr/bin/env python3
import numpy as np

def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The numpy array to calculate the shape of.

    Returns:
        tuple: A tuple of integers representing the shape of the array.
    """
    return matrix.shape