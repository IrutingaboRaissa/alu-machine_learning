#!/usr/bin/env python3
"""
Function to convert one-hot encoding back to numeric labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): One-hot encoded array with shape (classes, m)
            classes is the maximum number of classes
            m is the number of examples

    Returns:
        numpy.ndarray: Array with shape (m,) containing numeric labels for each example,
                      or None on failure
    """
    try:
        # Validate input parameter
        if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
            return None
        
        # Get dimensions
        classes, m = one_hot.shape
        
        # Check if the matrix contains only 0s and 1s
        if not np.all(np.logical_or(one_hot == 0, one_hot == 1)):
            return None
        
        # Check if each column has exactly one 1 (valid one-hot encoding)
        if not np.all(np.sum(one_hot, axis=0) == 1):
            return None
        
        # Find the index of the maximum value (which should be 1) in each column
        # This gives us the class label for each example
        labels = np.argmax(one_hot, axis=0)
        
        return labels
        
    except Exception:
        return None
