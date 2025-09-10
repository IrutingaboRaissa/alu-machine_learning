#!/usr/bin/env python3
"""
Function to convert numeric labels into one-hot encoding
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): Array with shape (m,) containing numeric class labels
        classes (int): Maximum number of classes found in Y

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        # Validate input parameters
        if not isinstance(Y, np.ndarray) or Y.ndim != 1:
            return None
        
        if not isinstance(classes, int) or classes <= 0:
            return None
        
        # Check if all labels are valid (non-negative and less than classes)
        if np.any(Y < 0) or np.any(Y >= classes):
            return None
        
        # Get the number of examples
        m = Y.shape[0]
        
        # Create one-hot encoding matrix with shape (classes, m)
        one_hot = np.zeros((classes, m))
        
        # Set the appropriate elements to 1
        # For each example, set the element at [label, example_index] to 1
        one_hot[Y, np.arange(m)] = 1
        
        return one_hot
        
    except Exception:
        return None
