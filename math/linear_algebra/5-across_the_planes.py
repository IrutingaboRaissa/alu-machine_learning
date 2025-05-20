#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    # Check if both matrices have the same number of rows
    if len(mat1) != len(mat2):
        return None
    # Check if all rows have the same length
    if any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    # Return element-wise sum
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
