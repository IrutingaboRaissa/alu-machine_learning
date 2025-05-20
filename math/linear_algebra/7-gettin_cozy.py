#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    # Concatenate along rows
    if axis == 0:
        if len(mat1) == 0:
            return [row[:] for row in mat2]
        if len(mat2) == 0:
            return [row[:] for row in mat1]
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    # Concatenate along columns
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1[:] + r2[:] for r1, r2 in zip(mat1, mat2)]
    else:
        return None
