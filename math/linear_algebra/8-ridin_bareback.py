#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    # Check if multiplication is possible
    if len(mat1) == 0 or len(mat2) == 0 or len(mat1[0]) != len(mat2):
        return None
    # Matrix multiplication
    result = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            new_row.append(sum(a * b for a, b in zip(row, col)))
        result.append(new_row)
    return result
