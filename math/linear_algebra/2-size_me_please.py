#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    current_dim = matrix
    while isinstance(current_dim, list):
        shape.append(len(current_dim))
        if len(current_dim) > 0:
            current_dim = current_dim[0]
        else:
            break  
    return(shape)


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))  
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print(matrix_shape(mat2))  