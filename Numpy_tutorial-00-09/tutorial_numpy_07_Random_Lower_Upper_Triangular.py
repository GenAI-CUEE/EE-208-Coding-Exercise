import numpy as np

# Create a full 3x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Convert to lower triangular
lower_matrix = np.tril(matrix, -1)
print("lower_matrix")
print(lower_matrix)

breakpoint

upper_matrix = np.triu(matrix, -1) 
print("upper_matrix")
print(upper_matrix) 

breakpoint