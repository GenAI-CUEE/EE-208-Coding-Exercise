import numpy as np

# Create a full 3x3 matrix
A_matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Target
# sym_matrix = np.array([[1, 2, 3],
#                        [2, 5, 6],
#                        [3, 6, 9]])

#             =  upper_matrix (without diag) + upper_matrix.T (without diag) + diag_matrix
#                   [[0 2 3]                    [[0 0 0]                       [[1 0 0]
#                   [0 0 6]                      [6 0 0]                        [0 5 0]
#                   [0 0 0]]                     [3 2 0]]                       [0 0 9]]
                 
# Convert to lower triangular
upper_matrix = np.triu(A_matrix, 1)
print("upper_matrix")
# print(upper_matrix)
# [[0 2 3]
#  [0 0 6]
#  [0 0 0]] 

# Extract diagonal 
diag_matrix = np.diag(np.diag(A_matrix))
print("diag_matrix")
print(diag_matrix)
# [[1 0 0]
#  [0 5 0]
#  [0 0 9]]

# Symmetric matrix 

sym_matrix = upper_matrix + upper_matrix.T + diag_matrix
print("sym_matrix")
print(sym_matrix)
# sym_matrix = np.array([[1, 2, 3],
#                        [2, 5, 6],
#                        [3, 6, 9]]) ?? 


  