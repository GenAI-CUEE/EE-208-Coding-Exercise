import numpy as np

# Contruct an identity matrix 

A_NxN_matrix = np.eye(5,5)

print("A_NxN_matrix")
print(A_NxN_matrix)
breakpoint
 

############################################################ 
# Contrustruct a diagonal matrix (B_NxN_matrix) whose elements are random value... 
rng = np.random.default_rng()
random_vector = rng.random(size=(5,))
 
B_NxN_matrix =  random_vector*np.eye(5) 
print("B_NxN_matrix")
print(B_NxN_matrix)

breakpoint

# An alternative way to construct the same  diagonal matrix
C_NxN_matrix = np.diag(random_vector) 

print("C_NxN_matrix")
print(C_NxN_matrix)

breakpoint

print("B_NxN_matrix == C_NxN_matrix?")
assert (B_NxN_matrix == C_NxN_matrix).all()
print("YES, they are the same")
