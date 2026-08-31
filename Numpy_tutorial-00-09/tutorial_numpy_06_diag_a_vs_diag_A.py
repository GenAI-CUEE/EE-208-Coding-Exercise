import numpy as np

rng = np.random.default_rng()
random_vector = rng.random(size=(5,)) 

# Construct diagonal matrix from a random vector
C_NxN_matrix = np.diag(random_vector) 

print("C_NxN_matrix")
print(C_NxN_matrix)
breakpoint

# Reverse back to the random vector
C_vector = np.diag(C_NxN_matrix) 
assert (C_vector == random_vector).all()
print("C_vector == random_vector")
breakpoint

###################################
# Extract diagonal coefficients

random_matrix = rng.random(size=(5,5)) 
D_vector = np.diag(random_matrix) 

print("D_vector")
print(D_vector)
breakpoint
