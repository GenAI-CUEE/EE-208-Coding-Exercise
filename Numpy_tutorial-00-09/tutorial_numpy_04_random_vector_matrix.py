import numpy as np 

#  a matrix of random integers
A_vector = np.random.randint(0, 10, size=(5,))

print("A_vector")
print(A_vector)
breakpoint

A_matrix = np.random.randint(0, 10, size=(5,5))
print("A_matrix")
print(A_matrix) 
breakpoint

############################################
#  a matrix of random floating numbers

rng = np.random.default_rng()
B_vector = rng.random(size=(5,)) 

print("B_vector")
print(B_vector)
breakpoint


B_matrix = rng.random(size=(5,5))  
print("B_matrix")
print(B_matrix)
breakpoint



############################################ 
# Alternative way to generate a matrix of random integers
rng = np.random.default_rng() 
C_vector = rng.integers(0, 10, size=(5,)) 

print("C_vector")
print(C_vector)
breakpoint


C_matrix = rng.integers(0, 10, size=(5,5))  
print("C_matrix")
print(C_matrix)
breakpoint

