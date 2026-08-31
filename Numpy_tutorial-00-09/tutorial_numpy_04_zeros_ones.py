import numpy as np

A_matrix = np.zeros((5,5)) 
print("A_matrix")
print(A_matrix)
breakpoint()

a1_vect = np.zeros((5,)) 
print("a1_vect")
print(a1_vect); 
print(a1_vect.ndim)
breakpoint()

a2_vect = np.zeros((5,1)) 
print("a2_vect")
print(a2_vect)
print(a2_vect.ndim)
breakpoint()


################################################## 

B_matrix = np.ones((5,5)) 
print("B_matrix")
print(B_matrix)
breakpoint()

b1_vect = np.ones((5,))
print("b1_vect") 
print(b1_vect)
breakpoint()

b2_vect = np.ones((5,1))
print("b2_vect") 
print(b2_vect)
breakpoint()

################################################## 
# Generate matrix of 1 that has the same shape as A_matrix 
C_matrix = np.ones_like(A_matrix) 
print("C_matrix = np.ones_like(A_matrix)") 
print(C_matrix) 
breakpoint()

# Generate matrix of 0 that has the same shape as A_matrix 
D_matrix = np.zeros_like(A_matrix)
print("D_matrix = np.zeros_like(A_matrix)") 
print(D_matrix)  
breakpoint()